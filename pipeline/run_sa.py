"""End-to-end Selective Abstraction generation without evaluators."""

import argparse
import json
import math
import os
from numbers import Real
from pathlib import Path

from abstract import run_abstract
from atomize import run_atomize
from confidence_abstractions import run_abstractions_confidence
from confidence_atoms import run_atoms_confidence
from generate_response import run_generate_response
from generation_utils import get_response_text, init_client
from reconstruction import run_reconstruction
from refusal_classifier import run_refusal_classification
from selection import write_selected_statements
from utils.path_utils import normalize_output_root
from validation import validate_confidences_for_entity_separate


PAPER_MODELS = (
    "gpt-5.2",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "qwen3-30b-thinking",
    "qwen3-4b-thinking",
)


class PipelineOutputError(RuntimeError):
    """A pipeline stage returned an absent, partial, or misaligned artifact."""


def _require_directory(stage, value):
    if value is None:
        raise PipelineOutputError(f"{stage} returned no output directory.")
    try:
        path = Path(value)
    except TypeError as exc:
        raise PipelineOutputError(
            f"{stage} returned an invalid output directory: {value!r}."
        ) from exc
    if not path.is_dir():
        raise PipelineOutputError(
            f"{stage} output directory does not exist: {path}."
        )
    return path


def _require_file(stage, value):
    if value is None:
        raise PipelineOutputError(f"{stage} returned no output file.")
    try:
        path = Path(value)
    except TypeError as exc:
        raise PipelineOutputError(
            f"{stage} returned an invalid output file: {value!r}."
        ) from exc
    if not path.is_file():
        raise PipelineOutputError(f"{stage} output file does not exist: {path}.")
    return path


def _load_stage_json(stage, path, expected_type):
    path = _require_file(stage, path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineOutputError(f"{stage} output is not valid JSON: {path}.") from exc
    if not isinstance(value, expected_type):
        raise PipelineOutputError(
            f"{stage} output has invalid structure in {path}; "
            f"expected {expected_type.__name__}."
        )
    return value


def _valid_confidence(value):
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 100
    )


def _validate_generation_output(generation_dir, model):
    generation_dir = _require_directory("generation", generation_dir)
    response_path = generation_dir / "response.json"
    response = _load_stage_json("generation", response_path, dict)
    config = response.get("config")
    if not isinstance(config, dict):
        raise PipelineOutputError("generation response is missing its config object.")
    prompt = config.get("generate_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise PipelineOutputError("generation response has an empty generate_prompt.")
    try:
        response_text = get_response_text(response, model)
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise PipelineOutputError("generation response content is missing or malformed.") from exc
    if not isinstance(response_text, str) or not response_text.strip():
        raise PipelineOutputError(
            "generation produced empty content; this is not treated as abstention."
        )
    return generation_dir, response


def _validate_refusal_output(generation_dir, returned_value):
    if type(returned_value) is not bool:
        raise PipelineOutputError("refusal classification did not return a boolean.")
    response = _load_stage_json(
        "refusal classification", generation_dir / "response.json", dict
    )
    if type(response.get("is_refusal")) is not bool:
        raise PipelineOutputError(
            "refusal classification did not persist an is_refusal boolean."
        )
    if response["is_refusal"] != returned_value:
        raise PipelineOutputError(
            "refusal classification return value disagrees with response.json."
        )
    if not isinstance(response.get("refusal_classification"), dict):
        raise PipelineOutputError(
            "refusal classification metadata is missing; a failed classifier is not "
            "silently treated as a valid response."
        )


def _validate_atoms_output(atomize_dir):
    atomize_dir = _require_directory("atomization", atomize_dir)
    atoms = _load_stage_json("atomization", atomize_dir / "atoms.json", list)
    if not atoms:
        raise PipelineOutputError(
            "atomization produced zero atoms; this is not treated as abstention."
        )
    if any(not isinstance(atom, str) or not atom.strip() for atom in atoms):
        raise PipelineOutputError("atomization produced an empty or non-string atom.")
    if len(set(atoms)) != len(atoms):
        raise PipelineOutputError(
            "atomization produced duplicate atoms, which cannot align exactly with "
            "the confidence mapping."
        )
    return atomize_dir, atoms


def _validate_abstractions_output(abstractions_path, atoms):
    abstractions_path = _require_file("abstraction", abstractions_path)
    data = _load_stage_json("abstraction", abstractions_path, list)
    if len(data) != len(atoms):
        raise PipelineOutputError(
            f"abstraction count ({len(data)}) does not match atom count ({len(atoms)})."
        )
    for atom_index, (atom, item) in enumerate(zip(atoms, data)):
        if not isinstance(item, dict):
            raise PipelineOutputError(
                f"abstraction item {atom_index} is not an object."
            )
        if item.get("atom") != atom or item.get("index") != atom_index:
            raise PipelineOutputError(
                f"abstraction item {atom_index} is not exactly aligned with atoms.json."
            )
        choices = item.get("abstractions")
        if not isinstance(choices, list) or not choices:
            raise PipelineOutputError(
                f"abstraction item {atom_index} has no abstraction sequence."
            )
        for abstraction_index, choice in enumerate(choices):
            if not isinstance(choice, dict):
                raise PipelineOutputError(
                    f"abstraction item {atom_index}/{abstraction_index} is not an object."
                )
            text = choice.get("abstraction")
            if (
                choice.get("index") != abstraction_index
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise PipelineOutputError(
                    f"abstraction item {atom_index}/{abstraction_index} is malformed."
                )
    return abstractions_path, data


def _validate_atom_confidences(atom_confidence_path, atoms):
    atom_confidence_path = _require_file("atom confidence", atom_confidence_path)
    data = _load_stage_json("atom confidence", atom_confidence_path, dict)
    if list(data) != atoms:
        raise PipelineOutputError(
            "atom confidence keys are missing, reordered, or not exactly aligned "
            "with atoms.json."
        )
    scores = {}
    for atom in atoms:
        item = data.get(atom)
        if not isinstance(item, dict) or not _valid_confidence(item.get("confidence")):
            raise PipelineOutputError(
                f"atom confidence for {atom!r} is missing or outside [0, 100]."
            )
        scores[atom] = item["confidence"]
    return atom_confidence_path, scores


def _validate_abstraction_confidences(
    abstraction_confidence_path, abstractions, atom_scores
):
    abstraction_confidence_path = _require_file(
        "abstraction confidence", abstraction_confidence_path
    )
    data = _load_stage_json(
        "abstraction confidence", abstraction_confidence_path, list
    )
    if len(data) != len(abstractions):
        raise PipelineOutputError(
            f"abstraction-confidence count ({len(data)}) does not match abstraction "
            f"count ({len(abstractions)})."
        )
    for atom_index, (source, scored) in enumerate(zip(abstractions, data)):
        if not isinstance(scored, dict):
            raise PipelineOutputError(
                f"abstraction-confidence item {atom_index} is not an object."
            )
        atom = source["atom"]
        if scored.get("atom") != atom or scored.get("index") != source["index"]:
            raise PipelineOutputError(
                f"abstraction-confidence item {atom_index} is not exactly aligned."
            )
        atom_confidence = scored.get("confidence")
        if not _valid_confidence(atom_confidence):
            raise PipelineOutputError(
                f"atom confidence at abstraction item {atom_index} is invalid."
            )
        if atom_confidence != atom_scores[atom]:
            raise PipelineOutputError(
                f"atom confidence at abstraction item {atom_index} disagrees with "
                "atoms_confidence.json."
            )
        source_choices = source["abstractions"]
        scored_choices = scored.get("abstractions")
        if not isinstance(scored_choices, list) or len(scored_choices) != len(source_choices):
            raise PipelineOutputError(
                f"abstraction-confidence sequence {atom_index} is missing or truncated."
            )
        for abstraction_index, (source_choice, scored_choice) in enumerate(
            zip(source_choices, scored_choices)
        ):
            if not isinstance(scored_choice, dict):
                raise PipelineOutputError(
                    f"abstraction confidence {atom_index}/{abstraction_index} is not an object."
                )
            if (
                scored_choice.get("index") != source_choice["index"]
                or scored_choice.get("abstraction") != source_choice["abstraction"]
                or not _valid_confidence(scored_choice.get("confidence"))
            ):
                raise PipelineOutputError(
                    f"abstraction confidence {atom_index}/{abstraction_index} is "
                    "misaligned or invalid."
                )
    return abstraction_confidence_path, data


def _validate_selection_output(statements, statements_path):
    if not isinstance(statements, list):
        raise PipelineOutputError("selection did not return a list of statements.")
    if any(not isinstance(statement, str) or not statement.strip() for statement in statements):
        raise PipelineOutputError("selection returned an empty or non-string statement.")
    stored = _load_stage_json("selection", statements_path, list)
    if stored != statements:
        raise PipelineOutputError(
            "selection return value disagrees with selected_statements.json."
        )
    return statements, _require_file("selection", statements_path)


def _validate_reconstruction_output(reconstruction_dir, statements, model):
    reconstruction_dir = _require_directory("reconstruction", reconstruction_dir)
    response = _load_stage_json(
        "reconstruction", reconstruction_dir / "response.json", dict
    )
    config = response.get("config")
    if not isinstance(config, dict):
        raise PipelineOutputError("reconstruction response is missing its config object.")
    try:
        text = get_response_text(response, model)
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise PipelineOutputError(
            "reconstruction response content is missing or malformed."
        ) from exc
    if not isinstance(text, str):
        raise PipelineOutputError("reconstruction response content is not text.")
    if statements:
        if not text.strip() or config.get("abstained") is True:
            raise PipelineOutputError(
                "reconstruction returned empty/abstained content despite selected statements."
            )
    elif text != "" or config.get("abstained") is not True:
        raise PipelineOutputError(
            "empty selection must produce the explicit abstained reconstruction artifact."
        )
    return reconstruction_dir


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run generation, SA selection, and v6 reconstruction without evaluators."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Entity or full prompt inserted into the selected generation template.")
    input_group.add_argument("--input-file", help="UTF-8 file containing the entity or full prompt.")
    parser.add_argument("--output-dir", required=True, help="Directory that will retain every stage output.")
    parser.add_argument("--model", required=True, choices=PAPER_MODELS)
    parser.add_argument(
        "--model-base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible local endpoint; ignored for OpenAI-hosted models.",
    )
    parser.add_argument(
        "--generation-prompt",
        default="longfact",
        choices=("factscore", "longfact", "non_factual"),
        help="Generation prompt template. longfact passes a full prompt through unchanged.",
    )
    parser.add_argument("--theta", required=True, type=int, help="Confidence threshold from 0 to 100.")
    parser.add_argument("--selection", default="sa", choices=("sa", "redaction"))
    parser.add_argument("--prompts-dir", default="pipeline/prompts")
    parser.add_argument(
        "--generation-params-path",
        default="pipeline/utils/model_generation_params.json",
    )
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    args.output_dir = normalize_output_root(args.output_dir)
    if args.theta < 0 or args.theta > 100:
        raise ValueError("theta must be between 0 and 100")

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as handle:
            input_text = handle.read().strip()
    else:
        input_text = args.input
    if not input_text:
        raise ValueError("input must not be empty")

    with open(args.generation_params_path, "r", encoding="utf-8") as handle:
        all_generation_params = json.load(handle)
    if args.model not in all_generation_params:
        raise ValueError(
            f"Model {args.model!r} is missing from {args.generation_params_path}."
        )
    generation_params = all_generation_params[args.model]

    os.makedirs(args.output_dir, exist_ok=True)
    client = init_client(args.model, args.model_base_url)

    generation_dir = run_generate_response(
        entity=input_text,
        client=client,
        model=args.model,
        input_dir=args.output_dir,
        prompt=args.generation_prompt,
        prompts_dir=args.prompts_dir,
        generation_params=generation_params,
        reasoning_effort=args.reasoning_effort,
    )
    generation_dir, generation_data = _validate_generation_output(
        generation_dir, args.model
    )
    is_refusal = run_refusal_classification(
        client=client,
        model=args.model,
        input_dir=generation_dir,
        prompts_dir=args.prompts_dir,
        generation_params=generation_params,
    )
    _validate_refusal_output(generation_dir, is_refusal)
    if is_refusal:
        print(f"Refusal detected; preserved generation at {generation_dir}")
        return

    atomize_dir = run_atomize(
        client=client,
        model=args.model,
        sys_prompt="atomic_facts_brackets",
        input_dir=generation_dir,
        prompts_dir=args.prompts_dir,
        generation_params=generation_params,
        reasoning_effort=args.reasoning_effort,
    )
    atomize_dir, atoms = _validate_atoms_output(atomize_dir)
    abstractions_path = run_abstract(
        model=args.model,
        sys_prompt="abstract_separate",
        client=client,
        atomize_output_dir=atomize_dir,
        prompts_dir=args.prompts_dir,
        generation_params=generation_params,
        reasoning_effort=args.reasoning_effort,
    )
    abstractions_path, abstractions = _validate_abstractions_output(
        abstractions_path, atoms
    )
    atom_confidence_path = run_atoms_confidence(
        model=args.model,
        confidence_score="verbalized",
        client=client,
        input_files={"atoms_file_path": atomize_dir},
        prompts_dir=args.prompts_dir,
        generation_params=generation_params,
        reasoning_effort=args.reasoning_effort,
    )
    atom_confidence_path, atom_scores = _validate_atom_confidences(
        atom_confidence_path, atoms
    )
    abstraction_confidence_path = run_abstractions_confidence(
        model=args.model,
        confidence_score="verbalized",
        client=client,
        input_files={
            "abstract_file_path": abstractions_path,
            "atoms_conf_file_path": atom_confidence_path,
        },
        prompts_dir=args.prompts_dir,
        generation_params=generation_params,
        reasoning_effort=args.reasoning_effort,
    )
    abstraction_confidence_path, _ = _validate_abstraction_confidences(
        abstraction_confidence_path, abstractions, atom_scores
    )
    # Keep the historical validator unchanged for compatibility. The public
    # runner's checks above are intentionally stricter: they require complete,
    # exactly aligned stage outputs and reject every invalid confidence.
    validate_confidences_for_entity_separate(
        abstractions_path,
        abstraction_confidence_path,
    )

    selection_dir = os.path.join(
        generation_dir,
        "selection",
        f"mode={args.selection}__theta={args.theta}",
    )
    statements, statements_path = write_selected_statements(
        confidence_path=abstraction_confidence_path,
        output_dir=selection_dir,
        theta=args.theta,
        mode=args.selection,
    )
    statements, statements_path = _validate_selection_output(
        statements, statements_path
    )

    original_prompt = generation_data["config"]["generate_prompt"]
    reconstruction_dir = run_reconstruction(
        client=client,
        model=args.model,
        original_prompt=original_prompt,
        statements=statements,
        input_dir=selection_dir,
        prompts_dir=args.prompts_dir,
        generation_params=generation_params,
        theta=args.theta,
        reasoning_effort=args.reasoning_effort,
        sys_prompt="recon_rf_v6",
    )
    reconstruction_dir = _validate_reconstruction_output(
        reconstruction_dir, statements, args.model
    )
    print(f"Selected statements: {statements_path}")
    print(f"Reconstruction: {os.path.join(reconstruction_dir, 'response.json')}")


if __name__ == "__main__":
    main()
