import argparse
import os
import time
from generation_utils import init_client
from utils.path_utils import normalize_output_root
from utils.text_utils import load_json, create_numbered_list
from utils.time_utils import get_timestamp, format_duration
from generate_response import run_generate_response
from atomize import run_atomize
from abstract import run_abstract
from confidence_atoms import run_atoms_confidence
from confidence_abstractions import run_abstractions_confidence
from count_questions import run_count_questions
from validation import (
    validation_checkpoint,
    validate_questions_for_entity,
    validate_confidences_for_entity_separate,
)
from self_revision import run_self_revision
from trivial_abstractions import (
    build_trivial_abstractions_from_atoms,
    process_atoms_without_abstraction,
)
from cove import run_cove
from refusal_classifier import run_refusal_classification
from revise_atoms import run_revise_atoms
from sampling_baseline import run_sampling_baseline
import copy
import traceback
import json

PAPER_MODELS = (
    'gpt-5.2',
    'gpt-oss-120b',
    'gpt-oss-20b',
    'qwen3-30b-thinking',
    'qwen3-4b-thinking',
)


parser = argparse.ArgumentParser(description='Run the full selective abstraction pipeline.')
# dataset arguments
parser.add_argument('--sampled_prompts_file', type=str, default=None, help='Path to a sampled prompts JSON file for datasets that are defined by prompts (e.g., longfact_objects_sampled, non_factual). Default is None.')
parser.add_argument('--dataset', type=str, default='factscore_labeled', help='Dataset name for evaluation')
parser.add_argument('--experiment', type=str, default='sa', help='Experiment name for evaluation')
parser.add_argument('--generate_prompt', type=str, default=None, help='Prompt for generation')
parser.add_argument('--theta', type=int, default=None, help='Confidence threshold (0-100) used for *_theta experiments')

# general pipeline arguments
parser.add_argument('--results_root', type=str, default='results', help='Root directory for results')
parser.add_argument('--prompts_dir', type=str, default='pipeline/prompts', help='Path to prompts directory')
# generation arguments
parser.add_argument('--generating_model', type=str, default='gpt-oss-120b', choices=PAPER_MODELS, help='Model to use for generation')
parser.add_argument('--generating_model_base_url', type=str, default=os.environ.get('VLLM_BASE_URL', 'http://localhost:8000/v1'), help='OpenAI-compatible base URL for locally hosted models')
parser.add_argument('--generation_params_path', type=str, default='pipeline/utils/model_generation_params.json', help='Path to per-model generation hyperparameters JSON')

# atomization arguments
parser.add_argument('--atomize_sys_prompt', type=str, default='atomic_facts_brackets', help='Prompt name for atomization (without .txt)')
# confidence arguments
parser.add_argument('--confidence_score', type=str, default='verbalized', help='Confidence scoring method', choices=['verbalized', 'logprobs'])
parser.add_argument('--confidence_sys_prompt', type=str, default='verbal_reasoning', help='Confidence system prompt (without .txt)')
# abstraction arguments
parser.add_argument('--abstract_sys_prompt', type=str, default='abstract_separate', help='Prompt name for separate abstraction (without .txt)')
parser.add_argument('--abstraction_mode', type=str, default='separate', choices=['separate', 'none', 'self_revision'], help='Abstraction mode: separate (abstract then confidence), self_revision (redact then atomize) or none (only atomize)')
# questions arguments
parser.add_argument('--run_questions', action=argparse.BooleanOptionalAction, default=True, help='Whether to run questions or not')
parser.add_argument('--questions_model_base_url', type=str, default=os.environ.get('VLLM_BASE_URL', 'http://localhost:8000/v1'), help='OpenAI-compatible base URL for locally hosted models')
parser.add_argument('--questions_sys_prompt', type=str, default='pairs', help='Prompt for counting questions (without .txt)')
parser.add_argument('--questions_model', type=str, default='gpt-oss-120b', choices=PAPER_MODELS, help='Model for counting questions')
# hyperparameters for questions model will be loaded from hparams_config
# entity range arguments
parser.add_argument('--start_index', type=int, default=0, help='Starting index for entity processing (0-based)')
parser.add_argument('--count', type=int, default=None, help='Number of entities to process (None means process all remaining entities)')
parser.add_argument('--entities_file', type=str, default=None, help='Path to file containing specific entities to process (overrides start_index and count)')
# reasoning effort arguments
parser.add_argument('--reasoning_effort', type=str, default=None, choices=['low', 'medium', 'high'], help='Optional reasoning effort for supported pipeline stages')
parser.add_argument('--iterative_k', type=int, default=2, help='Number of self-revision passes for iterative_self_revision experiment')



def main():
    args = parser.parse_args()
    args.results_root = normalize_output_root(args.results_root)

    supported_experiments = {
        'sa', 'single_pass', 'single_pass_theta', 'double_pass',
        'double_pass_theta', 'iterative_self_revision',
        'atom_wise_self_revision', 'sampling_baseline', 'cove',
    }
    if args.experiment not in supported_experiments:
        raise ValueError(
            f"Unsupported experiment in the public paper release: {args.experiment}. "
            f"Choose one of: {', '.join(sorted(supported_experiments))}."
        )
    if args.dataset.startswith('non_factual') and args.experiment != 'sa':
        raise ValueError(
            "The public non_factual route supports only the paper's Aw-SA "
            "qualitative examples."
        )
    if args.confidence_score not in ('verbalized', 'logprobs'):
        raise ValueError(
            "The public paper release supports confidence_score=verbalized or logprobs only."
        )

    if args.theta is not None and (args.theta < 0 or args.theta > 100):
        raise ValueError(f"Invalid theta: {args.theta}. Must be between 0 and 100.")

    if args.dataset.startswith('factscore'):
        args.sampled_prompts_file = None
    elif args.dataset.startswith('longfact'):
        args.sampled_prompts_file = 'datasets/longfact_objects_sampled/sampled_prompts_2.json'
    elif args.dataset.startswith('non_factual'):
        args.sampled_prompts_file = 'datasets/non_factual/prompts.json'
    else:
        raise ValueError(
            f"Invalid dataset: {args.dataset}. Must start with 'factscore', "
            f"'longfact', or 'non_factual'."
        )

    if args.experiment in ('single_pass', 'single_pass_theta'):
        if args.dataset.startswith('factscore'):
            args.generate_prompt = 'single_pass_theta_factscore' if args.experiment == 'single_pass_theta' else 'single_pass_factscore'
        elif args.dataset.startswith('longfact'):
            args.generate_prompt = 'single_pass_theta_longfact' if args.experiment == 'single_pass_theta' else 'single_pass_longfact'
        args.abstraction_mode = 'none'
    else:
        if args.dataset.startswith('factscore'):
            args.generate_prompt = 'factscore'
        elif args.dataset.startswith('longfact'):
            args.generate_prompt = 'longfact'
        elif args.dataset.startswith('non_factual'):
            args.generate_prompt = 'non_factual'

        if args.experiment in ('double_pass', 'double_pass_theta', 'iterative_self_revision', 'cove'):
            args.abstraction_mode = 'self_revision'
        elif args.experiment == 'atom_wise_self_revision':
            args.abstraction_mode = 'none'
        elif args.experiment == 'sa':
            args.abstraction_mode = 'separate'
        elif args.experiment == 'sampling_baseline':
            args.abstraction_mode = 'self_revision'
        else:
            raise ValueError(f"Invalid experiment: {args.experiment}")
    
    
    print(f"Running pipeline with args: {args}")
    # Load per-model hyperparameters
    try:
        model_hparams = load_json(args.generation_params_path)
        gen_model_hparams = model_hparams[args.generating_model]
        questions_model_hparams = model_hparams[args.questions_model]
    except Exception as e:
        print(f"Error loading generation parameters: {e}")
        exit(1)
    entity_list, start_idx, end_idx, original_length, entity_metadata = get_entity_list(args)

    dataset_results_dir = os.path.join(args.results_root, args.dataset)
    # Initialize generation client 
    generation_client = init_client(args.generating_model, args.generating_model_base_url)
    
    for i, entity_id in enumerate(entity_list):
        entity_start_time = time.time()
        
        # Handle progress message for both modes
        if args.entities_file:
            # When using entities_file, we don't have global indices
            print(f"\n===Processing entity: {entity_id} ({i+1}/{len(entity_list)}) ===")
        else:
            # When using start_index/count, we can show global indices
            global_index = start_idx + i
            print(f"\n===Processing entity: {entity_id} (local: {i+1}/{len(entity_list)}, global: {global_index+1}/{original_length}) ===")
        
        # The directory name is always based on the entity ID (e.g., subject_index for longfact).
        entity_dir_path = os.path.join(dataset_results_dir, args.generating_model, entity_id)

        # Determine the text that will be injected into the generation prompt.
        # For longfact, this should be the full prompt string from the sampled JSON;
        # for other datasets, we fall back to using the entity ID itself.
        if args.dataset in ('longfact_objects_sampled', 'non_factual') and entity_metadata and entity_id in entity_metadata:
            entity_for_prompt = entity_metadata[entity_id].get('prompt', entity_id)
        else:
            entity_for_prompt = entity_id
        os.makedirs(entity_dir_path, exist_ok=True)
            
        try:
            if args.experiment == 'sampling_baseline':
                print(f"{get_timestamp()} [1/5] Running sampling_baseline (N=5)...")
                generate_output_dir = run_sampling_baseline(
                    entity=entity_for_prompt,
                    client=generation_client,
                    model=args.generating_model,
                    input_dir=entity_dir_path,
                    prompt=args.generate_prompt,
                    prompts_dir=args.prompts_dir,
                    generation_params=gen_model_hparams,
                )
            else:
                print(f"{get_timestamp()} [1/5] Running generate_response...")
                generate_output_dir = run_generate_response(
                    entity=entity_for_prompt,
                    client=generation_client,
                    model=args.generating_model,
                    input_dir=entity_dir_path,
                    prompt=args.generate_prompt,
                    prompts_dir=args.prompts_dir,
                    generation_params=gen_model_hparams,
                    theta=args.theta if args.experiment == 'single_pass_theta' else None,
                    reasoning_effort=args.reasoning_effort,
                )

            print(f"{get_timestamp()} [2/5] Running refusal classification...")
            is_refusal = run_refusal_classification(
                client=generation_client,
                model=args.generating_model,
                input_dir=generate_output_dir,
                prompts_dir=args.prompts_dir,
                generation_params=gen_model_hparams,
            )

            if is_refusal:
                print(f"{get_timestamp()} Detected refusal response for entity {entity_id}. Skipping downstream stages.")
                entity_duration = time.time() - entity_start_time
                print(f"{get_timestamp()} === Finished entity (refusal): {entity_id} - Time taken: {format_duration(entity_duration)} ===")
                continue

            # Default: atomize the original generated response directory.
            atomize_input_dir = generate_output_dir

            if args.experiment == 'cove':
                print(f"{get_timestamp()} [3/5] Running CoVe verification (Factor+Revise)...")
                cove_output_dir = run_cove(
                    model=args.generating_model,
                    client=generation_client,
                    input_dir=generate_output_dir,
                    prompts_dir=args.prompts_dir,
                    generation_params=gen_model_hparams,
                )
                atomize_input_dir = cove_output_dir

            # In self_revision mode, perform a redaction step first and
            # atomize the redacted response instead of the original.
            elif args.abstraction_mode == 'self_revision' and args.experiment != 'sampling_baseline':
                print(f"{get_timestamp()} [3/5] Running self_revision (redaction pass)...")
                redact_output_dir = run_self_revision(
                    model=args.generating_model,
                    client=generation_client,
                    input_dir=generate_output_dir,
                    prompts_dir=args.prompts_dir,
                    generation_params=gen_model_hparams,
                    sys_prompt='self_revision_theta' if args.experiment == 'double_pass_theta' else 'self_revision',
                    theta=args.theta if args.experiment == 'double_pass_theta' else None
                )
                atomize_input_dir = redact_output_dir

                if args.experiment == 'iterative_self_revision':
                    K = args.iterative_k
                    current_dir = redact_output_dir
                    for k in range(2, K + 1):
                        print(f"{get_timestamp()} [3/5] Running iterative self_revision pass {k}/{K}...")
                        current_dir = run_self_revision(
                            model=args.generating_model,
                            client=generation_client,
                            input_dir=current_dir,
                            prompts_dir=args.prompts_dir,
                            generation_params=gen_model_hparams,
                            sys_prompt='self_revision',
                            theta=None,
                            output_subdir_name=f"iterative_self_revision_k={k}",
                        )
                    atomize_input_dir = current_dir

            print(f"{get_timestamp()} [3/5] Running atomize...")
            atomize_output_dir = run_atomize(
                client=generation_client,
                model=args.generating_model,
                sys_prompt=args.atomize_sys_prompt,
                input_dir=atomize_input_dir,
                prompts_dir=args.prompts_dir,
                generation_params=gen_model_hparams,
                reasoning_effort=args.reasoning_effort,
            )

            print(f"{get_timestamp()} [4/5] Running abstraction and confidence...")
            if args.experiment == 'atom_wise_self_revision':
                print(f"{get_timestamp()} [4/5] Running atom-wise self-revision...")
                atoms_confidence_file = run_atoms_confidence(
                    model=args.generating_model,
                    confidence_score=args.confidence_score,
                    sys_prompt=args.confidence_sys_prompt,
                    client=generation_client,
                    input_files={'atoms_file_path': atomize_output_dir},
                    prompts_dir=args.prompts_dir,
                    generation_params=gen_model_hparams,
                )
                revise_atoms_output_dir = run_revise_atoms(
                    model=args.generating_model,
                    client=generation_client,
                    atomize_output_dir=atomize_output_dir,
                    prompts_dir=args.prompts_dir,
                    generation_params=gen_model_hparams,
                    confidence_file_path=atoms_confidence_file,
                )
                abstract_output_file_path = build_trivial_abstractions_from_atoms(revise_atoms_output_dir)
                question_input_file = abstract_output_file_path
            elif args.abstraction_mode == 'none':
                # Only atoms and their confidence (no abstraction)
                atoms_conf_file_path = run_atoms_confidence(
                    model=args.generating_model,
                    confidence_score=args.confidence_score,
                    client=generation_client,
                    input_files={'atoms_file_path': atomize_output_dir},
                    prompts_dir=args.prompts_dir,
                    generation_params=gen_model_hparams
                )
                question_input_file = process_atoms_without_abstraction(atoms_conf_file_path)
            elif args.abstraction_mode == 'separate':
                # Separate: abstract first, then confidence for atoms and abstractions
                abstract_output_file_path = run_abstract(
                    model=args.generating_model,
                    sys_prompt=args.abstract_sys_prompt,
                    client=generation_client,
                    atomize_output_dir=atomize_output_dir,
                    prompts_dir=args.prompts_dir,
                    generation_params=gen_model_hparams,
                    reasoning_effort=args.reasoning_effort,
                )
                atoms_conf_file_path = run_atoms_confidence(
                    model=args.generating_model,
                    confidence_score=args.confidence_score,
                    client=generation_client,
                    input_files={'atoms_file_path': atomize_output_dir},
                    prompts_dir=args.prompts_dir,
                    generation_params=gen_model_hparams,
                    reasoning_effort=args.reasoning_effort,
                )
                abstractions_conf_file_path = run_abstractions_confidence(
                    model=args.generating_model,
                    confidence_score=args.confidence_score,
                    client=generation_client,
                    input_files={'abstract_file_path': abstract_output_file_path, 'atoms_conf_file_path': atoms_conf_file_path},
                    prompts_dir=args.prompts_dir,
                    generation_params=gen_model_hparams,
                    reasoning_effort=args.reasoning_effort,
                )
                # In 'separate' mode, ensure every abstraction has a valid confidence
                # whenever we are using a confidence_score that actually produces
                # confidence values (i.e., not pure logprobs-only runs).
                if args.confidence_score != 'logprobs':
                    validate_confidences_for_entity_separate(
                        abstract_output_file_path,
                        abstractions_conf_file_path,
                    )
                question_input_file = abstract_output_file_path
            elif args.abstraction_mode == 'self_revision':
                # In self_revision mode, we do not perform real abstraction
                # or confidence scoring. Instead, we build trivial abstractions
                # from the atoms of the redacted response (atom-as-its-own-
                # abstraction) and feed those into the questions pipeline.
                abstract_output_file_path = build_trivial_abstractions_from_atoms(
                    atomize_output_dir
                )
                question_input_file = abstract_output_file_path

            if args.abstraction_mode != 'none':                    
                validation_checkpoint(atomize_output_dir, abstract_output_file_path, entity_id)

            if args.run_questions:
                print(f"{get_timestamp()} [5/5] Running count_questions...")
                questions_output_dir = run_count_questions(
                    model=args.questions_model,
                    client=init_client(args.questions_model, args.questions_model_base_url),
                    sys_prompt=args.questions_sys_prompt,
                    input_file=question_input_file,
                    prompts_dir=args.prompts_dir,
                    generation_params=questions_model_hparams,
                )
                # Validate questions.json integrity (fail entity on error)
                questions_json_path = os.path.join(questions_output_dir, "questions.json")
                print(f"Questions saved to:\n{questions_json_path}")
                validate_questions_for_entity(question_input_file, questions_json_path)
            entity_duration = time.time() - entity_start_time
            print(f"{get_timestamp()} === Finished entity: {entity_id} - Time taken: {format_duration(entity_duration)} ===")
        except Exception as e:
            print(f"{get_timestamp()} Error processing entity {entity_id}: {e}")
            print(traceback.format_exc())
            continue

    print(f"{get_timestamp()} Pipeline complete.")


def get_entity_list(args): 
    """
    Build the list of entities to process and, for some datasets, an accompanying
    metadata mapping keyed by entity ID.

    Returns:
        entity_list: List[str]
        start_idx: int
        end_idx: int
        original_length: int
        entity_metadata: Optional[Dict[str, dict]]
    """
    entity_list = []
    entity_metadata = None
    
    # Check if specific entities file is provided
    if args.entities_file and os.path.exists(args.entities_file):
        # Use entities from the specified file
        with open(args.entities_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    entity_list.append(line)
        print(f"Processing {len(entity_list)} entities from file: {args.entities_file}")

        # For datasets backed by sampled prompts (currently longfact_objects_sampled,
        # non_factual), also build metadata so we can still map entity IDs
        # back to the original prompts.
        if args.dataset in ('longfact_objects_sampled', 'non_factual'):
            sampled_path = args.sampled_prompts_file
            try:
                prompt_entries = load_json(sampled_path)
            except Exception as e:
                print(f"Error loading sampled prompts from {sampled_path}: {e}")
                prompt_entries = []

            if not isinstance(prompt_entries, list):
                print(f"Error: sampled_prompts_file {sampled_path} does not contain a JSON list.")
                prompt_entries = []

            entity_metadata = {}
            # Build a mapping for all known IDs, then filter to the requested subset.
            for idx, entry in enumerate(prompt_entries):
                if isinstance(entry, dict):
                    entity_id = entry.get('id') or entry.get('entity_id')
                    subject = entry.get('subject')
                    index = entry.get('index')
                    if not entity_id and subject is not None and index is not None:
                        entity_id = f"{subject}_{index}"
                    if not entity_id:
                        entity_id = f"prompt_{idx}"
                    if 'prompt' not in entry and 'text' in entry:
                        entry = {**entry, 'prompt': entry.get('text')}
                    entity_metadata[entity_id] = entry
                elif isinstance(entry, str):
                    entity_id = f"prompt_{idx}"
                    entity_metadata[entity_id] = {'prompt': entry}

            # Warn if any requested entities are missing from metadata
            missing = [eid for eid in entity_list if eid not in entity_metadata]
            if missing:
                raise ValueError(
                    f"Error: {len(missing)} entities from {args.entities_file} were "
                    f"not found in sampled prompts."
                )
        # Set variables for entities_file mode
        start_idx = 0  # Not used in this mode, but needed for compatibility
        original_length = len(entity_list)  # Total entities being processed
        end_idx = original_length
    else:
        # Use default dataset entities with index slicing
        if args.dataset == 'factscore_labeled':
            with open(os.path.join('datasets', args.dataset, 'entities.txt'), 'r') as f:
                for line in f:
                    entity_list.append(line.strip())
        elif args.dataset == 'longfact_objects_sampled':
            # Load sampled prompts and derive stable entity IDs of the form
            # "{subject}_{index}", using the sampled index field.
            sampled_path = args.sampled_prompts_file
            try:
                longfact_entries = load_json(sampled_path)
            except Exception as e:
                print(f"Error loading sampled prompts from {sampled_path}: {e}")
                longfact_entries = []

            if not isinstance(longfact_entries, list):
                print(f"Error: sampled_prompts_file {sampled_path} does not contain a JSON list.")
                longfact_entries = []

            entity_metadata = {}
            for entry in longfact_entries:
                subject = entry.get('subject')
                index = entry.get('index')
                if subject is None:
                    continue
                if index is None:
                    # If index is missing, we cannot form a stable ID; skip.
                    continue
                entity_id = f"{subject}_{index}"
                entity_list.append(entity_id)
                entity_metadata[entity_id] = entry
        elif args.dataset == 'non_factual':
            sampled_path = args.sampled_prompts_file
            try:
                prompt_entries = load_json(sampled_path)
            except Exception as e:
                print(f"Error loading sampled prompts from {sampled_path}: {e}")
                prompt_entries = []

            if not isinstance(prompt_entries, list):
                print(f"Error: sampled_prompts_file {sampled_path} does not contain a JSON list.")
                prompt_entries = []

            entity_metadata = {}
            for idx, entry in enumerate(prompt_entries):
                if isinstance(entry, dict):
                    entity_id = entry.get('id') or entry.get('entity_id')
                    subject = entry.get('subject')
                    index = entry.get('index')
                    if not entity_id and subject is not None and index is not None:
                        entity_id = f"{subject}_{index}"
                    if not entity_id:
                        entity_id = f"prompt_{idx}"
                    if 'prompt' not in entry and 'text' in entry:
                        entry = {**entry, 'prompt': entry.get('text')}
                    entity_list.append(entity_id)
                    entity_metadata[entity_id] = entry
                elif isinstance(entry, str):
                    entity_id = f"prompt_{idx}"
                    entity_list.append(entity_id)
                    entity_metadata[entity_id] = {'prompt': entry}

        # Slice entity list based on provided indices
        original_length = len(entity_list)
        start_idx = args.start_index
        
        # Determine end index
        if args.count is not None:
            end_idx = min(start_idx + args.count, original_length)
        else:
            end_idx = original_length
        
        # Validate indices
        if start_idx >= original_length:
            print(f"Error: start_index ({start_idx}) is greater than or equal to total entities ({original_length})")
            return [], start_idx, start_idx, original_length, entity_metadata
        
        if start_idx < 0:
            print(f"Error: start_index ({start_idx}) cannot be negative")
            return [], start_idx, start_idx, original_length, entity_metadata
            
        entity_list = entity_list[start_idx:end_idx]
        
        print(f"Processing entities {start_idx} to {end_idx-1} (total: {len(entity_list)} out of {original_length})")
    
    return entity_list, start_idx, end_idx, original_length, entity_metadata

if __name__ == "__main__":
    main() 
