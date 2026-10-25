import json
import os
import ast
from typing import Any, Dict

from generation_utils import *
from utils.text_utils import normalize_spaces


def _load_response_json(input_dir: str) -> Dict[str, Any]:
    """
    Load the existing response.json for a generated entity.
    """
    response_path = os.path.join(input_dir, "response.json")
    if not os.path.exists(response_path):
        raise FileNotFoundError(f"response.json not found at {response_path}")

    with open(response_path, "r") as f:
        return json.load(f)


def _save_response_json(input_dir: str, data: Dict[str, Any]) -> None:
    response_path = os.path.join(input_dir, "response.json")
    with open(response_path, "w") as f:
        json.dump(data, f, indent=4)


def _simplify_answer_for_classification(answer: str, max_chars: int = 2000) -> str:
    """
    Simplify the answer content before sending it to the classifier LLM.

    The classifier only needs to know whether the answer is a refusal vs a
    substantive response, not every formatting/detail. To avoid backend issues
    with very long markdown tables or HTML-like markup, we:
      - Drop markdown table rows (lines that look like `| ... |`).
      - Drop purely decorative separator lines.
      - Truncate to a safe maximum character length.
    """
    lines = answer.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Drop markdown table rows (and headers/separators).
        if stripped.startswith("|") and "|" in stripped:
            continue
        if set(stripped) <= {"|", "-", ":"}:
            continue
        kept.append(stripped)

    simplified = "\n".join(kept)
    if len(simplified) > max_chars:
        simplified = simplified[:max_chars]
    return simplified


def run_refusal_classification(
    client,
    model: str,
    input_dir: str,
    prompts_dir: str,
    generation_params: Dict[str, Any],
    classifier_prompt_name: str = "refusal_classifier",
) -> bool:
    """
    Classify the generated response as a refusal or a valid answer.

    This function:
      - Loads the existing response.json under `input_dir`.
      - If `is_refusal` is already present, returns it (idempotent behavior).
      - Otherwise, calls the generating LLM with a classification prompt.
      - Writes `is_refusal` (and optional helper fields) back into response.json.

    Returns:
      bool: True if the response is classified as a refusal, False otherwise.
    """
    try:
        response_data = _load_response_json(input_dir)
    except FileNotFoundError as e:
        print(f"[refusal_classifier] {e}")
        return False

    # Idempotent behavior: if already classified, just return the stored value.
    if "is_refusal" in response_data:
        stored_val = response_data["is_refusal"]
        # Be lenient about type here and coerce to bool if possible.
        if isinstance(stored_val, bool):
            return stored_val
        if isinstance(stored_val, str):
            lowered = stored_val.lower()
            if lowered in ("true", "yes", "1"):
                return True
            if lowered in ("false", "no", "0"):
                return False
        # Fallback if ambiguous.
        return False

    # Extract question and answer text from response.json
    try:
        question = response_data.get("config", {}).get("generate_prompt", "")
        answer = get_response_text(response_data, model)
    except Exception as e:
        print(f"[refusal_classifier] Failed to extract question/answer: {e}")
        return False

    # If we don't have an answer, treat as non-refusal for control flow.
    if not answer:
        print("[refusal_classifier] Empty answer content; treating as non-refusal.")
        response_data["is_refusal"] = False
        _save_response_json(input_dir, response_data)
        return False

    # Normalize known problematic unicode spaces before sending back to the model.
    # Other stages (e.g., atomization) do this via normalize_spaces; we mirror that
    # behavior here to avoid backend issues with certain unicode whitespace.
    norm_question = normalize_spaces(question or "")
    norm_answer = normalize_spaces(answer or "")

    # Load the classifier prompt template.
    prompt_path = os.path.join(
        prompts_dir, "classification", f"{classifier_prompt_name}.txt"
    )
    if not os.path.exists(prompt_path):
        print(
            f"[refusal_classifier] Classifier prompt file not found at {prompt_path}. "
            "Treating as non-refusal."
        )
        response_data["is_refusal"] = False
        _save_response_json(input_dir, response_data)
        return False

    with open(prompt_path, "r") as f:
        # Use the full instructions file as the system prompt.
        system_prompt = f.read()

    # First attempt: use the full normalized answer.
    classifier_prompt = f"QUESTION:\n{norm_question}\n\nANSWER:\n{norm_answer}"

    try:
        completion = get_model_completion(
            client=client,
            prompt=classifier_prompt,
            model=model,
            generation_params=generation_params,
            system_prompt=system_prompt,
        )
    except Exception as e:
        print(f"[refusal_classifier] Error calling classifier model (first attempt): {e}")
        # Second attempt: simplify the answer (e.g., drop markdown tables) and retry once.
        try:
            simplified_answer = _simplify_answer_for_classification(norm_answer)
            fallback_prompt = f"QUESTION:\n{norm_question}\n\nANSWER:\n{simplified_answer}"
            completion = get_model_completion(
                client=client,
                prompt=fallback_prompt,
                model=model,
                generation_params=generation_params,
                system_prompt=system_prompt,
            )
        except Exception as e2:
            print(f"[refusal_classifier] Error calling classifier model (fallback): {e2}")
            response_data["is_refusal"] = False
            _save_response_json(input_dir, response_data)
            return False

    # Extract the classifier's JSON content.
    classifier_content = get_response_text(completion, model)
    classifier_content_stripped = classifier_content.strip()

    # Try strict JSON first; many models will follow the instructions.
    classification = None
    if classifier_content_stripped:
        try:
            classification = json.loads(classifier_content_stripped.replace('"',''))
        except Exception:
            # Some models may emit Python-style dicts with single quotes.
            try:
                possible = ast.literal_eval(classifier_content_stripped)
                if isinstance(possible, dict):
                    classification = possible
            except Exception:
                classification = None

    if not isinstance(classification, dict):
        # As a very last resort, fall back to a simple textual heuristic on the raw output.
        upper = classifier_content_stripped.upper()
        if "REFUSAL" in upper:
            label = "REFUSAL"
        else:
            label = "VALID"
        classification = {
            "label": label,
            "explanation": classifier_content_stripped,
        }

    label = str(classification.get("label", "") or "").strip().upper()
    is_refusal = label == "REFUSAL"

    # Persist results back into response.json
    response_data["is_refusal"] = is_refusal
    # Optionally keep the raw classification for debugging/analysis.
    response_data["refusal_classification"] = classification

    _save_response_json(input_dir, response_data)

    return is_refusal




