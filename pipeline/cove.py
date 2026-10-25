"""
Chain-of-Verification (CoVe) — Factor+Revise variant.

Reference: Dhuliawala et al., 2023 "Chain-of-Verification Reduces Hallucination in
Large Language Models" (https://arxiv.org/abs/2309.11495)

Implementation notes:
- Step 1: Baseline response (reused from generate_response stage).
- Step 2: For each sentence, plan verification questions independently.
- Step 3: Answer each question in a separate LLM call WITHOUT the original response
          in context (factored execution), preventing the model from repeating its
          own hallucinations.
- Step 3b: Cross-check each (original sentence, question, answer) triple and label
           it CONSISTENT / INCONSISTENT / PARTIALLY CONSISTENT.
- Step 4: For each sentence with inconsistencies, generate a revised sentence using
          only the consistent information.  Sentences with no inconsistencies are
          kept as-is without an extra LLM call.
"""

import os
import json
import re
from typing import List, Tuple, Optional, Dict

from generation_utils import (
    get_model_completion,
    get_response_text,
    is_openai_model,
)
from utils.text_utils import load_json, normalize_spaces


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> List[str]:
    """
    Split a response into sentences for per-sentence CoVe processing.

    Strategy:
    1. Split on newlines to respect list items and paragraph structure.
    2. Within each non-empty line, split further on sentence boundaries
       (`.` `!` `?` followed by whitespace and an uppercase letter).
       This simple heuristic avoids splitting on common abbreviations such as
       "Dr." or "U.S." because those are rarely followed directly by an
       uppercase word that starts a new sentence without a subject.
    """
    sentences: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Split sentence boundaries within the line
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', line)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_verification_questions(model_output: str) -> List[str]:
    """Parse a numbered list of verification questions from model output."""
    if not model_output or not model_output.strip():
        return []

    stripped = model_output.strip()

    # Model signalled no verifiable claims
    if re.search(r'\bNO_QUESTIONS\b', stripped, re.IGNORECASE):
        return []

    questions: List[str] = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept "1. ...", "1) ...", or "- ..." prefixes
        m = re.match(r'^(?:\d+[.)]\s+|-\s+)(.*)', line)
        if m:
            q = m.group(1).strip()
            if q:
                questions.append(q)
        else:
            # Accept bare lines that look like questions (contain "?")
            if '?' in line:
                questions.append(line)

    return questions


def _parse_cross_check_result(
    model_output: str,
    original_fact: str,
) -> Tuple[str, Optional[str]]:
    """
    Parse the cross-check model output into (label, fact_text).

    Returns:
        label       : 'CONSISTENT' | 'INCONSISTENT' | 'PARTIALLY_CONSISTENT'
        fact_text   : the (possibly trimmed) fact to keep, or None if INCONSISTENT
    """
    if not model_output:
        return 'CONSISTENT', original_fact

    text = model_output.strip()

    # PARTIALLY CONSISTENT must be checked before CONSISTENT to avoid prefix clash
    if re.match(r'^PARTIALLY[\s_]CONSISTENT\.?', text, re.IGNORECASE):
        fact = re.sub(r'^PARTIALLY[\s_]CONSISTENT\.?\s*', '', text, flags=re.IGNORECASE).strip()
        return 'PARTIALLY_CONSISTENT', fact if fact else original_fact

    if re.match(r'^CONSISTENT\.?', text, re.IGNORECASE):
        fact = re.sub(r'^CONSISTENT\.?\s*', '', text, flags=re.IGNORECASE).strip()
        return 'CONSISTENT', fact if fact else original_fact

    if re.match(r'^INCONSISTENT\.?', text, re.IGNORECASE):
        return 'INCONSISTENT', None

    # Fallback: scan for keywords anywhere in the text
    upper = text.upper()
    if 'INCONSISTENT' in upper and 'PARTIALLY' not in upper:
        return 'INCONSISTENT', None
    if 'PARTIALLY' in upper and 'CONSISTENT' in upper:
        return 'PARTIALLY_CONSISTENT', original_fact
    if 'CONSISTENT' in upper:
        return 'CONSISTENT', original_fact

    # Default: trust the original fact when parsing fails
    return 'CONSISTENT', original_fact


# ---------------------------------------------------------------------------
# Building a response dict compatible with get_response_text()
# ---------------------------------------------------------------------------

def _build_response_dict(model: str, text: str, base_config: dict) -> dict:
    """
    Construct a minimal response dict that get_response_text() can parse,
    mirroring the structure returned by get_model_completion() for each
    model family.
    """
    if is_openai_model(model):
        return {
            "output": [{"content": [{"text": text}]}],
            "config": base_config,
        }
    # vLLM / OpenAI-compatible endpoint
    return {
        "choices": [{"message": {"content": text}}],
        "config": base_config,
    }


# ---------------------------------------------------------------------------
# Per-step LLM calls
# ---------------------------------------------------------------------------

def _plan_verification_questions(
    sentence: str,
    client,
    model: str,
    system_prompt: str,
    generation_params: dict,
    original_question: Optional[str] = None,
) -> List[str]:
    """Step 2: Generate verification questions for a single sentence."""
    if original_question:
        prompt = f"Context: Q: {original_question}\n\nSentence: {sentence}"
    else:
        prompt = f"Sentence: {sentence}"
    response = get_model_completion(
        client=client,
        system_prompt=system_prompt,
        prompt=prompt,
        model=model,
        generation_params=generation_params,
    )
    output = get_response_text(response, model)
    return _parse_verification_questions(output)


def _execute_verification(
    question: str,
    client,
    model: str,
    system_prompt: str,
    generation_params: dict,
) -> str:
    """
    Step 3: Answer a verification question independently.

    Crucially, the original baseline response is NOT in context here — only
    the question itself.  This prevents the model from simply parroting its
    own potentially hallucinated answer.
    """
    response = get_model_completion(
        client=client,
        system_prompt=system_prompt,
        prompt=question,
        model=model,
        generation_params=generation_params,
    )
    return get_response_text(response, model).strip()


def _cross_check_consistency(
    original_fact: str,
    question: str,
    answer: str,
    client,
    model: str,
    system_prompt: str,
    generation_params: dict,
) -> Tuple[str, Optional[str]]:
    """Step 3b: Determine whether the original fact is consistent with the verification answer."""
    prompt = (
        f"Context: {original_fact}\n"
        f"From another source,\n"
        f"{question}: {answer}\n"
        f"Response:"
    )
    response = get_model_completion(
        client=client,
        system_prompt=system_prompt,
        prompt=prompt,
        model=model,
        generation_params=generation_params,
    )
    output = get_response_text(response, model).strip()
    return _parse_cross_check_result(output, original_fact)


def _generate_final_passage(
    original_passage: str,
    qa_cross_checks: List[Dict],
    client,
    model: str,
    system_prompt: str,
    generation_params: dict,
) -> str:
    """
    Step 4: Generate a revised passage that retains only consistent information.

    The prompt provides the original passage together with each verification
    Q&A pair and its consistency label so the model knows exactly what to
    keep, trim, or drop.
    """
    check_lines = []
    for item in qa_cross_checks:
        check_lines.append(
            f"{item['question']}: {item['answer']} [{item['label']}]"
        )

    prompt = (
        f"Context: {original_passage}\n"
        f"From another source,\n"
        + "\n".join(check_lines)
        + "\nResponse:"
    )
    response = get_model_completion(
        client=client,
        system_prompt=system_prompt,
        prompt=prompt,
        model=model,
        generation_params=generation_params,
    )
    return get_response_text(response, model).strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_cove(
    model: str,
    client,
    input_dir: str,
    prompts_dir: str,
    generation_params: dict,
) -> str:
    """
    Run the CoVe Factor+Revise pipeline over an existing generated response.

    Args:
        model             : Model name (used for response parsing and LLM calls).
        client            : Initialised LLM client.
        input_dir         : Directory containing the baseline ``response.json``
                            (i.e. the generate_response output directory).
        prompts_dir       : Root prompts directory (``pipeline/prompts``).
        generation_params : Model hyperparameters dict.

    Returns:
        Path to the CoVe output directory, which contains:
          - ``response.json``  — verified response in the standard format
          - ``cove_log.json``  — per-sentence verification log
    """
    # ------------------------------------------------------------------
    # Load prompts
    # ------------------------------------------------------------------
    cove_prompts_dir = os.path.join(prompts_dir, "cove")

    def _load_prompt(name: str) -> str:
        path = os.path.join(cove_prompts_dir, f"{name}.txt")
        with open(path, "r") as fh:
            return fh.read()

    plan_prompt = _load_prompt("plan_verification")
    execute_prompt = _load_prompt("execute_verification")
    cross_check_prompt = _load_prompt("cross_check")
    final_revision_prompt = _load_prompt("final_revision")

    # ------------------------------------------------------------------
    # Set up output directory
    # ------------------------------------------------------------------
    output_dir = os.path.join(input_dir, "cove")
    os.makedirs(output_dir, exist_ok=True)

    output_response_path = os.path.join(output_dir, "response.json")
    if os.path.exists(output_response_path):
        print(
            f"CoVe response already exists at {output_response_path}. "
            f"Skipping CoVe step."
        )
        return output_dir

    # ------------------------------------------------------------------
    # Load baseline response
    # ------------------------------------------------------------------
    base_response_path = os.path.join(input_dir, "response.json")
    if not os.path.exists(base_response_path):
        raise FileNotFoundError(
            f"Base response.json not found at {base_response_path}"
        )

    base_response = load_json(base_response_path)
    base_config = base_response.get("config", {})
    response_text = normalize_spaces(get_response_text(base_response, model))

    # Extract the original generation prompt for Step 2 context
    original_question = base_config.get("generate_prompt", "") or ""

    # ------------------------------------------------------------------
    # Step 2–4: Process sentence by sentence
    # ------------------------------------------------------------------
    sentences = _split_into_sentences(response_text)
    print(f"CoVe: processing {len(sentences)} sentences.")

    verified_sentences: List[str] = []
    cove_log: List[Dict] = []

    for sent_idx, sentence in enumerate(sentences):
        print(
            f"  [{sent_idx + 1}/{len(sentences)}] "
            f"{sentence[:70]}{'...' if len(sentence) > 70 else ''}"
        )
        sent_log: Dict = {"sentence": sentence, "questions": [], "skipped": False}

        # Step 2: Plan verification questions (with original question as context)
        questions = _plan_verification_questions(
            sentence=sentence,
            client=client,
            model=model,
            system_prompt=plan_prompt,
            generation_params=generation_params,
            original_question=original_question,
        )

        if not questions:
            print(f"    → No verification questions. Keeping sentence as-is.")
            verified_sentences.append(sentence)
            sent_log["skipped"] = True
            sent_log["verified_sentence"] = sentence
            cove_log.append(sent_log)
            continue

        print(f"    → {len(questions)} verification question(s).")

        # Steps 3 + 3b: Execute each question and cross-check
        qa_cross_checks: List[Dict] = []
        for q_idx, question in enumerate(questions):
            # Step 3: factored execution (no original response in context)
            answer = _execute_verification(
                question=question,
                client=client,
                model=model,
                system_prompt=execute_prompt,
                generation_params=generation_params,
            )

            # Step 3b: cross-check consistency
            label, consistent_fact = _cross_check_consistency(
                original_fact=sentence,
                question=question,
                answer=answer,
                client=client,
                model=model,
                system_prompt=cross_check_prompt,
                generation_params=generation_params,
            )

            print(f"      Q{q_idx + 1}: {label}")
            qa_cross_checks.append(
                {
                    "question": question,
                    "answer": answer,
                    "label": label,
                    "consistent_fact": consistent_fact,
                }
            )

        sent_log["questions"] = qa_cross_checks

        # Step 4: Revise if any inconsistencies found
        has_inconsistency = any(
            item["label"] in ("INCONSISTENT", "PARTIALLY_CONSISTENT")
            for item in qa_cross_checks
        )

        if not has_inconsistency:
            # All claims verified as consistent — keep the original sentence
            verified_sentence = sentence
        else:
            verified_sentence = _generate_final_passage(
                original_passage=sentence,
                qa_cross_checks=qa_cross_checks,
                client=client,
                model=model,
                system_prompt=final_revision_prompt,
                generation_params=generation_params,
            )
            print(f"    → Revised sentence generated.")

        if verified_sentence:
            verified_sentences.append(verified_sentence)

        sent_log["verified_sentence"] = verified_sentence
        cove_log.append(sent_log)

    # ------------------------------------------------------------------
    # Combine verified sentences into the final response
    # ------------------------------------------------------------------
    final_text = " ".join(verified_sentences)

    cove_config = {
        "method": "cove",
        "base_generate_config": base_config,
    }
    verified_response = _build_response_dict(model, final_text, cove_config)

    with open(output_response_path, "w") as fh:
        json.dump(verified_response, fh, indent=4)

    log_path = os.path.join(output_dir, "cove_log.json")
    with open(log_path, "w") as fh:
        json.dump(cove_log, fh, indent=4)

    print(f"CoVe response saved to {output_response_path}.")
    return output_dir
