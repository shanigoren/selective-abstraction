import os
import json
import re
from generation_utils import *
from utils.text_utils import *
from extract_logprobs import token_logprobs_as_dicts
from confidence_utils import _count_sentence_groups, _find_upwards_file, parse_confidence


def run_atoms_confidence(model, confidence_score, sys_prompt=None, client=None, input_files=None, prompts_dir=None, generation_params=None, logprobs_strategy='all_statements', logprobs_chunking=True, logprobs_chunk_size=50, reasoning_effort=None):
    print(f'Atoms Confidence score: {confidence_score}')
    if confidence_score not in ('verbalized', 'logprobs'):
        raise ValueError(
            "The public paper pipeline supports verbalized or logprobs confidence only."
        )
    if confidence_score == 'verbalized':
        sys_prompt = 'verbal_atoms_reasoning'
    naming_config = {
        'confidence_score': confidence_score,
        'sys_prompt': sys_prompt
    }
    if reasoning_effort is not None:
        naming_config['reasoning_effort'] = reasoning_effort
    input_dir = input_files['atoms_file_path']
    output_dir = os.path.join(input_dir, "atoms_confidence", naming_config_to_str(naming_config))
    os.makedirs(output_dir, exist_ok=True)

    atoms_file_path = os.path.join(input_dir, 'atoms.json')
    if not os.path.exists(atoms_file_path):
        print(f"Atoms file not found in {input_dir}. Skipping...")
        return

    # Logprobs branch writes under atoms_confidence/logprobs/<config>/atoms_logprobs.json
    if confidence_score == 'logprobs':
        # Resolve output path and verify completeness before skipping
        logprobs_config = {
            "delims": "dot_newline",
            "strategy": logprobs_strategy
        }
        if logprobs_strategy == 'all_statements':
            logprobs_config["chunking"] = "auto" if logprobs_chunking else "none"
            if logprobs_chunking:
                logprobs_config["chunk_size"] = logprobs_chunk_size
        atoms_lp_dir = os.path.join(input_dir, "atoms_confidence", "logprobs", naming_config_to_str(logprobs_config))
        atoms_lp_path = os.path.join(atoms_lp_dir, "atoms_logprobs.json")
        entity_atoms = load_json(atoms_file_path)
        if os.path.exists(atoms_lp_path):
            try:
                existing = load_json(atoms_lp_path)
            except Exception:
                existing = None
            complete = isinstance(existing, list) and len(existing) == len(entity_atoms) and all(
                isinstance(entry, dict) and isinstance(entry.get('logprobs'), list)
                for entry in existing
            )
            if complete:
                print(f"Atoms logprobs already complete. Skipping... ({atoms_lp_path})")
                return atoms_lp_path
            else:
                print(f"Existing atoms logprobs incomplete; regenerating ({atoms_lp_path}).")
        # Load original generation prompt from generate_response/response.json (search upwards)
        gen_dir = _find_upwards_file(input_dir, "response.json", max_levels=4) or os.path.dirname(input_dir)
        gen_resp_path = os.path.join(gen_dir, "response.json")
        if not os.path.exists(gen_resp_path):
            print(f"Generation response.json not found when extracting logprobs. Searched under {gen_dir}.")
            return
        gen_resp = load_json(gen_resp_path)
        try:
            user_prompt = gen_resp["config"]["generate_prompt"]
        except Exception:
            print(f"Could not locate generate_prompt in response.json. Aborting logprobs extraction for atoms.")
            return
        # Strategy branch: current implementation is 'all_statements'
        if logprobs_strategy == 'all_statements':
            # Prepare entries list (one per atom) with de-bracketed text
            entries = []
            for i, atom in enumerate(entity_atoms):
                entries.append({
                    "index": i,
                    "text_raw": atom,
                    "text_debr": remove_first_last_brackets(atom)
                })
            # Build batches of entries
            batches = []
            if logprobs_chunking:
                if logprobs_chunk_size is None:
                    raise ValueError("logprobs_chunk_size must be provided when logprobs_chunking is True.")
                current = []
                current_len = 0
                for e in entries:
                    if current_len > 0 and (current_len + 1) > max(1, int(logprobs_chunk_size)):
                        batches.append(current)
                        current = []
                        current_len = 0
                    current.append(e)
                    current_len += 1
                if current:
                    batches.append(current)
            else:
                if entries:
                    batches = [entries]
            # Prepare output container, preserving original order
            results = [None] * len(entity_atoms)
            # Execute batch requests and map back
            for batch_idx, batch in enumerate(batches):
                assistant_content = "\n".join(stmt["text_debr"] for stmt in batch).strip()
                if len(assistant_content) == 0:
                    continue
                # Ensure provider receives prompt_logprobs request
                logprobs_params = dict(generation_params or {})
                logprobs_params["prompt_logprobs"] = True
                completion = get_model_completion(
                    client=client,
                    prompt=assistant_content,
                    model=model,
                    generation_params=logprobs_params,
                    extract_prompt_logprobs=True,
                    extract_prompt_logprobs_user_prompt=user_prompt
                )
                prompt_logprobs = completion.get("prompt_logprobs")
                if prompt_logprobs is None:
                    print(f"prompt_logprobs missing in model response for batch {batch_idx}. Skipping batch.")
                    continue
                # Group tokens by boundaries and then map back to per-entry
                grouped = token_logprobs_as_dicts(prompt_logprobs, prompt_text=assistant_content, sentence_terminators=['.'])
                grouped = [group for group in grouped if len(group) > 0]
                g_idx = 0
                for stmt in batch:
                    seg_text = stmt["text_debr"]
                    num_groups = _count_sentence_groups(seg_text)
                    if num_groups == 0:
                        num_groups = 1
                    segs = grouped[g_idx:g_idx + num_groups]
                    g_idx += num_groups
                    flat_tokens = []
                    for sent in segs:
                        flat_tokens.extend(sent if isinstance(sent, list) else [])
                    i = stmt["index"]
                    results[i] = {
                        "atom": entity_atoms[i],
                        "index": i,
                        "logprobs": flat_tokens
                    }
            os.makedirs(atoms_lp_dir, exist_ok=True)
            # Collapse any None (unprocessed) into empty lists to maintain shape
            final_results = []
            for i, res in enumerate(results):
                if res is None:
                    final_results.append({
                        "atom": entity_atoms[i],
                        "index": i,
                        "logprobs": []
                    })
                else:
                    final_results.append(res)
            with open(atoms_lp_path, 'w') as f:
                json.dump(final_results, f, indent=4)
            return atoms_lp_path
        else:
            raise NotImplementedError(f"Unsupported logprobs strategy for atoms: {logprobs_strategy}")

    output_file_path = os.path.join(output_dir, "atoms_confidence.json")
    if os.path.exists(output_file_path):
        atoms_confidence_dict = load_json(output_file_path)
        entity_atoms = load_json(atoms_file_path)
        if len(atoms_confidence_dict) == len(set(entity_atoms)):
            print(f"Atoms confidence already exists. Skipping...")
            return output_file_path
        else:
            print(f"Atoms confidence already exists but number of atoms do not match.\natoms: {len(entity_atoms)}, atoms confidence: {len(atoms_confidence_dict)}\nRemoving old atoms confidence file and running again...")
            os.remove(output_file_path)

    entity_atoms = load_json(atoms_file_path)
    system_prompt_path = f"{prompts_dir}/confidence/sys_prompts/{sys_prompt}.txt"
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read()

    call_costs = []
    print(f'Confidence score: {confidence_score}')
    numbered_entity_atoms = create_numbered_list(entity_atoms)
    confidence_response_dict = get_model_completion(
        client=client,
        system_prompt=system_prompt,
        prompt=numbered_entity_atoms,
        model=model,
        generation_params=generation_params,
        reasoning_effort=reasoning_effort,
    )
    if reasoning_effort is not None:
        call_costs.append(extract_call_cost(confidence_response_dict))
    response_text = get_response_text(confidence_response_dict, model)
    confidence_response = normalize_spaces(response_text)
    atoms_confidence_dict = parse_confidence(confidence_response, entity_atoms)
    assert len(atoms_confidence_dict) == len(set(entity_atoms)), f"Number of atoms and confidence scores do not match. atoms: {len(entity_atoms)}, confidence scores: {len(atoms_confidence_dict)}"
    if all(score == 100 for score in atoms_confidence_dict.values()):
        print(f"Warning: All confidence scores are 100.")

    with open(output_file_path, 'w') as f:
        json.dump(atoms_confidence_dict, f, indent=4)

    if reasoning_effort is not None and call_costs:
        stage_cost = {
            'stage': 'atoms_confidence',
            'model': model,
            'reasoning_effort': reasoning_effort,
            **aggregate_costs(call_costs),
        }
        with open(os.path.join(output_dir, 'cost.json'), 'w') as f:
            json.dump(stage_cost, f, indent=4)

    return output_file_path


