import os
import json
from generation_utils import *
from utils.text_utils import *
from extract_logprobs import token_logprobs_as_dicts
from confidence_utils import _count_sentence_groups, _find_upwards_file, parse_confidence


def run_abstractions_confidence(model, confidence_score, sys_prompt=None, client=None, input_files=None, prompts_dir=None, generation_params=None, logprobs_strategy='per_atom', logprobs_chunking=False, logprobs_chunk_size=None, reasoning_effort=None):
    print(f'Abstractions Confidence score: {confidence_score}')
    if confidence_score not in ('verbalized', 'logprobs'):
        raise ValueError(
            "The public paper pipeline supports verbalized or logprobs confidence only."
        )
    if confidence_score == 'verbalized':
        sys_prompt = 'verbal_abstractions_reasoning_w_atom_conf'
    naming_config = {
        'confidence_score': confidence_score,
        'sys_prompt': sys_prompt
    }
    if reasoning_effort is not None:
        naming_config['reasoning_effort'] = reasoning_effort
    abstractions_input_path = input_files['abstract_file_path']
    if not os.path.exists(abstractions_input_path):
        print(f"Abstractions file not found in {os.path.dirname(abstractions_input_path)}. Skipping...")
        return
    # Load abstractions produced by run_abstract:
    abstractions_data = load_json(abstractions_input_path)

    input_dir = os.path.dirname(abstractions_input_path)
    # Logprobs branch: write under abstractions_confidence/logprobs/<config>/abstractions_logprobs.json
    if confidence_score == 'logprobs':
        # Load original generation prompt
        gen_dir = _find_upwards_file(input_dir, "response.json", max_levels=6)
        if not gen_dir:
            print(f"Generation response.json not found when extracting logprobs for abstractions (starting at {input_dir}).")
            return
        gen_resp = load_json(os.path.join(gen_dir, "response.json"))
        try:
            user_prompt = gen_resp["config"]["generate_prompt"]
        except Exception:
            print(f"Could not locate generate_prompt in response.json. Aborting logprobs extraction for abstractions.")
            return
        # Prepare results list (resume if exists)
        logprobs_config = {
            "delims": "dot_newline",
            "strategy": logprobs_strategy,
        }
        if logprobs_strategy == 'all_statements':
            logprobs_config["chunking"] = "auto" if logprobs_chunking else "none"
            if logprobs_chunking:
                logprobs_config["chunk_size"] = logprobs_chunk_size
        abs_lp_dir = os.path.join(input_dir, "abstractions_confidence", "logprobs", naming_config_to_str(logprobs_config))
        os.makedirs(abs_lp_dir, exist_ok=True)
        abs_lp_path = os.path.join(abs_lp_dir, "abstractions_logprobs.json")
        results = []
        if os.path.exists(abs_lp_path):
            try:
                existing = load_json(abs_lp_path)
                results = existing if isinstance(existing, list) else []
            except Exception:
                results = []
            by_atom = {normalize_spaces(e.get('atom')): e for e in results if isinstance(e, dict)}
            def _entry_complete(atom_text_raw, abstractions_for_atom):
                entry = by_atom.get(normalize_spaces(atom_text_raw))
                if not entry:
                    return False
                abs_entries = entry.get('abstractions')
                if not isinstance(abs_entries, list) or len(abs_entries) != len(abstractions_for_atom):
                    return False
                return all(isinstance(ae, dict) and isinstance(ae.get('logprobs'), list) for ae in abs_entries)
            all_complete = all(_entry_complete(ad.get('atom'), ad.get('abstractions', [])) for ad in abstractions_data)
            if all_complete:
                print(f"Abstractions logprobs already complete. Skipping... ({abs_lp_path})")
                return abs_lp_path
        # Strategy branch
        if logprobs_strategy == 'per_atom':
            # Build aggregators per atom to preserve output shape
            aggregators = {}
            for atom_data in abstractions_data:
                atom_text = normalize_spaces(atom_data['atom'])
                atom_abstractions = atom_data.get('abstractions', [])
                aggregators[atom_text] = {
                    "atom": atom_text,
                    "index": atom_data.get('index'),
                    "abstractions": [
                        {
                            "abstraction": normalize_spaces(a.get('abstraction')),
                            "index": j,
                            "logprobs": None
                        } for j, a in enumerate(atom_abstractions)
                    ]
                }
            # Build special user prompt
            logprob_ext_user_prompt = (
                "Original user prompt:\n"
                f"{user_prompt}\n\n"
                "The following statements were generated by you as part of the response to the prompt above during earlier steps. "
                "Each statement is a standalone factual claim; statements may not be consistent or related to each other. "
                "Read them as-is; do not add, change, or continue any text."
            )
            # Execute one request per atom (do not mix atoms)
            for atom_data in abstractions_data:
                atom_text = normalize_spaces(atom_data['atom'])
                atom_abstractions = atom_data.get('abstractions', [])
                if len(atom_abstractions) == 0:
                    continue
                # If we already have a complete entry for this atom, skip it
                try:
                    existing_map = {normalize_spaces(e.get('atom')): e for e in results if isinstance(e, dict)}
                    existing_entry = existing_map.get(atom_text)
                    if existing_entry:
                        abs_entries = existing_entry.get('abstractions')
                        if isinstance(abs_entries, list) and len(abs_entries) == len(atom_abstractions):
                            if all(isinstance(ae, dict) and isinstance(ae.get('logprobs'), list) for ae in abs_entries):
                                print(f"Atom '{atom_text}' logprobs already complete. Skipping...")
                                continue
                except Exception:
                    pass
                print(f"Extracting logprobs for atom '{atom_text}'...")
                entries = []
                for j, a in enumerate(atom_abstractions):
                    a_raw = normalize_spaces(a.get('abstraction'))
                    entries.append({
                        "abs_index": j,
                        "text_raw": a_raw,
                        "text_debr": remove_first_last_brackets(a_raw)
                    })
                assistant_content = "\n".join(e["text_debr"] for e in entries).strip()
                if len(assistant_content) == 0:
                    continue
                logprobs_params = dict(generation_params or {})
                logprobs_params["prompt_logprobs"] = True
                completion = get_model_completion(
                    client=client,
                    prompt=assistant_content,
                    model=model,
                    generation_params=logprobs_params,
                    extract_prompt_logprobs=True,
                    extract_prompt_logprobs_user_prompt=logprob_ext_user_prompt
                )
                prompt_logprobs = completion.get("prompt_logprobs")
                if prompt_logprobs is None:
                    print(f"prompt_logprobs missing in model response for atom '{atom_text}'. Skipping.")
                    continue
                grouped = token_logprobs_as_dicts(prompt_logprobs, prompt_text=assistant_content, sentence_terminators=['.'])
                grouped = [group for group in grouped if len(group) > 0]
                # Map grouped tokens to each statement by counting sentence groups
                g_idx = 0
                for stmt in entries:
                    seg_text = stmt["text_debr"]
                    num_groups = _count_sentence_groups(seg_text)
                    if num_groups == 0:
                        num_groups = 1
                    segs = grouped[g_idx:g_idx + num_groups]
                    g_idx += num_groups
                    flat_tokens = []
                    for sent in segs:
                        flat_tokens.extend(sent if isinstance(sent, list) else [])
                    abs_index = stmt.get("abs_index")
                    if abs_index is not None and abs_index < len(aggregators[atom_text]["abstractions"]):
                        aggregators[atom_text]["abstractions"][abs_index]["logprobs"] = flat_tokens
                # Persist this atom immediately (merge-or-replace behavior)
                agg = aggregators[atom_text]
                result_entry = {
                    "atom": agg["atom"],
                    "index": agg["index"],
                    "abstractions": agg["abstractions"]
                }
                existing_by_atom = {normalize_spaces(e.get('atom')): i for i, e in enumerate(results) if isinstance(e, dict)}
                if atom_text in existing_by_atom:
                    results[existing_by_atom[atom_text]] = result_entry
                else:
                    results.append(result_entry)
                with open(abs_lp_path, 'w') as f:
                    json.dump(results, f, indent=4)
            return abs_lp_path
        elif logprobs_strategy == 'all_statements':
            # Build aggregators per atom to preserve output shape
            aggregators = {}
            for atom_data in abstractions_data:
                atom_text = normalize_spaces(atom_data['atom'])
                atom_abstractions = atom_data.get('abstractions', [])
                aggregators[atom_text] = {
                    "atom": atom_text,
                    "index": atom_data.get('index'),
                    "abstractions": [
                        {
                            "abstraction": normalize_spaces(a.get('abstraction')),
                            "index": j,
                            "logprobs": None
                        } for j, a in enumerate(atom_abstractions)
                    ]
                }
            # Flatten into statements by atom-group (only abstractions; first abstraction equals the atom)
            grouped_by_atom = []
            for atom_data in abstractions_data:
                atom_text = normalize_spaces(atom_data['atom'])
                atom_abstractions = atom_data.get('abstractions', [])
                if len(atom_abstractions) == 0:
                    continue
                group_entries = []
                for j, a in enumerate(atom_abstractions):
                    a_raw = normalize_spaces(a.get('abstraction'))
                    group_entries.append({
                        "atom": atom_text,
                        "abs_index": j,
                        "text_raw": a_raw,
                        "text_debr": remove_first_last_brackets(a_raw)
                    })
                grouped_by_atom.append(group_entries)
            # Chunk batches by atom groups (never split an atom)
            batches = []
            if logprobs_chunking:
                if logprobs_chunk_size is None:
                    raise ValueError("logprobs_chunk_size must be provided when logprobs_chunking is True.")
                current = []
                current_len = 0
                for group in grouped_by_atom:
                    group_len = len(group)
                    if current_len > 0 and (current_len + group_len) > max(1, int(logprobs_chunk_size)):
                        batches.append(current)
                        current = []
                        current_len = 0
                    current.extend(group)
                    current_len += group_len
                if current:
                    batches.append(current)
            else:
                # Single batch with all
                flat = []
                for group in grouped_by_atom:
                    flat.extend(group)
                if flat:
                    batches = [flat]
            logprob_ext_user_prompt = (
                "Original user prompt:\n"
                f"{user_prompt}\n\n"
                "The following statements were generated by you in response to the prompt above during earlier steps. "
                "Each statement is a standalone factual claim; statements may not be mutually consistent or connected. "
                "Read them as-is; do not add, change, or continue any text."
            )
            # Execute batch requests and map back
            for batch_idx, batch in enumerate(batches):
                assistant_content = "\n".join(stmt["text_debr"] for stmt in batch)
                if len(assistant_content.strip()) == 0:
                    continue
                logprobs_params = dict(generation_params or {})
                logprobs_params["prompt_logprobs"] = True
                completion = get_model_completion(
                    client=client,
                    prompt=assistant_content,
                    model=model,
                    generation_params=logprobs_params,
                    extract_prompt_logprobs=True,
                    extract_prompt_logprobs_user_prompt=logprob_ext_user_prompt
                )
                prompt_logprobs = completion.get("prompt_logprobs")
                if prompt_logprobs is None:
                    print(f"prompt_logprobs missing in model response for batch {batch_idx}. Skipping batch.")
                    continue
                grouped = token_logprobs_as_dicts(prompt_logprobs, prompt_text=assistant_content, sentence_terminators=['.'])
                grouped = [group for group in grouped if len(group) > 0]
                # Map grouped tokens to each statement by counting sentence groups
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
                    atom_key = stmt["atom"]
                    abs_index = stmt.get("abs_index")
                    if abs_index is not None and abs_index < len(aggregators[atom_key]["abstractions"]):
                        aggregators[atom_key]["abstractions"][abs_index]["logprobs"] = flat_tokens
                # end mapping for batch
            # Prepare final results list (preserve existing result merge behavior)
            # Replace existing entries or append new ones
            existing_by_atom = {normalize_spaces(e.get('atom')): i for i, e in enumerate(results) if isinstance(e, dict)}
            for atom_key, agg in aggregators.items():
                result_entry = {
                    "atom": agg["atom"],
                    "index": agg["index"],
                    "abstractions": agg["abstractions"]
                }
                if atom_key in existing_by_atom:
                    results[existing_by_atom[atom_key]] = result_entry
                else:
                    results.append(result_entry)
            with open(abs_lp_path, 'w') as f:
                json.dump(results, f, indent=4)
            return abs_lp_path
        else:
            raise NotImplementedError(f"Unsupported logprobs strategy: {logprobs_strategy}")

    atoms_conf_file_path = input_files['atoms_conf_file_path']
    if not os.path.exists(atoms_conf_file_path):
        print(f"Atoms confidence file not found in {input_dir}. Skipping...")
        return
    atoms_confidence_data = load_json(atoms_conf_file_path)
    normalized_atoms_confidence_data = {}
    for k,v in atoms_confidence_data.items():
        normalized_atoms_confidence_data[normalize_spaces(k)] = v
    atoms_confidence_data = normalized_atoms_confidence_data
    
    atoms_confidence_config_str = "atoms_confidence_config=" + os.path.basename(os.path.dirname(atoms_conf_file_path))
    output_dir = os.path.join(input_dir, "abstractions_confidence", atoms_confidence_config_str, 'abstractions_'+naming_config_to_str(naming_config))
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, "abstractions_confidence.json")
    if os.path.exists(output_file_path):
        abstractions_conf_data = load_json(output_file_path)
    else:
        abstractions_conf_data = []

    system_prompt_path = f"{prompts_dir}/confidence/sys_prompts/{sys_prompt}.txt"
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read()

    # Build a lookup for already processed atoms to resume safely
    processed_atoms = {a['atom'] for a in abstractions_conf_data} if isinstance(abstractions_conf_data, list) else set()
    results = abstractions_conf_data if isinstance(abstractions_conf_data, list) else []
    call_costs = []

    for atom_data in abstractions_data:
        atom = normalize_spaces(atom_data['atom'])
        atom_abstractions = atom_data['abstractions']

        # find the atom in abstractions_conf_data
        atom_conf_data = next((a for a in abstractions_conf_data if a['atom'] == atom), None)
        if atom_conf_data and all([a.get('confidence') is not None for a in atom_conf_data.get('abstractions')]):
            # print(f"Atom '{atom}' already processed. Skipping...")
            continue

        if len(atom_abstractions) == 0:
            print(f"Warning: Atom '{atom}' has no abstractions. Skipping...")
            continue
        abs_texts = [normalize_spaces(a['abstraction']) for a in atom_abstractions]

        # Verbalized confidence for abstractions
        atom_confidence = atoms_confidence_data[atom]['confidence']
        if confidence_score == 'verbalized':
            numbered_abs = ""
            for i, abs_text in enumerate(abs_texts):
                numbered_abs += f"{i+1}. {remove_first_last_brackets(abs_text)}{f' Confidence: {atom_confidence}' if i == 0 else ''}\n"
            try:
                response = get_model_completion(
                    client=client,
                    system_prompt=system_prompt,
                    prompt=numbered_abs,
                    model=model,
                    generation_params=generation_params,
                    reasoning_effort=reasoning_effort,
                )
                if reasoning_effort is not None:
                    call_costs.append(extract_call_cost(response))
                response_text = get_response_text(response, model)
            except Exception as e:
                print(f"Error getting abstraction confidence for atom '{atom}': {e}")
                continue
            try:
                abs_conf_map = parse_confidence(response_text, abs_texts)
                if atom_confidence != abs_conf_map[atom]['confidence']:
                    print(f"Warning: Atom confidence for atom '{atom}' ({atom_confidence}) != abstractions confidence ({abs_conf_map[atom]['confidence']}). Keeping the atom confidence in the outer dict and updating only the abstractions confidence.")
            except Exception as e:
                print(f"Error parsing abstractions confidence for atom '{atom}': {e}")
                abs_conf_map = {a: None for a in abs_texts}
            atom_entry = {
                'atom': atom,
                'index': atom_data['index'],
                'confidence': atom_confidence,
                'abstractions': [
                    {
                        'abstraction': a_text,
                        'index': j,
                        'confidence': abs_conf_map[a_text]['confidence'] if abs_conf_map[a_text] else None,
                        'reasoning': abs_conf_map[a_text]['reasoning'] if abs_conf_map[a_text] else None
                    }
                    for j, a_text in enumerate(abs_texts)
                ]
            }
            if len(results) > atom_data['index']:
                results[atom_data['index']] = atom_entry
            else:
                results.append(atom_entry)
            with open(output_file_path, 'w') as f:
                json.dump(results, f, indent=4)

        else:
            raise NotImplementedError(f"Unsupported confidence_score for abstractions: {confidence_score}")

    if reasoning_effort is not None and call_costs:
        stage_cost = {
            'stage': 'abstractions_confidence',
            'model': model,
            'reasoning_effort': reasoning_effort,
            **aggregate_costs(call_costs),
        }
        with open(os.path.join(output_dir, 'cost.json'), 'w') as f:
            json.dump(stage_cost, f, indent=4)

    return output_file_path


