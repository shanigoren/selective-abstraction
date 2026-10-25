import os
import json
from generation_utils import *
from utils.text_utils import *


def _chunk_response_by_newlines(text, chunk_size):
    """
    Split the response text into chunks using newline boundaries only.
    Each non-empty line is treated as an atomic unit that is never split.
    Consecutive lines are packed into chunks up to approximately `chunk_size`
    characters. If a single line exceeds `chunk_size`, it forms its own chunk.
    """
    if text is None:
        return []

    lines = text.splitlines()
    chunks = []
    current_lines = []
    current_len = 0

    for raw_line in lines:
        line = raw_line.strip()
        # Skip completely empty lines
        if not line:
            continue

        line_len = len(line)

        # If adding this line would exceed chunk_size and we already have
        # accumulated lines, flush the current chunk first.
        if current_lines and (current_len + 1 + line_len) > max(1, int(chunk_size)):
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_len = 0

        # Add line to current chunk (even if it is longer than chunk_size);
        # this guarantees we never split a line/sentence across chunks.
        current_lines.append(line)
        if current_len == 0:
            current_len = line_len
        else:
            # +1 accounts for the newline that will be inserted when joining
            current_len += 1 + line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    # Fallback: if everything was empty/whitespace, return empty list
    return chunks


def _deduplicate_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def run_atomize(client, model, sys_prompt, input_dir, prompts_dir, generation_params, chunking=False, chunk_size=None, reasoning_effort=None):
    sys_prompt_path = f"{prompts_dir}/atomization/{sys_prompt}.txt"
    with open(sys_prompt_path, 'r') as f:
        atomize_system_prompt = f.read()

    naming_config = {
        'sys_prompt': sys_prompt
    }
    if chunking:
        naming_config['chunking'] = True
        naming_config['chunk_size'] = int(chunk_size)
    if reasoning_effort is not None:
        naming_config['reasoning_effort'] = reasoning_effort

    output_dir = os.path.join(input_dir, "atomize", naming_config_to_str(naming_config))
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_path = os.path.join(output_dir, "atoms.json")

    if os.path.exists(output_file_path):
        print(f"Atomized response already exists. Skipping...")
        return output_dir

    generation_response_file_path = os.path.join(input_dir, "response.json")
    if not os.path.exists(generation_response_file_path):
        print(f"Response file not found. Skipping.")
        return

    entity_response_dict = load_json(generation_response_file_path)
    response_text = get_response_text(entity_response_dict, model)
    entity_response = normalize_spaces(response_text)
    print(f"Atomizing response using model {model}, config: {naming_config}")

    call_costs = []

    # Non-chunked path preserves existing behaviour (including special handling
    # for specific system prompts).
    if not chunking:
        if 'w_context' in sys_prompt:
            prompt = f"Prompt: {entity_response_dict['config']['generate_prompt']}\nResponse: {entity_response}"
        else:
            prompt = entity_response

        atomize_response_dict = get_model_completion(
            client=client,
            system_prompt=atomize_system_prompt,
            prompt=prompt,
            model=model,
            generation_params=generation_params,
            reasoning_effort=reasoning_effort,
        )
        if reasoning_effort is not None:
            call_costs.append(extract_call_cost(atomize_response_dict))

        response_text = get_response_text(atomize_response_dict, model)
        entity_atoms = parse_atoms(normalize_spaces(response_text))
    else:
        # Chunked path: split the response into newline-based chunks and
        # atomize each chunk separately.
        chunks = _chunk_response_by_newlines(entity_response, chunk_size)
        if not chunks:
            print("No non-empty chunks found in response; writing empty atoms list.")
            entity_atoms = []
        else:
            all_atoms = []
            base_prompt = entity_response_dict['config']['generate_prompt']
            for idx, chunk_text in enumerate(chunks):
                # Optionally incorporate the original generation prompt if available.
                if 'w_context' in sys_prompt:
                    chunk_prompt = f"Prompt: {base_prompt}\nResponse: {chunk_text}"
                else:
                    chunk_prompt = chunk_text

                print(f"Atomizing chunk {idx + 1}/{len(chunks)} (length={len(chunk_text)} chars)")
                try:
                    atomize_response_dict = get_model_completion(
                        client=client,
                        system_prompt=atomize_system_prompt,
                        prompt=chunk_prompt,
                        model=model,
                        generation_params=generation_params,
                        reasoning_effort=reasoning_effort,
                    )
                    if reasoning_effort is not None:
                        call_costs.append(extract_call_cost(atomize_response_dict))
                except Exception as e:
                    print(f"Error atomizing chunk {idx + 1}/{len(chunks)}: {e}")
                    continue

                response_text = get_response_text(atomize_response_dict, model)
                chunk_atoms = parse_atoms(normalize_spaces(response_text))
                all_atoms.extend(chunk_atoms)

            entity_atoms = all_atoms

    # de-duplicate while preserving order
    entity_atoms = _deduplicate_preserve_order(entity_atoms)

    # validate the atoms are in the correct format
    # TODO: might change if we change atomization!
    # count the number of atoms that are not in the correct format
    count_bad_format = 0
    valid_atoms = []
    for atom in entity_atoms:
        try:
            valid_atom = verify_sentence_format_for_abstraction(atom)
            valid_atoms.append(valid_atom)
        except Exception:
            print(f"Invalid atom format: {atom}. Skipping...")
            count_bad_format += 1
    if count_bad_format > 0:
        print(f"Number of atoms that are not in the correct format: {count_bad_format}/{len(entity_atoms)}")

    if count_bad_format / len(entity_atoms) > 0.2:
        print(f"Too many atoms that are not in the correct format. Skipping...")
        return None

    # keep only the atoms that are in the correct format
    entity_atoms = valid_atoms

    with open(output_file_path, 'w') as f:
        json.dump(entity_atoms, f, indent=4)

    if reasoning_effort is not None and call_costs:
        stage_cost = {
            'stage': 'atomize',
            'model': model,
            'reasoning_effort': reasoning_effort,
            **aggregate_costs(call_costs),
        }
        with open(os.path.join(output_dir, 'cost.json'), 'w') as f:
            json.dump(stage_cost, f, indent=4)

    print(f"Atoms for {input_dir} saved to {output_file_path}.")
    return output_dir

