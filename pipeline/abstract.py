import os
from generation_utils import *
from utils.text_utils import parse_atoms, verify_sentence_format_for_abstraction, load_json, normalize_spaces
import json
import re

def parse_abstractions(atom, abstraction_response_text):
    """
    Parse a numbered abstraction sequence where each item has:
      n. [ENTITY] [FACT].
         Reasoning: ...
    And ends with:
      K. STOP
         Reasoning: ...
    Returns a list of dicts: {'abstraction', 'index', 'reasoning'}, excluding STOP.
    """
    abstractions_list = []
    pattern = r"\d+\.\s+(.*?)\n\s*Reasoning:\s*(.*?)(?=\n\s*\d+\.|\Z)"
    matches = re.findall(pattern, abstraction_response_text, re.DOTALL)
    if not isinstance(matches, list):
        matches = []

    for idx, (candidate, reasoning) in enumerate(matches):
        candidate = candidate.strip()
        if candidate.upper().startswith('STOP'):
            break
        # verify bracketed sentence format; attempt light fixes if needed
        try:
            candidate_verified = verify_sentence_format_for_abstraction(candidate)
        except Exception:
            candidate_no_asterisk = candidate.replace('*', '')
            try:
                candidate_verified = verify_sentence_format_for_abstraction(candidate_no_asterisk)
            except Exception:
                print(f"Invalid abstraction format: {candidate}. Skipping...")
                continue
        abstractions_list.append({
            'abstraction': candidate_verified,
            'index': idx,
            'reasoning': (reasoning or '').strip()
        })

    # Ensure the first item is the original atom (as required by the prompt)
    if abstractions_list:
        if normalize_spaces(abstractions_list[0]['abstraction']) != normalize_spaces(atom):
            print(f"First abstraction does not match the atom. Adding atom as first item.\nfirst: {abstractions_list[0]['abstraction']} != atom: {atom}")
            try:
                atom_verified = verify_sentence_format_for_abstraction(atom)
            except Exception:
                atom_verified = atom.replace('*', '')
                atom_verified = verify_sentence_format_for_abstraction(atom_verified)
            # Use the first reasoning block for the inserted atom to mirror the with-confidence parser behavior
            first_reasoning = (matches[0][1] if matches else '') or ''
            abstractions_list.insert(0, {
                'abstraction': atom_verified,
                'index': 0,
                'reasoning': first_reasoning.strip()
            })
            # Reindex subsequent items
            for i in range(1, len(abstractions_list)):
                abstractions_list[i]['index'] = i

    return abstractions_list

def _process_atom_abstractions(atom, client, model, abstract_system_prompt, entity_abstractions, generation_params, reasoning_effort=None, call_costs=None):
    try:
        atom = verify_sentence_format_for_abstraction(atom)
    except Exception as e:
        print(f"Error verifying atom format for {atom}. Skipping...")
        return None
    if any(d.get('atom') == atom for d in entity_abstractions):
        # print(f"Abstraction for {atom} already exists. Skipping...")
        abstractions_for_atom = []
        for d in entity_abstractions:
            if d.get('atom') == atom:
                abstractions_for_atom = d.get('abstractions', [])
        if len(abstractions_for_atom) == 0:
            print(f"Warning: Atom '{atom}' has no abstractions. Returning None...")
        return None

    print(f"Abstracting atom: {atom}")
    response_dict = get_model_completion(
        client=client,
        system_prompt=abstract_system_prompt,
        prompt=atom,
        model=model,
        generation_params=generation_params,
        reasoning_effort=reasoning_effort,
    )
    if call_costs is not None:
        call_costs.append(extract_call_cost(response_dict))
    response_text = get_response_text(response_dict, model)

    try:
        abstractions_list = parse_abstractions(atom, normalize_spaces(response_text))
    except Exception as e:
        print(f"Error parsing abstractions for {atom}. Saving empty list.")
        abstractions_list = []

    return {'abstractions': abstractions_list}

def run_abstract(model, sys_prompt, client, atomize_output_dir, prompts_dir, generation_params, reasoning_effort=None):
    sys_prompt_path = os.path.join(prompts_dir, "abstraction", f"{sys_prompt}.txt")
    with open(sys_prompt_path, 'r') as f:
        abstract_system_prompt = f.read()

    naming_config = {
        'sys_prompt': sys_prompt
    }
    if reasoning_effort is not None:
        naming_config['reasoning_effort'] = reasoning_effort

    output_dir = os.path.join(atomize_output_dir, "abstract", naming_config_to_str(naming_config))
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_path = os.path.join(output_dir, "abstractions.json")

    entity_atoms_file_path = os.path.join(atomize_output_dir, "atoms.json")
    if not os.path.exists(entity_atoms_file_path):
        print(f"Atoms file not found in {atomize_output_dir}. Skipping...")
        return

    entity_atoms = load_json(entity_atoms_file_path)
    
    if not entity_atoms:
        print(f"No atoms found in {atomize_output_dir}. Skipping...")
        return
    else:
        print(f"Found {len(entity_atoms)} atoms in {atomize_output_dir}.")

    if os.path.exists(output_file_path):
        try:
            entity_abstractions = load_json(output_file_path)
        except Exception as e:
            print(f"Error loading abstractions from {output_file_path}. Creating empty list.")
            entity_abstractions = []
    else:
        entity_abstractions = []

    call_costs = []
    for i, atom in enumerate(entity_atoms):
        result = _process_atom_abstractions(atom, client, model, abstract_system_prompt, entity_abstractions, generation_params, reasoning_effort=reasoning_effort, call_costs=call_costs)
        if result:
            if len(result['abstractions']) == 0:
                print(f"Warning: Atom '{atom}' has no abstractions. Trying again...")
                result = _process_atom_abstractions(atom, client, model, abstract_system_prompt, entity_abstractions, generation_params, reasoning_effort=reasoning_effort, call_costs=call_costs)
            entity_abstractions.append({'atom': atom, 'index': i, **result})
            with open(output_file_path, 'w') as f:
                json.dump(entity_abstractions, f, indent=4)

    if reasoning_effort is not None and call_costs:
        stage_cost = {
            'stage': 'abstract',
            'model': model,
            'reasoning_effort': reasoning_effort,
            **aggregate_costs(call_costs),
        }
        with open(os.path.join(output_dir, 'cost.json'), 'w') as f:
            json.dump(stage_cost, f, indent=4)

    return output_file_path
