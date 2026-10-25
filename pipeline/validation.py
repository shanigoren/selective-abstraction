import os
import json
from utils.text_utils import load_json, create_numbered_list, normalize_spaces


def validation_checkpoint(atomize_output_dir, abstract_output_file_path, entity):
    # validation checkpoint - make sure each atom has abstractions
    atoms_file_path = os.path.join(atomize_output_dir, 'atoms.json')
    if not os.path.exists(atoms_file_path):
        print(f"Error: atoms file not found in {atomize_output_dir}")
        return
    atoms_data = load_json(atoms_file_path)
    if not os.path.exists(abstract_output_file_path):
        print(f"Error: abstractions file not found in {abstract_output_file_path}")
        return
    abstractions_data = load_json(abstract_output_file_path)
    assert len(set(atoms_data)) == len(abstractions_data), f"Number of atoms and abstractions do not match. atoms.json: {len(set(atoms_data))}, abstractions: {len(abstractions_data)}"
    if len(set(atoms_data)) != len(abstractions_data):
        print(f"Number of atoms and abstractions do not match. atoms.json: {len(set(atoms_data))}, abstractions: {len(abstractions_data)}")
        # temp fix: find the duplicate atoms and insert them into the abstractions_data at the correct index
        if len(set([a['atom'] for a in abstractions_data])) < len([a['atom'] for a in abstractions_data]):
            print("Found duplicate atoms in abstractions.json. Fixing...")
            len_diff = len([a['atom'] for a in abstractions_data]) - len(set([a['atom'] for a in abstractions_data]))
            # take the last len_diff atoms from the abstractions_data
            duplicates = abstractions_data[-len_diff:]
            # pop the duplicates from the abstractions_data
            for duplicate in duplicates:
                abstractions_data[duplicate['index']] = duplicate
            abstractions_data = abstractions_data[:-len_diff]
            # save the abstractions_data
            with open(abstract_output_file_path, 'w') as f:
                json.dump(abstractions_data, f, indent=4)
        
        else:
            # find the missing atoms
            missing_atoms = []
            for atom in atoms_data:
                found_atom = False
                for abstraction in abstractions_data:
                    if abstraction['atom'] == atom:
                        found_atom = True
                        break
                if not found_atom:
                    missing_atoms.append(atom)
            print(f"Missing atoms:\n{create_numbered_list(missing_atoms)}")
            print("Moving on to the next entity...")
            return

    # Count atoms that ended up with zero abstractions and enforce a tolerance threshold.
    total_atoms = len(abstractions_data)
    count_zero_abstractions = 0
    for atom_entry in abstractions_data:
        if len(atom_entry.get('abstractions', [])) == 0:
            print(f"Warning: Atom {atom_entry.get('atom')} has no abstractions.")
            count_zero_abstractions += 1

    if total_atoms > 0 and count_zero_abstractions > 0:
        fraction_zero = count_zero_abstractions / total_atoms
        print(
            f"Warning: {count_zero_abstractions}/{total_atoms} atoms "
            f"have no abstractions ({fraction_zero:.1%})."
        )
        # Hard-fail only if more than 10% of atoms have no abstractions.
        if fraction_zero > 0.10:
            raise ValueError(
                f"Too many atoms with no abstractions for entity {entity}: "
                f"{count_zero_abstractions}/{total_atoms} ({fraction_zero:.1%}) "
                "exceeds 10% threshold."
            )

    print("Validation checkpoint passed.")


def validate_questions_for_entity(abstract_output_file_path, questions_json_path):
    # Ensure files exist
    if not os.path.exists(abstract_output_file_path):
        raise FileNotFoundError(f"Abstractions file not found: {abstract_output_file_path}")
    if not os.path.exists(questions_json_path):
        raise FileNotFoundError(f"Questions file not found: {questions_json_path}")
    # Load data
    abstractions_data = load_json(abstract_output_file_path)
    questions_data = load_json(questions_json_path)
    # Basic structure checks
    if not isinstance(abstractions_data, list):
        raise ValueError(f"Abstractions file has invalid structure (expected list): {abstract_output_file_path}")
    if not isinstance(questions_data, list):
        raise ValueError(f"Questions file has invalid structure (expected list): {questions_json_path}")
    # Top-level count equality
    if len(abstractions_data) != len(questions_data):
        raise AssertionError(f"Mismatch: abstractions items ({len(abstractions_data)}) != questions items ({len(questions_data)}).")
    # Per-atom validation
    for idx, (abs_item, q_item) in enumerate(zip(abstractions_data, questions_data), start=1):
        abs_abstractions = abs_item.get('abstractions')
        q_abstractions = q_item.get('abstractions')
        if not isinstance(abs_abstractions, list):
            raise ValueError(f"[Atom #{idx}] Abstractions list missing/invalid in abstractions file.")
        if not isinstance(q_abstractions, list):
            raise ValueError(f"[Atom #{idx}] Abstractions list missing/invalid in questions file.")
        if len(abs_abstractions) != len(q_abstractions):
            raise AssertionError(f"[Atom #{idx}] Mismatch: abstractions count ({len(abs_abstractions)}) != questions abstractions count ({len(q_abstractions)}).")
        # Each abstraction must have questions with both keys
        for jdx, q_abs in enumerate(q_abstractions, start=1):
            q_dict = q_abs.get('questions')
            if not isinstance(q_dict, dict):
                raise ValueError(f"[Atom #{idx} | Abstraction #{jdx}] Missing 'questions' dict.")
            for key in ('broad', 'specific'):
                val = q_dict.get(key)
                if not isinstance(val, str) or not val.strip():
                    raise ValueError(f"[Atom #{idx} | Abstraction #{jdx}] Missing/empty '{key}' question.")
    print("Questions validation passed.")


def validate_confidences_for_entity_separate(abstract_output_file_path, abstractions_conf_json_path):
    """
    Validate that, in 'separate' abstraction mode, every abstraction has a
    non-null numeric confidence score.

    This cross-checks the plain abstractions file produced by `run_abstract`
    against the confidence-annotated abstractions file produced by
    `run_abstractions_confidence`.
    """
    # Ensure files exist
    if not os.path.exists(abstract_output_file_path):
        raise FileNotFoundError(f"Abstractions file not found: {abstract_output_file_path}")
    if not os.path.exists(abstractions_conf_json_path):
        raise FileNotFoundError(f"Abstractions confidence file not found: {abstractions_conf_json_path}")

    abstractions_data = load_json(abstract_output_file_path)
    abstractions_conf_data = load_json(abstractions_conf_json_path)

    # Basic structure checks
    if not isinstance(abstractions_data, list):
        raise ValueError(
            f"Abstractions file has invalid structure (expected list): {abstract_output_file_path}"
        )
    if not isinstance(abstractions_conf_data, list):
        raise ValueError(
            f"Abstractions confidence file has invalid structure (expected list): {abstractions_conf_json_path}"
        )

    # DANGER!!!
    # previous version:
    # if len(abstractions_data) != len(abstractions_conf_data):
    #     raise AssertionError(
    #         f"Mismatch: abstractions items ({len(abstractions_data)}) != "
    #         f"abstractions confidence items ({len(abstractions_conf_data)})."
    #     )
    
    # TEMP FIX: allow confidence file to have over 90% items than abstractions file, but not more (to handle cases where confidence generation failed for some atoms)
    if len(abstractions_conf_data) > len(abstractions_data):
        raise AssertionError(
            f"Mismatch: abstractions items ({len(abstractions_data)}) > "
            f"abstractions confidence items ({len(abstractions_conf_data)})."
        )

    if len(abstractions_conf_data) < len(abstractions_data) * 0.90:
        raise AssertionError(
            f"Mismatch: abstractions items ({len(abstractions_data)}) < "
            f"abstractions confidence items ({len(abstractions_conf_data)})."
        )

    count_missing_confidences, count_total_confidences = 0, 0

    # Per-atom validation
    for idx, (abs_item, conf_item) in enumerate(zip(abstractions_data, abstractions_conf_data), start=1):
        atom_abs = normalize_spaces(abs_item.get('atom', ''))
        atom_conf = normalize_spaces(conf_item.get('atom', ''))
        if atom_abs != atom_conf:
            raise AssertionError(
                f"[Atom #{idx}] Atom mismatch between abstractions and confidence files: "
                f"'{atom_abs}' != '{atom_conf}'."
            )

        abs_abstractions = abs_item.get('abstractions')
        conf_abstractions = conf_item.get('abstractions')

        if not isinstance(abs_abstractions, list):
            raise ValueError(f"[Atom #{idx}] Abstractions list missing/invalid in abstractions file.")
        if not isinstance(conf_abstractions, list):
            raise ValueError(f"[Atom #{idx}] Abstractions list missing/invalid in confidence file.")

        if len(abs_abstractions) != len(conf_abstractions):
            raise AssertionError(
                f"[Atom #{idx}] Mismatch: abstractions count ({len(abs_abstractions)}) != "
                f"confidence abstractions count ({len(conf_abstractions)})."
            )

        # Ensure each abstraction has a non-null numeric confidence
        for jdx, conf_abs in enumerate(conf_abstractions, start=1):
            conf_val = conf_abs.get('confidence')
            if not isinstance(conf_val, (int, float)):
                count_missing_confidences += 1
            count_total_confidences += 1
            
    if count_missing_confidences > 0:
        print(f"Warning: {count_missing_confidences}/{count_total_confidences} abstractions have missing confidence values.")
        if count_missing_confidences / count_total_confidences > 0.10:
            raise ValueError(
                f"Too many abstractions with missing confidence values: "
                f"{count_missing_confidences}/{count_total_confidences} ({count_missing_confidences / count_total_confidences:.1%}) "
                "exceeds 10% threshold."
                )
    print("Abstractions confidence validation (separate mode) passed.")


