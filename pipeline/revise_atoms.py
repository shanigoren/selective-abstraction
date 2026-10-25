import os
import json
from generation_utils import get_model_completion, naming_config_to_str, get_response_text
from utils.text_utils import load_json, verify_sentence_format_for_abstraction


def run_revise_atoms(
    model,
    client,
    atomize_output_dir,
    prompts_dir,
    generation_params,
    confidence_file_path,
    sys_prompt="atom_wise_self_revision",
):
    """
    Revise each atom by prompting the LLM to remove or generalize uncertain
    information, using the atom's confidence score as guidance.

    Reads `atoms.json` from `atomize_output_dir` and confidence scores from
    `confidence_file_path`, calls the LLM once per atom, and writes revised
    atoms to:
        {atomize_output_dir}/revise_atoms/{naming_config}/atoms.json

    If a revised atom fails the [ENTITY] [FACT]. bracket format check, the
    original atom is kept with a warning.

    Returns:
        Path to the output directory containing the revised atoms.json.
    """
    system_prompt_path = os.path.join(prompts_dir, "self_revision", f"{sys_prompt}.txt")
    if not os.path.exists(system_prompt_path):
        raise FileNotFoundError(
            f"Atom-wise self-revision prompt not found at {system_prompt_path}"
        )

    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()

    naming_config = {"sys_prompt": sys_prompt}
    output_dir = os.path.join(
        atomize_output_dir, "revise_atoms", naming_config_to_str(naming_config)
    )
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, "atoms.json")
    if os.path.exists(output_file_path):
        print(
            f"Revised atoms already exist at {output_file_path}. "
            f"Skipping revise_atoms step."
        )
        return output_dir

    atoms_file_path = os.path.join(atomize_output_dir, "atoms.json")
    if not os.path.exists(atoms_file_path):
        raise FileNotFoundError(f"Atoms file not found at {atoms_file_path}")

    atoms = load_json(atoms_file_path)
    if not isinstance(atoms, list):
        raise ValueError(
            f"Invalid atoms.json structure (expected list) at {atoms_file_path}"
        )

    confidence_dict = load_json(confidence_file_path)
    if not isinstance(confidence_dict, dict):
        raise ValueError(
            f"Invalid confidence file structure (expected dict) at {confidence_file_path}"
        )

    revised_atoms = []
    for i, atom in enumerate(atoms):
        confidence = confidence_dict.get(atom)
        if confidence is None:
            print(f"Warning: no confidence score found for atom {i}: {atom}. Using 50.")
            confidence = 50

        prompt = f"Atom: {atom}\nConfidence: {confidence}"
        response = get_model_completion(
            client=client,
            system_prompt=system_prompt,
            prompt=prompt,
            model=model,
            generation_params=generation_params,
        )
        revised = get_response_text(response, model).strip()
        
        try:
            verify_sentence_format_for_abstraction(revised)
        except ValueError:
            print(
                f"Warning: revised atom {i} failed format check.\n"
                f"  Original: {atom}\n"
                f"  Revised:  {revised}"
            )
            continue

        revised_atoms.append(revised)

    with open(output_file_path, "w") as f:
        json.dump(revised_atoms, f, indent=4)

    print(f"Revised atoms saved to {output_file_path}.")
    return output_dir
