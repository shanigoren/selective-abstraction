import os
import json
from utils.text_utils import load_json


def process_atoms_without_abstraction(atoms_confidence_path: str) -> str:
    """Represent each scored atom as its sole confidence-bearing abstraction."""
    if os.path.isdir(atoms_confidence_path):
        atoms_confidence_path = os.path.join(
            atoms_confidence_path, "atoms_confidence.json"
        )
    if not os.path.exists(atoms_confidence_path):
        raise FileNotFoundError(
            f"Atoms confidence file not found: {atoms_confidence_path}"
        )

    atoms_confidence = load_json(atoms_confidence_path)
    output_path = os.path.join(
        os.path.dirname(atoms_confidence_path),
        "atoms_confidence_processed.json",
    )
    if os.path.exists(output_path):
        return output_path

    abstractions = []
    for index, (atom, confidence) in enumerate(atoms_confidence.items()):
        abstractions.append(
            {
                "atom": atom,
                "index": index,
                "confidence": confidence,
                "abstractions": [
                    {
                        "abstraction": atom,
                        "index": 0,
                        "confidence": confidence,
                    }
                ],
            }
        )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(abstractions, handle, ensure_ascii=False, indent=2)
    return output_path


def build_trivial_abstractions_from_atoms(atomize_output_dir: str) -> str:
    """
    Build a trivial abstractions file from atoms produced by `run_atomize`.

    Each atom becomes its own single abstraction entry:
      {
        "atom": <atom>,
        "index": <i>,
        "abstractions": [
          {
            "abstraction": <atom>,
            "index": 0
          }
        ]
      }

    Returns:
        Path to the written abstractions.json file.
    """
    atoms_file_path = os.path.join(atomize_output_dir, "atoms.json")
    if not os.path.exists(atoms_file_path):
        raise FileNotFoundError(
            f"Atoms file not found in atomize output dir: {atoms_file_path}"
        )

    atoms = load_json(atoms_file_path)
    if not isinstance(atoms, list):
        raise ValueError(
            f"Invalid atoms.json structure (expected list) at {atoms_file_path}"
        )

    output_dir = os.path.join(atomize_output_dir, "abstract_trivial")
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, "abstractions.json")
    if os.path.exists(output_file_path):
        print(
            f"Trivial abstractions file already exists at {output_file_path}. "
            f"Reusing existing file."
        )
        return output_file_path

    abstractions_data = []
    for idx, atom in enumerate(atoms):
        abstractions_data.append(
            {
                "atom": atom,
                "index": idx,
                "abstractions": [
                    {
                        "abstraction": atom,
                        "index": 0,
                    }
                ],
            }
        )

    with open(output_file_path, "w") as f:
        json.dump(abstractions_data, f, indent=4)

    print(f"Trivial abstractions saved to {output_file_path}.")
    return output_file_path


