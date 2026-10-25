"""Deterministic confidence-threshold selection used by the paper pipeline."""

import json
import os


def select_statements(confidence_data, theta, mode="sa"):
    """Select statements in atom order with exact-string deduplication.

    ``sa`` selects the first (most specific) abstraction whose confidence is at
    least ``theta``. ``redaction`` keeps the original atom when its atom-level
    confidence is at least ``theta``.
    """
    if mode not in {"sa", "redaction"}:
        raise ValueError(f"Unsupported selection mode: {mode}")
    if theta < 0 or theta > 100:
        raise ValueError("theta must be between 0 and 100")

    selected = []
    seen = set()
    for atom_item in confidence_data:
        statement = None
        if mode == "sa":
            abstractions = sorted(
                atom_item.get("abstractions", []),
                key=lambda item: item["index"],
            )
            for abstraction in abstractions:
                confidence = (
                    atom_item.get("confidence")
                    if abstraction.get("index") == 0
                    else abstraction.get("confidence")
                )
                if confidence is not None and confidence >= theta:
                    statement = abstraction["abstraction"]
                    break
        else:
            confidence = atom_item.get("confidence")
            if confidence is not None and confidence >= theta:
                statement = atom_item.get("atom")

        if statement is not None and statement not in seen:
            selected.append(statement)
            seen.add(statement)

    return selected


def write_selected_statements(confidence_path, output_dir, theta, mode="sa"):
    with open(confidence_path, "r", encoding="utf-8") as handle:
        confidence_data = json.load(handle)
    statements = select_statements(confidence_data, theta=theta, mode=mode)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "selected_statements.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(statements, handle, ensure_ascii=False, indent=2)
    return statements, output_path
