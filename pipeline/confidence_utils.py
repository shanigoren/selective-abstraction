import os
import re


def _count_sentence_groups(text: str) -> int:
    if not isinstance(text, str) or text.strip() == "":
        return 0
    parts = re.split(r"[.\n]", text)
    return len([p for p in parts if p.strip() != ""])


def _find_upwards_file(start_dir: str, filename: str, max_levels: int = 6) -> str:
    """
    Search upwards from start_dir for a directory containing filename.
    Returns the directory path if found, else None.
    """
    current = start_dir
    for _ in range(max_levels):
        candidate = os.path.join(current, filename)
        if os.path.exists(candidate):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def match_atom_with_brackets(atom, entity_atoms):
    # compare the atom with the entity_atoms, ignoring any non alphanumeric characters
    return next((a for a in entity_atoms if re.sub(r'[^a-zA-Z0-9]', '', a) == re.sub(r'[^a-zA-Z0-9]', '', atom)), None)


def parse_confidence(confidence_response, entity_atoms):
    confidence_dict = {}
    pattern_with_reasoning = r"\d+\.\s+(.*?)\s+Reasoning:\s*(.*?)\s+Confidence:\s*(\d+)(?=\s+\d+\.|\s*$)"
    matches = re.findall(pattern_with_reasoning, confidence_response, re.DOTALL)
    if len(matches) < len(entity_atoms):
        print(f"Warning: Number of matches and entity atoms do not match. matches: {len(matches)} < entity_atoms: {len(entity_atoms)}. Attempting to parse without reasoning...")
        pattern_without_reasoning = r"\d+\.\s+(.*?)(?:\s+Reasoning:\s*.*?)?\s+Confidence:\s*(\d+)(?=\s+\d+\.|\s*$)"
        matches = re.findall(pattern_without_reasoning, confidence_response, re.DOTALL)
    if len(matches) < len(entity_atoms):
        print(f"Warning: Number of matches and entity atoms do not match. matches: {len(matches)} < entity_atoms: {len(entity_atoms)}. Attempting to clean text and parse again...")
        cleaned = re.sub(r"(?s)^(.*Confidence:\s*\d+).*", r"\1", confidence_response)
        matches = re.findall(pattern_without_reasoning, cleaned, re.DOTALL)
    assert len(matches) == len(entity_atoms), f"Number of matches and entity atoms do not match. matches: {len(matches)}, entity_atoms: {len(entity_atoms)}"
    for match, entity_atom in zip(matches, entity_atoms):
        if len(match) == 3:
            atom, reasoning, score = match
        else:
            atom, score = match
            reasoning = None
        confidence_dict[entity_atom] = {
            'confidence': int(score),
            'reasoning': reasoning.strip() if reasoning else None
        }
    return confidence_dict


