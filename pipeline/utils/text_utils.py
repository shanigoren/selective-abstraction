import re
import json


def normalize_spaces(s, remove_newlines=False):
    # Replace the known problematic unicode spaces
    s = s.replace("\u202f", " ")
    s = s.replace("\u00A0", " ")
    s = s.replace("\u2009", " ")
    # Collapse runs of spaces
    if remove_newlines:
        return " ".join(s.split())
    s = re.sub(r"[ \t]+", " ", s)
    return s


def remove_thinking_from_text(text):
    """Remove everything inside thinking tags from the text."""
    if '<think>' in text and '</think>' in text:
        text = text.split('</think>')[-1].strip()
    return text


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def create_numbered_list(sentences, remove_brackets=True):
    """Create a numbered list from the sentences."""
    numbered_list = ""
    for i, sentence in enumerate(sentences):
        if remove_brackets:
            sentence = remove_first_last_brackets(sentence)
        numbered_list += f"{i+1}. {sentence}\n"
    return numbered_list


def parse_atoms(atoms_raw):
    """Parse the atoms from the raw response."""
    atoms_raw = remove_thinking_from_text(atoms_raw)
    atoms = atoms_raw.split("\n")
    atoms = ['. '.join(atom.strip().split(". ")[1:]) for atom in atoms if atom]
    atoms = [atom for atom in atoms if atom]
    return atoms


def verify_sentence_format_for_abstraction(sentence):
    """Verify the bracket format ``[entity] [information].``"""
    if not sentence.endswith('.'):
        sentence = sentence + '.'
    if not re.match(r'^\[.*\] \[.*\]\.$', sentence):
        raise ValueError(
            f"Sentence: '{sentence}' is not in the correct format for abstraction."
        )
    return sentence


def remove_first_last_brackets(text):
    if text.endswith('.'):
        text = text[:-1]
    text = re.sub(r'^\s*\[([^\]]*)\]', r'\1', text, count=1)
    text = re.sub(r'\[([^\]]*)\]\s*$', r'\1', text, count=1)
    return text + '.'
