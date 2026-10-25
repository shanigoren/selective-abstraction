

import os
from generation_utils import *
from utils.text_utils import *
import json
import re

# format: 
# - Broad: How many people are there?
# - Specific: How many people have performances that showcase a deep respect and understanding of ancient musical traditions?
def parse_questions(response_text):
    """
    Parse the response to extract questions.
    """
    questions = {}
    lines = response_text.strip().split('\n')
    for line in lines:
        if not line.strip(): continue
        # make match not case sensitive
        match = re.match(r'^\s*-?\s*(\w+):\s*(.*)', line, re.IGNORECASE)
        if match:
            question_type, question = match.groups()
            questions[question_type.lower()] = question.strip()
    return questions

def _process_abstraction_for_questions(abstraction, atom_index, client, model, system_prompt, generation_params):
    # print(f"Creating questions for abstraction: {abstraction['abstraction']}")
    if abstraction.get('questions') and abstraction.get('questions').get('broad') and abstraction.get('questions').get('specific'):
        # print(f"Questions for abstraction {abstraction['abstraction']} already exist. Skipping...")
        return None

    response_dict = get_model_completion(
        client=client,
        system_prompt=system_prompt,
        prompt=remove_first_last_brackets(abstraction['abstraction']),
        model=model,
        generation_params=generation_params
    )
    try:
        response_text = get_response_text(response_dict, model)
        questions = parse_questions(normalize_spaces(response_text))
        if 'broad' not in questions or not questions['broad']:
            print(f"Warning: Broad question is missing for abstraction {abstraction['abstraction']}")
            return None
        if 'specific' not in questions or not questions['specific']:
            print(f"Warning: Specific question is missing for abstraction {abstraction['abstraction']}")
            return None
        print(f"Questions for atom {atom_index} - abstraction {abstraction['abstraction']}: {questions}")
        return questions
    except Exception as e:
        print(f"Error parsing questions for abstraction {abstraction['abstraction']}: {e}")
        # print(f"Response: {normalize_spaces(response_dict['choices'][0]['message']['content'])}")
        return None

def run_count_questions(model, client, sys_prompt, input_file, prompts_dir, generation_params):
    if not os.path.exists(input_file):
        print(f"Abstractions file not found in {input_file}. Skipping...")
        return

    system_prompt_path = f"{prompts_dir}/questions/{sys_prompt}.txt"
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read()
    
    naming_config = {
        'model': model,
        'sys_prompt': sys_prompt
    }

    output_dir = os.path.join(os.path.dirname(input_file), "questions", naming_config_to_str(naming_config))
    output_file_path = os.path.join(output_dir, "questions.json")

    # if output file doesn't exist, create the output directory
    if not os.path.exists(output_file_path):
        os.makedirs(output_dir, exist_ok=True)

    abstractions_data = load_json(input_file)
    try:
        questions_data = load_json(output_file_path)
    except Exception as e:
        # print(f"Error loading questions data from {output_file_path}: {e}")
        questions_data = abstractions_data

    if len(questions_data) != len(abstractions_data):
        print(f"Warning: Mismatch in number of abstractions and questions. {len(questions_data)} questions != {len(abstractions_data)} abstractions")
        questions_data = abstractions_data

    for atom in questions_data:
        for abstraction in atom['abstractions']:
            questions = _process_abstraction_for_questions(abstraction, atom['index'], client, model, system_prompt, generation_params)
            if questions:
                abstraction['questions'] = questions
                with open(output_file_path, 'w') as f:
                    json.dump(questions_data, f, indent=4)
                # print('----------------------------------')

    return output_dir

