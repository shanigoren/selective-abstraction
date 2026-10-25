

import os
from generation_utils import *
import json


def run_generate_response(entity, client, model, input_dir, prompt, prompts_dir, generation_params, theta=None, reasoning_effort=None):
    naming_config = {
        'prompt': prompt
    }
    if theta is not None:
        naming_config['theta'] = int(theta)
    if reasoning_effort is not None:
        naming_config['reasoning_effort'] = reasoning_effort
    
    generate_prompt_path = os.path.join(prompts_dir, f"generation/{prompt}.txt")
    with open(generate_prompt_path, 'r') as f:
        generate_prompt = f.read()

    if "{theta}" in generate_prompt and theta is None:
        raise ValueError(f"Generation prompt {prompt} requires theta but theta was None.")
    
    full_config = {
        **naming_config,
        'generate_prompt': generate_prompt.format(entity=entity, theta=theta)
    }

    output_dir = os.path.join(input_dir, "generate_response", naming_config_to_str(naming_config))
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_path = os.path.join(output_dir, "response.json")

    if os.path.exists(output_file_path):
        print(f"Response for {entity} with this configuration already exists. Skipping...")
        return output_dir

    print(f"Generating response for {entity}, config: {full_config}")
    
    generate_response_dict = get_model_completion(
        client=client,
        prompt=full_config['generate_prompt'],
        model=model,
        generation_params=generation_params,
        reasoning_effort=reasoning_effort,
    )

    generate_response_dict['config'] = full_config

    with open(output_file_path, 'w') as f:
        json.dump(generate_response_dict, f, indent=4)

    if reasoning_effort is not None:
        stage_cost = {
            'stage': 'generate_response',
            'model': model,
            'reasoning_effort': reasoning_effort,
            **aggregate_costs([extract_call_cost(generate_response_dict)]),
        }
        with open(os.path.join(output_dir, 'cost.json'), 'w') as f:
            json.dump(stage_cost, f, indent=4)

    print(f"Response for {entity} saved to {output_file_path}.")
    return output_dir


