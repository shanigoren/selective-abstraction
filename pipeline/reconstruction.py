import argparse
import os
import json
from generation_utils import (
    init_client,
    is_openai_model,
    naming_config_to_str,
    get_model_completion,
    extract_call_cost,
    aggregate_costs,
)
from utils.text_utils import remove_first_last_brackets


def run_reconstruction(client, model, original_prompt, statements, input_dir,
                       prompts_dir, generation_params, theta, reasoning_effort=None,
                       sys_prompt='recon_rf_v6'):
    """Reconstruct a fluent answer from a list of selected-abstraction statements.

    Mirrors ``generate_response.run_generate_response`` so that the downstream
    atomize/questions/coverage stages run unchanged on the output. The
    reconstruction prompt (``prompts/reconstruction/{sys_prompt}.txt``) is used
    as the system prompt; the original prompt and the statements are passed as the
    user message. ``sys_prompt`` selects which prompt variant to use (and the output
    dir name), so different candidate prompts write to separate, non-colliding dirs.
    """
    naming_config = {
        'sys_prompt': sys_prompt,
        'theta': int(round(theta)),
    }
    if reasoning_effort is not None:
        naming_config['reasoning_effort'] = reasoning_effort

    output_dir = os.path.join(input_dir, "reconstruction", naming_config_to_str(naming_config))
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "response.json")

    if os.path.exists(output_file_path):
        print(f"Reconstruction with this configuration already exists. Skipping...")
        return output_dir

    reconstruction_prompt_path = os.path.join(prompts_dir, f"reconstruction/{sys_prompt}.txt")
    with open(reconstruction_prompt_path, 'r') as f:
        reconstruction_system_prompt = f.read()

    # Statements are stored bracketed (e.g. "[Entity] [fact].");
    # turn them into natural sentences before handing them to the model.
    clean_statements = [remove_first_last_brackets(s) for s in statements]
    numbered = [f"{i}. {s}" for i, s in enumerate(clean_statements, 1)]
    user_prompt = (
        f"Prompt:\n{original_prompt}\n\n"
        f"Statements ({len(numbered)} total):\n" + "\n".join(numbered)
    )

    if not clean_statements:
        if is_openai_model(model):
            reconstruction_response_dict = {
                'output': [{'content': [{'text': ''}]}],
            }
        else:
            reconstruction_response_dict = {
                'choices': [{'message': {'content': ''}}],
            }
        reconstruction_response_dict['config'] = {
            **naming_config,
            'generate_prompt': original_prompt,
            'abstained': True,
        }
        with open(output_file_path, 'w') as f:
            json.dump(reconstruction_response_dict, f, indent=4)
        print(f"No statements survived selection; wrote an empty reconstruction to {output_file_path}.")
        return output_dir

    print(f"Reconstructing answer using model {model}, config: {naming_config} "
          f"({len(clean_statements)} statements)")

    reconstruction_response_dict = get_model_completion(
        client=client,
        system_prompt=reconstruction_system_prompt,
        prompt=user_prompt,
        model=model,
        generation_params=generation_params,
        reasoning_effort=reasoning_effort,
    )

    reconstruction_response_dict['config'] = {
        **naming_config,
        'generate_prompt': original_prompt,
    }

    with open(output_file_path, 'w') as f:
        json.dump(reconstruction_response_dict, f, indent=4)

    if reasoning_effort is not None:
        stage_cost = {
            'stage': 'reconstruction',
            'model': model,
            'reasoning_effort': reasoning_effort,
            **aggregate_costs([extract_call_cost(reconstruction_response_dict)]),
        }
        with open(os.path.join(output_dir, 'cost.json'), 'w') as f:
            json.dump(stage_cost, f, indent=4)

    print(f"Reconstructed answer saved to {output_file_path}.")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconstruct a fluent answer from selected abstractions.')
    parser.add_argument('--model', type=str, required=True, help='Model to use for reconstruction')
    parser.add_argument('--model_base_url', type=str, default=None, help='Base URL for locally hosted models')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Entity generate_response dir (.../generate_response/prompt:<dataset>)')
    parser.add_argument('--original_prompt', type=str, required=True)
    parser.add_argument('--statements_json', type=str, required=True,
                        help='Path to a JSON file with a list of statement strings')
    parser.add_argument('--theta', type=int, required=True)
    parser.add_argument('--prompts_dir', type=str, default='pipeline/prompts')
    parser.add_argument(
        '--sys_prompt',
        type=str,
        default='recon_rf_v6',
        choices=['recon_rf_v6', 'reconstruction'],
        help='Validated v6 prompt by default; reconstruction retains the original source prompt.',
    )
    parser.add_argument(
        '--generation_params_path',
        type=str,
        required=True,
        help='JSON file containing the selected model generation parameters.',
    )
    parser.add_argument('--reasoning_effort', type=str, default=None, choices=['low', 'medium', 'high'])
    args = parser.parse_args()

    with open(args.statements_json, 'r') as f:
        statements = json.load(f)
    with open(args.generation_params_path, 'r') as f:
        all_generation_params = json.load(f)
    if args.model not in all_generation_params:
        raise ValueError(
            f"Model {args.model!r} is missing from {args.generation_params_path}."
        )
    generation_params = all_generation_params[args.model]
    client = init_client(args.model, args.model_base_url)
    run_reconstruction(
        client=client,
        model=args.model,
        original_prompt=args.original_prompt,
        statements=statements,
        input_dir=args.input_dir,
        prompts_dir=args.prompts_dir,
        generation_params=generation_params,
        theta=args.theta,
        reasoning_effort=args.reasoning_effort,
        sys_prompt=args.sys_prompt,
    )
