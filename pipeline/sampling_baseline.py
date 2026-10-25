import os
import json

from generation_utils import get_model_completion, get_response_text, naming_config_to_str

N_SAMPLES = 5

RESOLVE_SYSTEM_PROMPT = (
    "You will be given a question and a list of responses.\n"
    "Write a single response that is as specific as possible while remaining consistent with all the original responses.\n"
    'If the responses have nothing meaningful in common with respect to the question, output "IDK." '
)


def run_sampling_baseline(entity, client, model, input_dir, prompt, prompts_dir, generation_params):
    """
    Sample N_SAMPLES generations from the model, then resolve them into a single
    response using a fixed prompt. The resolved response is saved as response.json
    in the output directory so the rest of the pipeline (atomize, trivial
    abstractions, questions) can consume it unchanged.

    Returns:
        Path to the output directory containing response.json.
    """
    naming_config = {'n': N_SAMPLES, 'prompt': prompt}
    output_dir = os.path.join(input_dir, "sampling_baseline", naming_config_to_str(naming_config))
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, "response.json")
    if os.path.exists(output_file_path):
        print(f"Sampling baseline response already exists. Skipping...")
        return output_dir

    # Load and format the generation prompt
    generate_prompt_path = os.path.join(prompts_dir, f"generation/{prompt}.txt")
    with open(generate_prompt_path, 'r') as f:
        generate_prompt_template = f.read()

    if "{theta}" in generate_prompt_template:
        # theta prompts are not expected for sampling_baseline; strip the placeholder
        entity_prompt = generate_prompt_template.replace("{theta}", "").format(entity=entity)
    else:
        entity_prompt = generate_prompt_template.format(entity=entity)

    # --- Step 1: Sample N_SAMPLES generations ---
    generations_dir = os.path.join(output_dir, "generations")
    os.makedirs(generations_dir, exist_ok=True)

    generation_texts = []
    for i in range(N_SAMPLES):
        gen_file = os.path.join(generations_dir, f"generation_{i}.json")
        if os.path.exists(gen_file):
            with open(gen_file, 'r') as f:
                gen_dict = json.load(f)
        else:
            print(f"Generating sample {i + 1}/{N_SAMPLES}...")
            gen_dict = get_model_completion(
                client=client,
                prompt=entity_prompt,
                model=model,
                generation_params=generation_params,
            )
            with open(gen_file, 'w') as f:
                json.dump(gen_dict, f, indent=4)

        generation_texts.append(get_response_text(gen_dict, model))

    # --- Step 2: Resolve the N generations into one response ---
    # Progressively truncate if the combined input is too long for the model
    current_texts = list(generation_texts)
    resolved_dict = None
    while current_texts:
        numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(current_texts))
        prompt_str = f"Question: {entity_prompt}\n\nResponses:\n{numbered}"
        try:
            resolved_dict = get_model_completion(
                client=client,
                prompt=prompt_str,
                model=model,
                system_prompt=RESOLVE_SYSTEM_PROMPT,
                generation_params=generation_params,
            )
            break
        except Exception as e:
            if len(current_texts) == 1:
                raise
            print(
                f"[sampling_baseline] Resolve call failed with {len(current_texts)} responses "
                f"(likely context-length error: {e}). Dropping last response and retrying..."
            )
            current_texts = current_texts[:-1]

    resolved_dict['config'] = {
        'prompt': prompt,
        'n': N_SAMPLES,
        'generate_prompt': entity_prompt,
    }

    with open(output_file_path, 'w') as f:
        json.dump(resolved_dict, f, indent=4)

    print(f"Sampling baseline response saved to {output_file_path}.")
    return output_dir
