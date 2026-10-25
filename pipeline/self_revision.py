import os
import json
from generation_utils import *
from utils.text_utils import load_json
from typing import Optional


def _get_original_prompt(config: dict) -> str:
    """Walk nested base_generate_config to find the original generate_prompt."""
    prompt = config.get("generate_prompt", "")
    if not prompt:
        nested = config.get("base_generate_config", {})
        if nested:
            prompt = _get_original_prompt(nested)
    return prompt


def run_self_revision(
    model,
    client,
    input_dir,
    prompts_dir,
    generation_params,
    sys_prompt: str = "self_revision",
    theta: Optional[int] = None,
    output_subdir_name: Optional[str] = None,
):
    """
    Run a second-pass redaction step over an existing generated response.

    This step:
      - Reads the original generation response from `input_dir/response.json`.
      - Shows the original prompt and answer to the model.
      - Uses a redaction system prompt to instruct the model to remove
        information it is uncertain about.
      - Writes a new `response.json` under an `self_revision` subdirectory
        that mirrors the standard generation output structure.
    """
    system_prompt_path = os.path.join(prompts_dir, "self_revision", f"{sys_prompt}.txt")
    if not os.path.exists(system_prompt_path):
        raise FileNotFoundError(
            f"Self-revision system prompt not found at {system_prompt_path}"
        )

    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()

    if "{theta}" in system_prompt:
        if theta is None:
            raise ValueError(
                f"Self-revision prompt {sys_prompt} requires theta but theta was None."
            )
        system_prompt = system_prompt.format(theta=int(theta))

    if output_subdir_name is not None:
        output_dir = os.path.join(input_dir, output_subdir_name)
    else:
        naming_config = {"sys_prompt": sys_prompt}
        if theta is not None:
            naming_config["theta"] = int(theta)
        output_dir = os.path.join(
            input_dir,
            "self_revision",
            naming_config_to_str(naming_config),
        )
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, "response.json")
    if os.path.exists(output_file_path):
        print(
            f"Self-revised response already exists in {output_file_path}. "
            f"Skipping self_revision step."
        )
        return output_dir

    base_response_path = os.path.join(input_dir, "response.json")
    if not os.path.exists(base_response_path):
        raise FileNotFoundError(
            f"Base response.json not found in generation output dir: {base_response_path}"
        )

    base_response = load_json(base_response_path)
    base_config = base_response.get("config", {})
    base_prompt = _get_original_prompt(base_config)
    base_message = get_response_text(base_response, model)

    # The redaction behavior is controlled entirely by the system prompt.
    # The user-facing content here simply provides the original prompt and
    # answer as context.
    redaction_input = f"Original prompt:\n{base_prompt}\n\nAnswer:\n{base_message}"

    redacted_response = get_model_completion(
        client=client,
        system_prompt=system_prompt,
        prompt=redaction_input,
        model=model,
        generation_params=generation_params,
    )

    # Attach a config block that records how this redacted response was produced,
    # while keeping a reference to the original generation config.
    redacted_response["config"] = {
        "redaction_sys_prompt": sys_prompt,
        "theta": int(theta) if theta is not None else None,
        "base_generate_config": base_config,
    }

    with open(output_file_path, "w") as f:
        json.dump(redacted_response, f, indent=4)

    print(f"Self-revised response saved to {output_file_path}.")
    return output_dir


