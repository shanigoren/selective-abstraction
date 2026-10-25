from openai import OpenAI
import os
import json
import time

OPENAI_KEYS = {
    "model","messages","max_tokens","temperature","top_p","stop","seed",
    "logprobs","top_logprobs","tools","tool_choice","response_format","presence_penalty",
    "frequency_penalty","n","stream"
}

def _split_params(params: dict):
    # drop None-valued keys
    params = {k: v for k, v in params.items() if v is not None}
    std = {k: v for k, v in params.items() if k in OPENAI_KEYS}
    extra = {k: v for k, v in params.items() if k not in OPENAI_KEYS}
    return std, extra


def init_vllm_client(base_url=None, timeout=300):
    base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=1800,
    )

    return client

def is_openai_model(full_model_name):
    return full_model_name.startswith('gpt') and 'gpt-oss' not in full_model_name

def init_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for OpenAI-hosted models.")
    client = OpenAI(
        api_key=api_key,
    )

    return client

def init_client(full_model_name, model_base_url):
    if is_openai_model(full_model_name):
        return init_openai_client()
    return init_vllm_client(model_base_url)

def get_openai_response_text(completion):
    return completion.get("output", [{}])[-1]['content'][0]['text']

def get_response_text(response, model):
    if is_openai_model(model):
        return get_openai_response_text(response)
    return response['choices'][0]['message']['content']
    
def naming_config_to_str(naming_config):
    return '__'.join([f"{k}={str(naming_config[k])}" for k in sorted(naming_config.keys())])

def extract_call_cost(response_dict):
    """Extract token counts and latency from a get_model_completion response dict."""
    usage = response_dict.get('usage', {}) or {}
    details = usage.get('completion_tokens_details', {}) or {}
    return {
        'completion_tokens': usage.get('completion_tokens', 0) or 0,
        'reasoning_tokens': details.get('reasoning_tokens', 0) or 0,
        'prompt_tokens': usage.get('prompt_tokens', 0) or 0,
        'latency_seconds': response_dict.get('_latency_seconds', 0.0) or 0.0,
    }


def aggregate_costs(call_costs):
    """Sum a list of per-call cost dicts into a stage-level total."""
    return {
        'total_completion_tokens': sum(c['completion_tokens'] for c in call_costs),
        'total_reasoning_tokens': sum(c['reasoning_tokens'] for c in call_costs),
        'total_prompt_tokens': sum(c['prompt_tokens'] for c in call_costs),
        'total_latency_seconds': sum(c['latency_seconds'] for c in call_costs),
        'n_calls': len(call_costs),
    }


def get_model_completion(
    client: OpenAI,
    prompt: str,
    model: str,
    max_tokens=None,
    temperature=None,
    top_p=None,
    stop=None,
    seed=123,
    tools=None,
    logprobs=False,  # whether to return log probabilities; enable only when needed (e.g., confidence scoring)
    top_logprobs=None,
    system_prompt=None,  # System prompt to be used in the messages list
    generation_params=None,
    extract_prompt_logprobs=False,
    extract_prompt_logprobs_user_prompt=None,
    reasoning_effort=None,
) -> str:
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt}
        ]
    if extract_prompt_logprobs:
        messages = [
            {"role": "user", "content": extract_prompt_logprobs_user_prompt},
            {"role": "assistant","content": prompt}
        ]
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    # Merge any extra generation parameters (e.g., top_k, presence_penalty, etc.)
    # Values in extra_params take precedence over the defaults above
    if generation_params:
        params.update({k: v for k, v in generation_params.items() if v is not None})
    if tools:
        params["tools"] = tools

    std_params, extra_params = _split_params(params)
    
    if is_openai_model(model):
        effort = reasoning_effort if reasoning_effort is not None else "medium"
        std_params['reasoning'] = {"effort": effort}
        std_params['max_output_tokens'] = 25000
        std_params['input'] = std_params.pop('messages')
        std_params.pop('seed')
        std_params.pop('logprobs')
        _t0 = time.time()
        completion = client.responses.create(**std_params, extra_body=extra_params)
        result = completion.to_dict()
        result['_latency_seconds'] = time.time() - _t0
        return result
    if reasoning_effort is not None:
        extra_params['reasoning_effort'] = reasoning_effort
    _t0 = time.time()
    completion = client.chat.completions.create(**std_params, extra_body=extra_params)
    result = completion.to_dict()
    result['_latency_seconds'] = time.time() - _t0
    return result

