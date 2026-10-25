import json
import pickle
import redis
import pandas as pd
import re
import time
import argparse
import signal
import copy
import shutil
from pathlib import Path

# set environment variable
import os
import sys

# Ensure repository root is on sys.path so `cache` can be imported when
# this script is run as a subprocess from run_coverage_parallel.py
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cache.result_cache import coverage_cache

# Signal handler for graceful shutdown
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals (SIGTERM/SIGINT) gracefully."""
    global _shutdown_requested
    if _shutdown_requested:
        # Already shutting down, force exit
        sys.exit(1)
    _shutdown_requested = True
    print("\n[coverage.py] Shutdown signal received. Cleaning up cache connections...")
    try:
        coverage_cache.cleanup()
        print("[coverage.py] Cache cleanup completed.")
    except Exception as e:
        print(f"[coverage.py] Error during cache cleanup: {e}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

redis_path = os.environ.get("SA_REDIS_SERVER") or shutil.which("redis-server")

# Check if Redis is running, and start it if not
def initialize_cache(port=6379):
    try:
        r = redis.Redis(host="localhost", port=port)
        r.ping()
        print("Redis server is running.")
    except redis.exceptions.ConnectionError:
        print("Redis server not found, starting it now...")
        if not redis_path:
            raise RuntimeError(
                "Redis is not running and redis-server was not found. Install Redis or set SA_REDIS_SERVER."
            )
        # Start Redis server
        import subprocess
        subprocess.run([redis_path, "--port", str(port), "--daemonize", "yes"])
        # Verify it started
        try:
            r = redis.Redis(host="localhost", port=port)
            r.ping()
            print("Redis server started successfully.")
        except redis.exceptions.ConnectionError:
            print("Failed to start Redis server. You may need to install it using: conda install -c conda-forge redis")


def parse_count(response_text):
    response_text = response_text.replace('*','')
    # First try to find the standardized COUNT format
    count_match = re.search(r'COUNT:\s*(\d[\d,]*)', response_text)
    if count_match:
        return int(count_match.group(1).replace(',', ''))
    return None

def validate_count(count):
    if count is None:
        raise ValueError("Count is None")
    if not isinstance(count, int):
        raise TypeError(f"Count is not an integer: {count}")
    if count < 0:
        raise ValueError(f"Count is negative: {count}")

def get_question_count(question, question_type, logs_path, chain_output_path, verbose=False):
    log_message(f"Running count for {question_type} question: {question}", logs_path, print_to_console=verbose)
    try:
        chain_output = semantic_parser_class.run_batch(questions=[{"question": question, "conversation_history": []}])
        log_message(f"Response:{chain_output[0]['response']}", logs_path, print_to_console=verbose)
        # Prefer logging the chain if available from the openai-based agent
        if isinstance(chain_output[0], dict) and 'chain' in chain_output[0]:
            # Write chain steps to both chain_outputs file and main logs for visibility
            log_message(f"Chain Output: {chain_output[0]['chain']}\n"+"-"*100, chain_output_path, print_to_console=verbose)
            log_message(f"Chain Output: {chain_output[0]['chain']}\n"+"-"*100, logs_path, print_to_console=verbose)
        else:
            # As an additional attempt, if a `log_reasoning` helper exists on the class, call it
            try:
                if hasattr(semantic_parser_class, 'last_response') and hasattr(semantic_parser_class, 'log_reasoning'):
                    semantic_parser_class.log_reasoning(semantic_parser_class.last_response, chain_output_path, print_to_console=True)
            except Exception:
                pass
            log_message(f"Chain Output: {chain_output[0]}\n"+"-"*100, chain_output_path, print_to_console=False)
        try:
            count_result = parse_count(chain_output[0]['response'])
            validate_count(count_result)
            log_message(f"Obtained count: {count_result}", logs_path)
            if count_result == 0:
                log_message(f"Warning: {question_type} count is 0 for question '{question}'.", logs_path)
            return count_result
        except Exception as e:
            log_message(f"Error parsing count for question '{question}': {e}", logs_path)
            raise
    except Exception as e:
        # Log the full error details
        log_message(f"Full error details for question '{question}': {str(e)}", logs_path)
        log_message(f"Error type: {type(e).__name__}", logs_path)
        raise

# log_message and log to file
def log_message(message, logs_path, print_to_console=True):
    # add timestamp to message
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    if print_to_console:
        print(message)
    with open(logs_path, 'a') as f:
        f.write(message + '\n')

QUESTION_TYPES = ['broad', 'specific']
FAILED_COUNT_VALUE = -1

parser = argparse.ArgumentParser(description='Get counts for questions using the Spinach agent.')
parser.add_argument('--model', type=str, default='gpt-oss-120b', help='Model to use for counting questions')
parser.add_argument('--reasoning_effort', type=str, default='high', help='Reasoning effort to use for counting questions')
parser.add_argument('--max_response_tokens', type=int, default=30000, help='Max response tokens to use for counting questions')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature to use for counting questions')
parser.add_argument('--top_p', type=float, default=1.0, help='Top p to use for counting questions')
parser.add_argument('--top_k', type=int, default=0, help='Top k to use for counting questions')
parser.add_argument(
    '--backend',
    type=str,
    default='legacy',
    choices=['legacy'],
    help='Coverage backend. This release contains the paper-relevant legacy SPINACH backend only.',
)
parser.add_argument(
    '--model_base_url',
    type=str,
    default=os.environ.get('SA_MODEL_BASE_URL', 'http://127.0.0.1:8000/v1'),
    help='Base URL for the model endpoint; overrides llm_config.yaml when passed explicitly or through SA_MODEL_BASE_URL.',
)
parser.add_argument(
    "--verbose", action="store_true", default=False, help="Print everything."
)
parser.add_argument(
    "--use_cache",
    action="store_true",
    default=False,
    help="Opt in to the global SQLite cache. Disabled by default for reproducibility.",
)
parser.add_argument(
    "--no_cache",
    action="store_true",
    default=False,
    help="Explicitly disable cache reuse (the default; retained for compatible commands).",
)
parser.add_argument(
    '--questions_path',
    type=str,
    required=True,
    help='Path to questions.json.',
)
    

def _merge_missing_questions_into_counts(
    *,
    counts_data: list,
    questions_data: list,
    question_types: list[str],
) -> tuple[list, bool, dict]:
    """
    Ensure `counts_data` contains the questions present in `questions_data`.

    This fixes cases where questions exist in questions.json but are missing from counts.json.
    We merge *only* question text (and do not overwrite existing questions unless missing).

    Returns:
      (updated_counts_data, changed, stats)
    """
    changed = False
    stats = {
        "atoms_appended": 0,
        "abstractions_appended": 0,
        "abstractions_questions_filled": 0,
        "question_fields_filled": 0,
    }

    # If counts.json is truncated, append missing atoms wholesale (with no counts).
    if len(counts_data) < len(questions_data):
        for atom in questions_data[len(counts_data):]:
            counts_data.append(copy.deepcopy(atom))
            stats["atoms_appended"] += 1
            changed = True

    for atom_idx, q_atom in enumerate(questions_data):
        if atom_idx >= len(counts_data):
            break
        c_atom = counts_data[atom_idx]
        q_abs_list = q_atom.get("abstractions") or []
        c_abs_list = c_atom.get("abstractions") or []

        # If an atom's abstractions list is truncated, append missing abstractions.
        if len(c_abs_list) < len(q_abs_list):
            for abs_obj in q_abs_list[len(c_abs_list):]:
                c_abs_list.append(copy.deepcopy(abs_obj))
                stats["abstractions_appended"] += 1
                changed = True
            c_atom["abstractions"] = c_abs_list

        # Build a fallback lookup within this atom by abstraction text.
        q_by_abstraction_text = {}
        for qa in q_abs_list:
            key = qa.get("abstraction")
            if isinstance(key, str) and key not in q_by_abstraction_text:
                q_by_abstraction_text[key] = qa

        for abs_idx, c_abs in enumerate(c_abs_list):
            # Prefer index alignment, fallback to matching by abstraction string.
            q_abs = q_abs_list[abs_idx] if abs_idx < len(q_abs_list) else None
            if not isinstance(q_abs, dict):
                q_abs = q_by_abstraction_text.get(c_abs.get("abstraction"))
            if not isinstance(q_abs, dict):
                continue

            q_questions = q_abs.get("questions") or {}
            if not isinstance(q_questions, dict):
                q_questions = {}

            if not c_abs.get("questions") or not isinstance(c_abs.get("questions"), dict):
                c_abs["questions"] = {}
                stats["abstractions_questions_filled"] += 1
                changed = True

            c_questions = c_abs["questions"]
            for qt in question_types:
                if (qt not in c_questions or c_questions.get(qt) in (None, "")) and q_questions.get(qt):
                    c_questions[qt] = q_questions[qt]
                    stats["question_fields_filled"] += 1
                    changed = True

    return counts_data, changed, stats


if __name__ == "__main__":
    args = parser.parse_args()
    model = args.model

    # Make sure coverage/ is importable even if we later chdir() elsewhere (e.g., a temp llm_config dir).
    _COVERAGE_DIR = os.path.abspath(os.path.dirname(__file__))
    if _COVERAGE_DIR not in sys.path:
        sys.path.insert(0, _COVERAGE_DIR)

    # Set base URL for the OpenAI-compatible endpoint (default aligns with fact-check agent)
    os.environ["OPENAI_BASE_URL"] = args.model_base_url

    naming_config = {
        'model': model,
        'reasoning_effort': args.reasoning_effort,
        'max_response_tokens': args.max_response_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
    }
    
    # ChainLite changes the working directory below; keep all artifacts anchored
    # to the original input location, including when the CLI receives a relative path.
    questions_json_path = Path(args.questions_path).resolve()
    config_component = '__'.join(
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', str(k))}="
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', str(v))}"
        for k, v in naming_config.items()
    )
    output_dir = questions_json_path.parent / "spinach_counts" / config_component
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_json_path = output_dir / "counts.json"
    logs_path = output_dir / "logs.txt"
    chain_output_path = output_dir / "chain_outputs.txt"
    print(f"Saving results to {counts_json_path}")

    use_cache = args.use_cache and not args.no_cache

    answered_questions = {}
    # Always load the source questions.json so we can backfill missing questions into counts.json.
    with open(questions_json_path, 'r') as f:
        questions_data = json.load(f)
    try:
        with open(counts_json_path, 'r') as f:
            counts_data = json.load(f)
        for atom in counts_data:
            for abstraction in atom['abstractions']:
                if abstraction.get('counts'):
                    for question_type, count in abstraction['counts'].items():
                            if count != FAILED_COUNT_VALUE:
                                q = (abstraction.get('questions') or {}).get(question_type)
                                if q:
                                    answered_questions[q] = count
    except Exception as e:
        # Initialize counts from questions if counts.json doesn't exist yet (or is unreadable)
        counts_data = copy.deepcopy(questions_data)

    # Backfill missing *questions* into counts_data from questions_data, then persist immediately.
    try:
        counts_data, changed, merge_stats = _merge_missing_questions_into_counts(
            counts_data=counts_data,
            questions_data=questions_data,
            question_types=QUESTION_TYPES,
        )
        # Now that questions may have been backfilled, populate answered_questions for reuse.
        for atom in counts_data:
            for abstraction in atom.get('abstractions', []) or []:
                counts = abstraction.get('counts') or {}
                questions = abstraction.get('questions') or {}
                if not isinstance(counts, dict) or not isinstance(questions, dict):
                    continue
                for question_type, count in counts.items():
                    if count != FAILED_COUNT_VALUE and questions.get(question_type):
                        answered_questions[questions[question_type]] = count
        if changed:
            with open(counts_json_path, 'w') as f:
                json.dump(counts_data, f, indent=4)
            log_message(
                "Backfilled missing questions from questions.json into counts.json. "
                f"Stats: {merge_stats}",
                logs_path,
                print_to_console=True,
            )
    except Exception as merge_e:
        # Don't fail the run if merging fails; keep previous behavior but surface the issue.
        log_message(
            f"Warning: failed to merge missing questions from questions.json into counts.json: {merge_e}",
            logs_path,
            print_to_console=True,
        )

    # Ensure Redis is running before initializing the parser
    initialize_cache()

    def _explicitly_set_cli_flag(flag_names: tuple[str, ...]) -> bool:
        # We want to preserve prior behavior even though --model_base_url has a default.
        # Only override llm_config.yaml when the user explicitly passes the flag.
        argv = sys.argv[1:]
        return any(f in argv for f in flag_names)

    def _write_temp_llm_config_override_api_base(
        *,
        src_llm_config_path: str,
        dst_llm_config_path: str,
        override_api_base: str,
        prompt_log_file: str,
    ) -> str:
        """
        Create a modified copy of llm_config.yaml that overrides ONLY llm_endpoints[0].api_base.
        Also rewrites prompt_dirs and prompt_logging.log_file to absolute paths so the config
        can live outside coverage/.

        Returns the original api_base from llm_endpoints[0] (best-effort).
        """
        import yaml

        with open(src_llm_config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        original_api_base = None
        try:
            original_api_base = (cfg.get("llm_endpoints") or [{}])[0].get("api_base")
        except Exception:
            original_api_base = None

        # Override only the first endpoint
        cfg.setdefault("llm_endpoints", [])
        if not cfg["llm_endpoints"]:
            cfg["llm_endpoints"] = [{}]
        if not isinstance(cfg["llm_endpoints"][0], dict):
            cfg["llm_endpoints"][0] = {}
        cfg["llm_endpoints"][0]["api_base"] = override_api_base

        # Make prompt_dirs absolute (relative to the original YAML location)
        src_dir = os.path.abspath(os.path.dirname(src_llm_config_path))
        prompt_dirs = cfg.get("prompt_dirs") or []
        if isinstance(prompt_dirs, list):
            cfg["prompt_dirs"] = [
                (p if os.path.isabs(p) else os.path.abspath(os.path.join(src_dir, p)))
                for p in prompt_dirs
            ]

        # Make prompt log file absolute (prefer run output dir)
        cfg.setdefault("prompt_logging", {})
        if not isinstance(cfg["prompt_logging"], dict):
            cfg["prompt_logging"] = {}
        cfg["prompt_logging"]["log_file"] = prompt_log_file

        os.makedirs(os.path.dirname(dst_llm_config_path), exist_ok=True)
        with open(dst_llm_config_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        return original_api_base or ""

    # ChainLite auto-discovers llm_config.yaml based on CWD. Preserve the
    # default CWD=coverage behavior unless the endpoint is explicitly overridden.
    model_base_url_explicit = (
        _explicitly_set_cli_flag(("--model_base_url",))
        or "SA_MODEL_BASE_URL" in os.environ
    )
    if model_base_url_explicit:
        src_llm_config = os.path.join(_COVERAGE_DIR, "llm_config.yaml")
        tmp_dir = os.path.join(output_dir, "chainlite_tmp")
        dst_llm_config = os.path.join(tmp_dir, "llm_config.yaml")
        prompt_log_file = os.path.join(output_dir, "prompt_logs.jsonl")
        original_api_base = _write_temp_llm_config_override_api_base(
            src_llm_config_path=src_llm_config,
            dst_llm_config_path=dst_llm_config,
            override_api_base=args.model_base_url,
            prompt_log_file=prompt_log_file,
        )
        os.chdir(tmp_dir)
        log_message(
            f"Legacy backend: overriding llm_config.yaml llm_endpoints[0].api_base "
            f"from '{original_api_base}' to '{args.model_base_url}'. Using temp config: {dst_llm_config}",
            logs_path,
            print_to_console=True,
        )
    else:
        os.chdir(_COVERAGE_DIR)
        log_message(
            "Legacy backend: using coverage/llm_config.yaml (no endpoint override provided).",
            logs_path,
            print_to_console=True,
        )
    from spinach_agent.part_to_whole_parser import PartToWholeParser as semantic_parser_class
    semantic_parser_class.initialize(
        engine=model, 
        max_response_tokens = args.max_response_tokens, 
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        top_p=args.top_p,
        # top_k=args.top_k,
    ) 

    # count total number of abstractions
    num_abstractions = 0
    for atom in counts_data:
        for abstraction in atom['abstractions']:
            num_abstractions += 1
    log_message(f"Total number of abstractions: {num_abstractions}\n" + "="*100, logs_path)

    i = 0
    try:
        for atom in counts_data:
            if _shutdown_requested:
                break
            for abstraction in atom['abstractions']:
                if _shutdown_requested:
                    break
                i += 1
                if not abstraction.get('counts'):
                    abstraction['counts'] = {}
                log_message(f"Abstraction #{i}/{num_abstractions}: {abstraction['abstraction']}", logs_path, print_to_console=True)
                for question_type in QUESTION_TYPES:
                    if _shutdown_requested:
                        break
                    question_start_time = time.time()
                    if abstraction['counts'].get(question_type) and abstraction['counts'][question_type] != FAILED_COUNT_VALUE:
                        log_message(f"Counts for {question_type} question already exist. Skipping...", logs_path, print_to_console=args.verbose)
                        continue
                    question = abstraction.get('questions', {}).get(question_type, None)
                    if question is None:
                        log_message(f"Question {question_type} not found for abstraction {i}/{num_abstractions}: {abstraction['abstraction']}", logs_path, print_to_console=args.verbose)
                        continue
                    if question in answered_questions:
                        abstraction['counts'][question_type] = answered_questions[question]
                        log_message(f"Using existing count for {question_type} question: {question} - Count: {abstraction['counts'][question_type]}", logs_path, print_to_console=args.verbose)
                    else:
                        try:
                            if use_cache:
                                cache_payload, source = coverage_cache.get_or_compute(
                                    task_type="coverage",
                                    abstraction_key=question,
                                    compute_fn=lambda: {
                                        "count": get_question_count(
                                            question, question_type, logs_path, chain_output_path
                                        )
                                    },
                                )
                                count_result = cache_payload["count"]
                                log_message(
                                    f"Got count for {question_type} question from {source}: "
                                    f"{question} - Count: {count_result}",
                                    logs_path,
                                    print_to_console=True,
                                )
                            else:
                                count_result = get_question_count(
                                    question, question_type, logs_path, chain_output_path
                                )
                            abstraction['counts'][question_type] = count_result
                            answered_questions[question] = count_result
                            print(f"\033[32mGot count successfully for abstraction {i}/{num_abstractions}: {question}. Count: {count_result}\033[0m")
                            log_message(f"Time taken to run question: {time.strftime('%H:%M:%S', time.gmtime(time.time() - question_start_time))}", logs_path, print_to_console=args.verbose)
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            log_message(f"Error running {question_type} question: {question}. Error: {e}", logs_path, print_to_console=args.verbose)
                            abstraction['counts'][question_type] = FAILED_COUNT_VALUE
                    log_message(f"Time taken for question: {time.strftime('%H:%M:%S', time.gmtime(time.time() - question_start_time))}", logs_path, print_to_console=args.verbose)
                    with open(counts_json_path, 'w') as f:
                        json.dump(counts_data, f, indent=4)
                log_message('----------------------------------', logs_path, print_to_console=args.verbose)
    except KeyboardInterrupt:
        print("\n[coverage.py] KeyboardInterrupt received. Cleaning up...")
    finally:
        # Ensure cache is cleaned up even on normal exit or interruption
        try:
            coverage_cache.cleanup()
        except Exception:
            pass




