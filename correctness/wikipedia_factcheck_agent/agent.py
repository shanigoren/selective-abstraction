# factcheck_agent.py
# Wikipedia fact-checking agent using OpenAI Agents SDK + gpt-oss-120b.
# - Uses ONLY Wikipedia via tools
# - Gathers evidence first, then decides
# - Final label is strictly: SUPPORTED or UNSUPPORTED
# - Structured Outputs (JSON schema) enforce the final response contract

import os
import sys
import re
import math
import json
import time
import asyncio
import logging
import signal
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure repository root is on sys.path so `cache` can be imported when
# this script is run as a subprocess from run_correctness_parallel.py
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import wikipedia
from bs4 import BeautifulSoup as _BS
from functools import partial
wikipedia.BeautifulSoup = partial(_BS, features="lxml")

from openai import AsyncOpenAI
from infobox_parser import WikiInfobox

from cache.result_cache import correctness_cache, cache_correctness_key_for_abstraction

# Signal handler for graceful shutdown
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals (SIGTERM/SIGINT) gracefully."""
    global _shutdown_requested
    if _shutdown_requested:
        # Already shutting down, force exit
        sys.exit(1)
    _shutdown_requested = True
    print("\n[agent.py] Shutdown signal received. Cleaning up cache connections...")
    try:
        correctness_cache.cleanup()
        print("[agent.py] Cache cleanup completed.")
    except Exception as e:
        print(f"[agent.py] Error during cache cleanup: {e}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

from agents import (
    Agent,
    AgentOutputSchemaBase,
    Runner,
    function_tool,
    OpenAIResponsesModel,
    ModelSettings,
    set_tracing_disabled,
)

FAILED = 'FAILED'

# --------------------------------- Config ---------------------------------

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("factcheck_agent")
logger.setLevel(logging.DEBUG)
set_tracing_disabled(True)  # avoids tracing noise unless you want it

# Ensure English Wikipedia
wikipedia.set_lang("en")

# vLLM endpoint should be provided via:
#   export SA_MODEL_BASE_URL="http://127.0.0.1:8000/v1"
#   export SA_MODEL_API_KEY="<local-endpoint-key>"

# Compact, deterministic tool I/O parameters
_CHUNK_SIZE = 1200
_OVERLAP = 200
_PAGE_CACHE: Dict[str, str] = {}

# Structured output schema (STRICT)
FACTCHECK_SCHEMA = {
    "name": "factcheck_result",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED"]},
            "rationale": {"type": "string", "maxLength": 280},
            "evidence": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "quote": {"type": "string", "maxLength": 600},
                    },
                    "required": ["title", "url", "quote"],
                },
            },
        },
        "required": ["label", "rationale", "evidence"],
    },
    "strict": True,
}

# --------------------------------- Utils ----------------------------------

def _normalize_wiki_text(text: str) -> str:
    # drop citation-like [12], [a], [citation needed]
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _OVERLAP) -> List[Tuple[int, str]]:
    chunks: List[Tuple[int, str]] = []
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + size)
        chunks.append((i, text[i:j]))
        if j == n:
            break
        i = j - overlap
    return chunks

def _score_chunk(query: str, chunk: str) -> float:
    # very light, deterministic scoring
    q_terms = [t for t in re.split(r"\W+", query.lower()) if t]
    if not q_terms:
        return 0.0
    cl = chunk.lower()
    score = sum(cl.count(t) for t in q_terms)
    return score / math.sqrt(max(1, len(chunk)))

def _with_backoff(fn, *args, retries: int = 1, base_delay: float = 0.5, **kwargs):
    # simple sync backoff wrapper for wikipedia lib calls
    ex: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            ex = e
            time.sleep(base_delay * (2**attempt))
    if ex:
        raise ex

# --------------------------------- Tools ----------------------------------

@function_tool
def search_wikipedia(query: str) -> Dict[str, Any]:
    """
    Search Wikipedia for page titles matching the query and return up to 10 candidates.

    Args:
        query: Natural language query, usually the full claim or entity.

    Returns:
        {
          "query": str,
          "results": [{"title": str, "summary": str}],
        }
    """
    # print(f"[debug] search_wikipedia: {query}")
    try:
        titles = _with_backoff(wikipedia.search, query, results=10)
        results = []
        for title in titles:
            summary = ""
            try:
                summary = _with_backoff(wikipedia.summary, title, auto_suggest=False, sentences=2)
            except Exception:
                pass
            results.append({"title": title, "summary": summary})
        return {"query": query, "results": results}
    except Exception as e:
        return {"query": query, "results": [], "error": str(e)}

@function_tool
def get_infobox(title: str) -> Dict[str, Any]:
    """
    Get the infobox of a Wikipedia page.
    """
    # print(f'[debug] get_infobox: {title}')
    wiki = WikiInfobox(lang="en")
    data = wiki.get_infobox(title)
    return data

@function_tool
def open_wikipedia_page(title: str) -> Dict[str, Any]:
    """
    Fetch and cache the text of a Wikipedia page by exact title.
    On disambiguation, returns options so the agent can pick a specific page.

    Args:
        title: Exact page title to open.

    Returns:
        {
          "title": str,
          "url": str,
          "num_chars": int,
          "lead": str,
          # on disambiguation: "disambiguation_options": [str, ...]
        }
    """
    # print(f"[debug] open_wikipedia_page: {title}")
    try:
        page = _with_backoff(wikipedia.page, title, auto_suggest=False, preload=True)
        content = _normalize_wiki_text(page.content or "")
        _PAGE_CACHE[page.title] = content
        return {
            "title": page.title,
            "url": page.url,
            "num_chars": len(content),
            "lead": content[:300],
        }
    except wikipedia.DisambiguationError as e:
        return {
            "title": title,
            "url": "",
            "num_chars": 0,
            "lead": "",
            "disambiguation_options": list(e.options)[:15],
            "error": "disambiguation",
        }
    except Exception as e:
        return {"title": title, "url": "", "num_chars": 0, "lead": "", "error": str(e)}

@function_tool
def rank_page_spans(title: str, query: str, k: int = 6) -> Dict[str, Any]:
    """
    Return the top-K relevant spans from a previously opened page.

    Args:
        title: Page title previously opened via open_wikipedia_page.
        query: Claim or focused sub-question.
        k:     Number of spans to return (default 6).

    Returns:
        {
          "title": str,
          "query": str,
          "spans": [{"start": int, "end": int, "score": float, "text": str}],
          "url": str
        }
    """
    # print(f"[debug] rank_page_spans: {title}, {query}, {k}")
    text = _PAGE_CACHE.get(title)
    if not text:
        return {"title": title, "query": query, "spans": [], "error": "page_not_cached"}

    url = ""
    try:
        page = _with_backoff(wikipedia.page, title, auto_suggest=False, preload=False)
        url = page.url
    except Exception:
        pass

    chunks = _chunk_text(text, size=_CHUNK_SIZE, overlap=_OVERLAP)
    scored = [( _score_chunk(query, ch), st, st + len(ch), ch ) for st, ch in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)

    spans = []
    for s, st, en, ch in scored[: max(1, k)]:
        spans.append({"start": st, "end": en, "score": round(float(s), 6), "text": ch[:800]})

    return {"title": title, "query": query, "spans": spans, "url": url}

# ----------------------------- Agent System Prompt ------------------------

SYSTEM_INSTRUCTIONS = """
You are a precise Wikipedia-only fact-checker.

Tasks:
1) Use the tools to SEARCH and READ Wikipedia (and only Wikipedia).
2) Gather relevant snippets from opened pages as evidence.
3) Decide whether the claim is SUPPORTED or UNSUPPORTED by Wikipedia.

Label policy:
- "SUPPORTED" only if Wikipedia contains clear supporting evidence.
- "UNSUPPORTED" if (a) the claim is contradicted/refuted or (b) you cannot find supporting evidence.

Output policy:
- You MUST use the tools to obtain evidence (do not rely on prior knowledge).
- Cite evidence using page titles/URLs and short quotes taken from spans.
- Your final output MUST follow the provided JSON schema exactly
  (Structured Outputs). Keep rationale ≤ 2 sentences, no chain-of-thought.

Output Schema:
{
  "label": "SUPPORTED" | "UNSUPPORTED",
  "rationale": "<string, max 280 chars, ≤2 sentences>",
  "evidence": [
    {
      "title": "<string>",
      "url": "<string>",
      "quote": "<string, max 600 chars>"
    },
    ...
  ]
}

Workflow:
- Start with search_wikipedia(query=claim).
- Open promising pages with open_wikipedia_page(title=...).
- If you think the infobox may contain relevant information, call get_infobox(title=...).
- For each opened page, call rank_page_spans(title, query=claim or sub-question).
- Decide and return the structured result.
"""

# ------------------------------- Builder ----------------------------------

def build_agent(args) -> Agent:
    api_key = (
        os.environ.get("SA_MODEL_API_KEY")
        or os.environ.get("VLLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Set SA_MODEL_API_KEY, VLLM_API_KEY, or OPENAI_API_KEY before running correctness evaluation."
        )
    model = OpenAIResponsesModel(
        openai_client=AsyncOpenAI(
            base_url=args.model_base_url,
            api_key=api_key,
        ),
        model=args.model,
    )
    reasoning = (
        None
        if getattr(args, "reasoning_effort", None) in (None, "none", "omit")
        else {"effort": args.reasoning_effort}
    )
    return Agent(
        name="factcheck_agent",
        instructions=SYSTEM_INSTRUCTIONS.strip(),
        model=model,
        tools=[search_wikipedia, open_wikipedia_page, rank_page_spans, get_infobox],
        model_settings=ModelSettings(
            # Current openai-agents uses `max_tokens`; the older
            # `max_output_tokens` name is silently ignored by this Pydantic
            # dataclass and would leave the requested token bound unapplied.
            max_tokens=args.max_tokens,
            reasoning=reasoning,
            tool_choice=None,
        ),
        output_type = FactCheckOutput
    )

# ------------------------------- Runner API --------------------------------

def log_reasoning(resp, logs_path, print_to_console=False):
    try:
        for i,r in enumerate(resp.raw_responses):
            log_message(f'=============== STEP {i} ===============', logs_path, print_to_console)
            if len(r.output) != 2:
                log_message(f'output len is not 2, it is {len(r.output)}', logs_path, print_to_console)
                log_message(f'{[_r for _r in r.output]}', logs_path, print_to_console)
            log_message(f'1) {r.output[0].type}: {r.output[0].content[0].text.strip()}', logs_path, print_to_console)
            log_message(f'2) {r.output[1].type}: {r.output[1]}', logs_path, print_to_console)
        log_message(f'=============== FINAL OUTPUT ===============', logs_path, print_to_console)
        log_message(resp.final_output, logs_path, print_to_console)
    except Exception as e:
        log_message(f'Error printing reasoning: {e}', logs_path, print_to_console)
        log_message(resp.final_output, logs_path, print_to_console)

async def fact_check_claim(agent: Agent, claim: str, max_steps: int = 30, timeout_s: int = 240) -> Dict[str, Any]:
    """
    Run the agent on a single claim and return the enforced-JSON result dict.
    """
    resp = await Runner.run(agent, claim, max_turns=max_steps)
    log_reasoning(resp, logs_path)
    return resp.final_output

from typing import Literal, List, TypedDict

class Evidence(TypedDict, total=False):
    title: str
    url: str
    quote: str
    source: str      # accept either title or source
    snippet: str     # accept either quote or snippet

class FactCheckOutput(TypedDict, total=False):
    label: Literal["SUPPORTED", "UNSUPPORTED"]
    rationale: str
    justification: str   # accept justification too
    evidence: List[Evidence] | str
    claim: str

# ---------------------------------- CLI -----------------------------------
def log_message(message, logs_path, print_to_console=True):
    # add timestamp to message
    message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    if print_to_console:
        print(message)
    with open(logs_path, 'a') as f:
        f.write(message + '\n')

def remove_first_last_brackets(text):
    # remove the dot from the end of the text if it exists
    if text.endswith('.'):
        text = text[:-1]
    # Remove the first set of brackets
    text = re.sub(r'^\s*\[([^\]]*)\]', r'\1', text, count=1)
    # Remove the last set of brackets
    text = re.sub(r'\[([^\]]*)\]\s*$', r'\1', text, count=1)
    # put the dot back 
    text = text + '.'
    return text

def correctness_result_exists(abs, skip_failed):
    if abs.get('correctness_result') and abs['correctness_result'] == FAILED:
        return skip_failed
    elif abs.get('correctness_result') and abs['correctness_result'].get('annotation') in ['SUPPORTED', 'UNSUPPORTED']:
        return True
    else:
        return False

def raise_exception(ex):
    raise ex

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wikipedia fact-checking agent")
    parser.add_argument("--model", type=str, default="gpt-oss-120b", help="Model name served by your vLLM endpoint.")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for the model.")
    parser.add_argument("--max_tokens", type=int, default=64000, help="Max tokens for the model.")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="Reasoning effort.")
    parser.add_argument("--max_steps", type=int, default=30, help="Max agent steps.")
    parser.add_argument("--timeout", type=int, default=240, help="Timeout in seconds.")
    parser.add_argument(
        "--skip_failed",
        action="store_true",
        default=False,
        help="Skip failed abstractions.",
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to abstractions with confidence.',
    )
    parser.add_argument('--max_attempts', type=int, default=3, help='Max attempts to run the agent for the same claim.')
    parser.add_argument(
        '--model_base_url',
        type=str,
        default=os.environ.get('SA_MODEL_BASE_URL', 'http://127.0.0.1:8000/v1'),
        help='OpenAI-compatible evaluator URL (default: SA_MODEL_BASE_URL or local port 8000).',
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
    args = parser.parse_args()
    abstractions_path = Path(args.input_path)
    with open(abstractions_path, 'r') as f:
        abstractions_data = json.load(f)

    config = {
        'model': args.model,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'reasoning_effort': args.reasoning_effort,
    }
    
    config_component = '__'.join(
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', str(k))}="
        f"{re.sub(r'[^A-Za-z0-9._-]+', '_', str(v))}"
        for k, v in config.items()
    )
    correctness_dir_path = abstractions_path.parent / 'wikifact_agent' / config_component
    correctness_dir_path.mkdir(parents=True, exist_ok=True)
    correctness_path = correctness_dir_path / 'correctness.json'
    logs_path = correctness_dir_path / 'correctness_logs.txt'
    
    try:
        with open(correctness_path, 'r') as f:
            correctness_results = json.load(f)
    except Exception as e:
        correctness_results = abstractions_data.copy()

    num_abstractions = 0
    for atom in correctness_results:
        for abstraction in atom['abstractions']:
            num_abstractions += 1
    log_message(f"Total number of abstractions: {num_abstractions}\n" + "="*100, logs_path)

    use_cache = args.use_cache and not args.no_cache

    async def _main():
        print(f"args: {args}")
        agent = build_agent(args)
        # count total number of abstractions
        i = 0    
        try:
            for atom in correctness_results:
                if _shutdown_requested:
                    break
                for abs in atom['abstractions']:
                    if _shutdown_requested:
                        break
                    i += 1
                    max_steps = args.max_steps
                    if correctness_result_exists(abs, args.skip_failed):
                        log_message(f'Skipping abstraction {i}/{num_abstractions}: {abs["abstraction"]}', logs_path, print_to_console=args.verbose)
                        continue
                    else:
                        cache_key = cache_correctness_key_for_abstraction(abs['abstraction']) if use_cache else None
                        if use_cache and cache_key is not None:
                            try:
                                cache_payload, source = correctness_cache.get_or_compute(
                                    task_type="correctness",
                                    abstraction_key=cache_key,
                                    compute_fn=lambda: None,
                                )
                            except Exception as e:
                                log_message(f'Error getting cached correctness: {e}', logs_path, print_to_console=args.verbose)
                                cache_payload = None
                            if cache_payload is not None and type(cache_payload) == dict and abs.get('correctness_result') != FAILED:
                                # Reuse cached annotation for this abstraction.
                                print(f'CACHE HIT for abstraction {i}/{num_abstractions}: {abs["abstraction"]}')
                                abs['correctness_result'] = cache_payload
                                log_message(
                                    f'Using cached correctness for abstraction {i}/{num_abstractions}: {abs["abstraction"]}',
                                    logs_path,
                                    print_to_console=args.verbose,
                                )
                                with open(correctness_path, 'w') as f:
                                    json.dump(correctness_results, f, indent=4)
                                log_message('='*100, logs_path)
                                continue
                        for attempt in range(1, args.max_attempts + 1):
                            if _shutdown_requested:
                                break
                            try:
                                start_time = time.time()
                                log_message(f'verifying abstraction {i}/{num_abstractions}: {abs["abstraction"]}, attempt {attempt+1}/{args.max_attempts}', logs_path, print_to_console=args.verbose)
                                result = await fact_check_claim(agent, remove_first_last_brackets(abs['abstraction']), max_steps=max_steps, timeout_s=args.timeout)
                                if not "label" in result:
                                    raise Exception('No label in final output')
                                log_message(f'{i}/{num_abstractions}: CLAIM: {abs["abstraction"]} ANNOTATION: {result["label"]}', logs_path)
                                abs['correctness_result'] = {
                                    'annotation': result.pop('label')
                                }
                                for k,v in result.items():
                                    abs['correctness_result'][k] = v
                                break
                            except KeyboardInterrupt:
                                raise
                            except Exception as e:
                                log_message(f'Attempt {attempt+1}/{args.max_attempts}: Error fact-checking claim: {e}', logs_path, print_to_console=args.verbose)
                                if "Max turns" in str(e):
                                    max_steps += 10
                                    log_message(f'Increasing max steps to {max_steps}', logs_path, print_to_console=args.verbose)
                                continue
                        if attempt == args.max_attempts and abs.get('correctness_result') and abs.get('correctness_result').get('annotation') not in ['SUPPORTED', 'UNSUPPORTED']:
                            abs['correctness_result'] = FAILED
                            log_message(f'Failed to verify abstraction {i}/{num_abstractions}: {abs["abstraction"]} after {args.max_attempts} attempts', logs_path)
                        log_message(f"Time taken for abstraction: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}", logs_path)
                        # In case of success, store the correctness_result in the cache.
                        if use_cache and cache_key is not None and 'correctness_result' in abs and abs.get('correctness_result') != FAILED:
                            try:
                                correctness_cache.store_success("correctness", cache_key, abs['correctness_result'])
                            except Exception as e:
                                log_message(f'Error storing correctness in cache: {e}', logs_path, print_to_console=args.verbose)
                        with open(correctness_path, 'w') as f:
                            json.dump(correctness_results, f, indent=4)
                    log_message('='*100, logs_path)
        except KeyboardInterrupt:
            print("\n[agent.py] KeyboardInterrupt received. Cleaning up...")
        finally:
            # Ensure cache is cleaned up even on normal exit or interruption
            try:
                correctness_cache.cleanup()
            except Exception:
                pass

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\n[agent.py] KeyboardInterrupt in asyncio.run. Cleaning up...")
        correctness_cache.cleanup()
