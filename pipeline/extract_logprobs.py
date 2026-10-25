from generation_utils import init_client, get_model_completion
from utils.text_utils import remove_first_last_brackets, load_json

def _select_realized_and_rank1(entry):
    """
    From a single provider entry (dict of token_id -> {decoded_token, logprob, rank}),
    return:
    - realized (token, logprob, rank) using the provider ordering (first item)
    - rank1 (token, logprob) using the candidate with rank==1 if present, else best logprob
    """
    if not isinstance(entry, dict) or len(entry) == 0:
        return None, None, None, None, None
    # Realized token: assume provider returns realized first
    first_key = next(iter(entry))
    realized = entry.get(first_key, {})
    realized_token = realized.get('decoded_token')
    realized_logprob = realized.get('logprob')
    realized_rank = realized.get('rank')
    # Rank1 candidate: prefer rank==1; fallback to max logprob
    best_rank = None
    best_rank_item = None
    best_lp_item = None
    for _, item in entry.items():
        if not isinstance(item, dict):
            continue
        # Track best by rank
        rank_val = item.get('rank')
        if isinstance(rank_val, int) and (best_rank is None or rank_val < best_rank):
            best_rank = rank_val
            best_rank_item = item
        # Track best by logprob
        lp_val = item.get('logprob')
        if isinstance(lp_val, (int, float)):
            if best_lp_item is None or lp_val > best_lp_item.get('logprob', float('-inf')):
                best_lp_item = item
    rank1 = best_rank_item if (isinstance(best_rank, int) and best_rank == 1) else best_rank_item or best_lp_item
    rank1_token = rank1.get('decoded_token') if isinstance(rank1, dict) else None
    rank1_logprob = rank1.get('logprob') if isinstance(rank1, dict) else None
    return (
        realized_token,
        (float(realized_logprob) if isinstance(realized_logprob, (int, float)) else None),
        realized_rank if isinstance(realized_rank, int) else None,
        rank1_token,
        (float(rank1_logprob) if isinstance(rank1_logprob, (int, float)) else None),
    )


def realized_token_logprobs_from_prompt(prompt_logprobs_list, source_prompt=None):
    """
    For each entry, choose the realized token (provider ordering) and also extract
    the rank-1 candidate. Returns a list of dicts:
      {
        'realized_token': { 'token': str, 'logprob': float, 'rank': Optional[int] },
        'rank_1_token': { 'token': Optional[str], 'logprob': Optional[float] }
      }
    """
    tokens = []
    if not isinstance(prompt_logprobs_list, list):
        return tokens
    for entry in prompt_logprobs_list:
        if not entry:
            continue
        tok, lp, rk, rank1_tok, rank1_lp = _select_realized_and_rank1(entry)
        if tok is None or lp is None:
            continue
        tokens.append({
            "realized_token": {
                "token": tok,
                "logprob": float(lp),
                "rank": rk
            },
            "rank_1_token": {
                "token": rank1_tok,
                "logprob": (float(rank1_lp) if isinstance(rank1_lp, (int, float)) else None)
            }
        })
    return tokens


def extract_prompt_tokens_and_logprobs(prompt_logprobs_list, source_prompt=None):
    # Returns structured tokens as produced by realized_token_logprobs_from_prompt
    return realized_token_logprobs_from_prompt(prompt_logprobs_list, source_prompt)


def _is_sentence_boundary(token, sentence_terminators):
    if not isinstance(token, str):
        return False
    if "\n" in token:
        return True
    if sentence_terminators:
        for term in sentence_terminators:
            if term and term in token:
                return True
    return False


def split_token_logprobs_by_sentence(tokens_and_logprobs, include_whitespace=False, sentence_terminators=None):
    """
    Group tokens and logprobs by sentence-like boundaries. A boundary is any token that
    contains a newline or one of the 'sentence_terminators' (default ['.']), or special markers.
    The boundary token is included in the current group before starting a new one.
    Special handling: tokens like '.\\n' are included in the current group and then split.
    """
    if not isinstance(tokens_and_logprobs, list):
        return []

    groups = [[]]
    for item in tokens_and_logprobs:
        # Legacy tuple form: (token, logprob)
        if isinstance(item, (tuple, list)) and len(item) == 2:
            token, lp = item
            if not isinstance(token, str):
                continue
            has_newline = ("\n" in token)
            terms = sentence_terminators or ['.']
            has_terminator = any((t in token) for t in terms if t)
            if include_whitespace or token.strip() != "":
                groups[-1].append((token, float(lp)))
            if has_terminator or has_newline:
                groups.append([])
        # Structured dict forms
        elif isinstance(item, dict):
            # New nested structure
            if "realized_token" in item and isinstance(item["realized_token"], dict):
                token = item["realized_token"].get("token")
                lp = item["realized_token"].get("logprob")
            else:
                # Old flat dict fallback
                token = item.get("token")
                lp = item.get("logprob")
            if not isinstance(token, str):
                continue
            has_newline = ("\n" in token)
            terms = sentence_terminators or ['.']
            has_terminator = any((t in token) for t in terms if t)
            if include_whitespace or token.strip() != "":
                groups[-1].append((token, float(lp)))
            if has_terminator or has_newline:
                groups.append([])
    return groups
def _split_token_logprobs_by_sentence_structured(tokens_and_logprobs, include_whitespace=False, sentence_terminators=None):
    """
    Same as split_token_logprobs_by_sentence but preserves structured dicts with
    realized/rank1 fields so downstream consumers can access them.
    """
    if not isinstance(tokens_and_logprobs, list):
        return []
    groups = [[]]
    for item in tokens_and_logprobs:
        # Expect dicts with 'token'/'logprob'
        if not isinstance(item, dict):
            continue
        if "realized_token" in item and isinstance(item["realized_token"], dict):
            token = item["realized_token"].get("token")
        else:
            token = item.get("token")
        if not isinstance(token, str):
            continue
        has_newline = ("\n" in token)
        terms = sentence_terminators or ['.']
        has_terminator = any((t in token) for t in terms if t)
        if include_whitespace or token.strip() != "":
            groups[-1].append(item)
        if has_terminator or has_newline:
            groups.append([])
    return groups


def _filter_tokens_to_last_message(tokens_and_logprobs):
    """
    Deprecated: provider-agnostic alignment should use _slice_tokens_to_prompt.
    Currently returns the input unchanged.
    """
    return tokens_and_logprobs


def _slice_tokens_to_prompt(tokens_and_logprobs, prompt_text):
    """
    Find a contiguous slice of tokens that exactly concatenates to prompt_text.
    No assumptions on provider formatting or special markers.
    """
    if not isinstance(tokens_and_logprobs, list) or not isinstance(prompt_text, str) or prompt_text == "":
        return []
    # Try to find a token-aligned slice whose concatenation equals prompt_text
    for start_idx in range(len(tokens_and_logprobs)):
        p_idx = 0
        end_idx = start_idx
        while end_idx < len(tokens_and_logprobs) and p_idx <= len(prompt_text):
            item = tokens_and_logprobs[end_idx]
            if isinstance(item, (tuple, list)) and len(item) == 2:
                tok_str = item[0] if isinstance(item[0], str) else ""
            elif isinstance(item, dict):
                if "realized_token" in item and isinstance(item["realized_token"], dict):
                    tok_str = item["realized_token"].get("token") if isinstance(item["realized_token"].get("token"), str) else ""
                else:
                    tok_str = item.get("token") if isinstance(item.get("token"), str) else ""
            else:
                tok_str = ""
            # If the remaining prompt does not start with the token, break
            if not prompt_text.startswith(tok_str, p_idx):
                break
            p_idx += len(tok_str)
            end_idx += 1
            if p_idx == len(prompt_text):
                # Exact match achieved
                return tokens_and_logprobs[start_idx:end_idx]
        # otherwise move start forward and try again
    return []


def sum_logprobs_by_sentence(tokens_and_logprobs, include_whitespace=False, sentence_terminators=None):
    """
    Compute summed logprob per line given a pre-extracted list of (token, logprob).
    Sentence boundaries are inferred from tokens containing '\n' or any of the
    provided 'sentence_terminators' (default ['.']) or special markers.
    """
    if not isinstance(tokens_and_logprobs, list):
        return []

    groups = split_token_logprobs_by_sentence(
        tokens_and_logprobs,
        include_whitespace=include_whitespace,
        sentence_terminators=sentence_terminators or ['.']
    )
    return [sum(lp for _, lp in group) for group in groups]


def compute_line_logprobs(tokens_and_logprobs, include_whitespace=False):
    # Backwards-compatible wrapper for previous name; default splits on '.'
    return sum_logprobs_by_sentence(tokens_and_logprobs, include_whitespace, sentence_terminators=['.'])


def group_token_logprobs_by_sentence(tokens_and_logprobs, include_whitespace=False, sentence_terminators=None):
    """
    Group tokens by sentence boundaries:
     - If input is tuples (token, logprob), returns list[list[tuple]]
     - If input is dicts (flat or nested), returns list[list[dict]] preserving items
    """
    if isinstance(tokens_and_logprobs, list) and len(tokens_and_logprobs) > 0 and isinstance(tokens_and_logprobs[0], dict):
        return _split_token_logprobs_by_sentence_structured(
            tokens_and_logprobs,
            include_whitespace=include_whitespace,
            sentence_terminators=sentence_terminators or ['.']
        )
    return split_token_logprobs_by_sentence(
        tokens_and_logprobs,
        include_whitespace=include_whitespace,
        sentence_terminators=sentence_terminators or ['.']
    )

def token_logprobs_as_dicts(prompt_logprobs_list, include_whitespace=False, sentence_terminators=None, last_message_only=False, prompt_text=None):
    """
    Convert a prompt-level logprobs structure into a list of sentences, where each
    sentence is a list of dicts with fields:
      {
        'realized_token': { 'token': str, 'logprob': float, 'rank': Optional[int] },
        'rank_1_token': { 'token': Optional[str], 'logprob': Optional[float] }
      }
    """
    tokens = realized_token_logprobs_from_prompt(prompt_logprobs_list)
    if isinstance(prompt_text, str) and prompt_text != "":
        tokens = _slice_tokens_to_prompt(tokens, prompt_text)
    elif last_message_only:
        tokens = _filter_tokens_to_last_message(tokens)
    grouped = group_token_logprobs_by_sentence(
        tokens,
        include_whitespace=include_whitespace,
        sentence_terminators=sentence_terminators or ['.']
    )
    # Already structured; just ensure types are normalized
    normalized = []
    for group in grouped:
        norm_group = []
        for item in group:
            if not isinstance(item, dict):
                continue
            rt = item.get("realized_token", {})
            r1 = item.get("rank_1_token", {})
            norm_group.append({
                "realized_token": {
                    "token": rt.get("token"),
                    "logprob": (float(rt.get("logprob")) if isinstance(rt.get("logprob"), (int, float)) else None),
                    "rank": rt.get("rank") if isinstance(rt.get("rank"), int) else None
                },
                "rank_1_token": {
                    "token": r1.get("token"),
                    "logprob": (float(r1.get("logprob")) if isinstance(r1.get("logprob"), (int, float)) else None)
                }
            })
        normalized.append(norm_group)
    return normalized


