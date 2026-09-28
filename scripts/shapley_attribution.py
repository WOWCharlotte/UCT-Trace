"""Shapley value attribution via attention-mask intervention.

This module replaces the previous string-replacement masking strategy
([REMOVED_*] placeholders) with embedding-level intervention. For each
coalition of "players" (text regions), the corresponding token embeddings
are zeroed out and the rest of the sequence is left intact. The output
logprob under that intervention becomes v(S), the coalition's value.

Why this is better than string-replacement masking:
    - Sequence length and RoPE positions are preserved.
    - The placeholder text is not a confounder — masked tokens truly
      contribute nothing to the residual stream.
    - All players see identical "removal" treatment, so v(S) is comparable
      across coalitions.

Public API:
    compute_shapley_values(players, values) -> dict[str, float]
    extract_region_spans(model, prompt, sample, granularity) -> dict
    logprob_with_masked_regions(...) -> float
    shapley(model, sample, output_text, granularity) -> dict
    shapley_rows(model, rows, scope) -> list[dict]
"""

"""
Complete computation process
 1. Build the exact chat-templated prompt used at inference time from
    `<system>`, `<user>`, `<data_fact>`, and `<data_attack>`.
 2. Define players according to the scheme in the design doc:
    coarse = [`auth`, `data`], fine = [`auth`, `data_fact`, `data_attack`].
 3. Tokenize the full prompt once and locate each player's token span in
    that exact prompt.
 4. Embed the prompt once with the model's input embedding layer.
 5. For each coalition S, keep the full token sequence unchanged and only
    zero the embeddings of players not in S.
 6. Concatenate masked prompt embeddings with the generated output tokens O
    and compute `v(S) = log P_theta(O | X_S)`.
 7. Enumerate the complete coalition-value table and compute exact Shapley
    values from weighted marginal contributions.

 Why the Attention mask is preserved
 - We do not delete tokens.
 - We do not insert placeholder text.
 - We keep sequence length, token positions, and the all-ones attention
   mask unchanged across coalitions.
 - Only embeddings are perturbed, so the measured difference is attributed
   to region semantics rather than prompt-length or masking-side effects.
"""


import argparse
import itertools
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# Player keys match the keys used in run_qwen3_shift_experiment.py so the
# JSON output schema stays identical.
AUTH_KEY = "auth"
DATA_KEY = "data"
FACT_KEY = "data_fact"
ATTACK_KEY = "data_attack"
NONE_LIKE_TEXTS = {"", "none", "none.", "null", "null.", "n/a", "na"}


def compute_shapley_values(
    players: List[str],
    values: Dict[Tuple[str, ...], float],
) -> Dict[str, float]:
    """Compute Shapley values from a complete coalition-value table.

    Args:
        players: Ordered list of player names. Order only affects tuple
            representation, not numerical results.
        values: Mapping from a coalition (tuple of player names) to its
            characteristic value v(S). Must contain every coalition
            2^len(players), including the empty coalition and the full
            coalition.

    Returns:
        Mapping from each player to its Shapley value phi_i.

    Formula:
        phi_i = sum over S in N \\ {i} of
            |S|! * (n - |S| - 1)! / n!  *  [v(S ∪ {i}) - v(S)]
    """
    n = len(players)
    phi: Dict[str, float] = {}
    for player in players:
        total = 0.0
        others = [p for p in players if p != player]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                with_player = tuple(sorted(subset + (player,), key=players.index))
                subset_key = tuple(sorted(subset, key=players.index))
                weight = (
                    math.factorial(len(subset))
                    * math.factorial(n - len(subset) - 1)
                    / math.factorial(n)
                )
                total += weight * (values[with_player] - values[subset_key])
        phi[player] = float(total)
    return phi


def build_auth(sample: dict) -> str:
    """Re-export so callers don't need to import the main script."""
    return f"<system>\n{sample['system']}\n</system>\n\n<user>\n{sample['user']}\n</user>"


def normalize_attack_text(value) -> str:
    """Normalize placeholder / missing attack text to the empty string."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in NONE_LIKE_TEXTS:
        return ""
    return text


def build_data(data_fact: str, data_attack: str) -> str:
    """Re-export so callers don't need to import the main script."""
    data_attack = normalize_attack_text(data_attack)
    return (
        "<data>\n"
        "<data_fact>\n"
        f"{data_fact}\n"
        "</data_fact>\n"
        "<data_attack>\n"
        f"{data_attack}\n"
        "</data_attack>\n"
        "</data>"
    )


def _build_prompt(sample: dict) -> str:
    """Build the canonical (template-free) auth+data block.

    This string is used as a fallback for span extraction only when the
    caller does not provide a chat-templated prompt explicitly. The
    production code paths thread the chat-templated prompt through so
    Shapley is scored on the exact same token sequence the model was
    conditioned on at inference time.
    """
    return f"{build_auth(sample)}\n\n{build_data(sample.get('data_fact', ''), normalize_attack_text(sample.get('data_attack', '')))}"


def build_chat_prompt(model, sample: dict, add_generation_prompt: bool = True) -> str:
    """Build the same chat-templated prompt used by the Qwen shift experiment."""
    auth = build_auth(sample)
    data = build_data(sample.get("data_fact", ""), normalize_attack_text(sample.get("data_attack", "")))
    if getattr(model, "provider", "") == "attn-hf-no-sys":
        messages = [
            {"role": "user", "content": auth + "\nData: " + data},
        ]
    else:
        messages = [
            {"role": "system", "content": auth},
            {"role": "user", "content": "Data: " + data},
        ]
    return model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def _tokenize_piece(model, text: str) -> List[int]:
    """Encode a piece of text into token IDs (no special tokens)."""
    return model.tokenizer.encode(text, add_special_tokens=False)


def extract_region_spans(
    model,
    prompt: str,
    sample: dict,
    granularity: str,
) -> Dict[str, Tuple[int, int]]:
    """Locate the token span for each Shapley player inside the prompt.

    Strategy:
        1. Tokenize the prompt once.
        2. Tokenize each region's literal text and find its first
           occurrence after the previous region. Spans cannot overlap
           and are returned in left-to-right order matching `players`.

    Returns a dict like:
        {"auth": (s, e), "data_fact": (s, e), "data_attack": (s, e)}
        or, for coarse granularity:
        {"auth": (s, e), "data": (s, e)}  # data covers both fact and attack

    Spans are inclusive-exclusive [start, end) over the tokenized prompt.
    Missing or empty regions are simply omitted from the result.
    """
    spans: Dict[str, Tuple[int, int]] = {}
    prompt_token_spans = _prompt_token_spans(model, prompt)
    if not prompt_token_spans:
        return {}

    # 1. auth region: from <system> to </user> inclusive of the wrapper text.
    auth_text = build_auth(sample)
    auth_chars = _find_text_span(prompt, auth_text, start=0)
    if auth_chars is not None:
        auth_span = _token_span_from_char_span(prompt_token_spans, *auth_chars)
    else:
        auth_span = None
    if auth_span is not None:
        spans[AUTH_KEY] = auth_span

    # 2. data_fact and data_attack regions.
    fact_text = sample.get("data_fact", "")
    attack_text = normalize_attack_text(sample.get("data_attack", ""))

    # Both regions sit inside the data block. Anchor on the data block
    # to keep fact/attack search ranges small and bounded.
    data_block_text = build_data(fact_text, attack_text)
    auth_char_end = auth_chars[1] if auth_chars is not None else 0
    data_block_chars = _find_text_span(prompt, data_block_text, start=auth_char_end)
    if data_block_chars is not None:
        data_block_span = _token_span_from_char_span(
            prompt_token_spans, data_block_chars[0], data_block_chars[1]
        )
        data_block_text_start = data_block_chars[0]
    else:
        data_block_span = None
        data_block_text_start = len(prompt)

    fact_span: Tuple[int, int] | None = None
    attack_span: Tuple[int, int] | None = None
    if fact_text:
        local_chars = _find_text_span(data_block_text, fact_text, start=0)
        if local_chars is not None:
            fact_span = _token_span_from_char_span(
                prompt_token_spans,
                data_block_text_start + local_chars[0],
                data_block_text_start + local_chars[1],
            )
        if fact_span is not None:
            spans[FACT_KEY] = fact_span

    if attack_text:
        # Search after the fact region to avoid matching against fact content.
        start_after_fact_chars = 0
        if fact_text:
            fact_chars_in_data = _find_text_span(data_block_text, fact_text, start=0)
            if fact_chars_in_data is not None:
                start_after_fact_chars = fact_chars_in_data[1]
        local_chars = _find_text_span(
            data_block_text, attack_text, start=start_after_fact_chars
        )
        if local_chars is not None:
            attack_span = _token_span_from_char_span(
                prompt_token_spans,
                data_block_text_start + local_chars[0],
                data_block_text_start + local_chars[1],
            )
        if attack_span is not None:
            spans[ATTACK_KEY] = attack_span

    # 3. Coarse data span = union of fact + attack spans.
    if granularity == "coarse":
        pieces = [s for s in (fact_span, attack_span) if s is not None]
        if pieces:
            data_start = min(s for s, _ in pieces)
            data_end = max(e for _, e in pieces)
            spans[DATA_KEY] = (data_start, data_end)
        elif data_block_span is not None:
            spans[DATA_KEY] = data_block_span

    return spans


def _find_text_span(text: str, needle: str, start: int = 0) -> Tuple[int, int] | None:
    """Find a literal substring span [start, end) in character space."""
    if not needle:
        return None
    pos = text.find(needle, start)
    if pos < 0:
        return None
    return pos, pos + len(needle)


def _prompt_token_spans(model, prompt: str) -> List[Tuple[int, int]]:
    """Return per-token character offsets for the full prompt."""
    encoded = model.tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError(
            "Tokenizer does not provide offset_mapping, which is required "
            "for robust Shapley span extraction."
        )
    return [(int(start), int(end)) for start, end in offsets]


def _token_span_from_char_span(
    offsets: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> Tuple[int, int] | None:
    """Map a character span [char_start, char_end) to a token span [i, j)."""
    indices = [
        i for i, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not indices:
        return None
    return indices[0], indices[-1] + 1


def _find_token_span(
    haystack: List[int],
    needle: List[int],
    start: int = 0,
    end: int | None = None,
) -> Tuple[int, int] | None:
    """Find the [start, end) span of `needle` inside `haystack`.

    Returns None if `needle` is empty or not found. Used to locate
    region boundaries in token space without depending on the model's
    special prompt template (so it works across providers).
    """
    if not needle:
        return None
    end = len(haystack) if end is None else min(end, len(haystack))
    last = end - len(needle)
    for i in range(max(0, start), last + 1):
        if haystack[i:i + len(needle)] == needle:
            return i, i + len(needle)
    return None


def _mask_span(embeds: "torch.Tensor", span: Tuple[int, int]) -> "torch.Tensor":
    """Return a copy of `embeds` with positions [span[0], span[1]) zeroed.

    Immutability note: we never mutate the input tensor — a fresh tensor
    is always returned so callers can reuse the original prompt embeds
    across coalitions without state leakage.
    """
    start, end = span
    masked = embeds.clone()
    masked[:, start:end, :] = 0.0
    return masked


def logprob_with_masked_regions(
    model,
    prompt_embeds: "torch.Tensor",
    output_ids: "torch.Tensor",
    region_spans: Dict[str, Tuple[int, int]],
    masked_players: Iterable[str],
) -> float:
    """Compute the output logprob with `masked_players` regions zeroed out.

    Args:
        model: A HF causal-LM model wrapper exposing `.model.device`,
            `.get_input_embeddings()` and `.forward(inputs_embeds=...)`.
            We rely only on the inner model's API, not on any wrapper
            specifics.
        prompt_embeds: Tensor of shape [1, T_prompt, H] — the prompt's
            input embeddings (output of get_input_embeddings()).
        output_ids: Tensor of shape [1, T_output] — token IDs to score.
        region_spans: Map from player key to its token span in the prompt.
        masked_players: Iterable of player keys to mask (i.e., remove
            from the model's view). All others are kept intact.

    Returns:
        The sum of log-probabilities assigned to output_ids under the
        masked prompt. Returns -inf if output_ids is empty.
    """
    if output_ids.shape[1] == 0:
        return float("-inf")

    masked_set = set(masked_players)
    embeds = prompt_embeds
    for player in masked_set:
        span = region_spans.get(player)
        if span is None:
            # Player's region not in the prompt (e.g., empty data_attack).
            # Treating it as already masked is the conservative choice —
            # v(S) when S includes such a player equals the v(S') where
            # S' excludes it, so the marginal contribution is 0.
            continue
        embeds = _mask_span(embeds, span)

    # Build the full sequence by concatenating prompt + output IDs.
    # We re-embed only the output portion; prompt embeddings are already
    # prepared and possibly masked.
    embed_layer = _get_input_embedding_layer(model)
    output_embeds = embed_layer(output_ids)
    full_embeds = torch.cat([embeds, output_embeds], dim=1)
    # Keep the attention mask fully visible. Shapley masking is an
    # embedding-level intervention, so positions and attention-mask shape
    # remain identical across every coalition.
    attention_mask = torch.ones(
        full_embeds.shape[:2], device=full_embeds.device, dtype=torch.long
    )

    inner = model.model if hasattr(model, "model") else model
    with torch.no_grad():
        logits = inner.forward(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
        ).logits

    prompt_len = prompt_embeds.shape[1]
    start = prompt_len - 1
    end = start + output_ids.shape[1]
    token_logits = logits[:, start:end, :]
    log_probs = F.log_softmax(token_logits.float(), dim=-1)
    gathered = log_probs.gather(2, output_ids.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum().item())


def shapley(
    model,
    sample: dict,
    output_text: str,
    granularity: str,
    prompt_text: str | None = None,
) -> dict:
    """Compute coarse or fine Shapley values for a single sample.

    The function:
        1. Resolves the prompt text (chat-templated if `prompt_text` is
           provided; otherwise the template-free auth+data block).
        2. Computes per-player token spans inside that exact prompt.
        3. For every coalition, runs a single forward pass with the
           masked regions' embeddings zeroed.
        4. Computes Shapley values from the coalition logprob table.

    Args:
        model: HF causal-LM wrapper.
        sample: Dataset row with `system`, `user`, `data_fact`, `data_attack`.
        output_text: The already-generated model output to score against.
        granularity: "coarse" → [auth, data] players; "fine" →
            [auth, data_fact, data_attack] players.
        prompt_text: Optional chat-templated prompt. When provided,
            Shapley is scored on the exact same token sequence the model
            saw at inference time. When None, the canonical
            template-free block is used (only useful for unit tests).

    Returns:
        Dict with keys:
            - players: ordered list of player names
            - values: {"+"-joined-coalition: v(S)} for every coalition
            - phi: {player_name: shapley_value}
    """
    if granularity == "coarse":
        players = [AUTH_KEY, DATA_KEY]
    elif granularity == "fine":
        players = [AUTH_KEY, FACT_KEY, ATTACK_KEY]
    else:
        raise ValueError(f"Unknown granularity: {granularity!r}")

    if prompt_text is None:
        prompt_text = _build_prompt(sample)
    region_spans = extract_region_spans(model, prompt_text, sample, granularity)

    # Embed the prompt once; every coalition clones and masks this tensor.
    prompt_ids = _tokenize_piece(model, prompt_text)
    embed_layer = _get_input_embedding_layer(model)
    prompt_embeds = embed_layer(
        torch.tensor([prompt_ids], device=_device_of(model))
    )
    output_ids_list = _tokenize_output(model, output_text)
    output_ids = torch.tensor(
        [output_ids_list], device=_device_of(model), dtype=torch.long
    )

    values: Dict[Tuple[str, ...], float] = {}
    for r in range(len(players) + 1):
        for coalition in itertools.combinations(players, r):
            masked_players = [p for p in players if p not in coalition]
            values[coalition] = logprob_with_masked_regions(
                model, prompt_embeds, output_ids, region_spans, masked_players
            )

    phi = compute_shapley_values(players, values)

    return {
        "players": players,
        "values": {"+".join(k) if k else "empty": v for k, v in values.items()},
        "phi": phi,
    }


def select_rows_for_shapley(rows: List[dict], scope: str) -> List[dict]:
    """Select rows for standalone Shapley attribution."""
    if scope == "candidates":
        return [row for row in rows if row.get("candidate_attention_shift_attack_failed")]
    if scope == "attention_shift":
        return [row for row in rows if row.get("attention_shift")]
    if scope == "all":
        return list(rows)
    raise ValueError(f"Unknown Shapley scope: {scope!r}")


def shapley_for_row(
    model,
    row: dict,
    granularities: Iterable[str] = ("coarse", "fine"),
) -> dict:
    """Attach Shapley results to one experiment result row."""
    sample = row["sample"]
    output = row["output"]
    prompt_text = build_chat_prompt(model, sample, add_generation_prompt=True)
    for granularity in granularities:
        row[f"shapley_{granularity}"] = shapley(
            model,
            sample,
            output,
            granularity,
            prompt_text=prompt_text,
        )
    return row


def shapley_rows(
    model,
    rows: List[dict],
    scope: str = "candidates",
    granularities: Iterable[str] = ("coarse", "fine"),
) -> List[dict]:
    """Compute Shapley values for selected rows from an experiment JSONL."""
    work_rows = select_rows_for_shapley(rows, scope)
    for row in tqdm(work_rows, desc="shapley"):
        shapley_for_row(model, row, granularities)
    return work_rows


def read_jsonl(path: str, limit: int | None = None) -> List[dict]:
    if limit == 0:
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _device_of(model) -> "torch.device":
    """Resolve the device the model lives on."""
    if hasattr(model, "model") and hasattr(model.model, "device"):
        return model.model.device
    if hasattr(model, "device"):
        return model.device
    return torch.device("cpu")


def _get_input_embedding_layer(model):
    """Resolve the HF input embeddings from either a wrapper or raw model."""
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    if hasattr(model, "model") and hasattr(model.model, "get_input_embeddings"):
        return model.model.get_input_embeddings()
    raise AttributeError(
        f"{type(model).__name__} does not expose get_input_embeddings() "
        "either directly or via `.model`."
    )


def _tokenize_output(model, output_text: str) -> List[int]:
    """Encode `output_text` into token IDs (no special tokens)."""
    return model.tokenizer.encode(output_text, add_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute coarse/fine Shapley attribution for JSONL rows produced "
            "by run_qwen3_shift_experiment.py. Interventions zero selected "
            "region embeddings while keeping the attention mask unchanged."
        )
    )
    parser.add_argument("--model_name", default="qwen3_8b-attn")
    parser.add_argument("--input", required=True, help="Input experiment JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL with Shapley fields.")
    parser.add_argument("--limit", type=int, help="Optional maximum rows to read.")
    parser.add_argument(
        "--shapley_scope",
        choices=["candidates", "attention_shift", "all"],
        default="candidates",
        help="Rows to compute Shapley for.",
    )
    parser.add_argument(
        "--granularity",
        choices=["both", "coarse", "fine"],
        default="coarse",
        help="Which Shapley granularity to compute.",
    )
    args = parser.parse_args()

    from baselines.attention_tracker.utils import create_model, open_config

    granularities = ("coarse", "fine") if args.granularity == "both" else (args.granularity,)
    rows = read_jsonl(args.input, args.limit)
    config = open_config(f"./configs/model_configs/{args.model_name}_config.json")
    model = create_model(config)
    attributed = shapley_rows(
        model,
        rows,
        scope=args.shapley_scope,
        granularities=granularities,
    )
    write_jsonl(args.output, attributed)


if __name__ == "__main__":
    main()
