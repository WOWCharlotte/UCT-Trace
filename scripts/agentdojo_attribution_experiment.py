"""Attention and Shapley attribution for AgentDojo trajectories."""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import inspect
import itertools
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from collections import Counter, defaultdict
import statistics
from typing import Iterable

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.cache_utils import DynamicCache


# Phi-3 remote code from older Transformers releases calls this renamed API.
if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = _get_usable_length


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shapley_attribution import compute_shapley_values
from attack_judge_support import attack_dominant, contribution_fields


DEFAULT_INPUT_ROOT = "data/agentdojo/runs/qwen3-8b"
DEFAULT_OUTPUT_DIR = "result/agentdojo_qwen3_important_instructions"
DEFAULT_MODEL_CONFIG = "configs/model_configs/qwen3_8b-attn_config.json"
INPUT_PATTERN = "*/user_task_*/important_instructions/*.json"
EXPECTED_CANDIDATES = 629
EXPECTED_ELIGIBLE = 565
ATTACK_TYPE = "important_instructions"
INFORMATION_RE = re.compile(r"<INFORMATION>.*?</INFORMATION>", re.DOTALL | re.IGNORECASE)


LLAMA3_CHAT_TEMPLATE = """{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + (message['content'] or '') | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"""


@dataclass(frozen=True)
class SelectedCase:
    source_path: str
    trajectory: dict
    polluted_tool_index: int
    target_assistant_index: int
    attack_char_spans: tuple[tuple[int, int], ...]
    injection_match: bool
    target_id: str


@dataclass(frozen=True)
class RenderedTarget:
    prompt: str
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    target_text: str
    target_kind: str
    messages: tuple[dict, ...]


AUTH_KEY = "auth"
FACT_KEY = "data_fact"
ATTACK_KEY = "data_attack"
SPECIAL_KEY = "special"
PLAYERS = (AUTH_KEY, FACT_KEY, ATTACK_KEY)
RegionSpans = dict[str, list[tuple[int, int]]]


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL row in {path}:{line_number}: {error}") from error
    return rows


def rows_by_target_id(rows: Iterable[dict]) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        target = row.get("target_id")
        if target:
            indexed[str(target)] = row
    return indexed


def discover_input_paths(input_root: str) -> list[str]:
    pattern = os.path.join(input_root, INPUT_PATTERN)
    return sorted(glob.glob(pattern))


def information_spans(content: str | None) -> list[tuple[int, int]]:
    if not content:
        return []
    return [match.span() for match in INFORMATION_RE.finditer(content)]


def normalize_injection_text(text: str) -> str:
    normalized = str(text).replace("\\n", "").replace("\\", "")
    normalized = re.sub(r"\s+", "", normalized).casefold()
    # Some YAML-rendered tool results escape apostrophes by doubling them.
    return normalized.replace("''", "'")


def attack_blocks_match_injections(trajectory: dict, blocks: list[str]) -> bool:
    declared = [
        normalize_injection_text(value)
        for value in trajectory.get("injections", {}).values()
        if isinstance(value, str)
    ]
    normalized_blocks = [normalize_injection_text(block) for block in blocks]
    if not declared or not normalized_blocks:
        return False
    return all(any(block in injection or injection in block for injection in declared) for block in normalized_blocks)


def first_polluted_tool(messages: list[dict]) -> tuple[int, list[tuple[int, int]]] | None:
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        spans = information_spans(message.get("content"))
        if spans:
            return index, spans
    return None


def next_assistant_index(messages: list[dict], after_index: int) -> int | None:
    for index in range(after_index + 1, len(messages)):
        if messages[index].get("role") == "assistant":
            return index
    return None


def case_id(trajectory: dict) -> str:
    return "/".join([
        str(trajectory.get("suite_name", "unknown")),
        str(trajectory.get("user_task_id", "unknown")),
        str(trajectory.get("attack_type", ATTACK_TYPE)),
        str(trajectory.get("injection_task_id", "unknown")),
    ])


def target_id(trajectory: dict, tool_index: int, assistant_index: int) -> str:
    return f"{case_id(trajectory)}@tool-{tool_index}:assistant-{assistant_index}"


def select_case(path: str, trajectory: dict) -> tuple[SelectedCase | None, str | None]:
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        return None, "invalid_messages"
    selected_tool = first_polluted_tool(messages)
    if selected_tool is None:
        return None, "no_information_attack_block"
    tool_index, spans = selected_tool
    assistant_index = next_assistant_index(messages, tool_index)
    if assistant_index is None:
        return None, "no_post_attack_assistant"
    content = messages[tool_index].get("content") or ""
    blocks = [content[start:end] for start, end in spans]
    return SelectedCase(
        source_path=path,
        trajectory=trajectory,
        polluted_tool_index=tool_index,
        target_assistant_index=assistant_index,
        attack_char_spans=tuple(spans),
        injection_match=attack_blocks_match_injections(trajectory, blocks),
        target_id=target_id(trajectory, tool_index, assistant_index),
    ), None


def mistral_tool_call_id(value: object) -> str:
    """Return the 9-character alphanumeric id required by Mistral templates."""
    raw = str(value or "tool-call")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:9]


def qwen_message(message: dict, *, normalize_mistral_ids: bool = False) -> dict:
    # Phi-3's template concatenates message content directly. AgentDojo uses
    # None for assistant messages that contain only a tool call.
    converted = {"role": message["role"], "content": str(message.get("content") or "")}
    if message.get("role") == "tool" and message.get("tool_call_id") is not None:
        converted["tool_call_id"] = mistral_tool_call_id(message["tool_call_id"]) if normalize_mistral_ids else message["tool_call_id"]
    tool_calls = message.get("tool_calls")
    if message.get("role") == "assistant" and tool_calls:
        converted["tool_calls"] = [{
            "type": "function",
            "function": {
                "name": call["function"],
                "arguments": call.get("args", {}),
            },
            "id": mistral_tool_call_id(call.get("id")) if normalize_mistral_ids else call.get("id"),
        } for call in tool_calls]
    return converted


def render_messages_for_tokenizer(tokenizer, messages: list[dict]) -> list[dict]:
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).casefold()
    normalize_mistral_ids = "mistral" in tokenizer_name
    is_gemma = "gemma" in tokenizer_name
    is_phi3 = "phi-3" in tokenizer_name or "phi3" in tokenizer_name
    is_llama = "llama" in tokenizer_name
    rendered_messages = [qwen_message(message, normalize_mistral_ids=normalize_mistral_ids) for message in messages]
    if not is_gemma and not is_phi3 and not is_llama:
        return rendered_messages
    if is_gemma:
        if not any(message.get("role") == "system" for message in rendered_messages):
            return rendered_messages
        system_content = "\n\n".join(
            str(message.get("content") or "")
            for message in rendered_messages
            if message.get("role") == "system"
        )
        source_messages = [message for message in rendered_messages if message.get("role") != "system"]
        for index, message in enumerate(source_messages):
            if message.get("role") == "user":
                source_messages[index] = {
                    **message,
                    "content": system_content + "\n\n" + str(message.get("content") or ""),
                }
                break
    else:
        source_messages = rendered_messages
    normalized = []
    for message in source_messages:
        role = message.get("role")
        if role == "tool":
            normalized.append({
                "role": "user",
                "content": "[Tool result]\n" + str(message.get("content") or ""),
            })
            continue
        if role == "assistant" and message.get("tool_calls") and (is_phi3 or is_llama or not message.get("content")):
            calls = [
                {
                    "name": call["function"]["name"],
                    "arguments": call["function"].get("arguments", {}),
                }
                for call in message["tool_calls"]
            ]
            tool_call_text = "[Tool call]\n" + json.dumps(calls, ensure_ascii=False)
            content = str(message.get("content") or "")
            normalized.append({
                "role": "assistant",
                "content": content + "\n\n" + tool_call_text if content else tool_call_text,
            })
            continue
        normalized.append(message)
    return normalized


def assistant_target_kind(message: dict) -> str:
    has_content = bool(message.get("content"))
    has_calls = bool(message.get("tool_calls"))
    if has_content and has_calls:
        return "mixed"
    if has_calls:
        return "tool_calls"
    if has_content:
        return "content"
    raise ValueError("Selected assistant message has neither content nor tool calls")


def render_selected_target(tokenizer, selected: SelectedCase) -> RenderedTarget:
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "")).casefold()
    if not getattr(tokenizer, "chat_template", None) and "llama" in tokenizer_name:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    source_messages = selected.trajectory["messages"]
    rendered_messages = render_messages_for_tokenizer(
        tokenizer, source_messages[:selected.target_assistant_index + 1]
    )
    prefix_messages = tuple(rendered_messages[:-1])
    target_message = rendered_messages[-1]
    full_messages = (*prefix_messages, target_message)
    template_args = {"enable_thinking": False}
    prompt = tokenizer.apply_chat_template(
        list(prefix_messages), tokenize=False, add_generation_prompt=True, **template_args
    )
    prompt_ids = tokenizer.apply_chat_template(
        list(prefix_messages), tokenize=True, add_generation_prompt=True, **template_args
    )
    full_ids = tokenizer.apply_chat_template(
        list(full_messages), tokenize=True, add_generation_prompt=False, **template_args
    )
    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Native full-message rendering does not extend the decision prefix")
    target_ids = full_ids[len(prompt_ids):]
    if not target_ids:
        raise ValueError("Native chat template produced an empty assistant target")
    target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
    source_target = source_messages[selected.target_assistant_index]
    validate_target_round_trip(source_target, target_text)
    return RenderedTarget(
        prompt=prompt,
        prompt_ids=tuple(prompt_ids),
        target_ids=tuple(target_ids),
        target_text=target_text,
        target_kind=assistant_target_kind(source_target),
        messages=prefix_messages,
    )


def validate_target_round_trip(message: dict, target_text: str) -> None:
    content = message.get("content")
    if content and str(content) not in target_text:
        raise ValueError("Serialized target does not preserve assistant content")
    for call in message.get("tool_calls") or []:
        if str(call.get("function")) not in target_text:
            raise ValueError("Serialized target does not preserve a tool name")
        for value in (call.get("args") or {}).values():
            serialized = json.dumps(value, ensure_ascii=False)
            if isinstance(value, str):
                serialized = serialized[1:-1]
            if isinstance(value, (str, int, float, bool)) and str(serialized) not in target_text:
                raise ValueError("Serialized target does not preserve a tool argument")


def token_offsets(tokenizer, text: str) -> list[tuple[int, int]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer must provide offset_mapping")
    return [(int(start), int(end)) for start, end in offsets]


def char_to_token_span(
    offsets: list[tuple[int, int]], start: int, end: int
) -> tuple[int, int] | None:
    indices = [index for index, (left, right) in enumerate(offsets) if right > start and left < end]
    if not indices:
        return None
    return indices[0], indices[-1] + 1


def locate_source(prompt: str, source: str, start: int = 0) -> tuple[int, int]:
    position = prompt.find(source, start)
    if position < 0:
        raise ValueError("Source content is absent from native prompt rendering")
    return position, position + len(source)


def original_user_message(trajectory: dict) -> dict:
    for message in trajectory.get("messages", []):
        if message.get("role") == "user":
            return message
    raise ValueError("Trajectory has no original user message")


def complement_intervals(length: int, blocked: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    result = []
    cursor = 0
    for start, end in sorted(blocked):
        if start < cursor or end < start or end > length:
            raise ValueError("Attack intervals overlap or exceed selected tool content")
        if cursor < start:
            result.append((cursor, start))
        cursor = end
    if cursor < length:
        result.append((cursor, length))
    return result


def subtract_token_intervals(
    span: tuple[int, int], blocked: Iterable[tuple[int, int]]
) -> list[tuple[int, int]]:
    pieces = [span]
    for blocked_start, blocked_end in sorted(blocked):
        updated = []
        for start, end in pieces:
            if end <= blocked_start or blocked_end <= start:
                updated.append((start, end))
                continue
            if start < blocked_start:
                updated.append((start, blocked_start))
            if blocked_end < end:
                updated.append((blocked_end, end))
        pieces = updated
    return [piece for piece in pieces if piece[0] < piece[1]]


def extract_region_spans(tokenizer, selected: SelectedCase, rendered: RenderedTarget) -> RegionSpans:
    prompt = rendered.prompt
    offsets = token_offsets(tokenizer, prompt)
    spans: RegionSpans = {AUTH_KEY: [], FACT_KEY: [], ATTACK_KEY: []}
    user_content = str(original_user_message(selected.trajectory).get("content") or "")
    auth_chars = locate_source(prompt, user_content)
    auth_span = char_to_token_span(offsets, *auth_chars)
    if auth_span:
        spans[AUTH_KEY].append(auth_span)

    tool_content = str(selected.trajectory["messages"][selected.polluted_tool_index].get("content") or "")
    tool_chars = locate_source(prompt, tool_content, start=auth_chars[1])
    tool_base = tool_chars[0]
    for start, end in selected.attack_char_spans:
        attack_span = char_to_token_span(offsets, tool_base + start, tool_base + end)
        if attack_span:
            spans[ATTACK_KEY].append(attack_span)
    for start, end in complement_intervals(len(tool_content), selected.attack_char_spans):
        fact_span = char_to_token_span(offsets, tool_base + start, tool_base + end)
        if fact_span:
            spans[FACT_KEY].extend(subtract_token_intervals(fact_span, spans[ATTACK_KEY]))
    return spans


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def validate_region_spans(rendered: RenderedTarget, spans: RegionSpans) -> dict:
    issues = []
    prompt_len = len(rendered.prompt_ids)
    for player in PLAYERS:
        if not spans.get(player):
            issues.append({"type": "missing_player", "player": player})
        for span in spans.get(player, []):
            if span[0] < 0 or span[1] > prompt_len or span[0] >= span[1]:
                issues.append({"type": "invalid_bounds", "player": player, "span": list(span)})
    assigned = [(player, span) for player in PLAYERS for span in spans.get(player, [])]
    for index, (left_player, left) in enumerate(assigned):
        for right_player, right in assigned[index + 1:]:
            if spans_overlap(left, right):
                issues.append({
                    "type": "player_overlap",
                    "left_player": left_player,
                    "right_player": right_player,
                    "left": list(left),
                    "right": list(right),
                })
    return {
        "ok": not issues,
        "issues": issues,
        "prompt_token_count": prompt_len,
        "target_token_count": len(rendered.target_ids),
        "num_spans": {player: len(spans.get(player, [])) for player in PLAYERS},
        "fact_source": "first_polluted_tool_only",
    }


def region_for_token(index: int, spans: RegionSpans) -> str:
    for region in (ATTACK_KEY, FACT_KEY, AUTH_KEY):
        if any(start <= index < end for start, end in spans.get(region, [])):
            return region
    return SPECIAL_KEY


def attention_metrics(region_scores: dict[str, float], threshold: float = 0.5) -> dict:
    prompt_mass = float(sum(region_scores.get(region, 0.0) for region in (*PLAYERS, SPECIAL_KEY)))
    prompt_denominator = prompt_mass or 1.0
    prompt_normalized = {
        region: float(region_scores.get(region, 0.0)) / prompt_denominator
        for region in (*PLAYERS, SPECIAL_KEY)
    }
    player_mass = float(sum(region_scores.get(region, 0.0) for region in PLAYERS))
    player_denominator = player_mass or 1.0
    player_normalized = {
        region: float(region_scores.get(region, 0.0)) / player_denominator
        for region in PLAYERS
    }
    auth_focus = player_normalized[AUTH_KEY]
    attack_is_dominant = attack_dominant(
        player_normalized[ATTACK_KEY], player_normalized[AUTH_KEY], player_normalized[FACT_KEY]
    )
    return {
        "region_scores_raw": {region: float(region_scores.get(region, 0.0)) for region in (*PLAYERS, SPECIAL_KEY)},
        "region_scores_prompt_normalized": prompt_normalized,
        "region_scores_player_normalized": player_normalized,
        "prompt_attention_mass": prompt_mass,
        "player_attention_mass": player_mass,
        "special_attention_mass": float(region_scores.get(SPECIAL_KEY, 0.0)),
        "auth_focus_score": float(auth_focus),
        "threshold": float(threshold),
        "attention_shift": bool(auth_focus <= threshold),
        "attention_shift_basis": "auth_focus_score<=threshold_excluding_special",
        "attention_attack_dominant": bool(attack_is_dominant),
        "attention_attack_dominant_basis": "attack-margin-v1-on-player-normalized-regions",
        "attack_dominant_formula_version": "attack-margin-v1",
    }


def model_device(model) -> torch.device:
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "device"):
        return inner.device
    return next(inner.parameters()).device


def inner_model(model):
    return model.model if hasattr(model, "model") else model


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def supports_forward_arg(model, name: str) -> bool:
    try:
        return name in inspect.signature(inner_model(model).forward).parameters
    except (AttributeError, TypeError, ValueError):
        return False


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "cuda out of memory" in str(error).casefold()
    )


def read_target_id_file(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return {
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }


def compute_attention_streaming(
    model,
    rendered: RenderedTarget,
    spans: RegionSpans,
    top_k: int = 25,
    threshold: float = 0.5,
) -> dict:
    prompt_ids = list(rendered.prompt_ids)
    target_ids = list(rendered.target_ids)
    if not prompt_ids or not target_ids:
        raise ValueError("Attention requires non-empty prompt and target token sequences")
    heads = [tuple(head) for head in getattr(model, "important_heads", [])]
    if not heads:
        raise ValueError("Model has no configured important heads")
    prompt_len = len(prompt_ids)
    # Stream over target tokens to keep the peak attention tensor much smaller
    # for long AgentDojo trajectories. This preserves the original teacher-forced
    # query rule: prompt + target prefix predicts the next target token.
    token_scores = torch.zeros(prompt_len, dtype=torch.float32)
    non_prompt_masses = []
    past_key_values = None
    attention_mask_len = 0
    if prompt_len > 1:
        prefill_ids = torch.tensor([prompt_ids[:-1]], device=model_device(model), dtype=torch.long)
        prefill_mask = torch.ones_like(prefill_ids)
        with torch.inference_mode():
            prefill_output = inner_model(model)(
                input_ids=prefill_ids,
                attention_mask=prefill_mask,
                use_cache=True,
            )
        past_key_values = prefill_output.past_key_values
        attention_mask_len = prefill_ids.shape[1]
        del prefill_output, prefill_ids, prefill_mask
        clear_cuda_cache()
    for target_index in range(len(target_ids)):
        if target_index == 0:
            query_token_id = prompt_ids[-1]
        else:
            query_token_id = target_ids[target_index - 1]
        input_ids = torch.tensor([[query_token_id]], device=model_device(model), dtype=torch.long)
        attention_mask = torch.ones((1, attention_mask_len + 1), device=model_device(model), dtype=torch.long)
        with torch.inference_mode():
            output = inner_model(model)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=True,
                use_cache=True,
            )
        past_key_values = output.past_key_values
        attention_mask_len += 1
        per_target_scores = torch.zeros(prompt_len, dtype=torch.float32)
        per_target_non_prompt = 0.0
        for layer, head in heads:
            attention = output.attentions[layer][0, head, -1]
            per_target_scores += attention[:prompt_len].detach().float().cpu()
            per_target_non_prompt += float(attention[prompt_len:prompt_len + target_index].sum().item())
        token_scores += per_target_scores / len(heads)
        non_prompt_masses.append(per_target_non_prompt / len(heads))
        del output, input_ids, attention_mask, per_target_scores
        clear_cuda_cache()
    del past_key_values
    clear_cuda_cache()
    token_scores /= len(target_ids)
    region_scores = {region: 0.0 for region in (*PLAYERS, SPECIAL_KEY)}
    tokens = []
    input_tokens = model.tokenizer.convert_ids_to_tokens(prompt_ids)
    for index, score_tensor in enumerate(token_scores):
        score = float(score_tensor.item())
        region = region_for_token(index, spans)
        region_scores[region] += score
        tokens.append({"i": index, "t": input_tokens[index], "s": score, "r": region})
    metrics = attention_metrics(region_scores, threshold=threshold)
    return {
        **metrics,
        "non_prompt_attention_mass": float(sum(non_prompt_masses) / len(non_prompt_masses)),
        "num_input_tokens": len(prompt_ids),
        "num_target_tokens": len(target_ids),
        "token_ranges": {key: [list(span) for span in value] for key, value in spans.items()},
        "tokens": tokens,
        "top_tokens": sorted(tokens, key=lambda row: row["s"], reverse=True)[:top_k],
        "source": {
            "attn_step": "all_next_assistant_tokens_teacher_forced",
            "target_token_inclusion": "all_native_serialized_target_tokens",
            "aggregation": "mean_over_target_tokens_then_important_heads",
            "important_heads": [list(head) for head in heads],
            "query_index_rule": "prompt_len-1+target_index",
        },
    }


def compute_attention_full(
    model,
    rendered: RenderedTarget,
    spans: RegionSpans,
    top_k: int = 25,
    threshold: float = 0.5,
) -> dict:
    prompt_ids = list(rendered.prompt_ids)
    target_ids = list(rendered.target_ids)
    if not prompt_ids or not target_ids:
        raise ValueError("Attention requires non-empty prompt and target token sequences")
    forward_ids = prompt_ids + target_ids[:-1]
    input_ids = torch.tensor([forward_ids], device=model_device(model), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = inner_model(model)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
    heads = [tuple(head) for head in getattr(model, "important_heads", [])]
    if not heads:
        raise ValueError("Model has no configured important heads")
    prompt_len = len(prompt_ids)
    target_context_scores = []
    non_prompt_masses = []
    for target_index in range(len(target_ids)):
        query_index = prompt_len - 1 + target_index
        per_head = []
        per_head_non_prompt = []
        for layer, head in heads:
            attention = output.attentions[layer][0, head, query_index]
            per_head.append(attention[:prompt_len].detach().float().cpu())
            per_head_non_prompt.append(float(attention[prompt_len:query_index + 1].sum().item()))
        target_context_scores.append(torch.stack(per_head).mean(dim=0))
        non_prompt_masses.append(sum(per_head_non_prompt) / len(per_head_non_prompt))
    token_scores = torch.stack(target_context_scores).mean(dim=0)
    region_scores = {region: 0.0 for region in (*PLAYERS, SPECIAL_KEY)}
    tokens = []
    input_tokens = model.tokenizer.convert_ids_to_tokens(prompt_ids)
    for index, score_tensor in enumerate(token_scores):
        score = float(score_tensor.item())
        region = region_for_token(index, spans)
        region_scores[region] += score
        tokens.append({"i": index, "t": input_tokens[index], "s": score, "r": region})
    metrics = attention_metrics(region_scores, threshold=threshold)
    result = {
        **metrics,
        "non_prompt_attention_mass": float(sum(non_prompt_masses) / len(non_prompt_masses)),
        "num_input_tokens": len(prompt_ids),
        "num_target_tokens": len(target_ids),
        "token_ranges": {key: [list(span) for span in value] for key, value in spans.items()},
        "tokens": tokens,
        "top_tokens": sorted(tokens, key=lambda row: row["s"], reverse=True)[:top_k],
        "source": {
            "attn_step": "all_next_assistant_tokens_teacher_forced",
            "target_token_inclusion": "all_native_serialized_target_tokens",
            "aggregation": "mean_over_target_tokens_then_important_heads",
            "important_heads": [list(head) for head in heads],
            "query_index_rule": "prompt_len-1+target_index",
        },
    }
    del output, input_ids, attention_mask, target_context_scores, token_scores
    clear_cuda_cache()
    return result


def compute_attention(
    model,
    rendered: RenderedTarget,
    spans: RegionSpans,
    top_k: int = 25,
    threshold: float = 0.5,
    full_forward_token_threshold: int = 1024,
) -> dict:
    total_tokens = len(rendered.prompt_ids) + max(0, len(rendered.target_ids) - 1)
    if full_forward_token_threshold > 0 and total_tokens > full_forward_token_threshold:
        result = compute_attention_streaming(model, rendered, spans, top_k=top_k, threshold=threshold)
        result["source"]["memory_strategy"] = "kv_cache_preemptive_for_long_sequence"
        result["source"]["full_forward_token_threshold"] = int(full_forward_token_threshold)
        return result
    try:
        result = compute_attention_full(model, rendered, spans, top_k=top_k, threshold=threshold)
        result["source"]["memory_strategy"] = "full_forward"
        return result
    except Exception as error:
        if not is_cuda_oom(error):
            raise
        clear_cuda_cache()
        result = compute_attention_streaming(model, rendered, spans, top_k=top_k, threshold=threshold)
        result["source"]["memory_strategy"] = "kv_cache_fallback_after_cuda_oom"
        result["source"]["fallback_error"] = f"{type(error).__name__}: {error}"
        return result


def embedding_layer(model):
    inner = inner_model(model)
    if hasattr(inner, "get_input_embeddings"):
        return inner.get_input_embeddings()
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    raise AttributeError("Model does not expose get_input_embeddings")


def teacher_forced_target_inputs(masked_prompt_embeds: torch.Tensor, target_embeds: torch.Tensor) -> torch.Tensor:
    if masked_prompt_embeds.shape[1] == 0:
        raise ValueError("Shapley prompt must contain at least one token")
    if target_embeds.shape[1] == 0:
        raise ValueError("Shapley target must contain at least one token")
    return torch.cat([masked_prompt_embeds[:, -1:, :], target_embeds[:, :-1, :]], dim=1)


def mean_target_logprob_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    target_count = target_ids.shape[1]
    token_logits = logits[:, -target_count:, :]
    if token_logits.shape[1] != target_count:
        raise ValueError(
            "Shapley target-logit alignment failed: "
            f"expected {target_count} logits, got {token_logits.shape[1]}"
        )
    log_probs = F.log_softmax(token_logits.float(), dim=-1)
    gathered = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    result = float(gathered.mean().item())
    del token_logits, log_probs, gathered
    return result


def prefill_embeds_with_kv_cache(model, embeds: torch.Tensor, chunk_size: int) -> tuple[object | None, int]:
    if chunk_size <= 0:
        raise ValueError("Shapley prefill chunk size must be positive")
    past_key_values = None
    attention_mask_len = 0
    logits_to_keep = 1 if supports_forward_arg(model, "logits_to_keep") else None
    for start in range(0, embeds.shape[1], chunk_size):
        chunk = embeds[:, start:start + chunk_size, :]
        attention_mask = torch.ones(
            (1, attention_mask_len + chunk.shape[1]), device=embeds.device, dtype=torch.long
        )
        forward_kwargs = {
            "inputs_embeds": chunk,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
        if logits_to_keep is not None:
            forward_kwargs["logits_to_keep"] = logits_to_keep
        with torch.inference_mode():
            output = inner_model(model)(**forward_kwargs)
        past_key_values = output.past_key_values
        attention_mask_len += chunk.shape[1]
        del output, chunk, attention_mask
    return past_key_values, attention_mask_len


def mean_logprob_with_masked_regions_streaming(
    model,
    prompt_embeds: torch.Tensor,
    target_ids: torch.Tensor,
    spans: RegionSpans,
    masked_players: Iterable[str],
    chunk_size: int = 16,
    prefill_chunk_size: int = 256,
) -> float:
    if target_ids.shape[1] == 0:
        raise ValueError("Shapley target must contain at least one token")
    if chunk_size <= 0:
        raise ValueError("Shapley streaming chunk size must be positive")
    if prefill_chunk_size <= 0:
        raise ValueError("Shapley prefill chunk size must be positive")
    masked_embeds = prompt_embeds.clone()
    for player in masked_players:
        for start, end in spans.get(player, []):
            masked_embeds[:, start:end, :] = 0.0
    target_embeds = embedding_layer(model)(target_ids)
    past_key_values = None
    attention_mask_len = 0
    if masked_embeds.shape[1] > 1:
        prefill_embeds = masked_embeds[:, :-1, :]
        past_key_values, attention_mask_len = prefill_embeds_with_kv_cache(
            model, prefill_embeds, prefill_chunk_size
        )
        del prefill_embeds
    # Each input position predicts the target token at the same position.  Feeding
    # several positions at once preserves teacher forcing while avoiding one model
    # invocation and allocator synchronization per target token.
    teacher_forced_inputs = teacher_forced_target_inputs(masked_embeds, target_embeds)
    logprob_total = 0.0
    if supports_forward_arg(model, "logits_to_keep"):
        logits_to_keep = chunk_size
    else:
        logits_to_keep = None
    for target_start in range(0, target_ids.shape[1], chunk_size):
        current_embeds = teacher_forced_inputs[:, target_start:target_start + chunk_size, :]
        current_target_ids = target_ids[:, target_start:target_start + current_embeds.shape[1]]
        attention_mask = torch.ones(
            (1, attention_mask_len + current_embeds.shape[1]),
            device=current_embeds.device,
            dtype=torch.long,
        )
        forward_kwargs = {
            "inputs_embeds": current_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
        if logits_to_keep is not None:
            forward_kwargs["logits_to_keep"] = logits_to_keep
        with torch.inference_mode():
            output = inner_model(model)(**forward_kwargs)
        logits = output.logits
        block_mean_logprob = mean_target_logprob_from_logits(logits, current_target_ids)
        logprob_total += block_mean_logprob * current_target_ids.shape[1]
        past_key_values = output.past_key_values
        attention_mask_len += current_embeds.shape[1]
        del output, logits, current_embeds, current_target_ids, attention_mask
    result = logprob_total / target_ids.shape[1]
    del masked_embeds, target_embeds, teacher_forced_inputs, past_key_values
    clear_cuda_cache()
    return result


def mean_logprob_with_masked_regions_full(
    model,
    prompt_embeds: torch.Tensor,
    target_ids: torch.Tensor,
    spans: RegionSpans,
    masked_players: Iterable[str],
) -> float:
    if target_ids.shape[1] == 0:
        raise ValueError("Shapley target must contain at least one token")
    masked_embeds = prompt_embeds.clone()
    for player in masked_players:
        for start, end in spans.get(player, []):
            masked_embeds[:, start:end, :] = 0.0
    target_embeds = embedding_layer(model)(target_ids)
    target_inputs = teacher_forced_target_inputs(masked_embeds, target_embeds)
    full_embeds = torch.cat([masked_embeds[:, :-1, :], target_inputs], dim=1)
    attention_mask = torch.ones(full_embeds.shape[:2], device=full_embeds.device, dtype=torch.long)
    forward_kwargs = {
        "inputs_embeds": full_embeds,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    if supports_forward_arg(model, "logits_to_keep"):
        forward_kwargs["logits_to_keep"] = target_ids.shape[1]
    with torch.inference_mode():
        output = inner_model(model)(**forward_kwargs)
    result = mean_target_logprob_from_logits(output.logits, target_ids)
    del masked_embeds, target_embeds, target_inputs, full_embeds, attention_mask, output
    clear_cuda_cache()
    return result


def mean_logprob_with_masked_regions(
    model,
    prompt_embeds: torch.Tensor,
    target_ids: torch.Tensor,
    spans: RegionSpans,
    masked_players: Iterable[str],
    prefer_full_forward: bool = True,
    streaming_chunk_size: int = 16,
    prefill_chunk_size: int = 256,
) -> tuple[float, str, str | None]:
    if not prefer_full_forward:
        value = mean_logprob_with_masked_regions_streaming(
            model,
            prompt_embeds,
            target_ids,
            spans,
            masked_players,
            chunk_size=streaming_chunk_size,
            prefill_chunk_size=prefill_chunk_size,
        )
        return value, "kv_cache_chunked_preemptive_for_long_sequence", None
    try:
        value = mean_logprob_with_masked_regions_full(
            model, prompt_embeds, target_ids, spans, masked_players
        )
        return value, "full_forward", None
    except Exception as error:
        if not is_cuda_oom(error):
            raise
        clear_cuda_cache()
        value = mean_logprob_with_masked_regions_streaming(
            model,
            prompt_embeds,
            target_ids,
            spans,
            masked_players,
            chunk_size=streaming_chunk_size,
            prefill_chunk_size=prefill_chunk_size,
        )
        return value, "kv_cache_chunked_fallback_after_cuda_oom", f"{type(error).__name__}: {error}"


def compute_shapley(
    model,
    rendered: RenderedTarget,
    spans: RegionSpans,
    full_forward_token_threshold: int = 1024,
    streaming_chunk_size: int = 16,
    prefill_chunk_size: int = 256,
) -> dict:
    if streaming_chunk_size <= 0:
        raise ValueError("Shapley streaming chunk size must be positive")
    if prefill_chunk_size <= 0:
        raise ValueError("Shapley prefill chunk size must be positive")
    device = model_device(model)
    prompt_ids = torch.tensor([rendered.prompt_ids], device=device, dtype=torch.long)
    target_ids = torch.tensor([rendered.target_ids], device=device, dtype=torch.long)
    prompt_embeds = embedding_layer(model)(prompt_ids).detach()
    total_tokens = prompt_ids.shape[1] + target_ids.shape[1]
    prefer_full_forward = full_forward_token_threshold <= 0 or total_tokens <= full_forward_token_threshold
    values: dict[tuple[str, ...], float] = {}
    memory_strategies: dict[tuple[str, ...], str] = {}
    fallback_errors: dict[tuple[str, ...], str] = {}
    for size in range(len(PLAYERS) + 1):
        for coalition in itertools.combinations(PLAYERS, size):
            masked_players = [player for player in PLAYERS if player not in coalition]
            value, strategy, fallback_error = mean_logprob_with_masked_regions(
                model,
                prompt_embeds,
                target_ids,
                spans,
                masked_players,
                prefer_full_forward=prefer_full_forward,
                streaming_chunk_size=streaming_chunk_size,
                prefill_chunk_size=prefill_chunk_size,
            )
            values[coalition] = value
            memory_strategies[coalition] = strategy
            if fallback_error:
                fallback_errors[coalition] = fallback_error
    phi = compute_shapley_values(list(PLAYERS), values)
    full_coalition = tuple(PLAYERS)
    efficiency_error = sum(phi.values()) - (values[full_coalition] - values[()])
    del prompt_ids, target_ids, prompt_embeds
    clear_cuda_cache()
    return {
        "players": list(PLAYERS),
        "values": {"+".join(coalition) if coalition else "empty": value for coalition, value in values.items()},
        "phi": phi,
        "phi_auth": phi[AUTH_KEY],
        "phi_data_fact": phi[FACT_KEY],
        "phi_data_attack": phi[ATTACK_KEY],
        "target_token_count": len(rendered.target_ids),
        "efficiency_error": float(efficiency_error),
        **contribution_fields(phi[ATTACK_KEY], phi[AUTH_KEY], phi[FACT_KEY]),
        "shapley_attack_dominant": attack_dominant(phi[ATTACK_KEY], phi[AUTH_KEY], phi[FACT_KEY]),
        "source": {
            "value_function": "teacher_forced_mean_target_token_logprob",
            "value_function_version": "teacher-forced-mean-logprob-v2",
            "target_logit_alignment": "prompt-plus-target-prefix-predicts-full-target",
            "coalition_count": len(values),
            "masking": "zero_player_token_embeddings",
            "structure_preserved": ["token_count", "positions", "attention_mask", "fixed_context"],
            "target_scope": "next_assistant_message",
            "memory_strategy": "auto_full_forward_then_kv_cache_fallback",
            "full_forward_token_threshold": int(full_forward_token_threshold),
            "streaming_chunk_size": int(streaming_chunk_size),
            "prefill_chunk_size": int(prefill_chunk_size),
            "coalition_memory_strategies": {
                "+".join(coalition) if coalition else "empty": strategy
                for coalition, strategy in memory_strategies.items()
            },
            "fallback_errors": {
                "+".join(coalition) if coalition else "empty": error
                for coalition, error in fallback_errors.items()
            },
            "coalition_streaming_chunk_sizes": {
                "+".join(coalition) if coalition else "empty": (
                    int(streaming_chunk_size) if strategy.startswith("kv_cache_chunked") else None
                )
                for coalition, strategy in memory_strategies.items()
            },
        },
    }


def audit_corpus(input_root: str) -> tuple[list[SelectedCase], list[dict], dict]:
    paths = discover_input_paths(input_root)
    selected: list[SelectedCase] = []
    manifest: list[dict] = []
    for path in paths:
        trajectory = read_json(path)
        case, reason = select_case(path, trajectory)
        row = {
            "source_path": path,
            "case_id": case_id(trajectory),
            "eligible": case is not None,
            "reason": reason,
        }
        if case is not None:
            selected.append(case)
            row.update({
                "target_id": case.target_id,
                "polluted_tool_message_index": case.polluted_tool_index,
                "target_assistant_message_index": case.target_assistant_index,
                "assistant_immediately_adjacent": case.target_assistant_index == case.polluted_tool_index + 1,
                "attack_occurrence_count": len(case.attack_char_spans),
                "injection_metadata_match": case.injection_match,
            })
        manifest.append(row)
    reasons: dict[str, int] = {}
    for row in manifest:
        if row["reason"]:
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    summary = {
        "schema_version": 1,
        "input_root": input_root,
        "input_pattern": INPUT_PATTERN,
        "attack_type": ATTACK_TYPE,
        "candidate_count": len(paths),
        "eligible_count": len(selected),
        "excluded_count": len(paths) - len(selected),
        "exclusion_reasons": reasons,
        "unique_target_count": len({case.target_id for case in selected}),
        "injection_metadata_mismatch_count": sum(not case.injection_match for case in selected),
    }
    return selected, manifest, summary


def validate_canonical_counts(summary: dict) -> None:
    actual = (summary["candidate_count"], summary["eligible_count"])
    expected = (EXPECTED_CANDIDATES, EXPECTED_ELIGIBLE)
    if actual != expected:
        raise ValueError(f"Canonical corpus count mismatch: expected {expected}, got {actual}")
    if summary["unique_target_count"] != EXPECTED_ELIGIBLE:
        raise ValueError("Eligible trajectories did not produce unique target ids")


def player_texts(selected: SelectedCase) -> dict:
    user = str(original_user_message(selected.trajectory).get("content") or "")
    tool_content = str(selected.trajectory["messages"][selected.polluted_tool_index].get("content") or "")
    attacks = [tool_content[start:end] for start, end in selected.attack_char_spans]
    facts = [tool_content[start:end] for start, end in complement_intervals(len(tool_content), selected.attack_char_spans)]
    return {AUTH_KEY: user, FACT_KEY: facts, ATTACK_KEY: attacks}


def attribution_dedup_key(selected: SelectedCase) -> tuple[str, tuple[str, ...], tuple[str, ...], str]:
    players = player_texts(selected)
    target_message = selected.trajectory["messages"][selected.target_assistant_index]
    target_text = str(target_message.get("content") or "")
    return (
        players[AUTH_KEY],
        tuple(players[FACT_KEY]),
        tuple(players[ATTACK_KEY]),
        target_text,
    )


def fact_attack_lengths(selected: SelectedCase) -> tuple[int, int]:
    """Return character counts for FACT and ATTACK in the first polluted tool."""
    content = str(selected.trajectory["messages"][selected.polluted_tool_index].get("content") or "")
    attack_chars = sum(end - start for start, end in selected.attack_char_spans)
    return len(content) - attack_chars, attack_chars


def selected_target_kind(selected: SelectedCase) -> str:
    return assistant_target_kind(selected.trajectory["messages"][selected.target_assistant_index])


def result_base(selected: SelectedCase, rendered: RenderedTarget, spans: RegionSpans, validation: dict) -> dict:
    trajectory = selected.trajectory
    security = trajectory.get("security")
    return {
        "schema_version": 1,
        "target_id": selected.target_id,
        "case_id": case_id(trajectory),
        "source_path": selected.source_path,
        "suite_name": trajectory.get("suite_name"),
        "user_task_id": trajectory.get("user_task_id"),
        "injection_task_id": trajectory.get("injection_task_id"),
        "attack_type": trajectory.get("attack_type"),
        "polluted_tool_message_index": selected.polluted_tool_index,
        "target_assistant_message_index": selected.target_assistant_index,
        "target_selection": "next_assistant_after_first_polluted_tool",
        "fact_source": "first_polluted_tool_content_excluding_attack",
        "target_scope": "next_assistant_message",
        "target_kind": rendered.target_kind,
        "target_text": rendered.target_text,
        "target_message": trajectory["messages"][selected.target_assistant_index],
        "player_text": player_texts(selected),
        "token_ranges": {key: [list(span) for span in value] for key, value in spans.items()},
        "span_validation": validation,
        "security": security,
        "utility": trajectory.get("utility"),
        "attack_success": (not security) if isinstance(security, bool) else None,
        "attack_success_source": "logical_not_of_agentdojo_security",
        "valid_for_stats": False,
    }


def summarize_numbers(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {"count": len(values), "mean": float(statistics.fmean(values)), "median": float(statistics.median(values))}


def group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped["|".join(f"{key}={row.get(key)}" for key in keys)].append(row)
    result = {}
    for name, subset in sorted(grouped.items()):
        result[name] = {
            "count": len(subset),
            "auth_focus_score": summarize_numbers([row["auth_focus_score"] for row in subset if row.get("auth_focus_score") is not None]),
            "phi_auth": summarize_numbers([row["phi_auth"] for row in subset if row.get("phi_auth") is not None]),
            "phi_data_fact": summarize_numbers([row["phi_data_fact"] for row in subset if row.get("phi_data_fact") is not None]),
            "phi_data_attack": summarize_numbers([row["phi_data_attack"] for row in subset if row.get("phi_data_attack") is not None]),
        }
    return result


def summarize_outputs(audit: dict, attention_rows: list[dict], shapley_rows: list[dict]) -> dict:
    valid_attention = [row for row in attention_rows if row.get("valid_for_stats")]
    valid_shapley = [row for row in shapley_rows if row.get("valid_for_stats")]
    all_rows = attention_rows or shapley_rows
    return {
        "schema_version": 1,
        "target_scope": "next_assistant_message",
        "audit": audit,
        "counts": {
            "attempted_targets": len({row.get("target_id") for row in all_rows}),
            "attention_successful": len(valid_attention),
            "attention_failed": len(attention_rows) - len(valid_attention),
            "shapley_successful": len(valid_shapley),
            "shapley_failed": len(shapley_rows) - len(valid_shapley),
            "suite": dict(Counter(row.get("suite_name") for row in all_rows)),
            "security": dict(Counter(str(row.get("security")) for row in all_rows)),
            "target_kind": dict(Counter(row.get("target_kind") for row in all_rows)),
        },
        "attention_groups": group_rows(valid_attention, ("target_scope", "suite_name", "security", "target_kind")),
        "shapley_groups": group_rows(valid_shapley, ("target_scope", "suite_name", "security", "target_kind")),
    }


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def run_attribution(args: argparse.Namespace, selected: list[SelectedCase], audit_summary: dict) -> None:
    from baselines.attention_tracker.utils import create_model, open_config
    from visualize_agentdojo_attribution import merge_rows, render_html

    attention_path = os.path.join(args.output_dir, "results.attention.jsonl")
    shapley_path = os.path.join(args.output_dir, "results.shapley.jsonl")
    summary_path = os.path.join(args.output_dir, "results.summary.json")

    cases = selected
    if args.action_targets_only:
        cases = [case for case in cases if selected_target_kind(case) in {"tool_calls", "mixed"}]
    if args.fact_dominant:
        filtered = []
        for case in cases:
            fact_chars, attack_chars = fact_attack_lengths(case)
            if fact_chars < args.min_fact_chars:
                continue
            if fact_chars > args.max_fact_chars:
                continue
            if fact_chars < args.fact_ratio * max(1, attack_chars):
                continue
            filtered.append(case)
        cases = filtered
        if args.rank_by_fact_ratio:
            cases = sorted(cases, key=lambda case: (
                fact_attack_lengths(case)[0] / max(1, fact_attack_lengths(case)[1]),
                fact_attack_lengths(case)[0],
            ), reverse=True)
    if args.target_id:
        cases = [case for case in cases if case.target_id == args.target_id]
        if not cases:
            raise ValueError(f"Unknown target id: {args.target_id}")
    if args.target_id_file:
        target_ids = read_target_id_file(args.target_id_file)
        if not target_ids:
            raise ValueError(f"Target id file is empty: {args.target_id_file}")
        cases = [case for case in cases if case.target_id in target_ids]
        if not cases:
            raise ValueError(f"No selected cases matched target ids from: {args.target_id_file}")
    if args.limit is not None:
        cases = cases[:args.limit]
    unique_cases = []
    seen_dedup_keys = set()
    duplicate_count = 0
    for case in cases:
        dedup_key = attribution_dedup_key(case)
        if dedup_key in seen_dedup_keys:
            duplicate_count += 1
            continue
        seen_dedup_keys.add(dedup_key)
        unique_cases.append(case)
    cases = unique_cases
    audit_summary = dict(audit_summary)
    audit_summary["deduplication"] = {
        "selected_count": len(cases) + duplicate_count,
        "deduplicated_count": len(cases),
        "duplicate_count": duplicate_count,
    }
    if args.use_cache:
        attention_rows = [] if args.skip_attention else read_jsonl(attention_path)
        shapley_rows = [] if args.skip_shapley else read_jsonl(shapley_path)
        attention_cache = rows_by_target_id(attention_rows)
        shapley_cache = rows_by_target_id(shapley_rows)
    else:
        attention_rows = []
        shapley_rows = []
        attention_cache = {}
        shapley_cache = {}

    def has_cached_attention(target_id_value: str) -> bool:
        return args.skip_attention or (target_id_value in attention_cache and attention_cache[target_id_value].get("valid_for_stats") is True)

    def has_cached_shapley(target_id_value: str) -> bool:
        return args.skip_shapley or (target_id_value in shapley_cache and shapley_cache[target_id_value].get("valid_for_stats") is True)

    remaining_cases = [
        case for case in cases
        if not args.use_cache or not (has_cached_attention(case.target_id) and has_cached_shapley(case.target_id))
    ]

    model = create_model(open_config(args.model_config)) if remaining_cases else None
    for selected_case in tqdm(remaining_cases, desc="AgentDojo attribution", unit="target"):
        needs_attention = not has_cached_attention(selected_case.target_id)
        needs_shapley = not has_cached_shapley(selected_case.target_id)
        if not needs_attention and not needs_shapley:
            continue
        try:
            rendered = render_selected_target(model.tokenizer, selected_case)
            spans = extract_region_spans(model.tokenizer, selected_case, rendered)
            validation = validate_region_spans(rendered, spans)
            if not validation["ok"]:
                raise ValueError(f"Invalid player spans: {validation['issues']}")
            base = result_base(selected_case, rendered, spans, validation)
            base["attention_time_seconds"] = 0.0
            base["shapley_time_seconds"] = 0.0
        except Exception as error:
            failure = {
                "schema_version": 1,
                "target_id": selected_case.target_id,
                "case_id": case_id(selected_case.trajectory),
                "source_path": selected_case.source_path,
                "target_scope": "next_assistant_message",
                "security": selected_case.trajectory.get("security"),
                "utility": selected_case.trajectory.get("utility"),
                "valid_for_stats": False,
                "attention_time_seconds": 0.0,
                "shapley_time_seconds": 0.0,
                "error": f"{type(error).__name__}: {error}",
            }
            if needs_attention:
                if args.use_cache:
                    attention_rows[:] = [cached for cached in attention_rows if cached.get("target_id") != selected_case.target_id]
                attention_rows.append(dict(failure))
                attention_cache[selected_case.target_id] = attention_rows[-1]
                if args.use_cache:
                    append_jsonl(attention_path, attention_rows[-1])
            if needs_shapley:
                if args.use_cache:
                    shapley_rows[:] = [cached for cached in shapley_rows if cached.get("target_id") != selected_case.target_id]
                shapley_rows.append(dict(failure))
                shapley_cache[selected_case.target_id] = shapley_rows[-1]
                if args.use_cache:
                    append_jsonl(shapley_path, shapley_rows[-1])
            if args.use_cache:
                write_json(summary_path, summarize_outputs(audit_summary, attention_rows, shapley_rows))
            clear_cuda_cache()
            continue
        if needs_attention:
            row = dict(base)
            attention_started = time.perf_counter()
            try:
                row.update(compute_attention(
                    model,
                    rendered,
                    spans,
                    top_k=args.attention_top_k,
                    full_forward_token_threshold=args.full_forward_token_threshold,
                ))
                row["valid_for_stats"] = True
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["attention_time_seconds"] = float(time.perf_counter() - attention_started)
            if args.use_cache:
                attention_rows[:] = [cached for cached in attention_rows if cached.get("target_id") != selected_case.target_id]
            attention_rows.append(row)
            attention_cache[selected_case.target_id] = row
            if args.use_cache:
                append_jsonl(attention_path, row)
        if needs_shapley:
            row = dict(base)
            shapley_started = time.perf_counter()
            try:
                row.update(compute_shapley(
                    model,
                    rendered,
                    spans,
                    full_forward_token_threshold=args.full_forward_token_threshold,
                    streaming_chunk_size=args.shapley_streaming_chunk_size,
                    prefill_chunk_size=args.shapley_prefill_chunk_size,
                ))
                row["valid_for_stats"] = True
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["shapley_time_seconds"] = float(time.perf_counter() - shapley_started)
            if args.use_cache:
                shapley_rows[:] = [cached for cached in shapley_rows if cached.get("target_id") != selected_case.target_id]
            shapley_rows.append(row)
            shapley_cache[selected_case.target_id] = row
            if args.use_cache:
                append_jsonl(shapley_path, row)
        del rendered, spans, validation, base
        if args.use_cache:
            write_json(summary_path, summarize_outputs(audit_summary, attention_rows, shapley_rows))
        clear_cuda_cache()

    if not args.use_cache and not args.skip_attention:
        write_jsonl(attention_path, attention_rows)
    if not args.use_cache and not args.skip_shapley:
        write_jsonl(shapley_path, shapley_rows)
    gallery_rows = merge_rows(shapley_rows, attention_rows)
    gallery_source = {
        "shapley": None if args.skip_shapley else shapley_path,
        "attention": None if args.skip_attention else attention_path,
    }
    write_text(
        os.path.join(args.output_dir, "attribution_gallery.html"),
        render_html(gallery_rows, gallery_source, "AgentDojo Next-Assistant Attribution"),
    )
    write_json(summary_path, summarize_outputs(audit_summary, attention_rows, shapley_rows))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgentDojo next-assistant attribution.")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--allow-corpus-drift", action="store_true")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fact-dominant", action="store_true", help="Only run cases where FACT is substantially longer than ATTACK.")
    parser.add_argument("--action-targets-only", action="store_true", help="Only run tool_calls or mixed next-assistant targets.")
    parser.add_argument("--min-fact-chars", type=int, default=500)
    parser.add_argument("--max-fact-chars", type=int, default=100000)
    parser.add_argument("--fact-ratio", type=float, default=1.0)
    parser.add_argument("--rank-by-fact-ratio", action="store_true")
    parser.add_argument("--target-id")
    parser.add_argument("--target-id-file", help="Run only target ids listed one per line; blank lines and # comments are ignored.")
    parser.add_argument("--use-cache", action="store_true", help="Resume from existing JSONL rows and append each completed target immediately.")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--skip-shapley", action="store_true")
    parser.add_argument("--attention-top-k", type=int, default=25)
    parser.add_argument(
        "--shapley-streaming-chunk-size",
        type=positive_int,
        default=16,
        help="Target tokens evaluated per KV-cache Shapley forward; must be positive.",
    )
    parser.add_argument(
        "--shapley-prefill-chunk-size",
        type=positive_int,
        default=256,
        help="Prompt tokens evaluated per KV-cache Shapley prefill forward; must be positive.",
    )
    parser.add_argument(
        "--full-forward-token-threshold",
        type=int,
        default=1024,
        help="Use full forward only when prompt+target tokens are at or below this threshold; use <=0 to always try full forward before OOM fallback.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    selected, manifest, summary = audit_corpus(args.input_root)
    if not args.allow_corpus_drift:
        validate_canonical_counts(summary)
    write_jsonl(os.path.join(args.output_dir, "audit.jsonl"), manifest)
    write_json(os.path.join(args.output_dir, "audit.summary.json"), summary)
    if not args.audit_only:
        if args.skip_attention and args.skip_shapley:
            raise ValueError("At least one attribution method must be enabled")
        run_attribution(args, selected, summary)


if __name__ == "__main__":
    main(parse_args())
