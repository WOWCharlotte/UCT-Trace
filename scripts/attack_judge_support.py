"""Shared data contracts and attribution-cascade helpers for attack judging."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable


FORMULA_VERSION = "attack-margin-v1"
FORMULA = "ATTACK - (max(0, AUTH) + max(0, FACT)) > 0"


def attack_margin(attack: Any, auth: Any, fact: Any) -> float:
    """Return the versioned ATTACK contribution margin."""
    return float(attack or 0.0) - (max(0.0, float(auth or 0.0)) + max(0.0, float(fact or 0.0)))


def attack_dominant(attack: Any, auth: Any, fact: Any) -> bool:
    return attack_margin(attack, auth, fact) > 0.0


def contribution_fields(attack: Any, auth: Any, fact: Any) -> dict[str, Any]:
    margin = attack_margin(attack, auth, fact)
    return {
        "attack_margin": margin,
        "attack_dominant": margin > 0.0,
        "attack_dominant_formula": FORMULA,
        "attack_dominant_formula_version": FORMULA_VERSION,
        "clamped_auth": max(0.0, float(auth or 0.0)),
        "clamped_fact": max(0.0, float(fact or 0.0)),
    }


@dataclass(frozen=True)
class JudgeInput:
    """Dataset-neutral context presented to a behavior judge."""

    record_id: str
    dataset: str
    user_task: str
    tool_response: str
    attack_text: str
    assistant_output: str
    target_kind: str = "text"
    tool_definitions: str = ""
    target_tool_name: str | None = None
    target_tool_parameters: str | None = None
    source_path: str | None = None
    source_metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_input_hash(record: JudgeInput | dict[str, Any]) -> str:
    payload = record.as_dict() if isinstance(record, JudgeInput) else record
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_key(record: JudgeInput | dict[str, Any], model: str, prompt_version: str) -> str:
    return f"{stable_input_hash(record)}:{model}:{prompt_version}"


def _first_value(row: dict[str, Any], names: Iterable[str], default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def injecagent_input(row: dict[str, Any], case_id: str | int | None = None, tool_schema: str = "") -> JudgeInput:
    """Adapt an InjecAgent action row without changing the source row."""
    identifier = str(_first_value(row, ("record_id", "case_id"), case_id if case_id is not None else ""))
    action_tool = _first_value(row, ("action_tool", "tool_name"), None)
    action_parameters = _first_value(row, ("action_input", "action_parameters"), None)
    return JudgeInput(
        record_id=identifier,
        dataset="injecagent",
        user_task=str(_first_value(row, ("User Instruction", "user_instruction", "user_task"))),
        tool_response=str(_first_value(row, ("Tool Response", "tool_response", "data_fact_response"))),
        attack_text=str(_first_value(row, ("Attacker Instruction", "attack_text", "data_attack"))),
        assistant_output=str(_first_value(row, ("assistant_output", "model_output", "action_text", "output"))),
        target_kind=str(_first_value(row, ("target_kind", "action_kind"), "tool" if action_tool else "text")),
        tool_definitions=tool_schema or str(_first_value(row, ("tool_schema_text", "tool_definitions"))),
        target_tool_name=str(action_tool) if action_tool is not None else None,
        target_tool_parameters=str(action_parameters) if action_parameters is not None else None,
        source_metadata={"eval": row.get("eval"), "security": row.get("security"), "utility": row.get("utility")},
    )


def agentdojo_input(
    row: dict[str, Any], trajectory: dict[str, Any], polluted_tool_index: int | None = None,
    assistant_index: int | None = None,
) -> JudgeInput:
    """Adapt a selected AgentDojo trajectory to the shared judge context."""
    messages = trajectory.get("messages", [])
    tool_index = polluted_tool_index if polluted_tool_index is not None else row.get("polluted_tool_message_index")
    target_index = assistant_index if assistant_index is not None else row.get("target_assistant_message_index")
    tool_message = messages[int(tool_index)] if tool_index is not None else {}
    assistant_message = messages[int(target_index)] if target_index is not None else {}
    user_message = next((message for message in messages if message.get("role") == "user"), {})
    tool_calls = assistant_message.get("tool_calls") or []
    first_call = tool_calls[0] if tool_calls else {}
    function = first_call.get("function") or {}
    return JudgeInput(
        record_id=str(row.get("target_id") or row.get("case_id") or ""),
        dataset="agentdojo",
        user_task=str(user_message.get("content") or ""),
        tool_response=str(tool_message.get("content") or ""),
        attack_text=str(row.get("attack_text") or row.get("data_attack") or (row.get("player_text") or {}).get("data_attack") or ""),
        assistant_output=str(assistant_message.get("content") or ""),
        target_kind=str(row.get("target_kind") or ("tool" if tool_calls else "text")),
        tool_definitions=json.dumps(row.get("tool_definitions") or [], ensure_ascii=False, sort_keys=True),
        target_tool_name=function.get("name"),
        target_tool_parameters=function.get("arguments"),
        source_path=row.get("source_path"),
        source_metadata={"security": trajectory.get("security"), "utility": trajectory.get("utility"), "eval": row.get("eval")},
    )


def select_candidates(rows: list[dict[str, Any]], mode: str = "shapley") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return candidates and non-candidates, annotating both audit paths."""
    if mode not in {"shapley", "attention", "both"}:
        raise ValueError("mode must be shapley, attention, or both")
    selected, skipped = [], []
    for row in rows:
        shapley = attack_dominant(row.get("phi_data_attack"), row.get("phi_auth"), row.get("phi_data_fact"))
        attention_scores = row.get("attention_region_scores") or row.get("region_scores") or {}
        attention = attack_dominant(attention_scores.get("data_attack"), attention_scores.get("auth"), attention_scores.get("data_fact"))
        trigger = shapley if mode == "shapley" else attention if mode == "attention" else shapley and attention
        annotated = dict(row)
        annotated.update({"shapley_attack_dominant": shapley, "attention_attack_dominant": attention,
                          "attack_trigger_rule": mode, "attack_attribution_trigger": bool(trigger),
                          "attack_dominant_formula": FORMULA, "attack_dominant_formula_version": FORMULA_VERSION})
        (selected if trigger else skipped).append(annotated)
    return selected, skipped


def audit_counts(total: int, candidates: int, judged: int = 0) -> dict[str, Any]:
    return {"total_count": total, "candidate_count": candidates, "judged_count": judged,
            "not_judged_count": total - candidates, "selection_rate": candidates / total if total else 0.0}
