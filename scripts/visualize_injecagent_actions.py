"""Render InjecAgent action-level Shapley/attention results as HTML."""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import os
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from attack_judge_support import attack_margin as compute_attack_margin


REGION_LABELS = {
    "auth": "AUTH",
    "data_fact": "FACT",
    "data_attack": "ATTACK",
    "special": "SPECIAL",
}

REGION_CLASSES = {
    "auth": "region-auth",
    "data_fact": "region-fact",
    "data_attack": "region-attack",
    "special": "region-special",
}

DEFAULT_CASES = "data/injecagent/qwen3-8b/test_cases_dh_base.jsonl"


def read_jsonl(path: str | None) -> list[dict]:
    if not path:
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def case_index(rows: list[dict]) -> dict[str, dict]:
    return {str(index): row for index, row in enumerate(rows)}


def select_record(rows: list[dict], case_id: str | None, index: int) -> dict:
    if case_id is not None:
        for row in rows:
            if str(row.get("case_id")) == str(case_id):
                return row
        raise ValueError(f"No InjecAgent action record found with case_id={case_id!r}.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"--index {index} is out of range for {len(rows)} records.")
    return rows[index]


def attention_index(rows: list[dict]) -> dict[str, dict]:
    return {str(row.get("case_id")): row for row in rows}


def judge_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(str(row.get("case_id")), str(row.get("target_scope", "full_action"))): row for row in rows}


def merge_attention(row: dict, attn_by_case: dict[str, dict]) -> dict:
    row = dict(row)
    attention = attn_by_case.get(str(row.get("case_id")))
    if attention:
        row["attention"] = attention
        row.setdefault("attention_shift", attention.get("attention_shift", attention.get("attention_shift_attack")))
        row.setdefault("attention_shift_attack", attention.get("attention_shift_attack", attention.get("attention_shift")))
        row.setdefault("attention_attack_dominant", attention.get("attention_attack_dominant"))
        row.setdefault("attention_region_scores", attention.get("region_scores"))
    return row


def merge_judge(row: dict, judges: dict[tuple[str, str], dict]) -> dict:
    merged = dict(row)
    judge = judges.get((str(row.get("case_id")), str(row.get("target_scope", "full_action"))))
    if judge is None:
        merged["judge"] = {"status": "judge_unavailable"}
        merged["attack_action_executed"] = None
        merged["attack_success_strict"] = False
    else:
        action_executed = judge.get("attack_action_executed")
        attribution_trigger = judge.get("attack_attribution_trigger", row.get("shapley_attack_dominant", False))
        merged.update({"judge": judge.get("judge") or judge, "attack_action_executed": action_executed,
                       "attack_attribution_trigger": bool(attribution_trigger),
                       "attack_success_strict": bool(attribution_trigger and action_executed is True)})
    return merged


def should_expand_record(row: dict) -> bool:
    return str(row.get("eval", "")).lower() == "succ" or bool(row.get("attention_shift", row.get("attention_shift_attack")))


def normalize_tool_response(value: str) -> str:
    text = str(value)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text.strip('"')
    return text


def mask_attack_in_response(tool_response: str, attack_text: str) -> str:
    response = normalize_tool_response(tool_response)
    if attack_text and attack_text in response:
        return response.replace(attack_text, "<DATA_ATTACK>")
    return response


def enrich_with_case(row: dict, cases_by_id: dict[str, dict]) -> dict:
    row = dict(row)
    case = cases_by_id.get(str(row.get("case_id")))
    if not case:
        return row
    attack = case.get("Attacker Instruction", "")
    tool_response = case.get("Tool Response", "")
    row["player_text"] = {
        "auth": (
            "System/ReAct policy and official tool schemas are fixed in the prompt.\n\n"
            f"User Instruction:\n{case.get('User Instruction', '')}"
        ),
        "data_fact": mask_attack_in_response(tool_response, attack),
        "data_attack": attack,
    }
    row["case"] = {
        "user_instruction": case.get("User Instruction", ""),
        "tool_parameters": case.get("Tool Parameters", ""),
        "thought": case.get("Thought", ""),
        "tool_response": normalize_tool_response(tool_response),
    }
    return row


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if math.isnan(numeric) or math.isinf(numeric):
        return str(numeric)
    return f"{numeric:.{digits}f}"


def bool_label(value) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return "UNKNOWN"


def status_class(value) -> str:
    text = str(value).lower()
    if text in {"succ", "success", "true", "yes"}:
        return "status-bad"
    if text in {"unsucc", "failed", "false", "no"}:
        return "status-good"
    if text in {"invalid", "unknown", "none", "n/a"}:
        return "status-warn"
    return "status-neutral"


def phi_payload(row: dict) -> dict:
    return {
        "players": row.get("players") or ["auth", "data_fact", "data_attack"],
        "phi": {
            "auth": row.get("phi_auth"),
            "data_fact": row.get("phi_data_fact"),
            "data_attack": row.get("phi_data_attack"),
        },
        "values": row.get("values") or {},
    }


def attack_margin(row: dict) -> float | None:
    try:
        return compute_attack_margin(row["phi_data_attack"], row["phi_auth"], row["phi_data_fact"])
    except (KeyError, TypeError, ValueError):
        return None


def value_bound(values: list[float]) -> float:
    finite = [abs(v) for v in values if not math.isnan(v) and not math.isinf(v)]
    return max(finite, default=1.0) or 1.0


def render_phi_cards(row: dict) -> str:
    payload = phi_payload(row)
    phi = payload["phi"]
    numeric = []
    for value in phi.values():
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            pass
    bound = value_bound(numeric)
    cards = []
    for player in payload["players"]:
        value = phi.get(player)
        if value is None:
            continue
        value = float(value)
        width = min(abs(value) / bound * 100.0, 100.0)
        fill = "fill-pos" if value >= 0 else "fill-neg"
        region_cls = REGION_CLASSES.get(player, "region-special")
        cards.append(
            f"""
            <div class="phi-card">
              <div class="card-head">
                <span class="pill {region_cls}">{REGION_LABELS.get(player, player.upper())}</span>
                <strong>{fmt(value)}</strong>
              </div>
              <div class="bar"><span class="{fill}" style="width:{width:.2f}%"></span></div>
            </div>
            """
        )
    return "\n".join(cards)


def render_player_text(row: dict) -> str:
    player_text = row.get("player_text") or {}
    blocks = []
    for player in ("auth", "data_fact", "data_attack"):
        text = player_text.get(player, "")
        region_cls = REGION_CLASSES[player]
        blocks.append(
            f"""
            <section class="player-block {region_cls}">
              <div class="block-head"><span class="pill {region_cls}">{REGION_LABELS[player]}</span></div>
              <pre>{html.escape(str(text)) if text else '<span class="empty">(missing)</span>'}</pre>
            </section>
            """
        )
    return f"""
    <section class="panel">
      <h2>Players</h2>
      <div class="player-grid">{''.join(blocks)}</div>
    </section>
    """


def render_attention(row: dict) -> str:
    attention = row.get("attention") or {}
    scores = attention.get("region_scores") or row.get("attention_region_scores") or {}
    if not scores:
        return '<section class="panel"><h2>Attention</h2><p class="empty">No attention record found.</p></section>'
    prompt_mass = attention.get("prompt_attention_mass")
    if prompt_mass is None:
        prompt_mass = sum(float(v) for v in scores.values())
    prompt_mass = float(prompt_mass)
    normalized = attention.get("region_scores_normalized") or {
        region: float(value) / (prompt_mass or 1.0)
        for region, value in scores.items()
    }
    player_mass = attention.get("player_attention_mass")
    if player_mass is None:
        player_mass = sum(float(scores.get(region, 0.0)) for region in ("auth", "data_fact", "data_attack"))
    player_mass = float(player_mass)
    player_normalized = attention.get("region_scores_player_normalized") or {
        region: float(scores.get(region, 0.0)) / (player_mass or 1.0)
        for region in ("auth", "data_fact", "data_attack")
    }
    player_rows = []
    for region in ("auth", "data_fact", "data_attack"):
        raw_value = float(scores.get(region, 0.0))
        player_value = float(player_normalized.get(region, 0.0))
        width = min(player_value * 100.0, 100.0)
        player_rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">
                <span class="pill {REGION_CLASSES.get(region)}">{REGION_LABELS.get(region)}</span>
                <span class="score-pair">player norm {fmt(player_value)} · raw {fmt(raw_value)}</span>
              </div>
              <div class="bar"><span class="fill-attn" style="width:{width:.2f}%"></span></div>
            </div>
            """
        )
    prompt_rows = []
    for region in ("auth", "data_fact", "data_attack", "special"):
        value = float(scores.get(region, 0.0))
        normalized_value = float(normalized.get(region, 0.0))
        width = min(normalized_value * 100.0, 100.0)
        prompt_rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">
                <span class="pill {REGION_CLASSES.get(region)}">{REGION_LABELS.get(region)}</span>
                <span class="score-pair">prompt norm {fmt(normalized_value)} · raw {fmt(value)}</span>
              </div>
              <div class="bar"><span class="fill-attn" style="width:{width:.2f}%"></span></div>
            </div>
            """
        )
    tokens = attention.get("tokens") or attention.get("top_tokens") or []
    token_rows = []
    for token in tokens:
        region = str(token.get("r", "special"))
        token_rows.append(
            f"""
            <span class="token-chip token-{html.escape(region)}" data-region="{html.escape(region)}" title="i={html.escape(str(token.get('i', '')))} | {html.escape(REGION_LABELS.get(region, region.upper()))} | attention={fmt(token.get('s'))}">
              {html.escape(str(token.get("t", "")))}
            </span>
            """
        )
    tokens_html = ""
    if token_rows:
        tokens_html = f"""
        <div class="token-toolbar">
          <strong>Tokens</strong>
          <label><input type="checkbox" data-filter="auth" checked> AUTH</label>
          <label><input type="checkbox" data-filter="data_fact" checked> FACT</label>
          <label><input type="checkbox" data-filter="data_attack" checked> ATTACK</label>
          <label><input type="checkbox" data-filter="special" checked> SPECIAL</label>
        </div>
        <div class="token-cloud">
          {''.join(token_rows)}
        </div>
        """
    return f"""
    <section class="panel">
      <div class="section-head">
        <h2>Attention</h2>
        <span class="chip">Shift: {bool_label(row.get("attention_shift", row.get("attention_shift_attack")))}</span>
      </div>
      <div class="attention-score-grid">
        <div class="attention-score"><span>Auth Focus</span><strong>{fmt(attention.get("auth_focus_score", player_normalized.get("auth")))}</strong></div>
        <div class="attention-score"><span>Threshold</span><strong>{fmt(attention.get("threshold"))}</strong></div>
        <div class="attention-score"><span>Attack Dominant</span><strong>{bool_label(attention.get("attention_attack_dominant"))}</strong></div>
      </div>
      <h3 class="mini-heading">Player-Normalized Attention</h3>
      {''.join(player_rows)}
      <details class="attention-details">
        <summary>Prompt-Normalized Attention</summary>
        <div class="attention-details-body">
          {''.join(prompt_rows)}
        </div>
      </details>
      {tokens_html}
    </section>
    """


def render_status(row: dict) -> str:
    return f"""
    <div class="status-grid">
      <div class="status-card">
        <span>Eval</span>
        <strong class="{status_class(row.get("eval"))}">{html.escape(str(row.get("eval", "unknown")).upper())}</strong>
      </div>
      <div class="status-card">
        <span>Action</span>
        <strong>{html.escape(str(row.get("action_kind", "unknown")))}</strong>
      </div>
      <div class="status-card">
        <span>Shapley Attack Dominant</span>
        <strong class="{status_class(row.get("shapley_attack_dominant"))}">{bool_label(row.get("shapley_attack_dominant"))}</strong>
      </div>
      <div class="status-card"><span>Experiment Joint Method</span><strong class="{status_class(row.get("attack_success_strict"))}">{bool_label(row.get("attack_success_strict"))}</strong></div>
      <div class="status-card"><span>Judge Behavior</span><strong>{html.escape(str((row.get("judge") or {}).get("behavior_label") or (row.get("judge") or {}).get("status", "unknown")))}</strong></div>
      <div class="status-card"><span>Shapley Seconds</span><strong>{fmt(row.get("shapley_time_seconds"))}</strong></div>
      <div class="status-card"><span>Attention Seconds</span><strong>{fmt(row.get("attention_time_seconds"))}</strong></div>
    </div>
    """


def render_prompt_metadata(row: dict) -> str:
    fields = [
        ("Attack Type", row.get("attack_type")),
        ("User Tool", row.get("user_tool")),
        ("Attacker Tools", ", ".join(row.get("attacker_tools") or [])),
        ("Action Tool", row.get("action_tool")),
        ("Token Count", row.get("action_token_count")),
        ("Efficiency Error", fmt(row.get("efficiency_error"))),
    ]
    items = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value)) if value is not None else 'N/A'}</td></tr>"
        for label, value in fields
    )
    return f"""
    <section class="panel">
      <h2>Metadata</h2>
      <div class="table-wrap compact"><table><tbody>{items}</tbody></table></div>
    </section>
    """


def render_action(row: dict) -> str:
    action = row.get("action_text") or row.get("parse_error") or ""
    return f"""
    <section class="panel action-panel">
      <h2>Explained Action</h2>
      <pre>{html.escape(str(action))}</pre>
    </section>
    """


def render_record(row: dict) -> str:
    open_attr = " open" if should_expand_record(row) else ""
    return f"""
    <details class="record" id="case-{html.escape(str(row.get("case_id")))}" data-eval="{html.escape(str(row.get('eval', 'unknown')))}" data-action="{html.escape(str(row.get('action_kind', 'unknown')))}" data-shapley="{str(bool(row.get('shapley_attack_dominant'))).lower()}" data-joint="{str(bool(row.get('attack_success_strict'))).lower()}"{open_attr}>
      <summary>
        <span class="case-title">CASE {html.escape(str(row.get("case_id")))} · {html.escape(str(row.get("attack_type", "InjecAgent Action")))}</span>
        <span class="case-meta">{html.escape(str(row.get("eval", "unknown")).upper())} · {html.escape(str(row.get("action_kind", "unknown")))}</span>
      </summary>
      <div class="record-body">
        {render_status(row)}
        <div class="layout">
          <div class="stack">
            {render_prompt_metadata(row)}
            {render_player_text(row)}
            {render_action(row)}
          </div>
          <div class="stack">
            <section class="panel">
              <div class="section-head">
                <h2>Shapley</h2>
                <span class="chip">player contributions</span>
              </div>
              <div class="phi-grid">{render_phi_cards(row)}</div>
            </section>
            {render_attention(row)}
          </div>
        </div>
      </div>
    </details>
    """


def summary_counts(rows: list[dict]) -> dict:
    return {
        "records": len(rows),
        "eval": dict(Counter(str(row.get("eval")) for row in rows)),
        "action_kind": dict(Counter(str(row.get("action_kind")) for row in rows)),
        "valid_for_stats": sum(bool(row.get("valid_for_stats")) for row in rows),
        "shapley_attack_dominant": sum(bool(row.get("shapley_attack_dominant")) for row in rows),
        "attention_shift": sum(bool(row.get("attention_shift", row.get("attention_shift_attack"))) for row in rows),
        "attention_attack_dominant": sum(bool(row.get("attention_attack_dominant")) for row in rows),
    }


def render_summary_cards(counts: dict) -> str:
    cards = [
        ("Records", counts.get("records")),
        ("Valid For Stats", counts.get("valid_for_stats")),
        ("Shapley Attack Dominant", counts.get("shapley_attack_dominant")),
        ("Attention Shift", counts.get("attention_shift")),
        ("Attention Attack Dominant", counts.get("attention_attack_dominant")),
        ("Eval", ", ".join(f"{k}: {v}" for k, v in sorted((counts.get("eval") or {}).items()))),
        ("Action Kind", ", ".join(f"{k}: {v}" for k, v in sorted((counts.get("action_kind") or {}).items()))),
    ]
    return "".join(
        f"""
        <div class="summary-card">
          <span>{html.escape(label)}</span>
          <strong>{html.escape(str(value))}</strong>
        </div>
        """
        for label, value in cards
    )


def render_html(rows: list[dict], source: dict, title: str) -> str:
    counts = summary_counts(rows)
    records = "\n".join(render_record(row) for row in rows)
    eval_options = "".join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in sorted({str(row.get("eval", "unknown")) for row in rows}))
    action_options = "".join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in sorted({str(row.get("action_kind", "unknown")) for row in rows}))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --line: #d8dee9;
      --text: #172033;
      --muted: #667085;
      --auth: #1f5ea8;
      --fact: #167457;
      --attack: #b42318;
      --special: #6b7280;
      --good: #177245;
      --bad: #b42318;
      --warn: #9a6700;
      --attn: #7c3aed;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); }}
    .shell {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    .page-head {{ border-bottom: 1px solid var(--line); padding-bottom: 18px; margin-bottom: 18px; }}
    .page-head h1 {{ margin: 0 0 8px; font-size: 32px; letter-spacing: 0; }}
    .meta {{ color: var(--muted); display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; }}
    .meta code, code {{ background: #eef2f7; border: 1px solid var(--line); border-radius: 5px; padding: 1px 4px; }}
    .overview {{ margin-bottom: 22px; }}
    .filters {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-top: 16px; }}
    .filters label {{ color: var(--muted); font-size: 12px; }}
    .filters select {{ display: block; min-width: 150px; margin-top: 4px; padding: 7px; border: 1px solid var(--line); border-radius: 6px; background: white; color: var(--text); }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: 0 8px 22px rgba(16, 24, 40, 0.04); }}
    .panel h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .panel h3 {{ margin: 18px 0 10px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }}
    .summary-card {{ border: 1px solid var(--line); border-radius: 8px; background: #fbfcfe; padding: 12px; min-height: 72px; }}
    .summary-card span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }}
    .summary-card strong {{ display: block; font-size: 20px; overflow-wrap: anywhere; }}
    .record {{ display: block; margin: 16px 0; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); box-shadow: 0 8px 22px rgba(16, 24, 40, 0.04); overflow: hidden; }}
    .record.is-filtered {{ display: none !important; }}
    .record > summary {{ cursor: pointer; list-style: none; display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 16px 18px; background: #fbfcfe; border-bottom: 1px solid var(--line); }}
    .record > summary::-webkit-details-marker {{ display: none; }}
    .record > summary::before {{ content: "v"; color: var(--muted); margin-right: 2px; }}
    .record:not([open]) > summary::before {{ content: ">"; }}
    .case-title {{ font-size: 20px; font-weight: 800; }}
    .case-meta {{ color: var(--muted); font-size: 13px; }}
    .record-body {{ padding: 16px; }}
    .layout {{ display: grid; grid-template-columns: minmax(320px, 0.42fr) minmax(0, 0.58fr); gap: 16px; align-items: start; }}
    .stack {{ display: grid; gap: 16px; }}
    .status-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 16px; }}
    .status-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    .status-card span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .status-card strong {{ font-size: 18px; overflow-wrap: anywhere; }}
    .status-good {{ color: var(--good); }} .status-bad {{ color: var(--bad); }} .status-warn {{ color: var(--warn); }} .status-neutral {{ color: var(--muted); }}
    .section-head, .card-head, .bar-label {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; }}
    .score-pair {{ color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; }}
    .chip {{ border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; color: var(--muted); font-size: 12px; }}
    .phi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .phi-card {{ border: 1px solid var(--line); border-radius: 8px; background: #fbfcfe; padding: 12px; }}
    .phi-card strong {{ font-variant-numeric: tabular-nums; }}
    .margin-card small {{ color: var(--muted); }}
    .bar, .bar-track {{ height: 10px; background: #edf1f6; border-radius: 999px; overflow: hidden; border: 1px solid #e2e8f0; }}
    .bar span {{ display: block; height: 100%; border-radius: inherit; }}
    .fill-pos {{ background: var(--fact); }} .fill-neg {{ background: var(--attack); }} .fill-attn {{ background: var(--attn); }}
    .bar-row {{ margin-bottom: 10px; }}
    .attention-details {{ margin-top: 18px; }}
    .attention-details > summary {{ cursor: pointer; list-style: none; display: flex; align-items: center; gap: 6px; width: fit-content; color: var(--muted); font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0; }}
    .attention-details > summary::-webkit-details-marker {{ display: none; }}
    .attention-details > summary::before {{ content: ">"; color: var(--muted); }}
    .attention-details[open] > summary::before {{ content: "v"; }}
    .attention-details-body {{ margin-top: 10px; }}
    .pill {{ display: inline-block; border: 1px solid currentColor; border-radius: 999px; padding: 2px 7px; font-size: 11px; font-weight: 700; margin-right: 4px; }}
    .region-auth {{ color: var(--auth); }} .region-fact {{ color: var(--fact); }} .region-attack {{ color: var(--attack); }} .region-special, .region-empty {{ color: var(--special); }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf1f6; text-align: left; vertical-align: top; }}
    th {{ background: #f3f6fa; color: var(--muted); }}
    .num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .compact th, .compact td {{ padding: 6px 8px; }}
    .action-panel pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .player-grid {{ display: grid; gap: 10px; }}
    .player-block {{ border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #fbfcfe; }}
    .block-head {{ padding: 8px 10px; border-bottom: 1px solid var(--line); background: #f7f9fc; }}
    .player-block pre {{ margin: 0; padding: 10px; max-height: 220px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .token-toolbar {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 16px 0 10px; color: var(--muted); font-size: 13px; }}
    .token-toolbar strong {{ color: var(--text); margin-right: 4px; }}
    .token-toolbar label {{ display: inline-flex; align-items: center; gap: 4px; }}
    .token-cloud {{ border: 1px solid var(--line); border-radius: 8px; background: #fbfcfe; padding: 10px; max-height: 360px; overflow: auto; font: 12px/1.7 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .token-chip {{ display: inline-block; margin: 1px 2px; padding: 1px 3px; border-radius: 4px; cursor: help; }}
    .token-auth {{ background: rgba(31, 94, 168, 0.12); color: var(--auth); }}
    .token-data_fact {{ background: rgba(22, 116, 87, 0.13); color: var(--fact); }}
    .token-data_attack {{ background: rgba(180, 35, 24, 0.14); color: var(--attack); }}
    .token-special {{ background: rgba(107, 114, 128, 0.12); color: var(--special); }}
    .token-chip.is-hidden {{ display: none; }}
    .attention-score-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .attention-score {{ border: 1px solid var(--line); border-radius: 8px; background: #fbfcfe; padding: 10px; }}
    .attention-score span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .attention-score strong {{ font-size: 18px; font-variant-numeric: tabular-nums; }}
    .empty {{ color: var(--muted); font-style: italic; }}
    @media (max-width: 980px) {{
      .layout, .status-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell" id="top">
    <header class="page-head">
      <h1>{html.escape(title)}</h1>
      <div class="meta">
        <span>Shapley: <code>{html.escape(source.get("shapley", ""))}</code></span>
        <span>Attention: <code>{html.escape(source.get("attention", "") or "none")}</code></span>
        <span>Cases: <code>{html.escape(source.get("cases", "") or "none")}</code></span>
      </div>
    </header>
    <section class="overview">
      <div class="panel summary-panel">
        <h2>Summary</h2>
        <div class="summary-grid">{render_summary_cards(counts)}</div>
        <div class="filters">
          <label>Eval<select id="eval-filter"><option value="">All</option>{eval_options}</select></label>
          <label>Action<select id="action-filter"><option value="">All</option>{action_options}</select></label>
          <label>Shapley Attack Dominant<select id="shapley-filter"><option value="">All</option><option value="true">True</option><option value="false">False</option></select></label>
          <label>Experiment Joint Method<select id="joint-filter"><option value="">All</option><option value="true">True</option><option value="false">False</option></select></label>
          <button id="clear-filters" type="button">Clear filters</button>
        </div>
      </div>
    </section>
    {records}
  </main>
  <script>
    function applyFilters() {{
      const evalValue = document.getElementById("eval-filter").value;
      const actionValue = document.getElementById("action-filter").value;
      const shapleyValue = document.getElementById("shapley-filter").value;
      const jointValue = document.getElementById("joint-filter").value;
      document.querySelectorAll(".record").forEach((record) => {{
        const matches = (!evalValue || record.dataset.eval === evalValue)
          && (!actionValue || record.dataset.action === actionValue)
          && (!shapleyValue || record.dataset.shapley === shapleyValue)
          && (!jointValue || record.dataset.joint === jointValue);
        record.classList.toggle("is-filtered", !matches);
      }});
    }}
    document.querySelectorAll(".filters select").forEach((select) => select.addEventListener("change", applyFilters));
    document.getElementById("clear-filters").addEventListener("click", () => {{
      document.querySelectorAll(".filters select").forEach((select) => {{ select.value = ""; }});
      applyFilters();
    }});
    document.querySelectorAll(".token-toolbar input[type='checkbox']").forEach((box) => {{
      box.addEventListener("change", () => {{
        const panel = box.closest(".panel");
        const region = box.dataset.filter;
        panel.querySelectorAll(`.token-chip[data-region="${{region}}"]`).forEach((token) => {{
          token.classList.toggle("is-hidden", !box.checked);
        }});
      }});
    }});
  </script>
</body>
</html>
"""


def prepare_rows(
    shapley_rows: list[dict],
    attention_rows: list[dict],
    judge_rows: list[dict],
    case_rows: list[dict],
    all_rows: bool,
    index: int,
    case_id: str | None,
    limit: int | None,
    target_scope: str = "full_action",
) -> list[dict]:
    attn = attention_index(attention_rows)
    judges = judge_index(judge_rows)
    cases = case_index(case_rows)
    if target_scope != "all":
        rows_source = [
            row for row in shapley_rows
            if str(row.get("target_scope", "full_action")) == target_scope
        ]
    else:
        rows_source = shapley_rows
    if all_rows:
        rows = rows_source
        if limit is not None:
            rows = rows[:limit]
    else:
        rows = [select_record(rows_source, case_id, index)]
    return [enrich_with_case(merge_judge(merge_attention(row, attn), judges), cases) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render InjecAgent action-level Shapley/attention JSONL as HTML.")
    parser.add_argument("--shapley", required=True, help="results.action_shapley.jsonl from injecagent_action_experiment.py")
    parser.add_argument("--attention", help="Optional results.action_attention.jsonl from injecagent_action_experiment.py")
    parser.add_argument("--judge", help="Optional independent LLM judge JSONL")
    parser.add_argument("--cases", default=DEFAULT_CASES, help="Original qwen InjecAgent JSONL used to show AUTH/FACT/ATTACK text.")
    parser.add_argument("--output", required=True, help="HTML output path.")
    parser.add_argument("--case_id", help="Render a specific case_id.")
    parser.add_argument("--index", type=int, default=0, help="Record index when --case_id is not provided.")
    parser.add_argument("--all", action="store_true", help="Render all records into one gallery.")
    parser.add_argument("--limit", type=int, help="Optional limit in --all mode.")
    parser.add_argument("--target-scope", choices=("full_action", "tool_name", "all"), default="full_action",
                        help="Target scope to display; defaults to one full-action record per case.")
    parser.add_argument("--title", default="InjecAgent Action-Level Attribution")
    args = parser.parse_args()

    shapley_rows = read_jsonl(args.shapley)
    attention_rows = read_jsonl(args.attention)
    judge_rows = read_jsonl(args.judge)
    case_rows = read_jsonl(args.cases)
    rows = prepare_rows(shapley_rows, attention_rows, judge_rows, case_rows, args.all, args.index, args.case_id, args.limit, args.target_scope)
    page = render_html(rows, {"shapley": args.shapley, "attention": args.attention, "judge": args.judge, "cases": args.cases}, args.title)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(page)


if __name__ == "__main__":
    main()
