"""Render AgentDojo next-assistant Attention and Shapley results as HTML."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from attack_judge_support import attack_margin


REGIONS = ("auth", "data_fact", "data_attack", "special")
LABELS = {"auth": "AUTH", "data_fact": "FACT", "data_attack": "ATTACK", "special": "SPECIAL"}
CLASSES = {"auth": "auth", "data_fact": "fact", "data_attack": "attack", "special": "special"}


def read_jsonl(path: str | None) -> list[dict]:
    if not path:
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_index(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        target_id = str(row.get("target_id"))
        if target_id in result:
            raise ValueError(f"Duplicate target_id in visualization input: {target_id}")
        result[target_id] = row
    return result


def merge_rows(shapley_rows: list[dict], attention_rows: list[dict], judge_rows: list[dict] | None = None) -> list[dict]:
    shapley = row_index(shapley_rows)
    attention = row_index(attention_rows)
    judges = row_index(judge_rows or [])
    target_ids = list(shapley) if shapley else list(attention)
    merged = []
    for target_id in target_ids:
        row = dict(shapley.get(target_id) or attention[target_id])
        if target_id in attention:
            row["attention"] = attention[target_id]
        if target_id in shapley:
            row["shapley"] = shapley[target_id]
        judge = judges.get(target_id)
        if judge is None:
            row["judge"] = {"status": "judge_unavailable"}
            row["attack_action_executed"] = None
            row["attack_success_strict"] = False
        else:
            row["judge"] = judge.get("judge") or judge
            row["attack_action_executed"] = judge.get("attack_action_executed")
            attribution_trigger = judge.get("attack_attribution_trigger",
                                             (row.get("shapley") or {}).get("shapley_attack_dominant", False))
            row["attack_attribution_trigger"] = bool(attribution_trigger)
            row["attack_success_strict"] = bool(attribution_trigger and row["attack_action_executed"] is True)
        merged.append(row)
    return merged


def select_rows(
    rows: list[dict], all_rows: bool, index: int, target_id: str | None, limit: int | None
) -> list[dict]:
    if target_id is not None:
        selected = [row for row in rows if str(row.get("target_id")) == target_id]
        if not selected:
            raise ValueError(f"No AgentDojo record found with target_id={target_id!r}")
        return selected
    if all_rows:
        return rows[:limit] if limit is not None else rows
    if index < 0 or index >= len(rows):
        raise IndexError(f"--index {index} is out of range for {len(rows)} records")
    return [rows[index]]


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if math.isnan(number) or math.isinf(number):
        return str(number)
    return f"{number:.{digits}f}"


def yes_no(value) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return "UNKNOWN"


def player_value(row: dict, player: str):
    shapley = row.get("shapley") or row
    return shapley.get({"auth": "phi_auth", "data_fact": "phi_data_fact", "data_attack": "phi_data_attack"}[player])


def shapley_attack_margin(row: dict) -> float:
    return attack_margin(player_value(row, "data_attack"), player_value(row, "auth"), player_value(row, "data_fact"))


def render_shapley(row: dict) -> str:
    values = [player_value(row, player) for player in REGIONS[:3]]
    finite = [abs(float(value)) for value in values if value is not None]
    bound = max(finite, default=1.0) or 1.0
    cards = []
    for player in REGIONS[:3]:
        value = player_value(row, player)
        if value is None:
            continue
        number = float(value)
        width = min(abs(number) / bound * 100, 100)
        cards.append(f"""<div class="metric-card"><div class="metric-head"><span class="pill {CLASSES[player]}">{LABELS[player]}</span><strong>{fmt(number)}</strong></div><div class="bar"><span class="{'positive' if number >= 0 else 'negative'}" style="width:{width:.2f}%"></span></div></div>""")
    shapley = row.get("shapley") or row
    if not cards:
        return '<section class="panel"><h2>Shapley</h2><p class="empty">No Shapley record.</p></section>'
    return f"""<section class="panel"><div class="section-head"><h2>Shapley</h2><span class="chip">Attack dominant: {yes_no(shapley.get('shapley_attack_dominant'))}</span></div><div class="metric-grid">{''.join(cards)}</div><p class="subtle">Attack margin (v1): {fmt(shapley_attack_margin(row))} · 8-coalition teacher-forced mean logprob · efficiency error {fmt(shapley.get('efficiency_error'))}</p></section>"""


def render_attention(row: dict) -> str:
    attention = row.get("attention") or {}
    raw = attention.get("region_scores_raw") or {}
    prompt = attention.get("region_scores_prompt_normalized") or {}
    players = attention.get("region_scores_player_normalized") or {}
    if not raw:
        return '<section class="panel"><h2>Attention</h2><p class="empty">No Attention record.</p></section>'
    player_bars = []
    for region in REGIONS[:3]:
        value = float(players.get(region, 0.0))
        player_bars.append(f"""<div class="score-row"><div><span class="pill {CLASSES[region]}">{LABELS[region]}</span><span>player norm {fmt(value)} · raw {fmt(raw.get(region))}</span></div><div class="bar"><span class="attention-fill" style="width:{min(value * 100, 100):.2f}%"></span></div></div>""")
    prompt_bars = []
    for region in REGIONS:
        value = float(prompt.get(region, 0.0))
        prompt_bars.append(f"""<div class="score-row"><div><span class="pill {CLASSES[region]}">{LABELS[region]}</span><span>prompt norm {fmt(value)} · raw {fmt(raw.get(region))}</span></div><div class="bar"><span class="attention-fill" style="width:{min(value * 100, 100):.2f}%"></span></div></div>""")
    token_chips = []
    for token in attention.get("tokens") or []:
        region = str(token.get("r", "special"))
        token_chips.append(f"""<span class="token token-{html.escape(region)}" data-region="{html.escape(region)}" title="index {html.escape(str(token.get('i')))} · attention {fmt(token.get('s'))}">{html.escape(str(token.get('t', '')))}</span>""")
    toolbar = "".join(f'<label><input type="checkbox" data-region-toggle="{region}" checked> {LABELS[region]}</label>' for region in REGIONS)
    return f"""<section class="panel"><div class="section-head"><h2>Attention</h2><span class="chip">Shift: {yes_no(attention.get('attention_shift'))}</span></div><div class="status-grid"><div><span>Auth Focus</span><strong>{fmt(attention.get('auth_focus_score'))}</strong></div><div><span>Threshold</span><strong>{fmt(attention.get('threshold'))}</strong></div><div><span>Attack Dominant</span><strong>{yes_no(attention.get('attention_attack_dominant'))}</strong></div></div><h3>Player-Normalized Attention</h3>{''.join(player_bars)}<details><summary>Prompt-Normalized Attention Including SPECIAL</summary><div class="details-body">{''.join(prompt_bars)}</div></details><div class="token-toolbar"><strong>Context tokens</strong>{toolbar}</div><div class="tokens">{''.join(token_chips)}</div></section>"""


def render_players(row: dict) -> str:
    source = row.get("shapley") or row.get("attention") or row
    texts = source.get("player_text") or {}
    blocks = []
    for player in REGIONS[:3]:
        value = texts.get(player, "")
        text = "".join(value) if isinstance(value, list) else str(value)
        blocks.append(f"""<div class="player-block {CLASSES[player]}"><span class="pill {CLASSES[player]}">{LABELS[player]}</span><pre>{html.escape(text)}</pre></div>""")
    return f'<section class="panel"><h2>Players</h2><div class="player-grid">{"".join(blocks)}</div></section>'


def render_record(row: dict) -> str:
    source = row.get("shapley") or row.get("attention") or row
    attack_success = source.get("attack_success")
    open_attr = " open" if attack_success or (row.get("attention") or {}).get("attention_shift") else ""
    target_id = html.escape(str(source.get("target_id")))
    judge = row.get("judge") or {}
    metadata = [
        ("Suite", source.get("suite_name")), ("Security", source.get("security")),
        ("Utility", source.get("utility")), ("Ground-truth attack success", attack_success),
        ("Experiment joint method", row.get("attack_success_strict")),
        ("Judge behavior", judge.get("behavior_label") or judge.get("status")),
        ("Shapley seconds", source.get("shapley_time_seconds")),
        ("Attention seconds", source.get("attention_time_seconds")),
        ("Target kind", source.get("target_kind")), ("Target scope", source.get("target_scope")),
        ("Polluted tool index", source.get("polluted_tool_message_index")),
        ("Assistant index", source.get("target_assistant_message_index")),
    ]
    rows = "".join(f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value))}</td></tr>" for label, value in metadata)
    target = html.escape(str(source.get("target_text") or source.get("error") or ""))
    return f"""<details class="record" id="target-{target_id}" data-suite="{html.escape(str(source.get('suite_name')))}" data-security="{html.escape(str(source.get('security')).lower())}" data-kind="{html.escape(str(source.get('target_kind')))}"{open_attr}><summary><span><strong>{html.escape(str(source.get('case_id')))}</strong><small>{target_id}</small></span><span>{html.escape(str(source.get('target_kind')))} · ground-truth success {yes_no(attack_success)} · joint method {yes_no(row.get('attack_success_strict'))}</span></summary><div class="record-body"><div class="layout"><div class="stack"><section class="panel"><h2>Selection</h2><table>{rows}</table></section>{render_players(row)}<section class="panel"><h2>Next Assistant Target</h2><pre>{target}</pre></section></div><div class="stack">{render_shapley(row)}{render_attention(row)}</div></div></div></details>"""


def summary(rows: list[dict]) -> dict:
    sources = [row.get("shapley") or row.get("attention") or row for row in rows]
    return {
        "records": len(rows),
        "suite": dict(Counter(str(row.get("suite_name")) for row in sources)),
        "target_kind": dict(Counter(str(row.get("target_kind")) for row in sources)),
        "attack_success": sum(row.get("attack_success") is True for row in sources),
        "attack_success_strict": sum(row.get("attack_success_strict") is True for row in rows),
        "judge_behavior": dict(Counter(str((row.get("judge") or {}).get("behavior_label") or (row.get("judge") or {}).get("status")) for row in rows)),
        "attention_shift": sum((row.get("attention") or {}).get("attention_shift") is True for row in rows),
        "shapley_attack_dominant": sum((row.get("shapley") or {}).get("shapley_attack_dominant") is True for row in rows),
    }


def render_html(rows: list[dict], source: dict, title: str, page_size: int = 20) -> str:
    if page_size < 1:
        raise ValueError("page_size must be positive")
    counts = summary(rows)
    suites = sorted(counts["suite"])
    kinds = sorted(counts["target_kind"])
    options = lambda values: "".join(f'<option value="{html.escape(value)}">{html.escape(value)}</option>' for value in values)
    cards = "".join(f'<div class="summary-card"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>' for label, value in [
        ("Records", counts["records"]), ("Ground-truth Attack Success", counts["attack_success"]),
        ("Experiment Joint Method", counts["attack_success_strict"]),
        ("Attention Shift", counts["attention_shift"]), ("Shapley Attack Dominant", counts["shapley_attack_dominant"]),
    ])
    pages = [rows[index:index + page_size] for index in range(0, len(rows), page_size)] or [[]]
    page_templates = "".join(
        f'<template id="records-page-{index}">{"".join(render_record(row) for row in page)}</template>'
        for index, page in enumerate(pages)
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>
:root{{--bg:#f6f7f9;--panel:#fff;--line:#d8dee8;--text:#172033;--muted:#667085;--auth:#1f5ea8;--fact:#167457;--attack:#b42318;--special:#6b7280;--attn:#7c3aed}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}}main{{max-width:1440px;margin:auto;padding:24px}}h1{{font-size:30px;margin:0 0 6px;letter-spacing:0}}h2{{font-size:18px;margin:0 0 12px}}h3{{font-size:12px;text-transform:uppercase;color:var(--muted);letter-spacing:0;margin:18px 0 10px}}.subtle,small{{display:block;color:var(--muted);overflow-wrap:anywhere}}.summary-grid,.metric-grid,.status-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}.summary-card,.status-grid>div,.metric-card{{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fbfcfe}}.summary-card span,.status-grid span{{display:block;color:var(--muted);font-size:12px}}.summary-card strong,.status-grid strong{{font-size:19px}}.toolbar{{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:18px 0}}label{{color:var(--muted);font-size:12px}}select{{display:block;min-width:150px;margin-top:4px;padding:7px;border:1px solid var(--line);border-radius:6px;background:white}}.record{{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:14px 0;overflow:hidden}}.record>summary{{cursor:pointer;display:flex;justify-content:space-between;gap:16px;padding:15px;background:#fbfcfe}}.record-body{{padding:16px}}.layout{{display:grid;grid-template-columns:minmax(320px,.42fr) minmax(0,.58fr);gap:16px}}.stack,.player-grid{{display:grid;gap:14px}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}}.section-head,.metric-head,.score-row>div:first-child{{display:flex;justify-content:space-between;align-items:center;gap:8px}}table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid #edf1f6;padding:7px;text-align:left;vertical-align:top}}th{{color:var(--muted);width:42%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.55 ui-monospace,monospace;margin:0;max-height:300px;overflow:auto}}.player-block{{border-left:5px solid;padding:10px;background:#fbfcfe}}.auth{{color:var(--auth);border-color:var(--auth)}}.fact{{color:var(--fact);border-color:var(--fact)}}.attack{{color:var(--attack);border-color:var(--attack)}}.special{{color:var(--special);border-color:var(--special)}}.pill,.chip{{display:inline-block;border:1px solid currentColor;border-radius:999px;padding:2px 7px;font-size:11px;font-weight:700}}.bar{{height:9px;background:#edf1f6;border-radius:99px;overflow:hidden;margin-top:7px}}.bar span{{display:block;height:100%}}.positive{{background:var(--fact)}}.negative{{background:var(--attack)}}.attention-fill{{background:var(--attn)}}.score-row{{margin:10px 0}}details>summary{{cursor:pointer}}.details-body{{margin-top:10px}}.token-toolbar{{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 8px}}.tokens{{border:1px solid var(--line);padding:9px;max-height:360px;overflow:auto;font:12px/1.7 ui-monospace,monospace}}.token{{display:inline-block;padding:1px 3px;margin:1px;border-radius:3px}}.token-auth{{background:#1f5ea820;color:var(--auth)}}.token-data_fact{{background:#16745720;color:var(--fact)}}.token-data_attack{{background:#b4231824;color:var(--attack)}}.token-special{{background:#6b728020;color:var(--special)}}.hidden{{display:none!important}}@media(max-width:900px){{.layout{{grid-template-columns:1fr}}main{{padding:12px}}}}
</style></head><body><main><header><h1>{html.escape(title)}</h1><p class="subtle">Shapley: {html.escape(source.get('shapley') or 'none')} · Attention: {html.escape(source.get('attention') or 'none')}</p></header><section class="summary-grid">{cards}</section><div class="toolbar"><label>Suite<select id="suite-filter"><option value="">All</option>{options(suites)}</select></label><label>Security<select id="security-filter"><option value="">All</option><option value="true">True</option><option value="false">False</option></select></label><label>Target kind<select id="kind-filter"><option value="">All</option>{options(kinds)}</select></label></div><div class="pagination"><button id="previous-page" type="button">Previous</button><span id="page-label"></span><button id="next-page" type="button">Next</button></div><section id="records"></section><div class="page-templates">{page_templates}</div></main><script>
let currentPage = 0;
const pageCount = {len(pages)};
function bindTokens(){{document.querySelectorAll('#records [data-region-toggle]').forEach(box=>box.addEventListener('change',()=>{{const panel=box.closest('.panel'),region=box.dataset.regionToggle;panel.querySelectorAll(`.token[data-region="${{region}}"]`).forEach(token=>token.classList.toggle('hidden',!box.checked))}}));}}
function filterRecords(){{const s=document.querySelector('#suite-filter').value,q=document.querySelector('#security-filter').value,k=document.querySelector('#kind-filter').value;document.querySelectorAll('#records .record').forEach(r=>r.classList.toggle('hidden',(s&&r.dataset.suite!==s)||(q&&r.dataset.security!==q)||(k&&r.dataset.kind!==k)))}}
function renderPage(){{document.querySelector('#records').replaceChildren(document.getElementById(`records-page-${{currentPage}}`).content.cloneNode(true));document.querySelector('#page-label').textContent=`Page ${{currentPage+1}} / ${{pageCount}}`;document.querySelector('#previous-page').disabled=currentPage===0;document.querySelector('#next-page').disabled=currentPage===pageCount-1;bindTokens();filterRecords();}}
document.querySelectorAll('.toolbar select').forEach(x=>x.addEventListener('change',filterRecords));
document.querySelector('#previous-page').addEventListener('click',()=>{{if(currentPage>0){{currentPage-=1;renderPage();}}}});
document.querySelector('#next-page').addEventListener('click',()=>{{if(currentPage<pageCount-1){{currentPage+=1;renderPage();}}}});
renderPage();
</script></body></html>"""


def write_html(path: str, page: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render AgentDojo next-assistant attribution JSONL as HTML")
    parser.add_argument("--shapley", help="results.shapley.jsonl")
    parser.add_argument("--attention", help="results.attention.jsonl")
    parser.add_argument("--judge", help="Optional independent LLM judge JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-id")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--valid-only", action="store_true", help="Exclude records where either supplied attribution method failed.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--title", default="AgentDojo Next-Assistant Attribution")
    parser.add_argument("--page-size", type=int, default=20, help="Number of records rendered in the active page.")
    args = parser.parse_args()
    if not args.shapley and not args.attention:
        parser.error("at least one of --shapley or --attention is required")
    merged = merge_rows(read_jsonl(args.shapley), read_jsonl(args.attention), read_jsonl(args.judge))
    if args.valid_only:
        merged = [row for row in merged if all(
            row.get(method, {}).get("valid_for_stats")
            for method in ("shapley", "attention")
            if method in row
        )]
    selected = select_rows(merged, args.all, args.index, args.target_id, args.limit)
    write_html(args.output, render_html(selected, {"shapley": args.shapley, "attention": args.attention}, args.title, args.page_size))


if __name__ == "__main__":
    main()
