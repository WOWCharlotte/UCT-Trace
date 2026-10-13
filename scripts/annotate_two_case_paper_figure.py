"""Highlight Attention and Shapley focus in the two-case paper figure."""

from __future__ import annotations

from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "result" / "agentdojo_fact_longer_than_attack_qwen_8b_chunk16_fixed"
SOURCE = CASE_DIR / "two_case_paper_figure.pdf"
OUTPUT = CASE_DIR / "two_case_paper_figure_highlighted.pdf"


def highlight_summary(page: fitz.Page) -> None:
    attention_color = (255 / 255, 211 / 255, 115 / 255)  # #FFD373
    shapley_color = (242 / 255, 120 / 255, 115 / 255)

    # The same three FACT lines appear in both examples. Yellow fill marks the
    # Attention focus; a red outline also marks Shapley focus in Example 1.
    fact_lines = [
        [(82, 90, 239, 105), (82, 105, 232, 120), (82, 120, 193, 135)],
        [(335, 90, 492, 105), (335, 105, 485, 120), (335, 120, 446, 135)],
    ]
    for lines in fact_lines:
        for box in lines:
            page.draw_rect(
                fitz.Rect(*box), fill=attention_color, color=None,
                fill_opacity=0.42, overlay=True,
            )
    page.draw_rect(
        fitz.Rect(81, 89, 240, 136), color=shapley_color,
        width=1.4, overlay=True,
    )

    # Example 2: only the quoted transfer instruction is the Shapley focus.
    # The remaining attack sentence is deliberately outside the outline.
    page.draw_rect(
        fitz.Rect(335, 135, 485, 165), color=shapley_color,
        width=1.4, overlay=True,
    )

    page.insert_text(
        (20, 477),
        "Yellow area = Attention focus  |  Red outline = Shapley focus",
        fontname="helv", fontsize=6.7, color=(0.27, 0.31, 0.34), overlay=True,
    )


def main() -> None:
    doc = fitz.open(SOURCE)
    if len(doc) != 1:
        raise ValueError("Expected the original one-page paper figure")
    highlight_summary(doc[0])
    doc.save(OUTPUT, garbage=4, deflate=True)
    doc.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
