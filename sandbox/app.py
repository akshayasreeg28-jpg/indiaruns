"""
HuggingFace Spaces sandbox demo for the Redrob hackathon ranker.

This app imports the exact same src/scoring.py and src/reasoning.py used by
rank.py — it is not a reimplementation or a simplified stand-in. Running this
on the bundled 80-candidate sample (or an uploaded small JSONL) reproduces
the same logic that produced submission.csv, just scoped down for a fast,
free-tier-CPU demo per the spec's Section 10.5 sandbox requirement.
"""

import json
import csv
import io
import gradio as gr
import pandas as pd

from src.scoring import (
    extract_features, detect_honeypot, hard_disqualifiers,
    score_title_fit, score_career_substance, score_skills_with_trust,
    score_experience_band, score_location, score_behavioral_multiplier,
)
from src.reasoning import build_reasoning

WEIGHTS = {
    "title_fit": 0.32,
    "career_substance": 0.23,
    "skills_trust": 0.23,
    "experience_band": 0.12,
    "location": 0.10,
}
HONEYPOT_PENALTY = 0.05
DISQUALIFIER_PENALTY = 0.15


def score_candidate(c: dict) -> dict:
    feat = extract_features(c)
    honeypot = detect_honeypot(c, feat)
    disq = hard_disqualifiers(feat)

    comp = {
        "title_fit": score_title_fit(feat),
        "career_substance": score_career_substance(feat),
        "skills_trust": score_skills_with_trust(feat),
        "experience_band": score_experience_band(feat),
        "location": score_location(feat),
    }
    base = sum(WEIGHTS[k] * v for k, v in comp.items())
    behavioral_mult = score_behavioral_multiplier(feat)
    final = base * behavioral_mult
    if honeypot:
        final *= HONEYPOT_PENALTY
    if disq:
        final *= DISQUALIFIER_PENALTY
    final = round(max(0.0, min(final, 1.0)), 4)

    return {
        "candidate_id": feat.candidate_id, "score": final, "feat": feat,
        "components": comp, "honeypot": honeypot, "disqualifiers": disq,
    }


def rank_file(file_obj, top_n):
    if file_obj is None:
        path = "sandbox/sample_candidates.jsonl"
    else:
        path = file_obj.name

    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            results.append(score_candidate(c))

    for r in results:
        r["score"] = round(r["score"], 4)
    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = results[: int(top_n)]

    rows = []
    for i, r in enumerate(top, start=1):
        reasoning = build_reasoning(r["feat"], r["components"], r["honeypot"], r["disqualifiers"])
        rows.append({"rank": i, "candidate_id": r["candidate_id"], "score": r["score"], "reasoning": reasoning})

    df = pd.DataFrame(rows)

    # also produce a downloadable CSV
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for row in rows:
        writer.writerow([row["candidate_id"], row["rank"], f"{row['score']:.4f}", row["reasoning"]])
    out_path = "/tmp/submission_preview.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    summary = f"Scored {len(results)} candidates. Honeypots flagged: {sum(1 for r in results if r['honeypot'])}. Hard disqualifiers flagged: {sum(1 for r in results if r['disqualifiers'])}."
    return df, out_path, summary


with gr.Blocks(title="Redrob Ranker — Sandbox Demo") as demo:
    gr.Markdown(
        "# Redrob Hackathon — Candidate Ranker (Sandbox Demo)\n"
        "Rule-based ranker for the **Senior AI Engineer — Founding Team** JD. "
        "No network calls, no GPU, pure Python stdlib. This demo runs the exact "
        "same `src/` code as the full submission, scoped to a small sample for "
        "fast iteration. Upload your own small `.jsonl` (same schema as "
        "`candidate_schema.json`) or just click Run to use the bundled 80-candidate sample.\n\n"
        "**Note on the bundled sample:** it's a random 80-candidate draw from the "
        "100K pool, not a curated 'good' subset — so it may contain few or no "
        "strong AI/ML matches, and the top rows here can show low absolute scores "
        "(e.g. the best-of-a-weak-pool candidate). This is expected, not a bug. "
        "On the full 100K pool, top-100 scores range ~0.48–0.81; see `submission.csv` "
        "in the main repo for the real output."
    )
    with gr.Row():
        file_in = gr.File(label="Upload candidates.jsonl (optional — leave empty to use bundled sample)", file_types=[".jsonl"])
        top_n = gr.Slider(5, 80, value=20, step=1, label="Top N to show")
    run_btn = gr.Button("Run ranker", variant="primary")
    summary_out = gr.Textbox(label="Summary", interactive=False)
    table_out = gr.Dataframe(label="Ranked results")
    file_out = gr.File(label="Download ranked CSV")

    run_btn.click(rank_file, inputs=[file_in, top_n], outputs=[table_out, file_out, summary_out])

if __name__ == "__main__":
    demo.launch()
