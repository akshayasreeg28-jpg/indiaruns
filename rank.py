#!/usr/bin/env python3
"""
rank.py — produces the top-100 ranked submission CSV for the Redrob
Intelligent Candidate Discovery & Ranking Challenge.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

Design summary (full detail in README.md):
  1. Stream candidates.jsonl line by line (no full-file load needed — keeps
     memory flat regardless of pool size).
  2. For each candidate: extract features, run honeypot checks, run hard
     disqualifier checks, compute 5 component scores (title fit, career
     substance, trust-weighted skills, experience band, location).
  3. Combine components into a base fit score, then apply:
       - a multiplicative behavioral-availability modifier
       - a multiplicative honeypot penalty (near-zero, not literal zero --
         see README for why a hard zero is avoided)
       - a multiplicative hard-disqualifier penalty
  4. Sort descending by final score, deterministic tie-break by candidate_id
     ascending (per spec Section 3).
  5. Take top 100, write CSV with the required header and column order,
     with a real per-row reasoning string (see reasoning.py).

No network calls, no GPU, no LLM calls happen anywhere in this file or its
imports — verify with `grep -ri "api\\|requests\\|openai\\|anthropic" src/`.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.scoring import (
    extract_features, detect_honeypot, hard_disqualifiers,
    score_title_fit, score_career_substance, score_skills_with_trust,
    score_experience_band, score_location, score_behavioral_multiplier,
)
from src.reasoning import build_reasoning

# Composite weights for the base (pre-modifier) fit score.
# Title fit is weighted heaviest deliberately: it's the single strongest
# defense against the keyword-stuffer trap (an HR Manager with 9 AI skills
# still scores low here because their title isn't AI/ML/data at all).
WEIGHTS = {
    "title_fit": 0.32,
    "career_substance": 0.23,
    "skills_trust": 0.23,
    "experience_band": 0.12,
    "location": 0.10,
}

HONEYPOT_PENALTY = 0.05   # multiply score by this if any honeypot flag fires
DISQUALIFIER_PENALTY = 0.15  # multiply score by this if any hard disqualifier fires


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

    final = max(0.0, min(final, 1.0))

    return {
        "candidate_id": feat.candidate_id,
        "score": final,
        "feat": feat,
        "components": comp,
        "honeypot": honeypot,
        "disqualifiers": disq,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl (plain or .gz)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()

    path = Path(args.candidates)
    opener = open
    if path.suffix == ".gz":
        import gzip
        opener = gzip.open

    results = []
    n_read = 0
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            results.append(score_candidate(c))
            n_read += 1

    # Round to the same precision we write to the CSV *before* sorting, so
    # the tie-break (candidate_id ascending) is computed on the value the
    # validator actually sees. Two candidates can have distinct raw floats
    # that round to the same 4-decimal score — sorting on the raw float
    # first and writing the rounded value afterward breaks the spec's
    # "equal scores -> candidate_id ascending" rule. Caught by running the
    # official validate_submission.py against our own output.
    for r in results:
        r["score"] = round(r["score"], 4)

    # Sort: score descending, then candidate_id ascending for deterministic ties
    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))

    top = results[: args.top_n]

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, r in enumerate(top, start=1):
            reasoning = build_reasoning(r["feat"], r["components"], r["honeypot"], r["disqualifiers"])
            writer.writerow([r["candidate_id"], i, f"{r['score']:.4f}", reasoning])

    elapsed = time.time() - t0
    honeypots_in_top100 = sum(1 for r in top if r["honeypot"])
    print(f"Scored {n_read} candidates in {elapsed:.1f}s.")
    print(f"Wrote top {len(top)} to {out_path}")
    print(f"Honeypots in top 100: {honeypots_in_top100} ({honeypots_in_top100}%)")


if __name__ == "__main__":
    main()
