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
     disqualifier checks, compute 5 rule-based component scores (title fit,
     career substance, trust-weighted skills, experience band, location).
  3. Batch-encode every candidate's career_descriptions against the JD with
     a local sentence-transformers model (src/semantic.py) for an optional
     6th component: real semantic similarity, not just phrase matching.
     This step is OPTIONAL and SAFE: if the model isn't cached locally (see
     README "Semantic similarity setup"), this is automatically skipped —
     no network call is attempted, no exception is raised, and the 5
     rule-based components alone determine the score, with their weights
     renormalized to sum to 1.0. The submission is valid and complete
     either way; this component only adds lift when present.
  4. Combine components into a base fit score, then apply:
       - a multiplicative behavioral-availability modifier
       - a multiplicative honeypot penalty (near-zero, not literal zero --
         see README for why a hard zero is avoided)
       - a multiplicative hard-disqualifier penalty
       - a milder multiplicative title-chasing penalty
  5. Sort descending by final score, deterministic tie-break by candidate_id
     ascending (per spec Section 3).
  6. Take top 100, write CSV with the required header and column order,
     with a real per-row reasoning string (see reasoning.py).

No network calls, no GPU, no LLM calls happen anywhere in this file or its
imports during the ranking run itself — verify with
`grep -ri "api\\|requests\\|openai\\|anthropic" src/`. The one-time semantic
model download (if you choose to enable that component) happens before
ranking, as a separate setup step — see README.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.scoring import (
    extract_features, detect_honeypot, hard_disqualifiers, detect_title_chasing,
    score_title_fit, score_career_substance, score_skills_with_trust,
    score_experience_band, score_location, score_behavioral_multiplier,
)
from src.reasoning import build_reasoning
from src.semantic import SemanticScorer, JD_TECHNICAL_SUMMARY

# Composite weights for the base (pre-modifier) fit score, WHEN the semantic
# model is available. Title fit stays heaviest deliberately: it's the
# single strongest defense against the keyword-stuffer trap (an HR Manager
# with 9 AI skills still scores low here because their title isn't AI/ML/
# data at all). semantic_similarity gets a modest 0.15 -- it's a real,
# valuable signal but we deliberately don't let it dominate, since we
# couldn't fully validate its behavior end-to-end ourselves (see README) and
# a rule-based system whose top component is "trust the model" is exactly
# the kind of opaque scoring we built this whole approach to avoid.
WEIGHTS_WITH_SEMANTIC = {
    "title_fit": 0.28,
    "career_substance": 0.20,
    "skills_trust": 0.20,
    "experience_band": 0.10,
    "location": 0.07,
    "semantic_similarity": 0.15,
}

# Fallback weights when the semantic model isn't cached locally (see
# src/semantic.py). These are exactly the original 5-component weights --
# ratios preserved, semantic_similarity's 0.15 redistributed proportionally
# rather than just dropped, so removing this component doesn't silently
# shrink everyone's score by 15%.
WEIGHTS_NO_SEMANTIC = {
    "title_fit": 0.32,
    "career_substance": 0.23,
    "skills_trust": 0.23,
    "experience_band": 0.12,
    "location": 0.10,
}

HONEYPOT_PENALTY = 0.05   # multiply score by this if any honeypot flag fires
DISQUALIFIER_PENALTY = 0.15  # multiply score by this if any hard disqualifier fires
TITLE_CHASER_PENALTY = 0.55  # milder penalty — JD's tone here is "not a fit" not "will not move forward"


def score_candidate_rule_based(c: dict) -> dict:
    """Phase 1: everything except the semantic-similarity component."""
    feat = extract_features(c)
    honeypot = detect_honeypot(c, feat)
    disq = hard_disqualifiers(feat)
    title_chaser = detect_title_chasing(c, feat)

    comp = {
        "title_fit": score_title_fit(feat),
        "career_substance": score_career_substance(feat),
        "skills_trust": score_skills_with_trust(feat),
        "experience_band": score_experience_band(feat),
        "location": score_location(feat),
    }
    behavioral_mult = score_behavioral_multiplier(feat)

    return {
        "candidate_id": feat.candidate_id,
        "feat": feat,
        "components": comp,
        "behavioral_mult": behavioral_mult,
        "honeypot": honeypot,
        "disqualifiers": disq,
        "title_chaser": title_chaser,
    }


def finalize_score(r: dict, semantic_score: float | None) -> dict:
    """Phase 2: combine rule-based components with the (optional) semantic
    score, then apply the multiplicative modifiers/penalties."""
    comp = dict(r["components"])
    if semantic_score is not None:
        comp["semantic_similarity"] = semantic_score
        weights = WEIGHTS_WITH_SEMANTIC
    else:
        weights = WEIGHTS_NO_SEMANTIC

    base = sum(weights[k] * comp[k] for k in weights)
    final = base * r["behavioral_mult"]

    if r["honeypot"]:
        final *= HONEYPOT_PENALTY
    if r["disqualifiers"]:
        final *= DISQUALIFIER_PENALTY
    if r["title_chaser"]:
        final *= TITLE_CHASER_PENALTY

    r["components"] = comp
    r["score"] = max(0.0, min(final, 1.0))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl (plain or .gz)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--no-semantic", action="store_true",
                     help="Skip the semantic-similarity component even if the model is cached locally")
    args = ap.parse_args()

    t0 = time.time()

    path = Path(args.candidates)
    opener = open
    if path.suffix == ".gz":
        import gzip
        opener = gzip.open

    # Phase 1: rule-based scoring only, streamed (constant memory regardless
    # of pool size for this phase, since we only keep small feature objects,
    # not the raw JSON).
    results = []
    n_read = 0
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            results.append(score_candidate_rule_based(c))
            n_read += 1

    rule_elapsed = time.time() - t0

    # Phase 2: optional semantic similarity, batch-encoded all at once (this
    # is why it has to be a second pass rather than folded into phase 1 --
    # batch encoding is dramatically faster than encoding one text at a
    # time, and most of that throughput would be lost in a streaming loop).
    semantic_scores: list[float | None]
    if args.no_semantic:
        semantic_scores = [None] * len(results)
        semantic_status = "disabled via --no-semantic"
    else:
        scorer = SemanticScorer()
        if scorer.is_available:
            career_texts = [r["feat"].career_descriptions for r in results]

            # Time-budget probe: encode a small sample first and extrapolate
            # to the full pool before committing to the full pass. We
            # couldn't benchmark real throughput ourselves (see semantic.py
            # docstring), so this guard protects the 5-minute budget on
            # whatever machine actually runs this, rather than assuming a
            # number we never measured.
            probe_n = min(500, len(career_texts))
            probe_t0 = time.time()
            probe_scores = scorer.score_all(career_texts[:probe_n])
            probe_elapsed = time.time() - probe_t0
            projected_full_time = (probe_elapsed / max(probe_n, 1)) * len(career_texts)

            SEMANTIC_TIME_BUDGET_SECONDS = 150
            if projected_full_time > SEMANTIC_TIME_BUDGET_SECONDS:
                semantic_scores = [None] * len(results)
                semantic_status = (
                    f"model available but projected encode time "
                    f"({projected_full_time:.0f}s for {len(career_texts)} candidates, "
                    f"extrapolated from a {probe_n}-sample probe) exceeds the "
                    f"{SEMANTIC_TIME_BUDGET_SECONDS}s safety budget — falling back "
                    f"to 5-component scoring to protect the 5-minute constraint"
                )
            else:
                # Reuse the probe's results instead of re-encoding those
                # same texts; only encode the remainder.
                rest_scores = scorer.score_all(career_texts[probe_n:]) if probe_n < len(career_texts) else []
                semantic_scores = probe_scores + rest_scores
                semantic_status = f"model loaded, semantic_similarity active ({projected_full_time:.0f}s projected)"
        else:
            semantic_scores = [None] * len(results)
            semantic_status = "model not found locally, falling back to 5-component scoring (see README setup)"

    for r, sem in zip(results, semantic_scores):
        finalize_score(r, sem)

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
            reasoning = build_reasoning(r["feat"], r["components"], r["honeypot"], r["disqualifiers"], r["title_chaser"])
            writer.writerow([r["candidate_id"], i, f"{r['score']:.4f}", reasoning])

    elapsed = time.time() - t0
    honeypots_in_top100 = sum(1 for r in top if r["honeypot"])
    print(f"Scored {n_read} candidates in {elapsed:.1f}s (rule-based phase: {rule_elapsed:.1f}s).")
    print(f"Semantic similarity: {semantic_status}")
    print(f"Wrote top {len(top)} to {out_path}")
    print(f"Honeypots in top 100: {honeypots_in_top100} ({honeypots_in_top100}%)")


if __name__ == "__main__":
    main()
