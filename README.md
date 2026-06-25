# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

Solo submission. Ranks the top 100 candidates from a 100,000-candidate pool
against the **Senior AI Engineer — Founding Team** job description, with
per-candidate reasoning.

## Quick start

```bash
pip install -r requirements.txt
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv
```

Runs in **~13 seconds** and **~1.3 GB peak RAM** on a 100,000-row pool —
CPU-only, no network, no GPU. (Constraint budget: 5 min / 16 GB.)

There are no dependencies beyond the Python standard library — `requirements.txt`
is intentionally empty/minimal. This was a deliberate choice (see "Why no
embeddings model" below).

## What this does

1. Streams `candidates.jsonl` line by line (constant memory regardless of pool size).
2. For each candidate, extracts structured features from `profile`, `career_history`,
   `skills`, and `redrob_signals`.
3. Runs two independent integrity checks and two hard-disqualifier checks (see below).
4. Computes 5 interpretable component scores, combines them into a base fit score,
   then applies a multiplicative behavioral-availability modifier and any penalties.
5. Sorts, breaks ties by `candidate_id` ascending, writes the top 100 with a
   real, candidate-specific reasoning string for each row.

## Why rule-based, not embeddings/an LLM

The JD explicitly states the role wants someone who can ship something
"obviously suboptimal" quickly and reason about latency/quality tradeoffs,
and the compute constraints (CPU-only, 5 min, no network) rule out per-candidate
LLM calls outright for a 100K pool. Given that, a transparent, debuggable,
rule-based system has three real advantages over a black-box embedding
similarity score for *this specific JD*:

- **The JD is mostly disqualifier logic, not similarity.** "We will not move
  forward if X" is much easier to express as an explicit rule than to hope a
  cosine-similarity score happens to encode it.
- **Reasoning quality is scored at Stage 4.** A rule-based score decomposes
  cleanly into "why" (title fit was X, skill trust was Y, behavioral modifier
  was Z) — which is what let us write honest, non-templated, hallucination-free
  reasoning per row instead of an LLM post-hoc rationalizing a black-box score.
- **It is the most defensible architecture in a Stage 5 interview.** Every
  score is traceable to a specific field in the candidate JSON. There is no
  "the model decided this" step that I, personally, couldn't explain.

The tradeoff (and we're explicit about this rather than pretending otherwise):
a learned ranker over precomputed local embeddings (e.g. a small sentence-transformers
model run once, offline, to build a similarity feature, separate from any
per-candidate LLM call) would likely do better at catching paraphrased synonyms
("led the recommendation system rewrite" vs. "owned ranking architecture")
that this keyword/phrase-based version can miss. That's the most natural next
iteration; see "Honest limitations" below.

## Scoring architecture

| Component | Weight | What it captures |
|---|---|---|
| `title_fit` | 0.32 | Is the current title actually AI/ML/data, not an adjacent or unrelated role with AI keywords bolted onto the skills list. Heaviest weight — this is the direct defense against the "keyword stuffer" trap named in the hackathon README. |
| `career_substance` | 0.23 | Keyword/phrase hits inside `career_history[].description` (e.g. "embedding", "retrieval", "production", "A/B test") — rewards candidates who *describe having built* the JD's core systems, not just listed them as skills. |
| `skills_trust` | 0.23 | Skills score discounted by an endorsement+duration "trust" factor, so a skill claimed at `expert` with 0 months used / 0 endorsements counts far less than the same skill backed by real tenure. Weighted toward the JD's "things you absolutely need" (embeddings, vector DB, Python, eval frameworks) over "nice to have" (fine-tuning, learning-to-rank). |
| `experience_band` | 0.12 | Soft scoring around the JD's 5–9 yr band, full credit in-band, partial credit just outside it (the JD itself says the band is "a range, not a requirement"). |
| `location` | 0.10 | Full credit for Pune/Noida (named office cities), partial for other JD-named "welcome" cities, reduced for elsewhere in India or abroad scaled by `willing_to_relocate`, since the JD does not sponsor visas. |

`final_score = (Σ weight·component) × behavioral_multiplier × honeypot_penalty × disqualifier_penalty`

### Behavioral multiplier (~0.3–1.15×)
Built from `redrob_signals`: recency of `last_active_date`, `recruiter_response_rate`,
`open_to_work_flag`, `interview_completion_rate`, verification flags. Directly
implements the JD's own instruction: *"a perfect-on-paper candidate who hasn't
logged in for 6 months and has a 5% recruiter response rate is, for hiring
purposes, not actually available. Down-weight them appropriately."*

### Hard disqualifiers (×0.15 penalty, not a hard zero)
- **Consulting-only career**: every employer in `career_history` is a listed
  consulting firm (TCS/Infosys/Wipro/Cognizant/Accenture/Capgemini), matching the
  JD's explicit "people who have only worked at consulting firms in their
  entire career" rejection criterion.
- **CV/Speech/Robotics-only without NLP/IR**: 2+ computer-vision/speech/robotics
  skills present with zero NLP/IR/retrieval skills, matching the JD's explicit
  "primary expertise is computer vision, speech, or robotics without
  significant NLP/IR exposure" rejection criterion.

We chose a strong penalty multiplier (0.15×) rather than a literal exclusion so
a candidate with an overwhelming raw fit score could theoretically still
surface — matching the JD's own hedge language ("we will *probably* not move
forward") rather than treating these as absolute the way honeypots are.

### Honeypot detection (×0.05 penalty)
Two independent integrity checks, found by direct inspection of the dataset
during development (see `notebooks/` — not included in repo, see commit history
for the exploration script):
1. A skill is marked `proficiency: expert` with `duration_months: 0`.
2. `years_of_experience` disagrees with the sum of `career_history[].duration_months`
   by more than 2.5 years.

Either flag triggers a near-zero (not literal-zero) penalty multiplier. We
chose a strong-but-nonzero penalty deliberately: a hard 0 score for a flagged
profile collapses to an arbitrary tie-break against other zero-scored
candidates, which is harder to defend than "this score is real but multiplied
down by a specific, named penalty." Result on the full 100K pool: **0 honeypots
in the top 100** (validated by cross-referencing against the honeypot IDs found
during manual inspection).

## Honest limitations (what I'd improve with more time)

- **Phrase matching beats true semantic understanding.** `career_substance`
  is keyword/phrase-based, so a candidate who describes the exact same work
  with very different vocabulary could be under-scored. The natural fix is a
  small, locally-run sentence-transformers model (still CPU-only, still no
  network) to get a JD–career-history similarity feature as an additional
  signal — this is the first thing I'd add given more time.
- **Disqualifier rules are necessarily approximate.** "Consulting-only career"
  and "CV/speech-only" are pattern matches against the JD's explicit language,
  not a learned model — they will have false positives/negatives at the margins.
- **No cross-validation against ground truth**, since none is available during
  the competition (per the spec, the leaderboard is hidden). All tuning here
  was done by hand-inspecting the dataset and the JD, not by fitting to a score.

## Data

`data/candidates.jsonl` (465 MB, 100,000 rows) is **not committed to this
repo** — it's listed in `.gitignore`. Reasons:
1. It exceeds GitHub's 100 MB hard file-size limit as plain JSONL.
2. The organizers already hold the canonical copy and will supply it inside
   the Stage 3 sandbox container for reproduction, per `submission_spec.md`.

To run locally, place the released `candidates.jsonl` (or `candidates.jsonl.gz`
— both are supported, see below) at `data/candidates.jsonl` before running
the reproduce command. `rank.py` accepts a `.gz` path directly:

```bash
python rank.py --candidates ./data/candidates.jsonl.gz --out ./submission.csv
```

We verified the `.gz` and uncompressed paths produce byte-identical output.



```
.
├── rank.py                       # entry point
├── src/
│   ├── scoring.py                # feature extraction + component scores + honeypot/disqualifier logic
│   └── reasoning.py               # per-row reasoning string builder
├── data/
│   ├── candidates.jsonl           # full 100K pool (as released)
│   └── candidate_schema.json
├── sandbox/
│   └── sample_run.py              # small-sample (<=100 candidates) demo for the hosted sandbox
├── validate_submission.py         # organizer-provided validator (unmodified)
├── submission.csv                 # final output
├── submission_metadata.yaml
└── requirements.txt
```

## AI tools used

Declared honestly in `submission_metadata.yaml`. Summary: Claude was used as
a development collaborator — exploring the dataset to find the honeypot
signatures, reviewing the scoring logic for bugs (it caught two real bugs
during review: a tie-break bug where rounding the score for the CSV after
sorting could produce ties the validator rejects, and a reasoning-text bug
that mislabeled an India-based candidate's location as "outside India"),
and structuring this README. No candidate data was sent to any external LLM
API as part of the ranking pipeline itself — `rank.py` makes zero network calls.
