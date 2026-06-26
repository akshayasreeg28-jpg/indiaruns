# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

Team **TechNest** (Akshaya Sree G, Ponguru Aasrith Sairam). Ranks the top 100 candidates from a 100,000-candidate pool
against the **Senior AI Engineer — Founding Team** job description, with
per-candidate reasoning.

## Quick start

```bash
pip install -r requirements.txt
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv
```

Runs in **~13–35 seconds** and **~1.3 GB peak RAM** on a 100,000-row pool —
CPU-only, no network, no GPU. (Constraint budget: 5 min / 16 GB.) This works
immediately, no setup, and is what `submission.csv` in this repo was
generated with.

## Optional: semantic similarity component

The base mode above scores `career_substance` by phrase/keyword matching
against career history text. That's a deliberate, defensible choice (see
"Why rule-based" below) but it has one real limitation: a candidate who
describes the JD's exact work in different words ("owned the personalization
stack rewrite" vs. "built a recommendation system") can be under-scored.
`src/semantic.py` adds an optional 6th component — real semantic similarity
between the JD and each candidate's career narrative, via a small local
sentence-transformers model (`all-MiniLM-L6-v2`, ~80MB) — to address this.

**This component is fully optional and safe by construction.** If the model
isn't set up, `rank.py` detects that automatically, prints a clear status
line, and falls back to the original 5-component scoring with weights
renormalized to sum to 1.0 — same output as if this feature didn't exist.
No exception, no crash, no silent network call.

One-time setup (run once, requires internet — this download happens
*before* ranking, not during the timed run):

```bash
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('./models/all-MiniLM-L6-v2')"
```

After that, `rank.py` runs exactly as before, fully offline, and the
console output will say `semantic_similarity active` instead of `model not
found locally`. Use `--no-semantic` to force the original 5-component
behavior even if the model is cached.

**Honest disclosure: we could not benchmark this ourselves end-to-end.**
Our own development sandbox's network allowlist blocks both huggingface.co
and GitHub's release-asset CDN, so the model could never actually be
downloaded there — we wrote and unit-tested the integration logic (weight
switching, fallback behavior, the time-budget safety probe) with mocked
semantic scores, and confirmed the fallback path produces byte-identical
output to the original 5-component version, but we have not run the real
model against the full 100K pool ourselves. `rank.py` includes a time-budget
probe (encodes a 500-candidate sample first, extrapolates, and aborts back
to the fallback path if projected full-pool time would exceed a 150-second
safety budget) specifically because we couldn't verify real throughput and
wanted a guard against blowing the 5-minute constraint on an unknown
machine. If you run this with the model installed, please sanity-check the
printed timing against `submission.csv`'s reproducibility before relying on
it for the actual submission.

There are no required dependencies beyond the Python standard library for
the base mode — `requirements.txt`'s sentence-transformers/torch lines are
commented out and optional, only needed for the semantic component.

## What this does

1. Streams `candidates.jsonl` line by line (constant memory regardless of pool size).
2. For each candidate, extracts structured features from `profile`, `career_history`,
   `skills`, and `redrob_signals`.
3. Runs two independent integrity checks, two hard-disqualifier checks, and a
   softer title-chasing check (see below).
4. Computes 5 interpretable rule-based component scores. If the optional
   semantic model is set up (see above), batch-encodes every candidate's
   career narrative against the JD for a 6th component.
5. Combines components into a base fit score, then applies a multiplicative
   behavioral-availability modifier and any penalties.
6. Sorts, breaks ties by `candidate_id` ascending, writes the top 100 with a
   real, candidate-specific reasoning string for each row.

## Why rule-based first, with an optional semantic layer

The JD explicitly states the role wants someone who can ship something
"obviously suboptimal" quickly and reason about latency/quality tradeoffs,
and the compute constraints (CPU-only, 5 min, no network *during ranking*)
rule out per-candidate LLM calls outright for a 100K pool. Given that, a
transparent, debuggable, rule-based system has three real advantages over a
black-box embedding similarity score for *this specific JD*:

- **The JD is mostly disqualifier logic, not similarity.** "We will not move
  forward if X" is much easier to express as an explicit rule than to hope a
  cosine-similarity score happens to encode it.
- **Reasoning quality is scored at Stage 4.** A rule-based score decomposes
  cleanly into "why" (title fit was X, skill trust was Y, behavioral modifier
  was Z) — which is what let us write honest, non-templated, hallucination-free
  reasoning per row instead of an LLM post-hoc rationalizing a black-box score.
- **It is the most defensible architecture in a Stage 5 interview.** Every
  score is traceable to a specific field in the candidate JSON. There is no
  "the model decided this" step we couldn't explain.

The known tradeoff with a purely phrase-based approach is that it can miss
paraphrased synonyms ("led the recommendation system rewrite" vs. "owned
ranking architecture"). We addressed this with `src/semantic.py` — an
**optional** 6th component using a small local sentence-transformers model
(see "Optional: semantic similarity component" above) — rather than leaving
it as a stated limitation. It's optional and additive, not a replacement,
specifically so the three advantages above still hold for the 5 rule-based
components even when the semantic layer is active: the semantic score is
disclosed in the reasoning text as a real number, gets a modest 0.15 weight
so it can't single-handedly override the rule-based signals, and the whole
system degrades safely to the original 5-component version if the model
isn't set up — see the disclosure in that section about what we could and
couldn't verify ourselves.

## Scoring architecture

| Component | Weight (no semantic / with semantic) | What it captures |
|---|---|---|
| `title_fit` | 0.32 / 0.28 | Is the current title actually AI/ML/data, not an adjacent or unrelated role with AI keywords bolted onto the skills list. Heaviest weight — this is the direct defense against the "keyword stuffer" trap named in the hackathon README. Three tiers below "core": adjacent titles (Software/Data/Backend/Analytics Engineer, Data Analyst), AI-titled-but-wrong-specialization (AI Specialist, Computer Vision Engineer — real ML work, just not retrieval/NLP), and everything else. |
| `career_substance` | 0.23 / 0.20 | Keyword/phrase hits inside `career_history[].description` (e.g. "embedding", "retrieval", "production", "A/B test") — rewards candidates who *describe having built* the JD's core systems, not just listed them as skills. |
| `skills_trust` | 0.23 / 0.20 | Skills score discounted by an endorsement+duration "trust" factor, so a skill claimed at `expert` with 0 months used / 0 endorsements counts far less than the same skill backed by real tenure. Weighted toward the JD's "things you absolutely need" (embeddings, vector DB, Python, eval frameworks) over "nice to have" (fine-tuning, learning-to-rank). |
| `location` | 0.10 / 0.07 | Full credit for Pune/Noida (named office cities), high credit for the JD's 4 other named "welcome" cities, a distinct Tier-1-Indian-city tier (Bangalore, Chennai, Kolkata, Ahmedabad — the JD separately invites "Tier-1 Indian city" relocators, not just the 4 named cities), reduced for elsewhere in India or abroad scaled by `willing_to_relocate`, since the JD does not sponsor visas. |
| `semantic_similarity` | n/a / 0.15 | **Optional.** Cosine similarity between a sentence-transformers embedding of the JD's technical summary and each candidate's `career_history` description text. Only active if the local model is set up (see above); falls back cleanly otherwise. |
| `experience_band` | 0.12 / 0.10 | Soft-scored around the JD's 5–9yr range; full credit in-band, graduated partial credit outside it. Floor is 0.35, not a near-zero — the JD explicitly says this "is a range, not a requirement" and it will "seriously consider candidates outside the band if other signals are strong," so this component is deliberately kept from dominating the score for an otherwise excellent candidate. |

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

### Soft penalty: title-chasing (×0.55 penalty)
The JD separately names career trajectories that show someone optimizing for
"Senior → Staff → Principal" by switching companies every ~1.5 years.
`detect_title_chasing()` flags a monotonically escalating seniority ladder
across 3+ roles with ≤18-month average tenure. This gets a milder penalty
(0.55×) than the two hard disqualifiers above, matching the JD's softer
"we're not a fit" phrasing for this criterion versus "we will not move
forward" for consulting-only/CV-only careers.

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

**Calibration note.** Both thresholds were chosen by inspecting the actual
distribution, not guessed: the YOE-vs-career-history mismatch has a clean
gap (99.95% of the pool sits under 0.45 years of natural rounding noise,
then jumps straight to multi-year mismatches with no gray zone), and the
"expert proficiency, 0 months used" pattern has an even sharper cliff (the
next-lowest duration value after exactly 0 is 38 months — no 1-12 month
gray zone exists). 68 unique candidates are flagged in the full 100K pool
(0 overlap between the two rules), against the spec's "~80" estimate. Before
accepting that gap, we checked for additional honeypot categories: date-logic
violations (end-before-start, overlapping roles, future dates — found: 0
across the entire pool), education-timeline inconsistencies (found: large
numbers of false positives from legitimate multi-degree/part-time-study
timelines, not a real signal), salary-range `min > max` (found: ~19% of the
pool — too widespread to be a deliberate honeypot, looks like a dataset-wide
field-order quirk instead), and company-founding-date violations (found:
structurally impossible in this dataset — every one of the 63 distinct
employer names in the pool has an internally consistent earliest-start-date
floor across thousands of samples, so "N years at a company founded N-3
years ago" can't occur here). We also checked whether the keyword-stuffer
trap ever leaks into `career_history.description` text rather than just the
`skills` array — it doesn't, in any of the candidates we checked who have
non-technical titles; descriptions stay consistent with the stated role even
when skills are stuffed. We're treating the 68-vs-~80 gap as normal slack in
a rounded spec estimate rather than a missed pattern, but we're flagging the
investigation here in case Stage 5 wants to probe it further.

## Title classification audit

The dataset uses a **fixed, finite vocabulary of 49 distinct job titles**
(not freeform text), which let us check `score_title_fit` exhaustively
against every title actually present rather than guessing at coverage. This
caught a real bug: "AI Specialist" (130 candidates) and "Computer Vision
Engineer" (132 candidates) — both genuine AI/ML titles — were falling
through to the same 0.05 floor as "Civil Engineer" and "HR Manager", because
neither title contains any of our `CORE_TITLES` as a substring. Manual
inspection confirmed real candidates were affected: e.g. an "AI Specialist"
at Mad Street Den with Pinecone, Semantic Search, Vector Search, and
Information Retrieval in their skills list was scoring identically to a
keyword-stuffer. Fixed by adding an explicit `AI_ADJACENT_WRONG_SPECIALIZATION_TITLES`
tier (0.40 base, 0.65 if career history shows retrieval/NLP work despite the
title) and adding "Data Analyst" to `ADJACENT_TITLES` (it was inconsistently
excluded while "Data Engineer" and "Analytics Engineer" were included).
Result: 3 of the 100 final rankings changed, all three new entrants verified
to be genuine embeddings/vector-search practitioners; all three displaced
candidates were sitting at the rank 98–100 boundary with near-identical
scores, not collateral damage.

We also explicitly checked the reverse risk — whether gating the *adjacent*
tier's bonus credit on career-history keywords (rather than just skills)
was unfairly burying strong candidates whose skills list looks good but
whose career narrative doesn't mention it. Checked all 142 "Senior Software
Engineer (ML)" candidates: 91 have embeddings/vector-DB skills listed but
stay at the lower tier. Manually inspected several — in every case checked,
their actual career-history description was computer vision, fraud
detection, or time-series forecasting work, with the relevant skill present
only as a listed keyword, not work they describe having done. This matches
the keyword-stuffer pattern (skill claimed, not evidenced) rather than a
true negative, so no change was made here — the gate is working as intended.

## Location and experience-band review

The dataset's location field is also a small fixed vocabulary (18 Indian
cities, 8 countries) — checked it exhaustively the same way as titles. Found
that **Bangalore and Chennai** (4,238 and 4,164 candidates respectively, both
unambiguously Tier-1 Indian metros) were falling into the same generic
"India" bucket as smaller cities like Trivandrum or Bhubaneswar, even though
the JD separately and explicitly says it's *"open to relocation candidates
from Tier-1 Indian cities"* — a broader invitation than just the 4 cities it
names directly (Hyderabad, Pune, Mumbai, Delhi NCR). Added a distinct
Tier-1-city tier (Bangalore, Chennai, Kolkata, Ahmedabad) between the named
welcome cities and the generic-India fallback. Result: 4/100 rankings
changed, including a Recommendation Systems Engineer at Zomato (FAISS,
embeddings, semantic search) who'd been undervalued purely on a Chennai
address.

Separately, re-read the JD's own framing on experience — *"this is a range,
not a requirement... we'll seriously consider candidates outside the band if
other signals are strong"* — and concluded the `experience_band` component's
original 0.15 floor was too punishing relative to that explicit
anti-credentialist language: at this component's 0.12 weight, the gap
between in-band and the floor was eating roughly a fifth of the typical
top-100 competitive score spread, on a criterion the JD itself says
shouldn't be a hard filter. Raised the floor to 0.35. This was a real
principle fix but turned out to be a no-op on the current top 100 in this
run — the one strong 16+ year candidate we found who'd plausibly benefit
(a Senior AI Engineer-equivalent at Flipkart, real ranking-system career
history, strong skills) is still correctly excluded because their
`open_to_work_flag` is `False`, which is a genuine availability signal we
intentionally don't override.

**Two enhancement ideas we considered and explicitly rejected:**
- *Using `current_industry`* (e.g. "AI/ML", "Conversational AI" tags) as a
  new scoring component. Checked it against the existing components first:
  the 40 candidates with an AI-native industry tag but low `title_fit` are
  the same population already correctly identified via career-history
  inspection as doing CV/fraud-detection/forecasting work, not retrieval —
  industry tag didn't add independent signal beyond what title and career
  substance already capture, so we didn't add the complexity.
- *A dedicated "early-stage-startup-hopper" rule*, since the JD explicitly
  says people who've only bounced between early-stage startups avoiding
  process/structure aren't a fit. Checked the data: only 9 candidates in
  the entire 100K pool have 3+ roles entirely at sub-50-person companies,
  none with AI/ML titles. Too rare to matter for this dataset and would be
  dead code, so we documented the check instead of building the rule.

## Honest limitations (what we'd improve with more time)

- **The semantic component is implemented but unverified end-to-end by us.**
  `career_substance` alone is keyword/phrase-based, so a candidate who
  describes the exact same work with very different vocabulary could be
  under-scored on that component specifically. We addressed this with the
  optional `src/semantic.py` layer rather than leaving it as a stated gap —
  but we could not download or run the actual sentence-transformers model
  in our own development sandbox (network allowlist blocks both
  huggingface.co and GitHub's release-asset CDN), so the integration logic
  is unit-tested with mocked scores, not validated against the real model
  on the real 100K pool by us. If you run it with the model installed, the
  console's projected-time output is worth checking before trusting it for
  the final submission.
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



## Repository layout

```
.
├── rank.py                       # entry point
├── src/
│   ├── scoring.py                 # feature extraction + component scores + honeypot/disqualifier logic
│   ├── reasoning.py               # per-row reasoning string builder
│   └── semantic.py                # optional semantic-similarity component (local sentence-transformers)
├── models/
│   └── all-MiniLM-L6-v2/          # NOT committed — created by the one-time setup step, see "Optional" above
├── data/
│   ├── candidates.jsonl           # full 100K pool (as released) — NOT committed, see "Data" below
│   └── candidate_schema.json
├── hf_space/                      # standalone HuggingFace Spaces sandbox demo (same scoring code, 80-candidate sample)
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

A third, more substantive bug was caught by sanity-checking
`score_skills_with_trust` against synthetic boundary cases: the original
trust formula gave a guaranteed 0.5 floor to *any* claimed proficiency level
regardless of duration or endorsement evidence, so a fabricated "expert,
0 months used, 0 endorsements" skill scored *higher* (0.255) than a
genuinely-evidenced but lower-proficiency skill like a real Python expert
(0.17) — the opposite of what the function exists to do, since it's the
main defense against keyword-stuffing within the skills list itself. Fixed
by making the evidence factor multiplicative with no guaranteed floor (a
0-duration, 0-endorsement claim now scores ~0.05 regardless of claimed
proficiency, vs. ~0.17 for the genuinely-evidenced comparison). This changed
2 of the 100 final rankings, both swaps at the rank 94–100 boundary, both
replacements verified to have stronger real evidence.

A fourth addition (not a bug fix, a missing rule): the JD explicitly names
"title-chasers" — candidates whose career shows an escalating
Senior→Staff→Principal ladder via company-hopping every ~1.5 years — as a
deliberate non-fit. `detect_title_chasing()` checks for monotonically
increasing seniority titles with ≤18-month average tenure across 3+ roles,
applying a milder penalty (0.55×) than the hard disqualifiers, matching the
JD's softer "we're not a fit" language for this criterion vs. "we will not
move forward" for consulting-only/CV-only. It fires on 44 of the 100,000
candidates but doesn't change the top 100 in this run, since none of the
44 happened to also clear the title/skills bar — it's there as a tested
safety net, not because it was needed to hit this particular result.
