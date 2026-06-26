"""
Generates the per-candidate `reasoning` string for the submission CSV.

Hard constraint from submission_spec.md Section 3 / Stage 4 review:
  - no empty reasoning
  - no identical reasoning across rows
  - no name-templating ("X is a great fit because X...")
  - no mention of skills/facts not actually in the candidate's profile
  - reasoning must not contradict the rank

So every clause below is built only from fields we actually read off the
candidate (title, company, YOE, specific skills present, specific signal
values, specific disqualifier hits). Nothing is invented. If a component
score is low, the reasoning says so explicitly with the real number behind it
(e.g. "response rate 0.06") rather than a generic phrase, which is what
keeps reasoning from becoming a template across hundreds of rows.
"""

from .scoring import (
    CandidateFeatures, EMBEDDING_RETRIEVAL_SKILLS, VECTOR_DB_SKILLS,
    EVAL_FRAMEWORK_SKILLS, LLM_FT_SKILLS, LTR_SKILLS,
    TARGET_LOCATIONS, TIER_1_INDIAN_CITIES,
)


def _present_skills(feat: CandidateFeatures, pool: set) -> list:
    return sorted(s for s in pool if s in feat.skills)


def build_reasoning(feat: CandidateFeatures, scores: dict, honeypot: list, disq: list, title_chaser: bool = False) -> str:
    parts = []

    # Lead with title + company + YOE (always true, always specific)
    parts.append(f"{feat.title} at {feat.profile.get('current_company', 'unknown company')}, "
                 f"{feat.yoe:.1f} yrs experience")

    # Concrete skill evidence (only list skills that are actually present)
    embed_hits = _present_skills(feat, EMBEDDING_RETRIEVAL_SKILLS)
    vecdb_hits = _present_skills(feat, VECTOR_DB_SKILLS)
    eval_hits = _present_skills(feat, EVAL_FRAMEWORK_SKILLS)
    ft_hits = _present_skills(feat, LLM_FT_SKILLS)

    if embed_hits or vecdb_hits:
        tech_bits = (embed_hits + vecdb_hits)[:3]
        parts.append(f"hands-on with {', '.join(tech_bits)}")
    elif scores["career_substance"] > 0.3:
        parts.append("career history describes retrieval/ranking-adjacent production work")
    else:
        parts.append("no direct embeddings/vector-DB evidence in skills or career history")

    if eval_hits:
        parts.append(f"evaluation exposure ({', '.join(eval_hits)})")

    if ft_hits:
        parts.append(f"LLM fine-tuning exposure ({', '.join(ft_hits)})")

    # Semantic similarity — only mentioned if the component was actually
    # computed for this run (i.e. the local model was available). Reported
    # as the real number, not a vague "high/low" label, consistent with how
    # every other quantitative signal here is disclosed.
    if "semantic_similarity" in scores:
        parts.append(f"JD semantic similarity {scores['semantic_similarity']:.2f}")

    # Location — branch on the actual city/country FACTS the score was
    # computed from (set membership), never on the score's numeric value.
    # We deliberately re-derive membership here rather than reusing
    # scores["location"] thresholds: a generic-India-with-relocation-flag
    # score (0.6) and a Tier-1-city-without-relocation-flag score (0.55)
    # sit right next to each other numerically but mean different things,
    # and branching on the number alone caused exactly this kind of
    # mislabeling bug once already (see the "outside India" bug fixed
    # earlier) -- caught a second instance of the same bug class here
    # during review: Bhubaneswar (not Tier-1) was being labeled "Tier-1
    # Indian city" purely because its willing-to-relocate score (0.6)
    # happened to exceed the Tier-1 tier's own score range.
    loc = feat.location
    loc_lower = loc.lower()
    is_india = feat.country.lower() == "india"
    is_named_welcome_city = any(t in loc_lower for t in TARGET_LOCATIONS)
    is_tier1_city = any(t in loc_lower for t in TIER_1_INDIAN_CITIES) and not is_named_welcome_city

    if is_named_welcome_city:
        parts.append(f"based in {loc} (on-site target city)")
    elif is_tier1_city:
        parts.append(f"based in {loc} (Tier-1 Indian city); "
                     f"relocation flag {'set' if feat.redrob.get('willing_to_relocate') else 'not set'}")
    elif is_india:
        parts.append(f"based in {loc}, India, not a primary target city; "
                     f"relocation flag {'set' if feat.redrob.get('willing_to_relocate') else 'not set'}")
    else:
        parts.append(f"based in {loc}, {feat.country} — outside India, "
                     f"relocation flag {'set' if feat.redrob.get('willing_to_relocate') else 'not set'} "
                     f"(JD does not sponsor visas)")

    # Behavioral / availability signal — use real numbers, not adjectives
    r = feat.redrob
    rr = r.get("recruiter_response_rate")
    la = r.get("last_active_date")
    otw = r.get("open_to_work_flag")
    if rr is not None and la:
        parts.append(f"response rate {rr:.2f}, last active {la}, open_to_work={otw}")

    # Disqualifier / honeypot disclosure — must be visible if present, since
    # hiding it would contradict a low rank, which Stage 4 explicitly checks.
    if disq:
        parts.append(f"flagged: {', '.join(disq)}")
    if title_chaser:
        parts.append("title-escalation pattern: short average tenure with rising seniority titles across roles")
    if honeypot:
        parts.append(f"DATA INTEGRITY FLAG ({honeypot[0]})")

    text = "; ".join(parts) + "."
    # keep it readable, not a wall of semicolons -> light cleanup
    return text[0].upper() + text[1:]
