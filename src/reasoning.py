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

    # Location — branch on actual country, not on the score, so the text can
    # never claim "outside India" for an India-based candidate (a bug we
    # caught in review: the score and the country are independent facts).
    loc = feat.location
    is_india = feat.country.lower() == "india"
    if scores["location"] >= 0.85:
        parts.append(f"based in {loc} (on-site target city)")
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
