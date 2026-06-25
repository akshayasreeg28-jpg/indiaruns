"""
Scoring logic for the Redrob Senior AI Engineer (Founding Team) ranking challenge.

Design philosophy (see README.md for full writeup):
- Pure rule-based + lexical scoring. No embeddings, no LLM calls, no GPU.
  Everything here is O(1) dict lookups / string matching per candidate, so
  100K candidates score in low single-digit seconds on a laptop CPU.
- Every component score is interpretable, which is what lets us write
  non-templated, honest reasoning strings per candidate, and what lets a
  human (us, at the Stage 5 interview) actually defend *why* a candidate
  ranked where they did.
- The scoring is deliberately title/career-history led, not skills-keyword
  led. The JD and the trap design in sample_submission.csv both signal that
  keyword-stuffing (an HR Manager listing 9 AI skills) is the #1 trap to
  avoid. Title and career narrative are much harder to fake convincingly
  than a skills list.
"""

from __future__ import annotations
import re
from datetime import date, datetime
from dataclasses import dataclass, field

TODAY = date(2026, 6, 25)  # dataset's "current" reference date (last_active_date values cluster around early/mid 2026)

# ---------------------------------------------------------------------------
# Reference vocab derived from job_description.docx
# ---------------------------------------------------------------------------

CORE_TITLES = {
    "ai engineer", "senior ai engineer", "ml engineer", "machine learning engineer",
    "applied scientist", "research engineer", "ai research engineer",
    "data scientist", "senior data scientist", "nlp engineer",
    "recommendation systems engineer", "applied ml engineer", "search engineer",
    "ranking engineer", "mlops engineer",
}

ADJACENT_TITLES = {
    "software engineer", "senior software engineer", "backend engineer",
    "data engineer", "analytics engineer", "research scientist",
}

# Things the JD says are NOT a fit regardless of skills list
CV_SPEECH_ROBOTICS_ONLY_SKILLS = {
    "computer vision", "image classification", "object detection", "yolo", "opencv",
    "speech recognition", "tts", "robotics", "slam", "lidar", "ros",
}
NLP_IR_SKILLS = {
    "nlp", "rag", "retrieval", "vector search", "semantic search", "embeddings",
    "sentence transformers", "bge", "e5", "learning to rank", "ranking",
    "search", "information retrieval", "llm", "fine-tuning llms", "lora", "qlora", "peft",
}

# "Things you absolutely need" - production retrieval/vector-db/eval signals
EMBEDDING_RETRIEVAL_SKILLS = {
    "sentence transformers", "openai embeddings", "bge", "e5", "embeddings",
    "semantic search", "vector search", "rag",
}
VECTOR_DB_SKILLS = {
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "pgvector",
}
EVAL_FRAMEWORK_SKILLS = {
    "ndcg", "mrr", "map", "a/b testing", "learning to rank", "offline evaluation",
}
LLM_FT_SKILLS = {"lora", "qlora", "peft", "fine-tuning llms"}
LTR_SKILLS = {"learning to rank", "xgboost"}

CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "cognizant",
    "accenture", "capgemini",
}

TARGET_LOCATIONS = {"pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr", "gurugram", "gurgaon"}

LANGCHAIN_ONLY_MARKERS = {"langchain"}


def _lower_set(items):
    return {str(x).lower().strip() for x in items}


@dataclass
class CandidateFeatures:
    candidate_id: str
    title: str
    title_lower: str
    yoe: float
    location: str
    country: str
    skills: set
    skill_meta: dict  # name_lower -> {proficiency, duration_months, endorsements}
    career_titles: list
    career_companies: list
    career_descriptions: str
    redrob: dict
    profile: dict
    honeypot_flags: list = field(default_factory=list)


def extract_features(c: dict) -> CandidateFeatures:
    p = c["profile"]
    skills = c.get("skills", [])
    skill_meta = {}
    for s in skills:
        name = s["name"].lower().strip()
        skill_meta[name] = {
            "proficiency": s.get("proficiency", "beginner"),
            "duration_months": s.get("duration_months", 0),
            "endorsements": s.get("endorsements", 0),
        }
    career = c.get("career_history", [])
    return CandidateFeatures(
        candidate_id=c["candidate_id"],
        title=p.get("current_title", ""),
        title_lower=p.get("current_title", "").lower().strip(),
        yoe=p.get("years_of_experience", 0.0),
        location=p.get("location", ""),
        country=p.get("country", ""),
        skills=set(skill_meta.keys()),
        skill_meta=skill_meta,
        career_titles=[ch.get("title", "") for ch in career],
        career_companies=[ch.get("company", "") for ch in career],
        career_descriptions=" ".join(ch.get("description", "") for ch in career).lower(),
        redrob=c.get("redrob_signals", {}),
        profile=p,
    )


# ---------------------------------------------------------------------------
# Honeypot / data-integrity checks
# ---------------------------------------------------------------------------

def detect_honeypot(c: dict, feat: CandidateFeatures) -> list[str]:
    """Return a list of honeypot reason strings; empty list = looks legit."""
    reasons = []

    # 1. "Expert" proficiency claimed with 0 months of use.
    for name, meta in feat.skill_meta.items():
        if meta["proficiency"] == "expert" and meta.get("duration_months", 0) == 0:
            reasons.append(f"expert-claimed/0mo:{name}")

    # 2. years_of_experience grossly inconsistent with summed career_history duration.
    total_months = sum(ch.get("duration_months", 0) for ch in c.get("career_history", []))
    total_years = total_months / 12.0
    if feat.yoe > 0 and abs(total_years - feat.yoe) > 2.5:
        reasons.append(f"yoe-mismatch:stated={feat.yoe},history={round(total_years,1)}")

    # 3. Any single career entry implausibly long relative to total stated YOE
    #    (e.g. one role alone exceeds stated total experience).
    for ch in c.get("career_history", []):
        dm = ch.get("duration_months", 0)
        if feat.yoe > 0 and (dm / 12.0) > feat.yoe + 1:
            reasons.append("single-role-exceeds-total-yoe")
            break

    return reasons


# ---------------------------------------------------------------------------
# Hard disqualifiers (per JD's explicit "things we explicitly do NOT want" + bands)
# ---------------------------------------------------------------------------

def hard_disqualifiers(feat: CandidateFeatures) -> list[str]:
    reasons = []

    # Consulting-only career (every employer is a listed consulting firm)
    if feat.career_companies:
        companies_lower = [comp.lower() for comp in feat.career_companies]
        if all(any(cf in comp for cf in CONSULTING_FIRMS) for comp in companies_lower):
            reasons.append("consulting_only_career")

    # CV/Speech/Robotics-only without NLP/IR exposure
    has_cv_speech = any(s in feat.skills for s in CV_SPEECH_ROBOTICS_ONLY_SKILLS)
    has_nlp_ir = any(s in feat.skills for s in NLP_IR_SKILLS)
    if has_cv_speech and not has_nlp_ir:
        cv_count = sum(1 for s in feat.skills if s in CV_SPEECH_ROBOTICS_ONLY_SKILLS)
        if cv_count >= 2:
            reasons.append("cv_speech_robotics_without_nlp")

    return reasons


# ---------------------------------------------------------------------------
# Component scores (each returns 0.0-1.0)
# ---------------------------------------------------------------------------

def score_title_fit(feat: CandidateFeatures) -> float:
    t = feat.title_lower
    if t in CORE_TITLES:
        return 1.0
    if any(core in t for core in CORE_TITLES):
        return 0.9
    if t in ADJACENT_TITLES or any(adj in t for adj in ADJACENT_TITLES):
        # adjacent titles only count if career narrative shows ML/retrieval work
        if any(k in feat.career_descriptions for k in
               ("machine learning", "ml model", "embedding", "retrieval", "ranking", "recommendation")):
            return 0.55
        return 0.25
    return 0.05


def score_career_substance(feat: CandidateFeatures) -> float:
    """
    Reward evidence of having *built* the things the JD cares about, found in
    career_history descriptions, not just the skills list. This is the
    component meant to catch Tier-5 candidates who may not use buzzwords
    but clearly shipped a relevant system, and to discount keyword-stuffers
    who list skills without supporting narrative.
    """
    desc = feat.career_descriptions
    score = 0.0
    signals = [
        ("embedding", 0.18), ("vector search", 0.18), ("retrieval", 0.15),
        ("ranking", 0.15), ("recommendation", 0.12), ("search", 0.08),
        ("production", 0.10), ("deployed", 0.08), ("scale", 0.06),
        ("a/b test", 0.10), ("evaluation", 0.06), ("nlp", 0.06),
    ]
    for kw, w in signals:
        if kw in desc:
            score += w
    return min(score, 1.0)


def score_skills_with_trust(feat: CandidateFeatures) -> float:
    """
    Skills score, but discounted by an endorsement+duration "trust" factor so
    that listing a skill with 0 endorsements / 0 duration counts for much
    less than a skill backed by months of use and peer endorsement. This is
    the direct counter to the keyword-stuffer trap (see redrob_signals_doc /
    README: 'Keyword stuffers' are an explicitly named trap category).
    """
    def trust(name):
        meta = feat.skill_meta.get(name)
        if not meta:
            return 0.0
        dur = meta.get("duration_months", 0)
        end = meta.get("endorsements", 0)
        prof = meta.get("proficiency", "beginner")
        prof_w = {"beginner": 0.4, "intermediate": 0.65, "advanced": 0.85, "expert": 1.0}.get(prof, 0.4)
        dur_w = min(dur / 24.0, 1.0)  # 2 years = full credit
        end_w = min(end / 10.0, 1.0)  # 10 endorsements = full credit
        # an "expert" skill with no duration/endorsement backing is heavily discounted
        return prof_w * (0.5 + 0.3 * dur_w + 0.2 * end_w)

    embed_score = max((trust(s) for s in EMBEDDING_RETRIEVAL_SKILLS if s in feat.skills), default=0.0)
    vecdb_score = max((trust(s) for s in VECTOR_DB_SKILLS if s in feat.skills), default=0.0)
    eval_score = max((trust(s) for s in EVAL_FRAMEWORK_SKILLS if s in feat.skills), default=0.0)
    ft_score = max((trust(s) for s in LLM_FT_SKILLS if s in feat.skills), default=0.0)
    ltr_score = max((trust(s) for s in LTR_SKILLS if s in feat.skills), default=0.0)
    python_score = trust("python") if "python" in feat.skills else 0.0

    # weighted per JD: embeddings+vecdb+python+eval are "absolutely need"; ft/ltr are "nice to have"
    must_have = 0.32 * embed_score + 0.28 * vecdb_score + 0.20 * python_score + 0.20 * eval_score
    nice_to_have = 0.5 * ft_score + 0.5 * ltr_score
    return min(0.85 * must_have + 0.15 * nice_to_have, 1.0)


def score_experience_band(feat: CandidateFeatures) -> float:
    yoe = feat.yoe
    if 5 <= yoe <= 9:
        return 1.0
    if 4 <= yoe < 5 or 9 < yoe <= 11:
        return 0.7
    if 3 <= yoe < 4 or 11 < yoe <= 13:
        return 0.4
    return 0.15


def score_location(feat: CandidateFeatures) -> float:
    loc = feat.location.lower()
    country = feat.country.lower()
    if any(t in loc for t in ("pune", "noida")):
        return 1.0
    if any(t in loc for t in TARGET_LOCATIONS):
        return 0.85
    if country == "india":
        willing = feat.redrob.get("willing_to_relocate", False)
        return 0.6 if willing else 0.4
    # outside India: JD says case-by-case, no visa sponsorship
    willing = feat.redrob.get("willing_to_relocate", False)
    return 0.2 if willing else 0.05


def score_behavioral_multiplier(feat: CandidateFeatures) -> float:
    """
    Multiplicative modifier in [0.3, 1.05] reflecting whether a paper-perfect
    candidate is actually reachable/available. Mirrors the JD's explicit
    instruction to down-weight disengaged candidates.
    """
    r = feat.redrob
    if not r:
        return 0.7

    try:
        last_active = datetime.strptime(r["last_active_date"], "%Y-%m-%d").date()
        days_inactive = (TODAY - last_active).days
    except Exception:
        days_inactive = 9999

    recency_w = 1.0 if days_inactive <= 30 else (0.85 if days_inactive <= 90 else (0.6 if days_inactive <= 180 else 0.35))
    response_w = 0.5 + 0.5 * min(r.get("recruiter_response_rate", 0.0), 1.0)
    open_w = 1.05 if r.get("open_to_work_flag") else 0.85
    interview_w = 0.7 + 0.3 * min(r.get("interview_completion_rate", 0.5), 1.0)
    verified_w = 1.0 if (r.get("verified_email") and r.get("verified_phone")) else 0.9

    mult = recency_w * response_w * open_w * interview_w * verified_w
    return max(0.3, min(mult, 1.15))
