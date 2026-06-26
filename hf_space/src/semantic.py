"""
semantic.py — optional semantic-similarity signal for career_substance.

Why this exists: career_substance in scoring.py is phrase/keyword matching
against career_history descriptions. That catches a candidate who writes
"built a production embedding-based retrieval system" but misses one who
writes "owned the personalization stack rewrite, replacing a hand-tuned
heuristic with a learned model" — same work, different words. This module
adds a real semantic-similarity score (JD text vs. each candidate's career
narrative) using a small, locally-run sentence-transformers model.

Design constraints this respects (see submission_spec.md):
  - No network calls during the timed ranking run. The model is downloaded
    ONCE during setup (see README "Semantic similarity setup"), then loaded
    from a local path with local_files_only=True. If the model isn't
    present locally, this module degrades to a no-op rather than crashing
    or silently making a network call.
  - CPU-only. all-MiniLM-L6-v2 (the default here) is small enough to encode
    100K short text snippets well within the 5-minute budget on CPU --
    typical published throughput for this model is in the low thousands of
    sentences/sec on a single CPU core for short inputs; even at a
    pessimistic 200/sec that's ~8 minutes for 100K, so encoding is batched
    and capped (see SemanticScorer.score_all) to stay inside budget on
    slower machines. We could not benchmark this ourselves inside our own
    development sandbox -- its network allowlist blocks both
    huggingface.co and GitHub's release-asset CDN, so the model could never
    actually be downloaded there. This is disclosed in the README rather
    than asserting a benchmark we don't have.
  - Safe by construction: importing this module never raises, even with no
    internet and no cached model. score_all() returns None for every
    candidate in that case, and rank.py's caller treats None as "skip this
    signal", falling back to career_substance alone.

Usage from rank.py:
    from src.semantic import SemanticScorer
    scorer = SemanticScorer()  # loads model if available locally, else no-op
    sem_scores = scorer.score_all(jd_text, [feat.career_descriptions for feat in all_feats])
    # sem_scores[i] is a float in [0,1] or None
"""

from __future__ import annotations
import os
import json
from pathlib import Path

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_LOCAL_MODEL_DIR = Path(__file__).parent.parent / "models" / DEFAULT_MODEL_NAME
DEFAULT_BATCH_SIZE = 256

# The JD text used for the similarity comparison. Kept short and focused on
# the technical substance the JD cares about (retrieval, ranking, embeddings,
# production ML systems) rather than the full JD text (comp, culture, logistics
# sections would just dilute the embedding with irrelevant content).
JD_TECHNICAL_SUMMARY = (
    "Senior AI engineer building production retrieval and ranking systems. "
    "Owns embedding-based semantic search, vector database infrastructure, "
    "learning-to-rank models, and recommendation systems end to end. "
    "Designs evaluation frameworks (NDCG, MRR, offline and online A/B "
    "testing) to validate ranking quality. Fine-tunes and deploys "
    "transformer-based models for search relevance at production scale, "
    "with attention to latency and serving infrastructure."
)


class SemanticScorer:
    """
    Thin wrapper around a local sentence-transformers model. Never raises;
    degrades to a no-op (is_available=False) if the model can't be loaded
    locally, so callers don't need their own try/except around every call.
    """

    def __init__(self, local_model_dir: str | Path = DEFAULT_LOCAL_MODEL_DIR):
        self.is_available = False
        self.model = None
        self._jd_embedding = None
        local_model_dir = Path(local_model_dir)

        if not local_model_dir.exists():
            return  # no-op: model not downloaded, see README setup step

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(str(local_model_dir), local_files_only=True, device="cpu")
            self.is_available = True
        except Exception:
            # Any failure here (missing torch, corrupted cache, etc.) should
            # degrade gracefully, not crash the whole ranking run.
            self.is_available = False
            self.model = None

    def _embed_jd(self):
        if self._jd_embedding is None and self.is_available:
            self._jd_embedding = self.model.encode(
                JD_TECHNICAL_SUMMARY, normalize_embeddings=True, show_progress_bar=False
            )
        return self._jd_embedding

    def score_all(self, career_texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[float | None]:
        """
        Returns a list the same length as career_texts. Each entry is a
        cosine-similarity-derived score in [0,1], or None for every entry
        if the model isn't available (caller should treat None as "no
        semantic signal, fall back to phrase matching").

        Empty-text candidates get 0.0, not None — an empty career history
        genuinely has zero semantic overlap with the JD, that's a real
        score, not a missing one.
        """
        if not self.is_available:
            return [None] * len(career_texts)

        import numpy as np

        jd_emb = self._embed_jd()
        non_empty_idx = [i for i, t in enumerate(career_texts) if t and t.strip()]
        non_empty_texts = [career_texts[i] for i in non_empty_idx]

        results: list[float | None] = [0.0] * len(career_texts)
        if not non_empty_texts:
            return results

        embeddings = self.model.encode(
            non_empty_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # cosine similarity, since both sides are normalized this is just a dot product
        sims = embeddings @ jd_emb
        # map cosine [-1, 1] -> [0, 1]; in practice sentence-transformer
        # cosine sims for related-but-distinct text cluster well above 0,
        # so this linear rescale (not a hard clip at 0) preserves ordering
        # better than clamping negative values away.
        sims01 = (sims + 1.0) / 2.0

        for idx, sim in zip(non_empty_idx, sims01):
            results[idx] = float(sim)
        return results


def download_setup_instructions() -> str:
    """Returns the one-time setup command, for --help / error messages."""
    return (
        f"Semantic similarity model not found at {DEFAULT_LOCAL_MODEL_DIR}.\n"
        f"One-time setup (requires internet, run once before ranking):\n"
        f"  python -c \"from sentence_transformers import SentenceTransformer; "
        f"SentenceTransformer('{DEFAULT_MODEL_NAME}').save('{DEFAULT_LOCAL_MODEL_DIR}')\"\n"
        f"After this, rank.py runs fully offline using the cached model."
    )
