"""Relevance Evaluator for Multi-Stage Document Retrieval.

Evaluates evidence quality of retrieved document passages:
- Checks dense semantic similarity against configured thresholds.
- Computes query concept coverage & term overlap.
- Provides early-stopping signals (is_definitive) to avoid wasteful retrieval attempts.
- Flags when evidence is completely missing or only partial.
"""

from __future__ import annotations

import re
from typing import Any

STOP_WORDS = {
    'what', 'is', 'are', 'was', 'were', 'how', 'does', 'do', 'why', 'in', 'of',
    'the', 'a', 'an', 'and', 'or', 'for', 'to', 'with', 'from', 'it', 'its',
    'this', 'that', 'these', 'those', 'tell', 'me', 'about', 'explain', 'please',
}


class RelevanceEvaluator:
    """Evaluates passage relevance and retrieval completeness."""

    @classmethod
    def evaluate(
        cls,
        query: str,
        chunks: list[dict[str, Any]],
        min_similarity: float = 0.25,
        definitive_threshold: float = 0.38,
    ) -> dict[str, Any]:
        """Evaluates retrieved chunks for relevance to query.
        
        Returns:
            dict containing:
            - has_sufficient_evidence: bool
            - is_definitive: bool (triggers early stopping)
            - best_similarity: float
            - avg_top_similarity: float
            - confidence: 'high' | 'medium' | 'low' | 'insufficient'
            - is_partial: bool
            - matched_terms: list[str]
            - quality_chunks: list[dict]
        """
        if not chunks:
            return {
                'has_sufficient_evidence': False,
                'is_definitive': False,
                'best_similarity': 0.0,
                'avg_top_similarity': 0.0,
                'confidence': 'insufficient',
                'is_partial': False,
                'matched_terms': [],
                'quality_chunks': [],
            }

        q_clean = re.sub(r'[^a-zA-Z0-9\s_-]', ' ', query.lower())
        q_tokens = set(t for t in q_clean.split() if t not in STOP_WORDS and len(t) > 1)

        quality_chunks: list[dict[str, Any]] = []
        matched_terms: set[str] = set()

        for c in chunks:
            sim = float(c.get('similarity', 0.0))
            text = (c.get('text') or '').lower()
            section = (c.get('section') or '').lower()
            combined_text = f"{section} {text}"

            # Calculate token overlap
            chunk_tokens = set(re.findall(r'\b[a-zA-Z0-9_-]+\b', combined_text))
            overlap = q_tokens & chunk_tokens
            matched_terms |= overlap

            # A chunk is considered quality if similarity passes min_similarity
            # or if it has significant token overlap with query keywords
            if sim >= min_similarity or (q_tokens and len(overlap) / len(q_tokens) >= 0.5):
                quality_chunks.append({
                    **c,
                    'term_overlap_count': len(overlap),
                    'term_overlap_ratio': round(len(overlap) / max(len(q_tokens), 1), 2),
                })

        best_sim = max((float(c.get('similarity', 0.0)) for c in chunks), default=0.0)
        top3_sims = sorted([float(c.get('similarity', 0.0)) for c in quality_chunks], reverse=True)[:3]
        avg_top_sim = round(sum(top3_sims) / len(top3_sims), 4) if top3_sims else 0.0

        overlap_ratio = len(matched_terms) / max(len(q_tokens), 1) if q_tokens else 1.0

        # High confidence / definitive check (Early stopping condition)
        is_definitive = False
        if best_sim >= definitive_threshold and overlap_ratio >= 0.6:
            is_definitive = True
        elif len(quality_chunks) >= 3 and avg_top_sim >= min_similarity + 0.05 and overlap_ratio >= 0.7:
            is_definitive = True

        # Has sufficient evidence check
        has_sufficient_evidence = bool(
            quality_chunks and (
                best_sim >= min_similarity or
                (overlap_ratio >= 0.5 and len(quality_chunks) >= 1)
            )
        )

        # Confidence categorization
        if is_definitive or (best_sim >= 0.35 and overlap_ratio >= 0.7):
            confidence = 'high'
        elif has_sufficient_evidence and best_sim >= min_similarity:
            confidence = 'medium'
        elif quality_chunks:
            confidence = 'low'
        else:
            confidence = 'insufficient'

        # Partial information detection
        is_partial = bool(
            has_sufficient_evidence and
            (overlap_ratio < 0.6 or len(quality_chunks) == 1) and
            confidence != 'high'
        )

        return {
            'has_sufficient_evidence': has_sufficient_evidence,
            'is_definitive': is_definitive,
            'best_similarity': round(best_sim, 4),
            'avg_top_similarity': avg_top_sim,
            'confidence': confidence,
            'is_partial': is_partial,
            'matched_terms': list(matched_terms),
            'quality_chunks': quality_chunks,
        }
