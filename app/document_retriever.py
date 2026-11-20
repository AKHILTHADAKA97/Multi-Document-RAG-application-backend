"""Five-Stage Document Retrieval and Verification Agent.

Executes up to five intelligent retrieval attempts with early stopping:
- ATTEMPT 1: Search using the original user question.
- ATTEMPT 2: Search using the spelling-corrected question.
- ATTEMPT 3: Search using the grammar-normalized question.
- ATTEMPT 4: Search using semantic variations, extracted keywords, and related technical terms.
- ATTEMPT 5: Broaden retrieval across all authorized uploaded document collections using hybrid search.

Guarantees:
- Never executes duplicate/identical searches.
- Reuses cached embeddings.
- Deduplicates results across attempts.
- Ranks results by combined relevance.
- Stops early as soon as sufficient reliable evidence is found.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .relevance_evaluator import RelevanceEvaluator


class FiveStageDocumentRetriever:
    """Orchestrator for 5-Stage Document Retrieval and Verification."""

    def __init__(self, base_retriever: Any):
        self.base_retriever = base_retriever

    def search_5_stages(
        self,
        query_variants: dict[str, Any],
        document_ids: list[str],
        settings: Any,
        page: int | None = None,
        on_progress: Callable[[str, str, str], None] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Executes the five-stage retrieval pipeline with early stopping.
        
        Returns:
            (ranked_chunks, diagnostics)
        """
        min_sim = float(getattr(settings, 'min_similarity', 0.25))
        top_k = int(getattr(settings, 'top_k', 5))

        q1_original = query_variants.get('original_question', '').strip()
        q2_spelling = query_variants.get('spelling_corrected_query', '').strip()
        q3_grammar = query_variants.get('grammar_normalized_query', '').strip()
        q4_semantic = query_variants.get('semantic_variation_query', '').strip()
        q5_broad = query_variants.get('broad_search_query', '').strip()

        attempts_plan = [
            (1, "Original Question", q1_original, False),
            (2, "Spelling-Corrected Query", q2_spelling, False),
            (3, "Grammar-Normalized Query", q3_grammar, False),
            (4, "Semantic Variations & Keywords", q4_semantic, False),
            (5, "Broadened Collection Hybrid Search", q5_broad, True),
        ]

        seen_queries: set[str] = set()
        executed_attempts: list[dict[str, Any]] = []
        all_candidate_chunks: dict[str, dict[str, Any]] = {}
        early_stopped_at: str | None = None
        t_start = time.perf_counter()

        for attempt_num, attempt_name, attempt_query, is_broad_scope in attempts_plan:
            clean_q = attempt_query.strip().lower()
            if not clean_q or clean_q in seen_queries:
                continue

            seen_queries.add(clean_q)

            if on_progress:
                on_progress(
                    'searching_documents',
                    'Searching Uploaded Documents',
                    f"Executing Attempt {attempt_num}: {attempt_name}..."
                )

            # For Attempt 5 (Broadened scope), search all authorized ready docs if a subset had no hits
            target_doc_ids = [] if (is_broad_scope and len(document_ids) > 0) else document_ids

            try:
                chunks, diag = self.base_retriever.search(
                    attempt_query,
                    target_doc_ids,
                    settings,
                    page=page,
                    top_k_override=top_k * 2,
                )
            except Exception:
                chunks, diag = [], {}

            # Evaluate relevance of this attempt's results
            evaluation = RelevanceEvaluator.evaluate(
                attempt_query,
                chunks,
                min_similarity=min_sim,
                definitive_threshold=min_sim + 0.12,
            )

            executed_attempts.append({
                'attempt': attempt_num,
                'name': attempt_name,
                'query': attempt_query,
                'returned_chunks': len(chunks),
                'best_similarity': evaluation['best_similarity'],
                'has_sufficient_evidence': evaluation['has_sufficient_evidence'],
                'is_definitive': evaluation['is_definitive'],
                'confidence': evaluation['confidence'],
            })

            # Merge and deduplicate candidate chunks (keyed by document_id + chunk_index or rowid)
            for c in chunks:
                chunk_key = str(c.get('id') or f"{c.get('document_id')}_{c.get('chunk_index', c.get('rowid', ''))}")
                if chunk_key not in all_candidate_chunks:
                    all_candidate_chunks[chunk_key] = c
                else:
                    # Keep the chunk with higher similarity score
                    if float(c.get('similarity', 0.0)) > float(all_candidate_chunks[chunk_key].get('similarity', 0.0)):
                        all_candidate_chunks[chunk_key] = c

            # Early Stopping Check:
            # If definitive evidence is confirmed, stop immediately!
            if evaluation['is_definitive']:
                early_stopped_at = f"Attempt {attempt_num} ({attempt_name})"
                break

            # If Attempt 1, 2, or 3 found sufficient evidence with multiple quality chunks, stop early!
            if attempt_num <= 3 and evaluation['has_sufficient_evidence'] and evaluation['confidence'] in {'high', 'medium'}:
                early_stopped_at = f"Attempt {attempt_num} ({attempt_name})"
                break

        total_retrieval_ms = round((time.perf_counter() - t_start) * 1000)

        # Final Ranking and Selection
        pooled_chunks = list(all_candidate_chunks.values())
        final_eval = RelevanceEvaluator.evaluate(
            q3_grammar or q1_original,
            pooled_chunks,
            min_similarity=min_sim,
        )

        # Sort deduplicated chunks by similarity (and rerank score if available)
        pooled_chunks.sort(
            key=lambda c: (
                float(c.get('rerank_score') or 0.0),
                float(c.get('similarity', 0.0)),
                float(c.get('bm25', 0.0))
            ),
            reverse=True
        )

        # Slice to top_k and re-assign verified citation badges [S1], [S2]...
        final_chunks: list[dict[str, Any]] = []
        if final_eval['has_sufficient_evidence']:
            for idx, c in enumerate(pooled_chunks[:top_k], start=1):
                final_chunks.append({
                    **c,
                    'citation': f"S{idx}",
                })

        diagnostics = {
            'five_stage_retrieval': True,
            'attempts_executed': len(executed_attempts),
            'attempts_detail': executed_attempts,
            'early_stopped_at': early_stopped_at,
            'has_sufficient_evidence': final_eval['has_sufficient_evidence'],
            'confidence': final_eval['confidence'],
            'is_partial': final_eval['is_partial'],
            'best_similarity': final_eval['best_similarity'],
            'total_retrieval_ms': total_retrieval_ms,
        }

        return final_chunks, diagnostics
