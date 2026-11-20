"""Context-Aware Query Understanding Agent.

Resolves conversational context (pronouns, ordinal references, follow-ups),
performs intent routing, and produces multi-stage retrieval variants for
the 5-Stage Document Retrieval Agent.
"""

from __future__ import annotations

import re
from typing import Any

from .query_normalizer import QueryNormalizer, ACRONYM_EXPANSION_MAP

ORDINAL_WORDS = {
    'first': 0, '1st': 0, 'one': 0,
    'second': 1, '2nd': 1, 'two': 1,
    'third': 2, '3rd': 2, 'three': 2,
    'fourth': 3, '4th': 3, 'four': 3,
    'fifth': 4, '5th': 4, 'five': 4,
    'last': -1,
}

PRONOUNS_RE = re.compile(r'\b(it|its|this|that|these|those|them|they)\b', re.IGNORECASE)
FOLLOW_UP_RE = re.compile(
    r'^\s*(?:how\s+about|what\s+about|explain\s+more|tell\s+more|elaborate|and\s+its?|features?\??|types?\??|architecture\??|advantages?\??|disadvantages?\??|applications?\??)\b',
    re.IGNORECASE
)


class QueryUnderstandingAgent:
    """Conversational Context and Multi-Stage Query Formulator."""

    @classmethod
    def analyze_query(
        cls,
        question: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Analyzes user query, resolves conversational context, and forms the 5 retrieval variants.
        
        Returns:
            dict containing:
            - original_question
            - spelling_corrected_query
            - grammar_normalized_query
            - semantic_variation_query
            - broad_search_query
            - is_follow_up
            - referenced_entity
            - intent
            - key_entities
        """
        raw_question = (question or '').strip()
        history = conversation_history or []

        # 1. Base Normalization (Spelling + Grammar)
        norm_result = QueryNormalizer.normalize(raw_question)
        spelling_q = norm_result['spelling_corrected']
        grammar_q = norm_result['grammar_normalized']

        # 2. Extract context from recent conversation turns
        last_entity = cls._extract_last_entity(history)
        last_bullet_items = cls._extract_bullet_items(history)

        # 3. Context & Reference Resolution
        context_resolved_q = grammar_q
        is_follow_up = False
        referenced_entity = None

        q_lower = grammar_q.lower()

        # Check for ordinal references: "explain the second one", "what is the 1st one"
        ordinal_match = re.search(r'\b(?:the\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last)\s+(?:one|point|item|feature|concept|architecture)\b', q_lower)
        if ordinal_match and last_bullet_items:
            ord_word = ordinal_match.group(1).lower()
            idx = ORDINAL_WORDS.get(ord_word, 0)
            if idx == -1 and len(last_bullet_items) > 0:
                resolved_item = last_bullet_items[-1]
            elif 0 <= idx < len(last_bullet_items):
                resolved_item = last_bullet_items[idx]
            else:
                resolved_item = last_bullet_items[0]

            context_resolved_q = f"Explain {resolved_item} in {last_entity or 'the topic'}."
            is_follow_up = True
            referenced_entity = resolved_item

        # Check for pronouns: "what are its features?", "how does it work?"
        elif PRONOUNS_RE.search(grammar_q) and last_entity:
            # Replace pronoun with the referenced entity
            def _sub_pronoun(m: re.Match) -> str:
                word = m.group(1).lower()
                if word == 'its':
                    return f"{last_entity}'s"
                return last_entity

            context_resolved_q = PRONOUNS_RE.sub(_sub_pronoun, grammar_q)
            is_follow_up = True
            referenced_entity = last_entity

        # Check for short follow-up phrases: "architecture?", "features?"
        elif (len(grammar_q.split()) <= 4 or FOLLOW_UP_RE.match(grammar_q)) and last_entity:
            if last_entity.lower() not in grammar_q.lower():
                clean_clause = grammar_q.rstrip('?.')
                context_resolved_q = f"{clean_clause} of {last_entity}?"
                is_follow_up = True
                referenced_entity = last_entity

        # 4. Formulate the Five Search Queries
        # Attempt 1: Raw user input as-is
        attempt_1_query = raw_question

        # Attempt 2: Spelling-corrected query
        attempt_2_query = spelling_q

        # Attempt 3: Grammar-normalized query (with conversational pronouns resolved)
        attempt_3_query = context_resolved_q

        # Attempt 4: Semantic variations + technical keyword expansion
        semantic_terms = cls._build_semantic_variations(context_resolved_q)
        attempt_4_query = f"{context_resolved_q} {semantic_terms}".strip()

        # Attempt 5: Broad hybrid query (core entities & nouns for broad cross-document fallback)
        core_entities = cls._extract_entities(context_resolved_q)
        attempt_5_query = ' '.join(core_entities) if core_entities else context_resolved_q

        # Intent classification
        intent = cls._classify_intent(context_resolved_q)

        return {
            'original_question': raw_question,
            'spelling_corrected_query': attempt_2_query,
            'grammar_normalized_query': attempt_3_query,
            'semantic_variation_query': attempt_4_query,
            'broad_search_query': attempt_5_query,
            'is_follow_up': is_follow_up,
            'referenced_entity': referenced_entity,
            'intent': intent,
            'corrections': norm_result['corrections'],
            'is_corrected': norm_result['is_modified'],
            'key_entities': core_entities,
        }

    @classmethod
    def _extract_last_entity(cls, history: list[dict[str, Any]]) -> str | None:
        """Finds the main subject entity from recent history questions and answers."""
        if not history:
            return None

        # Inspect the last 2 user turns
        for turn in reversed(history[-2:]):
            q = (turn.get('question') or '').strip()
            # Check definition queries e.g. "What is Machine Learning?"
            match = re.search(r'\b(?:what is|what are|define|explain|about)\s+([A-Za-z0-9\s_-]+?)(?:\s*\?|\s*$)', q, re.IGNORECASE)
            if match:
                cand = match.group(1).strip()
                cand_clean = re.sub(r'^(?:a|an|the)\s+', '', cand, flags=re.IGNORECASE).strip()
                if cand_clean and len(cand_clean.split()) <= 4:
                    return cand_clean

            # Or take the longest technical capital or noun sequence
            caps = re.findall(r'\b[A-Z][a-zA-Z0-9_-]*(?:\s+[A-Z][a-zA-Z0-9_-]*)*\b', q)
            valid_caps = [c for c in caps if c.lower() not in {'what', 'how', 'why', 'when', 'where', 'explain'}]
            if valid_caps:
                return valid_caps[-1]

        return None

    @classmethod
    def _extract_bullet_items(cls, history: list[dict[str, Any]]) -> list[str]:
        """Extracts enumerated list items or bullets from the last AI answer."""
        if not history:
            return []

        last_ans = ""
        for turn in reversed(history[-1:]):
            resp = turn.get('response') or {}
            last_ans = resp.get('answer') or turn.get('answer') or ""
            if last_ans:
                break

        if not last_ans:
            return []

        # Find markdown items: 1. **Item Name**: or - **Item Name**:
        items: list[str] = []
        numbered = re.findall(r'^\s*(?:\d+\.|\*|-)\s+\*\*([^*]+)\*\*', last_ans, re.MULTILINE)
        if numbered:
            return [it.strip() for it in numbered[:6]]

        # Plain bullet lines
        bullets = re.findall(r'^\s*(?:\d+\.|\*|-)\s+([A-Z][^.:\n]{3,40})[:.]', last_ans, re.MULTILINE)
        if bullets:
            return [it.strip() for it in bullets[:6]]

        return []

    @classmethod
    def _build_semantic_variations(cls, text: str) -> str:
        """Expands text with domain synonyms and full acronym terms."""
        terms = re.findall(r'\b[A-Za-z0-9_-]+\b', text)
        additions: list[str] = []
        for t in terms:
            low = t.lower()
            if low in ACRONYM_EXPANSION_MAP:
                additions.append(ACRONYM_EXPANSION_MAP[low])
            if low == 'difference':
                additions.extend(['comparison', 'versus', 'contrast'])
            elif low in {'architecture', 'structure'}:
                additions.extend(['layers', 'components', 'pipeline'])
            elif low in {'features', 'characteristics'}:
                additions.extend(['properties', 'advantages', 'key components'])

        return ' '.join(dict.fromkeys(additions))

    @classmethod
    def _extract_entities(cls, text: str) -> list[str]:
        """Extracts key searchable tokens/entities."""
        stop = {'what', 'is', 'are', 'was', 'were', 'how', 'does', 'do', 'why', 'in', 'of', 'the', 'a', 'an', 'and', 'or', 'for', 'to'}
        tokens = [t for t in re.findall(r'\b[A-Za-z0-9_-]+\b', text) if t.lower() not in stop and len(t) > 1]
        return list(dict.fromkeys(tokens))

    @classmethod
    def _classify_intent(cls, question: str) -> str:
        q_low = question.lower()
        if re.search(r'\b(what is|what are|define|definition|meaning)\b', q_low):
            return "definition"
        if re.search(r'\b(how does|how to|architecture|pipeline|steps|algorithm)\b', q_low):
            return "technical"
        if re.search(r'\b(compare|difference|versus|vs\.?)\b', q_low):
            return "comparison"
        if re.search(r'\b(why|reasons?|benefits?|pros and cons)\b', q_low):
            return "analytical"
        if re.search(r'\b(summarize|summary|overview|brief)\b', q_low):
            return "summary"
        return "general"
