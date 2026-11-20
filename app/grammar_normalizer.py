"""Grammar Normalization Engine for Multi-Document RAG.

Normalizes:
- Missing auxiliary verbs (e.g. "what machine learning" -> "What is machine learning?").
- Incomplete phrases / bare topic queries (e.g. "features of MLP" -> "What are the features of MLP?").
- Mixed-language and transliterated constructs (Hinglish, Telugu-English).
- Incorrect word order and agreement.
- Punctuation and sentence capitalization.
- Strictly preserves technical meaning, acronyms, and proper nouns.
"""

from __future__ import annotations

import re
from typing import Any

# ==============================================================================
# MIXED-LANGUAGE & TRANSLITERATED PHRASE PATTERNS
# ==============================================================================
MIXED_LANGUAGE_PATTERNS = [
    # Hinglish patterns
    (re.compile(r'\b(?:kya\s+hai|kya\s+hota\s+hai|kya\s+h|kya\s+he)\b', re.IGNORECASE), 'what is', 'definition'),
    (re.compile(r'\b(?:kaise\s+kaam\s+karta\s+hai|kaise\s+work\s+karta\s+hai|kese\s+kam\s+karta\s+h)\b', re.IGNORECASE), 'how does it work', 'technical'),
    (re.compile(r'\b(?:explain\s+karo|batao|samjhao|bataiye|explain\s+kar\s+do)\b', re.IGNORECASE), 'explain', 'explanation'),
    (re.compile(r'\b(?:kya\s+difference\s+hai|kya\s+antar\s+hai)\b', re.IGNORECASE), 'what is the difference between', 'comparison'),
    (re.compile(r'\b(?:ka\s+matlab\s+kya\s+hai|ka\s+meaning\s+kya\s+hai)\b', re.IGNORECASE), 'what is the meaning of', 'definition'),
    (re.compile(r'\b(?:ke\s+features\s+kya\s+hai|ke\s+fayde\s+kya\s+hai)\b', re.IGNORECASE), 'what are the features of', 'features'),

    # Telugu-English transliteration patterns
    (re.compile(r'\b(?:ante\s+enti|ante\s+entidi|ante\s+emititi)\b', re.IGNORECASE), 'what is', 'definition'),
    (re.compile(r'\b(?:ela\s+pani\s+chestundi|ela\s+work\s+avutundi|ela\s+pani\s+chestadi)\b', re.IGNORECASE), 'how does it work', 'technical'),
    (re.compile(r'\b(?:cheppu|vivarinchu|teliyajeyandi)\b', re.IGNORECASE), 'explain', 'explanation'),
    (re.compile(r'\b(?:bhedam\s+enti|theda\s+enti)\b', re.IGNORECASE), 'what is the difference between', 'comparison'),
    (re.compile(r'\b(?:gurinchi\s+cheppu|gurinchi\s+vivarinchu)\b', re.IGNORECASE), 'tell me about', 'overview'),
]

# Structural grammar repairs
MISSING_COPULA_PATTERNS = [
    # "what <noun_phrase>" -> "What is <noun_phrase>?"
    (re.compile(r'^\s*what\s+([a-z0-9\s_-]+?)(?:\s*\?)?\s*$', re.IGNORECASE),
     lambda m: f"What is {m.group(1).strip()}?"),
    # "how <noun_phrase> works" -> "How does <noun_phrase> work?"
    (re.compile(r'^\s*how\s+([a-z0-9\s_-]+?)\s+works?(?:\s*\?)?\s*$', re.IGNORECASE),
     lambda m: f"How does {m.group(1).strip()} work?"),
    # "why <noun_phrase> used" -> "Why is <noun_phrase> used?"
    (re.compile(r'^\s*why\s+([a-z0-9\s_-]+?)\s+used(?:\s*\?)?\s*$', re.IGNORECASE),
     lambda m: f"Why is {m.group(1).strip()} used?"),
    # "features of <noun_phrase>" -> "What are the features of <noun_phrase>?"
    (re.compile(r'^\s*features\s+(?:of|in)\s+([a-z0-9\s_-]+?)(?:\s*\?)?\s*$', re.IGNORECASE),
     lambda m: f"What are the features of {m.group(1).strip()}?"),
    # "architecture of <noun_phrase>" -> "Explain the architecture of <noun_phrase>."
    (re.compile(r'^\s*architecture\s+(?:of|in)\s+([a-z0-9\s_-]+?)(?:\s*\?)?\s*$', re.IGNORECASE),
     lambda m: f"Explain the architecture of {m.group(1).strip()}."),
    # "difference <A> and <B>" -> "What is the difference between <A> and <B>?"
    (re.compile(r'^\s*(?:difference|diff)\s+(?:between\s+)?([a-z0-9\s_-]+?)\s+(?:and|vs\.?|versus)\s+([a-z0-9\s_-]+?)(?:\s*\?)?\s*$', re.IGNORECASE),
     lambda m: f"What is the difference between {m.group(1).strip()} and {m.group(2).strip()}?"),
    # "types of <noun_phrase>" -> "What are the types of <noun_phrase>?"
    (re.compile(r'^\s*types\s+of\s+([a-z0-9\s_-]+?)(?:\s*\?)?\s*$', re.IGNORECASE),
     lambda m: f"What are the types of {m.group(1).strip()}?"),
    # "examples of <noun_phrase>" -> "What are examples of <noun_phrase>?"
    (re.compile(r'^\s*examples?\s+of\s+([a-z0-9\s_-]+?)(?:\s*\?)?\s*$', re.IGNORECASE),
     lambda m: f"What are examples of {m.group(1).strip()}?"),
]


class GrammarNormalizer:
    """Intelligent Grammar Normalization Engine."""

    @staticmethod
    def normalize(text: str) -> tuple[str, list[dict[str, Any]], bool]:
        """Normalizes sentence structure, missing copula/auxiliary verbs, and mixed language.
        
        Returns:
            (normalized_text, list_of_grammar_actions, has_normalization)
        """
        if not text or not text.strip():
            return text, [], False

        working = text.strip()
        actions: list[dict[str, Any]] = []

        # 1. Mixed language / transliterated normalization
        for pattern, english_equiv, note in MIXED_LANGUAGE_PATTERNS:
            if pattern.search(working):
                # If question contains transliteration at end (e.g. "ML kya hai")
                working = pattern.sub('', working).strip()
                if not working.lower().startswith(('what', 'how', 'why', 'explain', 'tell')):
                    working = f"{english_equiv} {working}".strip()
                actions.append({
                    'type': 'mixed_language_normalization',
                    'intent': note,
                    'replacement': english_equiv,
                })

        # Clean multiple spaces resulting from regex substitutions
        working = re.sub(r'\s+', ' ', working).strip()

        # 2. Structural grammar and missing copula repair
        for pattern, repl_fn in MISSING_COPULA_PATTERNS:
            # Check if working already starts with "what is", "how does", etc. to avoid duplicate insertion
            if re.match(r'^\s*what\s+(?:is|are|was|were)\b', working, re.IGNORECASE):
                continue
            if re.match(r'^\s*how\s+(?:does|do|did|can|to)\b', working, re.IGNORECASE):
                continue
            if re.match(r'^\s*why\s+(?:is|are|was|were|do|does)\b', working, re.IGNORECASE):
                continue

            match = pattern.match(working)
            if match:
                repaired = repl_fn(match)
                actions.append({
                    'type': 'sentence_structure_repair',
                    'before': working,
                    'after': repaired,
                })
                working = repaired
                break

        # 3. Capitalization and trailing punctuation
        working = working.strip()
        if working:
            # Capitalize first letter if lowercase
            if working[0].islower():
                working = working[0].upper() + working[1:]

            # If it's a question without trailing '?', append '?'
            is_question = bool(re.match(r'^(?:what|how|why|when|where|which|who|can|could|is|are|does|do|will|would)\b', working, re.IGNORECASE))
            if is_question and not working.endswith(('?', '.', '!')):
                working += '?'

        has_normalization = len(actions) > 0 and working.lower() != text.strip().lower()
        return working, actions, has_normalization
