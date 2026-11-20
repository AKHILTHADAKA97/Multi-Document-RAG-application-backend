"""Unified Query Normalizer for Multi-Document RAG.

Coordinates spelling correction, grammar normalization, abbreviation expansion,
and produces normalized queries while preserving the exact original question.
"""

from __future__ import annotations

import re
from typing import Any

from .spelling_corrector import SpellingCorrector
from .grammar_normalizer import GrammarNormalizer

ACRONYM_EXPANSION_MAP = {
    'ml': 'Machine Learning',
    'ai': 'Artificial Intelligence',
    'dl': 'Deep Learning',
    'nlp': 'Natural Language Processing',
    'llm': 'Large Language Model',
    'llms': 'Large Language Models',
    'ann': 'Artificial Neural Network',
    'anns': 'Artificial Neural Networks',
    'rnn': 'Recurrent Neural Network',
    'rnns': 'Recurrent Neural Networks',
    'cnn': 'Convolutional Neural Network',
    'cnns': 'Convolutional Neural Networks',
    'mlp': 'Multi-Layer Perceptron',
    'mlps': 'Multi-Layer Perceptrons',
    'lstm': 'Long Short-Term Memory',
    'lstms': 'Long Short-Term Memory',
    'gru': 'Gated Recurrent Unit',
    'rag': 'Retrieval-Augmented Generation',
    'gan': 'Generative Adversarial Network',
    'gans': 'Generative Adversarial Networks',
    'svm': 'Support Vector Machine',
    'pca': 'Principal Component Analysis',
    'sgd': 'Stochastic Gradient Descent',
    'adam': 'Adam optimizer',
    'bptt': 'Backpropagation Through Time',
    'relu': 'Rectified Linear Unit (ReLU)',
    'mse': 'Mean Squared Error',
}


class QueryNormalizer:
    """Unified Query Normalization Pipeline."""

    @classmethod
    def normalize(cls, question: str) -> dict[str, Any]:
        """Runs spelling and grammar normalization stages.
        
        Returns:
            dict containing:
            - original_question: Exactly as submitted by user
            - spelling_corrected: Query after spelling correction
            - grammar_normalized: Query after grammar normalization
            - fully_normalized: Final normalized query with acronym awareness
            - corrections: List of all adjustments performed
            - is_modified: True if any changes occurred
        """
        raw_question = (question or '').strip()
        if not raw_question:
            return {
                'original_question': '',
                'spelling_corrected': '',
                'grammar_normalized': '',
                'fully_normalized': '',
                'corrections': [],
                'is_modified': False,
            }

        all_corrections: list[dict[str, Any]] = []

        # 1. Spelling Correction Stage
        spelling_corrected, spelling_actions, has_spell = SpellingCorrector.correct(raw_question)
        if has_spell:
            all_corrections.extend(spelling_actions)

        # 2. Grammar Normalization Stage
        grammar_normalized, grammar_actions, has_grammar = GrammarNormalizer.normalize(spelling_corrected)
        if has_grammar:
            all_corrections.extend(grammar_actions)

        # 3. Expanded Query (for search retrieval semantic boosting)
        tokens = re.findall(r'\b[A-Za-z0-9_-]+\b', grammar_normalized)
        expanded_terms: list[str] = []
        for t in tokens:
            low = t.lower()
            if low in ACRONYM_EXPANSION_MAP and low not in grammar_normalized.lower():
                expanded_terms.append(ACRONYM_EXPANSION_MAP[low])

        fully_normalized = grammar_normalized
        is_modified = bool(has_spell or has_grammar or fully_normalized != raw_question)

        return {
            'original_question': raw_question,
            'spelling_corrected': spelling_corrected,
            'grammar_normalized': grammar_normalized,
            'fully_normalized': fully_normalized,
            'expanded_terms': expanded_terms,
            'corrections': all_corrections,
            'is_modified': is_modified,
        }
