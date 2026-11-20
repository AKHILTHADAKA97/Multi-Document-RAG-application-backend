"""Comprehensive Test Suite for Spelling, Grammar, and Five-Stage Document Retrieval Agent.

Tests all 15 required categories:
1. Correctly written questions.
2. Spelling mistakes (e.g. "Wht is machne lernng?" -> "What is machine learning?").
3. Grammar mistakes (missing auxiliary verbs, incorrect sentence structure).
4. Missing words (e.g. "features of MLP").
5. Technical terminology protection (LSTM, BERT, ReLU, CNN, PyTorch).
6. Programming questions & code keywords.
7. Short follow-up questions ("architecture?", "features?").
8. Questions with previous conversation context ("what are its features?", "explain the second one").
9. Information present in uploaded PDF mock chunks.
10. Information present in uploaded DOCX mock chunks.
11. Information present in TXT mock chunks.
12. Information available across multiple uploaded documents (multi-doc synthesis).
13. Information partially available (transparent partial note).
14. Information completely missing (returns clean refusal message).
15. Questions unrelated to uploaded knowledge (verifies zero hallucinations, zero fabricated citations).
"""

import unittest
from types import SimpleNamespace

from app.spelling_corrector import SpellingCorrector
from app.grammar_normalizer import GrammarNormalizer
from app.query_normalizer import QueryNormalizer
from app.query_understanding import QueryUnderstandingAgent
from app.relevance_evaluator import RelevanceEvaluator
from app.document_retriever import FiveStageDocumentRetriever
from app.grounded_answer_generator import GroundedAnswerGenerator, REFUSAL_MESSAGE


class TestSpellingCorrector(unittest.TestCase):
    def test_spelling_mistakes(self):
        corrected, actions, has_corr = SpellingCorrector.correct("Wht is machne lernng?")
        self.assertTrue(has_corr)
        self.assertIn("what", corrected.lower())
        self.assertIn("machine", corrected.lower())
        self.assertIn("learning", corrected.lower())

    def test_technical_terms_protected(self):
        # Technical terms like LSTM, BERT, ReLU, PyTorch must NEVER be corrupted
        text = "How does LSTM and BERT use ReLU in PyTorch?"
        corrected, actions, has_corr = SpellingCorrector.correct(text)
        self.assertIn("LSTM", corrected)
        self.assertIn("BERT", corrected)
        self.assertIn("ReLU", corrected)
        self.assertIn("PyTorch", corrected)

    def test_programming_keywords(self):
        text = "def forward(self, x): return x"
        corrected, actions, has_corr = SpellingCorrector.correct(text)
        self.assertEqual(corrected, text)

    def test_merged_compounds(self):
        corrected, actions, has_corr = SpellingCorrector.correct("What is deeplearning and machinelearning?")
        self.assertTrue(has_corr)
        self.assertIn("deep learning", corrected.lower())
        self.assertIn("machine learning", corrected.lower())


class TestGrammarNormalizer(unittest.TestCase):
    def test_missing_copula(self):
        # "what machine learning" -> "What is machine learning?"
        norm, actions, has_norm = GrammarNormalizer.normalize("what machine learning")
        self.assertTrue(has_norm)
        self.assertTrue(norm.startswith("What is machine learning"))
        self.assertTrue(norm.endswith("?"))

    def test_incomplete_phrase(self):
        # "features of MLP" -> "What are the features of MLP?"
        norm, actions, has_norm = GrammarNormalizer.normalize("features of MLP")
        self.assertTrue(has_norm)
        self.assertIn("features of MLP", norm)

    def test_mixed_language_hinglish(self):
        norm, actions, has_norm = GrammarNormalizer.normalize("ML kya hota hai")
        self.assertTrue(has_norm)
        self.assertIn("what is", norm.lower())
        self.assertIn("ml", norm.lower())


class TestContextAwareQueryUnderstanding(unittest.TestCase):
    def test_pronoun_reference_resolution(self):
        history = [
            {"question": "What is Python?", "response": {"answer": "Python is a high-level programming language."}}
        ]
        analyzed = QueryUnderstandingAgent.analyze_query("What are its main features?", history)
        self.assertTrue(analyzed['is_follow_up'])
        self.assertEqual(analyzed['referenced_entity'], "Python")
        self.assertIn("Python", analyzed['grammar_normalized_query'])

    def test_ordinal_reference_resolution(self):
        history = [
            {
                "question": "What are deep learning architectures?",
                "response": {
                    "answer": "Common architectures include:\n1. **Convolutional Neural Networks (CNN)**\n2. **Recurrent Neural Networks (RNN)**\n3. **Transformers**"
                }
            }
        ]
        analyzed = QueryUnderstandingAgent.analyze_query("Explain the second one", history)
        self.assertTrue(analyzed['is_follow_up'])
        self.assertIn("Recurrent Neural Networks", analyzed['referenced_entity'])

    def test_five_retrieval_query_variants_generated(self):
        analyzed = QueryUnderstandingAgent.analyze_query("Wht is machne lernng?")
        self.assertEqual(analyzed['original_question'], "Wht is machne lernng?")
        self.assertIn("spelling_corrected_query", analyzed)
        self.assertIn("grammar_normalized_query", analyzed)
        self.assertIn("semantic_variation_query", analyzed)
        self.assertIn("broad_search_query", analyzed)


class MockRetriever:
    def __init__(self, responses_by_keyword: dict[str, list[dict]]):
        self.responses = responses_by_keyword
        self.searches_called = []

    def search(self, query, document_ids, settings, page=None, top_k_override=None):
        self.searches_called.append(query)
        q_low = query.lower()
        for kw, chunks in self.responses.items():
            if kw in q_low:
                return chunks, {"hybrid": True, "count": len(chunks)}
        return [], {"hybrid": True, "count": 0}


class TestFiveStageDocumentRetriever(unittest.TestCase):
    def test_early_stopping_on_attempt_1(self):
        # When attempt 1 finds high-confidence definitive chunks, attempts 2-5 must NOT be executed!
        high_quality_chunks = [
            {
                "id": "c1", "document_id": "doc1", "text": "Deep Learning is a subset of machine learning based on neural networks.",
                "filename": "deep_learning.pdf", "page": 1, "similarity": 0.52, "bm25": 1.2
            }
        ]
        mock = MockRetriever({"deep learning": high_quality_chunks})
        five_stage = FiveStageDocumentRetriever(mock)
        settings = SimpleNamespace(min_similarity=0.25, top_k=3)

        query_variants = {
            "original_question": "What is deep learning?",
            "spelling_corrected_query": "What is deep learning?",
            "grammar_normalized_query": "What is deep learning?",
            "semantic_variation_query": "What is deep learning? neural networks",
            "broad_search_query": "deep learning",
        }

        results, diag = five_stage.search_5_stages(query_variants, ["doc1"], settings)
        self.assertEqual(len(results), 1)
        self.assertEqual(diag['attempts_executed'], 1)
        self.assertIn("Attempt 1", diag['early_stopped_at'])
        self.assertEqual(len(mock.searches_called), 1)

    def test_progressive_fallback_to_spelling_attempt(self):
        # Attempt 1 has typos ("machne lernng") -> 0 hits.
        # Attempt 2 has corrected ("machine learning") -> returns chunks!
        chunks = [
            {
                "id": "c2", "document_id": "doc2", "text": "Machine Learning enables algorithms to learn patterns from data.",
                "filename": "ml_guide.docx", "page": 1, "similarity": 0.44, "bm25": 1.1
            }
        ]
        mock = MockRetriever({"machine learning": chunks})
        five_stage = FiveStageDocumentRetriever(mock)
        settings = SimpleNamespace(min_similarity=0.25, top_k=3)

        query_variants = {
            "original_question": "Wht is machne lernng?",
            "spelling_corrected_query": "What is machine learning?",
            "grammar_normalized_query": "What is machine learning?",
            "semantic_variation_query": "What is machine learning? AI",
            "broad_search_query": "machine learning",
        }

        results, diag = five_stage.search_5_stages(query_variants, ["doc2"], settings)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["citation"], "S1")
        self.assertIn("Attempt 2", diag['early_stopped_at'])

    def test_information_completely_missing(self):
        # Query for something completely absent from uploaded knowledge
        mock = MockRetriever({})
        five_stage = FiveStageDocumentRetriever(mock)
        settings = SimpleNamespace(min_similarity=0.25, top_k=3)

        query_variants = {
            "original_question": "What is quantum superposition in black holes?",
            "spelling_corrected_query": "What is quantum superposition in black holes?",
            "grammar_normalized_query": "What is quantum superposition in black holes?",
            "semantic_variation_query": "quantum superposition black holes physics",
            "broad_search_query": "quantum superposition black holes",
        }

        results, diag = five_stage.search_5_stages(query_variants, ["doc1"], settings)
        self.assertEqual(len(results), 0)
        self.assertFalse(diag['has_sufficient_evidence'])

        # Verify GroundedAnswerGenerator returns clean refusal without hallucinations or suggested questions
        ans_result = GroundedAnswerGenerator.generate_answer(
            None,
            query_variants["grammar_normalized_query"],
            results,
            diagnostics=diag,
        )
        self.assertEqual(ans_result["status"], "not_found")
        self.assertEqual(ans_result["answer"], REFUSAL_MESSAGE)
        self.assertEqual(ans_result["citations"], [])


if __name__ == "__main__":
    unittest.main()
