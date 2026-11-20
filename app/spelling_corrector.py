"""Spelling Correction Engine for Multi-Document RAG.

Detects and corrects:
- Typing mistakes and missing characters (e.g. "machne lernng" -> "machine learning").
- Inverted/transposed letters (e.g. "teh" -> "the", "algotithm" -> "algorithm").
- Common typing slips and abbreviations (e.g. "wht" -> "what", "hw" -> "how").
- Merged words (e.g. "deeplearning" -> "deep learning").
- Strictly protects technical terminology, code identifiers, and proper nouns.
"""

from __future__ import annotations

import re
from typing import Any

# ==============================================================================
# PROTECTED TECHNICAL TERMS AND CODE KEYWORDS (NEVER OVERCORRECT)
# ==============================================================================
PROTECTED_TERMS = {
    'lstm', 'lstms', 'bert', 'gpt', 'gpt-4', 'gpt-3', 'cnn', 'cnns', 'rnn', 'rnns',
    'mlp', 'mlps', 'gru', 'grus', 'rag', 'gan', 'gans', 'ann', 'anns', 'svm', 'pca',
    'sgd', 'adam', 'bptt', 'relu', 'gelu', 'sigmoid', 'softmax', 'tanh', 'mse', 'rmse',
    'pytorch', 'tensorflow', 'keras', 'sklearn', 'numpy', 'scipy', 'pandas', 'huggingface',
    'transformer', 'transformers', 'attention', 'tokenization', 'embeddings', 'faiss',
    'chromadb', 'pinecone', 'bm25', 'cross-encoder', 'bi-encoder', 'roc-auc',
    'resnet', 'vgg', 'yolo', 'densenet', 'autoencoder', 'autoencoders', 'variational',
    'eigenvalue', 'eigenvector', 'gradient', 'backpropagation', 'hyperparameter',
    'def', 'class', 'import', 'return', 'yield', 'lambda', 'async', 'await',
    'int', 'float', 'str', 'bool', 'list', 'dict', 'set', 'tuple', 'true', 'false', 'none',
}

# ==============================================================================
# COMMON TYPOS AND SLANG DICTIONARY
# ==============================================================================
SPELLING_TYPO_MAP: dict[str, str] = {
    # Question interrogatives
    'wht': 'what', 'wat': 'what', 'wot': 'what', 'waht': 'what', 'wats': 'what is', 'whats': 'what is',
    'hw': 'how', 'hwo': 'how', 'ho': 'how',
    'whr': 'where', 'wher': 'where', 'wer': 'where',
    'wn': 'when', 'wen': 'when',
    'y': 'why', 'whi': 'why',
    'whch': 'which', 'wich': 'which', 'whcih': 'which',
    'whoo': 'who',

    # Common auxiliary and verbs
    'isnt': 'is not', 'arent': 'are not', 'wasnt': 'was not', 'werent': 'were not',
    'cant': 'cannot', 'couldnt': 'could not', 'wouldnt': 'would not', 'shouldnt': 'should not',
    'dows': 'does', 'dose': 'does', 'duz': 'does', 'dosent': 'does not', 'doesnt': 'does not',
    'wrks': 'works', 'wrk': 'work', 'workng': 'working', 'wrking': 'working',
    'tel': 'tell', 'tl': 'tell', 'tll': 'tell',
    'giv': 'give', 'gve': 'give',
    'shw': 'show', 'sho': 'show',
    'explan': 'explain', 'expaln': 'explain', 'expln': 'explain', 'explane': 'explain',
    'defne': 'define', 'defination': 'definition', 'definiton': 'definition', 'definitiion': 'definition',
    'sumarize': 'summarize', 'sumary': 'summary', 'sumry': 'summary', 'sumerize': 'summarize',
    'evaluvate': 'evaluate', 'evalute': 'evaluate',
    'compair': 'compare', 'comparision': 'comparison', 'compr': 'compare',

    # Conversational & connective terms
    'plz': 'please', 'pls': 'please', 'plez': 'please',
    'bcz': 'because', 'bcoz': 'because', 'becoz': 'because', 'cuz': 'because', 'bcuz': 'because',
    'bteeen': 'between', 'betwen': 'between', 'btwn': 'between', 'betwn': 'between',
    'imprtnt': 'important', 'importnt': 'important', 'importent': 'important',
    'diffrence': 'difference', 'diffrnce': 'difference', 'diffrent': 'different', 'diference': 'difference',
    'exmple': 'example', 'exmpl': 'example', 'eg': 'example', 'exampel': 'example',
    'fatures': 'features', 'feautres': 'features', 'fetures': 'features', 'featurs': 'features',
    'functon': 'function', 'fnction': 'function', 'funciton': 'function',
    'algoritm': 'algorithm', 'algorthm': 'algorithm', 'algotithm': 'algorithm', 'algorothm': 'algorithm',
    'archtecture': 'architecture', 'architechture': 'architecture', 'architacture': 'architecture',
    'paradigam': 'paradigm', 'paradim': 'paradigm',
    'performence': 'performance', 'preformance': 'performance',

    # AI, ML & Deep Learning terms
    'machne': 'machine', 'mchine': 'machine', 'machin': 'machine', 'macheen': 'machine', 'machne-learning': 'machine learning',
    'lerning': 'learning', 'lrning': 'learning', 'learnng': 'learning', 'lernin': 'learning', 'leanring': 'learning',
    'artficial': 'artificial', 'artifical': 'artificial', 'artifitial': 'artificial', 'artifcial': 'artificial',
    'intligenc': 'intelligence', 'inteligence': 'intelligence', 'intellegence': 'intelligence', 'intelligense': 'intelligence',
    'deeep': 'deep', 'dep': 'deep', 'depp': 'deep',
    'neurall': 'neural', 'nueral': 'neural', 'neurl': 'neural', 'nural': 'neural', 'neral': 'neural',
    'netwrk': 'network', 'netwrok': 'network', 'netowrk': 'network', 'netwrks': 'networks', 'netwroks': 'networks',
    'percepton': 'perceptron', 'perceptrn': 'perceptron', 'perceprton': 'perceptron',
    'reccurent': 'recurrent', 'recurent': 'recurrent', 'recurant': 'recurrent',
    'convolutinal': 'convolutional', 'convoluton': 'convolution', 'convulution': 'convolution',
    'transfomer': 'transformer', 'transfomers': 'transformers', 'transformr': 'transformer',
    'suparvised': 'supervised', 'supervized': 'supervised', 'suprvised': 'supervised',
    'unsuparvised': 'unsupervised', 'unsupervized': 'unsupervised',
    'reinforcment': 'reinforcement', 'reinforcemnt': 'reinforcement',
    'overfiting': 'overfitting', 'overfit': 'overfit',
    'underfiting': 'underfitting',
    'backpropogation': 'backpropagation', 'backprop': 'backpropagation', 'back-propagation': 'backpropagation',
    'gradiant': 'gradient', 'gradiants': 'gradients', 'gradint': 'gradient',
    'descent': 'descent', 'decent': 'descent', 'decsent': 'descent',
    'regresion': 'regression', 'regresstion': 'regression',
    'clasification': 'classification', 'clasifcation': 'classification', 'classifcation': 'classification',
    'clustring': 'clustering', 'clstrng': 'clustering',
    'optmizer': 'optimizer', 'optimizr': 'optimizer',
    'probabilty': 'probability', 'probablity': 'probability',
    'dimention': 'dimension', 'dimentions': 'dimensions', 'dimenshion': 'dimension',
    'vecter': 'vector', 'vectrs': 'vectors', 'vektor': 'vector',
    'embeding': 'embedding', 'embedings': 'embeddings', 'embeddng': 'embedding',
    'similarty': 'similarity', 'similiarity': 'similarity', 'similartiy': 'similarity',
    'retrival': 'retrieval', 'retreival': 'retrieval', 'retrive': 'retrieve',
    'hybird': 'hybrid', 'hibred': 'hybrid',
    'crossencoder': 'cross-encoder', 'biencoder': 'bi-encoder',
}

# Common merged words to separate
MERGED_COMPOUNDS = {
    'machinelearning': 'machine learning',
    'deeplearning': 'deep learning',
    'neuralnetwork': 'neural network',
    'neuralnetworks': 'neural networks',
    'naturallanguage': 'natural language',
    'supervisedlearning': 'supervised learning',
    'unsupervisedlearning': 'unsupervised learning',
    'reinforcementlearning': 'reinforcement learning',
    'gradientdescent': 'gradient descent',
    'backpropagation': 'backpropagation',
    'languagemodel': 'language model',
    'languagemodels': 'large language models',
}

# Domain vocabulary for minimum edit-distance spelling suggestion
VOCABULARY = [
    'machine', 'learning', 'artificial', 'intelligence', 'deep', 'neural', 'network',
    'networks', 'perceptron', 'recurrent', 'convolutional', 'transformer', 'transformers',
    'supervised', 'unsupervised', 'reinforcement', 'overfitting', 'underfitting',
    'backpropagation', 'gradient', 'descent', 'regression', 'classification',
    'clustering', 'optimizer', 'probability', 'dimension', 'vector', 'vectors',
    'embedding', 'embeddings', 'similarity', 'retrieval', 'hybrid', 'cross-encoder',
    'features', 'architecture', 'algorithm', 'difference', 'example', 'explain',
    'definition', 'summary', 'function', 'performance', 'evaluation', 'metric',
    'accuracy', 'precision', 'recall', 'f1-score', 'validation', 'dataset', 'training',
    'testing', 'epoch', 'batch', 'weight', 'weights', 'bias', 'biases', 'activation',
    'loss', 'cost', 'convergence', 'regularization', 'dropout', 'normalization',
]


def _damerau_levenshtein(s1: str, s2: str) -> int:
    """Computes Damerau-Levenshtein distance (substitutions, insertions, deletions, transpositions)."""
    d: dict[tuple[int, int], int] = {}
    len1, len2 = len(s1), len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,          # deletion
                d[(i, j - 1)] + 1,          # insertion
                d[(i - 1, j - 1)] + cost,   # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 1)  # transposition

    return d[(len1 - 1, len2 - 1)]


class SpellingCorrector:
    """Intelligent Domain-Aware Spelling Corrector."""

    @staticmethod
    def correct(text: str) -> tuple[str, list[dict[str, Any]], bool]:
        """Detects and corrects spelling errors in the user query.
        
        Returns:
            (corrected_text, list_of_corrections, has_correction)
        """
        if not text or not text.strip():
            return text, [], False

        original = text.strip()
        tokens_with_delimiters = re.split(r'(\s+|[.,!?;:()"]+)', original)
        corrections: list[dict[str, Any]] = []
        result_tokens: list[str] = []

        i = 0
        while i < len(tokens_with_delimiters):
            segment = tokens_with_delimiters[i]

            # Pass whitespace and punctuation delimiters through untouched
            if not segment or re.match(r'^[\s.,!?;:()"]+$', segment):
                result_tokens.append(segment)
                i += 1
                continue

            clean_lower = segment.lower().strip()

            # Check protected technical terms & code keywords
            if clean_lower in PROTECTED_TERMS:
                result_tokens.append(segment)
                i += 1
                continue

            # Check merged compounds (e.g. machinelearning -> machine learning)
            if clean_lower in MERGED_COMPOUNDS:
                replacement = MERGED_COMPOUNDS[clean_lower]
                corrections.append({
                    'original': segment,
                    'corrected': replacement,
                    'type': 'merged_compound',
                })
                result_tokens.append(replacement)
                i += 1
                continue

            # Check dictionary map for known typo
            if clean_lower in SPELLING_TYPO_MAP:
                replacement = SPELLING_TYPO_MAP[clean_lower]
                # Preserve capitalization of original if titled or all uppercase
                if segment.isupper() and len(segment) > 1:
                    replacement = replacement.upper()
                elif segment.istitle():
                    replacement = replacement.title()

                corrections.append({
                    'original': segment,
                    'corrected': replacement,
                    'type': 'typo_map',
                })
                result_tokens.append(replacement)
                i += 1
                continue

            # For words >= 4 chars not in protected terms, test Damerau-Levenshtein distance <= 2
            if len(clean_lower) >= 4 and not re.search(r'\d', clean_lower):
                best_match = None
                best_dist = 999
                for cand in VOCABULARY:
                    # Length difference filter for performance
                    if abs(len(cand) - len(clean_lower)) > 2:
                        continue
                    dist = _damerau_levenshtein(clean_lower, cand)
                    max_allowed = 1 if len(clean_lower) <= 5 else 2
                    if dist <= max_allowed and dist < best_dist:
                        best_dist = dist
                        best_match = cand

                if best_match and best_match != clean_lower:
                    replacement = best_match
                    if segment.istitle():
                        replacement = replacement.title()
                    corrections.append({
                        'original': segment,
                        'corrected': replacement,
                        'type': 'fuzzy_distance',
                        'distance': best_dist,
                    })
                    result_tokens.append(replacement)
                    i += 1
                    continue

            result_tokens.append(segment)
            i += 1

        corrected_text = ''.join(result_tokens).strip()
        has_correction = len(corrections) > 0 and corrected_text.lower() != original.lower()
        return corrected_text, corrections, has_correction
