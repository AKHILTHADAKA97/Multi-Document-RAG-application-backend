"""Advanced Intelligent Query Understanding Agent for Multi-Document RAG.

Provides:
1. Smart Query Normalization & Typo Correction (Spelling, Grammar, Abbreviations).
2. Multilingual & Mixed-language understanding (English, Hinglish, Telugu-English transliterated).
3. Conversational Context & Reference Resolution (it, its, this, that, the second one).
4. Question Intent Classification & Search Entity Extraction.
5. Duplicate & Related Question Detection.
6. AI Follow-up Question Suggestion Agent (3-5 relevant suggestions).
7. Intelligent Ambiguity & Clarification Agent.
"""

from __future__ import annotations

import re
from typing import Any


# ==============================================================================
# 1. DOMAIN VOCABULARY & SPELLING CORRECTION MAPS
# ==============================================================================

TECHNICAL_ABBREVIATIONS = {
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
    'lstms': 'Long Short-Term Memory networks',
    'gru': 'Gated Recurrent Unit',
    'rag': 'Retrieval-Augmented Generation',
    'gan': 'Generative Adversarial Network',
    'gans': 'Generative Adversarial Networks',
    'bert': 'BERT (Bidirectional Encoder Representations from Transformers)',
    'gpt': 'Generative Pre-trained Transformer',
    'sgd': 'Stochastic Gradient Descent',
    'adam': 'Adam optimizer',
    'bptt': 'Backpropagation Through Time',
    'relu': 'Rectified Linear Unit (ReLU)',
    'mse': 'Mean Squared Error',
    'svm': 'Support Vector Machine',
    'pca': 'Principal Component Analysis',
    'rl': 'Reinforcement Learning',
}

COMMON_TYPO_MAP = {
    # Question words and common verbs
    'wht': 'what', 'wat': 'what', 'wot': 'what', 'waht': 'what',
    'hw': 'how', 'hwo': 'how', 'ho': 'how',
    'whr': 'where', 'wher': 'where', 'wer': 'where',
    'wn': 'when', 'wen': 'when',
    'y': 'why', 'whi': 'why',
    'bcz': 'because', 'bcoz': 'because', 'becoz': 'because', 'cuz': 'because',
    'plz': 'please', 'pls': 'please', 'plez': 'please',
    'tel': 'tell', 'tl': 'tell',
    'giv': 'give', 'gve': 'give',
    'shw': 'show', 'sho': 'show',
    'wrks': 'works', 'wrk': 'work', 'workng': 'working',
    'dows': 'does', 'dose': 'does', 'duz': 'does',
    'bteeen': 'between', 'betwen': 'between', 'btwn': 'between',
    'imprtnt': 'important', 'importnt': 'important',
    'diffrence': 'difference', 'diffrnce': 'difference', 'diffrent': 'different',
    'exmple': 'example', 'exmpl': 'example', 'eg': 'example',
    'explan': 'explain', 'expaln': 'explain', 'expln': 'explain',
    'defne': 'define', 'defination': 'definition', 'definiton': 'definition',
    'sumarize': 'summarize', 'sumary': 'summary', 'sumry': 'summary',
    'fatures': 'features', 'feautres': 'features', 'fetures': 'features',
    'functon': 'function', 'fnction': 'function',
    'algoritm': 'algorithm', 'algorthm': 'algorithm', 'algotithm': 'algorithm',
    'archtecture': 'architecture', 'architechture': 'architecture',

    # Technical & ML domain words
    'machne': 'machine', 'mchine': 'machine', 'machin': 'machine', 'macheen': 'machine',
    'lerning': 'learning', 'lrning': 'learning', 'learnng': 'learning', 'lernin': 'learning',
    'artficial': 'artificial', 'artifical': 'artificial', 'artifitial': 'artificial',
    'intligenc': 'intelligence', 'inteligence': 'intelligence', 'intellegence': 'intelligence',
    'deeep': 'deep', 'dep': 'deep', 'depp': 'deep',
    'neurall': 'neural', 'nueral': 'neural', 'neurl': 'neural',
    'netwrk': 'network', 'netwrok': 'network', 'netowrk': 'network', 'netwrks': 'networks',
    'percepton': 'perceptron', 'perceptrn': 'perceptron',
    'reccurent': 'recurrent', 'recurent': 'recurrent',
    'convolutinal': 'convolutional', 'convoluton': 'convolution',
    'transfomer': 'transformer', 'transfomers': 'transformers',
    'suparvised': 'supervised', 'supervized': 'supervised',
    'unsuparvised': 'unsupervised', 'unsupervized': 'unsupervised',
    'reinforcment': 'reinforcement', 'reinforcemnt': 'reinforcement',
    'overfiting': 'overfitting', 'overfit': 'overfit',
    'underfiting': 'underfitting',
    'backpropogation': 'backpropagation', 'backprop': 'backpropagation',
    'gradiant': 'gradient', 'gradiants': 'gradients',
    'hyperparamter': 'hyperparameter', 'hyperparamters': 'hyperparameters',
    'paramater': 'parameter', 'paramters': 'parameters',
    'acuracy': 'accuracy', 'accurasy': 'accuracy',
    'clasification': 'classification', 'clasify': 'classify',
    'regresion': 'regression',
    'evalution': 'evaluation', 'evalute': 'evaluate',
    'modle': 'model', 'modl': 'model', 'modles': 'models',
    'pytn': 'python', 'pythn': 'python', 'pyton': 'python',
    'vectr': 'vector', 'vectrs': 'vectors',
    'embeding': 'embedding', 'embedings': 'embeddings',
}

# Multilingual / Transliterated phrases (Telugu-English & Hinglish)
MULTILINGUAL_INTENT_PATTERNS = [
    # Telugu-English patterns
    (re.compile(r'\b(?:idi|idhi)\s+(?:explain\s+chey|cheppu|kenti)\b', re.IGNORECASE), 'Explain this'),
    (re.compile(r'\b(?:deeni|dheeni|deni)\s+gurinchi\s+(?:cheppu|explain\s+chey)\b', re.IGNORECASE), 'Tell me about this'),
    (re.compile(r'\b(?:deeni|dheeni|deni)\s+features\s*(?:cheppu|enti)?\b', re.IGNORECASE), 'What are its features?'),
    (re.compile(r'\b(?:idi|idhi)\s+ela\s+pani\s*(?:chesthundi|chesthadi)\b', re.IGNORECASE), 'How does this work?'),
    (re.compile(r'\b(?:rendodi|second\s*di|2nd\s*one)\s*(?:explain\s+chey|cheppu)?\b', re.IGNORECASE), 'Explain the second one'),
    (re.compile(r'\b(?:modatidi|first\s*di|1st\s*one)\s*(?:explain\s+chey|cheppu)?\b', re.IGNORECASE), 'Explain the first one'),
    (re.compile(r'\b(?:aakaridi|last\s*di|last\s*one)\s*(?:explain\s+chey|cheppu)?\b', re.IGNORECASE), 'Explain the last one'),
    (re.compile(r'\b(?:inkonchem|ekkuva)\s+(?:explain\s+chey|cheppu)\b', re.IGNORECASE), 'Explain in more detail'),
    (re.compile(r'\boka\s+example\s+(?:ivvu|cheppu)\b', re.IGNORECASE), 'Give an example of this'),
    (re.compile(r'\b(?:enti|emiti)\b', re.IGNORECASE), 'what is'),

    # Hinglish patterns
    (re.compile(r'\byeh\s+(?:kya\s+hai|kya\s+hota\s+hai)\b', re.IGNORECASE), 'What is this?'),
    (re.compile(r'\biska\s+matlab\s+kya\s+hai\b', re.IGNORECASE), 'What does this mean?'),
    (re.compile(r'\biske\s+features\s+batao\b', re.IGNORECASE), 'What are its features?'),
    (re.compile(r'\byeh\s+kaise\s+kaam\s+karta\s+hai\b', re.IGNORECASE), 'How does this work?'),
    (re.compile(r'\b(?:dusra\s+wala|second\s+wala)\s*(?:batao|explain\s+karo)?\b', re.IGNORECASE), 'Explain the second one'),
    (re.compile(r'\b(?:pehla\s+wala|first\s+wala)\s*(?:batao|explain\s+karo)?\b', re.IGNORECASE), 'Explain the first one'),
    (re.compile(r'\baur\s+detail\s+me\s+batao\b', re.IGNORECASE), 'Explain in more detail'),
    (re.compile(r'\bek\s+example\s+do\b', re.IGNORECASE), 'Give an example of this'),
]


# ==============================================================================
# 2. QUERY NORMALIZER & SPELLING CORRECTION
# ==============================================================================

class QueryNormalizer:
    """Corrects spelling, grammar slips, slang, and transliterated phrases without altering user intent."""

    @classmethod
    def normalize(cls, raw_query: str) -> dict[str, Any]:
        text = (raw_query or '').strip()
        if not text:
            return {'original': '', 'corrected': '', 'is_corrected': False, 'notes': []}

        original = text
        notes = []

        # 1. Translate multilingual/transliterated phrases
        for pattern, replacement in MULTILINGUAL_INTENT_PATTERNS:
            if pattern.search(text):
                matched_str = pattern.search(text).group(0)
                text = pattern.sub(replacement, text).strip()
                notes.append(f"Interpreted conversational phrase: '{matched_str}' -> '{replacement}'")

        # 2. Tokenize while preserving punctuation
        tokens = re.findall(r'[A-Za-z0-9_#+\-]+|[^\s\w]', text)
        corrected_tokens = []

        for token in tokens:
            lower = token.lower()
            # Check typo map
            if lower in COMMON_TYPO_MAP:
                replacement = COMMON_TYPO_MAP[lower]
                # Preserve capitalization
                if token.isupper() and len(token) <= 4:
                    replacement = replacement.upper()
                elif token[0].isupper():
                    replacement = replacement.capitalize()
                corrected_tokens.append(replacement)
                if lower != replacement.lower():
                    notes.append(f"Corrected: '{token}' -> '{replacement}'")
            # Check technical abbreviation (if standalone word)
            elif lower in TECHNICAL_ABBREVIATIONS and len(tokens) <= 5:
                full_name = TECHNICAL_ABBREVIATIONS[lower]
                corrected_tokens.append(f"{token.upper()} ({full_name})")
                notes.append(f"Expanded abbreviation: '{token.upper()}' -> '{full_name}'")
            else:
                corrected_tokens.append(token)

        reconstructed = ""
        for i, t in enumerate(corrected_tokens):
            if i > 0 and not re.match(r'^[,\.\?!;:)]', t) and not reconstructed.endswith('('):
                reconstructed += " "
            reconstructed += t

        # Clean multiple spaces and fix casing
        reconstructed = re.sub(r'\s+', ' ', reconstructed).strip()
        if reconstructed and reconstructed[0].islower():
            reconstructed = reconstructed[0].upper() + reconstructed[1:]

        # Add question mark if it is clearly an interrogative question without punctuation
        if re.match(r'^(what|how|why|when|where|who|which|can|is|are|does|do|explain|describe)\b', reconstructed, re.IGNORECASE):
            if not reconstructed.endswith(('?', '.', '!')):
                reconstructed += '?'

        is_corrected = reconstructed.lower() != original.lower() and len(notes) > 0

        return {
            'original': original,
            'corrected': reconstructed,
            'is_corrected': is_corrected,
            'notes': notes,
        }


# ==============================================================================
# 3. CONVERSATIONAL CONTEXT RESOLVER
# ==============================================================================

FOLLOW_UP_PRONOUNS = re.compile(
    r'\b(it|its|this|that|these|those|they|them|the\s+former|the\s+latter|the\s+second\s+(?:one|feature|type|point)?|the\s+first\s+(?:one|feature|type|point)?|the\s+third\s+(?:one|feature|type|point)?|the\s+last\s+(?:one|feature|type|point)?|more|example|difference)\b',
    re.IGNORECASE
)

ORDINAL_MAP = {
    'first': 0, '1st': 0, 'modatidi': 0, 'pehla': 0,
    'second': 1, '2nd': 1, 'rendodi': 1, 'dusra': 1,
    'third': 2, '3rd': 2, 'moododi': 2, 'teesra': 2,
    'fourth': 3, '4th': 3, 'naalgodi': 3, 'chautha': 3,
    'last': -1, 'aakaridi': -1, 'aakhri': -1,
}

class ConversationContextResolver:
    """Resolves conversational references, pronouns, and ordinal pointers using chat history."""

    @classmethod
    def resolve(cls, question: str, history: list[dict[str, Any]] | None) -> dict[str, Any]:
        if not history:
            return {
                'resolved_query': question,
                'is_follow_up': False,
                'referenced_entity': None,
                'context_carryover': '',
            }

        q_lower = question.lower()
        last_turn = history[-1] if history else {}
        last_question = last_turn.get('question', '')
        last_answer = last_turn.get('answer', '')

        referenced_entity = None
        resolved = question

        # 1. Check for ordinal resolution (e.g. "Explain the second one")
        ordinal_match = re.search(r'\b(first|second|third|fourth|last|1st|2nd|3rd|4th)\b', q_lower)
        if ordinal_match:
            ord_word = ordinal_match.group(1).lower()
            target_idx = ORDINAL_MAP.get(ord_word, 0)
            # Try to parse list items from previous answer
            items = cls._extract_enumerated_items(last_answer)
            if items:
                if target_idx == -1 and items:
                    referenced_entity = items[-1]
                elif 0 <= target_idx < len(items):
                    referenced_entity = items[target_idx]

        # 2. Check for pronoun resolution ("What are its features?", "How does it work?", "Explain it")
        if not referenced_entity and FOLLOW_UP_PRONOUNS.search(q_lower):
            # Extract main topic from previous question
            prev_subject = cls._extract_subject(last_question)
            if prev_subject:
                referenced_entity = prev_subject

        # 3. Construct resolved search query
        is_follow_up = False
        context_carryover = ""

        if referenced_entity:
            is_follow_up = True
            context_carryover = referenced_entity
            # Substitute pronoun or append entity
            if re.search(r'\b(the\s+(?:first|second|third|fourth|last)\s*(?:one|feature|type|point)?)\b', q_lower):
                resolved = re.sub(
                    r'\bthe\s+(?:first|second|third|fourth|last)\s*(?:one|feature|type|point)?\b',
                    referenced_entity,
                    question,
                    flags=re.IGNORECASE
                )
            elif re.search(r'\b(its|it|this|that)\b', q_lower):
                resolved = re.sub(
                    r'\b(its|it|this|that)\b',
                    f"{referenced_entity}",
                    question,
                    flags=re.IGNORECASE
                )
            else:
                resolved = f"{question} ({referenced_entity})"
        elif len(question.split()) <= 4 and re.search(r'\b(more|example|details|why|how)\b', q_lower):
            is_follow_up = True
            prev_subject = cls._extract_subject(last_question)
            if prev_subject:
                resolved = f"{question} regarding {prev_subject}"
                context_carryover = prev_subject

        return {
            'resolved_query': resolved,
            'is_follow_up': is_follow_up,
            'referenced_entity': referenced_entity,
            'context_carryover': context_carryover,
        }

    @classmethod
    def _extract_subject(cls, text: str) -> str | None:
        """Extracts the probable subject/topic noun phrase from a question."""
        cleaned = re.sub(r'^(what is|what are|explain|describe|tell me about|how does|define)\s+', '', text, flags=re.IGNORECASE).strip(' ?.')
        words = cleaned.split()
        if words:
            # Drop trailing filler
            if words[-1].lower() in {'work', 'works', 'do', 'mean', 'differ'}:
                cleaned = ' '.join(words[:-1])
            return cleaned.strip()
        return None

    @classmethod
    def _extract_enumerated_items(cls, text: str) -> list[str]:
        """Extracts numbered or bulleted items from an AI answer."""
        if not text:
            return []
        items = []
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r'^(?:\d+[\.\)]|\-|\*)\s+(.+)$', line)
            if m:
                content = m.group(1).strip()
                title_match = re.match(r'^\*?\*?([A-Za-z0-9_\s\(\)\/]+?)\*?\*?\s*[:\–\—]\s*(.*)$', content)
                if title_match:
                    item_title = title_match.group(1).strip()
                    if len(item_title) >= 3:
                        items.append(item_title)
                        continue
                cleaned = re.sub(r'[*_`]', '', content).strip()
                if cleaned and len(cleaned) < 120 and not cleaned.lower().startswith(('note', 'for example', 'e.g.')):
                    items.append(cleaned)
        return items


# ==============================================================================
# 4. INTENT CLASSIFICATION & DUPLICATE QUESTION DETECTION
# ==============================================================================

class IntentClassifier:
    """Classifies user intent to determine RAG retrieval strategy and answer formatting."""

    INTENTS = {
        'definition': re.compile(r'\b(what is|what are|define|definition|meaning of|explain what)\b', re.IGNORECASE),
        'comparison': re.compile(r'\b(difference between|compare|contrast|versus|vs\.?|how does .* differ)\b', re.IGNORECASE),
        'technical_explanation': re.compile(r'\b(how (?:does|do|can|works)|architecture|internal|equation|mechanism|workflow)\b', re.IGNORECASE),
        'code_implementation': re.compile(r'\b(code|implementation|implement|pytorch|tensorflow|python snippet|example code)\b', re.IGNORECASE),
        'summary': re.compile(r'\b(summarize|summary|overview|key takeaways|tl;?dr|recap|outline)\b', re.IGNORECASE),
        'factual_lookup': re.compile(r'\b(page|table|figure|section|statistic|number|dataset|accuracy|loss|metric)\b', re.IGNORECASE),
        'conversational': re.compile(r'^(hi|hello|hey|good\s+(?:morning|afternoon|evening)|thanks|thank you)\b', re.IGNORECASE),
    }

    @classmethod
    def classify(cls, query: str) -> dict[str, Any]:
        q_lower = query.strip().lower()
        matched_intent = 'general'

        for intent, pattern in cls.INTENTS.items():
            if pattern.search(q_lower):
                matched_intent = intent
                break

        tokens = [t for t in re.findall(r'\b[A-Za-z0-9_\-]{2,}\b', q_lower) if t not in {'the', 'what', 'and', 'for', 'with', 'does', 'how', 'are'}]

        return {
            'intent': matched_intent,
            'is_definition': matched_intent == 'definition',
            'is_comparison': matched_intent == 'comparison',
            'is_code': matched_intent == 'code_implementation',
            'is_summary': matched_intent == 'summary',
            'search_tokens': tokens[:8],
        }


class DuplicateDetector:
    """Detects whether a similar or identical question was already asked in the conversation."""

    @classmethod
    def check(cls, current_query: str, history: list[dict[str, Any]] | None) -> dict[str, Any]:
        if not history:
            return {'is_duplicate': False, 'is_related': False, 'matched_turn': None}

        curr_norm = re.sub(r'[^\w\s]', '', current_query.lower()).strip()
        curr_words = set(curr_norm.split())

        for idx, turn in enumerate(reversed(history)):
            prev_q = turn.get('question', '')
            prev_norm = re.sub(r'[^\w\s]', '', prev_q.lower()).strip()
            if not prev_norm:
                continue

            # Exact match
            if curr_norm == prev_norm:
                return {'is_duplicate': True, 'is_related': True, 'matched_turn': len(history) - 1 - idx}

            # Jaccard word similarity
            prev_words = set(prev_norm.split())
            if curr_words and prev_words:
                inter = len(curr_words.intersection(prev_words))
                union = len(curr_words.union(prev_words))
                jaccard = inter / union if union else 0
                if jaccard >= 0.85:
                    return {'is_duplicate': True, 'is_related': True, 'matched_turn': len(history) - 1 - idx}
                elif jaccard >= 0.50:
                    return {'is_duplicate': False, 'is_related': True, 'matched_turn': len(history) - 1 - idx}

        return {'is_duplicate': False, 'is_related': False, 'matched_turn': None}


# ==============================================================================
# 5. AI SUGGESTED QUESTIONS AGENT
# ==============================================================================

class QuestionSuggestionAgent:
    """Generates 3 to 5 smart, non-repetitive follow-up questions tailored to the topic and generated answer."""

    SUGGESTION_TEMPLATES = {
        'machine learning': [
            "How does Machine Learning work under the hood?",
            "What are the main types of Machine Learning (Supervised, Unsupervised, RL)?",
            "What is the difference between AI and Machine Learning?",
            "What are real-world applications of Machine Learning?",
            "Which Python libraries are used for Machine Learning?",
        ],
        'deep learning': [
            "What is the difference between Deep Learning and traditional Machine Learning?",
            "How do Artificial Neural Networks (ANNs) learn from data?",
            "What are Convolutional Neural Networks (CNNs) used for?",
            "How do activation functions introduce non-linearity in Deep Learning?",
            "What is Transfer Learning and how does it save compute?",
        ],
        'mlp': [
            "How does the backpropagation algorithm train an MLP?",
            "What role do hidden layers and neurons play in an MLP?",
            "What is the difference between an MLP and a CNN?",
            "How do activation functions like ReLU prevent vanishing gradients?",
            "What are practical examples where MLPs are deployed?",
        ],
        'multi-layer perceptron': [
            "How does backpropagation compute gradient descent in an MLP?",
            "What is the difference between an MLP and a standard perceptron?",
            "Why are activation functions essential between layers in an MLP?",
            "How do hidden layer dimensions affect model capacity?",
        ],
        'lstm': [
            "How do the input, forget, and output gates operate in an LSTM?",
            "How does an LSTM resolve the vanishing gradient problem of traditional RNNs?",
            "What is the difference between LSTM and GRU architectures?",
            "What are real-world time-series and NLP applications of LSTMs?",
            "How does an LSTM maintain its internal cell state?",
        ],
        'neural network': [
            "How do weights and biases adjust during neural network training?",
            "What is the purpose of the loss function in neural networks?",
            "What is the difference between forward propagation and backpropagation?",
            "How can we prevent overfitting in deep neural networks?",
        ],
        'python': [
            "What are the key features and advantages of Python?",
            "How is Python used in Data Science and Machine Learning?",
            "What is the difference between Python lists and NumPy arrays?",
            "What are Python's object-oriented programming concepts?",
        ],
        'rag': [
            "How does Retrieval-Augmented Generation (RAG) prevent hallucinations?",
            "What is the role of Vector Embeddings in RAG pipelines?",
            "What is the difference between Dense Retrieval and Sparse BM25 Search?",
            "How does Cross-Encoder Reranking improve RAG accuracy?",
        ],
    }

    @classmethod
    def generate(
        cls,
        question: str,
        answer: str,
        sources: list[dict[str, Any]] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        q_lower = question.lower()

        # 1. Check direct topic matches in templates
        for topic, suggestions in cls.SUGGESTION_TEMPLATES.items():
            if topic in q_lower:
                return suggestions[:4]

        # 2. Extract concepts from the answer (bold terms, bullet points, headings)
        concepts = []
        if answer:
            # Find bold words: **Concept**
            bolds = re.findall(r'\*\*([A-Za-z0-9_\s\-\(\)]+?)\*\*', answer)
            for b in bolds:
                cleaned = b.strip()
                if 3 <= len(cleaned) <= 40 and cleaned.lower() not in q_lower and cleaned not in concepts:
                    concepts.append(cleaned)

        # 3. Formulate dynamic relevant suggestions
        clean_q = re.sub(r'^(what is|what are|explain|describe|define)\s+', '', question, flags=re.IGNORECASE).strip(' ?.')
        suggestions = []

        if concepts:
            top_concept = concepts[0]
            suggestions.append(f"How does {top_concept} relate to {clean_q}?")
            if len(concepts) > 1:
                suggestions.append(f"What is the difference between {clean_q} and {concepts[1]}?")

        suggestions.append(f"What are practical real-world examples of {clean_q}?")
        suggestions.append(f"How is {clean_q} implemented in production systems?")
        suggestions.append(f"What are the main advantages and limitations of {clean_q}?")

        # Filter out anything identical to previous questions in conversation
        past_questions = {turn.get('question', '').lower() for turn in (history or [])}
        filtered = [s for s in suggestions if s.lower() not in past_questions]

        return filtered[:4]


# ==============================================================================
# 6. UNIFIED QUERY UNDERSTANDING AGENT ENTRY POINT
# ==============================================================================

class QueryUnderstandingAgent:
    """Master Agent coordinating normalization, context resolution, intent routing, and question suggestions."""

    @classmethod
    def process_query(
        cls,
        raw_question: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Main pipeline to analyze and optimize a user query before retrieval and generation."""
        # 1. Normalization & Spelling Correction
        norm_result = QueryNormalizer.normalize(raw_question)
        clean_text = norm_result['corrected']

        # 2. Conversational Context & Follow-up Resolution
        ctx_result = ConversationContextResolver.resolve(clean_text, conversation_history)
        retrieval_query = ctx_result['resolved_query']

        # 3. Intent & Topic Classification
        intent_info = IntentClassifier.classify(retrieval_query)

        # 4. Duplicate Question Detection
        dup_info = DuplicateDetector.check(clean_text, conversation_history)

        return {
            'original_question': raw_question,
            'corrected_query': clean_text if norm_result['is_corrected'] else None,
            'is_corrected': norm_result['is_corrected'],
            'correction_notes': norm_result['notes'],
            'retrieval_query': retrieval_query,
            'is_follow_up': ctx_result['is_follow_up'],
            'referenced_entity': ctx_result['referenced_entity'],
            'intent': intent_info['intent'],
            'intent_info': intent_info,
            'is_duplicate': dup_info['is_duplicate'],
            'is_related': dup_info['is_related'],
        }

    @classmethod
    def generate_suggestions(
        cls,
        question: str,
        answer: str,
        sources: list[dict[str, Any]] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        return QuestionSuggestionAgent.generate(question, answer, sources, history)
