"""Separate retrieval and answer quality evaluation."""
import math
from .errors import AppError
from .schemas import AskRequest
from .service import uid

def evaluate(service, cases):
    rows = []
    for case in cases:
        try:
            response = service.answer(AskRequest(question=case.question), persist=False)
            names = [s['name'] for s in response['sources']]
            rank = next((i + 1 for i, n in enumerate(names) if n == case.expected_document), None) if case.expected_document else None
            keyword = case.expected_text.casefold() in response['answer'].casefold() if case.expected_text else None
            abstained = response['status'] == 'not_found'

            # Evaluate retrieval quality
            retrieval_checks = [abstained] if case.should_abstain else [not abstained]
            if case.expected_document:
                retrieval_checks.append(rank is not None)
            retrieval_passed = all(retrieval_checks)

            # Evaluate answer quality
            answer_checks = [abstained] if case.should_abstain else [not abstained]
            if keyword is not None:
                answer_checks.append(keyword)
            answer_passed = all(answer_checks)

            overall_passed = retrieval_passed and answer_passed

            rows.append({
                'question': case.question,
                'passed': overall_passed,
                'retrieval_passed': retrieval_passed,
                'answer_passed': answer_passed,
                'rank': rank,
                'document_hit_at_k': float(rank is not None) if case.expected_document else None,
                'reciprocal_rank': 1 / rank if rank else (0 if case.expected_document else None),
                'ndcg_at_k': 1 / math.log2(rank + 1) if rank else (0 if case.expected_document else None),
                'expected_text_found': keyword,
                'abstained': abstained,
                'latency_ms': response['latency_ms'],
                'error': None
            })
        except AppError as exc:
            rows.append({
                'question': case.question,
                'passed': False,
                'retrieval_passed': False,
                'answer_passed': False,
                'error': exc.code
            })

    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    return {
        'id': uid(),
        'cases': len(rows),
        'passed': sum(r['passed'] for r in rows),
        'retrieval_accuracy': mean('retrieval_passed'),
        'answer_accuracy': mean('answer_passed'),
        'document_hit_at_k': mean('document_hit_at_k'),
        'mrr': mean('reciprocal_rank'),
        'ndcg_at_k': mean('ndcg_at_k'),
        'mean_latency_ms': mean('latency_ms'),
        'notes': 'Retrieval and answer quality evaluated separately before changing chunking, settings or reranking.',
        'results': rows
    }
