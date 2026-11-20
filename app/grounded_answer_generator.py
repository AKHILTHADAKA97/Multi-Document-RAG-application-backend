"""Document-Grounded Answer Generator.

Synthesizes factual, context-grounded responses strictly adhering to
uploaded document excerpts. When evidence is insufficient or missing,
transparently reports what was found and what is missing, with zero
hallucinations, zero invented citations, and no unsolicited question suggestions.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from .generation import generate, generate_stream
from .synthesis import compose

REFUSAL_MESSAGE = "I couldn't find sufficient information about this question in your uploaded documents."


class GroundedAnswerGenerator:
    """Document-Grounded Response Generator."""

    @classmethod
    def generate_answer(
        cls,
        config: Any,
        question: str,
        sources: list[dict[str, Any]],
        diagnostics: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        context: str = "",
        system_prompt: str | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """Generates grounded answer from verified sources.
        
        If evidence is missing, returns REFUSAL_MESSAGE with status='not_found'.
        """
        diag = diagnostics or {}
        has_evidence = diag.get('has_sufficient_evidence', bool(sources))
        is_partial = diag.get('is_partial', False)

        if not has_evidence or not sources:
            return {
                'answer': REFUSAL_MESSAGE,
                'citations': [],
                'status': 'not_found',
                'answer_type': 'not_found',
                'confidence': 'insufficient',
                'is_partial': False,
                'usage': None,
            }

        # Format partial instruction to prompt if evidence is partial
        custom_system = system_prompt
        if is_partial and custom_system:
            custom_system += "\nNote: The uploaded documents contain only partial information on this topic. Explain clearly what information was found in the documents and note what is missing."

        result = generate(
            config,
            question,
            sources,
            history=history,
            context=context,
            system_prompt=custom_system,
            model_override=model_override,
        )

        # Validate that cited sources actually exist in sources
        allowed = {s.get('citation') for s in sources}
        citations = [c for c in (result.get('citations') or []) if c in allowed]
        if not citations:
            inline = re.findall(r'\[(S?\d+)\]', result.get('answer', ''))
            citations = [c for c in inline if c in allowed]

        return {
            **result,
            'citations': list(dict.fromkeys(citations)),
            'is_partial': is_partial,
        }

    @classmethod
    def generate_answer_stream(
        cls,
        config: Any,
        question: str,
        sources: list[dict[str, Any]],
        diagnostics: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        context: str = "",
        system_prompt: str | None = None,
        model_override: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Streams grounded answer tokens.
        
        If evidence is missing, streams REFUSAL_MESSAGE.
        """
        diag = diagnostics or {}
        has_evidence = diag.get('has_sufficient_evidence', bool(sources))
        is_partial = diag.get('is_partial', False)

        if not has_evidence or not sources:
            # Yield token for the refusal message
            yield {'type': 'token', 'token': REFUSAL_MESSAGE}
            yield {
                'type': 'done',
                'answer': REFUSAL_MESSAGE,
                'citations': [],
                'status': 'not_found',
                'answer_type': 'not_found',
                'confidence': 'insufficient',
                'is_partial': False,
                'ttft_ms': 0,
            }
            return

        custom_system = system_prompt
        if is_partial and custom_system:
            custom_system += "\nNote: The uploaded documents contain only partial information on this topic. Explain clearly what information was found in the documents and note what is missing."

        for event in generate_stream(
            config,
            question,
            sources,
            history=history,
            context=context,
            system_prompt=custom_system,
            model_override=model_override,
        ):
            yield event
