from typing import Literal
from pydantic import AliasChoices, BaseModel, Field, field_validator

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] = Field(
        default_factory=list, max_length=100,
        validation_alias=AliasChoices('document_ids', 'documentIds'))
    conversation_id: str | None = Field(
        default=None, max_length=80,
        validation_alias=AliasChoices('conversation_id', 'conversationId'))

    model_config = {'populate_by_name': True}

    @field_validator('question')
    @classmethod
    def clean_question(cls, value):
        if not value.strip():
            raise ValueError('Enter a question.')
        return value.strip()

class Settings(BaseModel):
    top_k: int = Field(default=5, ge=1, le=10)
    candidate_k: int = Field(default=30, ge=10, le=50)
    min_similarity: float = Field(default=0.15, ge=0, le=1)
    rerank: bool = True
    system_prompt: str | None = None
    model: str | None = None

class DocumentUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=255)

class Feedback(BaseModel):
    rating: Literal['helpful', 'unhelpful']
    comment: str = Field(default='', max_length=1000)

class EvalCase(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    expected_document: str | None = Field(default=None, max_length=255)
    expected_text: str | None = Field(default=None, max_length=1000)
    should_abstain: bool = False

class EvalRequest(BaseModel):
    cases: list[EvalCase] = Field(min_length=1, max_length=50)
