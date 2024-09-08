
from pydantic import BaseModel

from app.types.db import EmbeddingOutput


class DraftInput(BaseModel):
    email_body: str


class QuestionOutput(BaseModel):
    question: str
    answer: str
    embeddings: list[EmbeddingOutput]

class DraftOutput(BaseModel):
    draft: str
    email_body: str
    questions: list[QuestionOutput] | None
    embeddings: list[EmbeddingOutput] | None

