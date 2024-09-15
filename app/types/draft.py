from pydantic import BaseModel

from app.types.db import SourceOutput


class DraftInput(BaseModel):
    from_user: str | None
    email_body: str


class QuestionOutput(BaseModel):
    question: str
    answer: str
    sources: list[SourceOutput]


class DraftOutput(BaseModel):
    draft: str
    email_body: str
    questions: list[QuestionOutput] | None = None
    sources: list[SourceOutput] | None = None
    fail_reason: str | None = None
