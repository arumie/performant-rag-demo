
from pydantic import BaseModel

from app.types.db import EmbeddingOutput


class DraftInput(BaseModel):
    email_body: str


class DraftOutput(BaseModel):
    draft: str
    email_body: str
    embeddings: list[EmbeddingOutput]
