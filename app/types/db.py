from pydantic import BaseModel


class EmbeddingOutput(BaseModel):
    text: str
    score: float

class QueryDbOutput(BaseModel):
    response: list[EmbeddingOutput]
