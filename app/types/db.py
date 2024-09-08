from pydantic import BaseModel


class EmbeddingOutput(BaseModel):
    text: str
    question: str | None
    score: float


class QueryDbOutput(BaseModel):
    response: list[EmbeddingOutput]


V2_FILES = [
    {"file_name": "fake_app.md", "product": "Fake App"},
    {"file_name": "fake_product.md", "product": "Fake Product 1.0"},
    {"file_name": "fake_product_v2.md", "product": "Fake Product 2.0"},
    {"file_name": "fake_widget.md", "product": "Fake Widget"},
]
