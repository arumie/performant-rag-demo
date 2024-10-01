from fastapi import Request
from llama_index.core import StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.types.db import SourceOutput


def get_qdrant_vector_store(request: Request, collection_name: str, enable_hybrid: bool = False) -> QdrantVectorStore:
    settings = request.state.settings
    client = QdrantClient(
        host=settings["QDRANT_HOST"],
        port=settings["QDRANT_PORT"],
    )

    aclient = AsyncQdrantClient(
        host=settings["QDRANT_HOST"],
        port=settings["QDRANT_PORT"],
    )
    return QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=collection_name,
        enable_hybrid=enable_hybrid,
    )


def get_storage_context(request: Request, collection_name: str, enable_hybrid: bool = False) -> StorageContext:
    vector_store = get_qdrant_vector_store(request, collection_name=collection_name, enable_hybrid=enable_hybrid)
    return StorageContext.from_defaults(vector_store=vector_store)


def nodes_to_embedding_output(nodes: list[NodeWithScore] | None) -> list[SourceOutput]:
    if nodes is None or len(nodes) == 0:
        return []
    question = nodes[0].metadata.get("question", None)
    return [SourceOutput(text=node.text, score=node.score, question=question) for node in nodes]
