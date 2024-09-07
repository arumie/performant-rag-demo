from fastapi import Request
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.types.db import EmbeddingOutput


class QdrantRepo:
    def __init__(self, request: Request, collection_name: str) -> None:
        """Initialize the QdrantRepo object.

        Args:
            request (Request): The FastAPI request object.
            collection_name (str): The name of the collection

        """
        self.request = request
        self.collection_name = collection_name
        self.__set_openai_embedding()

    def simple_populate_db(self) -> None:
        """Populate the database."""
        storage_context = get_storage_context(self.request)

        # Populate the database
        documents = SimpleDirectoryReader(
            "data/v1",
        ).load_data()
        VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context)

    async def query_db(self, query: str, collection_name: str) -> list[EmbeddingOutput]:
        """Query the database and return a list of EmbeddingOutput objects."""
        vector_store = get_qdrant_vector_store(self.request, collection_name=collection_name)

        # Query the database
        retriever = VectorStoreIndex.from_vector_store(vector_store=vector_store).as_retriever()
        nodes: list[NodeWithScore] = retriever.retrieve(query)
        return nodes_to_embedding_output(nodes)

    def __set_openai_embedding(self) -> None:
        settings = self.request.state.settings
        embed_model = OpenAIEmbedding(embed_batch_size=10, api_key=settings["OPENAI_API_KEY"])
        Settings.embed_model = embed_model


def get_qdrant_vector_store(request: Request, collection_name: str) -> QdrantVectorStore:
    settings = request.state.settings
    qdrant_client = QdrantClient(
        host=settings["QDRANT_HOST"],
        port=settings["QDRANT_PORT"],
    )
    return QdrantVectorStore(client=qdrant_client, collection_name=collection_name)


def get_storage_context(request: Request, collection_name: str) -> StorageContext:
    vector_store = get_qdrant_vector_store(request, collection_name=collection_name)
    return StorageContext.from_defaults(vector_store=vector_store)


def nodes_to_embedding_output (nodes: list[NodeWithScore]) -> list[EmbeddingOutput]:
    return [EmbeddingOutput(text=node.text, score=node.score) for node in nodes]
