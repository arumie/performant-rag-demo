
from fastapi import Request
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.openai import OpenAI

from app.db import get_qdrant_vector_store


class BaseDraftService:
    def __init__(self, request: Request, collection_name: str, enable_hybrid: bool = False) -> None:
        """Initialize the DraftService.

        Args:
            request (Request): The FastAPI request object.
            collection_name (str): The name of the collection.
            enable_hybrid (bool): Flag to enable hybrid search in the vector store.

        """
        self.request = request

        # Initialize the LLM to be used in the pipeline (OpenAI gpt-4o-mini)
        self.llm = self.__set_openai_model()

        # Initialize the vector store and index for retrieval
        self.collection_name = collection_name
        self.vector_store = get_qdrant_vector_store(self.request, collection_name=self.collection_name, enable_hybrid=enable_hybrid)
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)


    def __set_openai_model(self) -> OpenAI:
        settings = self.request.state.settings
        openai = OpenAI(api_key=settings["OPENAI_API_KEY"], model="gpt-4o-mini")
        Settings.llm = openai
        return openai
