
from fastapi import Request
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from app.db.qdrant_repo import get_qdrant_vector_store


class BaseDraftService:
    def __init__(self, request: Request, collection_name: str) -> None:
        """Initialize the DraftService.

        Args:
            request (Request): The FastAPI request object.
            collection_name (str): The name of the collection.

        """
        self.request = request
        self.collection_name = collection_name
        self.llm = self.__set_openai_model()
        self.vector_store = get_qdrant_vector_store(self.request, collection_name=self.collection_name)


    def __set_openai_model(self) -> OpenAI:
        settings = self.request.state.settings
        openai = OpenAI(api_key=settings["OPENAI_API_KEY"])
        Settings.llm = openai
        return openai
