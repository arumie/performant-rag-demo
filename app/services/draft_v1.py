from typing import TYPE_CHECKING

from fastapi import Request
from llama_index.core import PromptTemplate, VectorStoreIndex

from app.db.qdrant_repo import get_qdrant_vector_store, nodes_to_embedding_output
from app.services.basedraft import BaseDraftService
from app.types.draft import DraftInput, DraftOutput
from app.types.prompts import SIMPLE_TEXT_QA_PROMPT_TMPL

if TYPE_CHECKING:
    from llama_index.core.base.response.schema import Response


class DraftV1Service(BaseDraftService):
    def __init__(self, request: Request) -> None:
        """Initialize the DraftService.

        Args:
            request (Request): The FastAPI request object.
            collection_name (str): The name of the collection.

        """
        super().__init__(request, collection_name="V1")

    def create_draft(self, draft_input: DraftInput) -> DraftOutput:
        """Create a simple draft from an email body. Uses a baseline RAG pipeline to generate a response.

        Args:
            draft_input (DraftInput): The input data for the draft.

        Returns:
            DraftOutput: The output data for the draft.

        """
        vector_store = get_qdrant_vector_store(self.request, collection_name=self.collection_name)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        prompt_template = PromptTemplate(template=SIMPLE_TEXT_QA_PROMPT_TMPL)
        query_engine = index.as_query_engine(response_mode="compact", text_qa_template=prompt_template)
        response: Response = query_engine.query(draft_input.email_body)

        return DraftOutput(
            draft=response.response,
            email_body=draft_input.email_body,
            sources=nodes_to_embedding_output(response.source_nodes),
        )
