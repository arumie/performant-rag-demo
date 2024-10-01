from typing import TYPE_CHECKING

from fastapi import Request
from llama_index.core import PromptTemplate

from app.db import nodes_to_embedding_output
from app.services.base import BaseDraftService
from app.types import SIMPLE_TEXT_QA_PROMPT_TMPL, DraftInput, DraftOutput

if TYPE_CHECKING:
    from llama_index.core.base.response.schema import Response


class DraftV1Service(BaseDraftService):
    def __init__(self, request: Request, collection_name: str = "V1") -> None:
        """Initialize the DraftService.

        Args:
            request (Request): The FastAPI request object.
            collection_name (str): The name of the collection.

        """
        super().__init__(request=request, collection_name=collection_name, enable_hybrid=True)

    # ----------------------------------------------------------------------
    # -----------------------FIRST ITERATION--------------------------------
    # ----------------------------------------------------------------------

    def create_draft(self, draft_input: DraftInput) -> DraftOutput:
        """Create a simple draft from an email body. Uses a baseline RAG pipeline to generate a response.

        Args:
            draft_input (DraftInput): The input data for the draft.

        Returns:
            DraftOutput: The output data for the draft.

        """
        # Initialize the query engine
        query_engine = self.index.as_query_engine(
            response_mode="compact",
            text_qa_template=PromptTemplate(template=SIMPLE_TEXT_QA_PROMPT_TMPL),
            vector_store_query_mode="hybrid",
        )

        # Generate the draft
        response: Response = query_engine.query(draft_input.email_body)

        return DraftOutput(
            draft=response.response,
            email_body=draft_input.email_body,
            sources=nodes_to_embedding_output(response.source_nodes),
        )
