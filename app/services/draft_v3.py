
from typing import TYPE_CHECKING

from fastapi import Request
from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

from app.db import nodes_to_embedding_output
from app.services.base import BaseDraftService
from app.types import SIMPLE_TEXT_QA_PROMPT_TMPL, DraftInput, DraftOutput
from app.util import DistinctPostProcessor

if TYPE_CHECKING:
    from llama_index.core.base.response.schema import Response


class DraftV3Service(BaseDraftService):
    def __init__(self, request: Request) -> None:
        """Initialize the DraftService.

        Args:
            request (Request): The FastAPI request object.
            collection_name (str): The name of the collection.

        """
        super().__init__(request, collection_name="V3")


    def create_draft(self, draft_input: DraftInput) -> DraftOutput:
        """Retrieve the draft output for a given draft input by performing question indexing.

        Args:
            draft_input (DraftInput): The input for the draft.

        Returns:
            DraftOutput: The output of the draft.

        """
        prompt_template = PromptTemplate(template=SIMPLE_TEXT_QA_PROMPT_TMPL)
        query_engine = self.index.as_query_engine(
            similarity_top_k=5,  # DistinctPostProcessor will reduce the number of nodes
            response_mode="compact",
            text_qa_template=prompt_template,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="original_text"),
                DistinctPostProcessor(target_metadata_key="id"),
            ],
        )
        response: Response = query_engine.query(draft_input.email_body)

        return DraftOutput(
            draft=response.response,
            email_body=draft_input.email_body,
            sources=nodes_to_embedding_output(response.source_nodes),
            questions=None,
        )
