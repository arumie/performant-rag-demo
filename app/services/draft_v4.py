import nest_asyncio
from fastapi import Request
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from app.db import nodes_to_embedding_output
from app.services.base import BaseDraftService
from app.types import REFINE_DRAFT_PROMPT, SIMPLE_TEXT_QA_PROMPT_TMPL, DraftInput, DraftOutput
from app.util import CustomerInfoQueryEngine, FilterAndQueryWorkflow, Result


class DraftV4Service(BaseDraftService):
    def __init__(self, request: Request) -> None:
        """Initialize the DraftService.

        Args:
            request (Request): The FastAPI request object.
            collection_name (str): The name of the collection.

        """
        super().__init__(request, collection_name="V1")
        self.text_qa_template = PromptTemplate(template=SIMPLE_TEXT_QA_PROMPT_TMPL)
        nest_asyncio.apply()

    # ----------------------------------------------------------------------
    # -----------------------FOURTH ITERATION-------------------------------
    # ----------------------------------------------------------------------

    async def create_draft(self, draft_input: DraftInput) -> DraftOutput:
        """Retrieve the draft output for a given draft input by performing query routing.

        Args:
            draft_input (DraftInput): The input for the draft.

        Returns:
            DraftOutput: The output of the draft.

        """
        # Define the query tools
        product_tool, customer_lookup_tool = self.__get_query_tools()

        # Initialize the query engine
        summarizer = TreeSummarize(verbose=True)
        query_engine = RouterQueryEngine(
            selector=LLMMultiSelector.from_defaults(),
            query_engine_tools=[product_tool, customer_lookup_tool],
            summarizer=summarizer,
            verbose=True,
        )

        # Initialize the filter workflow
        workflow = FilterAndQueryWorkflow(verbose=True, timeout=20)
        filters = [
            "The user seems to be angry",
            "The user is referring to a previous interaction",
        ]

        # Run the workflow
        query = f"from {draft_input.from_user}\n\n{draft_input.email_body}"
        response: Result = await workflow.run(query=query, query_engine=query_engine, refine=True, refine_prompt=REFINE_DRAFT_PROMPT, filters=filters)

        if response.filtered:
            return DraftOutput(draft="", email_body=query, fail_reason=response.stop_reason)

        return DraftOutput(
            draft=response.answer,
            email_body=draft_input.email_body,
            sources=nodes_to_embedding_output(response.source_nodes),
        )

    def __get_query_tools(self) -> tuple[QueryEngineTool, QueryEngineTool]:
        """Define the query tools for the query engine."""
        product_query_engine = self.index.as_query_engine(
            similarity_top_k=2,
            response_mode="compact",
            text_qa_template=self.text_qa_template,
            verbose=True,
        )

        customer_lookup_query_engine = CustomerInfoQueryEngine(verbose=True)

        product_tool = QueryEngineTool(
            query_engine=product_query_engine,
            metadata=ToolMetadata(
                name="product",
                description=(
                    "Useful for answering questions about Fake Product, "
                    "including features, pricing etc. "
                    "Does not answer questions about customers"
                ),
            ),
        )
        customer_lookup_tool = QueryEngineTool(
            query_engine=customer_lookup_query_engine,
            metadata=ToolMetadata(
                name="customer",
                description=(
                    "Useful for answering specific questions about the customer, "
                    "like plan, plan end date, etc. "
                    "Does not answer questions about products"
                ),
            ),
        )

        return product_tool, customer_lookup_tool
