from typing import TYPE_CHECKING

import nest_asyncio
from fastapi import Request
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine
from llama_index.core.response_synthesizers import Refine, TreeSummarize
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from app.db.qdrant_repo import nodes_to_embedding_output
from app.services.basedraft import BaseDraftService
from app.types.draft import DraftInput, DraftOutput
from app.types.prompts import (
    REFINE_ANSWER_PROMPT,
    REPLACE_USER_WITH_ID_PROMPT,
    SIMPLE_TEXT_QA_PROMPT_TMPL,
    SUMMARIZE_DRAFT_PROMPT,
)
from app.util.filterworkflow import FilterAndQueryWorkflow, Result
from app.util.query_engine import CustomerInfoQueryEngine

if TYPE_CHECKING:
    from llama_index.core.base.response.schema import Response


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

    def create_draft(self, draft_input: DraftInput) -> DraftOutput:
        """Retrieve the draft output for a given draft input by performing query routing.

        Args:
            draft_input (DraftInput): The input for the draft.

        Returns:
            DraftOutput: The output of the draft.

        """
        product_tool, customer_lookup_tool = self.__get_query_tools()
        query = self.llm.predict(
            PromptTemplate(REPLACE_USER_WITH_ID_PROMPT),
            query_str=draft_input.email_body,
            user_id=draft_input.from_user,
        )
        refine = Refine(text_qa_template=PromptTemplate(SIMPLE_TEXT_QA_PROMPT_TMPL), refine_template=PromptTemplate(REFINE_ANSWER_PROMPT))

        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=[product_tool, customer_lookup_tool],
            use_async=False,
            response_synthesizer=refine,
        )

        response: Response = query_engine.query(query)
        print(response.source_nodes)

        return DraftOutput(
            draft=response.response,
            email_body=draft_input.email_body,
            sources=nodes_to_embedding_output(response.source_nodes),
            questions=None,
        )

    async def create_draft_v2(self, draft_input: DraftInput) -> DraftOutput:
        """Retrieve the draft output for a given draft input by performing query routing.

        Args:
            draft_input (DraftInput): The input for the draft.

        Returns:
            DraftOutput: The output of the draft.

        """
        product_tool, customer_lookup_tool = self.__get_query_tools()
        summarizer = TreeSummarize(summary_template=PromptTemplate(SUMMARIZE_DRAFT_PROMPT))
        query_engine = RouterQueryEngine(
            selector=LLMMultiSelector.from_defaults(),
            query_engine_tools=[product_tool, customer_lookup_tool],
            summarizer=summarizer,
            verbose=True,
        )

        workflow = FilterAndQueryWorkflow(verbose=True, timeout=20)
        filters = [
            "The user seems to be angry",
            "The user is referring to a previous interaction",
        ]

        query = f"from {draft_input.from_user}\n\n{draft_input.email_body}"
        response: Result = await workflow.run(query=query, query_engine=query_engine, refine=False, filters=filters)

        if response.filtered:
            return DraftOutput(draft="", email_body=query, fail_reason=response.stop_reason)

        return DraftOutput(
            draft=response.answer,
            email_body=draft_input.email_body,
            sources=nodes_to_embedding_output(response.source_nodes),
        )

    def __get_query_tools(self) -> tuple[QueryEngineTool, QueryEngineTool]:
        """Define the query tools for the query engine."""
        index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        product_query_engine = index.as_query_engine(
            similarity_top_k=2,
            response_mode="compact",
            text_qa_template=self.text_qa_template,
            verbose=True,
        )

        customer_lookup_query_engine = CustomerInfoQueryEngine(verbose=True)

        product_tool = QueryEngineTool(
            query_engine=product_query_engine,
            metadata=ToolMetadata(name="product", description="Useful for answering questions about Fake Product, including features, pricing etc. Does not answer questions about customers"),
        )
        customer_lookup_tool = QueryEngineTool(
            query_engine=customer_lookup_query_engine,
            metadata=ToolMetadata(name="customer", description="Useful for answering specific questions about the customer, like plan, plan end date, etc. Does not answer questions about products"),
        )

        return product_tool, customer_lookup_tool
