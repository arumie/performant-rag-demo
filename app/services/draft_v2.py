from typing import TYPE_CHECKING

from fastapi import Request
from llama_index.core import PromptTemplate, QueryBundle, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.tools import ToolMetadata
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.question_gen.openai import OpenAIQuestionGenerator

from app.db.qdrant_repo import get_qdrant_vector_store, nodes_to_embedding_output
from app.services.basedraft import BaseDraftService
from app.types.db import V2_FILES
from app.types.draft import DraftInput, DraftOutput, QuestionOutput
from app.types.prompts import SIMPLE_TEXT_QA_PROMPT_TMPL

if TYPE_CHECKING:
    from llama_index.core.base.response.schema import Response


class DraftV2Service(BaseDraftService):
    def __init__(self, request: Request) -> None:
        """Initialize the DraftService.

        Args:
            request (Request): The FastAPI request object.
            collection_name (str): The name of the collection.

        """
        super().__init__(request, collection_name="V2")

    def create_draft(self, draft_input: DraftInput) -> DraftOutput:
        """Retrieve sub-questions and generates a draft based on the given draft input.

        Args:
            draft_input (DraftInput): The input for generating the draft.

        Returns:
            DraftOutput: The generated draft, email body, list of questions, and embeddings.

        """
        product_categories = [file["product"] for file in V2_FILES]
        question_gen = OpenAIQuestionGenerator.from_defaults(verbose=True)
        tool_choices = [
            ToolMetadata(name=product, description=f"Questions and answers about the product {product}")
            for product in product_categories
        ]
        questions = question_gen.generate(tools=tool_choices, query=QueryBundle(query_str=draft_input.email_body))

        question_answers: list[QuestionOutput] = [
            self.__auto_retrieval_draft(query=question.sub_question, product_categories=product_categories)
            for question in questions
        ]
        answer_str = "\n".join([f"{question.answer}" for question in question_answers])

        draft = self.llm.predict(
            PromptTemplate(template=SIMPLE_TEXT_QA_PROMPT_TMPL),
            context_str=answer_str,
            query_str=draft_input.email_body,
        )

        return DraftOutput(draft=draft, email_body=draft_input.email_body, questions=question_answers, sources=None)

    def __auto_retrieval_draft(self, query: str, product_categories: list[str]) -> QuestionOutput:
        retriever = self.__get_auto_retriever(categories=product_categories)
        query_engine = RetrieverQueryEngine(retriever=retriever)
        response: Response = query_engine.query(query)

        return QuestionOutput(
            question=query,
            answer=response.response,
            sources=nodes_to_embedding_output(response.source_nodes),
        )

    def __get_auto_retriever(self, categories: list[str]) -> VectorIndexAutoRetriever:
        vector_store = get_qdrant_vector_store(self.request, collection_name=self.collection_name)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        metadata_description = f"Name of the product, one of {categories}"
        vector_store_info = VectorStoreInfo(
            content_info="Guides on how to answer customer queries about products",
            metadata_info=[
                MetadataInfo(
                    name="product",
                    type="str",
                    description=metadata_description,
                ),
            ],
        )
        return VectorIndexAutoRetriever(
            index=index,
            vector_store_info=vector_store_info,
            verbose=True,
        )
