from typing import TYPE_CHECKING

from fastapi import Request
from llama_index.core import PromptTemplate, QueryBundle, Settings, VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import ToolMetadata
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.llms.openai import OpenAI
from llama_index.question_gen.openai import OpenAIQuestionGenerator
from pydantic import Field

from app.services.qdrant_repo import get_qdrant_vector_store, nodes_to_embedding_output
from app.types.db import V2_FILES
from app.types.draft import DraftInput, DraftOutput, QuestionOutput

if TYPE_CHECKING:
    from llama_index.core.base.response.schema import Response

SIMPLE_TEXT_QA_PROMPT_TMPL = (
    "You work as a support center agent and you answer questions from emails from customers."
    "You always start with 'Hello, thank you for reaching out to us. I am happy to help you with your query.' and end with 'Please let me know if you have any further questions.' separated by new lines\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context and no prior knowledge you generate an answer to the following email.\n"
    "Query: {query_str}\n"
    "Answer: "
)

class DistinctPostProcessor(BaseNodePostprocessor):
    target_metadata_key: str = Field(
        description="Target metadata key to distinct node by.",
    )

    def __init__(self, target_metadata_key: str) -> None:
        """Initialize the DistinctPostProcessor."""
        super().__init__(target_metadata_key=target_metadata_key)

    @classmethod
    def class_name(cls) -> str:
        """Return the name of the class."""
        return "DistinctPostProcessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,  # noqa: ARG002
    ) -> list[NodeWithScore]:
        distinct_nodes = []
        seen = set()
        for node in nodes:
            if node.node.metadata[self.target_metadata_key] not in seen:
                seen.add(node.node.metadata[self.target_metadata_key])
                distinct_nodes.append(node)

        return distinct_nodes


class DraftService:
    def __init__(self, request: Request, collection_name: str) -> None:
        """Initialize the DraftService.

        Args:
            request (Request): The FastAPI request object.
            collection_name (str): The name of the collection.

        """
        self.request = request
        self.collection_name = collection_name
        self.llm = self.__set_openai_model()

    def create_simple_draft(self, draft_input: DraftInput) -> DraftOutput:
        """Create a simple draft from an email body.

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
            embeddings=nodes_to_embedding_output(response.source_nodes),
        )

    def sub_question_auto_retrieval_draft(self, draft_input: DraftInput) -> DraftOutput:
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

        return DraftOutput(draft=draft, email_body=draft_input.email_body, questions=question_answers, embeddings=None)

    def doc_question_index_retrieval_draft(self, draft_input: DraftInput) -> DraftOutput:
        """Retrieve the draft output for a given draft input by performing question indexing.

        Args:
            draft_input (DraftInput): The input for the draft.

        Returns:
            DraftOutput: The output of the draft.

        """
        vector_store = get_qdrant_vector_store(self.request, collection_name=self.collection_name)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        prompt_template = PromptTemplate(template=SIMPLE_TEXT_QA_PROMPT_TMPL)
        query_engine = index.as_query_engine(
            similarity_top_k=5, # DistinctPostProcessor will reduce the number of nodes
            response_mode="compact",
            text_qa_template=prompt_template,
            node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="original_text"), DistinctPostProcessor(target_metadata_key="id")],
        )
        response: Response = query_engine.query(draft_input.email_body)

        return DraftOutput(
            draft=response.response,
            email_body=draft_input.email_body,
            embeddings=nodes_to_embedding_output(response.source_nodes),
            questions=None,
        )

    def __auto_retrieval_draft(self, query: str, product_categories: list[str]) -> QuestionOutput:
        retriever = self.__get_auto_retriever(categories=product_categories)
        query_engine = RetrieverQueryEngine(retriever=retriever)
        response: Response = query_engine.query(query)

        return QuestionOutput(
            question=query,
            answer=response.response,
            embeddings=nodes_to_embedding_output(response.source_nodes),
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

    def __set_openai_model(self) -> OpenAI:
        settings = self.request.state.settings
        openai = OpenAI(api_key=settings["OPENAI_API_KEY"])
        Settings.llm = openai
        return openai
