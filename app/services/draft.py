from typing import TYPE_CHECKING

from fastapi import Request
from llama_index.core import PromptTemplate, QueryBundle, Settings, VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.core.response_synthesizers import Refine
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.llms.openai import OpenAI
from llama_index.question_gen.openai import OpenAIQuestionGenerator
from pydantic import Field

from app.services.customer_lookup import CustomerInfoQueryEngine
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

REFINE_ANSWER_PROMPT = """
    You work as a support center agent and you answer questions from emails from customers.
    The original query is as follows: {query_str}
    We have provided an existing answer: {existing_answer}
    We have the opportunity to refine the existing answer into a helpful response and ensure that the response has the following characteristics:
    - Start with 'Hello, thank you for reaching out to us. I am happy to help you with your query.' and end with 'Please let me know if you have any further questions.' separated by new lines.
    - Ensure that the response answers the query in a helpful and informative way.
    - References to the user ID should be replaced "you", "your", etc. where appropriate.
    If the existing answer follows these guidelines already, return the existing answer.
    Refined Answer:
"""


REPLACE_USER_WITH_ID_PROMPT = """
    Replace references to 'me', 'my', etc. with the user ID.
    example:
    User ID: user123
    Query: "What is the status of my order?"
    Answer: "What is the status of user123's order?"
    Query: {query_str}
    User ID: {user_id}
    Answer:
"""


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
            sources=nodes_to_embedding_output(response.source_nodes),
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

        return DraftOutput(draft=draft, email_body=draft_input.email_body, questions=question_answers, sources=None)

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

    def query_router_draft(self, draft_input: DraftInput) -> DraftOutput:
        """Retrieve the draft output for a given draft input by performing query routing.

        Args:
            draft_input (DraftInput): The input for the draft.

        Returns:
            DraftOutput: The output of the draft.

        """
        vector_store = get_qdrant_vector_store(self.request, collection_name=self.collection_name)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        prompt_template = PromptTemplate(template=SIMPLE_TEXT_QA_PROMPT_TMPL)
        product_query_engine = index.as_query_engine(
            similarity_top_k=2,
            response_mode="compact",
            text_qa_template=prompt_template,
        )

        customer_lookup_query_engine = CustomerInfoQueryEngine(verbose=True)

        product_tool = QueryEngineTool(
            query_engine=product_query_engine,
            metadata=ToolMetadata(name="product", description="Useful for answering questions about Fake Product"),
        )
        customer_lookup_tool = QueryEngineTool(
            query_engine=customer_lookup_query_engine,
            metadata=ToolMetadata(name="customer", description=customer_lookup_query_engine.get_engine_description()),
        )
        query = self.llm.predict(
            PromptTemplate(REPLACE_USER_WITH_ID_PROMPT),
            query_str=draft_input.email_body,
            user_id=draft_input.from_user,
        )
        refine = Refine(text_qa_template=prompt_template, refine_template=PromptTemplate(REFINE_ANSWER_PROMPT))

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

    def __set_openai_model(self) -> OpenAI:
        settings = self.request.state.settings
        openai = OpenAI(api_key=settings["OPENAI_API_KEY"])
        Settings.llm = openai
        return openai
