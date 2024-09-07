from typing import TYPE_CHECKING

from fastapi import Request
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.llms.openai import OpenAI

from app.services.qdrant_repo import get_qdrant_vector_store, nodes_to_embedding_output
from app.types.draft import DraftInput, DraftOutput

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


class DraftService:
    def __init__(self, request: Request) -> None:
        """Initialize the DraftService.

        Args:
            request (Request): The FastAPI request object.

        """
        self.request = request
        self.__set_openai_model()

    def create_simple_draft(self, draft_input: DraftInput) -> DraftOutput:
        """Create a simple draft from an email body.

        Args:
            draft_input (DraftInput): The input data for the draft.

        Returns:
            DraftOutput: The output data for the draft.

        """
        vector_store = get_qdrant_vector_store(self.request, collection_name="V1")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        prompt_template = PromptTemplate(template=SIMPLE_TEXT_QA_PROMPT_TMPL)
        query_engine = index.as_query_engine(response_mode="compact", text_qa_template=prompt_template)
        response: Response = query_engine.query(draft_input.email_body)

        return DraftOutput(draft=response.response, email_body=draft_input.email_body, embeddings=nodes_to_embedding_output(response.source_nodes))

    def __set_openai_model(self) -> None:
        settings = self.request.state.settings
        openai = OpenAI(api_key=settings["OPENAI_API_KEY"])
        Settings.llm = openai
