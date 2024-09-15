
from fastapi import Request
from llama_index.core import PromptTemplate, QueryBundle
from llama_index.core.tools import ToolMetadata
from llama_index.question_gen.openai import OpenAIQuestionGenerator

from app.services.basedraft import BaseDraftService
from app.types.db import V2_FILES
from app.types.draft import DraftInput, DraftOutput, QuestionOutput
from app.types.prompts import SIMPLE_TEXT_QA_PROMPT_TMPL


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
