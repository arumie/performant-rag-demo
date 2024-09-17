from llama_index.core import PromptTemplate, Settings
from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.query_engine import BaseQueryEngine, CustomQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.openai import OpenAI

from app.services.customer_lookup import CustomerLookupService
from app.types.prompts import EXTRACT_USER_IDS_PROMPT
from app.util.util import pretty_print


class CustomerInfoQueryEngine(CustomQueryEngine):
    """A query engine for customer information queries."""

    verbose: bool | None = False
    llm: OpenAI | None = None
    service: CustomerLookupService | None = None

    def __init__(self, verbose: bool | None = None, llm: OpenAI | None = None) -> None:
        """Initialize the CustomerInfoQueryEngine object."""
        super().__init__(llm=llm, verbose=verbose)
        self.verbose = verbose or False
        self.llm = llm or Settings.llm
        self.service = CustomerLookupService()

    def get_engine_description(self) -> str:
        """Get the description of the query engine."""
        return "Useful for answering questions about customers, given their user IDs."

    def custom_query(self, query_str: str) -> RESPONSE_TYPE:
        """Query the customer information based on the user ID."""
        user_ids_response = self.llm.predict(PromptTemplate(EXTRACT_USER_IDS_PROMPT, query_str=query_str))
        user_ids = user_ids_response.split("\n")
        if user_ids_response == "[None]":
            return "No user IDs found in the query."
        self.__print(pre_text="Extracted user IDs:", text=f"{user_ids}", color="yellow")

        user_info_str = ""
        for user_id in user_ids:
            user_info_dict = self.service.lookup(user_id.strip())
            self.__print(pre_text="User Info: ", text=f"{user_info_dict}")
            user_info_str += f"User ID: {user_id}\n"
            if user_info_dict:
                user_info_str += "\n".join(f"{key}: {value}" for key, value in user_info_dict.items())
            else:
                user_info_str += "No information found for this user."
        self.__print(pre_text="User Info String: ", text=f"{user_info_str}")
        source_nodes: list[NodeWithScore] = [NodeWithScore(node=TextNode(text=user_info_str))]

        question_answer_query = """
            You are a helpful assistant that, answers questions about customers.
            ----
            {user_info_str}
            ----
            Query: {query_str}
            Answer:"""
        result = self.llm.predict(PromptTemplate(question_answer_query, user_info_str=user_info_str, query_str=query_str))
        return Response(response=result, source_nodes=source_nodes)


    def __print(self, text: str, pre_text: str | None = None, color: str | None = None) -> None:
        """Print the text if verbose mode is enabled."""
        pretty_print(verbose=self.verbose, text=text, step="CustomerInfoQueryEngine", pre_text=pre_text, color=color)


class FilterQueryEngine(CustomQueryEngine):
    """A query engine for decision-making based on filters.

    Args:
    filters (list[str]): The list of filters to apply.
    verbose (bool | None): Whether to print verbose output.
    llm (OpenAI | None): The OpenAI language model to use.

    """

    verbose: bool | None = False
    llm: OpenAI | None = None
    filters: list[str]
    success_query_engine: BaseQueryEngine

    def __init__(self, filters: list[str], verbose: bool | None = None, llm: OpenAI | None = None) -> None:
        """Initialize the FilterQueryEngine object."""
        super().__init__(llm=llm, verbose=verbose)
        self.verbose = verbose or False
        self.llm = llm or Settings.llm
        self.filters = filters
