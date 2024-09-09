from llama_index.core import PromptTemplate, Settings
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.llms.openai import OpenAI

CUSTOMER_DB = {
    "user123": {"name": "John Smith", "current_plan": "Free Trial", "subscription_end_date": None},
    "user456": {"name": "Jane Doe", "current_plan": "Basic Plan", "subscription_end_date": "2025-5-1"},
    "user789": {"name": "Jim Dickens", "current_plan": "Pro Plan", "subscription_end_date": "2024-12-31"},
    "user999": {
        "name": "Jill Skelter",
        "current_plan": "Enterprise Plan",
        "subscription_end_date": None,
        "company_name": "ACME Inc.",
    },
}


class CustomerLookup:
    def __init__(self) -> None:
        """Initialize the CustomerLookup object."""

    def lookup(self, user_id: str) -> dict:
        """Look up a customer in the database based on the user ID.

        Args:
            user_id (str): The ID of the customer.

        Returns:
            dict: The customer information.

        """
        return CUSTOMER_DB.get(user_id, {})


class CustomerInfoQueryEngine(CustomQueryEngine):
    verbose: bool | None = False
    llm: OpenAI | None = None
    service: CustomerLookup | None = None

    def __init__(self, verbose: bool | None = None, llm: OpenAI | None = None) -> None:
        """Initialize the CustomerInfoQueryEngine object."""
        super().__init__(llm=llm, verbose=verbose)
        self.verbose = verbose or False
        self.llm = llm or Settings.llm
        self.service = CustomerLookup()

    def get_engine_description(self) -> str:
        """Get the description of the query engine."""
        return "Useful for answering questions about customers, given their user IDs."

    def custom_query(self, query_str: str) -> str:
        """Query the customer information based on the user ID."""
        extract_user_ids_query = """
            List the user IDs in the following text, each on a new line.
            User IDs are alphanumeric strings that start with 'user' followed by a number.
            Only include the user IDs on each line. Return [None] if no user IDs are found.
            Example:
            Query: "from user123"
            Answer:
            user123

            Example:
            Query: "What is the status of user123 and user456?"
            Answer:
            user123
            user456



            Query: {query_str}
            Answer:"""

        if self.verbose:
            print(f"Extracting user IDs from the query: {query_str}")
        user_ids_response = self.llm.predict(PromptTemplate(extract_user_ids_query, query_str=query_str))
        user_ids = user_ids_response.split("\n")
        if user_ids_response == "[None]":
            return "No user IDs found in the query."
        if self.verbose:
            print(f"Extracted user IDs: {user_ids}")
        user_info_str = ""
        for user_id in user_ids:
            user_info_dict = self.service.lookup(user_id.strip())
            if self.verbose:
                print(f"User ID: {user_id}")
                print(f"User Info: {user_info_dict}")
            user_info_str += f"User ID: {user_id}\n"
            if user_info_dict:
                user_info_str += "\n".join(f"{key}: {value}" for key, value in user_info_dict.items())
            else:
                user_info_str += "No information found for this user."

        question_answer_query = """
            You are a helpful assistant that, answers questions about customers.
            ----
            {user_info_str}
            ----
            Query: {query_str}
            Answer:"""
        return self.llm.predict(PromptTemplate(question_answer_query, user_info_str=user_info_str, query_str=query_str))
