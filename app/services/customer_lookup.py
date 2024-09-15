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


class CustomerLookupService:
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
