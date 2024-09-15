from typing import TYPE_CHECKING

from llama_index.core import PromptTemplate, Settings
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from openai import BaseModel

from app.util.util import pretty_print

if TYPE_CHECKING:
    from llama_index.core.base.response.schema import Response
    from llama_index.core.query_engine import BaseQueryEngine


class FilterEvent(Event):
    filters: list[str]


class QueryEvent(Event):
    query: str


class RefineEvent(Event):
    existing_answer: str
    query: str


class Result(BaseModel):
    answer: str
    filtered: bool = False
    stop_reason: str | None = None
    source_nodes: list[NodeWithScore] | None = None


class FilterAndQueryWorkflow(Workflow):
    @step
    async def setup(
        self,
        ctx: Context,
        ev: StartEvent,
    ) -> FilterEvent:
        """Set up the context of the workflow."""
        if self._verbose:
            print("Setting up the context of the workflow.")
        await ctx.set("query", ev.get("query"))
        await ctx.set("llm", Settings.llm)
        await ctx.set("query_engine", ev.get("query_engine"))
        await ctx.set("refine", ev.get("refine"))
        await ctx.set("refine_prompt", ev.get("refine_prompt"))

        filters = ev.get("filters")

        return FilterEvent(
            filters=filters,
        )

    @step
    async def filter(
        self,
        ctx: Context,
        ev: FilterEvent,
    ) -> QueryEvent | StopEvent:
        """Decide whether to continue or not based on filters."""
        filters: list[str] = ev.filters
        query = await ctx.get("query")
        llm = await ctx.get("llm")

        filter_prompt = """
            You are a helpful assistant that, decide whether to continue or stop based on filters.
            If the query falls under the filters, answer [Stop:<reason>]. Otherwise, answer [Continue].
            Examples:
            --------
            Filter: The user seems to be angry
            Query: Hi, I am very unhappy with the service you provided.
            Answer: [Stop:The user seems to be angry]
            ----
            Filters:
            - The user seems to be angry
            - The user is referring to a previous interaction
            Query: Hi, when we talked last time, you promised to fix the issue.
            Answer: [Stop:The user is referring to a previous interaction]
            --------

            A list of filters is provided below.
            --------
            {filters}
            --------
            Query:
            --------
            {query}
            --------
            Answer:"""

        self.__print(pre_text="Filtering based on the filters:", text=f"{filters}", color="yellow", step="Filter")
        filter_str = "\n- ".join(filters)
        response = llm.predict(PromptTemplate(filter_prompt, filters=filter_str, query=query))
        if "[Stop:" in response:
            reason = response.split("[Stop:")[1].split("]")[0]
            self.__print(text=f"Stopping the workflow: {reason}", color="red", step="Filter")
            return StopEvent(
                result=Result(answer=response, filtered=True, stop_reason=reason),
            )
        if self._verbose:
            self.__print(text="Continuing the workflow.", color="green", step="Filter")
        return QueryEvent(query=query)

    @step
    async def query(
        self,
        ctx: Context,
        ev: QueryEvent,
    ) -> RefineEvent | StopEvent:
        """Query the information based on the query."""
        query = ev.query
        query_engine: BaseQueryEngine = await ctx.get("query_engine")
        self.__print(pre_text="Querying the information based on the query:", text=f"{query}", color="yellow", step="Query")

        response: Response = await query_engine.aquery(query)

        refine = await ctx.get("refine")
        if refine:
            return RefineEvent(existing_answer=response.response, query=query)
        return StopEvent(result=Result(answer=response.response, source_nodes=response.source_nodes))

    @step
    async def refine(
        self,
        ctx: Context,
        ev: RefineEvent,
    ) -> StopEvent:
        """Refine the answer based on the query."""
        existing_answer = ev.get("existing_answer")
        query = ev.get("query")
        refine_prompt = await ctx.get("refine_prompt")
        self.__print(pre_text="Refining the answer based on the query:", text=f"{query}", color="yellow", step="Refine")
        self.__print(pre_text="Existing answer:", text=f"{existing_answer}", color="yellow", step="Refine")

        llm = await ctx.get("llm")

        response: Response = llm.predict(refine_prompt, existing_answer=existing_answer, query_str=query)

        self.__print(f"Refined answer: {response.response}", color="green", step="Refine")

        return StopEvent(result=Result(answer=response, filtered=False, stop_reason=None))

    def __print(self, text: str, step: str, pre_text: str | None = None, color: str | None = None) -> None:
        """Print the text if verbose mode is enabled."""
        pretty_print(verbose=self._verbose, text=text, step=step, pre_text=pre_text, color=color)
