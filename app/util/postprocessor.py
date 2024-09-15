from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field


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
