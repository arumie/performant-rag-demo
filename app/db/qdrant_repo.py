from uuid import uuid4

from fastapi import Request
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document, NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.types.db import V2_FILES, SourceOutput


class QdrantRepo:
    def __init__(self, request: Request, collection_name: str) -> None:
        """Initialize the QdrantRepo object.

        Args:
            request (Request): The FastAPI request object.
            collection_name (str): The name of the collection

        """
        self.request = request
        self.collection_name = collection_name
        self.embed_model = self.__set_openai_embedding()
        self.llm = self.__set_openai_model()

    def simple_populate_db(self) -> None:
        """Populate the database."""
        storage_context = get_storage_context(self.request, self.collection_name)

        # Populate the database
        documents = SimpleDirectoryReader(
            "data/v1",
        ).load_data()
        for doc in documents:
            doc.metadata["text"] = doc.text

        VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context)

    def metadata_populate_db(self) -> None:
        """Populate the database with metadata."""
        get_and_clear_qdrant_vector_store(self.request, collection_name=self.collection_name)
        storage_context = get_storage_context(self.request, self.collection_name)

        documents = []
        for file in V2_FILES:
            file_documents = SimpleDirectoryReader(input_files=[f"data/v2/{file["file_name"]}"]).load_data()
            for doc in file_documents:
                doc.metadata["product"] = file["product"]
                doc.metadata["file_name"] = file["file_name"]
                doc.metadata["text"] = doc.text
            documents.extend(file_documents)

        VectorStoreIndex.from_documents(storage_context=storage_context, documents=documents)

    def question_populate_db(self) -> None:
        """Populate the database with document questions index."""
        get_and_clear_qdrant_vector_store(self.request, collection_name=self.collection_name)
        storage_context = get_storage_context(self.request, self.collection_name)

        # Populate the database
        documents = SimpleDirectoryReader(
            "data/v3",
        ).load_data()
        question_docs = []
        for doc in documents:
            question_docs.extend(self.__generate_doc_questions(doc))
        VectorStoreIndex.from_documents(documents=question_docs, storage_context=storage_context)

    def query_db(self, query: str) -> list[SourceOutput]:
        """Query the database and return a list of EmbeddingOutput objects."""
        vector_store = get_qdrant_vector_store(self.request, collection_name=self.collection_name)

        # Query the database
        retriever = VectorStoreIndex.from_vector_store(vector_store=vector_store).as_retriever()
        nodes: list[NodeWithScore] = retriever.retrieve(query)
        return nodes_to_embedding_output(nodes)

    def __generate_doc_questions(self, document: Document) -> list[Document]:
        query_gen_str = """\
        You are a helpful assistant that, given a piece of text, generate questions that text answers.
        Questions should be answerable by the text. Questions can have the same answer, but should be different.
        Generate {num_queries} questions, one on each line, related to the following text:
        Text: {text}
        Questions:
        """
        query_gen_prompt = PromptTemplate(query_gen_str)
        response = self.llm.predict(query_gen_prompt, num_queries=10, text=document.text)

        question_docs = [Document(text=question[2:].strip()) for question in response.split("\n")]
        document_id = uuid4()
        for question_doc in question_docs:
            question_doc.metadata["original_text"] = document.text
            question_doc.metadata["question"] = question_doc.text
            question_doc.metadata["id"] = str(document_id)
        return question_docs

    def __set_openai_embedding(self) -> None:
        settings = self.request.state.settings
        embed_model = OpenAIEmbedding(embed_batch_size=10, api_key=settings["OPENAI_API_KEY"])
        Settings.embed_model = embed_model

    def __set_openai_model(self) -> OpenAI:
        settings = self.request.state.settings
        openai = OpenAI(api_key=settings["OPENAI_API_KEY"])
        Settings.llm = openai
        return openai


def get_qdrant_vector_store(request: Request, collection_name: str) -> QdrantVectorStore:
    settings = request.state.settings
    client = QdrantClient(
        host=settings["QDRANT_HOST"],
        port=settings["QDRANT_PORT"],
    )

    aclient = AsyncQdrantClient(
        host=settings["QDRANT_HOST"],
        port=settings["QDRANT_PORT"],
    )
    return QdrantVectorStore(client=client, aclient=aclient, collection_name=collection_name)


def get_and_clear_qdrant_vector_store(request: Request, collection_name: str) -> QdrantVectorStore:
    vector_store = get_qdrant_vector_store(request, collection_name)
    vector_store.clear()
    return vector_store


def get_storage_context(request: Request, collection_name: str) -> StorageContext:
    vector_store = get_and_clear_qdrant_vector_store(request, collection_name=collection_name)
    return StorageContext.from_defaults(vector_store=vector_store)


def nodes_to_embedding_output(nodes: list[NodeWithScore] | None) -> list[SourceOutput]:
    if nodes is None or len(nodes) == 0:
        return []
    question = nodes[0].metadata.get("question", None)
    return [SourceOutput(text=node.text, score=node.score, question=question) for node in nodes]
