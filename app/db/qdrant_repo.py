from uuid import uuid4

from fastapi import Request
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import Document, NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from app.db.util import (
    get_qdrant_vector_store,
    get_storage_context,
    nodes_to_embedding_output,
)
from app.types import V2_FILES, SourceOutput
from app.types.prompts import QUESTION_GENERATION_PROMPT


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

    # ----------------------------------------------------------------------
    # -----------------------FIRST ITERATION--------------------------------
    # ----------------------------------------------------------------------

    def simple_populate_db(self) -> None:
        """Populate the database."""
        storage_context = get_storage_context(self.request, self.collection_name, enable_hybrid=True)
        storage_context.vector_store.clear()

        # Read the document from the directory
        documents = SimpleDirectoryReader(
            "data/v1",
        ).load_data()
        for doc in documents:
            doc.metadata["text"] = doc.text

        # Add the documents to the database
        VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, show_progress=True)

    # ----------------------------------------------------------------------
    # -----------------------SECOND ITERATION--------------------------------
    # ----------------------------------------------------------------------

    def metadata_populate_db(self) -> None:
        """Populate the database with metadata."""
        storage_context = get_storage_context(self.request, self.collection_name)
        storage_context.vector_store.clear()

        documents = []
        for file in V2_FILES:
            # Read document from directory
            file_documents = SimpleDirectoryReader(input_files=[f"data/v2/{file["file_name"]}"]).load_data()

            # Add metadata to the documents for that file
            for doc in file_documents:
                doc.metadata["product"] = file["product"]
                doc.metadata["file_name"] = file["file_name"]
                doc.metadata["text"] = doc.text
            documents.extend(file_documents)

        # Add the documents to the database
        VectorStoreIndex.from_documents(storage_context=storage_context, documents=documents, show_progress=True)

    # ----------------------------------------------------------------------
    # -----------------------THIRD ITERATION--------------------------------
    # ----------------------------------------------------------------------

    def question_populate_db(self) -> None:
        """Populate the database with document questions index."""
        storage_context = get_storage_context(self.request, self.collection_name)
        storage_context.vector_store.clear()

        # Load the documents from the directory
        documents = SimpleDirectoryReader(
            "data/v3",
        ).load_data()

        # Generate questions for each document
        question_docs = []
        for doc in documents:
            question_docs.extend(self.__generate_doc_questions(doc))

        # Add the question to the database
        VectorStoreIndex.from_documents(documents=question_docs, storage_context=storage_context)

    def __generate_doc_questions(self, document: Document) -> list[Document]:
        # Generate questions in separate documents/nodes
        response = self.llm.predict(PromptTemplate(QUESTION_GENERATION_PROMPT), num_queries=10, text=document.text)
        question_docs = [Document(text=question[2:].strip()) for question in response.split("\n")]

        # Add metadata to the questions
        document_id = uuid4()
        for question_doc in question_docs:
            question_doc.metadata["original_text"] = document.text
            question_doc.metadata["question"] = question_doc.text
            question_doc.metadata["id"] = str(document_id)
        return question_docs

    # ----------------------------------------------------------------------
    # -------------------------------- UTIL --------------------------------
    # ----------------------------------------------------------------------

    def query_db(self, query: str) -> list[SourceOutput]:
        """Query the database and return a list of EmbeddingOutput objects."""
        vector_store = get_qdrant_vector_store(self.request, collection_name=self.collection_name)

        # Query the database
        retriever = VectorStoreIndex.from_vector_store(vector_store=vector_store).as_retriever()
        nodes: list[NodeWithScore] = retriever.retrieve(query)
        return nodes_to_embedding_output(nodes)

    def __set_openai_embedding(self) -> None:
        settings = self.request.state.settings
        embed_model = OpenAIEmbedding(embed_batch_size=10, api_key=settings["OPENAI_API_KEY"])
        Settings.embed_model = embed_model

    def __set_openai_model(self) -> OpenAI:
        settings = self.request.state.settings
        openai = OpenAI(api_key=settings["OPENAI_API_KEY"])
        Settings.llm = openai
        return openai
