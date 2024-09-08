from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypedDict

import fastenv
from fastapi import FastAPI, Request

from app.services.draft import DraftService
from app.services.qdrant_repo import QdrantRepo
from app.types.db import QueryDbOutput
from app.types.draft import DraftInput, DraftOutput


class LifespanState(TypedDict):
    settings: fastenv.DotEnv


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[LifespanState]:
    settings = await fastenv.load_dotenv(".env")
    lifespan_state: LifespanState = {"settings": settings}
    yield lifespan_state


app = FastAPI(lifespan=lifespan)


@app.get("/settings")
async def get_settings(request: Request) -> dict[str, str]:
    settings = request.state.settings
    return dict(settings)


@app.post("/v1/draft", description="Create a draft from an email body using simple RAG", summary="Create a draft")
async def create_draft_v1(request: Request, draft_input: DraftInput) -> DraftOutput:
    draft_service = DraftService(request, "V1")
    return draft_service.create_simple_draft(draft_input)


@app.post("/v2/draft", description="Create a draft from an email body using sub questions and auto retrieval", summary="Create a draft")
async def create_draft_v2(request: Request, draft_input: DraftInput) -> DraftOutput:
    draft_service = DraftService(request, "V2")
    return draft_service.sub_question_auto_retrieval_draft(draft_input)

@app.post("/v3/draft", description="Create a draft from an email body using document question index", summary="Create a draft")
async def create_draft_v3(request: Request, draft_input: DraftInput) -> DraftOutput:
    draft_service = DraftService(request, "V3")
    return draft_service.doc_question_index_retrieval_draft(draft_input)


@app.post("/v1/db/populate", description="Populate the Qdrant Vector Database", summary="Populate the database")
async def db_populate_v1(request: Request) -> None:
    repo = QdrantRepo(request, "V1")
    return repo.simple_populate_db()

@app.post("/v2/db/populate", description="Populate the Qdrant Vector Database with metadata", summary="Populate the database")
async def db_populate_v2(request: Request) -> None:
    repo = QdrantRepo(request, "V2")
    return repo.metadata_populate_db()

@app.post("/v3/db/populate", description="Populate the Qdrant Vector Database with Document Question Index", summary="Populate the database")
async def db_populate_v3(request: Request) -> None:
    repo = QdrantRepo(request, "V3")
    return repo.question_populate_db()

@app.get("/db/query", description="Query the Qdrant Vector Database", summary="Query the database")
async def get_db_query(request: Request, query: str, version: int) -> QueryDbOutput:
    collection_name = f"V{version}"
    repo = QdrantRepo(request, collection_name=collection_name)
    response = await repo.query_db(query)
    return {"response": response}
