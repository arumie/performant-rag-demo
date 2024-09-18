# Demo for Performant RAG presentation

This repository contains the code for the demo presented in the Performant RAG presentation. 
The demo is a simple question answering system that uses the RAG model to answer questions based on a given context.

## Installation

This project uses [astral/uv](https://docs.astral.sh/uv/) for managing the project
Run the following command to install the required packages:

```bash
uv sync
```

## Usage

### Starting Qdrant with Docker

To start Qdrant with Docker, use the following command:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

This will start Qdrant on localhost:6333. You can access the Qdrant dashboard by visiting [localhost:6333/dashboard](http://localhost:6333/dashboard) in your browser.

### Running the RAG application

To run the FastAPI app, use the following command:

```bash
uv run fastapi dev
```

This will run the FastAPI app on localhost:8000. You can access the Swagger UI by visiting [localhost:8000/docs](http://localhost:8000/docs) in your browser.

This gives access to the following endpoints:

#### Creating drafts
- POST localhost:8000/v1/draft
- POST localhost:8000/v2/draft
- POST localhost:8000/v3/draft
- POST localhost:8000/v4/draft

#### Populate the vector database
- POST localhost:8000/v1/db/populate
- POST localhost:8000/v2/db/populate
- POST localhost:8000/v3/db/populate
- POST localhost:8000/v4/db/populate
- GET localhost:8000/db/query (*to query the database*)

### Setting up the environment variables

To the root of the project, create a `.env` file and add the following environment variables:

```
QDRANT_HOST=localhost
QDRANT_PORT=6333

OPENAI_API_KEY=<your-openai-key>
```

### Running the frontend

To run the demo frontend, use the following command:

```bash
uv run frontend.py
```

This will run the frontend on localhost:8080. You can access the frontend by visiting [localhost:8080](http://localhost:8080) in your browser.