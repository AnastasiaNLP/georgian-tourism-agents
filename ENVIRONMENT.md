# Environment Contract

The application does not read `.env` by itself. Variables must be present in the
process environment when the API or scripts are started.

## Required For Core Planning

- `OPENAI_API_KEY` - response/planning/validation LLM calls.
- `ANTHROPIC_API_KEY` - orchestrator request classification.
- `QDRANT_URL` - tourism vector database.
- `QDRANT_API_KEY` - Qdrant authentication.
- `COLLECTION_NAME` - attraction collection, defaults to `georgian_attractions`.
- `ORS_API_KEY` - geocoding and driving routes.

## Optional Runtime Features

- `UPSTASH_REDIS_URL` - cache/profile memory backend.
- `UPSTASH_REDIS_TOKEN` - cache/profile memory token.
- `EMBEDDING_MODEL` - sentence-transformers model, defaults to multilingual MiniLM.
- `VECTOR_SIZE` - embedding vector size for memory collections, defaults to `384`.
- `LANGSMITH_API_KEY` - LangSmith tracing key.
- `LANGSMITH_PROJECT` - LangSmith project name.
- `LANGCHAIN_TRACING_V2` - LangChain tracing toggle.

## Present But Not Wired Into Current Code

These variables exist in the environment contract but are not used by the current
Python code path:

- `UNSPLASH_URL`
- `UNSPLASH_ACCESS_KEY`
- `UNSPLASH_SECRET_KEY`
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`
- `MAPBOX_TOKEN`

## Test Toggles

- `RUN_ONLINE_TESTS=true` - enables tests marked `online`.

Online tests call external APIs and may spend LLM/provider credits. They are
skipped by default.
