#!/usr/bin/env python3
"""
Georgian Tourism Agent API entry point.

Initialization order matters:
1. setup_langsmith_tracing() must run before importing LangGraph
2. FastAPI app + lifespan
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(override=True)

# Configure LangSmith tracing before importing LangGraph.
from monitoring.tracing import setup_langsmith_tracing
setup_langsmith_tracing()

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager.
    Build the graph and checkpointer once at startup.
    """
    from graph.main_graph import build_graph
    from agents.registry import get_registry

    logger.info("Starting Georgian Tourism Agent...")

    registry = get_registry()
    registry.initialize()
    logger.info("AgentRegistry initialized")

    from guardrails.input_guardrails import InputGuardrails
    app.state.guardrails = InputGuardrails()
    logger.info("InputGuardrails initialized")

    settings = get_settings()

    # Single shared pool for checkpointer + MemoryManager — avoids two separate
    # connections to the same Postgres instance.
    from src.memory.db import init_pool, close_pool
    pool = await init_pool(settings.database_url)

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    checkpointer = AsyncPostgresSaver(conn=pool)
    await checkpointer.setup()
    app.state.graph = build_graph(checkpointer=checkpointer)
    logger.info("LangGraph compiled with AsyncPostgresSaver (shared pool)")

    from src.memory.memory_manager import get_memory_manager
    await get_memory_manager().setup()
    logger.info("MemoryManager schema ready")

    from src.tools.embeddings import get_embedding_model
    logger.info("Warming up embedding model...")
    await asyncio.to_thread(get_embedding_model)
    logger.info("Embedding model ready")

    yield

    from src.tools.tool_cache import get_cache
    await get_cache().aclose()

    from src.memory.memory_manager import get_memory_manager
    await get_memory_manager().aclose()

    await close_pool()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Georgian Tourism Agent API",
    description="Multi-agent system for planning trips in Georgia",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

from api.router import router as api_router
from middleware.rate_limit import rate_limit_middleware
app.middleware("http")(rate_limit_middleware)
app.include_router(api_router, prefix="/api/v1", tags=["planning"])


@app.get("/")
async def root():
    return {
        "name": "Georgian Tourism Agent",
        "status": "running",
        "endpoints": {
            "health":  "/api/v1/health",
            "plan":    "/api/v1/plan",
            "metrics": "/metrics",
            "docs":    "/docs",
        },
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from monitoring.metrics import is_available, REGISTRY
    if not is_available() or REGISTRY is None:
        return Response(
            content="# prometheus_client not installed\n",
            media_type="text/plain",
        )
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
