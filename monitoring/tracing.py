"""
LangSmith tracing setup.

Call setup_langsmith_tracing() before importing LangGraph so LangChain can
pick up tracing environment variables.

Usage in main.py:
    from monitoring.tracing import setup_langsmith_tracing, get_run_config
    setup_langsmith_tracing()

Usage in graph.ainvoke():
    config = get_run_config(request_id="req-123", user_id="user-456")
    result = await graph.ainvoke(state, config=config)
"""

import os
import logging
from config.settings import get_settings

logger = logging.getLogger(__name__)


def setup_langsmith_tracing() -> bool:
    """
    Configure LangSmith tracing through environment variables.

    LangChain traces calls automatically when LANGCHAIN_TRACING_V2=true and
    LANGCHAIN_API_KEY is set.

    Returns:
        True when tracing is enabled, False when no key is available.
    """
    settings = get_settings()
    api_key = settings.langchain_api_key or settings.langsmith_api_key

    if not api_key:
        logger.warning("LangSmith: LANGCHAIN_API_KEY or LANGSMITH_API_KEY not found")
        return False

    # LangChain reads LANGCHAIN_API_KEY, so mirror the configured key there.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = (
        settings.langsmith_project or settings.langchain_project or "georgian-tourism-agents"
    )

    logger.info(
        f"LangSmith tracing enabled: project={os.environ['LANGCHAIN_PROJECT']}"
    )
    return True


def get_run_config(
    request_id: str,
    user_id: str = None,
    extra_tags: list = None,
) -> dict:
    """
    Create LangGraph config with LangSmith metadata.

    Adds request_id and user_id to each trace for filtering in LangSmith UI.

    Args:
        request_id: Unique request ID.
        user_id: Optional user ID.
        extra_tags: Extra filtering tags.

    Returns:
        Config dict for graph.ainvoke(state, config=config).

    Example:
        config = get_run_config("req-123", user_id="user-456")
        result = await graph.ainvoke(state, config=config)
    """
    tags = ["georgian-tourism"] + (extra_tags or [])

    metadata = {"request_id": request_id}
    if user_id:
        metadata["user_id"] = user_id

    return {
        "configurable": {"thread_id": request_id},
        "run_name": f"plan-{request_id[:8]}",
        "tags": tags,
        "metadata": metadata,
    }
