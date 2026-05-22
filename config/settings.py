"""
Runtime settings for Georgian Tourism Agent.

This module is the small, explicit configuration surface used by runtime code.
It reads only process environment variables; it does not load `.env`.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel


class Settings(BaseModel, frozen=True):
    environment: str = "development"
    debug: bool = True

    # API auth / web
    api_key: Optional[str] = None
    cors_origins: str = "http://localhost:3000"
    rate_limit_rpm: int = 10

    # LLM providers
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"

    # Vector search
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    collection_name: str = "georgian_attractions"

    # Embeddings / memory
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    vector_size: int = 384
    upstash_redis_url: Optional[str] = None
    upstash_redis_token: Optional[str] = None

    # Geo / weather
    ors_api_key: Optional[str] = None
    openweather_api_key: Optional[str] = None

    # LangSmith
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "georgian-tourism-agents"
    langchain_api_key: Optional[str] = None
    langchain_project: Optional[str] = None
    langchain_tracing_v2: bool = False

    # Feature flags
    feature_use_weather: bool = False
    feature_enable_rag: bool = True
    feature_enable_geo_routes: bool = True
    feature_enable_validation: bool = True
    feature_enable_eval: bool = False
    max_search_results: int = 10

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


@lru_cache
def get_settings() -> Settings:
    return Settings(
        environment=os.getenv("ENVIRONMENT", "development"),
        debug=_get_bool("DEBUG", True),
        api_key=os.getenv("API_KEY"),
        cors_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000"),
        rate_limit_rpm=_get_int("RATE_LIMIT_RPM", 10),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=os.getenv("COLLECTION_NAME", "georgian_attractions"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ),
        vector_size=_get_int("VECTOR_SIZE", 384),
        upstash_redis_url=os.getenv("UPSTASH_REDIS_URL"),
        upstash_redis_token=os.getenv("UPSTASH_REDIS_TOKEN"),
        ors_api_key=os.getenv("ORS_API_KEY"),
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "georgian-tourism-agents"),
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY"),
        langchain_project=os.getenv("LANGCHAIN_PROJECT"),
        langchain_tracing_v2=_get_bool("LANGCHAIN_TRACING_V2", False),
        feature_use_weather=_get_bool("FEATURE_USE_WEATHER", False),
        feature_enable_rag=_get_bool("FEATURE_ENABLE_RAG", True),
        feature_enable_geo_routes=_get_bool("FEATURE_ENABLE_GEO_ROUTES", True),
        feature_enable_validation=_get_bool("FEATURE_ENABLE_VALIDATION", True),
        feature_enable_eval=_get_bool("FEATURE_ENABLE_EVAL", False),
        max_search_results=_get_int("MAX_SEARCH_RESULTS", 10),
    )
