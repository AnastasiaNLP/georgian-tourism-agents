"""
Runtime settings for Georgian Tourism Agent.

This module is the small, explicit configuration surface used by runtime code.
It reads only process environment variables; it does not load `.env`.
All variables listed in `.env.example` should be reflected here.
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
    database_url: str = ""
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
    mapbox_token: Optional[str] = None
    openweather_api_key: Optional[str] = None

    # Media providers
    unsplash_url: Optional[str] = None
    unsplash_access_key: Optional[str] = None
    unsplash_secret_key: Optional[str] = None
    cloudinary_cloud_name: Optional[str] = None
    cloudinary_api_key: Optional[str] = None
    cloudinary_api_secret: Optional[str] = None

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

    # Validation thresholds (override per deployment via env)
    validation_max_km_relaxed: int = 100
    validation_max_km_moderate: int = 150
    validation_max_km_intensive: int = 200
    validation_max_acts_relaxed: int = 3
    validation_max_acts_moderate: int = 4
    validation_max_acts_intensive: int = 6
    validation_max_driving_hours: int = 4
    # How far a day's reported distance may fall below the measured route before
    # it is treated as an error (0.25 = reported may be up to 25% lower).
    validation_distance_tolerance: float = 0.25
    # Minimum fraction of route legs that must have a measured distance before
    # the itinerary is trusted; below this the run is marked degraded.
    validation_min_grounding: float = 0.5

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


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


@lru_cache
def get_settings() -> Settings:
    return Settings(
        environment=os.getenv("ENVIRONMENT", "development"),
        debug=_get_bool("DEBUG", True),
        api_key=os.getenv("API_KEY"),
        cors_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000"),
        rate_limit_rpm=_get_int("RATE_LIMIT_RPM", 10),
        database_url=os.getenv("DATABASE_URL", ""),
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
        mapbox_token=os.getenv("MAPBOX_TOKEN"),
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
        unsplash_url=os.getenv("UNSPLASH_URL"),
        unsplash_access_key=os.getenv("UNSPLASH_ACCESS_KEY"),
        unsplash_secret_key=os.getenv("UNSPLASH_SECRET_KEY"),
        cloudinary_cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        cloudinary_api_key=os.getenv("CLOUDINARY_API_KEY"),
        cloudinary_api_secret=os.getenv("CLOUDINARY_API_SECRET"),
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
        validation_max_km_relaxed=_get_int("VALIDATION_MAX_KM_RELAXED", 100),
        validation_max_km_moderate=_get_int("VALIDATION_MAX_KM_MODERATE", 150),
        validation_max_km_intensive=_get_int("VALIDATION_MAX_KM_INTENSIVE", 200),
        validation_max_acts_relaxed=_get_int("VALIDATION_MAX_ACTS_RELAXED", 3),
        validation_max_acts_moderate=_get_int("VALIDATION_MAX_ACTS_MODERATE", 4),
        validation_max_acts_intensive=_get_int("VALIDATION_MAX_ACTS_INTENSIVE", 6),
        validation_max_driving_hours=_get_int("VALIDATION_MAX_DRIVING_HOURS", 4),
        validation_distance_tolerance=_get_float("VALIDATION_DISTANCE_TOLERANCE", 0.25),
        validation_min_grounding=_get_float("VALIDATION_MIN_GROUNDING", 0.5),
    )
