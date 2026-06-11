# src/tools/search_tools.py
"""
Search Tools - Information Gathering

Pure functions for searching and collecting facts.
No decisions, no reasoning - just data retrieval.
"""

import threading
from typing import List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from config.settings import get_settings


# ============================================================================
# Qdrant client singleton — one connection pool per worker process
# ============================================================================

_qdrant_client = None
_qdrant_lock = threading.Lock()


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            if _qdrant_client is None:
                settings = get_settings()
                _qdrant_client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    timeout=10,
                )
    return _qdrant_client


# ============================================================================
# Schemas
# ============================================================================

class SearchFilters(BaseModel):
    """Filters for Qdrant search"""
    location: Optional[str] = Field(None, description="Filter by location (e.g., 'Tbilisi')")
    category: Optional[str] = Field(None, description="Filter by category (e.g., 'museum', 'church')")
    is_popular: Optional[bool] = Field(None, description="Filter popular places only")
    is_unesco: Optional[bool] = Field(None, description="Filter UNESCO sites only")


class SearchResult(BaseModel):
    """Single search result"""
    id: str = Field(..., description="Place ID")
    name: str = Field(..., description="Place name")
    description: str = Field(..., description="Description")
    location: str = Field(..., description="Location/city")
    category: Optional[str] = Field(None, description="Category (museum, church, etc.)")
    score: float = Field(..., description="Relevance score")
    # Coordinates are added by GeoAgent, not read from Qdrant payload.
    tags: List[str] = Field(default_factory=list, description="Place tags")
    metadata: dict = Field(default_factory=dict, description="Full metadata")


class PlaceDetails(BaseModel):
    """Detailed information about a place"""
    id: str
    name: str
    description: str
    location: str
    coordinates: dict  # {"lat": float, "lon": float}
    category: str
    opening_hours: Optional[str] = None
    price_range: Optional[str] = None
    contact: Optional[str] = None
    website: Optional[str] = None
    rating: Optional[float] = None
    metadata: dict = Field(default_factory=dict)


# ============================================================================
# Tool 1: Search Qdrant
# ============================================================================

@tool
def search_qdrant(
    query: str,
    top_k: int = 10,
    filters: Optional[dict] = None
) -> List[dict]:
    """Search Qdrant vector database for Georgian tourism places

    Use this tool to find places, attractions, hotels, restaurants in Georgia.

    Args:
        query: Search query (e.g., "churches in Tbilisi", "best restaurants")
        top_k: Number of results to return (1-20)
        filters: Optional filters as dict:
            - location: str (e.g., "Tbilisi")
            - category: str (e.g., "museum")
            - is_popular: bool
            - is_unesco: bool

    Returns:
        List of search results with place information

    Example:
        results = search_qdrant("best churches in Mtskheta", top_k=5)
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.collection_name

    # Build Qdrant filter
    qdrant_filter = None
    if filters:
        conditions = []

        if filters.get("location"):
            conditions.append(
                FieldCondition(
                    key="location",
                    match=MatchValue(value=filters["location"])
                )
            )

        if filters.get("category"):
            conditions.append(
                FieldCondition(
                    key="category",
                    match=MatchValue(value=filters["category"])
                )
            )

        if filters.get("is_popular") is not None:
            conditions.append(
                FieldCondition(
                    key="is_popular",
                    match=MatchValue(value=filters["is_popular"])
                )
            )

        if filters.get("is_unesco") is not None:
            conditions.append(
                FieldCondition(
                    key="is_unesco_site",
                    match=MatchValue(value=filters["is_unesco"])
                )
            )

        if filters.get("region"):
            conditions.append(
                FieldCondition(
                    key="region",
                    match=MatchValue(value=filters["region"])
                )
            )

        if conditions:
            qdrant_filter = Filter(must=conditions)

    # Create embedding for query using the shared singleton model.
    from src.tools.embeddings import embed
    query_vector = embed(query)

    # Execute search
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter
    ).points

    # Coordinates are added by GeoAgent, not read from Qdrant payload.
    # Tags are used for deduplication and intent filtering.
    return [
        {
            "id": str(r.id),
            "name": r.payload.get("name", "Unknown"),
            "description": r.payload.get("description", ""),
            "location": r.payload.get("location", "Georgia"),
            "category": r.payload.get("category"),
            "tags": r.payload.get("tags", []),
            "score": float(r.score),
            "metadata": r.payload,
        }
        for r in results
    ]
