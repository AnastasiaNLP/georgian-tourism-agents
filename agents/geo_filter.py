"""
Post-search filtering by Georgian region.

Qdrant location strings are free-form, so filtering is done after vector
search using region keywords from config/georgia_regions.json.
"""

import json
import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_regions() -> dict:
    """
    Load the region keyword dictionary once.
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "config", "georgia_regions.json")
    
    if not os.path.exists(path):
        logger.error(f"Region dictionary not found: {path}")
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Normalize keywords for case-insensitive matching.
    normalized = {}
    for region, keywords in data.items():
        normalized[region] = [kw.lower() for kw in keywords]
    
    logger.info(f"Loaded region dictionary: {len(normalized)} regions")
    return normalized


def match_region(location: str, region: str) -> bool:
    """
    Return whether a location belongs to the requested region.
    
    Matching is case-insensitive and keyword based.
    
    Args:
        location: Qdrant location string, for example:
                  "Adjara, Mtsvane-Kontsxi (Green Cape), 9 km from Batumi")
        region: Region name, for example "Adjara"
    
    Returns:
        True when the location contains a region keyword.
    """
    if not location or not region:
        return False
    
    regions = load_regions()
    keywords = regions.get(region, [])
    
    if not keywords:
        logger.warning(f"Region '{region}' not found in dictionary")
        return False
    
    location_lower = location.lower()
    return any(kw in location_lower for kw in keywords)


def filter_by_region(
    places: list,
    region: str,
    min_results: int = 3,
) -> list:
    """
    Filter places by region.
    
    Args:
        places: Places returned by Qdrant.
        region: Target region, for example "Adjara".
        min_results: Minimum filtered results before falling back.
    
    Returns:
        Filtered places, or the original list if filtering is too narrow.
    
    Fallback logic favors returning enough results over returning none.
    """
    if not places or not region:
        return places
    
    filtered = [
        p for p in places
        if match_region(p.get("location", ""), region)
    ]
    
    logger.info(
        f"geo_filter: region={region}, "
        f"total={len(places)}, "
        f"filtered={len(filtered)}"
    )
    
    if len(filtered) >= min_results:
        return filtered
    
    # Fall back when region filtering is too narrow.
    logger.warning(
        f"geo_filter: only {len(filtered)} results for {region}; "
        f"need {min_results}, returning all {len(places)}"
    )
    return places


def get_region_keywords(region: str) -> list:
    """Return region keywords for diagnostics."""
    regions = load_regions()
    return regions.get(region, [])
