from typing import Any, Dict, List, Optional, Tuple
"""
Distance estimation tools for planning.
"""

import json
import math
from pathlib import Path
from functools import lru_cache

from langchain_core.tools import tool


# ============================================================================
# Load Distance Matrix
# ============================================================================

@lru_cache(maxsize=1)
def load_distance_matrix() -> Dict[str, dict]:
    """
    Load pre-computed distance matrix.

    Cached distance matrix, loaded once.

    Returns:
        Distances between common locations.
    """
    config_path = Path(__file__).parent.parent / "config" / "distance_matrix.json"

    if not config_path.exists():
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("distances", {})


# ============================================================================
# Haversine Distance (fallback)
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points on Earth.

    Args:
        lat1, lon1: First point
        lat2, lon2: Second point

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # Haversine formula
    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# ============================================================================
# City Coordinates (for fallback)
# ============================================================================

CITY_COORDINATES = {
    "tbilisi": (41.7151, 44.8271),
    "batumi": (41.6168, 41.6367),
    "kutaisi": (42.2488, 42.6988),
    "mtskheta": (41.8458, 44.7208),
    "telavi": (41.9185, 45.4733),
    "sighnaghi": (41.6186, 45.9214),
    "kazbegi": (42.6589, 44.6456),
    "borjomi": (41.8417, 43.3833),
    "gori": (41.9839, 44.1089),
    "mestia": (43.0444, 42.7289),
    "bakuriani": (41.7500, 43.5333),
    "ananuri": (42.1667, 44.7),
    "gudauri": (42.4789, 44.4719),
}


def normalize_city_name(city: str) -> str:
    """Normalize city name for lookup"""
    return city.lower().strip().replace(" ", "").replace("-", "")


def get_city_coords(city: str) -> Optional[Tuple[float, float]]:
    """Get coordinates for city"""
    normalized = normalize_city_name(city)
    return CITY_COORDINATES.get(normalized)


# ============================================================================
# Main Tool: Estimate Travel Time
# ============================================================================

@tool
async def estimate_travel_time(from_city: str, to_city: str) -> dict:
    """
    Estimate travel time and distance between two cities in Georgia.

    This is a QUICK estimate for planning. Uses:
    1. Pre-computed matrix (if available)
    2. Haversine distance * 1.3 (accounting for roads)

    Args:
        from_city: Starting city
        to_city: Destination city

    Returns:
        dict with:
        - km: float (distance in kilometers)
        - hours: float (estimated travel time)
        - source: str ("matrix" | "estimated")

    Example:
        result = await estimate_travel_time("Tbilisi", "Batumi")
        # {"km": 380, "hours": 5.5, "source": "matrix"}
    """
    # Normalize names
    from_norm = normalize_city_name(from_city)
    to_norm = normalize_city_name(to_city)

    # Same city
    if from_norm == to_norm:
        return {"km": 0, "hours": 0, "source": "same_city"}

    # Try matrix (both directions)
    matrix = load_distance_matrix()

    for key in [f"{from_city}-{to_city}", f"{to_city}-{from_city}"]:
        if key in matrix:
            data = matrix[key]
            return {
                "km": data["km"],
                "hours": data["hours"],
                "source": "matrix"
            }

    # Fallback: haversine estimate
    from_coords = get_city_coords(from_city)
    to_coords = get_city_coords(to_city)

    if from_coords and to_coords:
        straight_km = haversine_distance(
            from_coords[0], from_coords[1],
            to_coords[0], to_coords[1]
        )

        # Roads are ~30% longer than straight line
        road_km = straight_km * 1.3

        # Estimate time (average speed ~60 km/h in Georgia)
        hours = road_km / 60

        return {
            "km": round(road_km, 1),
            "hours": round(hours, 1),
            "source": "estimated"
        }

    # No data available
    return {
        "km": None,
        "hours": None,
        "source": "unknown",
        "error": f"No data for route {from_city} → {to_city}"
    }


# ============================================================================
# Helper: Check Feasibility
# ============================================================================

def check_day_feasibility(
    activities: List[dict],
    pace: str = "moderate"
) -> dict:
    """
    Check if activities in one day are feasible.

    Args:
        activities: List of activities with locations
        pace: Trip pace (relaxed/moderate/intensive)

    Returns:
        dict with feasibility info
    """
    # Max distances per day based on pace
    MAX_DISTANCE = {
        "relaxed": 100,     # 100 km max
        "moderate": 150,    # 150 km max
        "intensive": 200,   # 200 km max
    }

    max_km = MAX_DISTANCE.get(pace, 150)

    # Calculate total distance
    total_km = 0
    total_hours = 0
    routes = []

    for i in range(len(activities) - 1):
        from_loc = activities[i].get("location", "")
        to_loc = activities[i + 1].get("location", "")

        # This would be async in real usage, but simplified here
        # In Planning Agent, use estimate_travel_time tool
        routes.append(f"{from_loc} → {to_loc}")

    return {
        "is_feasible": total_km <= max_km,
        "total_km_estimate": total_km,
        "total_hours_estimate": total_hours,
        "max_km_for_pace": max_km,
        "routes": routes,
        "warning": f"Day might be too long ({total_km} km > {max_km} km)" if total_km > max_km else None
    }


# ============================================================================
# Tool: Get Pace Limits
# ============================================================================

@tool
def get_pace_limits(pace: str) -> dict:
    """
    Get distance/activity limits for a given pace.

    Args:
        pace: Trip pace (relaxed/moderate/intensive)

    Returns:
        dict with limits:
        - max_distance_km: Maximum distance per day
        - max_activities: Maximum activities per day
        - max_driving_hours: Maximum driving time per day

    Example:
        limits = get_pace_limits("moderate")
        # {"max_distance_km": 150, "max_activities": 4, "max_driving_hours": 3}
    """
    PACE_LIMITS = {
        "relaxed": {
            "max_distance_km": 100,
            "max_activities": 3,
            "max_driving_hours": 2,
            "description": "Slow travel, lots of time at each place"
        },
        "moderate": {
            "max_distance_km": 150,
            "max_activities": 4,
            "max_driving_hours": 3,
            "description": "Balanced travel, reasonable pace"
        },
        "intensive": {
            "max_distance_km": 200,
            "max_activities": 6,
            "max_driving_hours": 4,
            "description": "Fast travel, many activities"
        }
    }

    return PACE_LIMITS.get(pace, PACE_LIMITS["moderate"])
