#src/tools/geo_tools.py
from typing import List, Dict, Optional
"""
Geo Tools - Spatial Logic and Geography

Pure functions for geocoding and routing.
No LLM reasoning - just precise geographic calculations.
"""

import requests
from langchain_core.tools import tool
from typing import Optional
from config.settings import get_settings


# ============================================================================
# Tool 4: Geocode City
# ============================================================================

@tool
def geocode_city(city: str) -> dict:
    """Get coordinates for a place name using OpenRouteService.

    Args:
        city: Place name, optionally with location context.
              Examples: "Batumi Botanical Garden, Batumi"
                        "Europe Square, Batumi"
                        "Mestia"

    Returns:
        {"lat": float, "lon": float, "display_name": str}
        or {"error": str} if not found.
    """
    ORS_API_KEY = get_settings().ors_api_key

    if not ORS_API_KEY:
        return {"error": "ORS_API_KEY not found in environment"}

    # OpenRouteService Geocoding API
    url = "https://api.openrouteservice.org/geocode/search"

    params = {
        "api_key": ORS_API_KEY,
        "text": city,
        "boundary.country": "GE",   # Limit to Georgia
        "focus.point.lat": 41.9,    # Georgia centroid — biases results toward Georgia
        "focus.point.lon": 44.0,    # when boundary.country has no exact match
        "size": 3,                  # Fetch top 3, pick first within Georgia bbox
    }

    # Georgia bounding box for post-filter.
    # ORS/Pelias may return out-of-country results when boundary.country finds no match.
    _LAT = (41.0, 43.6)
    _LON = (40.5, 46.7)

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if not data.get("features"):
            return {"error": f"City '{city}' not found"}

        # Pick first result that falls within Georgia bounding box.
        feature = None
        for f in data["features"]:
            c = f["geometry"]["coordinates"]
            if _LAT[0] <= c[1] <= _LAT[1] and _LON[0] <= c[0] <= _LON[1]:
                feature = f
                break
        if feature is None:
            return {"error": f"No result within Georgia for '{city}'"}

        coords = feature["geometry"]["coordinates"]
        # OpenRouteService returns [lon, lat]
        lon, lat = coords

        return {
            "lat": lat,
            "lon": lon,
            "display_name": feature["properties"].get("label", city)
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Geocoding failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# ============================================================================
# Tool 5: Get Route
# ============================================================================

@tool
def get_route(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float
) -> dict:
    """Calculate route between two points using coordinates

    Use this tool ONLY if you already have coordinates
    for start and end points.

    IMPORTANT:
    - Always call geocode_city() FIRST to get coordinates
    - Never make up coordinates yourself
    - Coordinates must be from geocode_city or database

    Args:
        start_lon: Longitude of starting point
        start_lat: Latitude of starting point
        end_lon: Longitude of destination
        end_lat: Latitude of destination

    Returns:
        Dictionary with route information:
        - distance_km: Distance in kilometers
        - duration_min: Duration in minutes
        - start_point: Starting coordinates
        - end_point: Ending coordinates

    Example:
        # First get coordinates
        tbilisi = geocode_city("Tbilisi")
        mtskheta = geocode_city("Mtskheta")

        # Then calculate route
        route = get_route(
            start_lon=tbilisi["lon"],
            start_lat=tbilisi["lat"],
            end_lon=mtskheta["lon"],
            end_lat=mtskheta["lat"]
        )
        # Returns: {"distance_km": 25.3, "duration_min": 30.5}
    """
    ORS_API_KEY = get_settings().ors_api_key

    if not ORS_API_KEY:
        return {"error": "ORS_API_KEY not found in environment"}

    url = "https://api.openrouteservice.org/v2/directions/driving-car"

    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }

    body = {
        "coordinates": [
            [start_lon, start_lat],
            [end_lon, end_lat]
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=15)
        response.raise_for_status()

        data = response.json()

        # ORS v2 directions returns "routes", not "features".
        if "routes" in data and data["routes"]:
            summary = data["routes"][0]["summary"]
        elif "features" in data and data["features"]:
            summary = data["features"][0]["properties"]["summary"]
        else:
            return {"error": "Could not calculate route"}

        return {
            "distance_km": round(summary["distance"] / 1000, 2),
            "duration_min": round(summary["duration"] / 60, 1),
            "start_point": {
                "lon": start_lon,
                "lat": start_lat
            },
            "end_point": {
                "lon": end_lon,
                "lat": end_lat
            }
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Routing failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# ============================================================================
# Helper: Calculate multiple routes
# ============================================================================

def calculate_multi_stop_route(cities: List[str]) -> dict:
    """Calculate route through multiple cities

    This is a helper function (not a tool) that:
    1. Geocodes all cities
    2. Calculates routes between consecutive pairs
    3. Returns total distance and time

    Args:
        cities: List of city names in order

    Returns:
        Route summary with total distance and time
    """
    if len(cities) < 2:
        return {"error": "Need at least 2 cities"}

    # Geocode all cities
    coordinates = []
    for city in cities:
        result = geocode_city(city)
        if "error" in result:
            return {"error": f"Could not geocode {city}: {result['error']}"}
        coordinates.append(result)

    # Calculate routes between consecutive pairs
    routes = []
    total_distance = 0
    total_duration = 0

    for i in range(len(coordinates) - 1):
        start = coordinates[i]
        end = coordinates[i + 1]

        route = get_route(
            start_lon=start["lon"],
            start_lat=start["lat"],
            end_lon=end["lon"],
            end_lat=end["lat"]
        )

        if "error" in route:
            return {"error": f"Could not route {cities[i]} → {cities[i+1]}: {route['error']}"}

        routes.append({
            "from": cities[i],
            "to": cities[i + 1],
            "distance_km": route["distance_km"],
            "duration_min": route["duration_min"]
        })

        total_distance += route["distance_km"]
        total_duration += route["duration_min"]

    return {
        "cities": cities,
        "routes": routes,
        "total_distance_km": round(total_distance, 2),
        "total_duration_min": round(total_duration, 1),
        "total_duration_hours": round(total_duration / 60, 1)
    }
