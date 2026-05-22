#src/tools/planning_tools.py
"""
Planning Tools - Itinerary Generation

Tools for creative planning and itinerary generation.
These tools help organize places into a coherent travel plan.
"""

from langchain_core.tools import tool
from typing import List, Optional
from datetime import datetime, timedelta


# ============================================================================
# Tool 8: Generate Itinerary
# ============================================================================

@tool
def generate_itinerary(
    days: int,
    places: List[dict],
    preferences: Optional[dict] = None
) -> dict:
    """Generate a multi-day travel itinerary

    Use this tool to organize places into a day-by-day travel plan.
    Takes into account:
    - Number of days
    - Place locations (to minimize travel)
    - Opening hours
    - Typical visit duration
    - User preferences

    Args:
        days: Number of days for the trip (1-14)
        places: List of places from search results, each with:
            - name: str
            - location: str
            - category: str
            - lat, lon: float (coordinates)
        preferences: Optional preferences:
            - start_location: str (starting city)
            - pace: str ("relaxed", "moderate", "intensive")
            - interests: List[str] (e.g., ["history", "nature"])

    Returns:
        Itinerary with daily plans:
        - days: List of day plans
        - summary: Overall summary

    Example:
        itinerary = generate_itinerary(
            days=3,
            places=[...],
            preferences={"pace": "moderate", "start_location": "Tbilisi"}
        )
    """

    if days < 1 or days > 14:
        return {"error": "Days must be between 1 and 14"}

    if not places:
        return {"error": "No places provided"}

    # Parse preferences
    prefs = preferences or {}
    start_location = prefs.get("start_location", "Tbilisi")
    pace = prefs.get("pace", "moderate")
    interests = prefs.get("interests", [])

    # Activity limits per day based on pace
    activities_per_day = {
        "relaxed": 2,
        "moderate": 3,
        "intensive": 4
    }
    max_activities = activities_per_day.get(pace, 3)

    # Group places by location for efficient routing
    places_by_location = {}
    for place in places:
        loc = place.get("location", "Unknown")
        if loc not in places_by_location:
            places_by_location[loc] = []
        places_by_location[loc].append(place)

    # Generate daily plans
    daily_plans = []
    remaining_places = places.copy()
    current_location = start_location

    for day_num in range(1, days + 1):
        if not remaining_places:
            break

        day_activities = []

        # Prioritize places in current location
        local_places = [p for p in remaining_places if p.get("location") == current_location]
        other_places = [p for p in remaining_places if p.get("location") != current_location]

        # Fill day with activities
        for _ in range(max_activities):
            if local_places:
                place = local_places.pop(0)
            elif other_places:
                place = other_places.pop(0)
                current_location = place.get("location", current_location)
            else:
                break

            day_activities.append({
                "name": place.get("name"),
                "location": place.get("location"),
                "category": place.get("category"),
                "description": place.get("description", "")[:200] + "...",
                "coordinates": {
                    "lat": place.get("lat"),
                    "lon": place.get("lon")
                }
            })

            remaining_places.remove(place)

        daily_plans.append({
            "day": day_num,
            "date": None,  # Will be filled by agent with actual dates
            "location": current_location,
            "activities": day_activities,
            "activity_count": len(day_activities)
        })

    return {
        "days": daily_plans,
        "total_days": len(daily_plans),
        "total_activities": sum(len(d["activities"]) for d in daily_plans),
        "start_location": start_location,
        "pace": pace,
        "summary": f"{len(daily_plans)}-day {pace} itinerary starting from {start_location}"
    }


# ============================================================================
# Helper: Optimize itinerary order
# ============================================================================

def optimize_itinerary_routing(itinerary: dict) -> dict:
    """Optimize the order of activities within each day to minimize travel

    This is a helper function that:
    1. Takes an itinerary
    2. Reorders activities per day to minimize total distance
    3. Returns optimized itinerary

    Args:
        itinerary: Itinerary from generate_itinerary

    Returns:
        Optimized itinerary
    """
    from tools_geo import get_route

    optimized_days = []

    for day in itinerary.get("days", []):
        activities = day.get("activities", [])

        if len(activities) <= 2:
            # No need to optimize
            optimized_days.append(day)
            continue

        # Simple greedy optimization: always go to nearest next place
        optimized_activities = []
        remaining = activities.copy()

        # Start with first activity
        current = remaining.pop(0)
        optimized_activities.append(current)

        while remaining:
            current_coords = current.get("coordinates", {})

            if not current_coords.get("lat") or not current_coords.get("lon"):
                # No coordinates, just add next
                current = remaining.pop(0)
                optimized_activities.append(current)
                continue

            # Find nearest remaining place
            min_distance = float('inf')
            nearest_idx = 0

            for i, place in enumerate(remaining):
                place_coords = place.get("coordinates", {})

                if not place_coords.get("lat") or not place_coords.get("lon"):
                    continue

                # Calculate distance (simplified)
                # In production, could use get_route here
                lat_diff = abs(current_coords["lat"] - place_coords["lat"])
                lon_diff = abs(current_coords["lon"] - place_coords["lon"])
                distance = (lat_diff ** 2 + lon_diff ** 2) ** 0.5

                if distance < min_distance:
                    min_distance = distance
                    nearest_idx = i

            current = remaining.pop(nearest_idx)
            optimized_activities.append(current)

        # Update day with optimized activities
        day["activities"] = optimized_activities
        optimized_days.append(day)

    itinerary["days"] = optimized_days
    return itinerary