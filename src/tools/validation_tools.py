#src/tools/validation_tools.py
"""
Validation Tools - Feasibility Checking (CORRECTED)

Tools for validating itineraries and checking if plans are realistic.
The "skeptical" agent that says "this won't work".

IMPORTANT: This tool does NOT call other tools!
It only validates data that was already computed by other agents.
"""

from langchain_core.tools import tool
from typing import Optional


# ============================================================================
# Tool 9: Check Feasibility (CORRECTED - NO TOOL CALLS)
# ============================================================================

@tool
def check_feasibility(itinerary: dict, constraints: Optional[dict] = None) -> dict:
    """Check if an itinerary is realistic and feasible

    IMPORTANT: This tool expects itinerary to ALREADY have route data!
    The Geo Agent must have added route information before validation.

    Use this tool to validate a travel plan before presenting to user.
    Checks for:
    - Excessive travel times
    - Too many activities per day
    - Impossible distances
    - Timing conflicts
    - Unrealistic expectations

    Expected itinerary structure:
    {
        "days": [
            {
                "day": 1,
                "activities": [...],
                "routes": [  # Added by Geo Agent!
                    {
                        "from": "Place A",
                        "to": "Place B",
                        "distance_km": 25.3,
                        "duration_min": 30.5
                    }
                ],
                "total_distance_km": 25.3,  # Sum by Geo Agent
                "total_duration_min": 30.5   # Sum by Geo Agent
            }
        ]
    }

    Args:
        itinerary: Itinerary with route data from Geo Agent
        constraints: Optional constraints:
            - max_driving_per_day: int (km, default: 300)
            - max_activities_per_day: int (default: 5)
            - min_time_per_activity: int (minutes, default: 60)
            - max_driving_hours_per_day: float (hours, default: 4)

    Returns:
        Feasibility report:
        - feasible: bool (overall verdict)
        - issues: List of problems found
        - warnings: List of warnings
        - suggestions: List of improvements
        - score: float (0-1, confidence in feasibility)

    Example:
        # After Geo Agent has enriched the itinerary:
        report = check_feasibility(
            itinerary_with_routes,
            constraints={"max_driving_per_day": 200}
        )
    """

    # Parse constraints
    const = constraints or {}
    max_driving_km = const.get("max_driving_per_day", 300)
    max_activities = const.get("max_activities_per_day", 5)
    min_time_per_activity = const.get("min_time_per_activity", 60)  # minutes
    max_driving_hours = const.get("max_driving_hours_per_day", 4)

    issues = []
    warnings = []
    suggestions = []

    days = itinerary.get("days", [])

    if not days:
        return {
            "feasible": False,
            "issues": ["No days in itinerary"],
            "warnings": [],
            "suggestions": [],
            "score": 0.0
        }

    # Check each day
    for day in days:
        day_num = day.get("day", 0)
        activities = day.get("activities", [])

        # ========================================================================
        # Check 1: Too many activities
        # ========================================================================
        if len(activities) > max_activities:
            issues.append(
                f"Day {day_num}: {len(activities)} activities exceeds limit of {max_activities}"
            )

        # ========================================================================
        # Check 2: Minimum time per activity
        # ========================================================================
        available_time = 12 * 60  # 12 hours in minutes (8am-8pm)
        required_activity_time = len(activities) * min_time_per_activity

        if required_activity_time > available_time:
            issues.append(
                f"Day {day_num}: Activities need {required_activity_time/60:.1f}h "
                f"but only {available_time/60}h available"
            )

        # ========================================================================
        # Check 3: Driving distance and time (FROM GEO AGENT DATA!)
        # ========================================================================

        # Get pre-calculated route data from Geo Agent
        total_distance = day.get("total_distance_km", 0)
        total_drive_time = day.get("total_duration_min", 0)
        routes = day.get("routes", [])

        # If no route data, check if we expected it
        if len(activities) > 1 and not routes:
            warnings.append(
                f"Day {day_num}: Multiple activities but no route data. "
                "Geo Agent should have calculated routes first."
            )
            # Continue with other checks

        # Check distance limit
        if total_distance > max_driving_km:
            issues.append(
                f"Day {day_num}: {total_distance:.1f}km driving exceeds "
                f"limit of {max_driving_km}km"
            )

        # Check driving time limit
        if total_drive_time / 60 > max_driving_hours:
            issues.append(
                f"Day {day_num}: {total_drive_time/60:.1f}h driving exceeds "
                f"limit of {max_driving_hours}h"
            )

        # ========================================================================
        # Check 4: Total time feasibility
        # ========================================================================

        activity_time = len(activities) * min_time_per_activity
        drive_time = total_drive_time
        buffer_time = 60  # 1 hour buffer for meals, breaks

        total_required = activity_time + drive_time + buffer_time

        if total_required > available_time:
            issues.append(
                f"Day {day_num}: Total time {total_required/60:.1f}h exceeds "
                f"available {available_time/60}h "
                f"(activities: {activity_time/60:.1f}h, "
                f"driving: {drive_time/60:.1f}h, buffer: 1h)"
            )

            # Suggest how many activities would fit
            max_possible_activities = int((available_time - drive_time - buffer_time) / min_time_per_activity)
            if max_possible_activities > 0:
                suggestions.append(
                    f"Day {day_num}: Reduce to {max_possible_activities} activities "
                    f"to fit the time"
                )
        elif total_required > available_time * 0.9:
            warnings.append(
                f"Day {day_num}: Very tight schedule "
                f"({total_required/60:.1f}h / {available_time/60}h used)"
            )

        # ========================================================================
        # Check 5: Individual route segments (if available)
        # ========================================================================

        for route in routes:
            segment_distance = route.get("distance_km", 0)
            segment_duration = route.get("duration_min", 0)

            # Check for unusually long single segments
            if segment_distance > 200:
                warnings.append(
                    f"Day {day_num}: Long drive from {route.get('from')} "
                    f"to {route.get('to')} ({segment_distance:.1f}km)"
                )

            if segment_duration > 180:  # 3 hours
                warnings.append(
                    f"Day {day_num}: Long drive from {route.get('from')} "
                    f"to {route.get('to')} ({segment_duration/60:.1f}h)"
                )

    # ========================================================================
    # Check 6: Multi-day optimization
    # ========================================================================

    if len(days) > 1:
        # Check if visiting same location multiple times across days
        locations_by_day = {}
        for day in days:
            day_num = day.get("day")
            activities = day.get("activities", [])
            locations = [a.get("location") for a in activities]
            locations_by_day[day_num] = set(locations)

        # Find repeated locations
        all_locations = set()
        for day_num, locs in locations_by_day.items():
            repeated = all_locations & locs
            if repeated:
                suggestions.append(
                    f"Consider grouping activities in {', '.join(repeated)} "
                    "on the same day to reduce travel"
                )
            all_locations.update(locs)

    # ========================================================================
    # Check 7: Backtracking
    # ========================================================================

    # Check if route goes A → B → A (backtracking)
    for day in days:
        day_num = day.get("day")
        routes = day.get("routes", [])

        if len(routes) >= 2:
            for i in range(len(routes) - 1):
                current = routes[i]
                next_route = routes[i + 1]

                # Simplified check: if going to same general area twice
                if current.get("from") == next_route.get("to"):
                    warnings.append(
                        f"Day {day_num}: Possible backtracking detected "
                        f"({current.get('from')} → {current.get('to')} → "
                        f"{next_route.get('to')})"
                    )

    # ========================================================================
    # Calculate feasibility score
    # ========================================================================

    total_checks = len(days) * 5  # 5 main checks per day
    failed_checks = len(issues)

    score = max(0.0, 1.0 - (failed_checks / total_checks))

    # Add bonus for no warnings
    if not warnings:
        score = min(1.0, score + 0.1)

    # Penalty for suggestions (mild)
    if suggestions:
        score = max(0.0, score - 0.05)

    # Determine overall feasibility
    feasible = len(issues) == 0

    return {
        "feasible": feasible,
        "score": round(score, 2),
        "issues": issues,
        "warnings": warnings,
        "suggestions": suggestions,
        "days_checked": len(days),
        "summary": (
            f"Plan is {'FEASIBLE' if feasible else 'NOT FEASIBLE'}. "
            f"Found {len(issues)} issues, {len(warnings)} warnings, "
            f"{len(suggestions)} suggestions. "
            f"Confidence: {score:.0%}"
        )
    }