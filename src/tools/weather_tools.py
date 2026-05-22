#src/tools/weather_toos.py
"""
Weather and Availability Tools

Tools for checking weather conditions and place availability.
"""

import requests
from datetime import datetime, timedelta
from langchain_core.tools import tool
from typing import Optional
from config.settings import get_settings


# ============================================================================
# Tool 6: Get Weather
# ============================================================================

@tool
def get_weather(location: str, date: Optional[str] = None) -> dict:
    """Get weather forecast for a location

    Use this tool to check weather conditions for travel planning.
    Helps determine if it's a good day for outdoor activities.

    Args:
        location: City name (e.g., "Tbilisi", "Batumi")
        date: Date in format "YYYY-MM-DD" (optional, defaults to today)

    Returns:
        Weather information:
        - temperature: Temperature in Celsius
        - condition: Weather condition (sunny, rainy, etc.)
        - precipitation: Chance of rain (%)
        - wind_speed: Wind speed in km/h
        - date: Date of forecast

    Example:
        weather = get_weather("Tbilisi", "2025-02-15")
    """

    # Use OpenWeatherMap API (free tier)
    API_KEY = get_settings().openweather_api_key

    if not API_KEY:
        # Return mock data if no API key (for development)
        return {
            "location": location,
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "temperature": 15,
            "condition": "partly cloudy",
            "precipitation": 20,
            "wind_speed": 10,
            "note": "Mock data - OPENWEATHER_API_KEY not configured"
        }

    try:
        # First geocode to get coordinates
        geocode_url = f"http://api.openweathermap.org/geo/1.0/direct"
        geocode_params = {
            "q": f"{location},GE",
            "limit": 1,
            "appid": API_KEY
        }

        geo_response = requests.get(geocode_url, params=geocode_params, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data:
            return {"error": f"Location '{location}' not found"}

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]

        # Get weather forecast
        weather_url = "http://api.openweathermap.org/data/2.5/forecast"
        weather_params = {
            "lat": lat,
            "lon": lon,
            "appid": API_KEY,
            "units": "metric"  # Celsius
        }

        weather_response = requests.get(weather_url, params=weather_params, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        # Parse target date
        if date:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        else:
            target_date = datetime.now()

        # Find forecast for target date (closest match)
        forecasts = weather_data.get("list", [])
        closest_forecast = None
        min_diff = float('inf')

        for forecast in forecasts:
            forecast_time = datetime.fromtimestamp(forecast["dt"])
            diff = abs((forecast_time - target_date).total_seconds())

            if diff < min_diff:
                min_diff = diff
                closest_forecast = forecast

        if not closest_forecast:
            return {"error": "No forecast data available"}

        return {
            "location": location,
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "temperature": round(closest_forecast["main"]["temp"], 1),
            "feels_like": round(closest_forecast["main"]["feels_like"], 1),
            "condition": closest_forecast["weather"][0]["description"],
            "precipitation": closest_forecast.get("pop", 0) * 100,  # Probability of precipitation
            "humidity": closest_forecast["main"]["humidity"],
            "wind_speed": round(closest_forecast["wind"]["speed"] * 3.6, 1),  # m/s to km/h
            "coordinates": {"lat": lat, "lon": lon}
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Weather API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# ============================================================================
# Tool 7: Check Availability
# ============================================================================

@tool
def check_availability(place_id: str, date: str) -> dict:
    """Check if a place is available on a specific date

    Use this tool to verify:
    - Opening hours
    - Special closures
    - Holidays
    - Booking requirements

    Args:
        place_id: Place ID from search results
        date: Date in format "YYYY-MM-DD"

    Returns:
        Availability information:
        - available: Boolean
        - reason: Explanation if not available
        - opening_hours: Hours on that day
        - requires_booking: Whether booking is needed

    Example:
        avail = check_availability("place_123", "2025-02-15")
    """

    # NOTE: This is a mock implementation
    # In production, this would:
    # 1. Query database for special closures
    # 2. Check national holidays
    # 3. Integrate with booking APIs if available
    # 4. Check seasonal closures

    try:
        # Parse date
        check_date = datetime.strptime(date, "%Y-%m-%d")
        day_of_week = check_date.strftime("%A")

        # Georgian national holidays (2025)
        holidays = [
            "2025-01-01",  # New Year
            "2025-01-02",  # New Year
            "2025-01-07",  # Orthodox Christmas
            "2025-01-19",  # Epiphany
            "2025-03-03",  # Mother's Day
            "2025-03-08",  # Women's Day
            "2025-04-09",  # National Unity Day
            "2025-04-18",  # Good Friday (Orthodox)
            "2025-04-19",  # Easter Saturday
            "2025-04-20",  # Easter Sunday
            "2025-04-21",  # Easter Monday
            "2025-05-09",  # Victory Day
            "2025-05-12",  # St. Andrew's Day
            "2025-05-26",  # Independence Day
            "2025-08-28",  # Assumption of Mary
            "2025-10-14",  # Svetitskhoveli Cathedral Day
            "2025-11-23",  # St. George's Day
        ]

        is_holiday = date in holidays

        # Mock logic
        return {
            "place_id": place_id,
            "date": date,
            "day_of_week": day_of_week,
            "available": not is_holiday,
            "reason": "National holiday" if is_holiday else "Open",
            "opening_hours": "09:00-18:00" if not is_holiday else "Closed",
            "requires_booking": False,
            "is_holiday": is_holiday,
            "note": "Mock data - integrate with real booking system in production"
        }

    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
