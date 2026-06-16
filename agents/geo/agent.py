"""
GeoAgent geocodes places and builds a driving distance matrix.

It uses direct async ORS-backed calls without an LLM.

Tools:
  - geocode_city(city) -> {lat, lon}
  - get_route(start_lat, start_lon, end_lat, end_lon) → {distance_km, duration_min}
"""
from __future__ import annotations
import asyncio
import logging

from langchain_core.tools import tool

from agents.base import make_scratchpad, route_key
from src.tools.geo_tools import GEO_FOCUS_LAT, GEO_FOCUS_LON

logger = logging.getLogger(__name__)


# ── Tools ───────────────────────────────────────────────────────────────────

@tool
def geocode_city(city: str) -> dict:
    """Get coordinates (lat, lon) for a place in Georgia.

    Args:
        city: Short place name. Examples: "Batumi", "Mestia", "Telavi"

    Returns:
        {"lat": float, "lon": float, "display_name": str}
        or {"error": str} if not found.
    """
    from src.tools.geo_tools import geocode_city as _geocode
    return _geocode.invoke({"city": city})


@tool
def get_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> dict:
    """Get driving distance and time between two coordinate points.

    Returns:
        {"distance_km": float, "duration_min": float}
        or {"error": str} if route not found.
    """
    from src.tools.geo_tools import get_route as _get_route
    return _get_route.invoke({
        "start_lon": start_lon, "start_lat": start_lat,
        "end_lon": end_lon, "end_lat": end_lat,
    })


async def geo_agent_node(state: dict) -> dict:
    """Geocode search results before itinerary planning."""
    has_search = bool(state.get("search_results"))

    if has_search:
        return await _geo_enrich_places(state)
    else:
        return {
            "errors": ["geo_agent: no search_results to enrich"],
            "agent_history": ["geo_agent"],
            "agent_scratchpad": make_scratchpad("geo_agent", "SKIPPED: no search_results"),
        }


# Georgia bounding box — only geographic constant we need.
# ORS already has boundary.country=GE, this catches malformed API responses.
_GEORGIA_LAT = (41.0, 43.6)
_GEORGIA_LON = (40.0, 46.7)


def _in_georgia(lat: float, lon: float) -> bool:
    return (_GEORGIA_LAT[0] <= lat <= _GEORGIA_LAT[1] and
            _GEORGIA_LON[0] <= lon <= _GEORGIA_LON[1])


def _in_envelope(lat: float, lon: float, envelope: tuple) -> bool:
    """Whether a point lies inside the (lat_min, lat_max, lon_min, lon_max) box."""
    lat_min, lat_max, lon_min, lon_max = envelope
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _city_from_location(location: str) -> str:
    """
    Extract city/area name from a Qdrant location string.

    Qdrant location field can use Latin or Cyrillic place names.
    Examples include "Batumi, Adjara, Georgia" and region-first formats
    such as "Georgia, Samegrelo-Upper Svaneti region, village Mazeri".
    We take the first comma-separated token as the most specific place hint.
    This value is appended to the geocoding query to help ORS disambiguate
    (e.g. "Europe Square, Batumi" instead of bare "Europe Square").
    No translation is needed because ORS/Pelias handles both Latin and Cyrillic.
    """
    if not location:
        return ""
    first = location.split(",")[0].strip()
    # Skip generic country-level tokens that don't help disambiguation
    skip = {"georgia", "грузия", ""}
    return "" if first.lower() in skip else first


async def _geo_enrich_places(state: dict) -> dict:
    """
    Geocode search results and build a distance matrix.

    Results are cached through cached_geocode_city and cached_get_route.
    """
    from src.tools.tool_cache import cached_geocode_city, cached_get_route  # noqa: F401 (async)

    request_id = state.get("request_id", "unknown")
    search_results = state.get("search_results") or []

    from config.settings import get_settings
    settings = get_settings()
    # Cap concurrent external tool calls so one request cannot saturate the shared
    # asyncio.to_thread pool, and concurrent requests queue predictably instead of
    # contending for every pool thread at once. Only actual tool calls hold a slot —
    # places resolved from payload coordinates take the fast path and skip it.
    sem = asyncio.Semaphore(settings.geo_max_concurrency)

    # Compute ORS focus point from places that already have Qdrant payload coordinates.
    # Using the centroid of known-good coordinates reduces hallucination for objects
    # (mountains, waterfalls) whose names ORS resolves by global text-matching.
    # Falls back to the Georgia centroid when no payload coordinates are present.
    # The focus point is only a soft ORS ranking hint; the acceptance envelope built
    # below is what actually rejects wrong-but-within-Georgia results, derived from the
    # result's own coordinates (no external region boundary data needed).
    # Note: on multi-region results the centroid drifts toward the country centre — that
    # weakens the focus hint, while the envelope simply widens to span both regions.
    _payload_coords = [
        (float((p.get("metadata") or {})["lat"]), float((p.get("metadata") or {})["lon"]))
        for p in search_results
        if isinstance((p.get("metadata") or {}).get("lat"), (int, float))
        and isinstance((p.get("metadata") or {}).get("lon"), (int, float))
        and _in_georgia(
            float((p.get("metadata") or {})["lat"]),
            float((p.get("metadata") or {})["lon"]),
        )
    ]
    if _payload_coords:
        focus_lat = sum(c[0] for c in _payload_coords) / len(_payload_coords)
        focus_lon = sum(c[1] for c in _payload_coords) / len(_payload_coords)
        logger.info(
            f"[{request_id}] geo: focus_point=({focus_lat:.2f},{focus_lon:.2f}) "
            f"from {len(_payload_coords)} Qdrant payload coords"
        )
    else:
        focus_lat, focus_lon = GEO_FOCUS_LAT, GEO_FOCUS_LON

    # Acceptance envelope for ORS results: the bounding box of the result's known-good
    # payload coordinates, expanded by a margin. ORS can return a plausible point that
    # is inside Georgia but in the wrong region (place-name collisions); such points
    # fall outside this envelope and are discarded.
    # Requires at least two anchors: a single payload coordinate would make the envelope
    # an arbitrary point±margin box that could wrongly reject legitimate places far from
    # the lone anchor inside a large region. With fewer than two anchors we cannot bound
    # the result without external region data, so only the country-level box applies.
    if len(_payload_coords) >= 2:
        margin = settings.geo_coord_envelope_margin_deg
        lats = [c[0] for c in _payload_coords]
        lons = [c[1] for c in _payload_coords]
        coord_envelope = (min(lats) - margin, max(lats) + margin,
                          min(lons) - margin, max(lons) + margin)
    else:
        coord_envelope = None

    # Step 1: geocode all places concurrently.
    async def geocode_place(place: dict) -> dict:
        name = place.get("name", "")
        loc = place.get("location", "")
        enriched = dict(place)

        # Prefer pre-computed coordinates from Qdrant payload (Wikidata backfill).
        # These are precise; ORS geocoding by name often hallucinates by 1000+ km.
        meta = place.get("metadata") or {}
        payload_lat = meta.get("lat")
        payload_lon = meta.get("lon")
        if isinstance(payload_lat, (int, float)) and isinstance(payload_lon, (int, float)):
            if _in_georgia(payload_lat, payload_lon):
                enriched["lat"] = float(payload_lat)
                enriched["lon"] = float(payload_lon)
                enriched["coord_source"] = "qdrant_payload"
                return enriched
            else:
                logger.warning(
                    f"[{request_id}] payload coords for '{name}' "
                    f"({payload_lat:.3f},{payload_lon:.3f}) outside Georgia — falling back to ORS"
                )

        # Fallback: ORS geocoding by name + city hint.
        # Prefer settlement > municipality from Qdrant payload (more specific than location field).
        # Payload values may be lists (e.g. ['Sagarejo']) — take first element.
        def _first(val):
            if isinstance(val, list):
                return val[0] if val else ""
            return val or ""

        city = (
            _first(meta.get("settlement"))
            or _first(meta.get("municipality"))
            or _city_from_location(loc)
        )
        query = f"{name}, {city}" if (name and city) else (name if name else loc)
        try:
            async with sem:
                result = await cached_geocode_city(query, focus_lat=focus_lat, focus_lon=focus_lon)
            if "lat" in result and "lon" in result:
                lat, lon = result["lat"], result["lon"]
                if not _in_georgia(lat, lon):
                    logger.warning(
                        f"[{request_id}] geocode '{query}' → ({lat:.3f},{lon:.3f}) "
                        f"outside Georgia — discarding"
                    )
                elif coord_envelope and not _in_envelope(lat, lon, coord_envelope):
                    logger.warning(
                        f"[{request_id}] geocode '{query}' → ({lat:.3f},{lon:.3f}) "
                        f"outside expected region envelope — discarding"
                    )
                else:
                    enriched["lat"] = lat
                    enriched["lon"] = lon
                    enriched["coord_source"] = "ors"
            return enriched
        except Exception as e:
            logger.warning(f"[{request_id}] geocode failed for '{query}': {e}")
            return enriched

    enriched_places = list(await asyncio.gather(
        *[geocode_place(p) for p in search_results]
    ))

    geocoded = [p for p in enriched_places if p.get("lat")]
    qdrant_coords = sum(1 for p in enriched_places if p.get("coord_source") == "qdrant_payload")
    ors_coords = sum(1 for p in enriched_places if p.get("coord_source") == "ors")
    logger.info(
        f"[{request_id}] geo_enrich_places: {len(geocoded)}/{len(enriched_places)} geocoded "
        f"(qdrant: {qdrant_coords}, ors: {ors_coords})"
    )

    # Step 2: build route pairs for chain routes and first-place fan-out.
    distance_matrix = {}

    async def get_distance(a: dict, b: dict) -> tuple:
        a_name, b_name = a["name"], b["name"]
        try:
            async with sem:
                result = await cached_get_route(a["lon"], a["lat"], b["lon"], b["lat"])
            if "distance_km" in result:
                val = {"km": result["distance_km"], "hours": result["duration_min"] / 60}
                return a_name, b_name, val
        except Exception as e:
            logger.warning(f"[{request_id}] route failed for '{route_key(a_name, b_name)}': {e}")
        return a_name, b_name, {"km": 0, "hours": 0}

    # Chain: 0→1, 1→2, ..., n-1→n.
    route_pairs = []
    for i in range(len(geocoded) - 1):
        route_pairs.append((geocoded[i], geocoded[i + 1]))
    # First place to each other place, capped for latency.
    if len(geocoded) > 2:
        for i in range(1, min(len(geocoded), 8)):
            route_pairs.append((geocoded[0], geocoded[i]))

    if route_pairs:
        route_results = await asyncio.gather(
            *[get_distance(a, b) for a, b in route_pairs]
        )
        for a_name, b_name, val in route_results:
            if val["km"] > 0:
                distance_matrix[route_key(a_name, b_name)] = val
                reverse_key = route_key(b_name, a_name)
                if reverse_key not in distance_matrix:
                    distance_matrix[reverse_key] = val

    logger.info(f"[{request_id}] geo_enrich_places: {len(distance_matrix)} routes built")

    return {
        "enriched_places": enriched_places,
        "distance_matrix": distance_matrix,
        "geocoded_locations": {
            p["name"][:30]: {"lat": p.get("lat"), "lon": p.get("lon")}
            for p in geocoded
        },
        "agent_history": ["geo_agent"],
        "agent_scratchpad": make_scratchpad(
            "geo_agent",
            f"Geocoded {len(geocoded)}/{len(enriched_places)} places, "
            f"{len(distance_matrix)} routes (no LLM)",
        ),
    }
