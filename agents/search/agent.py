"""
SearchAgent performs direct Qdrant search.

Node wrapper: search_agent_node(state: dict) → dict
"""
from __future__ import annotations
import logging
from typing import Optional

from agents.base import extract_search_params, make_scratchpad

logger = logging.getLogger(__name__)


async def _do_search(search_query: str, region: Optional[str], max_results: int) -> list:
    """Search Qdrant. search_query is expected to be in English."""
    from src.tools.tool_cache import cached_search_qdrant
    from agents.geo_filter import filter_by_region, load_regions

    # Canonical regions use native Qdrant filter on structured payload field.
    # Non-canonical values (Abkhazia, South Ossetia, typos, etc.) fall through
    # to legacy post-filter on free-text location field.
    canonical_regions = set(load_regions().keys())

    if region and region in canonical_regions:
        all_results = await cached_search_qdrant(
            search_query,
            top_k=max_results,
            filters={"region": region},
        )
        return all_results

    # Fallback: no region or non-canonical — over-fetch then post-filter.
    fetch_k = max_results * 3 if region else max_results
    all_results = await cached_search_qdrant(search_query, top_k=fetch_k)

    if region and all_results:
        filtered = filter_by_region(all_results, region, min_results=3)
        return filtered[:max_results]

    return all_results[:max_results]


def _primary_category(cat: str) -> str:
    """Extract the primary category label (first comma/slash-separated token)."""
    if not cat:
        return ""
    import re
    return re.split(r'[,/]', cat)[0].strip().lower()


def _is_en_name(name: str) -> bool:
    """Return True when the name uses Latin characters."""
    import re
    return not bool(re.search(r'[а-яёА-ЯЁ\u10D0-\u10FF]', name or ""))


def _deduplicate_results(results: list) -> list:
    """
    Deduplicate search results representing the same place.

    Uses exact names, exact locations, and very close vector scores within the
    same primary category. When duplicates exist, prefer Latin names because
    geocoding is usually more reliable.
    """
    if not results:
        return results

    kept = []
    used = set()

    for i, r in enumerate(results):
        if i in used:
            continue

        best = r
        best_idx = i

        for j, candidate in enumerate(results):
            if j <= i or j in used:
                continue

            same_name = bool(
                r.get("name") and r.get("name") == candidate.get("name")
            )
            same_location = bool(
                r.get("location") and r.get("location") == candidate.get("location")
            )
            r_cat = _primary_category(r.get("category", ""))
            c_cat = _primary_category(candidate.get("category", ""))
            same_primary_cat = bool(r_cat and r_cat == c_cat)
            # Tight threshold: only nearly identical vectors.
            score_very_close = abs(r.get("score", 0) - candidate.get("score", 0)) <= 0.03

            is_duplicate = (
                same_name
                or (same_primary_cat and score_very_close)
                or same_location
            )

            if is_duplicate:
                used.add(j)
                # Prefer EN name (geocodes reliably); fallback to longer description
                best_is_en = _is_en_name(best.get("name", ""))
                cand_is_en = _is_en_name(candidate.get("name", ""))
                if cand_is_en and not best_is_en:
                    best = candidate
                    best_idx = j
                elif not cand_is_en and best_is_en:
                    pass  # keep current best (EN)
                elif len(candidate.get("description", "")) > len(best.get("description", "")):
                    best = candidate
                    best_idx = j

        used.add(i)
        used.add(best_idx)
        kept.append(best)

    return kept


async def search_agent_node(state: dict) -> dict:
    """Run direct Qdrant search without an LLM call."""
    request_id = state.get("request_id", "unknown")
    params = extract_search_params(state)
    region = params.get("region", "Georgia")
    max_results = params.get("max_results", 10)
    user_query = state.get("user_query", "")
    user_language = state.get("user_language", "en")

    # The orchestrator should provide an English search query.
    # If it does not, fall back to the raw user query because embeddings are
    # partially multilingual.
    search_query = params.get("search_query") or user_query

    raw_results = await _do_search(search_query, region, max_results)
    results = _deduplicate_results(raw_results)

    logger.info(
        f"[{request_id}] search: {len(raw_results)} raw → "
        f"{len(results)} deduped | lang={user_language}"
    )

    return {
        "search_results": results,
        "search_context": {
            "region": region,
            "result_count": len(results),
            "raw_count": len(raw_results),
            "query": user_query,
        },
        "intent": state.get("intent") or "SEARCH",
        "user_language": user_language,
        "agent_history": ["search_agent"],
        "agent_scratchpad": make_scratchpad(
            "search_agent",
            f"Found {len(results)} unique places in {region}, user_lang={user_language}",
        ),
    }
