"""
SearchAgent performs direct Qdrant search.

Node wrapper: search_agent_node(state: dict) → dict
"""
from __future__ import annotations
import logging
from typing import Optional

from agents.base import extract_search_params, make_scratchpad

logger = logging.getLogger(__name__)


def _do_search(search_query: str, region: Optional[str], max_results: int) -> list:
    """Search Qdrant. search_query is expected to be in English."""
    from src.tools.tool_cache import cached_search_qdrant
    from agents.geo_filter import filter_by_region, load_regions

    load_regions.cache_clear()

    # Canonical regions use native Qdrant filter on structured payload field.
    # Non-canonical values (Abkhazia, South Ossetia, typos, etc.) fall through
    # to legacy post-filter on free-text location field.
    canonical_regions = set(load_regions().keys())

    if region and region in canonical_regions:
        all_results = cached_search_qdrant(
            search_query,
            top_k=max_results,
            filters={"region": region},
        )
        return all_results

    # Fallback: no region or non-canonical — over-fetch then post-filter.
    fetch_k = max_results * 3 if region else max_results
    all_results = cached_search_qdrant(search_query, top_k=fetch_k)

    if region and all_results:
        filtered = filter_by_region(all_results, region, min_results=3)
        return filtered[:max_results]

    return all_results[:max_results]


def _translate_query(query: str) -> str:  # DEPRECATED — kept only as fallback
    """Kept as last-resort fallback if orchestrator didn't provide search_query."""
    replacements = {
        # Nature and landscape
        "природа": "nature", "природные": "nature", "природный": "nature",
        "горы": "mountains", "гора": "mountain", "горный": "mountain",
        "море": "sea", "морской": "sea", "пляж": "beach",
        "водопад": "waterfall", "водопады": "waterfalls",
        "река": "river", "озеро": "lake", "ущелье": "gorge", "каньон": "canyon",
        "лес": "forest", "пещера": "cave", "ледник": "glacier",
        "заповедник": "nature reserve", "национальный парк": "national park",
        "парк": "park", "хребет": "ridge",
        # Activities
        "треккинг": "trekking", "трекинг": "trekking",
        "поход": "hiking", "пешеходный": "hiking",
        "восхождение": "climbing", "велосипед": "cycling",
        "сплав": "rafting", "плавание": "swimming",
        # Culture and history
        "музей": "museum", "музеи": "museums", "галерея": "gallery", "галереи": "galleries",
        "церковь": "church", "церкви": "churches", "храм": "temple", "собор": "cathedral",
        "монастырь": "monastery", "монастыри": "monasteries",
        "крепость": "fortress", "крепости": "fortresses", "замок": "castle", "башня": "tower",
        "памятник": "monument", "памятники": "monuments", "руины": "ruins",
        "достопримечательности": "attractions sights",
        # Places and objects
        "места": "places", "место": "place",
        "объекты": "objects", "объект": "object",
        "площадь": "square", "площади": "squares",
        "бульвар": "boulevard", "бульвары": "boulevards",
        "набережная": "promenade", "набережные": "promenades",
        "рынок": "market", "рынки": "markets", "базар": "market",
        "ботанический сад": "botanical garden",
        "зоопарк": "zoo",
        "парки": "parks", "пляжи": "beaches",
        # Quality/type adjectives
        "современные": "modern", "современный": "modern",
        "лучшие": "best", "лучший": "best",
        "красивые": "beautiful", "красивый": "beautiful",
        "популярные": "popular", "популярный": "popular",
        "известные": "famous", "известный": "famous",
        "исторические": "historic", "исторический": "historic",
        "старинные": "ancient", "старинный": "ancient",
        "интересные": "interesting", "интересный": "interesting",
        "самые": "",
        "все": "",
        # Food and drinks
        "вино": "wine", "виноград": "vineyard", "винодельня": "winery",
        "ресторан": "restaurant", "кафе": "cafe",
        # Georgian cities
        "батуми": "batumi", "тбилиси": "tbilisi", "кутаиси": "kutaisi",
        "местиа": "mestia", "местия": "mestia", "телави": "telavi",
        "сигнахи": "sighnaghi", "гори": "gori", "мцхета": "mtskheta",
        "казбеги": "kazbegi", "боржоми": "borjomi", "вардзиа": "vardzia",
        "кобулети": "kobuleti", "гонио": "gonio",
        # Regions with common inflections and typos
        "аджария": "adjara", "аджарию": "adjara", "аджаре": "adjara", "аджара": "adjara",
        "кахетия": "kakheti", "кахетию": "kakheti", "кахетии": "kakheti",
        "сванетия": "svaneti", "сванетию": "svaneti", "сванетии": "svaneti",
        "свенетию": "svaneti", "свенети": "svaneti",
        "имерети": "imereti", "имеретию": "imereti",
        "самегрело": "samegrelo",
        "гурия": "guria", "гурии": "guria",
        "картли": "kartli",
        # Time and conversational noise
        " дней": "", " дня": "", " день": "",
        " days": "", " day": "",
        " завтра": "", " сегодня": "", " послезавтра": "",
        " привет": "", "привет ": "", "привет,": "",
        " прилетаю": "", " приеду": "", " приезжаю": "",
        " хочу": "", " хотим": "", " планирую": "",
        " посетить": "", " посмотреть": "", " увидеть": "",
        " есть": "", " буду": "", " будем": "",
        # Numerals
        " один": "", " одного": "", " одну": "",
        " два": "", " две": "", " двух": "",
        " три": "", " трёх": "", " трех": "",
        # Function words
        " и ": " ", " в ": " ", " на ": " ", " по ": " ",
        " с ": " ", " для ": " ", " из ": " ",
        " у ": " ", " о ": " ", " об ": " ",
        " я ": " ", " мы ": " ", " меня ": " ", " нас ": " ",
        " чтобы ": " ", " который ": " ", " которые ": " ", " которых ": " ",
        " это ": " ", " этот ": " ", " эта ": " ",
    }
    result = query.lower()
    for ru, en in replacements.items():
        result = result.replace(ru, en)
    # Drop unrecognized Cyrillic terms that add embedding noise.
    import re
    result = re.sub(r'[а-яёА-ЯЁ]+', ' ', result)
    # Normalize whitespace and punctuation.
    result = re.sub(r'[,.\s]+', ' ', result).strip()
    return result


# RU→EN category normalization — so deduplication catches RU/EN pairs
# where the primary category is the same concept in different languages.
_RU_CATEGORY_TO_EN = {
    "музей": "museum", "галерея": "gallery",
    "водопад": "waterfall", "водопады": "waterfall",
    "парк": "park", "национальный парк": "national park",
    "крепость": "fortress", "замок": "castle",
    "церковь": "church", "храм": "church",
    "монастырь": "monastery",
    "площадь": "square", "бульвар": "boulevard",
    "ботанический сад": "botanical garden",
    "зоопарк": "zoo",
    "пляж": "beach",
    "озеро": "lake",
    "пещера": "cave",
    "ущелье": "gorge",
    "ресторан": "restaurant",
    "кафе": "cafe",
    "гора": "mountain",
    # Compound categories
    "городская территория": "urban area",
    "городской парк": "urban park",
}


def _primary_category(cat: str) -> str:
    """
    Extract and normalize the primary category.

    Qdrant category can contain comma- or slash-separated labels.
    """
    if not cat:
        return ""
    import re
    first = re.split(r'[,/]', cat)[0].strip().lower()
    return _RU_CATEGORY_TO_EN.get(first, first)


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


def _filter_by_query_intent(results: list, query: str) -> list:
    """
    Re-rank results according to explicit nature/culture intent.

    Results are not removed; only their order changes.
    """
    query_lower = query.lower()

    nature_keywords = {"природа", "nature", "горы", "mountains", "водопад", "waterfall",
                       "парк", "park", "треккинг", "hiking", "пляж", "beach", "море", "sea"}
    culture_keywords = {"музей", "museum", "церковь", "church", "крепость", "fortress",
                        "культура", "culture", "история", "history"}

    wants_nature = any(kw in query_lower for kw in nature_keywords)
    wants_culture = any(kw in query_lower for kw in culture_keywords)

    if not wants_nature and not wants_culture:
        return results

    def relevance_score(r):
        tags = set(t.lower() for t in (r.get("tags") or []))
        cat = (r.get("category") or "").lower()
        name = (r.get("name") or "").lower()
        desc = (r.get("description") or "").lower()[:200]
        all_text = f"{cat} {' '.join(tags)} {name} {desc}"

        score = r.get("score", 0)

        if wants_nature:
            if any(kw in all_text for kw in nature_keywords):
                score += 0.3
            if any(kw in all_text for kw in ["музей", "museum", "gallery"]):
                score -= 0.2

        if wants_culture:
            if any(kw in all_text for kw in culture_keywords):
                score += 0.3
            if any(kw in all_text for kw in ["парк", "park", "beach"]):
                score -= 0.2

        return score

    return sorted(results, key=relevance_score, reverse=True)


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

    raw_results = _do_search(search_query, region, max_results)
    deduped = _deduplicate_results(raw_results)
    filtered = _filter_by_query_intent(deduped, search_query)

    logger.info(
        f"[{request_id}] search: {len(raw_results)} raw → "
        f"{len(deduped)} deduped → {len(filtered)} filtered | "
        f"lang={user_language}"
    )

    return {
        "search_results": filtered,
        "search_context": {
            "region": region,
            "result_count": len(filtered),
            "raw_count": len(raw_results),
            "query": user_query,
        },
        "intent": state.get("intent") or "SEARCH",
        "user_language": user_language,
        "agent_history": ["search_agent"],
        "agent_scratchpad": make_scratchpad(
            "search_agent",
            f"Found {len(filtered)} unique places in {region}, user_lang={user_language}",
        ),
    }
