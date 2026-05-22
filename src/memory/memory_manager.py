"""
Memory manager for user profile and episodic trip memory.

Budget-aware write policy:
    NORMAL   → save profile + save episode
    DEGRADED -> save profile only and skip Qdrant episode writes.

Bounded caps:
    MAX_INTERESTS = 20
    MAX_VISITED   = 50
    MAX_TRIPS     = 10
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from config.settings import get_settings

logger = logging.getLogger(__name__)

PROFILE_TTL = 60 * 60 * 24 * 30
EPISODE_COLLECTION = "user_memory_episodes_v1"
EPISODE_VECTOR_SIZE = get_settings().vector_size

MAX_INTERESTS = 20
MAX_VISITED   = 50
MAX_TRIPS     = 10


def empty_profile(user_id: str) -> Dict[str, Any]:
    return {
        "user_id":        user_id,
        "preferred_pace": None,         # relaxed / moderate / intensive
        "preferred_language": None,     # ru / en / ka
        "interests":      [],           # ["history", "nature", "food", ...]
        "visited_places": [],           # ["Tbilisi", "Batumi", ...]
        "past_trips":     [],
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "updated_at":     datetime.now(timezone.utc).isoformat(),
    }


def merge_profile(
    old: Dict[str, Any],
    new_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge profile updates with bounded caps and no duplicates.

    Args:
        old: Existing profile.
        new_data: New data to merge.

    Returns:
        Updated profile.
    """
    merged = {**old}

    # Pace overwrites when present.
    if new_data.get("preferred_pace"):
        merged["preferred_pace"] = new_data["preferred_pace"]

    # Language
    if new_data.get("preferred_language"):
        merged["preferred_language"] = new_data["preferred_language"]

    # Interests — union + cap
    new_interests = new_data.get("interests", [])
    if new_interests:
        combined = list(set(old.get("interests", []) + new_interests))
        merged["interests"] = combined[:MAX_INTERESTS]

    # Visited places — union + cap
    new_places = new_data.get("visited_places", [])
    if new_places:
        combined = list(set(old.get("visited_places", []) + new_places))
        merged["visited_places"] = combined[:MAX_VISITED]

    # Past trips — append + keep last MAX_TRIPS
    new_trip = new_data.get("new_trip")
    if new_trip:
        old_trips = old.get("past_trips", [])
        merged["past_trips"] = (old_trips + [new_trip])[-MAX_TRIPS:]

    merged["updated_at"] = datetime.now(timezone.utc).isoformat()
    return merged


class MemoryManager:
    """
    Memory Manager — Profile (Redis) + Episodic (Qdrant).

    Usage:
        mm = MemoryManager()

        # Load
        profile = mm.load_profile(user_id)

        # Save
        mm.save_profile(user_id, new_data)
        await mm.save_episode(user_id, episode)

        # Retrieve episodes (only for PLAN + NORMAL)
        episodes = await mm.retrieve_episodes(user_id, query, top_k=3)
    """

    def __init__(self):
        from src.tools.tool_cache import get_cache
        self._cache = get_cache()
        self._qdrant = None  # lazy init

    def _profile_key(self, user_id: str) -> str:
        return f"profile:{user_id}"

    # ========================================================================
    # 6A — Profile Memory (Redis)
    # ========================================================================

    def load_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Load user profile from Redis.

        Metrics:
            memory_hit: bool
            load_latency_ms: float
        """
        t0 = time.time()
        key = self._profile_key(user_id)
        profile = self._cache.get(key)
        latency_ms = (time.time() - t0) * 1000

        if profile:
            logger.info(
                f"[memory] Profile HIT user={user_id} "
                f"latency={latency_ms:.1f}ms"
            )
            return profile
        else:
            logger.info(
                f"[memory] Profile MISS user={user_id} "
                f"latency={latency_ms:.1f}ms → creating empty"
            )
            return empty_profile(user_id)

    def save_profile(
        self,
        user_id: str,
        new_data: Dict[str, Any]
    ) -> bool:
        """
        Save or update profile in Redis.

        Metrics:
            write_success_rate: bool
        """
        key = self._profile_key(user_id)

        # Load previous profile.
        old = self._cache.get(key) or empty_profile(user_id)

        # Merge with caps.
        updated = merge_profile(old, new_data)

        # Save merged profile.
        success = self._cache.set(key, updated, ttl_seconds=PROFILE_TTL)

        logger.info(
            f"[memory] Profile SAVE user={user_id} "
            f"success={success} "
            f"interests={len(updated['interests'])} "
            f"visited={len(updated['visited_places'])}"
        )
        return success

    # ========================================================================
    # 6B — Episodic Memory (Qdrant)
    # ========================================================================

    def _get_qdrant(self):
        """Lazy init Qdrant client."""
        if self._qdrant is None:
            from qdrant_client import QdrantClient
            settings = get_settings()
            self._qdrant = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        return self._qdrant

    def _ensure_episode_collection(self):
        """Create collection and indexes if missing."""
        from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
        client = self._get_qdrant()
        collections = [c.name for c in client.get_collections().collections]
        if EPISODE_COLLECTION not in collections:
            client.create_collection(
                collection_name=EPISODE_COLLECTION,
                vectors_config=VectorParams(
                    size=EPISODE_VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            # Required for user_id filtering in Qdrant Cloud.
            client.create_payload_index(
                collection_name=EPISODE_COLLECTION,
                field_name="user_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info(f"[memory] Created collection + index: {EPISODE_COLLECTION}")

    def _embed(self, text: str) -> List[float]:
        """Create an embedding with the shared singleton model."""
        from src.tools.embeddings import embed
        return embed(text)

    async def save_episode(
        self,
        user_id: str,
        episode: Dict[str, Any]
    ) -> bool:
        """
        Save trip episode to Qdrant.

        Expected keys: task, plan_summary, places, result, eval_score.
        """
        try:
            self._ensure_episode_collection()

            # Embed task and summary.
            embed_text = f"{episode.get('task', '')} {episode.get('plan_summary', '')}"
            vector = self._embed(embed_text)

            # Payload
            payload = {
                "user_id":      user_id,
                "task":         episode.get("task", ""),
                "plan_summary": episode.get("plan_summary", ""),
                "places":       episode.get("places", []),
                "result":       episode.get("result", "unknown"),
                "eval_score":   episode.get("eval_score"),
                "created_at":   datetime.now(timezone.utc).isoformat(),
            }

            import uuid
            point_id = str(uuid.uuid4()).replace("-", "")[:16]
            # Qdrant point IDs must be int or UUID.
            import hashlib
            point_id_int = int(hashlib.md5(
                f"{user_id}:{embed_text}:{payload['created_at']}".encode()
            ).hexdigest()[:8], 16)

            from qdrant_client.models import PointStruct
            self._get_qdrant().upsert(
                collection_name=EPISODE_COLLECTION,
                points=[PointStruct(
                    id=point_id_int,
                    vector=vector,
                    payload=payload
                )]
            )

            logger.info(
                f"[memory] Episode SAVED user={user_id} "
                f"places={len(payload['places'])} "
                f"result={payload['result']}"
            )
            return True

        except Exception as e:
            logger.error(f"[memory] Episode SAVE failed: {e}")
            return False

    async def retrieve_episodes(
        self,
        user_id: str,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant episodes from Qdrant.

        Called only when user_id is present and memory retrieval is enabled.

        Metrics:
            episodes_used: int
        """
        try:
            self._ensure_episode_collection()

            vector = self._embed(query)

            from qdrant_client.models import Filter, FieldCondition, MatchValue
            results = self._get_qdrant().query_points(
                collection_name=EPISODE_COLLECTION,
                query=vector,
                limit=top_k,
                query_filter=Filter(
                    must=[FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )]
                )
            ).points

            episodes = [r.payload for r in results]

            logger.info(
                f"[memory] Episodes RETRIEVED user={user_id} "
                f"query='{query[:40]}' count={len(episodes)}"
            )
            return episodes

        except Exception as e:
            logger.warning(f"[memory] Episode RETRIEVE failed: {e}")
            return []

    # ========================================================================
    # Convenience: extract profile data from state
    # ========================================================================

    def extract_profile_update(self, state) -> Dict[str, Any]:
        """
        Extract profile update data from final state.
        """
        # Works with dict and Pydantic-like objects.
        def get(key, default=None):
            if isinstance(state, dict):
                return state.get(key, default)
            return getattr(state, key, default)

        update = {}

        # Language
        lang = get("user_language")
        if lang:
            update["preferred_language"] = lang

        # Pace from trip_parameters
        trip = get("trip_parameters") or {}
        if isinstance(trip, dict):
            pace = trip.get("pace")
            if pace:
                update["preferred_pace"] = pace

        # Visited places from itinerary
        visited = []
        itinerary = get("enriched_itinerary") or get("raw_itinerary")
        if itinerary and isinstance(itinerary, dict):
            for day in itinerary.get("days", []):
                for activity in day.get("activities", []):
                    loc = activity.get("location") or activity.get("name")
                    if loc:
                        visited.append(loc)
        if visited:
            update["visited_places"] = list(set(visited))

        # Short trip entry
        if itinerary:
            days_count = len(itinerary.get("days", []))
            # execution_mode can be a string or enum-like object.
            mode_raw = get("execution_mode", "normal")
            mode_str = mode_raw.value if hasattr(mode_raw, "value") else str(mode_raw)
            update["new_trip"] = {
                "date":   datetime.now(timezone.utc).isoformat()[:10],
                "query":  (get("user_query") or "")[:100],
                "days":   days_count,
                "intent": get("intent"),
                "mode":   mode_str,
            }

        return update

    def extract_episode(self, state) -> Dict[str, Any]:
        """
        Create an episode from final state for Qdrant.
        """
        def get(key, default=None):
            if isinstance(state, dict):
                return state.get(key, default)
            return getattr(state, key, default)

        places = []
        itinerary = get("enriched_itinerary") or get("raw_itinerary")
        if itinerary and isinstance(itinerary, dict):
            for day in itinerary.get("days", []):
                for activity in day.get("activities", []):
                    name = activity.get("name") or activity.get("location")
                    if name:
                        places.append(name)

        plan_summary = ""
        if itinerary and isinstance(itinerary, dict):
            days = itinerary.get("days", [])
            plan_summary = f"{len(days)}-day trip"
            trip = get("trip_parameters") or {}
            if isinstance(trip, dict):
                start = trip.get("start_location", "")
                if start:
                    plan_summary += f" from {start}"

        errors = get("errors") or []
        eval_score = get("eval_score")
        score_val = None
        if isinstance(eval_score, dict):
            score_val = eval_score.get("total")

        return {
            "task":         get("user_query", ""),
            "plan_summary": plan_summary,
            "places":       list(set(places))[:20],
            "result":       "success" if not errors else "degraded",
            "eval_score":   score_val,
        }


# ============================================================================
# Singleton
# ============================================================================

_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
