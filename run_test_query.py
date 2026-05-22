#!/usr/bin/env python3
"""
Запуск реального запроса через граф с подробными логами.
Использование: python run_test_query.py
"""

import asyncio
import json
import logging
import sys

# Подробные логи: INFO для наших модулей, WARNING для сторонних
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
for noisy in ("httpx", "httpcore", "anthropic", "openai", "langsmith",
              "urllib3", "asyncio", "langchain", "langgraph"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


async def main():
    query = "Кутаиси, Кахетия и Сванетия — 2 недели, природа и культура"
    request_id = "manual-test-005"

    print("\n" + "="*60)
    print(f"ЗАПРОС: {query!r}")
    print(f"REQUEST_ID: {request_id}")
    print("="*60 + "\n")

    from graph.main_graph import run_graph
    # user_language не передаём — должен определиться автоматически
    final_state = await run_graph(
        user_query=query,
        request_id=request_id,
    )

    # ── Итоги ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТ")
    print("="*60)

    history = final_state.get("agent_history") or []
    print(f"\n▶ Порядок агентов: {' → '.join(history)}")

    search = final_state.get("search_results") or []
    print(f"\n▶ search_results: {len(search)} мест")
    for p in search[:5]:
        print(f"   • {p.get('name')} | {p.get('category')} | score={p.get('score', '?'):.2f}")

    enriched = final_state.get("enriched_places") or []
    geocoded = [p for p in enriched if p.get("lat")]
    print(f"\n▶ enriched_places: {len(enriched)} мест, геокодировано: {len(geocoded)}")

    dm = final_state.get("distance_matrix") or {}
    print(f"\n▶ distance_matrix: {len(dm)} маршрутов")
    for k, v in list(dm.items())[:5]:
        print(f"   • {k}: {v.get('km', 0):.0f} km / {v.get('hours', 0):.1f}h")

    itinerary = final_state.get("raw_itinerary") or {}
    days = itinerary.get("days") or []
    print(f"\n▶ raw_itinerary: {len(days)} дней")
    for d in days:
        acts = d.get("activities") or []
        km = d.get("total_distance_km", 0)
        hrs = d.get("total_driving_hours", 0)
        print(f"   День {d.get('day')}: {len(acts)} активностей, {km} km, {hrs}h вождения")
        for a in acts:
            coord = a.get("coordinates") or {}
            lat = coord.get("lat", "?")
            lon = coord.get("lon", "?")
            print(f"      - {a.get('name')} ({a.get('category')}) lat={lat} lon={lon}")

    val = final_state.get("validation_result") or {}
    print(f"\n▶ validation_result:")
    print(f"   is_valid={val.get('is_valid')} score={val.get('score')} action={val.get('recommended_action')}")
    if val.get("errors"):
        print(f"   errors: {val['errors']}")
    if val.get("warnings"):
        print(f"   warnings: {val['warnings']}")

    budget = final_state.get("budget_state") or {}
    print(f"\n▶ budget_state:")
    print(f"   llm_calls={budget.get('llm_calls', 0)}")
    print(f"   prompt_tokens={budget.get('prompt_tokens', 0)}")
    print(f"   completion_tokens={budget.get('completion_tokens', 0)}")
    print(f"   estimated_cost_usd=${budget.get('estimated_cost_usd', 0):.5f}")

    errors = final_state.get("errors") or []
    if errors:
        print(f"\n▶ errors: {errors}")

    mode = final_state.get("execution_mode", "normal")
    print(f"\n▶ execution_mode: {mode}")

    router_trace = final_state.get("router_trace") or []
    if router_trace:
        print(f"\n▶ router_trace:")
        for t in router_trace:
            print(f"   {t.get('from')} → {t.get('next')} | {t.get('reason', '')[:60]}")

    final_resp = final_state.get("final_response")
    if final_resp:
        print(f"\n▶ final_response (первые 300 символов):")
        print(f"   {final_resp[:300]}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
