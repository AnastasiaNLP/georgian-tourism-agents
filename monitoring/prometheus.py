from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "travel_agent_requests_total",
    "Total number of itinerary requests",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "travel_agent_request_latency_ms",
    "Request latency in milliseconds",
)

LLM_TOKENS_USED = Counter(
    "travel_agent_llm_tokens_total",
    "Total LLM tokens used",
    ["agent"],
)

DEGRADATION_FLAGS = Counter(
    "travel_agent_degradation_flags_total",
    "Number of degradation flags raised",
    ["flag"],
)
