-- request / response audit log (optional but powerful)

CREATE TABLE IF NOT EXISTS itinerary_requests (
    request_id        TEXT PRIMARY KEY,
    correlation_id    TEXT,
    created_at        TIMESTAMPTZ DEFAULT now(),
    language          TEXT,
    days              INTEGER,
    start_location    TEXT,
    degraded          BOOLEAN,
    overall_confidence FLOAT
);

CREATE TABLE IF NOT EXISTS degradation_events (
    id          SERIAL PRIMARY KEY,
    request_id TEXT REFERENCES itinerary_requests(request_id),
    flag        TEXT NOT NULL,
    source      TEXT NOT NULL,
    message     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

