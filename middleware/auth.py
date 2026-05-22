"""Simple API key authentication."""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from config.settings import get_settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> None:
    expected = get_settings().api_key
    if not expected:
        return
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
