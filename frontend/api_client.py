import os
from typing import Any, Callable, TypeVar

import httpx
import streamlit as st
from dotenv import load_dotenv

# Local/dev entrypoints run `streamlit run frontend/app.py` from the project root
# without exporting env vars by hand; load the same .env the backend reads so
# API_URL/API_KEY stay in sync between the two processes. Real env vars (e.g. the
# ones docker-compose sets for the frontend service) still take precedence.
load_dotenv(override=False)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY")

T = TypeVar("T")


class ApiError(Exception):
    """Raised when a backend API call fails in a way the UI should show cleanly."""


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _error_message(exc: httpx.HTTPStatusError) -> str:
    detail = None
    try:
        body = exc.response.json()
        if isinstance(body, dict):
            detail = body.get("detail")
    except ValueError:
        detail = None
    if detail:
        return str(detail)
    reason = exc.response.reason_phrase or ""
    return f"Server returned {exc.response.status_code} {reason}".strip()


def post(path: str, payload: dict[str, Any], timeout: float = 120) -> dict[str, Any]:
    try:
        with httpx.Client(base_url=API_URL, timeout=timeout, headers=_headers()) as client:
            response = client.post(path, json=payload)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return {"raw": response.text, "content_type": content_type}
    except httpx.HTTPStatusError as exc:
        raise ApiError(_error_message(exc)) from exc
    except httpx.RequestError as exc:
        raise ApiError(
            f"Could not reach the server ({exc.__class__.__name__})."
        ) from exc


def post_bytes(path: str, payload: dict[str, Any], timeout: float = 120) -> tuple[bytes, str]:
    try:
        with httpx.Client(base_url=API_URL, timeout=timeout, headers=_headers()) as client:
            response = client.post(path, json=payload)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "application/octet-stream")
            return response.content, content_type
    except httpx.HTTPStatusError as exc:
        raise ApiError(_error_message(exc)) from exc
    except httpx.RequestError as exc:
        raise ApiError(
            f"Could not reach the server ({exc.__class__.__name__})."
        ) from exc


def get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        with httpx.Client(base_url=API_URL, timeout=30, headers=_headers()) as client:
            response = client.get(path, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise ApiError(_error_message(exc)) from exc
    except httpx.RequestError as exc:
        raise ApiError(
            f"Could not reach the server ({exc.__class__.__name__})."
        ) from exc


def safe_call(spinner_text: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run an API call under a spinner; on ApiError show a friendly message and stop the page."""
    with st.spinner(spinner_text):
        try:
            return fn(*args, **kwargs)
        except ApiError as exc:
            st.error(f"Couldn't reach the server: {exc}")
            st.stop()
    raise AssertionError("unreachable")  # st.stop() always halts execution
