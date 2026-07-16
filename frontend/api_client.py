import json
import os
from collections.abc import Iterator
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


def _error_message_from_response(response: httpx.Response) -> str:
    detail = None
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = body.get("detail")
    except ValueError:
        detail = None
    if detail:
        return str(detail)
    reason = response.reason_phrase or ""
    return f"Server returned {response.status_code} {reason}".strip()


def _error_message(exc: httpx.HTTPStatusError) -> str:
    return _error_message_from_response(exc.response)


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


def stream_post(
    path: str, payload: dict[str, Any], timeout: float = 180
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (event, data) tuples from a Server-Sent Events endpoint.

    Blank lines separate SSE messages; `:`-prefixed lines are keep-alive
    comments and are skipped, per the SSE wire format sse-starlette emits.
    """
    try:
        with httpx.Client(base_url=API_URL, timeout=timeout, headers=_headers()) as client:
            with client.stream("POST", path, json=payload) as response:
                if response.status_code >= 400:
                    response.read()
                    raise ApiError(_error_message_from_response(response))

                event_name: str | None = None
                data_lines: list[str] = []
                for line in response.iter_lines():
                    if line == "":
                        if event_name is not None or data_lines:
                            data = "\n".join(data_lines)
                            try:
                                parsed = json.loads(data) if data else {}
                            except json.JSONDecodeError:
                                parsed = {"raw": data}
                            yield (event_name or "message", parsed)
                        event_name = None
                        data_lines = []
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:") :].strip())
    except httpx.RequestError as exc:
        raise ApiError(f"Could not reach the server ({exc.__class__.__name__}).") from exc


def build_mindmap_with_progress(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a teaching map while showing real per-agent progress instead of a
    single opaque spinner. Raises ApiError on failure."""
    with st.status("Building your teaching map...", expanded=True) as status:
        result: dict[str, Any] | None = None
        try:
            for event_name, data in stream_post("/api/v1/mindmap/stream", payload):
                if event_name == "step":
                    label = data.get("label") or data.get("agent", "").title()
                    st.write(f"**{label}** — {data.get('message', '')}")
                elif event_name == "error":
                    status.update(label="Something went wrong", state="error")
                    raise ApiError(data.get("detail") or "Mind map generation failed.")
                elif event_name == "result":
                    result = data
        except ApiError:
            status.update(label="Something went wrong", state="error")
            raise
        if result is None:
            status.update(label="Something went wrong", state="error")
            raise ApiError("The server closed the connection before the map finished.")
        status.update(label="Teaching map ready", state="complete", expanded=False)
        return result
