import os
from typing import Any

import httpx


API_URL = os.environ.get("API_URL", "http://localhost:8000")


def post(path: str, payload: dict[str, Any], timeout: float = 120) -> dict[str, Any]:
    with httpx.Client(base_url=API_URL, timeout=timeout) as client:
        response = client.post(path, json=payload)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        return {"raw": response.text, "content_type": content_type}


def post_bytes(path: str, payload: dict[str, Any], timeout: float = 120) -> tuple[bytes, str]:
    with httpx.Client(base_url=API_URL, timeout=timeout) as client:
        response = client.post(path, json=payload)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "application/octet-stream")
        return response.content, content_type


def get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    with httpx.Client(base_url=API_URL, timeout=30) as client:
        response = client.get(path, params=params)
        response.raise_for_status()
        return response.json()
