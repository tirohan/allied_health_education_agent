from backend.app.schemas.api import MindMapRequest, SearchRequest


def test_search_request_defaults_to_dense_only() -> None:
    request = SearchRequest(query="opioid education")
    assert request.mode == "vector"


def test_mindmap_request_defaults_to_dense_only() -> None:
    request = MindMapRequest(query="opioid education")
    assert request.retrieval_mode == "vector"
