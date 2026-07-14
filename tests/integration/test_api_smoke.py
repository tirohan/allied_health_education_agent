import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import create_app


@pytest.mark.asyncio
async def test_health_endpoint_smoke() -> None:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Lifespan is not auto-entered here; exercise route wiring only through override.
        app.state.services = type(
            "Services",
            (),
            {
                "settings": type("Settings", (), {"app_env": "test"})(),
            },
        )()
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_extraction_handles_all_collections() -> None:
    from backend.app.agents.extraction import _deterministic_extract
    from backend.app.agents.state import RetrievalCollection, RetrievedDoc

    state = {
        "query": "opioid education in rural Georgia",
        "refined_query": "opioid education in rural Georgia",
        "retrieved_docs": [
            RetrievedDoc(
                id="1",
                collection=RetrievalCollection.PAPERS,
                source_table="research_papers",
                source_id="p1",
                title="Opioid paper",
                text="opioid training",
                score=0.8,
            ),
            RetrievedDoc(
                id="2",
                collection=RetrievalCollection.RESOURCES,
                source_table="resources",
                source_id="r1",
                title="Opioid resource",
                text="IPE opioid module",
                score=0.7,
            ),
            RetrievedDoc(
                id="3",
                collection=RetrievalCollection.COMMUNITIES,
                source_table="county_profiles",
                source_id="13121",
                title="Fulton, GA",
                text="priority county",
                score=0.6,
            ),
        ],
    }
    entities, relations = _deterministic_extract(state)  # type: ignore[arg-type]
    assert any(entity.entity_type == "Topic" for entity in entities)
    assert any(entity.entity_type == "Paper" for entity in entities)
    assert any(entity.entity_type == "Resource" for entity in entities)
    assert any(entity.entity_type == "County" for entity in entities)
    assert len(relations) >= 3
