from backend.app.agents.state import RetrievalCollection, RetrievedDoc
from backend.app.retrieval.hybrid import keyword_tsquery, reciprocal_rank_fusion


def test_keyword_tsquery_prefers_salient_terms() -> None:
    query = "What interprofessional education resources address opioid education in rural Georgia counties?"
    assert "opioid" in keyword_tsquery(query)
    assert " OR " in keyword_tsquery(query)


def _doc(source_id: str, vector_rank: int | None = None, sql_rank: int | None = None) -> RetrievedDoc:
    return RetrievedDoc(
        id=source_id,
        collection=RetrievalCollection.PAPERS,
        source_table="research_papers",
        source_id=source_id,
        title=f"Paper {source_id}",
        text="Example",
        score=1.0,
        vector_rank=vector_rank,
        sql_rank=sql_rank,
    )


def test_reciprocal_rank_fusion_merges_duplicate_sources() -> None:
    results = reciprocal_rank_fusion(
        vector_results=[_doc("a", vector_rank=1), _doc("b", vector_rank=2)],
        sql_results=[_doc("a", sql_rank=1), _doc("c", sql_rank=2)],
        top_k=10,
    )
    assert [doc.source_id for doc in results][:2] == ["a", "b"]
    assert len(results) == 3
