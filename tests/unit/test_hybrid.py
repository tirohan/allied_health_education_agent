from backend.app.agents.state import RetrievalCollection, RetrievedDoc
from backend.app.retrieval.hybrid import keyword_tsquery, reciprocal_rank_fusion, teaching_rerank


def test_keyword_tsquery_prefers_salient_terms() -> None:
    query = "What interprofessional education resources address opioid education in rural Georgia counties?"
    assert "opioid" in keyword_tsquery(query)
    assert " OR " in keyword_tsquery(query)


def _doc(
    source_id: str,
    *,
    title: str | None = None,
    text: str = "Example",
    vector_rank: int | None = None,
    sql_rank: int | None = None,
    score: float = 1.0,
) -> RetrievedDoc:
    return RetrievedDoc(
        id=source_id,
        collection=RetrievalCollection.PAPERS,
        source_table="research_papers",
        source_id=source_id,
        title=title or f"Paper {source_id}",
        text=text,
        score=score,
        vector_rank=vector_rank,
        sql_rank=sql_rank,
    )


def test_reciprocal_rank_fusion_merges_duplicate_sources() -> None:
    results = reciprocal_rank_fusion(
        vector_results=[_doc("a", vector_rank=1), _doc("b", vector_rank=2)],
        sql_results=[_doc("a", sql_rank=1), _doc("c", sql_rank=2)],
        top_k=10,
    )
    # Keyword-heavy fusion prefers consensus + SQL evidence.
    assert results[0].source_id == "a"
    assert {doc.source_id for doc in results} == {"a", "b", "c"}
    assert len(results) == 3


def test_teaching_rerank_boosts_teaching_terms_and_penalizes_generic_titles() -> None:
    docs = [
        _doc("weak", title="Social Work.", text="General program overview", score=0.9),
        _doc(
            "strong",
            title="Opioid IPE simulation for rural Georgia",
            text="Interprofessional opioid education",
            score=0.5,
        ),
    ]
    ranked = teaching_rerank(
        "opioid interprofessional education rural Georgia",
        docs,
        top_k=2,
        vector_keys={("research_papers", "weak"), ("research_papers", "strong")},
        sql_keys={("research_papers", "strong")},
    )
    assert ranked[0].source_id == "strong"
