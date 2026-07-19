from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from backend.app.schemas.api import (
    ChatRequest,
    ChatResponse,
    CurriculumRequest,
    EducatorEnrichRequest,
    FacultyReviewRequest,
    GapFinderRequest,
    TeachingPackRequest,
)
from backend.app.services.chat import answer_question
from backend.app.services.educator import (
    GUIDED_STARTERS,
    build_curriculum_outline,
    enrich_graph,
    role_options,
)
from backend.app.services.faculty_review import submit_faculty_review
from backend.app.services.gap_finder import find_education_gaps
from backend.app.services.teaching_pack import (
    build_teaching_pack,
    teaching_pack_to_docx_bytes,
    teaching_pack_to_markdown,
)

router = APIRouter(prefix="/api/v1/educator", tags=["educator"])


@router.get("/roles")
async def get_roles() -> dict:
    return {"roles": role_options()}


@router.get("/starters")
async def get_starters() -> dict:
    return {"starters": [item.model_dump() for item in GUIDED_STARTERS]}


@router.post("/enrich")
async def enrich_educator_graph(body: EducatorEnrichRequest) -> dict:
    cards = enrich_graph(body.graph, body.query, body.role)
    return {
        "cards": [card.model_dump() for card in cards],
        "summary": (
            "Each item below is explained for teaching and planning use. "
            "Technical details stay available in Advanced mode."
        ),
    }


@router.post("/curriculum")
async def curriculum_outline(body: CurriculumRequest) -> dict:
    return build_curriculum_outline(body.graph, body.query, body.role)


@router.post("/gaps")
async def gap_finder(request: Request, body: GapFinderRequest) -> dict:
    topic_keywords = body.topic or body.topic_keywords
    return await find_education_gaps(
        request.app.state.services.postgres,
        topic_keywords=topic_keywords,
        county=body.county,
        state=body.state,
        limit=body.limit,
    )


@router.post("/teaching-pack")
async def teaching_pack(body: TeachingPackRequest):
    pack = build_teaching_pack(body.graph, body.query, body.role)
    if body.format == "json":
        return pack
    if body.format == "markdown":
        content = teaching_pack_to_markdown(pack)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": 'attachment; filename="teaching_pack.md"'
            },
        )
    try:
        docx_bytes = teaching_pack_to_docx_bytes(pack)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(
        content=docx_bytes,
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        headers={
            "Content-Disposition": 'attachment; filename="teaching_pack.docx"'
        },
    )


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    settings = request.app.state.services.settings
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
    result = await answer_question(
        body.graph,
        body.query,
        [message.model_dump() for message in body.history],
        body.role,
        api_key,
        settings.openai_model,
    )
    return ChatResponse(**result)


@router.post("/review")
async def faculty_review(request: Request, body: FacultyReviewRequest) -> dict:
    return await submit_faculty_review(
        request.app.state.services.postgres,
        record_type=body.record_type,
        record_id=body.record_id,
        record_title=body.record_title,
        decision=body.decision,  # type: ignore[arg-type]
        reviewer=body.reviewer,
        notes=body.notes,
    )
