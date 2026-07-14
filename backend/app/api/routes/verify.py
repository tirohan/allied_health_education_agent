from fastapi import APIRouter, Request

from backend.app.agents.extraction import _coerce_relation
from backend.app.agents.state import Relation
from backend.app.agents.verification import verify_all
from backend.app.schemas.api import VerifyRequest, VerifyResponse

router = APIRouter(prefix="/api/v1", tags=["verification"])


@router.post("/verify")
async def verify(request: Request, body: VerifyRequest) -> VerifyResponse:
    relations: list[Relation] = []
    for item in body.relations:
        if isinstance(item, dict):
            coerced = _coerce_relation(item)
            if coerced is not None:
                relations.append(coerced)
                continue
        relations.append(Relation.model_validate(item))
    results = await verify_all(request.app.state.services.postgres, body.entities, relations)
    return VerifyResponse(results=results)
