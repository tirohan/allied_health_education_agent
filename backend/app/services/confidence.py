from backend.app.agents.state import VerificationStatus


VERIFICATION_MULTIPLIER: dict[VerificationStatus, float] = {
    VerificationStatus.CONFIRMED: 1.0,
    VerificationStatus.INFERRED: 0.65,
    VerificationStatus.UNVERIFIED: 0.30,
    VerificationStatus.REFUTED: 0.0,
}


def apply_verification_multiplier(
    extraction_confidence: float,
    status: VerificationStatus,
) -> float:
    bounded = max(0.0, min(1.0, extraction_confidence))
    return round(bounded * VERIFICATION_MULTIPLIER[status], 4)


def confidence_delta(extraction_confidence: float, status: VerificationStatus) -> float:
    final_confidence = apply_verification_multiplier(extraction_confidence, status)
    return round(final_confidence - extraction_confidence, 4)
