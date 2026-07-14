from backend.app.agents.state import VerificationStatus
from backend.app.services.confidence import apply_verification_multiplier, confidence_delta


def test_apply_verification_multiplier() -> None:
    assert apply_verification_multiplier(0.8, VerificationStatus.CONFIRMED) == 0.8
    assert apply_verification_multiplier(0.8, VerificationStatus.INFERRED) == 0.52
    assert apply_verification_multiplier(0.8, VerificationStatus.UNVERIFIED) == 0.24
    assert apply_verification_multiplier(0.8, VerificationStatus.REFUTED) == 0.0


def test_confidence_delta() -> None:
    assert confidence_delta(0.8, VerificationStatus.INFERRED) == -0.28
