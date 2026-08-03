from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from server.schemas import DecisionAuditLog, IssueAnalysis, Decision


LOGGER = logging.getLogger(__name__)

CONTROL_PRIORITY = {"proceed": 0, "ask": 1, "amend": 2, "hold": 3}  # PRD 12장: hold > amend > ask > proceed
HIGH_RISK_ISSUES = {"명의도용", "지원상품아님"}
HOLD_RISK_FLAGS = {
    "fact_conflict",
    "identity_theft",
    "unauthorized_transaction",
    "fraud_suspected",
    "legal_uncertainty",
    "suspicious_input",
    "prompt_injection",
    "missing_critical_facts",
    "evidence_contradiction",
    "insufficient_temporal_clarity",
    "institutional_ambiguity",
    "unsupported_claim",
    "unverified_claim",
}
AMEND_RISK_FLAGS = {"pii_detected", "masking_required", "scope_review_required"}
LOW_CONFIDENCE_THRESHOLD = 0.6
MEDIUM_CONFIDENCE_THRESHOLD = 0.8

# False Negative 위험 신호 - 위험한데도 proceed가 나올 가능성이 높은 패턴들
FALSE_NEGATIVE_INDICATORS = {
    "identity_theft",
    "fraud_suspected",
    "unauthorized_transaction",
    "fact_conflict",
    "legal_uncertainty",
    "low_routing_confidence",
    "institutional_ambiguity",
    "insufficient_temporal_clarity",
}



class GateDecision(BaseModel):
    control: str
    reasons: list[str] = Field(default_factory=list)
    supporting_reasons: list[str] = Field(default_factory=list)
    human_review: bool = False
    audit_log: DecisionAuditLog | None = None
    false_negative_risk: str = "low"
    false_negative_indicators: list[str] = Field(default_factory=list)


def assess_risk(
    *,
    issue_type: str,
    target: dict[str, object],
    routing_confidence: float | None,
    risk_flags: list[str],
) -> tuple[str, list[str]]:
    flags = set(risk_flags)
    reasons = sorted(flags)
    if issue_type in HIGH_RISK_ISSUES or HOLD_RISK_FLAGS & flags:
        return "critical", reasons or ["high_risk_issue"]
    if "fact_conflict" in flags:
        return "high", reasons
    if flags & {"missing_facts", "evidence_insufficient", "customer_data_unavailable"}:
        return "high", reasons
    if routing_confidence is not None and routing_confidence < LOW_CONFIDENCE_THRESHOLD:
        return "high", [*reasons, "low_routing_confidence"]
    if routing_confidence is not None and routing_confidence < MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium", [*reasons, "medium_routing_confidence"]
    return "low", reasons


def apply_decision_gate(issue: IssueAnalysis) -> GateDecision:
    """Apply B's conservative policy over A's baseline decision.

    A returns a deterministic server-side baseline. B applies product support,
    safety, target clarity, and content-scope policy before composing a user
    response. Logs all decisions for auditability and detects false negative risks.
    """
    candidates: list[GateDecision] = [
        GateDecision(control=issue.decision.control, reasons=["A API baseline decision"])
    ]

    if issue.issue_type in HIGH_RISK_ISSUES:
        candidates.append(
            GateDecision(
                control="hold",
                reasons=["명의도용 또는 지원 제외 유형은 자동 판단하지 않습니다."],
                human_review=True,
            )
        )
    if HOLD_RISK_FLAGS & set(issue.decision.risk_flags):
        candidates.append(
            GateDecision(
                control="hold",
                reasons=["사실 충돌 또는 고위험 신호가 있어 사람이 확인해야 합니다."],
                human_review=True,
            )
        )
    if issue.target.get("is_unclear") is True and not issue.mock_data.get("available"):
        candidates.append(GateDecision(control="ask", reasons=["처리 대상 금융회사 또는 접수 대상이 불명확합니다."]))
    if issue.missing_facts:
        candidates.append(GateDecision(control="ask", reasons=["핵심 사실이 부족합니다."]))
    support_risks = _logic_support_risks(issue)
    if support_risks:
        candidates.append(
            GateDecision(
                control="ask",
                reasons=support_risks,
                human_review="unverified_claim" in support_risks,
            )
        )
    if issue.routing_confidence is not None and issue.routing_confidence < LOW_CONFIDENCE_THRESHOLD:
        candidates.append(
            GateDecision(
                control="ask",
                reasons=["low_routing_confidence"],
                human_review=True,
            )
        )
    if "evidence_insufficient" in issue.decision.risk_flags and not issue.evidence_refs:
        candidates.append(GateDecision(control="ask", reasons=["검색 근거가 부족합니다."]))
    if AMEND_RISK_FLAGS & set(issue.decision.risk_flags) or _requires_user_confirmation(issue):
        candidates.append(GateDecision(control="amend", reasons=["개인정보 마스킹 또는 제출 범위 확인이 필요합니다."]))

    decision = max(candidates, key=lambda item: CONTROL_PRIORITY[item.control])
    same_level_reasons = [reason for item in candidates if item.control == decision.control for reason in item.reasons]
    all_reasons = [reason for item in candidates for reason in item.reasons]
    
    # False negative 위험도 계산
    risk_flags = set(issue.decision.risk_flags) | set(support_risks)
    false_negative_risk, false_negative_indicators = _assess_false_negative_risk(
        control=decision.control,
        prior_control=issue.decision.control,
        issue_type=issue.issue_type,
        risk_flags=risk_flags,
        routing_confidence=issue.routing_confidence,
        missing_facts=issue.missing_facts,
        evidence_refs=issue.evidence_refs,
    )
    
    # 감사 로그 생성
    audit_log = DecisionAuditLog(
        audit_id=str(uuid4()),
        case_id=issue.issue_id,
        issue_id=issue.issue_id,
        event_type="decision_gate",
        created_at=datetime.utcnow(),
        decision=decision.control,
        prior_control=issue.decision.control,
        risk_flags=sorted(risk_flags),
        applied_rules=_dedupe(same_level_reasons),
        confidence_score=issue.routing_confidence,
        false_negative_risk=false_negative_risk,
        false_negative_indicators=false_negative_indicators,
        supporting_evidence={
            "routing_method": issue.routing_method,
            "risk_level": issue.risk_level,
            "missing_facts_count": len(issue.missing_facts),
            "evidence_count": len(issue.evidence_refs),
            "support_chain_count": len(issue.logic_verification.support_chains),
            "unsupported_claims": issue.logic_verification.unsupported_claims,
        },
    )
    
    # 감사 로그 기록
    LOGGER.info(
        "decision_gate audit=%s issue=%s control=%s prior=%s false_negative_risk=%s",
        audit_log.audit_id,
        issue.issue_id,
        decision.control,
        issue.decision.control,
        false_negative_risk,
    )
    
    return GateDecision(
        control=decision.control,
        reasons=_dedupe(same_level_reasons),
        supporting_reasons=_dedupe(all_reasons),
        human_review=decision.human_review or decision.control == "hold",
        audit_log=audit_log,
        false_negative_risk=false_negative_risk,
        false_negative_indicators=false_negative_indicators,
    )



def _assess_false_negative_risk(
    *,
    control: str,
    prior_control: str,
    issue_type: str,
    risk_flags: set[str],
    routing_confidence: float | None,
    missing_facts: list[str],
    evidence_refs: list[object],
) -> tuple[str, list[str]]:
    """평가: 이 판정이 위험 케이스를 놓칠 가능성"""
    indicators: list[str] = []
    score = 0
    
    # HIGH_RISK 이슈는 그 자체로 false negative 위험
    if issue_type in HIGH_RISK_ISSUES:
        score += 40
        indicators.append(f"high_risk_issue_type_{issue_type}")
        # 원래 control이 proceed였다면 더 위험
        if prior_control == "proceed":
            score += 60
            indicators.append("high_risk_was_proceeding")
    
    # False negative 지시자 확인
    for flag in risk_flags & FALSE_NEGATIVE_INDICATORS:
        if control in {"proceed", "ask"}:
            score += 30
            indicators.append(f"risky_flag_underestimated_{flag}")
    
    # Routing confidence가 낮은데 proceed면 위험
    if routing_confidence is not None and routing_confidence < LOW_CONFIDENCE_THRESHOLD:
        if control == "proceed":
            score += 40
            indicators.append("low_confidence_proceeding")
        elif control == "ask":
            score += 15
            indicators.append("low_confidence_asking")
    
    # 핵심 사실이 많이 부족하면서 proceed면 위험
    if len(missing_facts) >= 3 and control == "proceed":
        score += 35
        indicators.append(f"critical_facts_missing_{len(missing_facts)}")
    
    # 근거 자료가 전무한데 proceed면 위험
    if len(evidence_refs) == 0 and control == "proceed":
        score += 40
        indicators.append("no_evidence_proceeding")
    
    # False negative 위험도 결정
    if score >= 70:
        return "high", indicators
    elif score >= 40:
        return "medium", indicators
    else:
        return "low", indicators


def _requires_user_confirmation(issue: IssueAnalysis) -> bool:
    focal_scope = issue.focal.get("content_scope") if isinstance(issue.focal.get("content_scope"), dict) else {}
    return bool(
        issue.content_scope.get("requires_user_confirmation")
        or issue.content_scope.get("masked_fields")
        or focal_scope.get("requires_user_confirmation")
        or focal_scope.get("masked_fields")
    )


def _logic_support_risks(issue: IssueAnalysis) -> list[str]:
    risks: list[str] = []
    chains = issue.logic_verification.support_chains
    if issue.logic_verification.unsupported_claims:
        risks.append("unsupported_claim")
    if any(chain.inference_type == "unverified" for chain in chains):
        risks.append("unverified_claim")
    evidence_chains = [chain for chain in chains if chain.supporting_evidence]
    # 직접근거와 사례가 함께 검색되는 경우도 포함해야 한다 - 사례가 다른 직접근거에
    # "묻어가는" 방식으로 최종 결론에 섞여 들어가는 걸 all()로는 못 잡는다.
    has_precedent = any(chain.evidence_role == "precedent_reference" for chain in evidence_chains)
    if has_precedent and issue.decision.control == "proceed":
        risks.append("precedent_only_support")
    return _dedupe(risks)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
