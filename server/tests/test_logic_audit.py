from __future__ import annotations

import unittest

from server.agents.decision_gate import apply_decision_gate
from server.agents.logic_verification import verify_issue_logic
from server.agents.report_composer import compose_issue_report
from server.schemas import Decision, EvidenceRef, Fact, FactResolution, IssueAnalysis


def _issue(
    *,
    evidence_refs: list[EvidenceRef] | None = None,
    control: str = "proceed",
    logic: bool = True,
) -> IssueAnalysis:
    issue = IssueAnalysis(
        issue_id="issue_logic_001",
        product="deposit",
        issue_type="interest_miscalculation",
        focal={},
        target={},
        facts=[
            Fact(
                field="user_statement",
                value="The maturity interest looks lower than expected.",
                source_type="USER_STATED",
                source_ref="user_input",
            )
        ],
        missing_facts=[],
        fact_resolution=FactResolution(),
        evidence_refs=evidence_refs or [],
        decision=Decision(control=control, risk_flags=[]),
        next_steps=[],
    )
    if logic:
        issue = issue.model_copy(update={"logic_verification": verify_issue_logic(issue)})
    return issue


class LogicAuditTests(unittest.TestCase):
    def test_unverified_claim_downgrades_proceed_to_ask(self) -> None:
        issue = _issue(evidence_refs=[], control="proceed")

        gate = apply_decision_gate(issue)

        self.assertEqual(gate.control, "ask")
        self.assertIn("unverified_claim", gate.supporting_reasons)
        self.assertIn("unsupported_claim", gate.audit_log.risk_flags)

    def test_precedent_only_support_is_not_treated_as_direct_conclusion(self) -> None:
        issue = _issue(
            evidence_refs=[
                EvidenceRef(
                    doc_id="case_doc",
                    chunk_id="case_doc-p1-c1",
                    path="local:cases/kca/dispute_case.md",
                    page=1,
                    section="case",
                    score=0.8,
                    snippet="A similar dispute case.",
                )
            ],
            control="proceed",
        )

        gate = apply_decision_gate(issue)
        report = compose_issue_report(issue)

        self.assertEqual(gate.control, "ask")
        self.assertIn("precedent_only_support", gate.supporting_reasons)
        self.assertIn("유사 사례는 참고용", " ".join(report.consumer_cautions))

    def test_precedent_mixed_with_direct_evidence_still_gets_caution(self) -> None:
        issue = _issue(
            evidence_refs=[
                EvidenceRef(
                    doc_id="terms_doc",
                    chunk_id="terms_doc-p1-c1",
                    path="local:products/deposit/terms.md",
                    page=1,
                    section="interest",
                    score=0.9,
                    snippet="Direct product terms.",
                ),
                EvidenceRef(
                    doc_id="case_doc",
                    chunk_id="case_doc-p1-c1",
                    path="local:cases/kca/dispute_case.md",
                    page=1,
                    section="case",
                    score=0.8,
                    snippet="A similar dispute case.",
                ),
            ],
            control="proceed",
        )

        gate = apply_decision_gate(issue)
        report = compose_issue_report(issue)

        self.assertIn("precedent_only_support", gate.supporting_reasons)
        self.assertIn("유사 사례는 참고용", " ".join(report.consumer_cautions))

    def test_direct_evidence_can_remain_proceed_when_no_other_risks(self) -> None:
        issue = _issue(
            evidence_refs=[
                EvidenceRef(
                    doc_id="terms_doc",
                    chunk_id="terms_doc-p1-c1",
                    path="local:products/deposit/terms.md",
                    page=1,
                    section="interest",
                    score=0.9,
                    snippet="Direct product terms.",
                )
            ],
            control="proceed",
        )

        gate = apply_decision_gate(issue)

        self.assertEqual(gate.control, "proceed")
        self.assertFalse(issue.logic_verification.unsupported_claims)


if __name__ == "__main__":
    unittest.main()
