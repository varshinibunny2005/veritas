import pytest

from app.arbiter import ArbitrationError, aggregate, collect_issues, decide_verdict
from app.schemas import CriticReport, Issue

W = {"accuracy": 0.3, "grounding": 0.3, "completeness": 0.2, "consistency": 0.2}


def rep(name, score, conf=0.8, issues=None, **kw):
    return CriticReport(critic=name, score=score, confidence=conf, issues=issues or [], **kw)


def issue(critic, sev, claim_id=None, cat="x"):
    return Issue(critic=critic, category=cat, severity=sev, description="d", claim_id=claim_id)


def test_score_bands_without_issues():
    assert decide_verdict(90, [])[0] == "ACCEPT"
    assert decide_verdict(75, [])[0] == "ACCEPT_WITH_WARNINGS"
    assert decide_verdict(55, [])[0] == "REVISE"
    assert decide_verdict(30, [])[0] == "REJECT"


def test_severity_caps():
    assert decide_verdict(95, [issue("accuracy", "medium")])[0] == "ACCEPT_WITH_WARNINGS"
    assert decide_verdict(95, [issue("accuracy", "high")])[0] == "REVISE"
    assert decide_verdict(95, [issue("accuracy", "critical", 1)])[0] == "REVISE"
    assert decide_verdict(55, [issue("accuracy", "critical", 1)])[0] == "REJECT"
    both = [issue("accuracy", "critical", 1), issue("grounding", "critical", 2)]
    assert decide_verdict(95, both)[0] == "REJECT"


def test_same_claim_counts_once():
    dup = [issue("accuracy", "critical", 1), issue("grounding", "critical", 1)]
    assert decide_verdict(95, dup)[0] == "REVISE"


def test_corroboration_raises_severity():
    reports = [rep("accuracy", 60, issues=[issue("accuracy", "high", 1)]),
               rep("grounding", 60, issues=[issue("grounding", "high", 1)])]
    issues = collect_issues(reports)
    assert all(i.severity == "critical" for i in issues)
    assert issues[0].corroborated_by
    assert [i.id for i in issues] == ["I1", "I2"]


def test_single_critic_not_corroborated():
    issues = collect_issues([rep("accuracy", 60, issues=[issue("accuracy", "high", 1)])])
    assert issues[0].severity == "high" and not issues[0].corroborated_by


def test_aggregate_ignores_unusable_critics_and_lowers_confidence():
    full = aggregate([rep("accuracy", 80), rep("grounding", 80), rep("completeness", 80), rep("consistency", 80)], W)
    part = aggregate([rep("accuracy", 80), rep("grounding", None, evaluable=False),
                      rep("completeness", 80), rep("consistency", 80)], W)
    assert round(full["overall"]) == 80 and round(part["overall"]) == 80
    assert part["confidence"] < full["confidence"]


def test_disagreement_lowers_confidence():
    agree = aggregate([rep("accuracy", 70), rep("grounding", 70)], W)
    split = aggregate([rep("accuracy", 100), rep("grounding", 20)], W)
    assert split["confidence"] < agree["confidence"] and split["disagreement"] > 30


def test_no_usable_critic_raises():
    with pytest.raises(ArbitrationError):
        aggregate([rep("accuracy", None, evaluable=False, error="boom")], W)
