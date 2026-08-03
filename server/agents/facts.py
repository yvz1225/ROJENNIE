from __future__ import annotations

import json
from collections import defaultdict
from datetime import date

from ..schemas import Fact, FactProvenanceEntry, FactResolution


# 원문 정규식 스캔으로 추출되는 필드는 한 문장에 값이 여러 개인 것이 정상이라
# (예: 예상 이자 30만원 + 실수령 279,180원) 값이 달라도 사실 충돌이 아니다.
# 충돌 판정은 "실제 적용 금리"처럼 의미가 하나로 정해진 필드에만 적용한다.
MULTI_VALUE_FIELDS = frozenset({"amount", "date_or_duration", "rate"})


def _value_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def resolve_facts(facts: list[Fact]) -> FactResolution:
    """Choose the latest recorded fact and retain conflicts instead of hiding them."""
    grouped: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        grouped[fact.field].append(fact)

    latest: dict[str, Fact] = {}
    conflicts: dict[str, list[str]] = {}
    provenance: dict[str, list[FactProvenanceEntry]] = {}
    for field, candidates in grouped.items():
        values = {_value_key(fact.value) for fact in candidates}
        if len(values) > 1 and field not in MULTI_VALUE_FIELDS:
            conflicts[field] = sorted(values)
            status = "conflict"
        else:
            status = "confirmed"

        latest[field] = max(
            candidates,
            key=lambda fact: (
                fact.recorded_date or fact.event_date or date.min,
                fact.event_date or date.min,
            ),
        )

        provenance[field] = [
            FactProvenanceEntry(
                field=fact.field,
                value=fact.value,
                source_type=fact.source_type,
                source_ref=fact.source_ref,
                status=status,
                confidence=fact.confidence,
            )
            for fact in sorted(candidates, key=lambda fact: (fact.recorded_date or fact.event_date or date.min, fact.event_date or date.min), reverse=True)
        ]
    return FactResolution(latest=latest, conflicts=conflicts, provenance=provenance)


def missing_facts(required: list[str], resolution: FactResolution) -> list[str]:
    return [field for field in required if field not in resolution.latest]
