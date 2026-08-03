from __future__ import annotations

import logging
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from server.agents.focal_builder import KNOWN_FACT_FIELDS, build_issue_input
from server.policy.gateway import LLMPolicyGateway
from server.schemas import CaseAnalyzeRequest, IssueInput


LOGGER = logging.getLogger(__name__)
RouteProduct = Literal["예금", "적금", "대출", "펀드", "ELS", "공통"]


class LLMRouteIssue(BaseModel):
    text: str
    product: RouteProduct
    issue_type: str
    focal: dict[str, Any] = Field(default_factory=dict)
    target: dict[str, Any] = Field(default_factory=dict)
    required_facts: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class LLMRouteResult(BaseModel):
    issues: list[LLMRouteIssue]


ROUTER_SYSTEM_PROMPT = """당신은 금융 소비자 민원 Issue Splitter다.
사용자 문장을 독립적으로 처리해야 하는 민원 이슈 목록으로 분해한다.

규칙:
- 허용 상품은 예금, 적금, 대출, 펀드, ELS, 공통이다.
- 상품과 쟁점이 다르고 필요한 규정·증빙·해결 조치가 다를 때만 분리한다.
- 같은 사건의 원인과 결과는 하나의 이슈로 유지한다.
- issue_type은 짧은 한국어 식별자로 작성한다. 법적 결론을 단정하지 않는다.
- text는 사용자의 원문에서 해당 이슈를 설명하는 부분을 최대한 그대로 사용한다.
- 상품이 불명확하면 공통으로 분류한다.
- 금융 규정, 사실, 금액, 날짜를 새로 만들지 않는다.
- 분리할 수 없으면 issues 하나로 반환한다.
"""


PRODUCT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ELS", ("ELS", "지수연계")),
    ("펀드", ("펀드", "환매", "운용보수", "보수")),
    ("대출", ("대출", "대출금", "대출금리", "원리금", "중도상환수수료", "대출 잔액", "연체", "상환")),
    ("적금", ("적금", "자동이체", "우대조건")),
    ("예금", ("예금", "정기예금", "계좌", "통장", "지급정지")),
)

# 상품을 못 찾아 "공통"으로 떨어진 문의 중, 아예 다른 업권(보험 등)이라 우리가
# 절대 매칭시킬 수 없는 경우를 구분한다. 이게 없으면 "정보가 더 필요합니다"를
# 영원히 반복하며 RAG/LLM 호출만 낭비한다 (평가 중 실측 확인됨).
OUT_OF_SCOPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("보험", ("보험금", "보험료", "실손보험", "상해보험", "종신보험", "자동차보험", "보험 가입", "보험사", "보험 계약")),
    ("상조", ("상조", "장례")),
    ("신용카드", ("신용카드", "체크카드", "카드대금", "카드 할부")),
    ("법률서비스", ("변호사 선임", "소송비용", "법률상담")),
)
OUT_OF_SCOPE_ISSUE_TYPE = "지원상품아님"

ISSUE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("명의도용", ("명의도용", "모르는", "신청한 적", "본인인증 없이", "저도 모르는 사이", "내가 신청하지 않은", "비인가")),
    ("계약해지_지연", ("해지 신청", "해지하려", "해지하러", "처리가 안", "처리를 안", "미뤄지고")),
    ("금리적용오류", ("우대금리를", "낮게 적용", "금리라고 들", "다르게 계산", "지급된 이자")),
    ("인출제한", ("인출", "출금", "지급정지", "못 찾고", "거부")),
    ("거래오류", ("거래내역", "만기 지급액", "이자 금액", "계산한 것", "차이가")),
    ("만기지급거절", ("만기", "지급을 안", "지급 처리가", "지급을 미루")),
    ("우대금리설명부족", ("우대금리", "우대조건", "조건을 제대로 설명", "안내를 못 받", "만기 때가 돼서야")),
    ("중도해지위약금", ("중도해지", "위약금", "계약서에 없던 수수료")),
    ("자동이체누락안내", ("자동이체", "이체 실패", "우대조건이 깨졌", "우대금리 0.5%를 놓쳤")),
    ("민원처리지연", ("민원", "문의", "답변이 없", "확인 중", "처리 상태")),
    ("위험설명부족", ("위험등급", "원금손실 위험", "손실 가능성", "위험에 대한 설명")),
    ("환매지연", ("환매", "환매금", "지연되고")),
    ("수수료미고지", ("수수료", "보수", "차감", "빠져나갔")),
    ("해지절차복잡", ("해지 신청 방법", "절차가 너무 복잡", "절차 안내")),
    ("손실민원부실", ("손실 원인", "손실 관련", "손실에 대한 민원", "답변이 형식적", "원론적인")),
    ("원금손실설명부족", ("ELS", "원금손실", "기초자산", "예금처럼 안전")),
    ("상환금과소지급", ("상환금", "조기상환", "만기 정산", "부족해")),
    ("배상비율불만", ("배상 비율", "배상비율", "산정 기준", "배상 협상")),
    ("중도해지손실", ("조기해지", "중도 해지", "중도상환", "손실 규모", "손실이")),
    ("분쟁조정안내부족", ("분쟁조정", "금감원", "신청 절차")),
    ("금리변경미통지", ("금리 변경", "금리변경", "변경 안내")),
    ("금리인상미통지", ("금리가", "금리", "인상 안내", "사전 안내")),
    ("설명의무위반", ("상품 설명", "조건에 대한 설명", "서류만 작성", "산정 기준을 설명")),
    ("대출금리변경미통지", ("대출 금리 변경", "대출금리 변경", "대출 금리 인상", "대출 사전 안내")),
    ("상환금액오류", ("대출 상환금", "원리금이 다르", "상환 금액이 다르", "상환내역 오류")),
    ("중도상환수수료", ("중도상환수수료", "중도상환 수수료")),
    ("연체이자산정", ("연체이자", "연체 이자", "연체금")),
)

# Decision Gate와 평가 데이터가 문자열 완전일치로 참조하는 고정 라벨 체계.
_EXTRA_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    ("연체이자과다", "연체이자가 과도하게 부과됨"),
    ("채권추심과다", "과도한 추심 연락·독촉"),
    (OUT_OF_SCOPE_ISSUE_TYPE, "보험·상조·카드·법률서비스 등 본 서비스가 다루지 않는 상품/서비스 문의"),
    ("미분류", "위 어디에도 해당하지 않을 때"),
)
ISSUE_TYPES: tuple[str, ...] = (
    *(name for name, _ in ISSUE_RULES),
    *(name for name, _ in _EXTRA_TYPE_HINTS),
)


def _issue_type_guide() -> str:
    lines = [f"- {name}: {', '.join(keywords)}" for name, keywords in ISSUE_RULES]
    lines.extend(f"- {name}: {hint}" for name, hint in _EXTRA_TYPE_HINTS)
    return "\n".join(lines)

SPLIT_RE = re.compile(r"(?:[.!?。]\s+)|(?:요\.\s+)|(?:니다\.\s+)|(?:,\s*(?=(?:예금|적금|대출|펀드|ELS)))|\s*(?:아 그리고|그리고 마지막으로|그런데|게다가|또|그리고)\s*")


def build_case_request(
    prompt: str,
    *,
    case_id: str | None = None,
    session_id: str | None = None,
    customer_id: str | None = "CUST-001",
    as_of: date | None = None,
    use_llm: bool | None = None,
    client: Any | None = None,
) -> CaseAnalyzeRequest:
    """Build the A-server payload using LLM routing when configured.

    ROUTER_MODE=auto uses the LLM only when GEMINI_API_KEY exists.
    ROUTER_MODE=llm forces an LLM attempt, then falls back to rules on failure.
    """
    return CaseAnalyzeRequest(
        case_id=case_id,
        session_id=session_id,
        customer_id=customer_id,
        prompt=prompt,
        as_of=as_of,
        issues=split_prompt_to_issues(prompt, use_llm=use_llm, client=client),
    )


def split_prompt_to_issues(
    prompt: str,
    *,
    use_llm: bool | None = None,
    client: Any | None = None,
) -> list[IssueInput]:
    if _llm_enabled(use_llm):
        try:
            issues = split_prompt_to_issues_llm(prompt, client=client)
            if issues:
                return issues
        except Exception as exc:
            LOGGER.warning("LLM routing failed; using rule fallback: %s", exc)
    return _split_prompt_to_issues_rules(prompt)


def split_prompt_to_issues_llm(prompt: str, *, client: Any | None = None) -> list[IssueInput]:
    schema = LLMRouteResult.model_json_schema()
    for definition in schema.get("$defs", {}).values():
        issue_type_prop = definition.get("properties", {}).get("issue_type")
        if issue_type_prop is not None:
            issue_type_prop["enum"] = list(ISSUE_TYPES)
    response = LLMPolicyGateway(client=client).generate_json(
        stage="issue_splitter",
        contents=ROUTER_SYSTEM_PROMPT
        + "\n\nissue_type은 반드시 다음 목록에서만 선택한다. 각 라벨 뒤는 해당 라벨의 대표 표현 예시다:\n"
        + _issue_type_guide()
        + "\n\nReturn structured focal, target, required_facts, and a confidence from 0 to 1 for every issue. Keep them grounded in the user input.\n\nUser input:\n"
        + prompt,
        response_schema=schema,
    )
    if not response.text:
        raise ValueError("Gemini returned no structured routing result")
    result = LLMRouteResult.model_validate_json(response.text)

    issues: list[IssueInput] = []
    for index, routed in enumerate(result.issues, start=1):
        text = routed.text.strip()
        issue_type = routed.issue_type.strip() or "미분류"
        if issue_type not in ISSUE_TYPES:
            issue_type = _classify_issue_type(text, routed.product)
        if not text:
            continue
        issue = build_issue_input(
            issue_id=f"issue_{index:03d}",
            product=routed.product,
            issue_type=issue_type,
            text=text,
            raw_product=routed.product,
            routing_method="llm",
            routing_confidence=routed.confidence if routed.confidence is not None else 0.8,
        )
        if routed.focal:
            issue = issue.model_copy(update={"focal": {**issue.focal, **routed.focal, "source": "agent.focal_builder.llm"}})
        if routed.target:
            issue = issue.model_copy(update={"target": {**issue.target, **routed.target}})
        # LLM은 "가입금액 1,000만원"처럼 값이 박힌 문장을 돌려주기도 한다. 그런 항목은
        # 영원히 미충족으로 남아 사용자가 이미 말한 사실을 다시 묻게 되므로 버린다.
        extra_facts = [field for field in routed.required_facts if field in KNOWN_FACT_FIELDS]
        if extra_facts:
            issue = issue.model_copy(
                update={"required_facts": list(dict.fromkeys([*issue.required_facts, *extra_facts]))}
            )
        issues.append(_apply_out_of_scope(issue, text))
    return issues


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _gemini_api_key() -> str | None:
    _load_dotenv()
    return next(
        (os.getenv(name) for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_KEY") if os.getenv(name)),
        None,
    )


def _gemini_client() -> Any:
    from google import genai

    return genai.Client(api_key=_gemini_api_key())


def _llm_enabled(use_llm: bool | None) -> bool:
    if use_llm is not None:
        return use_llm
    _load_dotenv()
    mode = os.getenv("ROUTER_MODE", "auto").strip().lower()
    if mode in {"off", "rule", "rules"}:
        return False
    if mode in {"llm", "gemini"}:
        return True
    return bool(_gemini_api_key())


def _split_prompt_to_issues_rules(prompt: str) -> list[IssueInput]:
    spans = _issue_spans(prompt)
    issues: list[IssueInput] = []
    previous_product: str | None = None
    for index, span in enumerate(spans, start=1):
        issue, previous_product = _build_issue(index, span, previous_product)
        issues.append(issue)
    return issues


def _build_issue(index: int, text: str, previous_product: str | None = None) -> tuple[IssueInput, str | None]:
    raw_product = _classify_product(text)
    issue_type = _classify_issue_type(text, raw_product)
    # 이 span 단독으로는 상품 키워드가 안 잡히면, issue_type만 보고 상품을 되짚어
    # 추측하는 _infer_product_from_issue보다 "직전 span에서 실제로 확인된 상품"을
    # 먼저 우선한다. "정기예금에 가입했는데... 그리고 중도해지 수수료가 다르대요"처럼
    # 복합 민원의 뒤 절이 상품명을 생략하는 게 자연스러운 한국어 화법이라, 문맥을
    # 버리고 하드코딩된 매핑(중도해지위약금 -> 적금 등)으로 넘어가면 오분류가 난다.
    resolved_product = raw_product or previous_product or _infer_product_from_issue(issue_type)
    issue = build_issue_input(
        issue_id=f"issue_{index:03d}",
        product=resolved_product or "공통",
        issue_type=issue_type,
        text=text,
        raw_product=resolved_product,
        routing_method="rules",
        routing_confidence=0.8 if raw_product else (0.7 if previous_product else 0.6),
    )
    return _apply_out_of_scope(issue, text), (raw_product or previous_product)


def _issue_spans(prompt: str) -> list[str]:
    raw_parts = [part.strip(" \t\r\n.,") for part in SPLIT_RE.split(prompt) if part and part.strip(" \t\r\n.,")]
    if not raw_parts:
        return [prompt.strip()]

    spans: list[str] = []
    current = ""
    for part in raw_parts:
        if current and _has_new_issue_signal(part):
            spans.append(current)
            current = part
        else:
            current = f"{current}. {part}" if current else part
    if current:
        spans.append(current)

    # raw_parts[0]은 첫 조각이라는 이유만으로 무조건 current에 들어가고 신호
    # 유무를 검사받지 않는다(위 루프의 `if current and ...`가 current가 아직
    # 비어 있을 때는 항상 거짓). "아니 진짜 너무 화나서 미치겠는데요!!!" 같은
    # 감정 표현만 있는 서두 뒤에 실제 상품/쟁점 신호가 오면, 그 경계에서
    # 분리되면서 신호 없는 서두가 "공통/미분류" 카드로 따로 떨어져 나간다.
    # 뒤 span에 흡수시켜 하나의 이슈로 합친다.
    if len(spans) > 1 and _classify_product(spans[0]) is None and _classify_issue_type(spans[0], None) == "미분류":
        spans[1] = f"{spans[0]}. {spans[1]}"
        spans = spans[1:]
    return spans


def _has_new_issue_signal(text: str) -> bool:
    if _classify_product(text) is None and text.startswith(("신청한 적", "벌써", "10일째", "14일째")):
        return False
    return _classify_product(text) is not None or _classify_issue_type(text, None) != "미분류"


def _classify_product(text: str) -> str | None:
    # 이전에는 PRODUCT_KEYWORDS 순서상 처음 매칭되는 상품을 무조건 채택했다
    # (ELS > 펀드 > 대출 > 적금 > 예금). 그러면 "대출 갚으려고 예금을 깼는데"처럼
    # 실제로는 예금이 중심인 문장도 앞쪽에 있는 "대출" 키워드 하나 때문에 대출로
    # 잘못 분류됐다. 키워드 매치 개수로 채점하고, 동점일 때만 기존 우선순위를 쓴다.
    scored = [
        (sum(1 for keyword in keywords if keyword in text), -order, product)
        for order, (product, keywords) in enumerate(PRODUCT_KEYWORDS)
        if any(keyword in text for keyword in keywords)
    ]
    if not scored:
        return None
    return max(scored)[2]


def _out_of_scope_label(text: str) -> str | None:
    return next((label for label, keywords in OUT_OF_SCOPE_KEYWORDS if any(keyword in text for keyword in keywords)), None)


def _apply_out_of_scope(issue: IssueInput, text: str) -> IssueInput:
    """공통으로 분류된 이슈가 보험 등 아예 다른 업권이면 issue_type을 덮어쓴다.

    라우팅 방식(rules/llm)에 상관없이 마지막에 한 번만 거치는 공통 지점이라,
    LLM이 이 라벨을 스스로 고르지 못해도 여기서 잡힌다.
    """
    if issue.product != "공통":
        return issue
    label = _out_of_scope_label(text)
    if not label:
        return issue
    return issue.model_copy(
        update={
            "issue_type": OUT_OF_SCOPE_ISSUE_TYPE,
            "focal": {**issue.focal, "out_of_scope_category": label},
        }
    )


def _classify_issue_type(text: str, product: str | None) -> str:
    if product == "대출":
        if any(keyword in text for keyword in ("중도상환수수료", "중도상환 수수료")):
            return "중도상환수수료"
        if any(keyword in text for keyword in ("연체이자", "연체 이자", "연체금")):
            return "연체이자산정"
        if any(keyword in text for keyword in ("상환 금액이 다르", "대출 상환금", "원리금이 다르", "상환내역 오류")):
            return "상환금액오류"
        if any(keyword in text for keyword in ("금리 변경", "금리변경", "금리 인상", "사전 안내")):
            return "대출금리변경미통지"
    if product == "적금" and any(keyword in text for keyword in ("자동이체", "이체 실패", "우대조건이 깨졌", "놓쳤")):
        return "자동이체누락안내"
    if product == "적금" and any(keyword in text for keyword in ("중도해지", "위약금", "계약서에 없던 수수료")):
        return "중도해지위약금"
    if product == "적금" and any(keyword in text for keyword in ("우대금리", "우대조건")) and not any(keyword in text for keyword in ("문의", "확인 중", "답변")):
        return "우대금리설명부족"
    if product == "적금" and any(keyword in text for keyword in ("만기", "지급을 안", "지급 처리가", "지급을 미루")):
        return "만기지급거절"
    if product == "펀드" and any(keyword in text for keyword in ("환매", "환매금")):
        return "환매지연"
    if product == "펀드" and any(keyword in text for keyword in ("수수료", "보수", "차감", "빠져나갔")):
        return "수수료미고지"
    if product == "펀드" and any(keyword in text for keyword in ("해지 신청 방법", "절차가 너무 복잡", "절차 안내")):
        return "해지절차복잡"
    if product == "ELS" and any(keyword in text for keyword in ("조기해지", "중도 해지", "중도상환", "손실 규모")):
        return "중도해지손실"
    if product == "ELS" and any(keyword in text for keyword in ("상환금", "조기상환", "만기 정산")):
        return "상환금과소지급"
    if product == "ELS" and any(keyword in text for keyword in ("배상 비율", "배상비율", "배상 협상")):
        return "배상비율불만"
    if product == "ELS" and any(keyword in text for keyword in ("분쟁조정", "금감원")):
        return "분쟁조정안내부족"

    scored: list[tuple[int, int, str]] = []
    for order, (issue_type, keywords) in enumerate(ISSUE_RULES):
        score = sum(1 for keyword in keywords if keyword in text)
        if score:
            scored.append((score, -order, issue_type))
    if not scored:
        return "미분류"

    issue_type = max(scored)[2]
    if product == "펀드" and issue_type == "위험설명부족":
        return "위험설명부족"
    if product == "ELS" and issue_type == "위험설명부족":
        return "원금손실설명부족"
    return issue_type


def _infer_product_from_issue(issue_type: str) -> str | None:
    if issue_type in {"중도해지위약금", "자동이체누락안내", "만기지급거절", "우대금리설명부족"}:
        return "적금"
    if issue_type in {"배상비율불만", "분쟁조정안내부족", "상환금과소지급", "중도해지손실", "원금손실설명부족"}:
        return "ELS"
    if issue_type in {"대출금리변경미통지", "상환금액오류", "중도상환수수료", "연체이자산정"}:
        return "대출"
    return None
