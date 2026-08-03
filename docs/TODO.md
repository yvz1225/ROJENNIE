# TODO

## 협업 원칙

두 사람이 같은 파일을 동시에 수정하지 않는다. 각자 담당 디렉터리에서 작업하고, 공통 계약 변경은 PR 전에 공유한다.

| 담당 | 담당 파일 | 책임 |
|---|---|---|
| A: 데이터·서버·MCP | `data/`, `server/`, `docs/todo/A_DATA_SERVER.md` | Mock 금융정보, RAG, Finance MCP, FastAPI, API·Tool 계약 |
| B: 에이전트·클라이언트 | `server/agents/`, `client/`, `docs/todo/B_AGENT_CLIENT.md` | 3개 에이전트, Policy Gate 연동, React Flow, 채팅·리포트 UI |
| 공동 | `README.md`, `docs/PRD.md`, `docs/TODO.md` | 구조 확정 후 한 명이 통합 수정 |

### Git 규칙

```text
A: feat/data-server
B: feat/agent-client
```

- A와 B는 각자 브랜치에서 작업한다.
- `main`에 직접 push하지 않는다.
- 공통 schema·API·MCP Tool 계약은 A가 먼저 제안하고 B가 사용한다.
- 공통 문서는 충돌 방지를 위해 한 번에 한 명만 수정한다.
- 통합 순서는 `A PR → 계약 확인 → B PR → 전체 테스트`다.
- 최신 원격 변경을 먼저 fetch하고, 충돌 해결 후 테스트한다.

## 구조 확정 사항

- 실제 LLM 에이전트는 `Case Builder`, `Evidence & Decision`, `Response` 3개다.
- Issue Splitter와 Focal Builder는 Case Builder 내부 단계다.
- My Info Resolver, RAG Retriever, Calculator, Logic Graph, Policy Gate는 일반 모듈 또는 MCP Tool이다.
- MCP는 새 데이터를 보관하지 않고 기존 함수와 RAG를 Tool로 노출한다.
- 브라우저는 MCP를 직접 호출하지 않고 FastAPI만 호출한다.
- 고객 ID는 LLM이 추측하지 않고 서버 세션에서 연결한다.
- 예금·적금·대출을 정식 지원한다.

## 통합 체크리스트

- [ ] `customer_ref`와 조회 동의 상태가 API에 포함됨
- [ ] `get_my_products` 결과가 issue별 account에 연결됨
- [ ] 사용자가 이미 입력한 사실을 재질문하지 않음
- [x] 로컬 `retrieval.py`가 기존 RAG를 호출하고 구조화된 결과를 반환함
- [ ] RAG 후보가 Logic Verification 이후에만 처리 결과에 반영됨
- [ ] 근거 후보가 별도 트리가 아니라 리포트 판단 근거에 표시됨
- [ ] 근거 상세 클릭 시 문서명·페이지·조항·인용문이 표시됨
- [ ] 검색 점수·검색 방식·내부 chunk ID가 기본 화면에 노출되지 않음
- [ ] `proceed / ask / amend / hold`가 모든 계층에서 동일함
- [ ] `issue_id`가 MCP 조회·검색·검증·리포트까지 유지됨
- [ ] 복합 민원 결과가 서로 섞이지 않음
- [ ] 사용자가 제공한 금액·금리·거래 사실을 다시 질문하지 않음
- [ ] `complex_issue_75.json` end-to-end 테스트 통과
- [ ] MCP 장애 시 근거 없는 proceed가 발생하지 않음
- [ ] 개인정보 원문이 기본 답변에 노출되지 않음

## 데이터 추가 기준

현재 MVP 구현에 필수 데이터는 추가하지 않는다. 다음은 기능 확장 시에만 수집한다.

- [ ] 공식 금융 분쟁조정·민원 사례 원문
- [ ] 공식 민원 접수 절차·제출 서류 정보
- [ ] 첨부 HWP·PDF 문서 추출용 테스트 파일
- [ ] 추가 예금·적금 상품 설명서

실제 고객정보, 실제 계좌번호, 주민번호는 수집하지 않는다.

## 완료 기준

- [ ] 사용자가 내 금융정보 조회 동의 후 문의를 입력할 수 있음
- [ ] Case Builder가 복합 문의를 issue별 Case로 변환함
- [x] Finance MCP가 내 금융정보·계산 Tool을 제공함
- [ ] Evidence & Decision Agent가 MCP 결과와 사용자 진술을 함께 검증함
- [ ] Response Agent가 세 가지 리포트 섹션을 생성함
- [ ] React Flow에서 민원 분기와 사용자 답변 노드를 확인할 수 있음
- [ ] 추가 정보가 필요한 단계만 채팅 질문과 빨간 표시를 가짐
- [ ] 서버 재시작 이후 민원 기록 보존 여부가 명확함
- [ ] 서버·클라이언트·MCP·회귀 테스트가 모두 통과함
