# 학습 & 소유권 확보 로드맵

> AI가 만든 코드를 나의 경험과 스킬로 만들기 위한 단계별 가이드

---

## 왜 "이해"만으로는 부족한가

면접에서 이 질문이 옵니다:

> "ParentDocumentRetriever를 쓰셨는데, InMemoryStore 대신 파일 기반 Persistent Store를 선택한 이유가 뭔가요? **직접 겪어보셨나요?**"

코드를 읽었다면 답할 수 있습니다.
하지만 "직접 겪었다면"이라는 질문에는 못 답합니다.
면접관은 그 차이를 압니다.

**소유감(Ownership)은 코드를 이해하는 것이 아니라, 직접 부딪히고 고친 경험에서 옵니다.**

---

## 프로젝트 구조 한 눈에 보기

이 프로젝트는 크게 **3개의 레이어**로 구성되어 있다.

### Layer 1 — 데이터 흐름 (RAG 파이프라인)

> 논문이 어떻게 처리되어 검색 가능한 형태가 되는지

```
PDF 파일
  → parser.py     : LlamaParse로 텍스트 + 페이지 번호 추출
  → chunker.py    : 부모/자식 청크로 분할
  → vectorstore.py: ChromaDB에 저장
```

### Layer 2 — 상태 관리 (LangGraph State)

> 에이전트들이 공유하는 "칠판"의 구조

```
state.py
  ├── ContextChunk  (검색 결과 단위 구조체)
  └── AgentState    (전체 워크플로우 상태)
        └── contexts: Annotated[list, operator.add]  ← 핵심!
```

`state.py:44`의 핵심 패턴:
```python
contexts: Annotated[list[ContextChunk], operator.add]
```
- `local_retriever_node`와 `web_searcher_node`가 동시에 실행
- 둘 다 `contexts`에 결과를 쓰려 할 때 → **덮어쓰기 충돌 발생** 가능
- `operator.add` 리듀서가 자동으로 **두 리스트를 합산**

### Layer 3 — 에이전트 워크플로우 (Graph + Nodes)

> 실제 실행 흐름

```
graph.py  (설계도)          nodes.py (실제 코드)
  START
    ↓↓ (fan-out)
  local_retriever  ←→  local_retriever_node()
  web_searcher     ←→  web_searcher_node()
    ↓ (fan-in)
  vision_router    ←→  vision_router_node() + route_vision()
    ↓ (조건부)
  vision_analyst   ←→  vision_analyst_node()
    ↓
  synthesis        ←→  synthesis_node()
```

### 특히 중요한 개념

| 개념 | 위치 | 한 줄 설명 |
|------|------|-----------|
| `operator.add` reducer | `state.py:44` | 병렬 노드의 리스트 충돌 방지 |
| fan-out/fan-in | `graph.py:62-66` | 동시 실행 → 합류 패턴 |
| Conditional Edge | `graph.py:70-77` | 조건에 따라 다른 노드로 분기 |
| ParentDocumentRetriever | `chunker.py` | 검색은 작은 조각으로, 반환은 큰 문맥으로 |

---

## 1단계: 구조 파악 + 해부

**목표**: 모든 파일을 읽고, 한 줄씩 설명할 수 있어야 한다.

### Step 1 — 그림 그리기

`graph.py`를 읽고 전체 흐름을 **손으로 다이어그램** 그려보기 (30분)

### Step 2 — 단위별 읽기 + 주석 재작성 (동시 수행)

아래 순서로 파일을 열고, 한 줄씩 읽으면서 주석을 자기 말로 재작성한다.
설명할 수 없는 줄이 아직 이해 못 한 부분이다.

```
state.py → graph.py → nodes.py → parser.py → chunker.py → vectorstore.py → main.py
```

각 줄을 읽을 때 두 가지 질문을 동시에 던진다:

| 이해 질문 | 소유 질문 |
|---|---|
| "이 코드가 없으면 어떻게 될까?" | "이 줄을 남에게 설명할 수 있나?" |

**Layer별 핵심 포인트:**

- `state.py` — `operator.add` 리듀서가 왜 contexts에만 붙어있는가
- `graph.py:62-66` — `START → 두 노드` 연결이 자동으로 병렬 실행되는 이유
- `nodes.py:176-233` — Vision Router 3단계 비용 최적화 로직
- `nodes.py:236-244` — `route_vision()`이 노드 함수가 아닌 이유
- `chunker.py` — `CHILD_CHUNK_SIZE = 1600` (≈400 tokens)과 `PARENT_CHUNK_SIZE = 8000`의 역할 차이

### Step 3 — 테스트로 검증

```bash
pytest tests/agents/test_nodes.py -v
pytest tests/agents/test_graph.py -v
```

테스트가 무엇을 검증하는지 읽으면서 동작 방식 확인

### Step 4 — 직접 수정해보기

`nodes.py:33`의 `_VISION_KEYWORDS`에 새 단어를 추가하고 테스트 돌려보기

```bash
pytest tests/agents/test_nodes.py::TestVisionRouterNode -v
```

### 10가지 질문 — 완료 기준

아래 질문에 모두 답할 수 있으면 1단계 완료다.
막히는 질문이 남아있으면 해당 파일로 돌아간다.

```
1. operator.add가 없으면 병렬 노드에서 무슨 일이 생기나?
2. vision_router가 3단계로 나뉜 이유는?
3. ainvoke와 astream_events의 반환 타입 차이는?
4. ChromaDB를 self-hosted로 쓰는 이유는?
5. TypedDict를 State에 쓰고 Pydantic은 API 경계에만 쓰는 이유는?
6. ParentDocumentRetriever에서 child/parent를 분리하는 이유는?
7. lifespan 함수가 없으면 어떻게 되나?
8. pdf_hash를 SHA-256으로 만드는 이유는?
9. route_vision()이 노드 함수가 아닌 이유는?
10. astream_events의 version="v2"는 무슨 의미인가?
```

---

## 2단계: 직접 실행하고 버그 고치기

**목표**: AI가 만든 코드에는 버그가 있을 가능성이 높다. 그것을 직접 찾고 고친다.

> **버그를 직접 고친 순간, 그 코드는 당신 것이 됩니다.**

### 실행 방법

```bash
# 1. .env 파일에 API 키 설정
OPENAI_API_KEY=...
TAVILY_API_KEY=...
LLAMA_CLOUD_API_KEY=...

# 2. 서버 실행
PYTHONPATH=. uv run python src/main.py

# 3. 논문 업로드
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/pdfs/논문파일.pdf"

# 4. 질문
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "핵심 방법론은?", "pdf_hash": "..."}'
```

### 완료 기준 — 확인 체크리스트

- [ ] 이미지 파일명 포맷이 실제로 맞는가 (`page_001.png` vs `page_1.png`)
- [ ] LlamaParse JSON 모드의 `page_number`가 메타데이터에 실제로 들어가는가
- [ ] ChromaDB 재시작 후 데이터가 실제로 유지되는가
- [ ] vision_analyst가 이미지를 제대로 로드하는가
- [ ] `/ask/stream` SSE 스트림이 클라이언트에서 실제로 동작하는가

---

## 3단계: PRD에 없는 기능 하나 직접 추가

**목표**: "추가했다"는 경험이 소유감을 만든다.

포트폴리오에서 "이 부분은 제가 직접 설계하고 추가했습니다"라고 말할 수 있어야 한다.

### 추천 기능 (난이도 순)

| 기능 | 난이도 | 건드리는 파일 |
|---|---|---|
| 저장된 PDF 목록 조회 API (`GET /pdfs`) | 낮음 | `main.py` |
| 대화 히스토리 유지 | 중간 | `state.py`, `nodes.py`, `main.py` (checkpointer 추가 필요) |
| GPT 토큰 비용 추적 로깅 | 중간 | `nodes.py`, `main.py` |
| 멀티 PDF 동시 질문 | 높음 | `state.py`, `graph.py`, `nodes.py`, `main.py` |

### 완료 기준

선택한 기능이 테스트 포함 완성됐을 때

---

## 4단계: README와 아키텍처 문서 직접 작성

**목표**: 설명할 수 있어야 진짜 아는 것이다. (Feynman Technique)

> Feynman Technique의 핵심: 개념을 처음 배우는 사람에게 설명하듯 써보면,
> 실제로 모르는 부분이 어디인지 드러난다.

### README 필수 항목

- 이 프로젝트가 해결하는 문제 (일반 RAG의 한계와 차별점)
- LangGraph 흐름도 (직접 그린 다이어그램, 텍스트로도 가능)
- 데모 GIF 또는 스크린샷
- 설치 및 실행 방법
- **왜 이 기술을 선택했는지** — 이 섹션이 가장 중요

### 트레이드오프 섹션 예시

```markdown
## 설계 결정

### TypedDict vs Pydantic (State 정의)
TypedDict를 선택한 이유: LangGraph 공식 권장 방식이고,
Pydantic보다 가볍고 오버헤드가 없기 때문.
(LangGraph는 Pydantic BaseModel도 State로 지원하지만,
 TypedDict가 더 관용적이고 reducer와 함께 쓰기 단순하다.)
Pydantic은 외부 API 입출력 경계(AskRequest, AskResponse)에만 사용.

### ParentDocumentRetriever
child chunk(~400 tokens)로 검색 정밀도를 높이고,
매칭된 경우 parent chunk로 확장하여 문맥을 제공.
InMemoryStore 대신 LocalFileStore를 사용하여 서버 재시작 시 데이터 유지.

### 3단계 Vision Router
LLM 호출 비용 절감을 위해 키워드 판단 → 메타데이터 판단 → LLM 판단 순서.
1·2단계는 API 호출이 없으므로 비용이 0원.
```

이 문서는 코드를 보지 않고 **자기 말로** 써야 한다.

### 완료 기준

README를 코드 없이 자기 말로 쓸 수 있을 때

---

## 5단계: 면접 질문 10개 준비

**목표**: 포트폴리오는 면접 대화를 위한 도구다.

아래 질문과 답변을 직접 써보세요. 막히면 코드를 다시 읽고, 실행해보세요.

```
Q1.  LangGraph를 쓴 이유가 뭔가요?
Q2.  병렬 처리에서 데이터 충돌을 어떻게 방지했나요?
Q3.  비전 분석 비용을 어떻게 최적화했나요?
Q4.  FastAPI의 /ask와 /ask/stream 차이는 무엇이고 언제 각각 쓰나요?
Q5.  ChromaDB를 선택한 이유는?
Q6.  LlamaParse를 JSON 모드로 쓰는 이유는?
Q7.  Docker 멀티스테이지 빌드를 쓴 이유는?
Q8.  이 프로젝트에서 가장 어려웠던 부분은?
Q9.  개선하고 싶은 부분이 있다면?
Q10. 실제로 돌려봤을 때 예상과 달랐던 점은?
```

Q8~Q10은 2단계(직접 실행)에서 겪은 경험으로 답해야 한다.
이 세 개를 구체적으로 답할 수 있으면 면접관은 "직접 만든 프로젝트"로 인식한다.

### 완료 기준

10가지 질문에 구체적 경험으로 답할 수 있을 때

---

## 단계 순서 요약

각 단계에 걸리는 시간은 사람마다 다릅니다.
"몇 주"보다 "완료 기준을 충족했는가"로 진행 여부를 판단하세요.

```
1단계  구조 파악 + 해부   10가지 질문에 모두 답할 수 있을 때까지
2단계  실행 + 수정        서버가 실제로 뜨고 PDF → 질문 → 답변이 동작할 때까지
3단계  기능 추가          선택한 기능이 테스트 포함 완성될 때까지
4단계  문서화             README를 코드 없이 자기 말로 쓸 수 있을 때까지
5단계  면접 준비          10가지 질문에 구체적 경험으로 답할 수 있을 때까지
```

일일 실행 방법은 → [`daily-workflow.md`](./daily-workflow.md)

---

## 핵심 원칙

> "내가 만들지 않았다"는 사실보다
> "내가 이해하고 수정하고 확장했다"는 사실이 중요합니다.

기존 코드를 이해하고, 문제를 찾고, 개선하는 능력이 실무 역량입니다.

**지금 당장 시작할 것**: 1단계 Step 1 — `graph.py`를 열고 흐름을 손으로 그려보세요.
