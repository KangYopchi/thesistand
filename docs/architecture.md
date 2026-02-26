# thesistand 아키텍처 문서

논문 분석 RAG 시스템 `thesistand`의 전체 구조와 각 파일의 설계 이유를 설명한다.

---

## 1. 프로젝트 개요

`thesistand`는 학술 논문 PDF를 업로드하면 자연어로 질문할 수 있는 AI 논문 분석 백엔드다.

핵심 가치:
- **표·그래프·수식 이해**: 텍스트만 보는 일반 RAG와 달리 페이지 이미지를 Vision 모델로 분석한다.
- **정확한 출처 추적**: 모든 답변에 논문 페이지 번호를 함께 반환한다.
- **비용 최적화**: Vision API 호출을 꼭 필요한 경우에만 수행한다.
- **실시간 스트리밍**: 처리 중인 노드를 SSE로 클라이언트에 실시간 전달한다.

---

## 2. 기술 스택

| 라이브러리 | 역할 | 선택 이유 |
|-----------|------|----------|
| **LangGraph** | 에이전트 워크플로우 | 병렬 노드(fan-out), 조건부 분기(conditional edge), 상태 기계(State Machine)를 선언적으로 표현 가능 |
| **FastAPI** | HTTP 서버 | async/await 네이티브 지원, SSE 스트리밍 내장, 자동 OpenAPI 문서 생성 |
| **ChromaDB** | 벡터 데이터베이스 | 로컬 Persistent 모드 지원 → 별도 DB 서버 없이 파일 시스템에 영속화 |
| **LlamaParse** | PDF 파싱 | JSON 결과 모드에서 페이지 번호·요소 타입(table/image/text) 메타데이터까지 추출. 마크다운 모드는 이 정보가 손실됨 |
| **gpt-5-mini** | LLM 추론 + Vision | Vision Router 3단계 판단, 이미지 분석, 최종 답변 합성 |
| **text-embedding-3-small** | 텍스트 임베딩 | OpenAI 임베딩 중 비용 대비 성능 최우수 (ada-002 대비 성능↑, 비용↓) |
| **Tavily** | 웹 검색 | RAG 기반 논문 내용 외 최신 정보를 보완하는 외부 검색 |
| **uv** | 패키지 관리 | pip보다 10-100× 빠른 설치 속도, `pyproject.toml` 기반 재현 가능한 환경 |

---

## 3. 전체 아키텍처

### 두 개의 LangGraph

인제스트(저장)와 쿼리(질문)를 **별도 그래프**로 분리했다.

**이유:** 인제스트는 PDF 업로드 시 1회만 실행되는 일회성 작업이고, 쿼리는 요청마다 반복 실행된다. 하나의 그래프에 합치면 불필요한 엣지와 조건이 추가되어 복잡도가 불필요하게 증가한다.

---

#### Ingest Graph

```
POST /ingest
    │
    ▼
START → ingest_node → END
          │
          ├─ LlamaParse API (JSON 모드)     ─┐
          └─ pdf2image (asyncio.gather 병렬) ─┘
          │
          ├─► data/chroma_db/      (Child chunk 벡터)
          ├─► data/parent_store/   (Parent chunk JSON)
          └─► data/images/{hash}/  (페이지 이미지 PNG)
```

---

#### Query Graph

```
POST /ask (또는 /ask/stream)
    │
    ▼
START ──► local_retriever_node ──┐
      └─► web_searcher_node    ──┤  fan-out: 병렬 실행
                                 │  fan-in:  contexts 리스트에 합산
                                 ▼
                        vision_router_node
                         (1단계: 키워드)
                         (2단계: 메타데이터)
                         (3단계: LLM)
                         /              \
              NEED_VISION              NO_VISION
                 /                        \
    vision_analyst_node              synthesis_node
         (gpt-5-mini Vision)              │
                 \                        │
                  └──────────────────────►┘
                                synthesis_node → END
```

---

### 데이터 흐름

```
사용자 PDF 업로드
    │
    ▼
SHA-256 해시 계산 → 중복 확인
    │ (신규)
    ├─► data/pdfs/{hash}.pdf 저장
    ├─► LlamaParse: elements[] 추출 (page_number, text, element_type)
    │       └─► Child chunk 임베딩 → ChromaDB
    │       └─► Parent chunk JSON → data/parent_store/
    └─► pdf2image: page_001.png ... → data/images/{hash}/

사용자 질문
    │
    ▼
local_retriever: Child 벡터 검색 → Parent 문맥 반환
web_searcher:    Tavily API 호출
    │ (contexts 합산)
    ▼
vision_router: 비전 분석 필요 여부 판단
    │ (NEED_VISION)
    ▼
vision_analyst: data/images/{hash}/page_NNN.png → base64 → gpt-5-mini
    │
    ▼
synthesis: 모든 컨텍스트 + 비전 결과 → 최종 답변 (출처 포함)
```

---

## 4. 파일별 상세 설명

### `src/agents/state.py`

AgentState와 ContextChunk를 정의하는 상태 스키마 모듈.

**TypedDict 선택 이유**

LangGraph는 상태를 딕셔너리로 관리하므로 TypedDict가 가장 자연스럽게 맞는다. Pydantic은 검증 오버헤드가 있고, TypedDict와 달리 LangGraph의 reducer(`Annotated`) 문법과 바로 호환되지 않는다.

**`Annotated[list[ContextChunk], operator.add]`의 역할**

병렬로 실행되는 `local_retriever_node`와 `web_searcher_node`는 둘 다 `contexts` 필드에 결과를 쓴다. reducer 없이 두 노드가 같은 필드에 동시에 쓰면 **마지막 write가 나머지를 덮어쓴다.** `operator.add`를 reducer로 지정하면 두 노드의 결과가 자동으로 리스트에 합산된다.

```python
# reducer가 없으면 (잘못된 예):
# local_retriever가 ["A", "B"] 반환
# web_searcher가 ["C"] 반환
# 최종 contexts = ["C"]  ← local_retriever 결과 유실!

# reducer operator.add 적용 시:
# 최종 contexts = ["A", "B", "C"]  ← 정상 합산
```

**ContextChunk의 Optional 필드**

| 필드 | local_rag | web_search |
|------|-----------|------------|
| `page_number` | 논문 페이지 번호 | `None` |
| `url` | `None` | 검색 결과 URL |
| `element_type` | "text" / "table" / "image" / "figure" | `None` |

같은 타입으로 두 출처를 통합하면서 Vision Router가 `element_type`을 조회할 수 있다.

---

### `src/agents/graph.py`

두 LangGraph를 빌드하는 팩토리 모듈.

**fan-out/fan-in 구현 방법**

LangGraph에서 같은 START에서 두 노드로 엣지를 연결하면 자동으로 병렬 실행된다. 두 노드 모두 `vision_router`로 엣지를 연결하면 둘이 완료된 후에야 `vision_router`가 시작된다(fan-in).

```python
graph.add_edge(START, "local_retriever")  # 동시 시작
graph.add_edge(START, "web_searcher")     # 동시 시작

graph.add_edge("local_retriever", "vision_router")  # 둘 다 완료 후
graph.add_edge("web_searcher", "vision_router")     # vision_router 진입
```

**conditional_edge**

`route_vision` 함수가 `"vision_analyst"` 또는 `"synthesis"` 문자열을 반환하면, LangGraph가 해당 노드로 동적 분기한다. 이를 통해 비전이 불필요한 질문에서 Vision API 비용을 완전히 제거한다.

---

### `src/agents/nodes.py`

6개의 노드 함수와 1개의 라우팅 함수를 담은 핵심 모듈.

**함수형 노드 선택 이유**

노드가 6개이고 State가 하나일 때는 함수가 클래스보다 명확하다. 각 함수는 `AgentState`를 받아 업데이트할 필드만 딕셔너리로 반환한다는 단순한 계약을 따른다. 클래스 전환이 유리한 시점은 (1) 같은 노드를 설정만 다르게 여러 개 만들 때, (2) 노드가 자체 State와 Graph를 가지는 서브에이전트가 될 때다.

**Vision Router 3단계 비용 최적화**

```
1단계 (무료): 질문에 "표", "table", "figure" 등 키워드 포함 → 즉시 NEED_VISION
    ↓ (해당 없으면)
2단계 (무료): contexts의 element_type이 "table"/"image"/"figure" → NEED_VISION
    ↓ (해당 없으면)
3단계 (유료): gpt-5-mini 호출로 최종 판단
```

"표를 설명해줘"처럼 명확한 질문은 1단계에서 즉시 처리되어 LLM 비용이 전혀 들지 않는다.

**`_get_openai_client()` 싱글톤**

`AsyncOpenAI` 인스턴스는 내부에 `httpx.AsyncClient`를 유지한다. 요청마다 새로 생성하면 TCP 연결을 매번 새로 수립(TLS 핸드셰이크 포함)하는 오버헤드가 발생한다. 싱글톤으로 관리하면 커넥션 풀을 재사용한다.

**`pdf_hash` 필터**

단일 ChromaDB 컬렉션 `"thesistand"`에 모든 PDF의 청크가 혼재한다. `retriever.search_kwargs = {"filter": {"source": pdf_hash}}`로 특정 논문 청크만 검색하여 다른 논문의 내용이 섞이는 것을 방지한다.

**vision_analyst_node의 페이지 선택 로직**

그림·표는 텍스트가 해당 요소를 참조하는 페이지 **앞뒤**에 위치하는 경우가 많다. 따라서 local_rag 검색 결과의 페이지 번호를 기준으로 ±1 페이지를 후보로 확장한 후 최대 5장을 Vision 모델에 전달한다.

---

### `src/rag/parser.py`

LlamaParse JSON 모드 호출과 pdf2image 페이지 이미지 생성을 담당.

**LlamaParse JSON 모드 선택 이유**

| 모드 | page_number 유지 | element_type 구분 |
|------|-----------------|------------------|
| 마크다운 | ✗ (손실됨) | ✗ |
| **JSON** | ✓ | ✓ (text/table/image) |

Vision 분석 시 "몇 번째 페이지 이미지를 분석할지" 결정하려면 `page_number`가 필수다. 마크다운 모드는 이 정보가 없어 이미지-텍스트 매핑이 불가능하다.

**`asyncio.gather()` 병렬 실행**

```python
json_result, page_images = await asyncio.gather(
    parse_pdf_with_llamaparse(pdf_path),  # 네트워크 I/O (LlamaParse API)
    generate_page_images(pdf_path),       # CPU 집약 (pdf2image PDF 렌더링)
)
```

두 작업은 서로 독립적이므로 순차 실행할 이유가 없다. 병렬 실행으로 인제스트 시간을 최대 절반으로 단축한다.

**`run_in_executor()` 사용 이유**

`pdf2image`의 `convert_from_path()`는 Poppler가 PDF를 픽셀 단위로 렌더링하는 **CPU 집약 작업**이다. CPU 바운드 블로킹 함수를 async 이벤트 루프에서 직접 호출하면 렌더링이 끝날 때까지 루프 전체가 멈춘다. `loop.run_in_executor(None, ...)`으로 스레드 풀에 위임하면 이벤트 루프는 다른 coroutine을 계속 처리할 수 있다. PIL의 `image.save()`도 같은 이유로 executor에서 실행한다.

---

### `src/rag/chunker.py`

Parent/Child 계층 청킹과 문서 영속화를 담당.

**Parent/Child 청킹 전략**

```
전체 문서
    │ parent_splitter (8000 chars ≈ 2000 tokens)
    ▼
Parent chunk: 섹션 단위 문맥 → data/parent_store/ (JSON 파일)
    │ child_splitter (1600 chars ≈ 400 tokens)
    ▼
Child chunk: 고밀도 벡터 검색용 → ChromaDB
```

짧은 Child chunk로 정밀하게 매칭하고, 매칭 후에는 넓은 Parent chunk의 문맥을 반환한다. Child만 저장하면 검색 정밀도는 높지만 답변 생성에 필요한 문맥이 부족하고, Parent만 저장하면 문맥은 풍부하지만 관계없는 내용까지 포함될 수 있다.

**`DocumentFileStore` Adapter 패턴**

| 인터페이스 | 타입 |
|-----------|------|
| `LocalFileStore` | `BaseStore[str, bytes]` |
| `ParentDocumentRetriever.docstore` 기대값 | `BaseStore[str, Document]` |

타입 불일치를 `DocumentFileStore` 래퍼로 해결한다. `mset`에서 Document를 JSON으로 직렬화하여 bytes로 저장하고, `mget`에서 bytes를 역직렬화하여 Document를 반환한다.

**시각 요소 placeholder 저장 이유**

텍스트가 없는 이미지·표 요소를 버리면 ChromaDB에 해당 페이지 기록이 전혀 남지 않는다. Vision Router 2단계에서 `element_type`이 "table"/"image"/"figure"인 청크를 확인하는 로직이 동작하지 않게 된다.

```python
# 텍스트 없는 시각 요소 → placeholder로 대체하여 저장
if elem["element_type"] in ("image", "figure"):
    text = f"[그림 - 페이지 {elem['page_number']}]"
elif elem["element_type"] == "table":
    text = f"[표 - 페이지 {elem['page_number']}]"
```

---

### `src/rag/vectorstore.py`

ChromaDB 초기화와 임베딩 모델 팩토리 함수.

**Persistent 모드**

ChromaDB를 InMemory로 사용하면 서버 재시작 시 모든 벡터가 사라지고 재인제스트가 필요하다. `persist_directory`를 지정하면 `data/chroma_db/`에 파일로 영속화되어 재시작 후에도 데이터가 유지된다.

**단일 컬렉션 전략**

모든 PDF를 하나의 컬렉션 `"thesistand"`에 저장하고 `metadata.source` 필드로 논문을 구분한다. 논문마다 별도 컬렉션을 만들면 컬렉션 수가 늘어날수록 관리가 복잡해지고, ChromaDB 클라이언트도 논문 수만큼 생성해야 한다.

---

### `src/rag/registry.py`

인제스트 이력을 `data/documents.json`에 영속 저장하는 레지스트리.

**JSON 파일 선택 이유**

수십~수백 개 수준의 문서 목록을 저장하는 데 PostgreSQL 같은 별도 DB는 과도하다. JSON 파일로 충분하며 인프라 의존성을 최소화한다.

**`get_latest()` 편의 기능**

`/ask` 호출 시 `pdf_hash`를 생략하면 `ingested_at` 타임스탬프 기준으로 가장 최근 인제스트 문서를 자동으로 사용한다. 논문 하나를 작업하는 일반적인 상황에서 매 요청마다 hash를 명시하지 않아도 된다.

**`ingested_at` 갱신 전략**

동일 `pdf_hash`를 재업로드하면 기존 레코드를 삭제하고 새 타임스탬프로 재등록한다. 이를 통해 `get_latest()`가 항상 실제 최신 문서를 반환한다.

---

### `src/main.py`

FastAPI 앱 진입점. HTTP 엔드포인트와 오케스트레이션 담당.

**SHA-256 파일 해시**

파일 내용을 기반으로 식별하므로 파일명이 달라도 내용이 같으면 동일 문서로 처리한다. 반대로 이름이 같더라도 내용이 다르면 별개 문서로 처리된다.

**이미지 폴더 존재 여부로 중복 판정**

```python
existing_images = sorted(image_dir.glob("page_*.png")) if image_dir.exists() else []
if existing_images:
    return IngestResponse(status="already_exists", ...)
```

ChromaDB 조회 없이 파일 시스템만 확인하여 LlamaParse API + 임베딩 비용을 절약한다. 이 두 작업이 인제스트의 가장 비싼 부분이기 때문이다.

**Lifespan context manager**

FastAPI 시작 시 ChromaDB를 미리 초기화해 두면 첫 요청에서 초기화 지연이 없다. 또한 애플리케이션 종료 시 정리 작업을 `yield` 이후에 배치할 수 있다.

**SSE 스트리밍 (`/ask/stream`)**

`astream_events`로 LangGraph 각 노드의 `on_chain_start`/`on_chain_end` 이벤트를 실시간 전달한다. 클라이언트는 어느 노드가 실행 중인지 즉시 알 수 있어 긴 처리 시간에도 UX가 개선된다.

```
data: {"event": "node_start", "node": "local_retriever"}
data: {"event": "node_start", "node": "web_searcher"}
data: {"event": "node_end",   "node": "local_retriever"}
data: {"event": "node_end",   "node": "web_searcher"}
data: {"event": "node_start", "node": "vision_router"}
...
data: {"event": "final_answer", "answer": "..."}
data: [DONE]
```

**Pydantic 모델 사용 범위**

State는 TypedDict를 사용하지만, HTTP 경계(요청/응답)에서는 Pydantic(`AskRequest`, `AskResponse`, `IngestResponse`)을 사용한다. 외부 입력만 검증하고 내부 State는 가볍게 유지하는 "경계에서만 Pydantic" 패턴이다.

---

## 5. 데이터 경로

```
data/
├── pdfs/
│   └── {sha256_hash}.pdf          # 업로드된 PDF (내용 기반 식별)
│
├── images/
│   └── {sha256_hash}/
│       ├── page_001.png           # 200 DPI PNG
│       ├── page_002.png
│       └── ...
│
├── chroma_db/                     # ChromaDB Persistent 모드
│   └── (ChromaDB 내부 파일들)     # Child chunk 벡터 저장
│
├── parent_store/                  # Parent chunk JSON 파일
│   └── {uuid}                     # Document(page_content, metadata) — 확장자 없음, 내용은 JSON
│
└── documents.json                 # 인제스트 이력 레지스트리
    # 예시:
    # [
    #   {
    #     "pdf_hash": "a3f9...",
    #     "filename": "attention_is_all_you_need.pdf",
    #     "page_count": 15,
    #     "ingested_at": "2025-08-01T12:00:00+00:00"
    #   }
    # ]
```

---

## 6. 테스트 전략

### 파일 구조

```
tests/
├── agents/
│   ├── test_state.py      # ContextChunk, AgentState 타입 검증
│   ├── test_graph.py      # 그래프 빌드, 엣지 구조 검증
│   └── test_nodes.py      # 각 노드 함수 단위 테스트
├── rag/
│   ├── test_parser.py     # 파싱 함수, elements 추출 검증
│   ├── test_chunker.py    # DocumentFileStore, Parent/Child 분할 검증
│   └── test_vectorstore.py # 벡터스토어 초기화 검증
└── test_main.py           # FastAPI 엔드포인트 통합 테스트
```

`src/`와 `tests/`가 동일한 디렉토리 구조를 미러링한다. 어떤 모듈의 테스트가 어디 있는지 즉시 알 수 있다.

### Mock 우선 원칙

외부 API(OpenAI, Tavily, LlamaParse)는 전량 `unittest.mock.AsyncMock`으로 대체한다.
- 실제 API 키 없이 CI에서 테스트 실행 가능
- API 비용 없음
- 네트워크 없이도 결정적(deterministic) 테스트 가능

```python
# 예시: OpenAI 클라이언트 mock
with patch("src.agents.nodes._get_openai_client") as mock_client:
    mock_client.return_value.chat.completions.create = AsyncMock(
        return_value=mock_response
    )
```

### conftest.py 미사용 이유

파일이 7개인 현재 규모에서 전역 fixture는 불필요하다. 각 테스트 파일에서 직접 fixture와 mock을 정의하는 것이 더 명확하고 읽기 쉽다. 공통 fixture가 5개 이상 반복되거나 DB 연결 같은 무거운 설정이 필요해질 때 `conftest.py`를 도입한다.

### `asyncio_mode = "auto"`

`pyproject.toml`의 `[tool.pytest.ini_options]` 섹션에 설정하면 `async def test_*` 함수에 `@pytest.mark.asyncio` 데코레이터 없이도 자동으로 비동기 실행된다. 모든 노드가 `async def`이므로 이 설정이 없으면 모든 테스트 함수에 데코레이터를 붙여야 한다.

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

---

## 관련 문서

- [production-roadmap.md](./production-roadmap.md) — 프로덕션 전환 로드맵 (멀티유저, 프론트엔드, 인프라)
- [library.md](./library.md) — 개발 초기 설정 및 라이브러리 설치 기록
