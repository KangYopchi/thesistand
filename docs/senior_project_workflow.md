# 프로젝트 워크플로우 — Ingest & Query 파이프라인

> **읽는 법**: 각 단계 옆 `(파일:줄)` 표기로 코드를 바로 찾을 수 있다.
> "무엇을 하는가"보다 **"왜 이렇게 설계했는가"** 를 중심으로 작성했다.

---

## 1. Ingest 파이프라인 — `POST /ingest`

### 전체 흐름

```
POST /ingest (UploadFile)
  │
  ├─ SHA-256 해싱 → 중복 검사
  │     이미 처리된 파일이면 → IngestResponse(status="already_exists") 즉시 반환
  │
  └─ [신규] PDF 저장 → ingest_graph.ainvoke()
        │
        └─ ingest_node
              │
              ├─ asyncio.gather() ──┬─ LlamaParse  (네트워크 I/O, async)
              │    [병렬 실행]       └─ pdf2image   (CPU → run_in_executor)
              │
              ├─ extract_elements_with_page_numbers()
              │     list[dict] → list[ParsedElement]
              │     각 요소: {page_number, text, element_type}
              │
              ├─ elements_to_documents()
              │     list[ParsedElement] → list[Document]
              │     빈 image/table → placeholder("[그림 - 페이지 N]")로 저장
              │
              ├─ retriever.aadd_documents()
              │     Parent (8000 chars) → DocumentFileStore (data/parent_store/)
              │     Child  (1600 chars) → ChromaDB          (벡터 임베딩)
              │
              └─ return {"image_dir": "data/images/{hash}", "contexts": []}
                    LangGraph가 AgentState에 병합(merge)
```

---

### 단계별 상세

#### Step 1. 앱 시작 — lifespan (`main.py:58`)

```python
get_vectorstore(embeddings=get_embeddings())
```

앱이 뜰 때 ChromaDB 커넥션을 초기화한다.
`lifespan`이 없으면 첫 요청에서 커넥션이 생성되어 첫 번째 요청만 느려진다.
FastAPI의 startup/shutdown을 `@asynccontextmanager`로 처리하는 표준 패턴이다.

---

#### Step 2. SHA-256 해싱 + 중복 검사 (`main.py:106-118`)

```python
pdf_hash = hashlib.sha256(file_bytes).hexdigest()  # 64자리 hex 문자열

existing_images = sorted(image_dir.glob("page_*.png")) if image_dir.exists() else []
if existing_images:  # 폴더 존재가 아니라 이미지 파일 존재 여부로 판단
    return IngestResponse(status="already_exists", ...)
```

**왜 SHA-256인가?**
파일명이 달라도 내용이 같으면 동일한 해시가 나온다.
이 64자리 hex 문자열이 hash 기반 경로의 기준이 된다:
`data/pdfs/{hash}.pdf`, `data/images/{hash}/`

단, `data/parent_store/`는 hash를 포함하지 않는 **공유 폴더**다. (`chunker.py:76`)
모든 PDF의 부모 문서가 한 폴더에 함께 저장된다.

**왜 이미지 파일로 중복을 판단하는가?**
ChromaDB에 들어갔는지 확인하는 것보다 단순하다.
파이프라인의 **마지막 산출물**(이미지 파일)이 있으면 전체가 완료된 것으로 간주한다.
폴더만 있고 `page_*.png`가 없으면 완료로 보지 않는다는 점에 주목.

---

#### Step 3. ingest 그래프 실행 (`main.py:125-135`, `graph.py:27-40`)

```python
graph = build_ingest_graph()   # START → ingest → END (단순 선형)
result = await graph.ainvoke({
    "question": "", "pdf_hash": pdf_hash,
    "image_dir": "", "contexts": [], ...
})
```

단순한 선형 그래프인데도 LangGraph를 쓰는 이유는
Query 파이프라인과 **동일한 AgentState 구조**를 재사용하기 위함이다.

---

#### Step 4. `ingest_node` 내부 (`nodes.py:74-107`)

##### 4-1. asyncio.gather — 파싱 + 이미지 생성 병렬 실행 (`parser.py:148-152`)

```python
gathered = await asyncio.gather(
    parse_pdf_with_llamaparse(pdf_path),  # 네트워크 I/O (LlamaParse Cloud API)
    generate_page_images(pdf_path),       # CPU → loop.run_in_executor (스레드풀)
)
json_result, page_images = gathered
```

둘 다 수십 초가 걸리는 작업이다.
LlamaParse는 네트워크 대기 중 이벤트 루프를 반환(await)하고,
pdf2image는 CPU 바운드라 `run_in_executor`로 스레드풀에 오프로드한다. (`parser.py:117-121`)
`asyncio.gather`가 두 작업을 동시에 시작해 총 대기 시간을 줄인다.

##### 4-2. element 추출 (`parser.py:62-88`)

```python
elements = extract_elements_with_page_numbers(json_result)
# list[ParsedElement]: [{page_number: int, text: str, element_type: str}, ...]
```

LlamaParse JSON 모드는 `pages[].items[]` 구조로 결과를 반환한다.
각 item의 `type` 필드가 `"text"`, `"table"`, `"image"` 등으로 구분된다.
이 `element_type`이 나중에 vision_router 2단계 판단에 사용된다.

##### 4-3. Parent/Child 청킹 + 저장 (`nodes.py:92-103`, `chunker.py`)

```
list[ParsedElement]
  → elements_to_documents()
      텍스트 없는 image/table → "[그림 - 페이지 N]" placeholder로 저장  (chunker.py:108-115)
      ※ 버리면 ChromaDB에 시각 요소 기록이 없어져 vision_router 2단계가 작동하지 않음
  → retriever.aadd_documents()
      Parent splitter: 8000 chars (≈2000 tokens) → DocumentFileStore (data/parent_store/)
      Child splitter:  1600 chars (≈400 tokens)  → ChromaDB (벡터 검색용)
```

**왜 Parent/Child 구조인가?**
Child로 검색해 정확한 위치를 찾고, Parent를 반환해 충분한 문맥을 제공한다.
Child만 반환하면 LLM에 문맥이 부족하고, Parent만 저장하면 검색 정밀도가 낮아진다.

**왜 DocumentFileStore인가?** (`chunker.py:27-71`)
`LocalFileStore`는 `bytes`를 저장하는데, `ParentDocumentRetriever`는 `Document` 객체를 기대한다.
`DocumentFileStore`는 그 사이의 JSON 직렬화/역직렬화 래퍼다.
`InMemoryStore` 대신 사용하는 이유: 서버 재시작 후에도 부모 문서가 유지된다.

##### 4-4. 반환 + AgentState 병합 (`nodes.py:106-107`)

```python
image_dir = str(IMAGE_DIR / pdf_hash)   # "data/images/{hash}"
return {"image_dir": image_dir, "contexts": []}
```

노드는 **AgentState 전체가 아니라 dict를 반환**한다.
LangGraph가 이 dict를 AgentState에 병합(merge)한다.
`contexts: []`를 명시 반환하는 이유: AgentState 스키마를 명시적으로 준수하기 위함.
`operator.add` 리듀서는 리셋을 지원하지 않는다 — 빈 리스트를 반환하면 기존 값에 빈 리스트를 더하는 것(no-op)이다.
Ingest 초기 state가 이미 `contexts: []`이므로 실질적으로 "이 노드에서는 contexts를 변경하지 않는다"는 선언이다.

---

## 2. Query 파이프라인 — `POST /ask/stream`

### 전체 흐름

```
POST /ask/stream {question, pdf_hash}
  │
  ├─ image_dir 존재 확인 → 없으면 404
  │
  └─ query_graph.astream_events(version="v2")
        │
        ├─ fan-out (START → 두 노드 동시 실행)
        │     ├─ local_retriever_node : question → ChromaDB 벡터 검색 → Parent 청크 반환
        │     └─ web_searcher_node    : question → Tavily API 검색
        │           ↓ 두 결과 모두 contexts에 기록됨
        │           operator.add 리듀서가 두 리스트를 자동 합산 (state.py:44)
        │
        ├─ fan-in → vision_router_node (3단계 비용 최적화)
        │     1단계 (무료): 질문 키워드 확인 ("표", "figure" ...)  → NEED_VISION
        │     2단계 (무료): contexts의 element_type 확인           → NEED_VISION
        │     3단계 (유료): gpt-5-mini 호출 (1·2단계 판단 불가 시) → NEED_VISION or NO_VISION
        │
        ├─ [조건부] vision_analyst_node
        │     참조 페이지 ±1 이미지 → base64 인코딩 → gpt-5-mini Vision
        │
        └─ synthesis_node
              contexts + vision_result → gpt-5-mini → final_answer
              → SSE 이벤트로 클라이언트에 전달
```

---

### 단계별 상세

#### Step 1. image_dir 존재 확인 (`main.py:191-193`)

```python
image_dir = IMAGE_DIR / request.pdf_hash
if not image_dir.exists():
    raise HTTPException(status_code=404, ...)
```

ingest 완료 여부를 image_dir로 판단한다. (Ingest Step 2와 같은 기준)

---

#### Step 2. fan-out — 두 노드 동시 실행 (`graph.py:63-68`)

```python
graph.add_edge(START, "local_retriever")  # START에서 두 노드로 동시 분기
graph.add_edge(START, "web_searcher")
graph.add_edge("local_retriever", "vision_router")  # 둘 다 완료해야 fan-in
graph.add_edge("web_searcher", "vision_router")
```

LangGraph에서 `START`에 두 엣지를 연결하면 자동으로 병렬 실행된다.
두 노드가 모두 `contexts`에 결과를 쓰려 하므로 덮어쓰기 충돌 가능성이 있다.
`state.py:44`의 `contexts: Annotated[list[ContextChunk], operator.add]` 리듀서가
두 리스트를 자동으로 합산해서 충돌을 방지한다.

**local_retriever_node** (`nodes.py:110-147`)
```python
docs = await retriever.ainvoke(question)
# ⚠️ pdf_hash로 필터링하지 않는다.
# 여러 PDF가 ingest된 경우 모든 PDF의 청크가 검색 대상이 된다.
```

**web_searcher_node** (`nodes.py:150-188`)
```python
response = await client.search(query=question, max_results=5, search_depth="advanced")
# 실패 시 빈 리스트 반환 → 전체 파이프라인이 멈추지 않는다
```

---

#### Step 3. vision_router_node — 3단계 비용 최적화 (`nodes.py:191-246`)

> ⚠️ 코드 docstring에는 "2단계 라우터"라고 쓰여 있지만 실제 구현은 3단계다. (`nodes.py:192`)

| 단계 | 방법 | 비용 | 조건 |
|------|------|------|------|
| 1단계 | 질문 키워드 확인 (`_VISION_KEYWORDS`) | 0원 | "표", "figure" 등 포함 시 즉시 NEED_VISION |
| 2단계 | contexts의 `element_type` 메타데이터 확인 | 0원 | "table"/"image"/"figure" 포함 시 NEED_VISION |
| 3단계 | gpt-5-mini 호출 | 유료 | 1·2단계 판단 불가 시에만 |

**2단계가 작동하는 조건**: ChromaDB에 image/table placeholder가 저장되어 있어야 한다.
Ingest Step 4-3의 placeholder 로직이 여기에 연결된다.

---

#### Step 4. [조건부] vision_analyst_node (`nodes.py:260-365`)

```python
# 참조 페이지 ±1 확장 (그림/표는 텍스트 참조 페이지 앞뒤에 위치하는 경우가 많다)
for p in ref_pages:
    expanded.update([p - 1, p, p + 1])
visual_pages = sorted(p for p in expanded if p > 0)[:5]  # 최대 5장

# 이미지 파일명 포맷: page_001.png (3자리 zero-padding)
candidate_paths = [image_dir_path / f"page_{p:03d}.png" for p in visual_pages]
```

이미지를 base64로 인코딩해서 gpt-5-mini Vision으로 전달한다.
최대 5장으로 제한하는 이유: 토큰 비용 및 컨텍스트 윈도우 한계.

---

#### Step 5. synthesis_node + SSE 스트리밍 (`nodes.py:368-434`, `main.py:195-221`)

```python
async for event in graph.astream_events(initial_state, version="v2"):
    kind = event.get("event", "")  # "on_chain_start", "on_chain_end"
    name = event.get("name", "")   # 노드명: "local_retriever", "synthesis" 등
    # on_chain_end의 output.final_answer에서 최종 답변 추출
```

`astream_events(version="v2")`: LangGraph 이벤트 스키마의 두 번째 버전.
`version="v1"`과 달리 중첩 그래프에서도 이벤트가 올바르게 전파된다.
`on_chain_start`/`on_chain_end` 이벤트를 SSE로 클라이언트에 전달하여
각 노드의 진행 상황을 실시간으로 알린다.

---

## 코드를 읽는 시각 — 시니어의 관점

### 1. "무엇"보다 "왜"를 먼저 묻는다

주니어는 `asyncio.gather()`를 보고 "비동기로 동시에 실행"이라고 읽는다.
시니어는 거기서 멈추지 않고 묻는다:

- "왜 이 두 작업을 동시에 돌렸는가?" → 둘 다 수십 초짜리 I/O/CPU 작업이라서
- "왜 pdf2image는 `run_in_executor`로 감쌌는가?" → CPU 바운드 작업은 async만으로는 이벤트 루프를 블록킹하기 때문
- "왜 Document placeholder를 버리지 않고 저장했는가?" → vision_router 2단계가 element_type 메타데이터를 참조하기 때문

설계의 "왜"를 이해하면 코드를 수정할 때 무엇을 건드리면 안 되는지가 보인다.

---

### 2. 데이터 흐름을 타입으로 추적한다

각 단계에서 **어떤 타입이 흘러가는지** 머릿속에 그리는 것이 코드 이해의 핵심이다.

```
bytes (UploadFile)
  → str (pdf_hash, 64자 hex)
  → ParseResult: {elements: list[ParsedElement], page_images: dict[int, Path]}
  → list[Document] (LangChain, 메타데이터 포함)
  → ChromaDB (child 벡터) + DocumentFileStore (parent 문서)
  → dict {"image_dir": str, "contexts": []}
  → AgentState (LangGraph merge)
  → IngestResponse (Pydantic → JSON → HTTP 응답)
```

타입이 변환되는 지점이 **설계 경계**다. 경계를 찾으면 책임 범위가 보인다.

---

### 3. 시스템 경계(Boundary)를 먼저 파악한다

코드를 처음 볼 때 어디서 외부 시스템과 만나는지 찾는다.

| 경계 | 위치 | 타입 변환 |
|------|------|-----------|
| HTTP ↔ 내부 | `main.py` 엔드포인트 | HTTP 요청 → Pydantic 모델 |
| RAG ↔ Agent | `ingest_node` 반환 | `ParseResult` → `AgentState` |
| Agent ↔ HTTP | `main.py` 응답 | `AgentState` → `AskResponse` |
| 내부 ↔ LLM | `nodes.py` 각 노드 | `str` → OpenAI API → `str` |
| 내부 ↔ 벡터DB | `vectorstore.py` | `list[Document]` → ChromaDB |

경계를 알면 어디서 오류가 날지, 어디에 테스트가 필요한지 알 수 있다.

---

### 4. 실패 경로를 먼저 읽는다

정상 흐름보다 예외 처리가 설계 의도를 더 잘 드러낸다.

- `nodes.py:171-173`: 웹 검색 실패 시 빈 리스트 반환 → 파이프라인이 멈추지 않는다
- `nodes.py:275-282`: `image_dir`가 없으면 vision_analyst가 메시지를 반환하고 조용히 종료
- `nodes.py:241-243`: gpt-5-mini 판단 실패 시 기본값 `NO_VISION`으로 폴백

이 패턴은 의도적이다. **어떤 하나의 실패가 전체 응답을 막으면 안 된다**는 설계 원칙.

---

### 5. 의도적으로 연결된 코드를 찾는다

서로 다른 파일에 있지만 하나의 의도로 연결된 코드들이 있다.
이것을 찾는 능력이 실무 역량이다.

**예시 — placeholder와 vision_router의 연결**

```
chunker.py:108  빈 image/table → "[그림 - 페이지 N]" placeholder 저장
      ↓
nodes.py:212    vision_router 2단계: contexts의 element_type 확인
      ↓
nodes.py:213    chunk.get("element_type") in ("table", "image", "figure")
```

chunker.py가 placeholder를 저장하지 않으면 vision_router 2단계가 작동하지 않는다.
파일이 다르더라도 설계 의도는 하나다.

---

## 알려진 한계 — 이해하고 개선할 부분

### 1. local_retriever가 pdf_hash로 필터링하지 않는다 (`nodes.py:130`)

```python
docs = await retriever.ainvoke(question)  # 모든 PDF 청크를 검색 대상으로 삼는다
```

여러 PDF를 ingest하면 PDF A에 대한 질문에 PDF B의 청크가 섞여 반환될 수 있다.
ChromaDB의 메타데이터 필터 기능으로 해결 가능하다:
```python
# 개선 방향 (현재 미구현)
retriever.search_kwargs = {"filter": {"source": pdf_hash}}
```

### 2. vision_router_node docstring 오류 (`nodes.py:192`)

docstring에 "2단계 라우터"라고 쓰여 있지만 실제 구현은 3단계다.
코드와 주석이 불일치하면 다음 사람이 코드를 읽을 때 혼란을 준다.

---

## 앞으로의 과제

1. `chunker.py` — `RecursiveCharacterTextSplitter` 파라미터(chunk_size, chunk_overlap)가 검색 품질에 미치는 영향 실험
2. `parser.py` — LlamaParse JSON 모드의 실제 출력 구조 직접 확인 (`pages[].items[].type` 필드값)
3. `state.py` — `operator.add` 리듀서를 제거하면 병렬 노드에서 어떤 오류가 발생하는지 직접 확인
4. local_retriever의 pdf_hash 필터링 미구현 문제 — ChromaDB 메타데이터 필터 추가 시도
