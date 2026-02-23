# `std_project_workflow.md` 수정 가이드라인

> **목적**: `std_project_workflow.md`를 `senior_project_workflow.md` 수준으로 직접 발전시키는 단계별 가이드.
> 시니어 버전을 복사하지 말 것. 각 단계에서 코드를 직접 열어 답을 찾고 자기 말로 쓴다.

---

## Step 1. 코드 참조(file:line) 추가

**목표**: 현재 문서의 각 줄이 코드 어디에 해당하는지 직접 찾아서 옆에 표기한다.

**예시**:
```
Before: hashing을 진행한다. (16진수) 변환된 값으로 중복 검사 진행
After:  hashing을 진행한다. (16진수) 변환된 값으로 중복 검사 진행  (main.py:106-118)
```

**찾아야 할 참조 목록**:

| 현재 문장 | 찾아야 할 파일 |
|-----------|---------------|
| `@app.post("/ingest")` 실행 | `main.py` |
| hashing + 중복 검사 | `main.py` |
| ingest Graph 생성 + ainvoke | `main.py`, `graph.py` |
| parse_pdf + asyncio.gather 병렬 실행 | `nodes.py`, `parser.py` |
| Parent/Child 청킹 + ChromaDB 저장 | `nodes.py`, `chunker.py` |
| image_dir 업데이트 | `nodes.py` |
| PDF 이미지 존재 확인 (Query 시작) | `main.py` |
| local_retriever + web_search 동시 실행 | `graph.py`, `nodes.py` |
| operator.add 리듀서 | `state.py` |
| vision router | `nodes.py`, `graph.py` |
| vision analyst | `nodes.py` |
| synthesis + 답변 전달 | `nodes.py`, `main.py` |

**완료 기준**: 모든 단계에 `(파일:줄)` 표기가 붙어있다.

---

## Step 2. "왜(why)" 질문 추가

**목표**: 각 단계 아래에 "왜 이 방법을 썼는가?"를 자기 말로 한 줄 이상 쓴다.

**예시**:
```
Before: hashing을 진행한다. (main.py:106)

After:  SHA-256 hashing → hexdigest() → 64자리 hex 문자열  (main.py:106)
        왜 SHA-256인가? → 파일명이 달라도 내용이 같으면 동일한 해시가 나온다.
                          이 값이 data/pdfs/{hash}.pdf, data/images/{hash}/ 경로의 기준이 된다.
```

**직접 답해야 할 질문 목록**:

```
Q1.  왜 SHA-256 hexdigest인가?
Q2.  왜 ChromaDB 조회가 아니라 이미지 파일(page_*.png) 존재 여부로 중복을 판단하는가?
Q3.  왜 단순한 선형 파이프라인인데도 LangGraph를 쓰는가?
Q4.  왜 LlamaParse와 pdf2image를 asyncio.gather로 묶었는가?
Q5.  왜 pdf2image는 run_in_executor로 감쌌는가?
Q6.  왜 Parent/Child 청크를 분리하는가?
Q7a. 왜 InMemoryStore 대신 LocalFileStore를 쓰는가?      ← 영속성 문제
Q7b. 왜 LocalFileStore를 DocumentFileStore로 감쌌는가?  ← 타입 불일치 문제
Q8.  왜 노드는 AgentState 전체가 아니라 dict를 반환하는가?
Q9.  왜 vision_router를 3단계로 나눴는가?
Q10. 왜 web_search 실패 시 예외를 올리지 않고 빈 리스트를 반환하는가?
Q11. ingest_node가 {"contexts": []}를 반환하는 이유는 무엇인가?
     이것이 contexts를 초기화(reset)하는가?  ← operator.add 동작 방식을 확인하라.
```

> ⚠️ Q11 힌트:
> `operator.add`는 더하는 것만 가능하고 리셋은 지원하지 않는다.
> `operator.add([], [])` = `[]` (no-op) — 빈 리스트를 더해도 기존 값은 바뀌지 않는다.
> Ingest 초기 state가 이미 `contexts: []`이므로 `{"contexts": []}` 반환은 실질적으로 no-op다.
> 정확한 이유는 "초기화"가 아니라 **AgentState 스키마 명시적 준수**다.
> `state.py:44`를 보면 왜 이 구분이 중요한지 알 수 있다.

> ⚠️ Q7이 두 개(Q7a, Q7b)인 이유:
> `DocumentFileStore`는 `InMemoryStore`의 대안이 아니다.
> 영속성 문제(왜 LocalFileStore) 와 타입 변환 문제(왜 DocumentFileStore 래퍼) 는 별개다.
> `chunker.py:27-71`과 `chunker.py:162`를 함께 읽어야 두 질문 모두 답할 수 있다.

**완료 기준**: 12개 질문에 코드를 보지 않고 자기 말로 답할 수 있다.

---

## Step 3. 타입(type) 흐름 추가

**목표**: 각 단계 사이에서 어떤 타입의 데이터가 흘러가는지 추적해서 추가한다.

**Ingest 파이프라인 타입 추적**:

| 단계 사이 | 흐르는 타입 | 찾을 위치 |
|-----------|------------|-----------|
| HTTP 요청 진입 | `UploadFile` | `main.py` 함수 시그니처 |
| 해싱 후 | `str` (64자 hex) | `main.py:106` |
| `parse_pdf()` 반환 | `ParseResult` | `parser.py:28-34` |
| `extract_elements_with_page_numbers()` 반환 | `list[ParsedElement]` | `parser.py:62` |
| `elements_to_documents()` 반환 | `list[Document]` | `chunker.py:90` |
| `ingest_node` 최종 반환 | `dict` | `nodes.py:107` |
| LangGraph merge 후 | `AgentState` | `state.py` |
| HTTP 응답 | `IngestResponse` (JSON) | `main.py:47-51` |

**Query 파이프라인 타입 추적**:

| 단계 사이 | 흐르는 타입 | 찾을 위치 |
|-----------|------------|-----------|
| HTTP 요청 진입 | `AskRequest` (Pydantic) | `main.py:37-39` |
| local_retriever_node 반환 | `dict` (`contexts: list[ContextChunk]`) | `nodes.py:147` |
| web_searcher_node 반환 | `dict` (`contexts: list[ContextChunk]`) | `nodes.py:188` |
| fan-in 후 merged contexts | `list[ContextChunk]` (두 결과 합산) | `state.py:44` |
| vision_router_node 반환 | `dict` (`vision_result: str`) | `nodes.py:210, 218, 246` |
| synthesis_node 반환 | `dict` (`final_answer: str`) | `nodes.py:434` |
| HTTP 응답 | `StreamingResponse` (SSE) | `main.py:223-226` |

**완료 기준**: Ingest와 Query 두 흐름 모두 타입이 끊기지 않고 연결된다.

---

## Step 4. 설계 연결고리 찾기

**목표**: 서로 다른 파일에 있지만 하나의 의도로 연결된 코드를 찾아 문서에 기록한다.

시니어 버전의 힌트를 보기 전에 아래 질문을 직접 추적해본다.

**찾아야 할 연결고리 3개**:

```
연결고리 1.
  chunker.py에서 텍스트가 없는 image/table 요소를 버리지 않고 placeholder로 저장한다.
  → 왜 버리지 않는가? nodes.py의 vision_router_node에서 이유를 찾아라.

연결고리 2.
  parser.py에서 image_dir 폴더를 생성한다.  ← 찾아야 할 줄: parser.py:115
  → 이 폴더를 Query 파이프라인의 어느 코드가 참조하는가? main.py에서 찾아라.

  ⚠️ 주의: image_dir를 생성하는 주체는 main.py가 아니다.
            parser.py:115의 output_dir.mkdir()가 실제로 폴더를 만든다.
            main.py는 ingest_node의 결과를 읽을 뿐이다.

연결고리 3.
  ingest_node가 {"image_dir": ...}를 반환한다.  ← nodes.py:107
  → vision_analyst_node는 이 값을 어떻게 받아서 쓰는가? nodes.py에서 찾아라.
```

**완료 기준**: 3개 연결고리를 코드 줄 번호와 함께 설명할 수 있다.

---

## Step 5. 알려진 한계 섹션 추가

**목표**: 현재 구현에서 개선이 필요한 부분을 스스로 발견해서 기록한다.

**찾아야 할 한계 2개**:

```
한계 1.
  nodes.py:130을 보라.
  retriever.ainvoke(question) 에 어떤 필터가 있는가?
  PDF를 2개 ingest하면 각 PDF에 대한 질문이 어떻게 처리되는가?

한계 2.
  nodes.py:192의 docstring을 보라.
  그리고 그 함수의 실제 구현 단계 수를 세어라.
  무엇이 불일치하는가?
```

**완료 기준**: 두 가지 한계를 발견하고, 각각 한 줄로 개선 방향을 쓸 수 있다.

---

## 단계 순서 요약

```
Step 1  코드 참조 추가        각 줄에 (파일:줄) 붙이기          → 완료 기준: 모든 줄에 참조 표기
Step 2  "왜" 추가             12개 질문에 자기 말로 답하기       → 완료 기준: 코드 없이 답 가능
Step 3  타입 흐름 추가        Ingest + Query 타입 테이블 채우기  → 완료 기준: 두 흐름 모두 연결
Step 4  설계 연결고리 찾기    3개 cross-file 연결 발견하기       → 완료 기준: 줄 번호와 함께 설명
Step 5  알려진 한계 추가      2개 한계 발견 + 개선 방향          → 완료 기준: 개선 방향 한 줄 작성
```

각 Step이 끝날 때마다 `senior_project_workflow.md`와 비교해서
내용이 같아야 하는 게 아니라 **빠진 관점이 없는지** 확인하는 용도로 사용한다.
