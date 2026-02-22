# PRD - ThesiStand: 논문 학습을 돕는 에이전트 (v2.0)

> **v1.0 → v2.0 주요 변경사항**: 전문가 검토를 통해 State 설계 개선, Graph 구조 분리, Vision Router 로직 구체화, API 명세 추가

---

## 1. 프로젝트 개요

- **프로젝트명:** ThesiStand - 논문 학습을 돕는 에이전트
- **목표:** 논문의 계층 구조(Parent-Child)를 유지하는 고성능 RAG 시스템을 구축하고, 비동기 병렬 처리와 비전 모델을 결합하여 복잡한 테이블과 방법론을 정확히 해석함.
- **핵심 가치:** 1년 차 개발자가 겪을 수 있는 병렬 상태 관리 오류 및 배포 환경 문제를 선제적으로 해결하며 실무형 AI 에이전트 개발 역량 습득.
- **타겟:** 논문을 읽고 분석하는 신입 개발자
- **목표 응답 시간:** 질문 후 답변 스트리밍 시작까지 3초 이내

---

## 2. 사용자 흐름 (User Flow)

```
1. PDF 업로드 (POST /ingest)
        ↓
2. "분석 중..." 로딩 표시 (비동기 처리)
        ↓
3. 분석 완료 → 질문 입력창 활성화
        ↓
4. 질문 입력 (POST /query)
        ↓
5. 답변 스트리밍 출력
        ↓
6. 답변 하단에 출처 표시 (예: "📄 출처: 3페이지, 7페이지")
```

---

## 3. 핵심 기능 상세 (Features)

### 3.1. 정밀 메타데이터 파싱 (LlamaParse JSON Mode)

- **기능:** `LlamaParse`를 **JSON 결과 모드**로 호출하여 텍스트와 함께 각 요소의 `page_number`, **요소 타입(text / table / image)** 을 추출.
- **중요:** LlamaParse JSON 모드는 각 파싱 결과에 `type` 필드를 포함함. 이 메타데이터는 이후 Vision Router에서 AI 호출 없이 시각적 요소 여부를 판단하는 데 활용됨.
- **MVP 범위:** LlamaParse 단일 파싱 전략으로 진행. fallback(대체 파싱) 전략은 서비스 안정화 이후 Phase 2에서 추가.
	- **Phase 2 예정:** LlamaParse 실패 시 `PyMuPDF` 등 로컬 파서로 대체 (단, 품질 저하 있음을 명시)
- **중복 업로드 방지:** 논문 업로드 시 파일의 SHA-256 해시값을 계산하여 이미 처리된 논문이면 재파싱 없이 기존 데이터를 반환함.
	-  해시 값을 사용하면 파일명이 다른 같은 내용의 파일의 중복 여부를 확인할 수 있음

```python
import hashlib

def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()
```

### 3.2. 영속성 계층적 RAG (Persistent Hierarchical RAG)

- **기능:** `ParentDocumentRetriever`를 활용한 계층적 청킹 구현.
  - **Child Chunks:** 고밀도 벡터 검색용 (목표: 400 tokens, 실험을 통해 조정 예정)
  - **Parent Chunks:** 요약 및 문맥 제공용 (Section 단위)
- **Persistent Store 구현 방식:** `InMemoryStore` 대신 `LocalFileStore` + `create_kv_docstore` 조합으로 서버 재시작 후에도 부모 문서를 보존함.

```python
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

fs = LocalFileStore("./cache_dir")
store = create_kv_docstore(fs)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

- **주의사항:** `LocalFileStore`는 내부적으로 LangChain의 직렬화 방식을 사용함. 패키지 버전 업그레이드 시 역직렬화 오류가 발생할 수 있으므로, `uv lock`으로 의존성 버전을 고정하여 관리함.
- **청크 크기 실험 계획:** 200 / 400 / 600 토큰 세 가지 설정으로 검색 정확도를 비교 실험 후 최적값 선택. 결과는 README에 기록.

### 3.3. 안전한 비동기 병렬 에이전트 (LangGraph Reducer)

- **기능:** 논문 검색과 웹 검색을 병렬로 수행하여 응답 속도 최적화.
- **병렬 충돌 방지:** `Annotated[List[ContextChunk], operator.add]` 리듀서를 적용하여 병렬 노드 결과가 덮어씌워지지 않고 누적됨.
- **출처 추적:** 검색 결과를 단순 문자열이 아닌 `ContextChunk` 구조체로 저장하여, 최종 답변에서 "논문 몇 페이지에서 나온 내용인지" 추적 가능하게 함.

### 3.4. 2단계 Vision Router (비용 최적화)

- **기능:** 텍스트로 해석이 어려운 테이블/수식을 Vision 모델로 분석.
- **판단 로직 (2단계):**
  - **1단계 (비용 없음):** 사용자 질문 키워드 확인. "표", "그림", "figure", "table", "수식", "graph" 등 명시적 단어가 있으면 즉시 Vision 모드 진입.
  - **2단계 (LlamaParse 메타데이터 활용):** 검색된 청크의 `element_type` 메타데이터가 `"table"` 또는 `"image"`이면 Vision 모드 진입. 1, 2단계에서 판단이 안 된 **애매한 경우에만** LLM 호출.
- **이미지 전처리:** 토큰 비용 절감을 위해 해당 페이지 이미지에서 필요한 영역만 크롭(Crop)하여 전달.

### 3.5. Reranker (검색 품질 향상)

- **기능:** `Synthesis_Node` 전에 검색 결과의 관련도를 재평가하여 노이즈 제거.
- **구현:** 로컬에서 무료로 실행 가능한 `CrossEncoderReranker` 사용. (MVP 단계에서 비용 없이 품질 향상 가능)
- **Phase 2 예정:** 필요 시 Cohere Rerank API로 교체 검토.

---

## 4. 기술 스택 (Technical Stacks)

| 분류                 | 기술                       | 선택 이유                                          |
| ------------------ | ------------------------ | ---------------------------------------------- |
| Package Management | `uv`                     | 빠른 의존성 해석 및 lock-file로 버전 고정                   |
| API Framework      | `FastAPI`                | Streaming response 지원                          |
| Orchestration      | `LangGraph`              | Async StateGraph with Reducers                 |
| Vector DB          | `ChromaDB`               | Self-hosted persistent mode                    |
| PDF 파싱             | `LlamaParse` (JSON Mode) | 페이지 번호 및 요소 타입 메타데이터 추출                        |
| 임베딩 모델             | `bge-m3`                 | 다국어 지원, 로컬 실행 가능, 학술 텍스트에 강함                   |
| Reranker           | `CxrossEncoderReranker`  | 로컬 실행, 무료                                      |
| 웹 검색               | `Tavily API`             | LangGraph 통합 용이                                |
| Deployment         | `Docker`                 | Multi-stage build with `uv` virtualenv pathing |

> **임베딩 모델 참고:** `bge-m3` 로컬 실행 시 GPU 메모리가 필요할 수 있음. 배포 환경의 스펙에 따라 OpenAI 임베딩 API로 대체 검토.

---

## 5. 시스템 아키텍처 (LangGraph Workflow)

### Graph 분리 전략

Ingest(논문 저장)와 Query(질문 처리)는 **서로 다른 Graph로 분리**함.

- **Ingest Graph:** 논문 업로드 시 1회만 실행. 잦은 호출이 없으므로 동기 처리도 가능.
- **Query Graph:** 사용자가 질문할 때마다 실행. 병렬 처리 및 스트리밍 최적화.

### Ingest Graph

```
[PDF 업로드]
    ↓
[Hash 중복 확인] → 이미 존재하면 → [기존 데이터 반환]
    ↓ (신규)
[LlamaParse JSON 파싱] → 페이지별 텍스트 + 요소 타입 + page_number 추출
    ↓
[페이지별 이미지 생성] → ./images/{pdf_hash}/{page_num}.png 저장
    ↓
[Parent/Child 청킹] → element_type 메타데이터 포함
    ↓
[ChromaDB 저장 + LocalFileStore 저장]
```

### Query Graph

```
[사용자 질문]
    ↓
[Parallel_Research (비동기 병렬)]
    ├── [Local_Retriever] → 논문 내 Parent 문맥 검색
    └── [Web_Searcher]   → Tavily 외부 검색
    ↓ (Reducer로 contexts에 누적)
[Reranker_Node] → 관련도 낮은 결과 필터링
    ↓
[Vision_Router]
    ├── 1단계: 질문 키워드 확인
    ├── 2단계: contexts의 element_type 메타데이터 확인
    └── 3단계: (애매한 경우만) LLM 판단
    ↓ (Vision 필요 시)
[Vision_Analyst] → page_number로 이미지 로드 → Vision 모델 분석
    ↓
[Synthesis_Node] → 최종 답변 생성 (출처 페이지 포함, 스트리밍)
```

---

## 6. 데이터 모델 (State Schema)

```python
from typing import Annotated, TypedDict, List, Optional
import operator


class ContextChunk(TypedDict):
    """검색 결과 단위. 출처 추적을 위해 메타데이터 포함."""
    content: str
    source: str           # "local_rag" | "web_search"
    page_number: Optional[int]   # 논문 페이지 번호 (웹 검색이면 None)
    url: Optional[str]           # 웹 검색 URL (논문이면 None)
    element_type: Optional[str]  # "text" | "table" | "image" (LlamaParse 메타데이터)


class AgentState(TypedDict):
    question: str
    pdf_hash: str          # 중복 확인용 해시값
    image_dir: str         # 이미지 폴더 경로만 저장 (전체 이미지 로드 방지)
    # operator.add: 병렬 노드 결과가 덮어씌워지지 않고 누적됨
    contexts: Annotated[List[ContextChunk], operator.add]
    vision_result: Optional[str]
    final_answer: str
```

> **`image_dir` 설계 의도:** 이전 버전의 `image_paths: dict[int, str]`는 논문 전체 이미지 경로를 State에 올려 메모리 낭비를 유발함. 개선된 버전은 폴더 경로만 저장하고, `Vision_Analyst` 노드가 필요한 `page_number`를 받아 해당 이미지만 로드함.

---

## 7. API 명세

### `GET /health`
서버 상태 및 DB 연결 확인.

**Response**
```json
{ "status": "ok", "chromadb": "connected" }
```

---

### `POST /ingest`
논문 PDF 업로드 및 분석. 동일 파일 재업로드 시 캐시된 결과 반환.

**Request** `multipart/form-data`
| 필드 | 타입 | 설명 |
|------|------|------|
| `file` | PDF | 업로드할 논문 파일 |

**Response**
```json
{
  "pdf_hash": "sha256_hash_string",
  "status": "created" | "already_exists",
  "page_count": 42
}
```

---

### `POST /query`
질문 입력 및 스트리밍 답변 수신.

**Request** `application/json`
```json
{
  "question": "이 논문의 핵심 기여가 무엇인가요?",
  "pdf_hash": "sha256_hash_string"
}
```

**Response** `text/event-stream` (SSE 스트리밍)
```
data: {"chunk": "이 논문의 핵심 기여는 "}
data: {"chunk": "Transformer 구조를 개선하여..."}
data: {"done": true, "sources": [{"page": 3}, {"page": 7}]}
```

---

## 8. 인프라 설계 (Docker & uv)

### Dockerfile 핵심 구조

```dockerfile
# Stage 1: 빌드
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 2: 실행 (uv 바이너리 제거로 경량화)
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Volume 마운트 (데이터 영속성)

```yaml
# docker-compose.yml
services:
  app:
    volumes:
      - ./data/chromadb:/app/data/chromadb   # ChromaDB 데이터
      - ./data/cache_dir:/app/cache_dir       # LocalFileStore (Parent 문서)
      - ./data/images:/app/images             # 논문 페이지 이미지
```

> **주의:** Volume 마운트 없이 컨테이너를 재시작하면 모든 데이터가 초기화됨.

---

## 9. 품질 측정 계획

### Phase 1 (MVP)
- 답변 하단에 **"도움이 됐나요? 👍 👎"** 버튼 제공하여 사용자 피드백 수집.
- 목표 응답 시간: 스트리밍 시작까지 3초 이내.

### Phase 2 (안정화 이후)
- `RAGAS` 프레임워크를 이용한 RAG 평가 파이프라인 구축.
  - **Faithfulness:** 답변이 검색된 문서에 근거하는지
  - **Answer Relevancy:** 답변이 질문에 얼마나 관련 있는지
  - **Context Recall:** 필요한 정보를 빠짐없이 검색했는지

---

## 10. 개발 로드맵

| Phase | 주요 작업 | 비고 |
|-------|-----------|------|
| **Phase 1 (MVP)** | Ingest Graph, Query Graph, API 3개, Docker 배포 | 현재 범위 |
| **Phase 2** | Reranker 도입, 하이브리드 검색(BM25+Vector), RAGAS 평가 | 성능 최적화 |
| **Phase 3** | LlamaParse fallback 전략, 다중 논문 지원 | 서비스 안정화 |

---

## 11. 알려진 제약사항 및 리스크

| 항목 | 내용 | 대응 |
|------|------|------|
| LlamaParse 의존성 | 외부 API 장애 시 서비스 중단 | Phase 3에서 fallback 추가 |
| `bge-m3` GPU 요구 | 로컬 임베딩 모델은 GPU 메모리 필요 | 배포 환경 스펙 확인 후 API 임베딩으로 대체 검토 |
| LocalFileStore 직렬화 | LangChain 버전 업 시 역직렬화 오류 가능 | `uv lock`으로 버전 고정 |
| 하이브리드 검색 난이도 | BM25 + ChromaDB 연동 설정이 복잡함 | Phase 2로 이연 |
