# 프로덕션 전환 로드맵

현재 상태: **백엔드 API 완성** (FastAPI + LangGraph + ChromaDB)
목표: 실제 사용자가 쓸 수 있는 프로덕션 웹앱

---

## 현재 구현 완료 항목

| 항목 | 파일 |
|------|------|
| Ingest Graph (PDF 파싱 + 벡터 저장) | `src/agents/graph.py` |
| Query Graph (병렬 검색 + 비전 분석 + 합성) | `src/agents/graph.py` |
| ContextChunk 기반 출처 추적 | `src/agents/state.py` |
| Vision Router 2단계 (키워드 → 메타데이터 → LLM) | `src/agents/nodes.py` |
| SHA-256 중복 업로드 방지 | `src/main.py` |
| FastAPI 엔드포인트 (`/ingest`, `/ask`, `/ask/stream`) | `src/main.py` |
| Docker 멀티스테이지 빌드 | `Dockerfile` |
| 테스트 74개 | `tests/` |

---

## Phase 1 — PRD2 잔여 기능 (백엔드 완성)

### 1-1. `GET /health` 엔드포인트

PRD2 명세에 있지만 미구현. 서버 상태와 ChromaDB 연결을 확인한다.

```python
# 예상 응답
{ "status": "ok", "chromadb": "connected" }
```

**작업:** `src/main.py`에 `/health` 엔드포인트 추가 + 테스트

---

### 1-2. Reranker 노드

검색된 컨텍스트의 관련도를 재평가하여 노이즈를 제거한다.
로컬에서 무료로 실행 가능한 CrossEncoder 모델 사용.

```
현재 Query Graph:
  병렬 검색 → vision_router → synthesis

변경 후:
  병렬 검색 → reranker → vision_router → synthesis
```

**필요 패키지:**
```bash
uv add sentence-transformers
```

**작업:**
1. `src/agents/nodes.py`에 `reranker_node` 함수 추가
2. `src/agents/graph.py`의 Query Graph에 노드 + 엣지 삽입
3. 테스트 추가

---

### 1-3. ChromaDB pdf_hash 메타데이터 필터링

현재 `local_retriever_node`가 모든 논문의 청크를 통합 검색한다.
논문이 여러 개 등록되면 관계없는 논문의 내용이 섞인다.

```python
# src/rag/vectorstore.py — 검색 시 where 필터 추가
retriever.search_kwargs = {"filter": {"pdf_hash": pdf_hash}}
```

**작업:** `local_retriever_node`에 `state["pdf_hash"]` 기반 필터링 추가

---

## Phase 2 — 멀티유저 지원

### 2-1. 사용자 인증

현재 인증이 없어 누구나 API를 호출할 수 있다.

**추천 스택:** Supabase Auth (Google OAuth 내장, PostgreSQL 포함)

```
흐름:
  사용자 로그인 → JWT 발급 → API 요청 시 Authorization 헤더 포함
  FastAPI에서 토큰 검증 → user_id 추출
```

**작업:**
1. `uv add python-jose supabase` 또는 FastAPI Users 라이브러리 도입
2. 모든 엔드포인트에 `Depends(get_current_user)` 추가
3. `IngestResponse`, `AskRequest`에 `user_id` 연동

---

### 2-2. 논문 메타데이터 DB

현재 업로드한 논문 목록을 관리하는 저장소가 없다.
서버가 재시작되면 어떤 논문이 있는지 알 수 없다.

**테이블 설계 (PostgreSQL):**

```sql
CREATE TABLE papers (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL,
    pdf_hash    VARCHAR(64) NOT NULL UNIQUE,
    filename    VARCHAR(255),
    page_count  INT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

**추천 스택:** Supabase (PostgreSQL + 인증 통합) + SQLAlchemy or asyncpg

---

### 2-3. 파일 스토리지 교체

현재 이미지 파일이 `data/images/` 로컬 폴더에 저장된다.
컨테이너 재시작 또는 스케일 아웃 시 데이터 유실 위험.

```
현재: data/images/{pdf_hash}/page_001.png (로컬)
개선: s3://bucket/{pdf_hash}/page_001.png (S3)
```

**추천 스택:** AWS S3 또는 Cloudflare R2 (S3 호환 API)

**작업:**
1. `uv add boto3`
2. `src/rag/parser.py`의 `generate_page_images`에 S3 업로드 로직 추가
3. `vision_analyst_node`에서 이미지 로드 시 S3에서 다운로드

---

## Phase 3 — 프론트엔드 (웹 UI)

### 3-1. 기술 스택

| 항목 | 선택 | 이유 |
|------|------|------|
| 프레임워크 | Next.js 15 (App Router) | SSE 스트리밍, 파일 업로드, React 생태계 |
| 스타일 | Tailwind CSS | 빠른 UI 구성 |
| 인증 | Supabase Auth | 백엔드와 동일 Auth Provider |
| 배포 | Vercel | Next.js 공식 플랫폼, 무료 티어 |

---

### 3-2. 필요한 화면

**1. 업로드 페이지 (`/upload`)**
```
┌─────────────────────────────────────┐
│  논문 PDF를 드래그하거나 클릭하여   │
│  업로드하세요.                       │
│                                     │
│     [ 파일 선택 ]                   │
│                                     │
│  ──────────────────────────────     │
│  최근 업로드한 논문                  │
│  • A Survey of Context Engineering  │
│  • Attention Is All You Need        │
└─────────────────────────────────────┘
```

**2. 채팅 페이지 (`/chat/{pdf_hash}`)**
```
┌─────────────────────────────────────┐
│  A Survey of Context Engineering    │
├─────────────────────────────────────┤
│                                     │
│  [AI] 이 논문의 핵심 기여는         │
│        Context Engineering의 정의와  │
│        ...                          │
│        📄 출처: 3페이지, 7페이지    │
│                                     │
│  [나] 표 1을 설명해줘               │
│                                     │
├─────────────────────────────────────┤
│  질문을 입력하세요...    [전송]      │
└─────────────────────────────────────┘
```

**3. 논문 목록 페이지 (`/papers`)**
- 사용자가 업로드한 논문 카드 목록
- 각 카드에 논문명, 페이지 수, 업로드 날짜

---

### 3-3. SSE 스트리밍 구현 (Next.js)

```typescript
// 백엔드의 /ask/stream을 Next.js에서 소비하는 예시
async function streamAnswer(question: string, pdfHash: string) {
  const response = await fetch('/api/ask/stream', {
    method: 'POST',
    body: JSON.stringify({ question, pdf_hash: pdfHash }),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const lines = decoder.decode(value).split('\n');
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (data.event === 'final_answer') {
          setAnswer(data.answer);
        }
      }
    }
  }
}
```

---

## Phase 4 — 프로덕션 인프라

### 4-1. 비동기 인제스트 큐

현재 `/ingest`는 LlamaParse + 임베딩을 동기로 처리한다.
논문 크기에 따라 30초~3분 소요 → HTTP 타임아웃 위험.

```
개선 흐름:
  POST /ingest → 즉시 { job_id: "..." } 반환
      ↓
  백그라운드 워커가 실제 파싱/임베딩 처리
      ↓
  GET /ingest/status/{job_id} → { status: "processing" | "done" }
```

**추천 스택:** ARQ (asyncio 기반 Redis 큐, 가볍고 FastAPI와 잘 맞음)

```bash
uv add arq redis
```

---

### 4-2. Rate Limiting

API 키 비용 폭발 방지. 사용자당 일일 호출 횟수 제한.

```bash
uv add slowapi  # FastAPI용 rate limiter
```

```python
# 예시: 사용자당 /ingest 하루 10회, /ask 분당 10회 제한
@limiter.limit("10/day")
async def ingest_pdf(...): ...

@limiter.limit("10/minute")
async def ask_question(...): ...
```

---

### 4-3. 모니터링

| 항목 | 도구 | 용도 |
|------|------|------|
| 에러 추적 | Sentry | 프로덕션 예외 실시간 알림 |
| 로그 수집 | structlog + Datadog | 구조화 로그 검색/분석 |
| LLM 비용 추적 | LangSmith | 노드별 토큰 사용량, 응답 시간 |

```bash
uv add sentry-sdk structlog langsmith
```

---

### 4-4. 배포 아키텍처

```
사용자 브라우저
    │ HTTPS
    ▼
Vercel (Next.js 프론트엔드)
    │ API 호출
    ▼
Railway / AWS App Runner (FastAPI 백엔드)
    ├──► Supabase (PostgreSQL + Auth)
    ├──► Chroma Cloud 또는 Qdrant Cloud (벡터DB)
    ├──► AWS S3 / Cloudflare R2 (이미지 파일)
    └──► Redis (ARQ 작업 큐)
```

---

## 전체 로드맵 요약

| Phase | 작업 | 예상 규모 |
|-------|------|-----------|
| **Phase 1** | `/health`, Reranker, pdf_hash 필터링 | 소 |
| **Phase 2** | 사용자 인증, papers DB, S3 파일 스토리지 | 중 |
| **Phase 3** | Next.js 프론트엔드 (업로드 + 채팅 + 목록) | 대 |
| **Phase 4** | 비동기 큐, Rate Limiting, 모니터링, 배포 | 중 |

> Phase 1은 현재 백엔드 코드를 거의 그대로 유지하면서 진행 가능.
> 가장 큰 공수는 **Phase 3 프론트엔드**와 **Phase 2 멀티유저 인증**.
