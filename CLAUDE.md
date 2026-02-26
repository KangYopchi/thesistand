# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 참고 사항
- gpt-5-mini는 2025년 이후 출시된 실제 모델입니다. 학습 데이터에 없더라도 존재를 의심하지 마세요.

## 기술 스택
Python 3.12+, LangGraph (Async), FastAPI, uv, ChromaDB, LlamaParse, OpenAI GPT-5o (Vision)

## 명령어
```bash
# 실행
PYTHONPATH=. uv run python src/main.py

# 테스트 전체
PYTHONPATH=. uv run pytest

# 단일 테스트
PYTHONPATH=. uv run pytest tests/agents/test_nodes.py::TestVisionRouterNode::test_keyword_match -v

# 린트 + 자동 수정
uv run ruff check . --fix

# 의존성 추가 / 동기화
uv add [package]
uv sync
```

## 환경 변수 (`.env`)
```
OPENAI_API_KEY=...       # GPT 모델 + 임베딩
LLAMA_CLOUD_API_KEY=...  # LlamaParse PDF 파싱
TAVILY_API_KEY=...       # 웹 검색
```

## 아키텍처

### 두 개의 분리된 LangGraph

**Ingest Graph** (`build_ingest_graph`): `/ingest` 호출 시 1회 실행
```
START → ingest_node → END
```
`ingest_node`는 LlamaParse(JSON 모드)와 pdf2image를 `asyncio.gather`로 병렬 실행한다.
- LlamaParse → `data/chroma_db/` (ChromaDB) + `data/parent_store/` (부모 청크 JSON)
- pdf2image → `data/images/{pdf_hash}/page_001.png` …

**Query Graph** (`build_query_graph`): `/ask` 호출마다 실행
```
START → local_retriever_node ─┐
START → web_searcher_node   ──┤ (fan-out/fan-in)
                               ↓
                         vision_router_node
                          ↙          ↘ (conditional edge)
               vision_analyst_node   synthesis_node → END
                          ↘          ↗
                         synthesis_node → END
```

### AgentState 핵심 설계
`contexts: Annotated[list[ContextChunk], operator.add]` — 병렬 노드(local_retriever, web_searcher)가 같은 필드에 결과를 append할 수 있도록 reducer를 반드시 유지해야 한다. reducer 없이 리스트 필드를 병렬로 업데이트하면 마지막 write가 나머지를 덮어쓴다.

### Vision Router 3단계 (비용 최적화)
1. **키워드 매칭** (무료): "표", "table", "figure" 등 → 즉시 `NEED_VISION`
2. **메타데이터 확인** (무료): `contexts`의 `element_type`이 "table"/"image"/"figure" → `NEED_VISION`
3. **LLM 판단** (유료): 위 두 단계에서 결론 못 내면 `gpt-5-mini` 호출

### Parent/Child 청킹 전략
`ParentDocumentRetriever` 사용: Child chunk(~400 tokens)로 벡터 검색하고, 매칭 시 Parent chunk(~2000 tokens)의 문맥을 반환한다. Parent 문서는 `DocumentFileStore`(JSON 직렬화 래퍼)를 통해 `data/parent_store/`에 파일로 영속 저장된다.

### 문서 레지스트리
`src/rag/registry.py`의 `DocumentRegistry`가 `data/documents.json`에 인제스트 이력을 저장한다. `/ask` 호출 시 `pdf_hash`를 생략하면 `get_latest()`로 가장 최근 문서를 자동 사용한다.

## 코드 규칙
- 모든 I/O는 `async/await` 필수
- 병렬 노드 리스트 필드: `Annotated[List, operator.add]` 리듀서 필수
- 타입 힌트 필수
- 함수/변수: `snake_case`, 클래스: `PascalCase`, 상수: `UPPER_SNAKE_CASE`
- pip 대신 `uv` 사용

## 데이터 경로
```
data/
├── pdfs/              # {pdf_hash}.pdf
├── images/            # {pdf_hash}/page_001.png …
├── chroma_db/         # ChromaDB (child chunks + 벡터)
├── parent_store/      # Parent chunks (JSON 파일)
└── documents.json     # 인제스트 이력 레지스트리
```

## Git 커밋 규칙
`build` / `feat` / `fix` / `docs` / `style` / `refactor` / `test` / `chore`
