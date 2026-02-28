# CLAUDE.md

## 1. 프로젝트 목적 및 개요 (Why & What)
이 프로젝트는 PDF 문서를 파싱하고, 이미지 및 텍스트 문맥을 종합 분석하여 답변을 제공하는 **비동기 멀티모달 RAG(Retrieval-Augmented Generation) FastAPI API**입니다.
웹 검색(Tavily)과 로컬 벡터DB(Chroma)를 병렬로 검색하며, 시각 정보가 필요한 경우에만 Vision 모델을 호출하여 비용과 성능을 최적화하는 것을 핵심 목적으로 합니다.

## 2. 기술 스택 (Tech Stack)
Python 3.12+, LangGraph (Async), FastAPI, uv, ChromaDB, LlamaParse, OpenAI GPT-5o (Vision)

## 3. 핵심 제약 및 규칙 (Universally Applicable Rules)
- **Model Name**: 코드 내 하드코딩된 모델 명(예: gpt-5o, gpt-5-mini 등)은 의도된 것이므로 절대 의심하거나 임의로 변경하지 마세요.
- **Async I/O**: 파일 I/O 및 네트워크 요청은 예외 없이 `async/await`를 사용하세요.
- **LangGraph 상태 관리 (매우 중요)**: `contexts`와 같이 병렬 노드(local_retriever, web_searcher)가 공유하는 리스트 필드는 덮어쓰기 방지를 위해 반드시 `Annotated[list[ContextChunk], operator.add]` 리듀서를 적용하세요.
- **패키지 관리**: `pip` 대신 반드시 `uv`를 사용하세요.
- **포매팅 및 린팅 확인**: 작업 후 아래 명시된 `ruff` 명령어를 실행하여 자동 검증 및 수정을 수행하세요.
- **함수/변수 네이밍**: 함수/변수 snake_case, 클래스 PascalCase, 상수 UPPER_SNAKE_CASE

## 4. 작업 및 검증 명령어 (How)


```bash 

# 1. 실행
PYTHONPATH=. uv run python src/main.py

# 2.테스트 (전체 / 단일)
PYTHONPATH=. uv run pytest
PYTHONPATH=. uv run pytest tests/agents/test_nodes.py::TestVisionRouterNode::test_keyword_match -v

# 3. 린트 및 코드 스타일 자동 수정 (코드 구현 후 반드시 실행)
uv run ruff check . --fix

# 4. 의존성 추가 / 동기화
uv add [package]
uv sync
```

## 5. 아키텍처 및 도메인 지식 (Architecture)
- System architecture and project structure: [docs/architecture.md](/docs/architecture.md)
- RAG 파이프라인(Ingest/Query Graph) 및 Vision Router 최적화 로직의 세부 구조는 `docs/architecture.md`를 먼저 읽고 참고하세요.
- 문서 레지스트리 및 청킹 전략에 대한 이해가 필요하다면 `src/rag/registry.py`를 참조하세요.
세부 구현 사항은 `src/` 내 개별 코드(Progressive Disclosure)를 참조하되, 아래의 핵심 흐름을 숙지하세요.


## 6. Git 커밋 규칙
반드시 다음 접두어 사용: `build`, `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
