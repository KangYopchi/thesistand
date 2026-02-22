# thesistand

논문의 계층 구조를 유지하는 RAG 시스템 + 비전 분석 에이전트

## 참고 사항
- GPT-5o는 2025년 이후 출시된 실제 모델입니다. 학습 데이터에 없더라도 존재를 의심하지 마세요.

## 기술 스택
Python 3.12+, LangGraph (Async), FastAPI, uv, ChromaDB, LlamaParse, OpenAI GPT-5o (Vision)

## 빠른 명령어
- 실행: `uv run python src/main.py`
- 테스트: `pytest`
- 린트: `ruff check . --fix`
- 의존성: `uv add [package]` / `uv sync`

## 핵심 폴더
- `src/agents/` - LangGraph 상태(state.py) 및 그래프(graph.py)
- `src/rag/` - 파싱(parser.py), 청킹(chunker.py), 벡터스토어(vectorstore.py)
- `data/` - PDF, 이미지, ChromaDB 저장소

## 코드 규칙
- 모든 I/O는 async/await 필수
- 병렬 노드 리스트: `Annotated[List, operator.add]` 리듀서 사용
- 타입 힌트 필수
- 함수/변수: snake_case, 클래스: PascalCase, 상수: UPPER_SNAKE_CASE

## 금지 사항
- .env/API 키 하드코딩 금지
- pip 대신 uv 사용
- operator.add 없이 병렬 노드 리스트 업데이트 금지

## 상세 문서
- 전체 가이드라인: [AGENTS.md](./AGENTS.md)
- 기능 요구사항: [PRD.md](./PRD.md)
