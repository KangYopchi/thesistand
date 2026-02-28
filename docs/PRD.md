# PRD - thesistand 논문 학습을 돕는 에이전트

## 1. 프로젝트 개요

*   **프로젝트명:** 논문 학습을 돕는 에이전트
*   **목표:** 논문의 계층 구조(Parent-Child)를 유지하는 고성능 RAG 시스템을 구축하고, 비동기 병렬 처리 및 비전 모델을 결합하여 복잡한 테이블과 방법론을 정확히 해석함.
*   **핵심 가치:** 1년 차 개발자가 겪을 수 있는 병렬 상태 관리 오류 및 배포 환경 문제를 선제적으로 해결하며 실무형 AI 에이전트 개발 역량 습득.
*   **타겟**: 논문을 읽고 분석하는 신입 개발자

## 2. 핵심 기능 상세 (Features)

### 2.1. 정밀 메타데이터 파싱 (LlamaParse JSON Mode)
*   **기능:** `LlamaParse`를 **JSON 결과 모드**로 호출하여 텍스트와 함께 각 요소의 `page_number`를 추출.
*   **보정사항:** 단순 마크다운 텍스트 파싱 시 유실되는 페이지 정보를 JSON 파싱 로직을 통해 보존하여, 추후 비전 노드와 1:1 매칭 보장.

### 2.2. 영속성 계층적 RAG (Persistent Hierarchical RAG)
*   **기능:** `ParentDocumentRetriever`를 활용한 계층적 청킹 구현.
    *   **Child Chunks:** 고밀도 벡터 검색용 (예: 400 tokens).
    *   **Parent Chunks:** 요약 및 문맥 제공용 (예: Section 단위).
*   **보정사항:** 서버 재시작 시 데이터 유실 방지를 위해 `InMemoryStore` 대신 **로컬 파일 시스템 기반의 Persistent Store**를 사용하여 부모 문서 저장.

### 2.3. 안전한 비동기 병렬 에이전트 (LangGraph Reducer)
*   **기능:** 논문 검색과 웹 검색을 병렬로 수행하여 응답 속도 최적화.
*   **보정사항:** 병렬 노드에서 데이터 충돌(Overwrite)을 방지하기 위해 LangGraph State의 리스트 필드에 `Annotated[list, operator.add]` 리듀서 적용.

### 2.4. 비용 최적화 비전 분석 (Cost-Effective Vision)
*   **기능:** 텍스트로 해석이 어려운 테이블/수식은 해당 페이지 이미지를 GPT-5o Vision으로 분석.
*   **보정사항:** 토큰 비용 절감을 위해 이미지를 'low' 해상도 모드로 전송하거나, 필요한 영역만 크롭(Crop)하여 전달하는 전처리 로직 포함.

## 3. 기술 스택 (Technical Stacks)
*   **Package Management:** `uv` (Fast dependency resolution & lock-file)
*   **API Framework:** `FastAPI` (Streaming response 지원)
*   **Orchestration:** `LangGraph` (Async StateGraph with Reducers)
*   **Vector DB:** `ChromaDB` (Self-hosted persistent mode)
*   **Deployment:** `Docker` (Multi-stage build with `uv` virtualenv pathing)

## 4. 시스템 아키텍처 (LangGraph Workflow)

1.  **`Ingest_Node`**: PDF를 LlamaParse JSON으로 분석 -> 페이지별 이미지 생성 -> 부모/자식 청크 벡터 DB 저장.
2.  **`Parallel_Research` (Async Group)**:
    *   **`Local_Retriever`**: 부모 문맥을 포함한 논문 내용 검색.
    *   **`Web_Searcher`**: Tavily API를 통한 외부 구현 사례 검색.
    *   *결과는 Reducer를 통해 `context_list`에 누적 저장.*
3.  **`Vision_Router`**: 질문에 시각적 요소가 포함되었는지 LLM이 판단.
4.  **`Vision_Analyst`**: (필요 시) `page_number` 메타데이터와 매칭된 이미지를 로드하여 분석.
5.  **`Synthesis_Node`**: 모든 컨텍스트를 결합하여 최종 답변 생성 (출처 페이지 링크 포함).

## 5. 데이터 모델 (State Schema)
```python
from typing import Annotated, TypedDict, List
import operator

class AgentState(TypedDict):
    question: str
    pdf_path: str
    # operator.add를 사용하여 병렬 노드의 결과가 덮어씌워지지 않고 합쳐지게 함
    contexts: Annotated[List[str], operator.add] 
    image_paths: dict[int, str]
    vision_result: str
    final_answer: str
```

## 6. 인프라 설계 (Docker & uv)
*   **Dockerfile 핵심 가이드:**
    *   `uv sync --frozen`으로 의존성 고정.
    *   `ENV PATH="/app/.venv/bin:$PATH"` 설정을 통해 가상환경 내부 파이썬 실행 보장.
    *   Multi-stage 빌드를 통해 최종 이미지에서 `uv` 바이너리를 제거하고 실행 환경만 유지하여 경량화.
