"""LangGraph StateGraph 워크플로우 구성 모듈

Ingest(논문 저장)와 Query(질문 처리)를 별도 그래프로 분리한다.

- Ingest Graph: PDF 업로드 시 1회 실행
  START → ingest → END

- Query Graph: 질문마다 실행, 병렬 처리 및 스트리밍 최적화
  START → (local_retriever ∥ web_searcher) → vision_router → (조건부) vision_analyst → synthesis → END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.nodes import (
    ingest_node,
    local_retriever_node,
    route_vision,
    synthesis_node,
    vision_analyst_node,
    vision_router_node,
    web_searcher_node,
)
from src.agents.state import AgentState


def build_ingest_graph() -> CompiledStateGraph:
    """PDF 파싱 및 벡터DB 저장을 담당하는 인제스트 그래프 생성

    Returns:
        컴파일된 LangGraph 실행 가능 그래프
    """
    graph = StateGraph(AgentState)  # type: ignore[type-var]

    graph.add_node("ingest", ingest_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", END)

    return graph.compile()


def build_query_graph() -> CompiledStateGraph:
    """질문 처리를 담당하는 쿼리 그래프 생성

    병렬 검색(local_retriever ∥ web_searcher) → 비전 판단 → 합성 순서로 실행한다.

    Returns:
        컴파일된 LangGraph 실행 가능 그래프
    """
    graph = StateGraph(AgentState)  # type: ignore[type-var]

    # ── 노드 등록 ────────────────────────────────────────────────────
    graph.add_node("local_retriever", local_retriever_node)
    graph.add_node("web_searcher", web_searcher_node)
    graph.add_node("vision_router", vision_router_node)
    graph.add_node("vision_analyst", vision_analyst_node)
    graph.add_node("synthesis", synthesis_node)

    # ── 엣지 정의 ────────────────────────────────────────────────────

    # START → 병렬 검색 (fan-out)
    graph.add_edge(START, "local_retriever")
    graph.add_edge(START, "web_searcher")

    # 병렬 검색 완료 → vision_router (fan-in)
    graph.add_edge("local_retriever", "vision_router")
    graph.add_edge("web_searcher", "vision_router")

    # vision_router → 조건부 라우팅
    graph.add_conditional_edges(
        "vision_router",
        route_vision,
        {
            "vision_analyst": "vision_analyst",
            "synthesis": "synthesis",
        },
    )

    # vision_analyst → synthesis → END
    graph.add_edge("vision_analyst", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()
