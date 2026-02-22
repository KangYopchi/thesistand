"""LangGraph StateGraph 워크플로우 구성 모듈

논문 분석 에이전트의 전체 흐름을 정의한다:
Ingest → Parallel(Local_Retriever + Web_Searcher) → Vision_Router
  → (조건부) Vision_Analyst → Synthesis
"""

from langgraph.graph import END, START, StateGraph

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


def build_graph() -> StateGraph:
    """논문 분석 에이전트 그래프를 생성하고 컴파일

    Returns:
        컴파일된 LangGraph 실행 가능 그래프
    """
    graph = StateGraph(AgentState)

    # ── 노드 등록 ────────────────────────────────────────────────────
    graph.add_node("ingest", ingest_node)
    graph.add_node("local_retriever", local_retriever_node)
    graph.add_node("web_searcher", web_searcher_node)
    graph.add_node("vision_router", vision_router_node)
    graph.add_node("vision_analyst", vision_analyst_node)
    graph.add_node("synthesis", synthesis_node)

    # ── 엣지 정의 ────────────────────────────────────────────────────

    # START → ingest
    graph.add_edge(START, "ingest")

    # ingest → 병렬 연구 (fan-out)
    graph.add_edge("ingest", "local_retriever")
    graph.add_edge("ingest", "web_searcher")

    # 병렬 연구 완료 → vision_router (fan-in)
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

    # vision_analyst → synthesis
    graph.add_edge("vision_analyst", "synthesis")

    # synthesis → END
    graph.add_edge("synthesis", END)

    return graph.compile()


# 컴파일된 그래프 인스턴스 (import하여 사용)
app = build_graph()
