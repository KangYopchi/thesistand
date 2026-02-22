"""graph.py 테스트 모듈 - LangGraph 워크플로우 통합 테스트"""

from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.graph import build_ingest_graph, build_query_graph


# ── Ingest Graph ──────────────────────────────────────────────────────


class TestBuildIngestGraph:
    """인제스트 그래프 구조 검증"""

    def test_graph_compiles(self) -> None:
        """인제스트 그래프가 정상적으로 컴파일되는지 확인"""
        graph = build_ingest_graph()
        assert graph is not None

    def test_graph_has_only_ingest_node(self) -> None:
        """인제스트 그래프에 ingest 노드만 존재"""
        graph = build_ingest_graph()
        node_names = set(graph.get_graph().nodes.keys())

        expected_nodes = {"ingest", "__start__", "__end__"}
        assert expected_nodes == node_names


class TestIngestGraphExecution:
    """인제스트 그래프 실행 통합 테스트"""

    @patch("src.agents.nodes.add_documents_to_retriever", new_callable=AsyncMock)
    @patch("src.agents.nodes.create_parent_document_retriever")
    @patch("src.agents.nodes.get_vectorstore")
    @patch("src.agents.nodes.get_embeddings")
    @patch("src.agents.nodes.parse_pdf", new_callable=AsyncMock)
    async def test_ingest_flow(
        self,
        mock_parse: AsyncMock,
        mock_embeddings: MagicMock,
        mock_vectorstore: MagicMock,
        mock_create_retriever: MagicMock,
        mock_add_docs: AsyncMock,
    ) -> None:
        """인제스트 그래프: PDF 파싱 후 image_dir 반환"""
        mock_parse.return_value = {
            "pdf_name": "abc123",
            "elements": [{"page_number": 1, "text": "내용", "element_type": "text"}],
            "page_images": {1: "/tmp/page_001.png"},
        }
        mock_add_docs.return_value = 1

        graph = build_ingest_graph()
        result = await graph.ainvoke({
            "question": "",
            "pdf_hash": "abc123",
            "image_dir": "",
            "contexts": [],
            "vision_result": None,
            "final_answer": "",
        })

        assert "abc123" in result["image_dir"]
        assert result["contexts"] == []
        mock_parse.assert_awaited_once()


# ── Query Graph ───────────────────────────────────────────────────────


class TestBuildQueryGraph:
    """쿼리 그래프 구조 검증"""

    def test_graph_compiles(self) -> None:
        """쿼리 그래프가 정상적으로 컴파일되는지 확인"""
        graph = build_query_graph()
        assert graph is not None

    def test_graph_has_query_nodes(self) -> None:
        """쿼리 그래프에 5개 노드(ingest 제외)가 존재"""
        graph = build_query_graph()
        node_names = set(graph.get_graph().nodes.keys())

        expected_nodes = {
            "local_retriever",
            "web_searcher",
            "vision_router",
            "vision_analyst",
            "synthesis",
            "__start__",
            "__end__",
        }
        assert expected_nodes == node_names

    def test_graph_has_no_ingest_node(self) -> None:
        """쿼리 그래프에 ingest 노드가 없음"""
        graph = build_query_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "ingest" not in node_names


class TestQueryGraphExecution:
    """쿼리 그래프 실행 통합 테스트"""

    @patch("src.agents.nodes.get_embeddings")
    @patch("src.agents.nodes.get_vectorstore")
    @patch("src.agents.nodes.create_parent_document_retriever")
    @patch("src.agents.nodes._get_openai_client")
    @patch("src.agents.nodes.AsyncTavilyClient")
    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    async def test_full_flow_no_vision(
        self,
        mock_tavily_cls: MagicMock,
        mock_get_client: MagicMock,
        mock_create_retriever: MagicMock,
        mock_vectorstore: MagicMock,
        mock_embeddings: MagicMock,
    ) -> None:
        """비전 불필요 경로: 검색 결과를 합성하여 최종 답변 반환"""
        # 로컬 retriever mock
        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "논문 검색 결과"
        mock_doc.metadata = {"page_number": 1, "source": "test", "element_type": "text"}
        mock_retriever.ainvoke = AsyncMock(return_value=[mock_doc])
        mock_create_retriever.return_value = mock_retriever

        # Tavily mock
        mock_tavily = MagicMock()
        mock_tavily.search = AsyncMock(return_value={
            "results": [{"title": "웹", "url": "https://example.com", "content": "웹 내용"}]
        })
        mock_tavily_cls.return_value = mock_tavily

        # OpenAI mock: vision_router → NO_VISION, synthesis → 최종 답변
        router_response = MagicMock()
        router_response.choices = [MagicMock()]
        router_response.choices[0].message.content = "NO_VISION"

        synthesis_response = MagicMock()
        synthesis_response.choices = [MagicMock()]
        synthesis_response.choices[0].message.content = "최종 답변"

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[router_response, synthesis_response]
        )
        mock_get_client.return_value = mock_client

        graph = build_query_graph()
        result = await graph.ainvoke({
            "question": "이 논문의 요약은?",
            "pdf_hash": "abc123def456",
            "image_dir": "",
            "contexts": [],
            "vision_result": None,
            "final_answer": "",
        })

        assert result["final_answer"] == "최종 답변"
        # 로컬(1개) + 웹(1개) = 2개
        assert len(result["contexts"]) == 2
