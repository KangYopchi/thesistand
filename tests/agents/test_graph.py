"""graph.py 테스트 모듈 - LangGraph 워크플로우 통합 테스트"""

from unittest.mock import AsyncMock, MagicMock, patch


from src.agents.graph import build_graph


class TestBuildGraph:
    """그래프 구조 검증 테스트"""

    def test_graph_compiles(self) -> None:
        """그래프가 정상적으로 컴파일되는지 확인"""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self) -> None:
        """6개 노드가 모두 등록되었는지 확인"""
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())

        expected_nodes = {
            "ingest",
            "local_retriever",
            "web_searcher",
            "vision_router",
            "vision_analyst",
            "synthesis",
            "__start__",
            "__end__",
        }
        assert expected_nodes == node_names


class TestGraphExecution:
    """그래프 실행 통합 테스트 (모든 외부 API mock)"""

    @patch("src.agents.nodes.parse_pdf", new_callable=AsyncMock)
    @patch("src.agents.nodes.get_embeddings")
    @patch("src.agents.nodes.get_vectorstore")
    @patch("src.agents.nodes.create_parent_document_retriever")
    @patch("src.agents.nodes.add_documents_to_retriever", new_callable=AsyncMock)
    @patch("src.agents.nodes._get_openai_client")
    @patch("src.agents.nodes.AsyncTavilyClient")
    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    async def test_full_flow_no_vision(
        self,
        mock_tavily_cls: MagicMock,
        mock_get_client: MagicMock,
        mock_add_docs: AsyncMock,
        mock_create_retriever: MagicMock,
        mock_vectorstore: MagicMock,
        mock_embeddings: MagicMock,
        mock_parse: AsyncMock,
    ) -> None:
        """비전 불필요 경로 전체 플로우 테스트"""
        # parse_pdf mock
        mock_parse.return_value = {
            "pdf_name": "test",
            "elements": [{"page_number": 1, "text": "content", "element_type": "text"}],
            "page_images": {},
        }
        mock_add_docs.return_value = 1

        # retriever mock
        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "검색 결과"
        mock_doc.metadata = {"page_number": 1, "source": "test"}
        mock_retriever.ainvoke = AsyncMock(return_value=[mock_doc])
        mock_create_retriever.return_value = mock_retriever

        # tavily mock
        mock_tavily = MagicMock()
        mock_tavily.search = AsyncMock(return_value={
            "results": [{"title": "웹", "url": "https://example.com", "content": "웹 내용"}]
        })
        mock_tavily_cls.return_value = mock_tavily

        # openai mock - vision router (NO_VISION) + synthesis
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

        graph = build_graph()
        result = await graph.ainvoke({
            "question": "이 논문의 요약은?",
            "pdf_path": "/tmp/test.pdf",
            "contexts": [],
            "image_paths": {},
            "vision_result": "",
            "final_answer": "",
        })

        assert result["final_answer"] == "최종 답변"
        assert len(result["contexts"]) >= 1
