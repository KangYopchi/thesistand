"""nodes.py 테스트 모듈 - 각 노드 함수 단위 테스트"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.nodes import (
    ingest_node,
    local_retriever_node,
    route_vision,
    synthesis_node,
    vision_analyst_node,
    vision_router_node,
    web_searcher_node,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def base_state() -> dict:
    """기본 AgentState"""
    return {
        "question": "이 논문의 방법론을 설명해주세요.",
        "pdf_path": "/tmp/test.pdf",
        "contexts": [],
        "image_paths": {},
        "vision_result": "",
        "final_answer": "",
    }


@pytest.fixture
def mock_openai_response() -> MagicMock:
    """OpenAI API 응답 mock"""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "테스트 응답"
    return response


# ── ingest_node ──────────────────────────────────────────────────────


class TestIngestNode:
    """인제스트 노드 테스트"""

    @patch("src.agents.nodes.add_documents_to_retriever", new_callable=AsyncMock)
    @patch("src.agents.nodes.create_parent_document_retriever")
    @patch("src.agents.nodes.get_vectorstore")
    @patch("src.agents.nodes.get_embeddings")
    @patch("src.agents.nodes.parse_pdf", new_callable=AsyncMock)
    async def test_ingest_returns_image_paths(
        self,
        mock_parse: AsyncMock,
        mock_embeddings: MagicMock,
        mock_vectorstore: MagicMock,
        mock_retriever: MagicMock,
        mock_add_docs: AsyncMock,
        base_state: dict,
    ) -> None:
        """인제스트 후 image_paths와 빈 contexts 반환"""
        mock_parse.return_value = {
            "pdf_name": "test",
            "elements": [{"page_number": 1, "text": "hello", "element_type": "text"}],
            "page_images": {1: Path("/tmp/page_001.png")},
        }
        mock_add_docs.return_value = 1

        result = await ingest_node(base_state)

        assert result["image_paths"] == {1: "/tmp/page_001.png"}
        assert result["contexts"] == []
        mock_parse.assert_awaited_once()
        mock_add_docs.assert_awaited_once()


# ── local_retriever_node ─────────────────────────────────────────────


class TestLocalRetrieverNode:
    """로컬 검색 노드 테스트"""

    @patch("src.agents.nodes.create_parent_document_retriever")
    @patch("src.agents.nodes.get_vectorstore")
    @patch("src.agents.nodes.get_embeddings")
    async def test_returns_contexts_with_metadata(
        self,
        mock_embeddings: MagicMock,
        mock_vectorstore: MagicMock,
        mock_create_retriever: MagicMock,
        base_state: dict,
    ) -> None:
        """검색 결과에 출처와 페이지 번호가 포함됨"""
        mock_doc = MagicMock()
        mock_doc.page_content = "검색된 내용"
        mock_doc.metadata = {"page_number": 3, "source": "paper"}

        mock_retriever = MagicMock()
        mock_retriever.ainvoke = AsyncMock(return_value=[mock_doc])
        mock_create_retriever.return_value = mock_retriever

        result = await local_retriever_node(base_state)

        assert len(result["contexts"]) == 1
        assert "[출처: paper, p.3]" in result["contexts"][0]
        assert "검색된 내용" in result["contexts"][0]

    @patch("src.agents.nodes.create_parent_document_retriever")
    @patch("src.agents.nodes.get_vectorstore")
    @patch("src.agents.nodes.get_embeddings")
    async def test_empty_results(
        self,
        mock_embeddings: MagicMock,
        mock_vectorstore: MagicMock,
        mock_create_retriever: MagicMock,
        base_state: dict,
    ) -> None:
        """검색 결과가 없으면 빈 리스트 반환"""
        mock_retriever = MagicMock()
        mock_retriever.ainvoke = AsyncMock(return_value=[])
        mock_create_retriever.return_value = mock_retriever

        result = await local_retriever_node(base_state)

        assert result["contexts"] == []


# ── web_searcher_node ────────────────────────────────────────────────


class TestWebSearcherNode:
    """웹 검색 노드 테스트"""

    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    @patch("src.agents.nodes.AsyncTavilyClient")
    async def test_returns_web_contexts(
        self,
        mock_tavily_cls: MagicMock,
        base_state: dict,
    ) -> None:
        """웹 검색 결과가 contexts로 반환됨"""
        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value={
            "results": [
                {"title": "Example", "url": "https://example.com", "content": "내용"},
            ]
        })
        mock_tavily_cls.return_value = mock_client

        result = await web_searcher_node(base_state)

        assert len(result["contexts"]) == 1
        assert "[웹: Example]" in result["contexts"][0]

    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    @patch("src.agents.nodes.AsyncTavilyClient")
    async def test_handles_search_failure(
        self,
        mock_tavily_cls: MagicMock,
        base_state: dict,
    ) -> None:
        """검색 실패 시 빈 contexts 반환"""
        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=Exception("API Error"))
        mock_tavily_cls.return_value = mock_client

        result = await web_searcher_node(base_state)

        assert result["contexts"] == []


# ── vision_router_node ───────────────────────────────────────────────


class TestVisionRouterNode:
    """비전 라우터 노드 테스트"""

    @patch("src.agents.nodes._get_openai_client")
    async def test_returns_need_vision(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
        mock_openai_response: MagicMock,
    ) -> None:
        """시각 분석 필요 판단"""
        mock_openai_response.choices[0].message.content = "NEED_VISION"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        result = await vision_router_node(base_state)

        assert result["vision_result"] == "NEED_VISION"

    @patch("src.agents.nodes._get_openai_client")
    async def test_returns_no_vision(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
        mock_openai_response: MagicMock,
    ) -> None:
        """시각 분석 불필요 판단"""
        mock_openai_response.choices[0].message.content = "NO_VISION"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        result = await vision_router_node(base_state)

        assert result["vision_result"] == "NO_VISION"

    @patch("src.agents.nodes._get_openai_client")
    async def test_defaults_to_no_vision_on_error(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
    ) -> None:
        """API 에러 시 NO_VISION 기본값"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        mock_get_client.return_value = mock_client

        result = await vision_router_node(base_state)

        assert result["vision_result"] == "NO_VISION"


# ── route_vision ─────────────────────────────────────────────────────


class TestRouteVision:
    """라우팅 함수 테스트"""

    def test_routes_to_vision_analyst(self) -> None:
        assert route_vision({"vision_result": "NEED_VISION"}) == "vision_analyst"

    def test_routes_to_synthesis(self) -> None:
        assert route_vision({"vision_result": "NO_VISION"}) == "synthesis"

    def test_routes_to_synthesis_when_empty(self) -> None:
        assert route_vision({"vision_result": ""}) == "synthesis"


# ── vision_analyst_node ──────────────────────────────────────────────


class TestVisionAnalystNode:
    """비전 분석 노드 테스트"""

    async def test_no_images_returns_message(self, base_state: dict) -> None:
        """이미지가 없으면 안내 메시지 반환"""
        result = await vision_analyst_node(base_state)

        assert "이미지가 없어" in result["vision_result"]

    @patch("src.agents.nodes._get_openai_client")
    async def test_analyzes_images(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
        mock_openai_response: MagicMock,
        tmp_path: Path,
    ) -> None:
        """이미지를 분석하여 결과 반환"""
        # 테스트 이미지 파일 생성
        img_path = tmp_path / "page_001.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        base_state["image_paths"] = {1: str(img_path)}
        mock_openai_response.choices[0].message.content = "테이블 분석 결과"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        result = await vision_analyst_node(base_state)

        assert result["vision_result"] == "테이블 분석 결과"


# ── synthesis_node ───────────────────────────────────────────────────


class TestSynthesisNode:
    """합성 노드 테스트"""

    @patch("src.agents.nodes._get_openai_client")
    async def test_synthesizes_answer(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
        mock_openai_response: MagicMock,
    ) -> None:
        """컨텍스트를 합성하여 최종 답변 생성"""
        base_state["contexts"] = ["컨텍스트 1", "컨텍스트 2"]
        mock_openai_response.choices[0].message.content = "최종 답변입니다."
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        result = await synthesis_node(base_state)

        assert result["final_answer"] == "최종 답변입니다."

    @patch("src.agents.nodes._get_openai_client")
    async def test_includes_vision_result(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
        mock_openai_response: MagicMock,
    ) -> None:
        """비전 분석 결과가 프롬프트에 포함됨"""
        base_state["contexts"] = ["컨텍스트"]
        base_state["vision_result"] = "비전 분석 내용"
        mock_openai_response.choices[0].message.content = "답변"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        await synthesis_node(base_state)

        # 프롬프트에 비전 결과가 포함되었는지 확인
        call_args = mock_client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        assert "비전 분석 내용" in user_content

    @patch("src.agents.nodes._get_openai_client")
    async def test_excludes_router_values_from_vision(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
        mock_openai_response: MagicMock,
    ) -> None:
        """라우터 결과값(NEED_VISION/NO_VISION)은 비전 섹션에 포함하지 않음"""
        base_state["vision_result"] = "NO_VISION"
        mock_openai_response.choices[0].message.content = "답변"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        await synthesis_node(base_state)

        call_args = mock_client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        assert "시각적 분석 결과" not in user_content

    @patch("src.agents.nodes._get_openai_client")
    async def test_handles_api_error(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
    ) -> None:
        """API 에러 시 에러 메시지 반환"""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        mock_get_client.return_value = mock_client

        result = await synthesis_node(base_state)

        assert "오류" in result["final_answer"]
