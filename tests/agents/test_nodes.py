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
from src.agents.state import ContextChunk


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def base_state() -> dict:
    """기본 AgentState"""
    return {
        "question": "이 논문의 방법론을 설명해주세요.",
        "pdf_hash": "abc123def456",
        "image_dir": "",
        "contexts": [],
        "vision_result": None,
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
    async def test_ingest_returns_image_dir(
        self,
        mock_parse: AsyncMock,
        mock_embeddings: MagicMock,
        mock_vectorstore: MagicMock,
        mock_retriever: MagicMock,
        mock_add_docs: AsyncMock,
        base_state: dict,
    ) -> None:
        """인제스트 후 image_dir(문자열 경로)와 빈 contexts 반환"""
        mock_parse.return_value = {
            "pdf_name": "abc123def456",
            "elements": [{"page_number": 1, "text": "hello", "element_type": "text"}],
            "page_images": {1: Path("/tmp/page_001.png")},
        }
        mock_add_docs.return_value = 1

        result = await ingest_node(base_state)

        assert "image_dir" in result
        assert "abc123def456" in result["image_dir"]  # pdf_hash가 경로에 포함
        assert result["contexts"] == []
        mock_parse.assert_awaited_once()
        mock_add_docs.assert_awaited_once()


# ── local_retriever_node ─────────────────────────────────────────────


class TestLocalRetrieverNode:
    """로컬 검색 노드 테스트"""

    @patch("src.agents.nodes.create_parent_document_retriever")
    @patch("src.agents.nodes.get_vectorstore")
    @patch("src.agents.nodes.get_embeddings")
    async def test_returns_context_chunks(
        self,
        mock_embeddings: MagicMock,
        mock_vectorstore: MagicMock,
        mock_create_retriever: MagicMock,
        base_state: dict,
    ) -> None:
        """검색 결과를 ContextChunk 구조로 반환"""
        mock_doc = MagicMock()
        mock_doc.page_content = "검색된 내용"
        mock_doc.metadata = {"page_number": 3, "source": "paper", "element_type": "table"}

        mock_retriever = MagicMock()
        mock_retriever.ainvoke = AsyncMock(return_value=[mock_doc])
        mock_create_retriever.return_value = mock_retriever

        result = await local_retriever_node(base_state)

        assert len(result["contexts"]) == 1
        chunk: ContextChunk = result["contexts"][0]
        assert chunk["content"] == "검색된 내용"
        assert chunk["source"] == "local_rag"
        assert chunk["page_number"] == 3
        assert chunk["element_type"] == "table"
        assert chunk["url"] is None

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
    async def test_returns_web_context_chunks(
        self,
        mock_tavily_cls: MagicMock,
        base_state: dict,
    ) -> None:
        """웹 검색 결과를 ContextChunk 구조로 반환"""
        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value={
            "results": [
                {"title": "Example", "url": "https://example.com", "content": "내용"},
            ]
        })
        mock_tavily_cls.return_value = mock_client

        result = await web_searcher_node(base_state)

        assert len(result["contexts"]) == 1
        chunk: ContextChunk = result["contexts"][0]
        assert chunk["source"] == "web_search"
        assert chunk["url"] == "https://example.com"
        assert chunk["content"] == "내용"
        assert chunk["page_number"] is None
        assert chunk["element_type"] is None

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
    """비전 라우터 노드 테스트 — 3단계 로직"""

    # ── 1단계: 질문 키워드 (LLM 호출 없음) ───────────────────────────

    @patch("src.agents.nodes._get_openai_client")
    async def test_stage1_korean_keyword_triggers_vision(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
    ) -> None:
        """1단계: 한국어 키워드('표')가 있으면 LLM 없이 NEED_VISION"""
        base_state["question"] = "3번 표의 정확도 수치를 설명해주세요."

        result = await vision_router_node(base_state)

        assert result["vision_result"] == "NEED_VISION"
        mock_get_client.assert_not_called()

    @patch("src.agents.nodes._get_openai_client")
    async def test_stage1_english_keyword_triggers_vision(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
    ) -> None:
        """1단계: 영어 키워드('figure')가 있으면 LLM 없이 NEED_VISION"""
        base_state["question"] = "What does Figure 2 show?"

        result = await vision_router_node(base_state)

        assert result["vision_result"] == "NEED_VISION"
        mock_get_client.assert_not_called()

    @patch("src.agents.nodes._get_openai_client")
    async def test_stage1_keyword_case_insensitive(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
    ) -> None:
        """1단계: 영어 키워드는 대소문자 무관"""
        base_state["question"] = "Explain the TABLE in section 3."

        result = await vision_router_node(base_state)

        assert result["vision_result"] == "NEED_VISION"
        mock_get_client.assert_not_called()

    # ── 2단계: contexts 메타데이터 (LLM 호출 없음) ────────────────────

    @patch("src.agents.nodes._get_openai_client")
    async def test_stage2_table_element_type_triggers_vision(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
    ) -> None:
        """2단계: contexts에 element_type='table'이 있으면 LLM 없이 NEED_VISION"""
        base_state["question"] = "이 논문의 정확도는?"  # 키워드 없음
        base_state["contexts"] = [
            ContextChunk(
                content="정확도 비교 데이터",
                source="local_rag",
                page_number=4,
                url=None,
                element_type="table",
            )
        ]

        result = await vision_router_node(base_state)

        assert result["vision_result"] == "NEED_VISION"
        mock_get_client.assert_not_called()

    @patch("src.agents.nodes._get_openai_client")
    async def test_stage2_image_element_type_triggers_vision(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
    ) -> None:
        """2단계: contexts에 element_type='image'가 있으면 LLM 없이 NEED_VISION"""
        base_state["question"] = "모델 구조를 설명해줘."  # 키워드 없음
        base_state["contexts"] = [
            ContextChunk(
                content="모델 아키텍처",
                source="local_rag",
                page_number=2,
                url=None,
                element_type="image",
            )
        ]

        result = await vision_router_node(base_state)

        assert result["vision_result"] == "NEED_VISION"
        mock_get_client.assert_not_called()

    @patch("src.agents.nodes._get_openai_client")
    async def test_stage2_text_only_context_skips_to_stage3(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
        mock_openai_response: MagicMock,
    ) -> None:
        """2단계: element_type='text'만 있으면 3단계 LLM으로 넘어감"""
        base_state["question"] = "이 논문의 기여는?"  # 키워드 없음
        base_state["contexts"] = [
            ContextChunk(
                content="텍스트 내용",
                source="local_rag",
                page_number=1,
                url=None,
                element_type="text",
            )
        ]
        mock_openai_response.choices[0].message.content = "NO_VISION"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        result = await vision_router_node(base_state)

        assert result["vision_result"] == "NO_VISION"
        mock_get_client.assert_called_once()  # 3단계 LLM 호출됨

    # ── 3단계: LLM 판단 (키워드·메타데이터 모두 없을 때) ──────────────

    @patch("src.agents.nodes._get_openai_client")
    async def test_stage3_llm_returns_need_vision(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
        mock_openai_response: MagicMock,
    ) -> None:
        """3단계: 애매한 질문 → LLM이 NEED_VISION 반환"""
        mock_openai_response.choices[0].message.content = "NEED_VISION"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        result = await vision_router_node(base_state)

        assert result["vision_result"] == "NEED_VISION"

    @patch("src.agents.nodes._get_openai_client")
    async def test_stage3_llm_returns_no_vision(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
        mock_openai_response: MagicMock,
    ) -> None:
        """3단계: 애매한 질문 → LLM이 NO_VISION 반환"""
        mock_openai_response.choices[0].message.content = "NO_VISION"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        result = await vision_router_node(base_state)

        assert result["vision_result"] == "NO_VISION"

    @patch("src.agents.nodes._get_openai_client")
    async def test_stage3_defaults_to_no_vision_on_error(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
    ) -> None:
        """3단계: LLM API 에러 시 NO_VISION 기본값"""
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

    def test_routes_to_synthesis_when_none(self) -> None:
        assert route_vision({"vision_result": None}) == "synthesis"


# ── vision_analyst_node ──────────────────────────────────────────────


class TestVisionAnalystNode:
    """비전 분석 노드 테스트"""

    async def test_no_image_dir_returns_message(self, base_state: dict) -> None:
        """image_dir가 없으면 안내 메시지 반환"""
        # base_state의 image_dir는 ""
        result = await vision_analyst_node(base_state)

        assert "이미지가 없어" in result["vision_result"]

    async def test_nonexistent_image_dir_returns_message(
        self, base_state: dict
    ) -> None:
        """존재하지 않는 image_dir이면 안내 메시지 반환"""
        base_state["image_dir"] = "/nonexistent/path/to/images"

        result = await vision_analyst_node(base_state)

        assert "이미지가 없어" in result["vision_result"]

    @patch("src.agents.nodes._get_openai_client")
    async def test_analyzes_images_from_context_page_numbers(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
        mock_openai_response: MagicMock,
        tmp_path: Path,
    ) -> None:
        """contexts의 element_type이 table인 페이지 이미지를 분석"""
        # 테스트 이미지 파일 생성
        img_path = tmp_path / "page_001.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        base_state["image_dir"] = str(tmp_path)
        base_state["contexts"] = [
            ContextChunk(
                content="테이블 내용",
                source="local_rag",
                page_number=1,
                url=None,
                element_type="table",
            )
        ]
        mock_openai_response.choices[0].message.content = "테이블 분석 결과"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        result = await vision_analyst_node(base_state)

        assert result["vision_result"] == "테이블 분석 결과"

    @patch("src.agents.nodes._get_openai_client")
    async def test_falls_back_to_all_images_when_no_visual_context(
        self,
        mock_get_client: MagicMock,
        base_state: dict,
        mock_openai_response: MagicMock,
        tmp_path: Path,
    ) -> None:
        """시각적 요소 없으면 전체 이미지 사용"""
        img_path = tmp_path / "page_001.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        base_state["image_dir"] = str(tmp_path)
        base_state["contexts"] = [
            ContextChunk(
                content="텍스트 내용",
                source="local_rag",
                page_number=1,
                url=None,
                element_type="text",  # 시각적 요소 아님
            )
        ]
        mock_openai_response.choices[0].message.content = "분석 결과"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        result = await vision_analyst_node(base_state)

        assert result["vision_result"] == "분석 결과"


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
        """ContextChunk 리스트를 합성하여 최종 답변 생성"""
        base_state["contexts"] = [
            ContextChunk(
                content="컨텍스트 1",
                source="local_rag",
                page_number=1,
                url=None,
                element_type="text",
            ),
            ContextChunk(
                content="컨텍스트 2",
                source="web_search",
                page_number=None,
                url="https://example.com",
                element_type=None,
            ),
        ]
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
        base_state["contexts"] = [
            ContextChunk(
                content="컨텍스트",
                source="local_rag",
                page_number=1,
                url=None,
                element_type="text",
            )
        ]
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
