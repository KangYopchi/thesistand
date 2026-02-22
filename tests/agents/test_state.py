"""state.py 테스트 모듈"""

import operator
from typing import get_type_hints

from src.agents.state import AgentState, ContextChunk


class TestAgentState:
    """AgentState TypedDict 정의 검증"""

    def test_has_all_required_fields(self) -> None:
        """AgentState에 필수 필드가 모두 정의되어 있는지 확인"""
        hints = get_type_hints(AgentState, include_extras=True)
        expected_fields = {
            "question",
            "pdf_hash",
            "image_dir",
            "contexts",
            "vision_result",
            "final_answer",
        }
        assert set(hints.keys()) == expected_fields

    def test_contexts_has_add_reducer(self) -> None:
        """contexts 필드에 operator.add 리듀서가 설정되어 있는지 확인"""
        hints = get_type_hints(AgentState, include_extras=True)
        contexts_type = hints["contexts"]

        # Annotated 타입에서 메타데이터 추출
        assert hasattr(contexts_type, "__metadata__"), "contexts에 Annotated 메타데이터 없음"
        assert operator.add in contexts_type.__metadata__, "contexts에 operator.add 리듀서 없음"

    def test_contexts_reducer_merges_lists(self) -> None:
        """operator.add 리듀서가 ContextChunk 리스트를 올바르게 합치는지 확인"""
        chunk_a = ContextChunk(
            content="논문 내용 A",
            source="local_rag",
            page_number=1,
            url=None,
            element_type="text",
        )
        chunk_b = ContextChunk(
            content="웹 검색 결과 B",
            source="web_search",
            page_number=None,
            url="https://example.com",
            element_type=None,
        )

        merged = operator.add([chunk_a], [chunk_b])

        assert merged == [chunk_a, chunk_b]
        # 원본 리스트는 변경되지 않음
        assert [chunk_a] == [chunk_a]
        assert [chunk_b] == [chunk_b]


class TestContextChunk:
    """ContextChunk TypedDict 정의 검증"""

    def test_has_all_required_fields(self) -> None:
        """ContextChunk에 필수 필드가 모두 정의되어 있는지 확인"""
        hints = get_type_hints(ContextChunk)
        expected_fields = {"content", "source", "page_number", "url", "element_type"}
        assert set(hints.keys()) == expected_fields

    def test_local_rag_chunk(self) -> None:
        """로컬 RAG 청크 생성 확인"""
        chunk = ContextChunk(
            content="논문 내용",
            source="local_rag",
            page_number=3,
            url=None,
            element_type="table",
        )
        assert chunk["source"] == "local_rag"
        assert chunk["page_number"] == 3
        assert chunk["url"] is None

    def test_web_search_chunk(self) -> None:
        """웹 검색 청크 생성 확인"""
        chunk = ContextChunk(
            content="웹 내용",
            source="web_search",
            page_number=None,
            url="https://example.com",
            element_type=None,
        )
        assert chunk["source"] == "web_search"
        assert chunk["page_number"] is None
        assert chunk["url"] == "https://example.com"
