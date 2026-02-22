"""state.py 테스트 모듈"""

import operator
from typing import get_type_hints

from src.agents.state import AgentState


class TestAgentState:
    """AgentState TypedDict 정의 검증"""

    def test_has_all_required_fields(self) -> None:
        """AgentState에 필수 필드가 모두 정의되어 있는지 확인"""
        hints = get_type_hints(AgentState, include_extras=True)
        expected_fields = {
            "question",
            "pdf_path",
            "contexts",
            "image_paths",
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
        """operator.add 리듀서가 리스트를 올바르게 합치는지 확인"""
        list_a = ["논문 내용 A"]
        list_b = ["웹 검색 결과 B"]

        merged = operator.add(list_a, list_b)

        assert merged == ["논문 내용 A", "웹 검색 결과 B"]
        # 원본 리스트는 변경되지 않음
        assert list_a == ["논문 내용 A"]
        assert list_b == ["웹 검색 결과 B"]
