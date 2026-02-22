"""LangGraph AgentState 정의 모듈

병렬 노드에서 안전하게 리스트를 합치기 위해
Annotated[list, operator.add] 리듀서를 사용한다.
"""

import operator
from typing import Annotated, Optional, TypedDict


class ContextChunk(TypedDict):
    """검색 결과 단위. 출처 추적을 위해 메타데이터 포함.

    Attributes:
        content: 검색된 텍스트 내용
        source: 출처 구분 ("local_rag" | "web_search")
        page_number: 논문 페이지 번호 (웹 검색이면 None)
        url: 웹 검색 URL (논문이면 None)
        element_type: LlamaParse 요소 타입 ("text" | "table" | "image")
    """

    content: str
    source: str
    page_number: Optional[int]
    url: Optional[str]
    element_type: Optional[str]


class AgentState(TypedDict):
    """에이전트 전체 워크플로우의 상태 스키마

    Attributes:
        question: 사용자 질문
        pdf_hash: SHA-256 해시값 (중복 확인 및 파일 식별)
        image_dir: 논문 페이지 이미지 폴더 경로
        contexts: 병렬 노드(local_retriever, web_searcher) 결과를 합산
        vision_result: 비전 분석 결과 텍스트 (라우터 결과 "NEED_VISION"/"NO_VISION" 포함)
        final_answer: 최종 합성 답변
    """

    question: str
    pdf_hash: str
    image_dir: str
    contexts: Annotated[list[ContextChunk], operator.add]
    vision_result: Optional[str]
    final_answer: str
