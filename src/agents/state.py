"""LangGraph AgentState 정의 모듈

병렬 노드에서 안전하게 리스트를 합치기 위해
Annotated[list, operator.add] 리듀서를 사용한다.
"""

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    """에이전트 전체 워크플로우의 상태 스키마

    Attributes:
        question: 사용자 질문
        pdf_path: 분석 대상 PDF 파일 경로
        contexts: 병렬 노드(local_retriever, web_searcher) 결과를 합산
        image_paths: {page_number: image_path} 매핑
        vision_result: 비전 분석 결과 텍스트
        final_answer: 최종 합성 답변
    """

    question: str
    pdf_path: str
    contexts: Annotated[list[str], operator.add]
    image_paths: dict[int, str]
    vision_result: str
    final_answer: str
