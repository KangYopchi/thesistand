"""LangGraph 에이전트 노드 함수 모듈

각 노드는 AgentState를 입력받아 상태 업데이트 딕셔너리를 반환한다.
병렬 노드(local_retriever, web_searcher)는 contexts 필드에
operator.add 리듀서로 결과를 합산한다.
"""

import base64
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient

from src.agents.state import AgentState
from src.rag.chunker import (
    add_documents_to_retriever,
    create_parent_document_retriever,
)
from src.rag.parser import parse_pdf
from src.rag.vectorstore import get_embeddings, get_vectorstore

load_dotenv()

logger = logging.getLogger(__name__)

# ── OpenAI 클라이언트 (지연 초기화) ──────────────────────────────────

_openai_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    """OpenAI 클라이언트 싱글톤 반환"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


# ── 노드 함수들 ─────────────────────────────────────────────────────


async def ingest_node(state: AgentState) -> dict:
    """PDF를 파싱하고 벡터DB에 저장하는 인제스트 노드

    1. LlamaParse로 PDF 파싱 (텍스트 + 페이지 이미지)
    2. Parent/Child 계층적 청킹 후 ChromaDB에 저장

    Returns:
        image_paths와 빈 contexts 리스트
    """
    pdf_path = Path(state["pdf_path"])
    logger.info("인제스트 시작: %s", pdf_path)

    # PDF 파싱
    parse_result = await parse_pdf(pdf_path)

    # 벡터스토어 및 retriever 생성
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings=embeddings)
    retriever = create_parent_document_retriever(
        vectorstore=vectorstore,
        embeddings=embeddings,
    )

    # 문서 저장
    doc_count = await add_documents_to_retriever(
        retriever=retriever,
        elements=parse_result["elements"],
        pdf_name=parse_result["pdf_name"],
    )
    logger.info("저장 완료: %d개 문서", doc_count)

    # page_images: dict[int, Path] → dict[int, str]
    image_paths = {k: str(v) for k, v in parse_result["page_images"].items()}

    return {"image_paths": image_paths, "contexts": []}


async def local_retriever_node(state: AgentState) -> dict:
    """로컬 벡터DB에서 부모 문맥 포함 검색을 수행하는 노드

    ParentDocumentRetriever를 사용하여 child chunk 매칭 후
    parent chunk 문맥을 반환한다.

    Returns:
        contexts 리스트 (reducer로 합산)
    """
    question = state["question"]
    logger.info("로컬 검색 시작: %s", question)

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings=embeddings)
    retriever = create_parent_document_retriever(
        vectorstore=vectorstore,
        embeddings=embeddings,
    )

    docs = await retriever.ainvoke(question)

    contexts = []
    for doc in docs:
        page_num = doc.metadata.get("page_number", "?")
        source = doc.metadata.get("source", "unknown")
        contexts.append(
            f"[출처: {source}, p.{page_num}]\n{doc.page_content}"
        )

    logger.info("로컬 검색 완료: %d개 결과", len(contexts))
    return {"contexts": contexts}


async def web_searcher_node(state: AgentState) -> dict:
    """Tavily API로 외부 웹 검색을 수행하는 노드

    Returns:
        contexts 리스트 (reducer로 합산)
    """
    question = state["question"]
    logger.info("웹 검색 시작: %s", question)

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY 환경변수가 설정되지 않았습니다.")

    client = AsyncTavilyClient(api_key=api_key)

    try:
        response = await client.search(
            query=question,
            max_results=5,
            search_depth="advanced",
        )
    except Exception:
        logger.exception("웹 검색 실패")
        return {"contexts": []}

    contexts = []
    for result in response.get("results", []):
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
        contexts.append(f"[웹: {title}]({url})\n{content}")

    logger.info("웹 검색 완료: %d개 결과", len(contexts))
    return {"contexts": contexts}


async def vision_router_node(state: AgentState) -> dict:
    """질문에 시각적 분석이 필요한지 판단하는 라우터 노드

    GPT-5o에 질문을 보내 테이블/수식/그래프 등 시각적 요소의
    분석 필요 여부를 판단한다.

    Returns:
        vision_result에 "NEED_VISION" 또는 "NO_VISION" 설정
    """
    question = state["question"]
    logger.info("비전 라우팅 판단: %s", question)

    client = _get_openai_client()

    try:
        response = await client.chat.completions.create(
            model="gpt-5o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a routing assistant. Determine if the user's question "
                        "about a research paper requires visual analysis of tables, figures, "
                        "charts, equations, or diagrams. "
                        "Reply with exactly 'NEED_VISION' or 'NO_VISION'."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=20,
        )
        decision = response.choices[0].message.content.strip()
    except Exception:
        logger.exception("비전 라우팅 판단 실패, 기본값 NO_VISION")
        decision = "NO_VISION"

    logger.info("비전 라우팅 결과: %s", decision)
    return {"vision_result": decision}


def route_vision(state: AgentState) -> str:
    """vision_router_node의 결과에 따라 다음 노드를 결정하는 라우팅 함수

    Returns:
        "vision_analyst" 또는 "synthesis"
    """
    if state.get("vision_result") == "NEED_VISION":
        return "vision_analyst"
    return "synthesis"


async def vision_analyst_node(state: AgentState) -> dict:
    """페이지 이미지를 GPT-5o Vision으로 분석하는 노드

    image_paths에서 관련 페이지 이미지를 로드하여
    low resolution 모드로 GPT-5o Vision에 전송한다.

    Returns:
        vision_result에 분석 텍스트 설정
    """
    question = state["question"]
    image_paths = state.get("image_paths", {})
    logger.info("비전 분석 시작: 이미지 %d장", len(image_paths))

    if not image_paths:
        logger.warning("분석할 이미지가 없습니다.")
        return {"vision_result": "이미지가 없어 시각적 분석을 수행할 수 없습니다."}

    client = _get_openai_client()

    # 이미지를 base64로 인코딩하여 메시지 구성
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"다음 논문 페이지 이미지를 분석하여 질문에 답해주세요.\n\n"
                f"질문: {question}\n\n"
                f"테이블, 수식, 그래프 등 시각적 요소를 중심으로 분석해주세요."
            ),
        }
    ]

    # 최대 5페이지로 제한 (비용 최적화)
    sorted_pages = sorted(image_paths.keys())[:5]

    for page_num in sorted_pages:
        image_path = Path(image_paths[page_num])
        if not image_path.exists():
            logger.warning("이미지 파일 없음: %s", image_path)
            continue

        image_data = image_path.read_bytes()
        b64_image = base64.b64encode(image_data).decode("utf-8")

        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_image}",
                "detail": "low",
            },
        })

    try:
        response = await client.chat.completions.create(
            model="gpt-5o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research paper visual analyst. "
                        "Analyze the provided page images and explain tables, "
                        "figures, equations, and diagrams in detail. "
                        "Always reference the page number in your analysis."
                    ),
                },
                {"role": "user", "content": content},
            ],
            max_tokens=2000,
        )
        analysis = response.choices[0].message.content
    except Exception:
        logger.exception("비전 분석 실패")
        analysis = "비전 분석 중 오류가 발생했습니다."

    logger.info("비전 분석 완료")
    return {"vision_result": analysis}


async def synthesis_node(state: AgentState) -> dict:
    """모든 컨텍스트를 결합하여 최종 답변을 생성하는 합성 노드

    contexts(로컬 검색 + 웹 검색)와 vision_result를 결합하여
    출처 페이지 번호를 포함한 최종 답변을 생성한다.

    Returns:
        final_answer에 최종 답변 설정
    """
    question = state["question"]
    contexts = state.get("contexts", [])
    vision_result = state.get("vision_result", "")
    logger.info("합성 시작: 컨텍스트 %d개", len(contexts))

    # 컨텍스트 결합
    context_text = "\n\n---\n\n".join(contexts) if contexts else "검색된 컨텍스트가 없습니다."

    # 비전 분석 결과 포함 (라우터 결과값 제외)
    vision_section = ""
    if vision_result and vision_result not in ("NEED_VISION", "NO_VISION"):
        vision_section = f"\n\n## 시각적 분석 결과\n{vision_result}"

    client = _get_openai_client()

    try:
        response = await client.chat.completions.create(
            model="gpt-5o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research paper analysis assistant. "
                        "Synthesize the provided contexts and visual analysis "
                        "into a comprehensive answer. "
                        "Always cite page numbers when referencing specific content. "
                        "Answer in the same language as the question."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"## 질문\n{question}\n\n"
                        f"## 검색된 컨텍스트\n{context_text}"
                        f"{vision_section}"
                    ),
                },
            ],
            max_tokens=3000,
        )
        final_answer = response.choices[0].message.content
    except Exception:
        logger.exception("합성 실패")
        final_answer = "답변 생성 중 오류가 발생했습니다."

    logger.info("합성 완료")
    return {"final_answer": final_answer}
