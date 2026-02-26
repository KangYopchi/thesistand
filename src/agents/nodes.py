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

from src.agents.state import AgentState, ContextChunk
from src.rag.chunker import (
    add_documents_to_retriever,
    create_parent_document_retriever,
)
from src.rag.parser import parse_pdf
from src.rag.vectorstore import get_embeddings, get_vectorstore

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
IMAGE_DIR = DATA_DIR / "images"

# Vision Router 1단계: LLM 호출 없이 즉시 판단하는 키워드 목록
_VISION_KEYWORDS: frozenset[str] = frozenset(
    {
        # 한국어
        "표",
        "그림",
        "수식",
        "그래프",
        "차트",
        "이미지",
        "도표",
        "공식",
        # 영어
        "table",
        "figure",
        "graph",
        "chart",
        "diagram",
        "equation",
        "image",
    }
)

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

    PDF 파일은 data/pdfs/{pdf_hash}.pdf 경로에 저장되어 있어야 한다.
    이미지는 data/images/{pdf_hash}/ 폴더에 생성된다.

    Returns:
        image_dir(이미지 폴더 경로)와 빈 contexts 리스트
    """
    pdf_hash = state["pdf_hash"]
    pdf_path = DATA_DIR / "pdfs" / f"{pdf_hash}.pdf"
    logger.info("인제스트 시작: %s", pdf_path)

    # PDF 파싱 (이미지는 data/images/{pdf_hash}/에 자동 저장)
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

    image_dir = str(IMAGE_DIR / pdf_hash)
    return {"image_dir": image_dir, "contexts": []}


async def local_retriever_node(state: AgentState) -> dict:
    """로컬 벡터DB에서 부모 문맥 포함 검색을 수행하는 노드

    ParentDocumentRetriever를 사용하여 child chunk 매칭 후
    parent chunk 문맥을 반환한다. element_type 메타데이터를
    ContextChunk에 포함하여 Vision Router가 활용할 수 있게 한다.

    Returns:
        contexts 리스트 (reducer로 합산)
    """
    question = state["question"]
    pdf_hash = state["pdf_hash"]
    logger.info("로컬 검색 시작: %s", question)

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings=embeddings)
    retriever = create_parent_document_retriever(
        vectorstore=vectorstore,
        embeddings=embeddings,
    )
    # pdf_hash별로 결과를 격리: ChromaDB metadata "source" == pdf_hash 필터링
    retriever.search_kwargs = {"filter": {"source": pdf_hash}}

    docs = await retriever.ainvoke(question)

    contexts: list[ContextChunk] = []
    for doc in docs:
        page_num: int | None = doc.metadata.get("page_number")
        element_type: str | None = doc.metadata.get("element_type", "text")
        contexts.append(
            ContextChunk(
                content=doc.page_content,
                source="local_rag",
                page_number=page_num,
                url=None,
                element_type=element_type,
            )
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

    contexts: list[ContextChunk] = []
    for result in response.get("results", []):
        contexts.append(
            ContextChunk(
                content=result.get("content", ""),
                source="web_search",
                page_number=None,
                url=result.get("url", ""),
                element_type=None,
            )
        )

    logger.info("웹 검색 완료: %d개 결과", len(contexts))
    return {"contexts": contexts}


async def vision_router_node(state: AgentState) -> dict:
    """질문에 시각적 분석이 필요한지 판단하는 2단계 라우터 노드

    비용 최적화를 위해 LLM 호출 전에 저비용 판단을 먼저 수행한다.

    1단계 (무료): 질문 키워드 확인 — "표", "table", "figure" 등이 있으면 즉시 NEED_VISION
    2단계 (무료): contexts의 element_type 메타데이터 확인 — "table"/"image"이면 NEED_VISION
    3단계 (유료): 1·2단계에서 판단 불가한 경우에만 LLM 호출

    Returns:
        vision_result에 "NEED_VISION" 또는 "NO_VISION" 설정
    """
    question = state["question"]
    contexts = state.get("contexts", [])

    # ── 1단계: 질문 키워드 확인 (LLM 호출 없음) ───────────────────────
    question_lower = question.lower()
    if any(kw in question_lower for kw in _VISION_KEYWORDS):
        logger.info("비전 라우팅: 1단계 키워드 매칭 → NEED_VISION")
        return {"vision_result": "NEED_VISION"}

    # ── 2단계: contexts 메타데이터 확인 (LLM 호출 없음) ───────────────
    has_visual_element = any(
        chunk["element_type"] in ("table", "image", "figure") for chunk in contexts
    )
    if has_visual_element:
        logger.info("비전 라우팅: 2단계 메타데이터 매칭 → NEED_VISION")
        return {"vision_result": "NEED_VISION"}

    # ── 3단계: 애매한 경우에만 LLM 판단 ──────────────────────────────
    logger.info("비전 라우팅: 3단계 LLM 판단")
    client = _get_openai_client()

    try:
        response = await client.chat.completions.create(
            model="gpt-5-mini",
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
        )
        decision = (response.choices[0].message.content or "NO_VISION").strip()
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
    """페이지 이미지를 gpt-5-mini Vision으로 분석하는 노드

    local_rag 검색 결과의 페이지 번호를 기준으로 ±1 페이지를 후보로 선정한다.
    그림/표는 텍스트 참조 페이지의 앞뒤에 위치하는 경우가 많기 때문이다.
    검색 결과가 없으면 전체 이미지 중 최대 5개를 사용한다.

    Returns:
        vision_result에 분석 텍스트 설정
    """
    question = state["question"]
    image_dir = state.get("image_dir", "")
    contexts = state.get("contexts", [])
    logger.info("비전 분석 시작")

    if not image_dir:
        logger.warning("image_dir가 설정되지 않았습니다.")
        return {"vision_result": "이미지가 없어 시각적 분석을 수행할 수 없습니다."}

    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        logger.warning("이미지 폴더 없음: %s", image_dir_path)
        return {"vision_result": "이미지가 없어 시각적 분석을 수행할 수 없습니다."}

    # local_rag 검색 결과의 페이지 번호를 기준으로 후보 페이지 결정.
    # 그림/표는 텍스트 참조 페이지의 앞뒤에 위치하는 경우가 많으므로 ±1 확장.
    ref_pages = {
        chunk["page_number"]
        for chunk in contexts
        if chunk["source"] == "local_rag" and chunk["page_number"]
    }

    if ref_pages:
        expanded: set[int] = set()
        for p in ref_pages:
            expanded.update([p - 1, p, p + 1])
        visual_pages = sorted(p for p in expanded if p > 0)[:5]
    else:
        # 검색 결과가 없으면 전체 이미지 중 최대 5개
        all_pages = sorted(image_dir_path.glob("page_*.png"))
        visual_pages = [
            int(p.stem.split("_")[1]) for p in all_pages[:5]
        ]

    candidate_paths = [
        image_dir_path / f"page_{p:03d}.png" for p in visual_pages
    ]

    image_paths_to_analyze = [p for p in candidate_paths if p.exists()]

    if not image_paths_to_analyze:
        logger.warning("분석할 이미지 파일이 없습니다.")
        return {"vision_result": "이미지가 없어 시각적 분석을 수행할 수 없습니다."}

    logger.info("비전 분석 대상: %d장", len(image_paths_to_analyze))

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

    for image_path in image_paths_to_analyze:
        image_data = image_path.read_bytes()
        b64_image = base64.b64encode(image_data).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_image}",
                    "detail": "low",
                },
            }
        )

    try:
        response = await client.chat.completions.create(
            model="gpt-5-mini",
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
        )
        analysis = response.choices[0].message.content or "비전 분석 결과를 가져오지 못했습니다."
    except Exception:
        logger.exception("비전 분석 실패")
        analysis = "비전 분석 중 오류가 발생했습니다."

    logger.info("비전 분석 완료")
    return {"vision_result": analysis}


async def synthesis_node(state: AgentState) -> dict:
    """모든 컨텍스트를 결합하여 최종 답변을 생성하는 합성 노드

    ContextChunk 리스트에서 출처 정보(페이지 번호/URL)를 추출하여
    포맷팅한 뒤 gpt-5-mini에 전달한다.

    Returns:
        final_answer에 최종 답변 설정
    """
    question = state["question"]
    contexts = state.get("contexts", [])
    vision_result = state.get("vision_result")
    logger.info("합성 시작: 컨텍스트 %d개", len(contexts))

    # ContextChunk를 포맷팅된 텍스트로 변환
    context_texts = []
    for chunk in contexts:
        if chunk["source"] == "local_rag":
            page_num = chunk["page_number"]
            header = f"[논문 p.{page_num}]" if page_num is not None else "[논문]"
        else:
            header = f"[웹: {chunk['url'] or ''}]"
        context_texts.append(f"{header}\n{chunk['content']}")

    context_text = (
        "\n\n---\n\n".join(context_texts)
        if context_texts
        else "검색된 컨텍스트가 없습니다."
    )

    # 비전 분석 결과 포함 (라우터 결과값 제외)
    vision_section = ""
    if vision_result and vision_result not in ("NEED_VISION", "NO_VISION"):
        vision_section = f"\n\n## 시각적 분석 결과\n{vision_result}"

    client = _get_openai_client()

    try:
        response = await client.chat.completions.create(
            model="gpt-5-mini",
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
        )
        final_answer = response.choices[0].message.content or "답변을 생성하지 못했습니다."
    except Exception:
        logger.exception("합성 실패")
        final_answer = "답변 생성 중 오류가 발생했습니다."

    logger.info("합성 완료")
    return {"final_answer": final_answer}
