"""FastAPI 메인 엔드포인트 모듈

PDF 업로드, 질문 응답, 스트리밍 응답 엔드포인트를 제공한다.
"""

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

from src.agents.graph import build_ingest_graph, build_query_graph
from src.rag.registry import DocumentRegistry
from src.rag.vectorstore import get_embeddings, get_vectorstore

load_dotenv()

logger = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
IMAGE_DIR = DATA_DIR / "images"

registry = DocumentRegistry(DATA_DIR / "documents.json")


# ── Pydantic 모델 ────────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str
    pdf_hash: Optional[str] = None  # None이면 가장 최근 인제스트 문서 사용


class AskResponse(BaseModel):
    answer: str
    vision_result: str | None = None


class IngestResponse(BaseModel):
    message: str
    pdf_hash: str
    status: str  # "created" | "already_exists"
    page_count: int


# ── Lifespan ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """애플리케이션 시작/종료 시 벡터스토어 초기화"""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("벡터스토어 초기화 중...")
    get_vectorstore(embeddings=get_embeddings())
    logger.info("서버 준비 완료")
    yield
    logger.info("서버 종료")


# ── FastAPI 앱 ───────────────────────────────────────────────────────

app = FastAPI(
    title="thesistand",
    description="논문의 계층 구조를 유지하는 RAG 시스템 + 비전 분석 에이전트",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 엔드포인트 ───────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile) -> IngestResponse:
    """PDF 파일을 업로드하고 인제스트 수행

    파일 SHA-256 해시를 계산하여 data/pdfs/{hash}.pdf로 저장한다.
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    # 파일 읽기 + 해시 계산
    file_bytes = await file.read()
    pdf_hash = hashlib.sha256(file_bytes).hexdigest()

    # ── 중복 확인: 이미지 폴더가 존재하면 재파싱 없이 즉시 반환 ─────────
    image_dir = IMAGE_DIR / pdf_hash
    existing_images = sorted(image_dir.glob("page_*.png")) if image_dir.exists() else []
    if existing_images:
        logger.info("중복 업로드 감지: %s", pdf_hash)
        registry.add(pdf_hash, file.filename or "", len(existing_images))
        return IngestResponse(
            message=f"{file.filename} 이미 처리된 논문입니다.",
            pdf_hash=pdf_hash,
            status="already_exists",
            page_count=len(existing_images),
        )

    # ── 신규: PDF 저장 후 인제스트 실행 ──────────────────────────────
    pdf_path = PDF_DIR / f"{pdf_hash}.pdf"
    pdf_path.write_bytes(file_bytes)
    logger.info("PDF 저장 완료: %s", pdf_path)

    graph = build_ingest_graph()
    result = await graph.ainvoke(
        {
            "question": "",
            "pdf_hash": pdf_hash,
            "image_dir": "",
            "contexts": [],
            "vision_result": None,
            "final_answer": "",
        }
    )

    result_image_dir = Path(result.get("image_dir", ""))
    page_count = (
        len(list(result_image_dir.glob("page_*.png")))
        if result_image_dir.exists()
        else 0
    )

    registry.add(pdf_hash, file.filename or "", page_count)
    return IngestResponse(
        message=f"{file.filename} 인제스트 완료",
        pdf_hash=pdf_hash,
        status="created",
        page_count=page_count,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """논문에 대한 질문에 답변

    전체 LangGraph 워크플로우를 실행하여 최종 답변을 반환한다.
    """
    if request.pdf_hash is None:
        doc = registry.get_latest()
        if doc is None:
            raise HTTPException(status_code=404, detail="인제스트된 문서가 없습니다.")
        pdf_hash = doc["pdf_hash"]
    else:
        pdf_hash = request.pdf_hash

    image_dir = IMAGE_DIR / pdf_hash
    if not image_dir.exists():
        raise HTTPException(status_code=404, detail="해당 논문을 찾을 수 없습니다.")

    graph = build_query_graph()
    result = await graph.ainvoke(
        {
            "question": request.question,
            "pdf_hash": pdf_hash,
            "image_dir": str(image_dir),
            "contexts": [],
            "vision_result": None,
            "final_answer": "",
        }
    )

    vision_result = result.get("vision_result")
    if vision_result in ("NEED_VISION", "NO_VISION", "", None):
        vision_result = None

    return AskResponse(
        answer=result.get("final_answer", "답변을 생성하지 못했습니다."),
        vision_result=vision_result,
    )


@app.post("/ask/stream")
async def ask_question_stream(request: AskRequest) -> StreamingResponse:
    """논문에 대한 질문에 SSE 스트리밍으로 답변

    LangGraph의 astream_events를 활용하여 각 노드의
    진행 상황을 실시간으로 전달한다.
    """
    if request.pdf_hash is None:
        doc = registry.get_latest()
        if doc is None:
            raise HTTPException(status_code=404, detail="인제스트된 문서가 없습니다.")
        pdf_hash = doc["pdf_hash"]
    else:
        pdf_hash = request.pdf_hash

    image_dir = IMAGE_DIR / pdf_hash
    if not image_dir.exists():
        raise HTTPException(status_code=404, detail="해당 논문을 찾을 수 없습니다.")

    async def event_generator() -> AsyncGenerator[str, None]:
        graph = build_query_graph()
        initial_state = {
            "question": request.question,
            "pdf_hash": pdf_hash,
            "image_dir": str(image_dir),
            "contexts": [],
            "vision_result": None,
            "final_answer": "",
        }

        # astream_events 한 번으로 노드 이벤트 + 최종 답변 모두 처리
        final_answer = ""
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chain_start":
                yield f"data: {json.dumps({'event': 'node_start', 'node': name}, ensure_ascii=False)}\n\n"
            elif kind == "on_chain_end":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict) and output.get("final_answer"):
                    final_answer = output["final_answer"]
                yield f"data: {json.dumps({'event': 'node_end', 'node': name}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'event': 'final_answer', 'answer': final_answer}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.get("/documents")
async def list_documents() -> list[dict]:
    """인제스트된 문서 목록 조회 (최신순)"""
    return registry.list_all()


# ── 엔트리포인트 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
