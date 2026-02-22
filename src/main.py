"""FastAPI 메인 엔드포인트 모듈

PDF 업로드, 질문 응답, 스트리밍 응답 엔드포인트를 제공한다.
"""

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.graph import build_graph
from src.rag.vectorstore import get_embeddings, get_vectorstore

load_dotenv()

logger = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"


# ── Pydantic 모델 ────────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str
    pdf_path: str


class AskResponse(BaseModel):
    answer: str
    vision_result: str | None = None


class IngestResponse(BaseModel):
    message: str
    pdf_path: str
    page_count: int


# ── Lifespan ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """애플리케이션 시작/종료 시 벡터스토어 초기화"""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
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


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile) -> IngestResponse:
    """PDF 파일을 업로드하고 인제스트 수행

    PDF를 data/pdfs/에 저장한 후 LlamaParse 파싱 및
    벡터DB 저장을 실행한다.
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    # PDF 저장
    pdf_path = PDF_DIR / file.filename
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info("PDF 저장 완료: %s", pdf_path)

    # 인제스트 그래프 실행 (ingest_node만 필요)
    graph = build_graph()
    result = await graph.ainvoke({
        "question": "",
        "pdf_path": str(pdf_path),
        "contexts": [],
        "image_paths": {},
        "vision_result": "",
        "final_answer": "",
    })

    page_count = len(result.get("image_paths", {}))

    return IngestResponse(
        message=f"{file.filename} 인제스트 완료",
        pdf_path=str(pdf_path),
        page_count=page_count,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """논문에 대한 질문에 답변

    전체 LangGraph 워크플로우를 실행하여 최종 답변을 반환한다.
    """
    pdf_path = Path(request.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF 파일을 찾을 수 없습니다.")

    graph = build_graph()
    result = await graph.ainvoke({
        "question": request.question,
        "pdf_path": str(pdf_path),
        "contexts": [],
        "image_paths": {},
        "vision_result": "",
        "final_answer": "",
    })

    vision_result = result.get("vision_result", "")
    if vision_result in ("NEED_VISION", "NO_VISION", ""):
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
    pdf_path = Path(request.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF 파일을 찾을 수 없습니다.")

    async def event_generator() -> AsyncGenerator[str, None]:
        graph = build_graph()
        initial_state = {
            "question": request.question,
            "pdf_path": str(pdf_path),
            "contexts": [],
            "image_paths": {},
            "vision_result": "",
            "final_answer": "",
        }

        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chain_start":
                yield f"data: {{\"event\": \"node_start\", \"node\": \"{name}\"}}\n\n"
            elif kind == "on_chain_end":
                yield f"data: {{\"event\": \"node_end\", \"node\": \"{name}\"}}\n\n"

        # 최종 결과 전송
        result = await graph.ainvoke(initial_state)
        final_answer = result.get("final_answer", "")
        import json

        yield f"data: {json.dumps({'event': 'final_answer', 'answer': final_answer}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ── 엔트리포인트 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
