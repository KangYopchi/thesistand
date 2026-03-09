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

DATA_DIR = Path(__file__).parent.parent / "data"  # data dir path 저장
PDF_DIR = DATA_DIR / "pdfs"  # pdf dir path 저장
IMAGE_DIR = DATA_DIR / "images"  # pdf 에서 추출한 image 파일을 저장할 path 저장

registry = DocumentRegistry(
    DATA_DIR / "documents.json"
)  # 추출한 파일의 정보를 저장하는 파일 객체

"""
왜? 미리 상수로 경로를 저장할까? - 경로가 바뀔 경우 상단의 상수만 변경하면 된다.
"""


# ── Pydantic 모델 ────────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str  # User의 질문
    pdf_hash: Optional[str] = None  # None이면 가장 최근 인제스트 문서 사용


class AskResponse(BaseModel):
    answer: str  # AI 의 정리된 답변
    vision_result: str | None = None  # chunking 된 그림, 표 데이터가 사용된 결과 저장


class IngestResponse(BaseModel):
    message: str  # pdf 처리 결과 저장
    pdf_hash: str  # 처리 된 pdf 파일 hash 데이터
    status: str  # "created" | "already_exists" pdf 파일 생성, 또는 기존 파일 확인 여부
    page_count: int  # 생성한 (또는 기존의 추출한) 이미지 파일 개수


"""
왜? Pydantic을 사용할까? - 데이터의 타입 검수와, 사용자의 입력을 제한하여 예상한 데이터만 처리 될 수 있도록 하기 위해
"""


# ── Lifespan ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """애플리케이션 시작/종료 시 벡터스토어 초기화"""
    PDF_DIR.mkdir(
        parents=True, exist_ok=True
    )  # PDF 폴더 생성, 기존 폴더 있을 경우 생략
    IMAGE_DIR.mkdir(
        parents=True, exist_ok=True
    )  # Image 폴더 생성, 기존 폴더 있을 경우 생략
    logger.info("벡터스토어 초기화 중...")
    get_vectorstore(
        embeddings=get_embeddings()
    )  # ChromaDB 인스턴스 생성후 불러오기, 단 변수에 저장하지 않기 때문에 생성만 된다.
    logger.info("서버 준비 완료")
    yield
    logger.info("서버 종료")


# ── FastAPI 앱 ───────────────────────────────────────────────────────

app = FastAPI(
    title="thesistand",
    description="논문의 계층 구조를 유지하는 RAG 시스템 + 비전 분석 에이전트",
    version="0.1.0",
    lifespan=lifespan,  # FastAPI 실행 시 작성한 lifespan 함수를 실행
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 엔드포인트 ───────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)  # FastAPI 실행 수 첫 화면 진입 시 실행하는 함수.
async def root() -> RedirectResponse:
    return RedirectResponse(
        url="/docs"
    )  # Test Project라 아직 첫 화면이 없어 /docs로 연결했다.


@app.post(
    "/ingest", response_model=IngestResponse
)  # pdf 파일 참조 후 excute 시 실행되는 함수
async def ingest_pdf(file: UploadFile) -> IngestResponse:
    """PDF 파일을 업로드하고 인제스트 수행

    파일 SHA-256 해시를 계산하여 data/pdfs/{hash}.pdf로 저장한다.
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="PDF 파일만 업로드 가능합니다."
        )  # 파일 확장자가 .pdf 일 때만 업로드 가능하도록 함, 만약 확장자만 .pdf로 되어 있을 경우엔 어떻게 해야할까?
        # 위의 경우에는 "Magic Bytes 검사"를 사용해야 한다. 하지만 지금은 MVP 단계로 생략한다.
        # TODO: MagicBytes 검사, pypdf로 파싱 검, 파일 크기 제한 등으로 방어

    # 파일 읽기 + 해시 계산
    file_bytes = await file.read()  # FastAPI의 Starlette가 생성한 SpooledTemporaryFile의 내부에서 PDF의 Raw Binary를 읽고 Byte로 반환
    pdf_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()  # 변환된 byte를 hash 키로 변환해서 저장

    # ── 중복 확인: 이미지 폴더가 존재하면 재파싱 없이 즉시 반환 ─────────
    image_dir = IMAGE_DIR / pdf_hash  # hash이름으로 된 폴더 생성
    existing_images = (
        sorted(image_dir.glob("page_*.png")) if image_dir.exists() else []
    )  # 설정한 경로에 이미지 파일이 존재할 경우 이미지 파일의 list를 반환
    if existing_images:
        logger.info("중복 업로드 감지: %s", pdf_hash)
        registry.add(
            pdf_hash, file.filename or "", len(existing_images)
        )  # 레지스트리에 사용자가 추가를 시도한 pdf 파일의 정보를 저장한다.
        return IngestResponse(
            message=f"{file.filename} 이미 처리된 논문입니다.",
            pdf_hash=pdf_hash,
            status="already_exists",
            page_count=len(existing_images),
        )

    # ── 신규: PDF 저장 후 인제스트 실행 ──────────────────────────────
    pdf_path = PDF_DIR / f"{pdf_hash}.pdf"  # pdf 파일 경로 생성
    pdf_path.write_bytes(file_bytes)  # pdf 파일 저장
    logger.info("PDF 저장 완료: %s", pdf_path)

    graph = build_ingest_graph()  # ingest langgraph 생성
    result = await graph.ainvoke(
        {
            "question": "",
            "pdf_hash": pdf_hash,
            "image_dir": "",
            "contexts": [],
            "vision_result": None,
            "final_answer": "",
        }
    )  # ingest 실행

    result_image_dir = Path(result.get("image_dir", ""))  # 이미지 생성 결과 경로 저장
    page_count = (
        len(list(result_image_dir.glob("page_*.png")))
        if result_image_dir.exists()
        else 0
    )  # 생성된 이미지 개수 확인

    registry.add(
        pdf_hash, file.filename or "", page_count
    )  # Registry에 ingest 된 pdf 내용 저장
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
        # pdf_hash 값으로 저장된 내용이 없는 경우, 가장 최근에 ingest 한 pdf 파일의 정보를 불러온다.
        # Why? - 사용자의 질문에 참고할 문서의 정보를 직접 입력하게 하는 것은 불편함을 줄 수 있기 때문에, 최신 정보를 불러오도록 만듬
        if doc is None:
            raise HTTPException(status_code=404, detail="인제스트된 문서가 없습니다.")
        pdf_hash = doc["pdf_hash"]
    else:
        pdf_hash = request.pdf_hash

    image_dir = IMAGE_DIR / pdf_hash  # pdf 페이지들의 image 파일 경로 저장
    if not image_dir.exists():
        raise HTTPException(status_code=404, detail="해당 논문을 찾을 수 없습니다.")

    graph = build_query_graph()  # 질문 langgraph 호출
    result = await graph.ainvoke(
        {
            "question": request.question,
            "pdf_hash": pdf_hash,
            "image_dir": str(image_dir),
            "contexts": [],
            "vision_result": None,
            "final_answer": "",
        }
    )  # 질문에 대한 답변으로 생성된 AgentState 저장

    vision_result = result.get(
        "vision_result"
    )  # vision 결과를 저장. # Why- Vision 결과를 따로 저장하는 이유는 뭘까?
    if vision_result in ("NEED_VISION", "NO_VISION", "", None):
        vision_result = None

    return AskResponse(
        answer=result.get(
            "final_answer", "답변을 생성하지 못했습니다."
        ),  # dead code, final_answer의 키값은 항상 존재하기 때문에, default값이 반환되는 경우는 없다.
        vision_result=vision_result,
    )


@app.post("/ask/stream")
async def ask_question_stream(request: AskRequest) -> StreamingResponse:
    """논문에 대한 질문에 SSE 스트리밍으로 답변

    LangGraph의 astream_events를 활용하여 각 노드의
    진행 상황을 실시간으로 전달한다.
    """
    if request.pdf_hash is None:
        doc = (
            registry.get_latest()
        )  # pdf_hash가 비어 있는 경우 가장 최근에 ingest 된 문서 정보 로드
        if doc is None:
            raise HTTPException(status_code=404, detail="인제스트된 문서가 없습니다.")
        pdf_hash = doc["pdf_hash"]
    else:
        pdf_hash = request.pdf_hash

    image_dir = IMAGE_DIR / pdf_hash
    if not image_dir.exists():
        raise HTTPException(status_code=404, detail="해당 논문을 찾을 수 없습니다.")

    async def event_generator() -> AsyncGenerator[str, None]:
        graph = build_query_graph()  # Query graph 생성
        initial_state = {
            "question": request.question,
            "pdf_hash": pdf_hash,
            "image_dir": str(image_dir),
            "contexts": [],
            "vision_result": None,
            "final_answer": "",
        }  # 입력 State 생성

        # astream_events 한 번으로 노드 이벤트 + 최종 답변 모두 처리
        final_answer = ""
        async for event in graph.astream_events(
            initial_state, version="v2"
        ):  # LangGraph가 처리하는 과정 중에 나온 결과를 지속적으로 반환
            kind = event.get("event", "")  # 노드 위치
            name = event.get("name", "")  # 노드 네임

            if kind == "on_chain_start":
                yield f"data: {json.dumps({'event': 'node_start', 'node': name}, ensure_ascii=False)}\n\n"  # 어느 단계에서 처리 중인지 안내
            elif kind == "on_chain_end":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict) and output.get("final_answer"):
                    final_answer = output["final_answer"]  # 답변 저장
                yield f"data: {json.dumps({'event': 'node_end', 'node': name}, ensure_ascii=False)}\n\n"  # 단계 종료 알림

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
