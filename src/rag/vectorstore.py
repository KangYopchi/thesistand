"""ChromaDB 벡터스토어 설정 모듈

Persistent 모드로 ChromaDB를 초기화하고,
OpenAI 임베딩과 연동된 벡터스토어 인스턴스를 제공한다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "thesistand"


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI 임베딩 모델 인스턴스 반환

    Returns:
        OpenAIEmbeddings 인스턴스

    Raises:
        ValueError: OPENAI_API_KEY가 설정되지 않은 경우
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    return OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small",
    )


def get_vectorstore(
    collection_name: str = COLLECTION_NAME,
    persist_directory: Path | str | None = None,
    embeddings: OpenAIEmbeddings | None = None,
) -> Chroma:
    """ChromaDB 벡터스토어 인스턴스 생성

    Args:
        collection_name: ChromaDB 컬렉션 이름
        persist_directory: DB 저장 경로 (기본: data/chroma_db/)
        embeddings: 임베딩 모델 (기본: OpenAI text-embedding-3-small)

    Returns:
        Chroma 벡터스토어 인스턴스
    """
    if persist_directory is None:
        persist_directory = CHROMA_DB_DIR
    else:
        persist_directory = Path(persist_directory)

    persist_directory.mkdir(parents=True, exist_ok=True)

    if embeddings is None:
        embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    return vectorstore
