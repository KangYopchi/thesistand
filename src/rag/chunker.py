"""계층적 청킹 모듈 - Parent/Child 구조 기반 문서 분할

Parent(Section 단위)와 Child(~400 tokens) 청크를 생성하고,
ParentDocumentRetriever를 통해 검색 시 부모 문맥을 함께 반환한다.
부모 문서는 DocumentFileStore를 통해 JSON 직렬화하여 영속 저장된다.
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers.parent_document_retriever import (
    ParentDocumentRetriever,
)
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.parser import ParsedElement

# ── Document 직렬화 저장소 ─────────────────────────────────────────────


class DocumentFileStore(BaseStore[str, Document]):
    """LocalFileStore(bytes)를 감싸 Document 객체를 JSON으로 직렬화/역직렬화하는 래퍼.

    ParentDocumentRetriever의 docstore는 BaseStore[str, Document]를 기대하지만,
    LocalFileStore는 BaseStore[str, bytes]이므로 이 래퍼로 타입을 맞춘다.
    """

    def __init__(self, file_store: LocalFileStore) -> None:
        self._store = file_store

    def mget(self, keys: Sequence[str]) -> list[Document | None]:
        raw_values = self._store.mget(list(keys))
        results: list[Document | None] = []
        for raw in raw_values:
            if raw is None:
                results.append(None)
            else:
                data = json.loads(raw.decode("utf-8"))
                results.append(
                    Document(
                        page_content=data["page_content"],
                        metadata=data.get("metadata", {}),
                    )
                )
        return results

    def mset(self, key_value_pairs: Sequence[tuple[str, Document]]) -> None:
        serialized = [
            (
                k,
                json.dumps(
                    {"page_content": v.page_content, "metadata": v.metadata},
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            for k, v in key_value_pairs
        ]
        self._store.mset(serialized)

    def mdelete(self, keys: Sequence[str]) -> None:
        self._store.mdelete(list(keys))

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        yield from self._store.yield_keys(prefix=prefix)


# ── 설정 ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PARENT_STORE_DIR = DATA_DIR / "parent_store"

# Child: 고밀도 벡터 검색용 (~400 tokens ≈ 1600 chars)
CHILD_CHUNK_SIZE = 1600
CHILD_CHUNK_OVERLAP = 200

# Parent: Section 단위 문맥 제공용 (~2000 tokens ≈ 8000 chars)
PARENT_CHUNK_SIZE = 8000
PARENT_CHUNK_OVERLAP = 400


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────


def elements_to_documents(
    elements: list[ParsedElement], pdf_name: str
) -> list[Document]:
    """ParsedElement 리스트를 LangChain Document 리스트로 변환

    Args:
        elements: parser.py에서 추출된 요소 리스트
        pdf_name: PDF 파일명 (확장자 제외)

    Returns:
        메타데이터가 포함된 Document 리스트
    """
    documents: list[Document] = []

    for elem in elements:
        text = elem["text"].strip()

        if not text:
            # 이미지/그림/표 element는 텍스트가 없어도 placeholder로 저장.
            # 버리면 벡터DB에 시각 요소 기록이 전혀 남지 않는다.
            if elem["element_type"] in ("image", "figure"):
                text = f"[그림 - 페이지 {elem['page_number']}]"
            elif elem["element_type"] == "table":
                text = f"[표 - 페이지 {elem['page_number']}]"
            else:
                continue

        doc = Document(
            page_content=text,
            metadata={
                "source": pdf_name,
                "page_number": elem["page_number"],
                "element_type": elem["element_type"],
            },
        )
        documents.append(doc)

    return documents


def create_parent_document_retriever(
    vectorstore: Chroma,
    embeddings: Embeddings,
    parent_store_path: Path | str | None = None,
) -> ParentDocumentRetriever:
    """ParentDocumentRetriever 인스턴스를 생성

    Args:
        vectorstore: ChromaDB 벡터스토어 인스턴스
        embeddings: 임베딩 모델 (사용되지 않지만 확장성 위해 유지)
        parent_store_path: 부모 문서 저장 경로 (기본: data/parent_store/)

    Returns:
        ParentDocumentRetriever 인스턴스
    """
    if parent_store_path is None:
        parent_store_path = PARENT_STORE_DIR
    else:
        parent_store_path = Path(parent_store_path)

    parent_store_path.mkdir(parents=True, exist_ok=True)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )

    doc_store = DocumentFileStore(LocalFileStore(str(parent_store_path)))

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=doc_store,  # type: ignore[arg-type]
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever


async def add_documents_to_retriever(
    retriever: ParentDocumentRetriever,
    elements: list[ParsedElement],
    pdf_name: str,
) -> int:
    """파싱된 요소들을 Parent/Child 구조로 분할하여 저장

    Args:
        retriever: ParentDocumentRetriever 인스턴스
        elements: parser.py에서 추출된 요소 리스트
        pdf_name: PDF 파일명 (확장자 제외)

    Returns:
        저장된 Document 수
    """
    documents = elements_to_documents(elements, pdf_name)

    if not documents:
        return 0

    await retriever.aadd_documents(documents)

    return len(documents)
