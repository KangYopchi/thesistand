"""chunker.py 테스트 모듈"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.rag.chunker import (
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_SIZE,
    add_documents_to_retriever,
    create_parent_document_retriever,
    elements_to_documents,
)
from src.rag.parser import ParsedElement


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_elements() -> list[ParsedElement]:
    """테스트용 ParsedElement 리스트"""
    return [
        {"page_number": 1, "text": "Introduction to the paper", "element_type": "heading"},
        {"page_number": 1, "text": "This is the first paragraph.", "element_type": "text"},
        {"page_number": 2, "text": "Methods section content.", "element_type": "text"},
        {"page_number": 2, "text": "", "element_type": "text"},  # 빈 텍스트 - 필터링 대상
        {"page_number": 3, "text": "   ", "element_type": "text"},  # 공백만 - 필터링 대상
    ]


# ── elements_to_documents ─────────────────────────────────────────────


class TestElementsToDocuments:
    """ParsedElement → Document 변환 테스트"""

    def test_basic_conversion(self, sample_elements: list[ParsedElement]) -> None:
        """기본 변환: 빈 텍스트 요소는 필터링됨"""
        docs = elements_to_documents(sample_elements, "test_paper")

        # 빈 텍스트 2개 제외 → 3개
        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)

    def test_metadata_preserved(self, sample_elements: list[ParsedElement]) -> None:
        """메타데이터가 올바르게 설정되는지 확인"""
        docs = elements_to_documents(sample_elements, "my_paper")

        assert docs[0].metadata["source"] == "my_paper"
        assert docs[0].metadata["page_number"] == 1
        assert docs[0].metadata["element_type"] == "heading"

    def test_page_content_matches(self, sample_elements: list[ParsedElement]) -> None:
        """page_content가 원본 텍스트와 일치하는지 확인"""
        docs = elements_to_documents(sample_elements, "paper")

        assert docs[0].page_content == "Introduction to the paper"
        assert docs[2].page_content == "Methods section content."

    def test_empty_elements_returns_empty(self) -> None:
        """빈 리스트 입력 시 빈 리스트 반환"""
        docs = elements_to_documents([], "empty")
        assert docs == []

    def test_all_empty_text_filtered(self) -> None:
        """모든 요소의 텍스트가 빈 경우 빈 리스트 반환"""
        elements: list[ParsedElement] = [
            {"page_number": 1, "text": "", "element_type": "text"},
            {"page_number": 2, "text": "   ", "element_type": "text"},
        ]
        docs = elements_to_documents(elements, "paper")
        assert docs == []


# ── create_parent_document_retriever ──────────────────────────────────


class TestCreateParentDocumentRetriever:
    """ParentDocumentRetriever 생성 테스트"""

    def test_creates_retriever_with_default_path(self, tmp_path: Path) -> None:
        """기본 경로로 retriever가 정상 생성되는지 확인"""
        mock_vectorstore = MagicMock(spec=VectorStore)
        mock_embeddings = MagicMock()

        with patch("src.rag.chunker.PARENT_STORE_DIR", tmp_path / "parent_store"):
            retriever = create_parent_document_retriever(
                vectorstore=mock_vectorstore,
                embeddings=mock_embeddings,
            )

        assert retriever is not None
        assert (tmp_path / "parent_store").exists()

    def test_creates_retriever_with_custom_path(self, tmp_path: Path) -> None:
        """커스텀 경로로 retriever가 정상 생성되는지 확인"""
        custom_path = tmp_path / "custom_store"
        mock_vectorstore = MagicMock(spec=VectorStore)
        mock_embeddings = MagicMock()

        retriever = create_parent_document_retriever(
            vectorstore=mock_vectorstore,
            embeddings=mock_embeddings,
            parent_store_path=custom_path,
        )

        assert retriever is not None
        assert custom_path.exists()

    def test_splitter_sizes_configured(self) -> None:
        """청크 사이즈 상수가 올바르게 설정되었는지 확인"""
        assert CHILD_CHUNK_SIZE == 1600
        assert PARENT_CHUNK_SIZE == 8000


# ── add_documents_to_retriever ────────────────────────────────────────


class TestAddDocumentsToRetriever:
    """문서 추가 기능 테스트"""

    async def test_adds_documents_and_returns_count(
        self, sample_elements: list[ParsedElement]
    ) -> None:
        """문서가 추가되고 올바른 개수가 반환되는지 확인"""
        mock_retriever = MagicMock()
        mock_retriever.aadd_documents = AsyncMock()

        count = await add_documents_to_retriever(
            retriever=mock_retriever,
            elements=sample_elements,
            pdf_name="test_paper",
        )

        # 빈 텍스트 2개 제외 → 3개
        assert count == 3
        mock_retriever.aadd_documents.assert_awaited_once()

        # aadd_documents에 전달된 Document 리스트 확인
        call_args = mock_retriever.aadd_documents.call_args[0][0]
        assert len(call_args) == 3
        assert all(isinstance(d, Document) for d in call_args)

    async def test_empty_elements_returns_zero(self) -> None:
        """빈 요소 리스트 입력 시 0 반환, aadd_documents 호출 안됨"""
        mock_retriever = MagicMock()
        mock_retriever.aadd_documents = AsyncMock()

        count = await add_documents_to_retriever(
            retriever=mock_retriever,
            elements=[],
            pdf_name="empty",
        )

        assert count == 0
        mock_retriever.aadd_documents.assert_not_awaited()
