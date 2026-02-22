"""vectorstore.py 테스트 모듈"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_chroma import Chroma

from src.rag.vectorstore import COLLECTION_NAME, get_embeddings, get_vectorstore


# ── get_embeddings ────────────────────────────────────────────────────


class TestGetEmbeddings:
    """OpenAI 임베딩 생성 테스트"""

    def test_missing_api_key_raises(self) -> None:
        """OPENAI_API_KEY가 없으면 ValueError 발생"""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                get_embeddings()

    def test_returns_embeddings_with_valid_key(self) -> None:
        """API 키가 있으면 OpenAIEmbeddings 인스턴스 반환"""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            embeddings = get_embeddings()

        assert embeddings is not None
        assert embeddings.model == "text-embedding-3-small"


# ── get_vectorstore ───────────────────────────────────────────────────


class TestGetVectorstore:
    """ChromaDB 벡터스토어 생성 테스트"""

    def test_creates_vectorstore_with_defaults(self, tmp_path: Path) -> None:
        """기본 설정으로 벡터스토어 생성"""
        mock_embeddings = MagicMock()

        with patch("src.rag.vectorstore.CHROMA_DB_DIR", tmp_path / "chroma"):
            vs = get_vectorstore(embeddings=mock_embeddings)

        assert isinstance(vs, Chroma)
        assert (tmp_path / "chroma").exists()

    def test_creates_vectorstore_with_custom_path(self, tmp_path: Path) -> None:
        """커스텀 경로로 벡터스토어 생성"""
        custom_dir = tmp_path / "custom_chroma"
        mock_embeddings = MagicMock()

        vs = get_vectorstore(
            persist_directory=custom_dir,
            embeddings=mock_embeddings,
        )

        assert isinstance(vs, Chroma)
        assert custom_dir.exists()

    def test_custom_collection_name(self, tmp_path: Path) -> None:
        """커스텀 컬렉션 이름으로 벡터스토어 생성"""
        mock_embeddings = MagicMock()

        vs = get_vectorstore(
            collection_name="my_collection",
            persist_directory=tmp_path,
            embeddings=mock_embeddings,
        )

        assert isinstance(vs, Chroma)

    def test_default_collection_name(self) -> None:
        """기본 컬렉션 이름이 'thesistand'인지 확인"""
        assert COLLECTION_NAME == "thesistand"
