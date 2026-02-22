"""main.py 테스트 모듈 - FastAPI 엔드포인트 테스트"""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client() -> TestClient:
    """FastAPI 테스트 클라이언트"""
    return TestClient(app, raise_server_exceptions=False)


class TestIngestEndpoint:
    """POST /ingest 엔드포인트 테스트"""

    def test_rejects_non_pdf(self, client: TestClient) -> None:
        """PDF가 아닌 파일은 400 에러"""
        response = client.post(
            "/ingest",
            files={"file": ("test.txt", b"content", "text/plain")},
        )
        assert response.status_code == 400

    @patch("src.main.build_ingest_graph")
    def test_accepts_pdf_and_returns_hash(
        self, mock_build_graph: MagicMock, client: TestClient, tmp_path: Path
    ) -> None:
        """신규 PDF 업로드 성공: pdf_hash, status='created' 반환"""
        # 그래프가 반환할 image_dir (중복 체크에 걸리지 않는 별도 경로)
        result_image_dir = tmp_path / "result_images"
        result_image_dir.mkdir()
        (result_image_dir / "page_001.png").touch()

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "image_dir": str(result_image_dir),
            "contexts": [],
            "vision_result": None,
            "final_answer": "",
        })
        mock_build_graph.return_value = mock_graph

        # IMAGE_DIR을 tmp_path로 패치 → hash 디렉토리 없음 → 신규로 처리됨
        with patch("src.main.IMAGE_DIR", tmp_path), patch("src.main.PDF_DIR", tmp_path):
            response = client.post(
                "/ingest",
                files={"file": ("paper.pdf", b"%PDF-1.4 content", "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "pdf_hash" in data
        assert len(data["pdf_hash"]) == 64  # SHA-256 hex digest 길이
        assert data["message"] == "paper.pdf 인제스트 완료"
        assert data["status"] == "created"
        assert data["page_count"] == 1

    @patch("src.main.build_ingest_graph")
    def test_skips_ingest_for_duplicate(
        self, mock_build_graph: MagicMock, client: TestClient, tmp_path: Path
    ) -> None:
        """같은 파일 재업로드 시 인제스트 건너뜀: status='already_exists' 반환"""
        file_bytes = b"%PDF-1.4 duplicate content"
        pdf_hash = hashlib.sha256(file_bytes).hexdigest()

        # 이미 처리된 것처럼 image_dir와 PNG 파일 생성
        image_dir = tmp_path / pdf_hash
        image_dir.mkdir(parents=True)
        (image_dir / "page_001.png").touch()
        (image_dir / "page_002.png").touch()

        with patch("src.main.IMAGE_DIR", tmp_path), patch("src.main.PDF_DIR", tmp_path):
            response = client.post(
                "/ingest",
                files={"file": ("paper.pdf", file_bytes, "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "already_exists"
        assert data["pdf_hash"] == pdf_hash
        assert data["page_count"] == 2
        mock_build_graph.assert_not_called()  # 인제스트 그래프 실행 없음


class TestAskEndpoint:
    """POST /ask 엔드포인트 테스트"""

    def test_returns_404_for_missing_hash(self, client: TestClient) -> None:
        """존재하지 않는 pdf_hash면 404"""
        response = client.post(
            "/ask",
            json={"question": "질문", "pdf_hash": "nonexistent_hash_that_does_not_exist"},
        )
        assert response.status_code == 404

    @patch("src.main.build_query_graph")
    def test_returns_answer(
        self, mock_build_graph: MagicMock, client: TestClient, tmp_path: Path
    ) -> None:
        """정상적으로 답변 반환"""
        # image_dir 생성 (404 방지)
        image_dir = tmp_path / "abc123"
        image_dir.mkdir()

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "final_answer": "테스트 답변입니다.",
            "vision_result": "NO_VISION",
            "contexts": [],
            "image_dir": str(image_dir),
        })
        mock_build_graph.return_value = mock_graph

        with patch("src.main.IMAGE_DIR", tmp_path):
            response = client.post(
                "/ask",
                json={"question": "요약해주세요", "pdf_hash": "abc123"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "테스트 답변입니다."
        assert data["vision_result"] is None  # NO_VISION은 None으로 변환

    @patch("src.main.build_query_graph")
    def test_returns_vision_result(
        self, mock_build_graph: MagicMock, client: TestClient, tmp_path: Path
    ) -> None:
        """비전 분석 결과가 있으면 포함"""
        image_dir = tmp_path / "abc123"
        image_dir.mkdir()

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "final_answer": "답변",
            "vision_result": "테이블 분석 결과",
            "contexts": [],
            "image_dir": str(image_dir),
        })
        mock_build_graph.return_value = mock_graph

        with patch("src.main.IMAGE_DIR", tmp_path):
            response = client.post(
                "/ask",
                json={"question": "테이블 설명", "pdf_hash": "abc123"},
            )

        data = response.json()
        assert data["vision_result"] == "테이블 분석 결과"
