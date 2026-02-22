"""main.py 테스트 모듈 - FastAPI 엔드포인트 테스트"""

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

    @patch("src.main.build_graph")
    def test_accepts_pdf(self, mock_build_graph: MagicMock, client: TestClient, tmp_path: Path) -> None:
        """PDF 파일 업로드 성공"""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "image_paths": {1: "/tmp/page_001.png"},
            "contexts": [],
            "vision_result": "",
            "final_answer": "",
        })
        mock_build_graph.return_value = mock_graph

        with patch("src.main.PDF_DIR", tmp_path):
            response = client.post(
                "/ingest",
                files={"file": ("paper.pdf", b"%PDF-1.4 content", "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "paper.pdf 인제스트 완료"
        assert data["page_count"] == 1


class TestAskEndpoint:
    """POST /ask 엔드포인트 테스트"""

    def test_returns_404_for_missing_pdf(self, client: TestClient) -> None:
        """존재하지 않는 PDF 경로면 404"""
        response = client.post(
            "/ask",
            json={"question": "질문", "pdf_path": "/nonexistent/path.pdf"},
        )
        assert response.status_code == 404

    @patch("src.main.build_graph")
    def test_returns_answer(self, mock_build_graph: MagicMock, client: TestClient, tmp_path: Path) -> None:
        """정상적으로 답변 반환"""
        # 테스트 PDF 파일 생성
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "final_answer": "테스트 답변입니다.",
            "vision_result": "NO_VISION",
            "contexts": [],
            "image_paths": {},
        })
        mock_build_graph.return_value = mock_graph

        response = client.post(
            "/ask",
            json={"question": "요약해주세요", "pdf_path": str(pdf_path)},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "테스트 답변입니다."
        assert data["vision_result"] is None  # NO_VISION은 None으로 변환

    @patch("src.main.build_graph")
    def test_returns_vision_result(self, mock_build_graph: MagicMock, client: TestClient, tmp_path: Path) -> None:
        """비전 분석 결과가 있으면 포함"""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "final_answer": "답변",
            "vision_result": "테이블 분석 결과",
            "contexts": [],
            "image_paths": {},
        })
        mock_build_graph.return_value = mock_graph

        response = client.post(
            "/ask",
            json={"question": "테이블 설명", "pdf_path": str(pdf_path)},
        )

        data = response.json()
        assert data["vision_result"] == "테이블 분석 결과"
