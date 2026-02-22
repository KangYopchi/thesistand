"""parser.py 테스트 모듈"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.parser import (
    extract_elements_with_page_numbers,
    generate_page_images,
    parse_pdf,
    parse_pdf_with_llamaparse,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_json_result() -> list[dict]:
    """LlamaParse가 반환하는 JSON 형태의 샘플 데이터"""
    return [
        {
            "pages": [
                {
                    "page": 1,
                    "items": [
                        {"type": "heading", "value": "Introduction"},
                        {"type": "text", "value": "This is the first paragraph."},
                    ],
                },
                {
                    "page": 2,
                    "items": [
                        {"type": "text", "value": "Second page content."},
                        {"type": "table", "value": "col1 | col2"},
                    ],
                },
            ]
        }
    ]


@pytest.fixture
def multi_doc_json_result() -> list[dict]:
    """여러 문서가 포함된 JSON 결과"""
    return [
        {
            "pages": [
                {
                    "page": 1,
                    "items": [{"type": "text", "value": "Doc1 Page1"}],
                }
            ]
        },
        {
            "pages": [
                {
                    "page": 1,
                    "items": [{"type": "text", "value": "Doc2 Page1"}],
                }
            ]
        },
    ]


# ── extract_elements_with_page_numbers ────────────────────────────────


class TestExtractElementsWithPageNumbers:
    """순수 함수 extract_elements_with_page_numbers 테스트"""

    def test_basic_extraction(self, sample_json_result: list[dict]) -> None:
        """기본 추출: 4개 아이템이 올바르게 추출되는지 확인"""
        elements = extract_elements_with_page_numbers(sample_json_result)

        assert len(elements) == 4
        assert elements[0] == {
            "page_number": 1,
            "text": "Introduction",
            "element_type": "heading",
        }
        assert elements[2]["page_number"] == 2
        assert elements[3]["element_type"] == "table"

    def test_empty_input(self) -> None:
        """빈 리스트 입력 시 빈 리스트 반환"""
        assert extract_elements_with_page_numbers([]) == []

    def test_doc_without_pages_key(self) -> None:
        """pages 키가 없는 문서는 건너뜀"""
        result = extract_elements_with_page_numbers([{"no_pages": True}])
        assert result == []

    def test_page_without_items(self) -> None:
        """items 키가 없는 페이지는 건너뜀"""
        json_result = [{"pages": [{"page": 1}]}]
        result = extract_elements_with_page_numbers(json_result)
        assert result == []

    def test_missing_item_fields_use_defaults(self) -> None:
        """item에 value/type 키가 없으면 기본값 사용"""
        json_result = [{"pages": [{"page": 3, "items": [{}]}]}]
        elements = extract_elements_with_page_numbers(json_result)

        assert len(elements) == 1
        assert elements[0] == {
            "page_number": 3,
            "text": "",
            "element_type": "text",
        }

    def test_default_page_number(self) -> None:
        """page 키가 없으면 기본값 1 사용"""
        json_result = [{"pages": [{"items": [{"type": "text", "value": "hello"}]}]}]
        elements = extract_elements_with_page_numbers(json_result)
        assert elements[0]["page_number"] == 1

    def test_multi_doc(self, multi_doc_json_result: list[dict]) -> None:
        """여러 문서의 요소가 모두 추출되는지 확인"""
        elements = extract_elements_with_page_numbers(multi_doc_json_result)

        assert len(elements) == 2
        assert elements[0]["text"] == "Doc1 Page1"
        assert elements[1]["text"] == "Doc2 Page1"


# ── parse_pdf_with_llamaparse ─────────────────────────────────────────


class TestParsePdfWithLlamaparse:
    """LlamaParse API 호출 함수 테스트 (mock 사용)"""

    async def test_missing_api_key_raises(self) -> None:
        """API 키가 없으면 ValueError 발생"""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="LLAMA_CLOUD_API_KEY"):
                await parse_pdf_with_llamaparse("/fake/path.pdf")

    async def test_calls_llamaparse_and_returns_result(self) -> None:
        """LlamaParse.aget_json이 호출되고 결과가 반환되는지 확인"""
        fake_result = [{"pages": []}]

        with patch.dict("os.environ", {"LLAMA_CLOUD_API_KEY": "test-key"}):
            with patch("src.rag.parser.LlamaParse") as MockParser:
                mock_instance = MagicMock()
                mock_instance.aget_json = AsyncMock(return_value=fake_result)
                MockParser.return_value = mock_instance

                result = await parse_pdf_with_llamaparse("/fake/path.pdf")

        assert result == fake_result
        MockParser.assert_called_once_with(
            api_key="test-key",
            result_type="json",
            verbose=True,
        )
        mock_instance.aget_json.assert_awaited_once_with("/fake/path.pdf")


# ── generate_page_images ──────────────────────────────────────────────


class TestGeneratePageImages:
    """PDF → 이미지 변환 함수 테스트 (mock 사용)"""

    async def test_generates_images_with_default_output_dir(
        self, tmp_path: Path
    ) -> None:
        """기본 output_dir 사용 시 IMAGE_DIR/{pdf_name}/ 에 저장"""
        fake_images = [MagicMock(), MagicMock()]

        with patch("src.rag.parser.convert_from_path", return_value=fake_images):
            with patch("src.rag.parser.IMAGE_DIR", tmp_path):
                result = await generate_page_images("/fake/sample.pdf")

        assert len(result) == 2
        assert result[1] == tmp_path / "sample" / "page_001.png"
        assert result[2] == tmp_path / "sample" / "page_002.png"
        # 이미지 save 호출 확인
        for img in fake_images:
            img.save.assert_called_once()

    async def test_custom_output_dir(self, tmp_path: Path) -> None:
        """커스텀 output_dir 지정 시 해당 경로에 저장"""
        custom_dir = tmp_path / "custom"
        fake_images = [MagicMock()]

        with patch("src.rag.parser.convert_from_path", return_value=fake_images):
            result = await generate_page_images("/fake/test.pdf", output_dir=custom_dir)

        assert result[1] == custom_dir / "page_001.png"
        assert custom_dir.exists()

    async def test_empty_pdf_returns_empty_dict(self, tmp_path: Path) -> None:
        """페이지가 없는 PDF는 빈 딕셔너리 반환"""
        with patch("src.rag.parser.convert_from_path", return_value=[]):
            with patch("src.rag.parser.IMAGE_DIR", tmp_path):
                result = await generate_page_images("/fake/empty.pdf")

        assert result == {}


# ── parse_pdf (통합 함수) ─────────────────────────────────────────────


class TestParsePdf:
    """parse_pdf 통합 함수 테스트 (하위 함수 mock)"""

    async def test_combines_parsing_and_images(self) -> None:
        """파싱 결과와 이미지 결과가 올바르게 합쳐지는지 확인"""
        fake_json = [
            {
                "pages": [
                    {
                        "page": 1,
                        "items": [{"type": "text", "value": "Hello"}],
                    }
                ]
            }
        ]
        fake_images = {1: Path("/img/page_001.png")}

        with (
            patch(
                "src.rag.parser.parse_pdf_with_llamaparse",
                new_callable=AsyncMock,
                return_value=fake_json,
            ),
            patch(
                "src.rag.parser.generate_page_images",
                new_callable=AsyncMock,
                return_value=fake_images,
            ),
        ):
            result = await parse_pdf("/fake/paper.pdf")

        assert result["pdf_name"] == "paper"
        assert len(result["elements"]) == 1
        assert result["elements"][0]["text"] == "Hello"
        assert result["page_images"] == fake_images
