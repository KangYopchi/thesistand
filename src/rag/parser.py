"""PDF 파싱 모듈 - LlamaParse JSON 모드 + 페이지 이미지 생성"""

import asyncio
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from llama_parse import LlamaParse
from pdf2image import convert_from_path

load_dotenv()

# 설정
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
IMAGE_DIR = DATA_DIR / "images"


class ParsedElement(TypedDict):
    """파싱된 요소의 타입 정의"""

    page_number: int
    text: str
    element_type: str


class ParseResult(TypedDict):
    """파싱 결과의 타입 정의"""

    pdf_name: str
    elements: list[ParsedElement]
    page_images: dict[int, Path]


async def parse_pdf_with_llamaparse(pdf_path: Path | str) -> list[dict]:
    """
    LlamaParse를 JSON 결과 모드로 호출하여 PDF 파싱

    Args:
        pdf_path: PDF 파일 경로

    Returns:
        파싱된 JSON 결과 리스트
    """
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY 환경변수가 설정되지 않았습니다.")

    parser = LlamaParse(
        api_key=api_key,
        result_type="json",
        verbose=True,
    )

    pdf_path = Path(pdf_path)
    json_result: list[dict] = await parser.aget_json(str(pdf_path))

    return json_result


def extract_elements_with_page_numbers(json_result: list[dict]) -> list[ParsedElement]:
    """
    json 결과에서 텍스트와 page_number 추출

    Args:
        json_result: LlamaParse JSON 결과

    Returns:
        ParsedElement 리스트
    """
    elements: list[ParsedElement] = []

    for doc in json_result:
        pages = doc.get("pages", [])
        for page in pages:
            page_number = page.get("page", 1)
            items = page.get("items", [])

            for item in items:
                element: ParsedElement = {
                    "page_number": page_number,
                    "text": item.get("value", ""),
                    "element_type": item.get("type", "text"),
                }
                elements.append(element)

    return elements


async def generate_page_images(
    pdf_path: Path | str,
    output_dir: Path | str | None = None,
    dpi: int = 200,
) -> dict[int, Path]:
    """
    PDF의 각 페이지를 이미지로 변환하여 저장

    Args:
        pdf_path: PDF 파일 경로
        output_dir: 이미지 저장 디렉토리 (기본: data/images/{pdf_name}/)
        dpi: 이미지 해상도

    Returns:
        {page_number: image_path} 딕셔너리
    """
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem

    if output_dir is None:
        output_dir = IMAGE_DIR / pdf_name
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()
    images = await loop.run_in_executor(
        None,
        lambda: convert_from_path(str(pdf_path), dpi=dpi),
    )

    page_images: dict[int, Path] = {}

    for page_num, image in enumerate(images, start=1):
        image_path = output_dir / f"page_{page_num:03d}.png"
        await loop.run_in_executor(
            None,
            lambda img=image, path=image_path: img.save(str(path), "PNG"),
        )
        page_images[page_num] = image_path

    return page_images


async def parse_pdf(pdf_path: Path | str) -> ParseResult:
    """
    PDF 파싱 + 페이지 이미지 생성 통합 함수

    Args:
        pdf_path: PDF 파일 경로

    Returns:
        ParseResult (elements, page_images 포함)
    """
    pdf_path = Path(pdf_path)

    # 병렬로 LlamaParse 파싱과 이미지 생성 실행
    gathered: tuple[list[dict], dict[int, Path]] = await asyncio.gather(
        parse_pdf_with_llamaparse(pdf_path),
        generate_page_images(pdf_path),
    )

    # gathered tuple을 언팩한다. 두 변수에 대한 type hinting은 3.12.12 버전에서 작동하지 않으므로 명시하지 않는다.
    json_result, page_images = gathered

    elements = extract_elements_with_page_numbers(json_result)

    return ParseResult(
        pdf_name=pdf_path.stem,
        elements=elements,
        page_images=page_images,
    )


async def main() -> None:
    """테스트용 메인 함수"""
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"PDF 파일이 없습니다. {PDF_DIR}에 PDF를 추가하세요.")
        return

    pdf_path = pdf_files[0]
    print(f"파싱 시작: {pdf_path}")

    result = await parse_pdf(pdf_path)

    print("\n=== 파싱 결과 ===")
    print(f"PDF: {result['pdf_name']}")
    print(f"총 요소 수: {len(result['elements'])}")
    print(f"총 페이지 수: {len(result['page_images'])}")

    print("\n=== 처음 5개 요소 ===")
    for elem in result["elements"][:5]:
        text_preview = (
            elem["text"][:50] + "..." if len(elem["text"]) > 50 else elem["text"]
        )
        print(f"  [p.{elem['page_number']}] ({elem['element_type']}) {text_preview}")

    print("\n=== 생성된 이미지 ===")
    for page_num, image_path in sorted(result["page_images"].items()):
        print(f"  페이지 {page_num}: {image_path}")


if __name__ == "__main__":
    asyncio.run(main())
