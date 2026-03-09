"""문서 레지스트리 모듈

인제스트된 PDF 문서 목록을 data/documents.json에 영속 저장한다.
앱 재시작 후에도 유지되며, /ask 호출 시 pdf_hash 자동 조회에 사용된다.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentRegistry:
    """인제스트된 문서 목록을 JSON 파일로 관리하는 레지스트리"""

    def __init__(self, registry_path: Path) -> None:
        self._path = registry_path  # registry 저장 경로
        self._docs: list[dict] = self._load()  # json 타입으로 저장한 정보 로드

    def _load(self) -> list[dict]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("레지스트리 로드 실패, 빈 목록으로 시작: %s", e)
        return []

    def _save(self) -> None:
        self._path.parent.mkdir(
            parents=True, exist_ok=True
        )  # path = DATA_DIR / "documents.json", Data 폴더 생성, 있을 경우 생략
        self._path.write_text(
            json.dumps(self._docs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )  # "documents.json"에 json 타입으로 저장한 정보 저장

    def add(self, pdf_hash: str, filename: str, page_count: int) -> None:
        """문서를 레지스트리에 추가하고 파일에 저장한다.

        이미 존재하는 pdf_hash는 덮어쓴다.
        """
        self._docs = [
            d for d in self._docs if d["pdf_hash"] != pdf_hash
        ]  # "documents.json" 파일의 내용 중 pdf_hash와 다른 내용들만 저장
        self._docs.append(
            {
                "pdf_hash": pdf_hash,
                "filename": filename,
                "page_count": page_count,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
        )  # pdf의 정보룰 추가
        self._save()  # "documents.json 파일에 작성"
        logger.info("레지스트리 등록: %s (%s)", filename, pdf_hash[:8])

    def get(self, pdf_hash: str) -> dict | None:
        """pdf_hash로 문서 정보를 반환한다. 없으면 None."""
        for doc in self._docs:
            if doc["pdf_hash"] == pdf_hash:
                return doc
        return None

    def get_latest(self) -> dict | None:
        """ingested_at 기준 가장 최근 문서를 반환한다. 없으면 None."""
        if not self._docs:
            return None
        return max(
            self._docs, key=lambda d: d.get("ingested_at", "")
        )  # ingested_at elements를 읽어서 가장 최신의 데이터를 불러옴

    def list_all(self) -> list[dict]:
        """인제스트된 문서 전체 목록을 최신순으로 반환한다."""
        return sorted(self._docs, key=lambda d: d.get("ingested_at", ""), reverse=True)
