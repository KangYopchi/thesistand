# 개념 정리 노트

---

## 엔드포인트 (EndPoint)

**FastAPI의 엔드포인트**는 개발자가 정의한 URL과 HTTP 메서드(GET, POST, PUT, PATCH, DELETE 등)의 조합에 따라 실행되는 동작을 정의한 것이다. HTTP 메서드란 클라이언트(웹/앱)가 서버에 어떤 행동을 요청할지를 나타내며, 데이터를 **조회(GET)**, **생성(POST)**, **수정(PUT/PATCH)**, **삭제(DELETE)** 하는 동작을 한다

## 응답 데이터 직렬화 (Serialization)

응답 데이터 직렬화란 **Pydantic 등으로 정의한 데이터 객체**를 **JSON 등의 형태로 변환**하는 것으로, 서버에서 클라이언트로 응답할 때 가장 자주 사용된다

---

## 개념 정리 vs 코드에서 왜 필요한가 — 비교 예시

개념을 정의하는 것과, 실제 코드에서 그 줄이 왜 존재하는지를 설명하는 것은 다르다.

**개념 정리**
> 엔드포인트는 URL과 HTTP 메서드의 조합이다.

**코드에서 왜 필요한가**
> main.py:90의 `@app.post('/ingest')`가 없으면
> PDF 업로드 요청이 어디로 가야 할지 서버가 모른다.

두 가지를 같이 쓰는 것이 목표다.

```python
# [내 정리]
# HTTP POST 요청이 /ingest 경로로 오면 이 함수를 실행하라는 데코레이터.
# 이 줄이 없으면 클라이언트가 /ingest로 PDF를 보내도
# 서버가 어떤 함수를 실행할지 몰라서 404 에러가 난다.
@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile) -> IngestResponse:
```
