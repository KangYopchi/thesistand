# asyncio.gather 란? — 12살도 이해하는 설명

`src/rag/parser.py` 149–152번 줄에 있는 코드를 쉽게 풀어 쓴 문서예요.

---

## 사용하는 코드

```python
json_result, page_images = await asyncio.gather(
    parse_pdf_with_llamaparse(pdf_path),
    generate_page_images(pdf_path),
)
```

---

## 1. .gather를 쓰는 이유 (왜 필요한가?)

**한 줄 요약:** “두 가지 일을 **동시에** 하기 위해서”예요.

- **일 1:** PDF를 LlamaParse로 분석해서 **글/구조 정보**(JSON) 가져오기 → `parse_pdf_with_llamaparse`
- **일 2:** 같은 PDF에서 **페이지마다 이미지** 만들기 → `generate_page_images`

이 두 일은 **서로 독립적**이에요.  
한쪽이 끝나야 다른 쪽을 시작할 필요가 없어요.

- **gather 없이 하면:**  
  “1번 끝나고 → 2번 시작” → **총 시간 = 1번 시간 + 2번 시간**
- **gather로 하면:**  
  “1번이랑 2번 동시에 시작해서 둘 다 끝날 때까지 기다림” → **총 시간 ≈ 둘 중 더 오래 걸리는 쪽**

그래서 **빠르게 하려고** `.gather`를 써요.

---

## 2. 실행 과정 (어떤 순서로 돌아가나?)

쉽게 말하면 이렇게 돼요.

1. **시작**
   - `parse_pdf_with_llamaparse(pdf_path)` → “PDF 텍스트/구조 분석” 작업 예약
   - `generate_page_images(pdf_path)` → “페이지 이미지 만들기” 작업 예약  
   → **두 작업이 거의 동시에 시작**돼요.

2. **진행**
   - 둘 다 **동시에** 돌아가요 (네트워크 대기, 이미지 변환 등).
   - 한 작업이 I/O에서 기다리는 동안 다른 작업이 CPU를 쓸 수 있어요.

3. **끝날 때**
   - **둘 다 끝날 때까지** `await asyncio.gather(...)`에서 기다려요.
   - 끝나면 결과가 **순서대로** 나와요:
     - 첫 번째 자리 → `parse_pdf_with_llamaparse` 결과 → `json_result`
     - 두 번째 자리 → `generate_page_images` 결과 → `page_images`

4. **그 다음**
   - `json_result`와 `page_images`를 받아서, `extract_elements_with_page_numbers(json_result)` 같은 걸 하고, 최종 `ParseResult`를 만들어요.

---

## 3. 결과 (뭘 받나?)

`gather`가 반환하는 건 **두 개**예요. 왼쪽부터 1번, 2번 자리예요.

| 변수          | 의미 |
|---------------|------|
| `json_result` | LlamaParse가 PDF를 분석한 **JSON 형태 결과** (글, 제목, 표 등 구조 정보) |
| `page_images` | 페이지 번호 → 이미지 파일 경로를 담은 **딕셔너리** (예: `{1: Path(...), 2: Path(...)}`) |

그래서 이렇게 쓸 수 있어요:

- `json_result` → 텍스트/요소 목록 만들 때 사용
- `page_images` → 나중에 비전/이미지 분석할 때 “몇 페이지 이미지”로 넘길 때 사용

---

## 비유로 정리

- **일 1:** 친구 A한테 “이 PDF 요약해 줘” 시킴  
- **일 2:** 친구 B한테 “이 PDF 페이지마다 그림으로 저장해 줘” 시킴  

`.gather`는  
“A한테도 시키고, B한테도 시키고, **둘 다 끝날 때까지 기다렸다가**, A 결과랑 B 결과를 **한 번에** 받는 것”이에요.  
한 명 끝날 때까지 기다린 다음에 다른 한 명한테 시키는 것보다 **훨씬 빨라요.**

---

## 복사용 요약 (한 줄씩)

- **이유:** 두 작업(파싱 + 이미지 생성)을 **동시에** 돌려서 **시간을 줄이려고**.
- **과정:** 두 작업을 동시에 시작 → 둘 다 끝날 때까지 대기 → 결과를 **순서대로** 두 개 받음.
- **결과:** `json_result`(파싱 결과), `page_images`(페이지별 이미지 경로).

이걸 그대로 마크다운이나 메모에 복사해서 써도 돼요.
