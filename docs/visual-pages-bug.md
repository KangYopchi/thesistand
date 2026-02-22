# visual_pages 항상 비어있는 버그 분석

## 문제 현상

"Figure 1 그림 보여줘"라고 질문하면 AI가 페이지 1~5만 분석하고
"해당 페이지에 그림이 없다"고 답변하는 현상.

---

## 도서관 사서 비유로 이해하기

시스템 전체를 **도서관**이라고 생각하자.

---

## 1단계: 인제스트 — 책을 도서관에 넣는 과정

PDF를 업로드하면 이런 일이 일어난다:

```
PDF 파일
  └── LlamaParse가 읽음
        ├── [텍스트] "Figure 1 provides a comprehensive overview..." (page 5)
        ├── [텍스트] "Context Engineering의 개요는..." (page 5)
        ├── [이미지] "" ← 내용이 비어있음! (page 6)
        └── [텍스트] "As shown in the figure..." (page 7)
```

그런데 `chunker.py`에 이런 코드가 있다:

```python
# chunker.py:105
if not elem["text"].strip():
    continue  # ← "텍스트가 없으면 그냥 버려!"
```

**비유**: 사서가 책 목록을 만들 때, "내용이 없는 건 목록에 안 넣어도 되겠지?"라고 생각하고
**그림 페이지를 목록에서 빼버린 것.**

결과:
```
벡터DB에 저장됨 ✅
  ├── [텍스트] "Figure 1 provides..." (page 5)
  └── [텍스트] "Context Engineering의 개요는..." (page 5)

벡터DB에서 사라짐 ❌
  └── [이미지] (page 6) ← 이게 실제 그림인데!
```

---

## 2단계: 검색 — 질문에 맞는 내용 찾기

"그림을 보여줘"라고 물으면:

```
질문: "Context Engineering 그림 보여줘"
       ↓
벡터DB에서 비슷한 내용 검색
       ↓
검색 결과:
  ✅ [텍스트] "Figure 1 provides..." (page 5, element_type="text")
  ✅ [텍스트] "Context Engineering의 개요는..." (page 5, element_type="text")
  ❌ [이미지] (page 6) ← 애초에 저장이 안 됐으니 검색될 리가 없음
```

**비유**: 도서관에서 "그림이 있는 페이지 찾아줘"라고 했는데, 사서가 **그림 목록을 버렸기 때문에**
"Figure 1이 있다는 설명이 쓰여진 페이지(5쪽)"만 찾아준다.
실제 그림(6쪽)은 목록에 없으니 찾을 수가 없다.

---

## 3단계: vision_analyst — "어느 페이지 이미지를 AI에게 보여줄까?" 결정

여기서 **핵심 버그**가 나온다.

```python
# nodes.py:284~291
visual_pages = sorted(
    {
        chunk["page_number"]
        for chunk in contexts
        if chunk.get("element_type") in ("table", "image")  # ← 이게 문제!
        and chunk.get("page_number")
    }
)
```

**이 코드가 하는 말**: "검색된 결과 중에서 `element_type`이 'image'나 'table'인 것만 페이지 번호를 뽑아!"

**실제 검색 결과**:
```python
contexts = [
    {"content": "Figure 1 provides...", "page_number": 5, "element_type": "text"},  # ← "text"야!
    {"content": "Context Engineering...", "page_number": 5, "element_type": "text"}, # ← "text"야!
]
```

검색된 chunk는 전부 `element_type = "text"`다. "image"나 "table"인 게 하나도 없다.

```python
visual_pages = sorted({...})
# 결과: [] ← 빈 리스트!
```

**비유**: 사서에게 "그림이 표시된 카드만 골라줘"라고 했는데, 카드에는 전부 "텍스트"라고 적혀 있다.
그림 카드는 처음부터 목록에 없었으니까. 결국 아무 카드도 못 고른다.

---

## 4단계: 어쩔 수 없이 쓰는 fallback

`visual_pages`가 비어있으니까 이 코드가 실행된다:

```python
if visual_pages:
    # 이건 실행 안 됨
    candidate_paths = [image_dir_path / f"page_{p:03d}.png" for p in visual_pages[:5]]
else:
    # 이게 실행됨 ← "그냥 첫 5페이지 쓰자"
    candidate_paths = sorted(image_dir_path.glob("page_*.png"))[:5]
```

**결과**: AI가 보는 이미지는 **page_001.png ~ page_005.png** (표지, 목차, 서론...)

실제 Figure 1은 page_006.png인데 AI는 그걸 못 본다.

---

## 전체 흐름 요약

```
[사용자 질문] "Figure 1 그림 보여줘"
       ↓
[1] 검색 → "Figure 1을 언급한 텍스트" 발견 (page 5, element_type="text")
       ↓
[2] 비전 라우터 → "figure" 키워드 있음 → NEED_VISION ✅
       ↓
[3] visual_pages 계산 →
    "element_type이 image인 게 있나?" → 없음 ❌
    → visual_pages = []
       ↓
[4] fallback → 그냥 첫 5페이지 분석
       ↓
[5] AI 답변 → "페이지 1~5에는 그림이 없어요..."
       ↓
[사용자] 😡 "왜 그림을 못 찾아?"
```

---

## 버그 정리

| 위치 | 버그 내용 | 원인 |
|------|-----------|------|
| `chunker.py:105` | 이미지 element를 DB에서 버림 | `if not elem["text"].strip(): continue` |
| `nodes.py:284~291` | `visual_pages`가 항상 빈 리스트 | 검색 결과엔 전부 `element_type="text"` |
| `nodes.py:299~300` | fallback으로 첫 5페이지만 분석 | 실제 그림은 6페이지 이후에 있음 |

---

## 해결 방향

### Fix 1 (가장 중요): `visual_pages` 로직 수정

"텍스트가 figure를 언급한 페이지 ±1"을 확인하도록 변경한다.

```python
# nodes.py — vision_analyst_node 내부

# 기존: element_type이 "image"인 chunk 페이지만 (→ 항상 빈 리스트)
# visual_pages = sorted({chunk["page_number"] for chunk in contexts
#                         if chunk.get("element_type") in ("table", "image")})

# 수정: local_rag 검색 결과의 페이지 번호 + 인접 페이지
ref_pages = {
    chunk["page_number"]
    for chunk in contexts
    if chunk.get("source") == "local_rag" and chunk.get("page_number")
}

expanded: set[int] = set()
for p in ref_pages:
    expanded.update([p - 1, p, p + 1])  # 앞뒤 페이지까지 포함
visual_pages = sorted(p for p in expanded if p > 0)[:5]
```

**왜 효과 있나**: Figure 1이 page 5에서 언급되면 → pages 4, 5, 6을 확인 → 실제 그림이 page 6에 있으면 발견 가능.

### Fix 2: 이미지 element도 저장하기

```python
# chunker.py — elements_to_documents() 내부

text = elem["text"].strip()
if not text:
    if elem["element_type"] in ("image", "figure"):
        text = f"[그림 - 페이지 {elem['page_number']}]"
    elif elem["element_type"] == "table":
        text = f"[표 - 페이지 {elem['page_number']}]"
    else:
        continue  # 일반 텍스트가 비어있으면 건너뜀
```

> **주의**: Fix 2를 적용하면 기존 ChromaDB를 **다시 인제스트**해야 한다.

### Fix 3: element_type 체크 집합에 "figure" 추가

LlamaParse가 `"image"` 대신 `"figure"`를 반환할 수 있다.

```python
if chunk.get("element_type") in ("table", "image", "figure")
```

---

## 우선순위

- **Fix 1만 해도 현재 문제는 즉시 해결된다.** 재인제스트 불필요.
- Fix 2, 3은 더 완전한 해결이지만 기존 데이터 재인제스트가 필요하다.
