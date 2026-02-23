Ingest 부분

Start APP
↓
`@app.post("/ingest")`실행
↓
hashing을 진행한다. (16진수) 변환된 값으로 중복 검사 진행
↓
새로운 파일이면 ingest Graph를 생성하고 ainvoke를 실행한다.
↓
graph의 ingest_node가 실행된다.
↓
parse_pdf를 실행한다. LlamaParse로 파싱을 한다. pdf2image 라이브러리를 사용해 이미지를 생성하고, 저장한다. 두 동작은 비동기로 동시에 진행된다.
↓
Parents chunk, 그리고 연결된 Child chunk를 생성하고, chromaDB에 Parents/Child 구조로 저장한다. 
↓
이미지 폴더 경로와 pdf_hash를 연결해 이미지의 위치를 저장한다.
↓
state의 image_dir 값을 업데이트 한다.
↓
결과를 서버로 전달한다.



Stream 부분

(Ingest 를 실행하고 나서 실행)


PDF 파일의 이미지가 있는지 확인
↓
local_retriever_node와 web_search node에서 질문 값으로 벡터 데이터를 추출, 로컬과 tavily api 사용해 검색 동시에 비동기로 실행한다.
↓
router node 에서 vision analyse 가 필요한지 확인
↓
필요할 경우 vision Node로 이동해서 vision analyse 실행
↓
생성된 결과를 합쳐, 답변을 생성
↓
답변 전달


---

## 앞으로의 과제  

1. chunker.py의 로직에 대해서 이해가 필요하다.
2. LlamaParse의 Parsing 값을 확인한다.
3. RecursiveCharacterTextSplitter 의 결과값과 각 파라미터 값의 의미
