uv add fastapi uvicorn langgraph langchain-openai python-dotenv
uv add llama-parse pdf2image chromadb  # RAG와 이미지 관련

uv sync --extra dev 
uv sync --extra test



git remote add origin https://github.com/KangYopchi/thesistand.git/
├── .venv/              # uv가 관리하는 가상환경
├── data/               # 분석된 PDF, 이미지, DB가 저장되는 곳 (Persistence)
│   ├── pdfs/
│   ├── images/
│   └── chroma_db/
├── src/
│   ├── main.py         # FastAPI 실행 엔드포인트
│   ├── agents/         # LangGraph 관련 로직
│   │   ├── graph.py    # 전체 그래프 구성 (Nodes & Edges)
│   │   ├── state.py    # AgentState 정의 및 Reducer 설정
│   │   └── nodes.py    # 각 노드의 실제 함수들 (retrieval, vision 등)
│   ├── rag/            # RAG 관련 로직
│   │   ├── parser.py   # LlamaParse (JSON) 처리
│   │   ├── chunker.py  # 계층적 청킹 로직
│   │   └── vectorstore.py # ChromaDB 설정
│   └── utils/          # 공통 도구 (이미지 처리, 로거 등)
├── pyproject.toml      # uv 설정 파일
├── Dockerfile          # Multi-stage 빌드 파일
├── README.md           # Project 소개 파일
└── .env                # API 키 관리 (OpenAI, Tavily, LlamaIndex)
