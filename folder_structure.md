universal-ai-agent/
│
├── backend/                          # All server-side code
│   │
│   ├── api/                          # API Layer - FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app entrypoint
│   │   ├── dependencies.py           # Shared FastAPI dependencies (auth, db sessions)
│   │   ├── middleware.py             # CORS, logging, rate limiting middleware
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── chat.py               # POST /chat
│   │       ├── upload.py             # POST /upload
│   │       ├── search.py             # POST /search
│   │       ├── agents.py             # GET/POST /agents
│   │       └── health.py             # GET /health
│   │
│   ├── agents/                       # Agent Orchestration Layer
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseAgent abstract class
│   │   ├── react_agent.py            # Single ReAct agent (LangGraph)
│   │   ├── planner.py                # Planner Agent
│   │   ├── researcher.py             # Research Agent
│   │   ├── analyst.py                # Analyst Agent
│   │   ├── writer.py                 # Writer Agent
│   │   ├── orchestrator.py           # Multi-agent workflow coordinator
│   │   └── state.py                  # LangGraph shared state definitions
│   │
│   ├── llm/                          # LLM Provider Layer
│   │   ├── __init__.py
│   │   ├── base.py                   # LLMProvider Protocol/ABC
│   │   ├── litellm_provider.py       # LiteLLM unified adapter
│   │   ├── openai_provider.py        # OpenAI-specific adapter (if needed)
│   │   └── router.py                 # LLM routing logic (cost, latency, fallback)
│   │
│   ├── tools/                        # Tool Layer
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseTool abstract class
│   │   ├── registry.py               # Tool registry - discover and load tools
│   │   ├── web_search.py             # Web search (Tavily/SerpAPI)
│   │   ├── calculator.py             # Safe math evaluation
│   │   ├── wikipedia.py              # Wikipedia lookup
│   │   ├── file_reader.py            # Local file reading
│   │   ├── document_loader.py        # Document loading + parsing
│   │   └── database_query.py         # Safe SQL query execution
│   │
│   ├── rag/                          # RAG Layer
│   │   ├── __init__.py
│   │   ├── pipeline.py               # Orchestrates full RAG pipeline
│   │   ├── ingestion.py              # Document ingestion coordinator
│   │   ├── chunker.py                # Text chunking strategies
│   │   ├── embedder.py               # Embedding model abstraction
│   │   ├── retriever.py              # Vector retrieval + reranking
│   │   └── context_generator.py      # Formats retrieved docs into context
│   │
│   ├── vectordb/                     # Vector Database Adapters
│   │   ├── __init__.py
│   │   ├── base.py                   # VectorDB Protocol
│   │   ├── qdrant_adapter.py         # Qdrant implementation
│   │   └── factory.py                # Creates correct VectorDB from config
│   │
│   ├── memory/                       # Memory Layer
│   │   ├── __init__.py
│   │   ├── base.py                   # Memory interface
│   │   ├── short_term.py             # Redis-backed conversation memory
│   │   ├── long_term.py              # VectorDB-backed persistent memory
│   │   └── manager.py                # Unified memory manager
│   │
│   ├── context/                      # Context Builder
│   │   ├── __init__.py
│   │   ├── builder.py                # Assembles full LLM context
│   │   └── templates.py              # System prompt templates
│   │
│   ├── ingestion/                    # Document Ingestion Pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py               # End-to-end ingestion orchestrator
│   │   ├── parsers/
│   │   │   ├── __init__.py
│   │   │   ├── pdf_parser.py
│   │   │   ├── docx_parser.py
│   │   │   ├── txt_parser.py
│   │   │   └── url_parser.py
│   │   └── chunkers/
│   │       ├── __init__.py
│   │       ├── recursive_chunker.py
│   │       └── semantic_chunker.py
│   │
│   ├── mcp/                          # MCP Integration Layer
│   │   ├── __init__.py
│   │   ├── client.py                 # Base MCP client
│   │   ├── registry.py               # MCP server registry
│   │   └── connectors/
│   │       ├── __init__.py
│   │       ├── google_drive.py
│   │       ├── notion.py
│   │       └── slack.py
│   │
│   ├── db/                           # Database Layer
│   │   ├── __init__.py
│   │   ├── session.py                # SQLAlchemy session management
│   │   ├── base.py                   # Base model declaration
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── chat.py
│   │   │   ├── document.py
│   │   │   └── agent_run.py
│   │   └── repositories/
│   │       ├── __init__.py
│   │       ├── user_repo.py
│   │       ├── chat_repo.py
│   │       ├── document_repo.py
│   │       └── agent_run_repo.py
│   │
│   ├── guardrails/                   # Safety & Validation Layer
│   │   ├── __init__.py
│   │   ├── input_validator.py        # Prompt injection detection
│   │   ├── output_validator.py       # Structured output validation
│   │   └── permissions.py            # Tool permission system
│   │
│   ├── observability/                # Observability Layer
│   │   ├── __init__.py
│   │   ├── tracer.py                 # OpenTelemetry / LangSmith setup
│   │   ├── metrics.py                # Token, latency, error tracking
│   │   └── logger.py                 # Structured logging setup
│   │
│   └── config/                       # Configuration Management
│       ├── __init__.py
│       ├── settings.py               # Pydantic Settings (reads .env)
│       └── prompts.py                # Centralized prompt management
│
├── tests/                            # Test suite (mirrors backend structure)
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures
│   ├── unit/
│   │   ├── test_tools.py
│   │   ├── test_rag.py
│   │   ├── test_memory.py
│   │   ├── test_llm.py
│   │   └── test_guardrails.py
│   ├── integration/
│   │   ├── test_chat_endpoint.py
│   │   ├── test_agent_workflow.py
│   │   └── test_ingestion_pipeline.py
│   └── e2e/
│       └── test_full_research_workflow.py
│
├── scripts/                          # Dev utility scripts
│   ├── setup_db.py                   # Initialize PostgreSQL schema
│   ├── seed_data.py                  # Load test documents
│   └── test_connection.py            # Verify all service connections
│
├── docs/                             # Architecture documentation
│   ├── architecture.md
│   ├── api_reference.md
│   └── deployment.md
│
├── .env.example                      # Template for environment variables
├── .env                              # Local secrets (git-ignored)
├── .gitignore
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Dev + test dependencies
└── README.md