# RAG System with Google Gemini

A production-ready Retrieval-Augmented Generation system for document processing, semantic search, and AI-powered question answering.

## Overview

This system implements a complete RAG pipeline that processes documents, generates embeddings, performs semantic search, and provides context-aware responses using Google Gemini LLM.

### Architecture

```
Document Upload → Text Extraction → Chunking → Embedding Generation
                                                        ↓
User Query → Query Embedding → Vector Search ← Pinecone Storage
                    ↓                               ↓
            Context Retrieval ← SQLite Metadata
                    ↓
            Gemini LLM → AI Response
```

### Technology Stack

- **Framework**: FastAPI
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2, 384-dim)
- **Vector Store**: Pinecone
- **Metadata DB**: SQLite
- **Supported Formats**: PDF, DOCX, TXT, HTML, CSV, JSON, MD, XML

---

## Installation

### Prerequisites
- Python 3.9+
- Pinecone API Key
- Google Gemini API Key

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Docker Setup

```bash
# Using docker-compose (recommended)
docker-compose up -d

# Or manual Docker
docker build -t rag-system .
docker run -p 8000:8000 -v $(pwd)/data:/app/data --env-file .env rag-system
```
### Quick commands for comman tasks:

`install:
	pip install -r requirements.txt

run:
	python main.py

test:
	python tests/test.py

docker-up:
	docker-compose up -d`

---

## Configuration

### Required Environment Variables

```bash
# API Keys (Required)
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key

# LLM Configuration
GEMINI_MODEL=gemini-2.5-flash

# Pinecone Settings
PINECONE_INDEX_NAME=rag-documents
PINECONE_ENVIRONMENT=us-east-1

# Processing Limits
MAX_DOCUMENTS=20
MAX_FILE_SIZE_MB=50
MAX_PAGES_PER_DOC=1000

# Chunking
DEFAULT_CHUNK_SIZE=512
DEFAULT_CHUNK_OVERLAP=50
CHUNKING_STRATEGY=sliding_window

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_CONTEXT_CHUNKS=5

# Database
DATABASE_PATH=data/metadata.db

# Server
HOST=0.0.0.0
PORT=8000
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `GEMINI_MODEL` | Gemini model version | `gemini-2.5-flash` |
| `DEFAULT_CHUNK_SIZE` | Words per chunk | `512` |
| `CHUNKING_STRATEGY` | Chunking method | `sliding_window` |
| `MAX_CONTEXT_CHUNKS` | Chunks for LLM context | `5` |

---
## 🔒 Security & Legal Notice

### 🛡️ API Key Protection
**CRITICAL:** The API keys in the `.env` file are placeholders. You must:

- ✅ Replace with your own valid API keys  
- ✅ Never commit real API keys to public repositories  
- ✅ Keep your API keys confidential  
- ✅ Regenerate keys immediately if exposed  

---

### ⚖️ Terms of Use
This software is provided for **educational and legitimate development purposes only.**

#### Legal Disclaimer:
- API keys are **personal credentials** tied to your account.  
- Misuse, unauthorized sharing, or abuse of API keys may result in:
  - ❌ Account suspension or termination by service providers  
  - ⚠️ Legal action under applicable laws and terms of service  
  - 💰 Financial liability for unauthorized usage  

#### Users are solely responsible for:
- 🔐 Securing their API credentials  
- 📜 Complying with **Pinecone** and **Google API** terms of service  
- 💳 All costs incurred through API usage  
- 🚫 Any misuse or unauthorized access  

---

### 📘 By using this software, you agree to:
- Use API keys **responsibly and ethically**  
- Comply with **all applicable laws and regulations**  
- Accept **full responsibility** for your API usage  
- Indemnify the project authors from **any liability** arising from your use  

---

### 🌐 Service Provider Terms
Please review and comply with:
- [Pinecone Terms of Service](https://www.pinecone.io/terms/)  
- [Google Cloud Terms of Service](https://cloud.google.com/terms)  
- [Google AI/ML Terms](https://ai.google.dev/terms)  

---

🚨 **Unauthorized or malicious use of this system may result in civil and criminal penalties.**

## API Endpoints

### Primary Endpoints

#### 1. Upload Documents
```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data
```

**Parameters:**
- `files`: Document files (multiple)
- `chunk_size`: Optional, default 512
- `chunk_overlap`: Optional, default 50
- `chunking_strategy`: Optional, default "sliding_window"

**Response:**
```json
{
  "status": "success",
  "summary": {
    "documents_processed": 2,
    "total_chunks": 145,
    "vectors_stored": 145
  }
}
```

#### 2. Query System
```http
POST /api/v1/query
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "What is machine learning?",
  "top_k": 5,
  "use_llm": true,
  "filter": null
}
```

**Response:**
```json
{
  "query_id": "abc123",
  "llm_response": "Machine learning is...",
  "num_results": 5,
  "results": [...],
  "processing_time_ms": 1234.56
}
```

#### 3. List Documents
```http
GET /api/v1/documents
```

**Response:**
```json
{
  "total_documents": 5,
  "successful": 5,
  "documents": [...]
}
```

### Additional Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | System information |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/documents/{id}` | Document details |
| DELETE | `/api/v1/documents/{id}` | Delete document |
| GET | `/api/v1/queries/recent` | Query history |
| GET | `/api/v1/stats` | System statistics |
| GET | `/docs` | API documentation (Swagger) |

---

## Database Schema

### Tables

**documents**
| Column | Type | Description |
|--------|------|-------------|
| document_id | TEXT PK | UUID identifier |
| filename | TEXT | Original filename |
| file_type | TEXT | File extension |
| file_size_mb | REAL | File size |
| page_count | INTEGER | Number of pages |
| word_count | INTEGER | Total words |
| upload_timestamp | TEXT | ISO 8601 timestamp |
| status | TEXT | success/failed |

**chunks**
| Column | Type | Description |
|--------|------|-------------|
| chunk_id | TEXT PK | UUID identifier |
| document_id | TEXT FK | Parent document |
| chunk_index | INTEGER | Position in document |
| text | TEXT | Chunk content |
| word_count | INTEGER | Words in chunk |
| embedding_generated | BOOLEAN | Embedding status |
| stored_in_vectordb | BOOLEAN | Storage status |

**queries**
| Column | Type | Description |
|--------|------|-------------|
| query_id | TEXT PK | UUID identifier |
| query_text | TEXT | Search query |
| query_timestamp | TEXT | ISO 8601 timestamp |
| num_results | INTEGER | Results returned |
| llm_response | TEXT | AI answer |
| processing_time_ms | REAL | Processing time |

**query_results**
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| query_id | TEXT FK | Query reference |
| chunk_id | TEXT FK | Chunk reference |
| relevance_score | REAL | Similarity score |

---

## Usage Examples

### Upload Documents
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "files=@document.pdf" \
  -F "chunk_size=512"
```

### Query with AI Response
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain machine learning",
    "top_k": 5,
    "use_llm": true
  }'
```

### Search Without LLM
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "top_k": 3,
    "use_llm": false
  }'
```

### Filter by Document
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "key findings",
    "filter": {"filename": "report.pdf"}
  }'
```

---

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
RUN mkdir -p /app/data
EXPOSE 8000
CMD ["python", "main.py"]
```

### docker-compose.yml
```yaml
version: '3.8'
services:
  rag-system:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    env_file:
      - .env
    restart: unless-stopped
```

### .dockerignore
```
__pycache__/
*.pyc
.env.local
.venv/
venv/
tests/
.git/
*.md
```

### Commands
```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Testing

### Test Suite Overview

The system includes a comprehensive test suite (`tests/test.py`) that validates all API endpoints and system functionality.

### Running Tests

```bash
# Start the server first
python main.py

# In a separate terminal, run tests
python tests/test.py

# Or using pytest
pytest tests/test.py -v -s
```

### Test Coverage

The test suite validates 9 key areas:

| Test # | Test Name | Validates |
|--------|-----------|-----------|
| 1 | Server Status | API server availability |
| 2 | Health Check | Pinecone, Database, Gemini connectivity |
| 3 | Document Upload | Complete upload and processing pipeline |
| 4 | List Documents | Document metadata retrieval |
| 5 | Query (Search) | Semantic search without LLM |
| 6 | Query (LLM) | Full RAG pipeline with AI response |
| 7 | Document Details | Specific document information |
| 8 | Statistics | System-wide statistics |
| 9 | Recent Queries | Query history tracking |

### Expected Output

```
======================================================================
  TEST SUMMARY
======================================================================
  Server Status............................................... ✅ PASSED
  Health Check................................................ ✅ PASSED
  Document Upload............................................. ✅ PASSED
  List Documents.............................................. ✅ PASSED
  Query (Search).............................................. ✅ PASSED
  Query (LLM)................................................. ✅ PASSED
  Document Details............................................ ✅ PASSED
  Statistics.................................................. ✅ PASSED
  Recent Queries.............................................. ✅ PASSED

----------------------------------------------------------------------
  Total Tests: 9
  Passed: 9
  Failed: 0
  Success Rate: 100.0%
----------------------------------------------------------------------

🎉 All tests passed! API is fully functional.
```

### Test Data

Test files are automatically created in `tests/test_data/`:
- `sample.txt` - Sample text document with ML content
- `sample.json` - Sample JSON data structure

### Troubleshooting Tests

**All tests fail:**
```bash
# Verify server is running
curl http://localhost:8000/
```

**Health check fails:**
- Verify API keys in `.env`
- Check Pinecone connection
- Confirm Gemini API access

**Upload test fails:**
- Check file permissions in `tests/test_data/`
- Verify supported formats configuration
- Review server logs for processing errors

**Query tests timeout:**
- Increase timeout in test configuration
- Check network connectivity to Pinecone
- Verify Gemini API rate limits

---

## LLM Provider Configuration

### Google Gemini (Default)
```bash
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash  # or gemini-1.5-pro
```

### Alternative: OpenAI
Modify `GeminiLLMModule` class in `main.py`:

```python
import openai

class OpenAILLMModule:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
    
    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        # Implementation
```

Environment:
```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo
```

---

## Troubleshooting

### Server Issues
**Port in use:**
```bash
# Change port in .env
PORT=8001
```

### API Key Errors
- Remove quotes/spaces from .env
- Verify keys at provider dashboards
- Restart server after .env changes

### Pinecone Connection
- Use alphanumeric index names (no underscores)
- Verify region matches account
- Check API key permissions

### Document Processing
- Verify file format in SUPPORTED_FORMATS
- Check file size < MAX_FILE_SIZE_MB
- Ensure dependencies installed:
  ```bash
  pip install PyPDF2 pdfplumber python-docx
  ```

### Memory Issues
- Reduce EMBEDDING_BATCH_SIZE
- Decrease DEFAULT_CHUNK_SIZE
- Process fewer documents simultaneously

---

## System Requirements

**Minimum:**
- 4 GB RAM
- 2 GB disk space
- Python 3.9+

**Recommended:**
- 8 GB RAM
- 5 GB disk space
- Python 3.10+

---

## License

This project is provided for educational and development purposes.

---

## Support

- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/v1/health`
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Gemini API Documentation](https://ai.google.dev/)