# Tests Directory

Simple testing setup for RAG System API feasibility.

## 📁 Folder Structure

```
tests/
├── README.md           # This file
├── test.py            # Simple API feasibility tests (main test file)
└── test_data/         # Sample files for testing (auto-created)
    ├── sample.txt     # Sample text document
    └── sample.json    # Sample JSON document
```

## 🚀 Quick Start

### 1. Start the Server

```bash
# From project root
python main.py
```

Wait for the server to start (you should see "Server ready!" message).

### 2. Run Tests

```bash
# Option 1: Direct Python execution
python tests/test.py

# Option 2: Using pytest
pytest tests/test.py -v -s
```

## 📋 What Gets Tested

The `test.py` file runs **9 simple tests** that verify:

| # | Test | What It Checks |
|---|------|----------------|
| 1 | **Server Status** | Is the API server running? |
| 2 | **Health Check** | Are Pinecone, Database, and Gemini connected? |
| 3 | **Document Upload** | Can we upload and process documents? |
| 4 | **List Documents** | Can we retrieve all documents? |
| 5 | **Query (Search)** | Does semantic search work? |
| 6 | **Query (LLM)** | Does RAG with Gemini work? |
| 7 | **Document Details** | Can we get specific document info? |
| 8 | **Statistics** | Can we get system stats? |
| 9 | **Recent Queries** | Can we view query history? |

## ✅ Expected Output

```
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
  RAG SYSTEM - SIMPLE API FEASIBILITY TEST
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀

======================================================================
  TEST 1: Server Status
======================================================================
✅ Server is running
ℹ️  Service: Complete RAG System with Gemini LLM
ℹ️  Version: 3.1.0
ℹ️  Status: operational

======================================================================
  TEST 2: Health Check
======================================================================
✅ Health check passed
ℹ️  Pinecone: connected
ℹ️  Database: connected
ℹ️  Gemini: configured (Model: gemini-1.5-flash)

... (more tests) ...

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

## 🔧 Configuration

The tests use default configuration:

- **Base URL**: `http://localhost:8000`
- **Test Files**: Auto-created in `tests/test_data/`
- **Timeout**: 60 seconds per test

To change the base URL, edit `BASE_URL` in `test.py`:

```python
BASE_URL = "http://your-server:8000"
```

## 🐛 Troubleshooting

### Test Fails: "Cannot connect to server"

**Problem**: Server is not running

**Solution**:
```bash
# Start the server
python main.py

# In another terminal, run tests
python tests/test.py
```

### Test Fails: Health Check

**Problem**: API keys not configured

**Solution**:
1. Check your `.env` file exists in project root
2. Verify API keys are set:
   ```
   GEMINI_API_KEY=your_actual_key_here
   PINECONE_API_KEY=your_actual_key_here
   ```
3. Restart the server after updating `.env`

### Test Fails: Document Upload

**Problem**: File processing error

**Solution**:
- Check if `test_data` directory is writable
- Verify document processing dependencies are installed:
  ```bash
  pip install PyPDF2 pdfplumber python-docx beautifulsoup4
  ```

### All Tests Fail

**Problem**: Multiple issues

**Solution**:
1. Verify all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```
2. Check server logs for errors
3. Verify API keys are valid
4. Test server manually:
   ```bash
   curl http://localhost:8000/
   ```

## 📊 Understanding Results

### Success Rate
- **100%**: Perfect! All systems working
- **70-99%**: Good, but check failed tests
- **<70%**: Issues need attention

### Common Failure Patterns

| Pattern | Likely Cause | Solution |
|---------|--------------|----------|
| Test 1 fails | Server not running | Start server |
| Test 2 fails | API keys invalid | Check .env file |
| Test 3 fails | Upload issues | Check dependencies |
| Test 6 fails | Gemini API issue | Verify Gemini key |


