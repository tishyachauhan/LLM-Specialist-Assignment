"""
Simple Test Suite for RAG System API
Tests basic functionality and API feasibility

Run with: python tests/test.py
Or with: pytest tests/test.py -v
"""

import requests
import json
import time
from pathlib import Path
import tempfile

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL = "http://localhost:8000"
TEST_FILES_DIR = Path(__file__).parent / "test_data"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_success(text):
    """Print success message."""
    print(f"✅ {text}")


def print_error(text):
    """Print error message."""
    print(f"❌ {text}")


def print_info(text):
    """Print info message."""
    print(f"ℹ️  {text}")


def create_test_files():
    """Create sample test files if they don't exist."""
    TEST_FILES_DIR.mkdir(exist_ok=True)

    # Create sample TXT file
    txt_file = TEST_FILES_DIR / "sample.txt"
    if not txt_file.exists():
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("""Machine Learning Guide

Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.
It enables computers to improve their performance on tasks through experience without being explicitly programmed.

There are three main types of machine learning:
1. Supervised Learning - Learning from labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through trial and error

Deep learning is a subset of machine learning that uses neural networks with multiple layers.
It has revolutionized fields like computer vision, natural language processing, and speech recognition.
""")

    # Create sample JSON file
    json_file = TEST_FILES_DIR / "sample.json"
    if not json_file.exists():
        data = {
            "title": "Artificial Intelligence Overview",
            "author": "AI Researcher",
            "year": 2024,
            "topics": [
                "Machine Learning",
                "Deep Learning",
                "Neural Networks",
                "Natural Language Processing"
            ],
            "summary": "Artificial intelligence is transforming industries worldwide. From healthcare to finance, AI applications are making systems smarter and more efficient."
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    print_info(f"Test files ready in: {TEST_FILES_DIR}")
    return txt_file, json_file


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_1_server_running():
    """Test 1: Check if server is running."""
    print_header("TEST 1: Server Status")

    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)

        if response.status_code == 200:
            data = response.json()
            print_success("Server is running")
            print_info(f"Service: {data.get('service', 'Unknown')}")
            print_info(f"Version: {data.get('version', 'Unknown')}")
            print_info(f"Status: {data.get('status', 'Unknown')}")
            return True
        else:
            print_error(f"Server returned status code: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to server. Is it running?")
        print_info("Start server with: python main.py")
        return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_2_health_check():
    """Test 2: Check system health."""
    print_header("TEST 2: Health Check")

    try:
        response = requests.get(f"{BASE_URL}/api/v1/health", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print_success("Health check passed")

            # Check components
            if 'pinecone' in data:
                status = data['pinecone'].get('status', 'unknown')
                print_info(f"Pinecone: {status}")

            if 'database' in data:
                status = data['database'].get('status', 'unknown')
                print_info(f"Database: {status}")

            if 'gemini' in data:
                status = data['gemini'].get('status', 'unknown')
                model = data['gemini'].get('model', 'unknown')
                print_info(f"Gemini: {status} (Model: {model})")

            return True
        else:
            print_error(f"Health check failed with status: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Health check error: {e}")
        return False


def test_3_upload_document():
    """Test 3: Upload a document."""
    print_header("TEST 3: Document Upload")

    txt_file, _ = create_test_files()

    try:
        with open(txt_file, 'rb') as f:
            files = {'files': (txt_file.name, f, 'text/plain')}

            print_info("Uploading document...")
            start_time = time.time()

            response = requests.post(
                f"{BASE_URL}/api/v1/documents/upload",
                files=files,
                timeout=60
            )

            elapsed_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print_success(f"Document uploaded successfully in {elapsed_time:.2f}s")

            summary = data.get('summary', {})
            print_info(f"Documents processed: {summary.get('documents_processed', 0)}")
            print_info(f"Total chunks: {summary.get('total_chunks', 0)}")
            print_info(f"Embeddings generated: {summary.get('embeddings_generated', 0)}")
            print_info(f"Vectors stored: {summary.get('vectors_stored', 0)}")

            # Store document ID for later tests
            documents = data.get('documents', [])
            if documents and documents[0].get('status') == 'success':
                doc_id = documents[0].get('document_id')
                print_info(f"Document ID: {doc_id}")
                return True, doc_id

            return True, None
        else:
            print_error(f"Upload failed with status: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False, None

    except Exception as e:
        print_error(f"Upload error: {e}")
        return False, None


def test_4_list_documents():
    """Test 4: List all documents."""
    print_header("TEST 4: List Documents")

    try:
        response = requests.get(f"{BASE_URL}/api/v1/documents", timeout=10)

        if response.status_code == 200:
            data = response.json()
            total = data.get('total_documents', 0)
            successful = data.get('successful', 0)
            failed = data.get('failed', 0)

            print_success("Retrieved document list")
            print_info(f"Total documents: {total}")
            print_info(f"Successful: {successful}")
            print_info(f"Failed: {failed}")

            # Show recent documents
            documents = data.get('documents', [])
            if documents:
                print_info("\nRecent documents:")
                for doc in documents[:3]:  # Show max 3
                    filename = doc.get('filename', 'Unknown')
                    status = doc.get('status', 'Unknown')
                    word_count = doc.get('word_count', 0)
                    print(f"  • {filename} - {status} ({word_count} words)")

            return True
        else:
            print_error(f"Failed with status: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_5_query_without_llm():
    """Test 5: Query documents (semantic search only)."""
    print_header("TEST 5: Query Documents (Search Only)")

    query = "What is machine learning?"
    print_info(f"Query: {query}")

    try:
        payload = {
            "query": query,
            "top_k": 3,
            "use_llm": False
        }

        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/v1/query",
            json=payload,
            timeout=30
        )
        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print_success(f"Search completed in {elapsed_time:.2f}s")

            num_results = data.get('num_results', 0)
            print_info(f"Found {num_results} relevant chunks")

            results = data.get('results', [])
            if results:
                print_info("\nTop results:")
                for i, result in enumerate(results[:3], 1):
                    score = result.get('score', 0)
                    metadata = result.get('metadata', {})
                    filename = metadata.get('filename', 'Unknown')
                    text_preview = metadata.get('text', '')[:100]

                    print(f"\n  {i}. Score: {score:.3f} | Source: {filename}")
                    print(f"     Preview: {text_preview}...")

            return True
        else:
            print_error(f"Query failed with status: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False

    except Exception as e:
        print_error(f"Query error: {e}")
        return False


def test_6_query_with_llm():
    """Test 6: Query with LLM response (complete RAG)."""
    print_header("TEST 6: Query with LLM (Complete RAG)")

    query = "Explain the types of machine learning."
    print_info(f"Query: {query}")

    try:
        payload = {
            "query": query,
            "top_k": 5,
            "use_llm": True
        }

        print_info("Generating AI response...")
        start_time = time.time()

        response = requests.post(
            f"{BASE_URL}/api/v1/query",
            json=payload,
            timeout=60
        )

        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print_success(f"Query completed in {elapsed_time:.2f}s")

            llm_response = data.get('llm_response')
            if llm_response:
                print_info("\nAI Response:")
                print("-" * 70)
                print(llm_response)
                print("-" * 70)

            num_results = data.get('num_results', 0)
            print_info(f"\nBased on {num_results} relevant chunks")

            processing_time = data.get('processing_time_ms', 0)
            print_info(f"Processing time: {processing_time:.2f}ms")

            return True
        else:
            print_error(f"Query failed with status: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False

    except Exception as e:
        print_error(f"Query error: {e}")
        return False


def test_7_get_document_details(document_id=None):
    """Test 7: Get document details."""
    print_header("TEST 7: Document Details")

    if not document_id:
        print_info("Skipping (no document ID available)")
        return True

    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/documents/{document_id}",
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            print_success("Retrieved document details")

            doc = data.get('document', {})
            print_info(f"Filename: {doc.get('filename', 'Unknown')}")
            print_info(f"Pages: {doc.get('page_count', 0)}")
            print_info(f"Words: {doc.get('word_count', 0)}")

            total_chunks = data.get('total_chunks', 0)
            print_info(f"Total chunks: {total_chunks}")

            return True
        else:
            print_error(f"Failed with status: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_8_statistics():
    """Test 8: Get system statistics."""
    print_header("TEST 8: System Statistics")

    try:
        response = requests.get(f"{BASE_URL}/api/v1/stats", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print_success("Retrieved statistics")

            docs = data.get('documents', {})
            print_info(f"Total documents: {docs.get('total', 0)}")
            print_info(f"Successful: {docs.get('successful', 0)}")
            print_info(f"Failed: {docs.get('failed', 0)}")

            vectorstore = data.get('vectorstore', {})
            print_info(f"Total vectors: {vectorstore.get('total_vectors', 0)}")

            models = data.get('models', {})
            print_info(f"Embedding model: {models.get('embedding', 'Unknown')}")
            print_info(f"LLM model: {models.get('llm', 'Unknown')}")

            return True
        else:
            print_error(f"Failed with status: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_9_recent_queries():
    """Test 9: Get recent queries."""
    print_header("TEST 9: Recent Queries")

    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/queries/recent?limit=5",
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            total = data.get('total_queries', 0)

            print_success(f"Retrieved {total} recent queries")

            queries = data.get('queries', [])
            if queries:
                print_info("\nRecent queries:")
                for i, q in enumerate(queries[:5], 1):
                    query_text = q.get('query_text', 'Unknown')
                    processing_time = q.get('processing_time_ms', 0)
                    print(f"  {i}. {query_text} ({processing_time:.0f}ms)")

            return True
        else:
            print_error(f"Failed with status: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Error: {e}")
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests and provide summary."""
    print("\n" + "🚀" * 35)
    print("  RAG SYSTEM - SIMPLE API FEASIBILITY TEST")
    print("🚀" * 35)

    results = {}
    document_id = None

    # Test 1: Server Running
    results['Server Status'] = test_1_server_running()
    if not results['Server Status']:
        print_error("\nServer is not running. Cannot continue tests.")
        print_info("Start the server with: python main.py")
        return

    # Test 2: Health Check
    results['Health Check'] = test_2_health_check()

    # Test 3: Upload Document
    success, doc_id = test_3_upload_document()
    results['Document Upload'] = success
    if doc_id:
        document_id = doc_id

    # Test 4: List Documents
    results['List Documents'] = test_4_list_documents()

    # Test 5: Query (Search Only)
    results['Query (Search)'] = test_5_query_without_llm()

    # Test 6: Query with LLM
    results['Query (LLM)'] = test_6_query_with_llm()

    # Test 7: Document Details
    results['Document Details'] = test_7_get_document_details(document_id)

    # Test 8: Statistics
    results['Statistics'] = test_8_statistics()

    # Test 9: Recent Queries
    results['Recent Queries'] = test_9_recent_queries()

    # Print Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:.<50} {status}")

    print("\n" + "-" * 70)
    print(f"  Total Tests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {total - passed}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print("-" * 70)

    if success_rate == 100:
        print("\n🎉 All tests passed! API is fully functional.")
    elif success_rate >= 70:
        print("\n⚠️  Most tests passed. Check failed tests above.")
    else:
        print("\n❌ Many tests failed. Please check your configuration.")
        print("   - Verify API keys in .env file")
        print("   - Check Pinecone connection")
        print("   - Verify Gemini API key")

    print("\n" + "=" * 70 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_all_tests()