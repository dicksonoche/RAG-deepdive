#!/usr/bin/env python3.11
"""Simple script to add documents to the RAG system.

Usage:
    python3.11 add_documents.py /path/to/your/document.pdf
    python3.11 add_documents.py /path/to/your/document.docx
"""

import os
import sys
import requests
from pathlib import Path

def add_documents(file_paths, chat_uid="test-docs"):
    """Add documents to the RAG system."""
    
    # Check if backend is running
    try:
        health_response = requests.get("http://localhost:5000/health")
        if health_response.status_code != 200:
            print("❌ Backend is not responding. Make sure it's running with: uvicorn app:app --port 5000 --reload")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False
    
    print(f"✅ Backend is running")
    print(f"Adding {len(file_paths)} document(s) with chat_uid: {chat_uid}")
    
    # Prepare files for upload
    files_data = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        
        with open(file_path, 'rb') as f:
            files_data.append(('files', (os.path.basename(file_path), f.read())))
    
    if not files_data:
        print("❌ No valid files to upload")
        return False
    
    # Upload and index files
    try:
        response = requests.post(
            "http://localhost:5000/index",
            data={"chat_uid": chat_uid},
            files=files_data,
            timeout=60
        )
        
        if response.status_code == 200:
            print(f"✅ Documents indexed successfully!")
            print(f"📊 Response: {response.json()}")
            return True
        else:
            print(f"❌ Indexing failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        return False

def test_query(chat_uid, question="What is this document about?"):
    """Test a simple query to verify the system works."""
    print(f"\n🧪 Testing query: {question}")
    
    try:
        response = requests.post(
            "http://localhost:5000/chat",
            json={
                "query": question,
                "model": "llama-3.3-70b-versatile",
                "chat_uid": chat_uid,
                "chatbot_name": "TestBot"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Query successful! Response received.")
            return True
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during query: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print("🚀 RAG Document Indexer")
        print("=" * 40)
        print("Usage: python add_documents.py /path/to/document1.pdf /path/to/document2.docx")
        print("Example: python add_documents.py ./my_document.pdf")
        print("\nThis script will:")
        print("1. Check if the RAG backend is running")
        print("2. Upload and index your documents")
        print("3. Test a simple query to verify everything works")
        print("4. Give you a chat_uid to use in evaluation")
        sys.exit(0)
    
    # Get file paths from command line arguments
    file_paths = sys.argv[1:]
    chat_uid = f"docs-{int(os.path.getmtime(file_paths[0]))}"
    
    print("RAG Document Indexer")
    print("=" * 40)
    
    # Add documents
    if add_documents(file_paths, chat_uid):
        print(f"\n📚 Documents indexed with chat_uid: {chat_uid}")
        
        # Test a query
        test_query(chat_uid)
        
        print(f"\n💡 You can now use this chat_uid in your evaluation:")
        print(f"   export EVAL_CHAT_UID='{chat_uid}'")
        print(f"   python run_evaluation.py")
    else:
        print("\n❌ Failed to index documents")
        sys.exit(1)
