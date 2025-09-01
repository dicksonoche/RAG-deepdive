#!/usr/bin/env python3.11
"""
Simple script to evaluate a RAG system using Evidently AI cloud.

This script demonstrates how to use the `RAGEvaluator` class to assess the performance
of a Retrieval-Augmented Generation (RAG) system against reference data. It supports
uploading documents, querying the system, and generating evaluation metrics.

Usage:
    python3.11 run_evaluation.py
"""
import os
import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluator import RAGEvaluator, EvaluationConfig, create_sample_evaluation_data

def main():
    """
    Run the RAG system evaluation with sample data.

    Configures the evaluation, initializes the evaluator, generates sample questions and answers,
    runs the evaluation, and displays the results, including contradiction analysis.
    """
    # Configuration
    config = EvaluationConfig(
        backend_url=os.environ.get("RAG_BACKEND_URL", "http://localhost:5000"),
        evidently_cloud_url=os.environ.get("EVIDENTLY_CLOUD_URL", "https://app.evidently.cloud/"),
        project_id=os.environ.get("EVIDENTLY_PROJECT_ID"),  # Optional
        model="llama-3.3-70b-versatile",
        chatbot_name="RAGEvaluator"
    )
    
    print("RAG System Evaluation")
    print("=====================")
    print(f"Backend URL: {config.backend_url}")
    print(f"Evidently Cloud: {config.evidently_cloud_url}")
    print(f"Project ID: {config.project_id or 'Not set'}")
    print(f"Model: {config.model}")
    print()
    
    # Allow user to specify a chat_uid for existing documents
    custom_chat_uid = os.environ.get("EVAL_CHAT_UID")
    if custom_chat_uid:
        print(f"📚 Using existing documents with chat_uid: {custom_chat_uid}")
        chat_uid = custom_chat_uid
    else:
        print("📚 No existing documents specified. Creating new evaluation session.")
        chat_uid = f"demo-eval-{int(time.time())}"
    
    try:
        # Initialize evaluator
        print("Initializing evaluator...")
        evaluator = RAGEvaluator(config)
        print("✓ Evaluator initialized successfully")
        
        # Create sample evaluation data
        print("\nCreating sample evaluation data...")
        questions, answers = create_sample_evaluation_data()
        print(f"✓ Created {len(questions)} sample questions")
        
        # Run evaluation
        print(f"\nRunning evaluation with chat_uid: {chat_uid}")
        results = evaluator.evaluate_rag_pipeline(
            questions=questions,
            reference_answers=answers,
            chat_uid=chat_uid
        )
        
        # Display results
        print("\n" + "="*50)
        print(evaluator.get_evaluation_summary(results))
        print("="*50)
        
        # Display detailed contradiction analysis if available
        print("\n" + "="*50)
        print("DETAILED CONTRADICTION ANALYSIS")
        print("="*50)
        print(evaluator.get_contradiction_details(results))
        print("="*50)
        
        print("\n✓ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your RAG backend is running")
        print("2. Check your Evidently AI cloud credentials")
        print("3. Verify GROQ_API_KEY is set")
        print("4. Check the logs in logs/evaluation_logs.log")
        print("5. Make sure you have documents indexed with the specified chat_uid")
        sys.exit(1)

if __name__ == "__main__":
    """
    Entry point for the evaluation script.

    Executes the `main` function to run the RAG system evaluation.
    """
    main()