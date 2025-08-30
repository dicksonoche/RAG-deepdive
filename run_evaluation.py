#!/usr/bin/env python3
"""Simple script to run RAG system evaluation using Evidently AI cloud.

This script demonstrates how to use the RAGEvaluator class to evaluate
your RAG system performance against reference data.

Usage:
    python run_evaluation.py
"""

import os
import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluator import RAGEvaluator, EvaluationConfig, create_sample_evaluation_data


def main():
    """Run RAG system evaluation with sample data."""
    
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
        print(f"\nRunning evaluation with chat_uid: demo-eval-{int(time.time())}")
        results = evaluator.evaluate_rag_pipeline(
            questions=questions,
            reference_answers=answers,
            chat_uid=f"demo-eval-{int(time.time())}"
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
        sys.exit(1)


if __name__ == "__main__":
    main()
