"""RAG System Evaluation Module using Evidently AI Cloud.

This module provides comprehensive evaluation capabilities for the RAG-deepdive system,
integrating with Evidently AI cloud for metrics, reports, and monitoring.
"""

import os
import time
import json
import random
import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Evidently AI imports
from evidently.ui.workspace import CloudWorkspace
from evidently import Dataset, DataDefinition, Report
from evidently.descriptors import *
from evidently.presets import TextEvals
from evidently.llm.templates import BinaryClassificationPromptTemplate

# Local imports
from src.loghandler import set_logger, ColorFormmater
from src.config import GROQ_API_KEY
from src.models import LLMClient


@dataclass
class EvaluationConfig:
    """Configuration for RAG evaluation runs."""
    
    backend_url: str = "http://localhost:5000"
    evidently_cloud_url: str = "https://app.evidently.cloud/"
    project_id: Optional[str] = None
    model: str = "llama-3.3-70b-versatile"
    chatbot_name: str = "RAGEvaluator"
    max_retries: int = 3
    timeout: int = 300
    similarity_top_k: int = 5
    chunk_size: int = 1024
    chunk_overlap: int = 40


class RAGEvaluator:
    """Evaluates RAG system performance using Evidently AI cloud metrics."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize the RAG evaluator with configuration."""
        self.config = config
        self.logger = self._setup_logger()
        self.evidently_client = self._setup_evidently_client()
        self.llm_client = LLMClient()
        
        # Validate configuration
        self._validate_config()
        
    def _setup_logger(self) -> Any:
        """Set up logging for evaluation operations."""
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        return set_logger(
            logger_name="rag_evaluator",
            to_file=True,
            log_file_name=str(log_dir / "evaluation_logs.log"),
            to_console=True,
            custom_formatter=ColorFormmater
        )
    
    def _setup_evidently_client(self) -> CloudWorkspace:
        """Initialize connection to Evidently AI cloud workspace."""
        try:
            self.logger.info("Connecting to Evidently AI cloud workspace...")
            client = CloudWorkspace(url=self.config.evidently_cloud_url)
            self.logger.info("Successfully connected to Evidently AI cloud")
            return client
        except Exception as e:
            error_msg = f"Failed to connect to Evidently AI cloud: {e}"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def _validate_config(self):
        """Validate evaluation configuration parameters."""
        if not self.config.backend_url:
            raise ValueError("Backend URL must be provided")
        
        if not self.config.evidently_cloud_url:
            raise ValueError("Evidently cloud URL must be provided")
        
        if not GROQ_API_KEY:
            self.logger.warning("GROQ_API_KEY not found in environment variables")
        
        self.logger.info(f"Evaluation config validated: backend={self.config.backend_url}, model={self.config.model}")
    
    def _upload_and_index_files(self, files: List[str], chat_uid: str) -> bool:
        """Upload and index files for evaluation using the RAG backend."""
        try:
            self.logger.info(f"Uploading and indexing {len(files)} files for evaluation session {chat_uid}")
            
            # Prepare multipart form data
            with requests.Session() as session:
                files_data = []
                for file_path in files:
                    if not os.path.exists(file_path):
                        self.logger.warning(f"File not found: {file_path}")
                        continue
                    
                    with open(file_path, 'rb') as f:
                        files_data.append(('files', (os.path.basename(file_path), f.read())))
                
                if not files_data:
                    self.logger.error("No valid files found for indexing")
                    return False
                
                # Send indexing request
                response = session.post(
                    f"{self.config.backend_url}/index",
                    data={"chat_uid": chat_uid},
                    files=files_data,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    self.logger.info(f"Successfully indexed files for session {chat_uid}")
                    return True
                else:
                    self.logger.error(f"Indexing failed with status {response.status_code}: {response.text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error during file indexing: {e}")
            return False
    
    def _query_rag_system(self, question: str, chat_uid: str) -> Tuple[Optional[str], Optional[str]]:
        """Query the RAG system and retrieve response and context."""
        try:
            self.logger.info(f"Querying RAG system for question: {question[:100]}...")
            
            payload = {
                "query": question,
                "model": self.config.model,
                "chat_uid": chat_uid,
                "chatbot_name": self.config.chatbot_name
            }
            
            # Query the chat endpoint
            response = requests.post(
                f"{self.config.backend_url}/chat",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                self.logger.error(f"Chat request failed with status {response.status_code}: {response.text}")
                return None, None
            
            # For evaluation, we need both the response and context
            # Since the current backend doesn't return context separately,
            # we'll need to implement a separate context retrieval endpoint
            # For now, we'll return the response and a placeholder for context
            
            generated_response = response.text
            # TODO: Implement context retrieval endpoint for evaluation
            retrieved_context = "Context retrieval not yet implemented"
            
            self.logger.info(f"Successfully generated response for question (length: {len(generated_response)})")
            return generated_response, retrieved_context
            
        except Exception as e:
            self.logger.error(f"Error querying RAG system: {e}")
            return None, None
    
    def _create_contradiction_template(self) -> BinaryClassificationPromptTemplate:
        """Create the contradiction detection template for LLM-based evaluation."""
        return BinaryClassificationPromptTemplate(
            criteria="""Label an ANSWER as **contradictory** only if it directly contradicts any part of that REFERENCE.
            Differences in length or wording are acceptable. It is also acceptable if the ANSWER adds new details, but not acceptable if the ANSWER omits information, as long as **it is a fact and not contradictory**.
            Your task is to compare factual consistency only - not completeness, relevance, or style.
            
            REFERENCE:
            =====
            {reference}
            =====
            """,
            target_category="contradictory",
            non_target_category="non-contradictory",
            uncertainty="unknown",
            include_reasoning=True,
            pre_messages=[("system", "You are an expert evaluator. You will be given an ANSWER and a REFERENCE.")]
        )
    
    def evaluate_rag_pipeline(
        self,
        questions: List[str],
        reference_answers: List[str],
        chat_uid: str,
        files_to_index: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate the RAG pipeline performance against reference data."""
        try:
            self.logger.info(f"Starting RAG pipeline evaluation for {len(questions)} questions")
            
            # Index files if provided
            if files_to_index:
                if not self._upload_and_index_files(files_to_index, chat_uid):
                    raise RuntimeError("Failed to index files for evaluation")
            
            # Generate responses from RAG system
            generated_responses = []
            retrieved_contexts = []
            
            for i, question in enumerate(questions):
                self.logger.info(f"Processing question {i+1}/{len(questions)}")
                
                response, context = self._query_rag_system(question, chat_uid)
                if response is None:
                    self.logger.warning(f"Failed to get response for question {i+1}")
                    response = "ERROR: Failed to generate response"
                    context = "ERROR: Failed to retrieve context"
                
                generated_responses.append(response)
                retrieved_contexts.append(context)
                
                # Rate limiting for API calls
                time.sleep(0.5)
            
            # Create evaluation dataset
            evaluation_data = pd.DataFrame({
                'question': questions,
                'reference_answer': reference_answers,
                'generated_answer': generated_responses,
                'context': retrieved_contexts
            })
            
            # Run Evidently evaluation
            evaluation_results = self._run_evidently_evaluation(evaluation_data)
            
            # Compile results
            results = {
                'chat_uid': chat_uid,
                'total_questions': len(questions),
                'successful_responses': len([r for r in generated_responses if not r.startswith("ERROR")]),
                'evaluation_data': evaluation_data.to_dict('records'),
                'evidently_results': evaluation_results,
                'timestamp': time.time()
            }
            
            self.logger.info(f"Evaluation completed successfully. Results: {results}")
            return results
            
        except Exception as e:
            error_msg = f"Error during RAG pipeline evaluation: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _run_evidently_evaluation(self, evaluation_data: pd.DataFrame) -> Dict[str, Any]:
        """Run Evidently AI evaluation on the evaluation dataset."""
        try:
            self.logger.info("Running Evidently AI evaluation...")
            
            # Create evaluation dataset with Groq-compatible metrics only
            training_dataset = Dataset.from_pandas(
                evaluation_data,
                data_definition=DataDefinition(),
                descriptors=[
                    Sentiment("generated_answer"),
                    TextLength("generated_answer"),
                    # Note: FaithfulnessLLMEval and LLMEval require OpenAI API key
                    # We'll use basic text metrics instead
                ]
            )
            
            # Generate report
            report = Report([TextEvals()])
            evaluation_result = report.run(training_dataset, None)
            
            # Add to Evidently cloud if project ID is configured
            if self.config.project_id:
                try:
                    self.evidently_client.add_run(
                        self.config.project_id,
                        evaluation_result,
                        include_data=True
                    )
                    self.logger.info(f"Evaluation results added to Evidently cloud project {self.config.project_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to add results to Evidently cloud: {e}")
            
            # Add custom Groq-based contradiction detection if available
            contradiction_results = self._run_groq_contradiction_detection(evaluation_data)
            
            return {
                'report': evaluation_result,
                'dataset_size': len(evaluation_data),
                'evaluation_timestamp': time.time(),
                'contradiction_analysis': contradiction_results
            }
            
        except Exception as e:
            self.logger.error(f"Error during Evidently evaluation: {e}")
            return {'error': str(e)}
    
    def _run_groq_contradiction_detection(self, evaluation_data: pd.DataFrame) -> Dict[str, Any]:
        """Run contradiction detection using Groq instead of OpenAI."""
        try:
            if not GROQ_API_KEY:
                self.logger.warning("GROQ_API_KEY not available for contradiction detection")
                return {'status': 'no_groq_key', 'message': 'GROQ_API_KEY required for contradiction detection'}
            
            self.logger.info("Running Groq-based contradiction detection...")
            
            # Use Groq for contradiction detection
            groq_client = self.llm_client.get_groq()
            
            contradiction_prompt = """You are an expert evaluator. Your task is to determine if the GENERATED_ANSWER contradicts the REFERENCE_ANSWER.

REFERENCE_ANSWER:
{reference}

GENERATED_ANSWER:
{generated}

Instructions:
- Label as "CONTRADICTORY" only if the generated answer directly contradicts facts in the reference
- Label as "NON_CONTRADICTORY" if they are consistent (even if different in length/wording)
- Label as "UNKNOWN" if you cannot determine

Provide your response in this exact format:
LABEL: [CONTRADICTORY/NON_CONTRADICTORY/UNKNOWN]
REASONING: [Brief explanation of your decision]

Response:"""

            results = []
            for idx, row in evaluation_data.iterrows():
                try:
                    prompt = contradiction_prompt.format(
                        reference=row['reference_answer'],
                        generated=row['generated_answer']
                    )
                    
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=200
                    )
                    
                    result_text = response.choices[0].message.content
                    
                    # Parse the response
                    label = "UNKNOWN"
                    reasoning = "Could not parse response"
                    
                    if "LABEL:" in result_text:
                        label_line = [line for line in result_text.split('\n') if line.startswith('LABEL:')]
                        if label_line:
                            label = label_line[0].replace('LABEL:', '').strip()
                    
                    if "REASONING:" in result_text:
                        reasoning_line = [line for line in result_text.split('\n') if line.startswith('REASONING:')]
                        if reasoning_line:
                            reasoning = reasoning_line[0].replace('REASONING:', '').strip()
                    
                    results.append({
                        'question_idx': idx,
                        'label': label,
                        'reasoning': reasoning,
                        'raw_response': result_text
                    })
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to analyze contradiction for question {idx}: {e}")
                    results.append({
                        'question_idx': idx,
                        'label': 'ERROR',
                        'reasoning': f'Analysis failed: {str(e)}',
                        'raw_response': ''
                    })
            
            return {
                'status': 'success',
                'total_analyzed': len(results),
                'contradictory_count': len([r for r in results if r['label'] == 'CONTRADICTORY']),
                'non_contradictory_count': len([r for r in results if r['label'] == 'NON_CONTRADICTORY']),
                'unknown_count': len([r for r in results if r['label'] == 'UNKNOWN']),
                'error_count': len([r for r in results if r['label'] == 'ERROR']),
                'detailed_results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error during Groq contradiction detection: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_evaluation_summary(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable summary of evaluation results."""
        try:
            summary = f"""
RAG Pipeline Evaluation Summary
===============================

Session ID: {results.get('chat_uid', 'N/A')}
Total Questions: {results.get('total_questions', 0)}
Successful Responses: {results.get('successful_responses', 0)}
Success Rate: {(results.get('successful_responses', 0) / max(results.get('total_questions', 1), 1)) * 100:.1f}%

Evaluation Timestamp: {pd.to_datetime(results.get('timestamp', 0), unit='s')}

Evidently Results: {'Available' if results.get('evidently_results') else 'Not Available'}
"""
            
            # Add contradiction analysis summary if available
            evidently_results = results.get('evidently_results', {})
            if isinstance(evidently_results, dict) and 'contradiction_analysis' in evidently_results:
                contra_analysis = evidently_results['contradiction_analysis']
                if contra_analysis.get('status') == 'success':
                    summary += f"""
Contradiction Analysis (Groq-based):
====================================
Total Analyzed: {contra_analysis.get('total_analyzed', 0)}
Contradictory: {contra_analysis.get('contradictory_count', 0)}
Non-Contradictory: {contra_analysis.get('non_contradictory_count', 0)}
Unknown: {contra_analysis.get('unknown_count', 0)}
Errors: {contra_analysis.get('error_count', 0)}
Contradiction Rate: {(contra_analysis.get('contradictory_count', 0) / max(contra_analysis.get('total_analyzed', 1), 1)) * 100:.1f}%
"""
                elif contra_analysis.get('status') == 'no_groq_key':
                    summary += "\nContradiction Analysis: GROQ_API_KEY not available"
                else:
                    summary += f"\nContradiction Analysis: {contra_analysis.get('message', 'Unknown error')}"
            
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating evaluation summary: {e}")
            return f"Error generating summary: {e}"

    def get_contradiction_details(self, results: Dict[str, Any]) -> str:
        """Get detailed contradiction analysis results."""
        try:
            evidently_results = results.get('evidently_results', {})
            if not isinstance(evidently_results, dict) or 'contradiction_analysis' not in evidently_results:
                return "No contradiction analysis available"
            
            contra_analysis = evidently_results['contradiction_analysis']
            if contra_analysis.get('status') != 'success':
                return f"Contradiction analysis failed: {contra_analysis.get('message', 'Unknown error')}"
            
            details = f"""
Detailed Contradiction Analysis
==============================

Summary:
- Total Questions Analyzed: {contra_analysis.get('total_analyzed', 0)}
- Contradictory Responses: {contra_analysis.get('contradictory_count', 0)}
- Non-Contradictory Responses: {contra_analysis.get('non_contradictory_count', 0)}
- Unknown/Unclear: {contra_analysis.get('unknown_count', 0)}
- Analysis Errors: {contra_analysis.get('error_count', 0)}

Detailed Results:
"""
            
            for result in contra_analysis.get('detailed_results', []):
                details += f"""
Question {result.get('question_idx', 'N/A')}:
- Label: {result.get('label', 'N/A')}
- Reasoning: {result.get('reasoning', 'N/A')}
- Raw Response: {result.get('raw_response', 'N/A')[:200]}{'...' if len(result.get('raw_response', '')) > 200 else ''}
"""
            
            return details.strip()
            
        except Exception as e:
            self.logger.error(f"Error getting contradiction details: {e}")
            return f"Error getting contradiction details: {e}"


def create_sample_evaluation_data() -> Tuple[List[str], List[str]]:
    """Create sample evaluation data for testing."""
    sample_questions = [
        "What is the primary method of diagnosis for Trichuris infections?",
        "What are the common symptoms of light infections caused by Strongyloides stercoralis?",
        "What is the name of the disease caused by the parasites Wuchereria bancrofti and Brugia malayi?",
        "What is the typical shape and size of eggs laid by Enterobius vermicularis?",
        "What are the potential complications of heavy, chronic infections caused by Strongyloides stercoralis?"
    ]
    
    sample_answers = [
        "The primary method of diagnosis for Trichuris infections is based on symptoms and the presence of eggs in faeces.",
        "Light infections of Strongyloides stercoralis are often asymptomatic, but when symptoms occur, they may include itching and red blotches due to skin penetration, bronchial verminous pneumonia, burning mid-epigastric pain and tenderness accompanied by nausea and vomiting, and alternating diarrhea and constipation.",
        "The disease caused by the parasites Wuchereria bancrofti and Brugia malayi is Filariasis, also known as elephantiasis.",
        "Eggs of Enterobius vermicularis are ovoid but asymmetrically flat on one side, measuring 60 micrometers x 27 micrometers.",
        "Heavy, chronic infections caused by Strongyloides stercoralis can result in anemia, weight loss, and chronic bloody dysentery. Additionally, secondary bacterial infection of damaged mucosa may produce serious complications."
    ]
    
    return sample_questions, sample_answers


if __name__ == "__main__":
    # Example usage
    config = EvaluationConfig(
        backend_url="http://localhost:5000",
        evidently_cloud_url="https://app.evidently.cloud/",
        project_id="your-project-id-here"
    )
    
    evaluator = RAGEvaluator(config)
    
    # Create sample data
    questions, answers = create_sample_evaluation_data()
    
    # Run evaluation
    results = evaluator.evaluate_rag_pipeline(
        questions=questions,
        reference_answers=answers,
        chat_uid="demo-eval-1"
    )
    
    # Print summary
    print(evaluator.get_evaluation_summary(results))
