"""
RAG System Evaluation Module using Evidently AI Cloud.

This module provides tools to evaluate the performance of the Retrieval-Augmented Generation (RAG)
system using Evidently AI cloud for metrics, reports, and monitoring. It includes a configuration
class (`EvaluationConfig`), an evaluator class (`RAGEvaluator`), and a utility function to generate
sample evaluation data for testing. Optimized for production with retry logic and robust error handling.
"""
import os
import time
import json
import random
import requests
import torch
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from retry import retry
from src.loghandler import set_logger, ColorFormatter
from src.config import GROQ_API_KEY, EVIDENTLY_CLOUD_URL, EVIDENTLY_API_KEY
from src.models import LLMClient
from src.exceptions import *

# Initialize logger first to ensure availability for imports
API_DIR = Path(__file__).resolve().parent.parent
LOG_FILENAME = str(API_DIR / "logs" / "status_logs.log")
logger = set_logger(
    to_file=True,
    log_file_name=LOG_FILENAME,
    to_console=True,
    custom_formatter=ColorFormatter
)

# Evidently AI imports with fallback
try:
    from evidently.ui.workspace import CloudWorkspace
    from evidently import Dataset, DataDefinition, Report
    from evidently.descriptors import Sentiment, TextLength
    from evidently.presets import TextEvals
    from evidently.llm.templates import BinaryClassificationPromptTemplate
except ImportError:
    logger.error("Evidently LLM modules not found. Falling back to basic metrics.")
    from evidently.ui.workspace import CloudWorkspace
    from evidently import Dataset, DataDefinition, Report
    from evidently.descriptors import Sentiment, TextLength
    TextEvals = None
    BinaryClassificationPromptTemplate = None

@dataclass
class EvaluationConfig:
    """
    Configuration class for RAG evaluation runs.

    Stores settings for connecting to the RAG backend, Evidently AI cloud, and evaluation parameters.

    Attributes:
        backend_url (str): URL of the FastAPI backend server.
        evidently_cloud_url (str): URL of the Evidently AI cloud service.
        evidently_api_key (str): API key for Evidently AI cloud.
        project_id (Optional[str]): ID of the Evidently project for storing results.
        model (str): Model identifier for the Groq LLM.
        chatbot_name (str): Name of the chatbot used in evaluations.
        max_retries (int): Maximum number of retries for API calls.
        timeout (int): Timeout for API requests in seconds.
        similarity_top_k (int): Number of top similar documents to retrieve.
        chunk_size (int): Size of text chunks for embedding generation.
        chunk_overlap (int): Overlap between text chunks for embedding.
    """
    backend_url: str = "http://localhost:5000"
    evidently_cloud_url: str = "https://app.evidently.cloud/"
    evidently_api_key: str = EVIDENTLY_API_KEY
    project_id: Optional[str] = None
    model: str = "llama-3.3-70b-versatile"
    chatbot_name: str = "RAGEvaluator"
    max_retries: int = 3
    timeout: int = 300
    similarity_top_k: int = 5
    chunk_size: int = 1024
    chunk_overlap: int = 40

class RAGEvaluator:
    """
    Evaluates the performance of the RAG system using Evidently AI cloud metrics.

    Manages the evaluation process, including file indexing, querying the RAG system,
    running metrics with Evidently AI, and performing contradiction detection using Groq.

    Attributes:
        config (EvaluationConfig): Configuration settings for the evaluation.
        logger (logging.Logger): Logger for tracking evaluation operations.
        evidently_client (CloudWorkspace): Cached client for Evidently AI cloud.
        llm_client (LLMClient): Client for Groq-based LLM operations.
    """
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the RAG evaluator with configuration.

        Sets up logging, connects to Evidently AI cloud, and validates the configuration.

        Args:
            config (EvaluationConfig): Configuration object for the evaluation.

        Raises:
            ConnectionError: If connection to Evidently AI cloud fails.
            ValueError: If required configuration fields are missing.
        """
        self.config = config
        self.logger = self._setup_logger()
        self.logger.info(f"PyTorch version: {torch.__version__}, Platform: {torch.__version__.__platform__ or 'unknown'}")
        self.llm_client = LLMClient()
        self.evidently_client = None
        self._validate_config()
        self.evidently_client = self._setup_evidently_client()

    def _setup_logger(self) -> Any:
        """
        Set up logging for evaluation operations.

        Creates a logger that writes to a file (`logs/evaluation_logs.log`) and console with colorized output.

        Returns:
            logging.Logger: Configured logger instance.
        """
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        return set_logger(
            logger_name="rag_evaluator",
            to_file=True,
            log_file_name=str(log_dir / "evaluation_logs.log"),
            to_console=True,
            custom_formatter=ColorFormatter
        )

    @retry(tries=3, delay=1, backoff=2, exceptions=(Exception,))
    def _setup_evidently_client(self) -> CloudWorkspace:
        """
        Initialize connection to Evidently AI cloud with retry logic.

        Returns:
            CloudWorkspace: Configured Evidently AI cloud client.

        Raises:
            ConnectionError: If connection fails after retries.
        """
        try:
            self.logger.info("Connecting to Evidently AI cloud workspace...")
            client = CloudWorkspace(url=self.config.evidently_cloud_url, token=self.config.evidently_api_key)
            self.logger.info("Successfully connected to Evidently AI cloud")
            return client
        except Exception as e:
            error_msg = f"Failed to connect to Evidently AI cloud: {e}"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)

    def _validate_config(self) -> None:
        """
        Validate evaluation configuration parameters.

        Ensures required fields are set and logs warnings for optional fields.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = [
            (self.config.backend_url, "Backend URL"),
            (self.config.evidently_cloud_url, "Evidently cloud URL"),
            (self.config.evidently_api_key, "Evidently API key")
        ]
        for value, name in required_fields:
            if not value:
                raise ValueError(f"{name} must be provided")
        if not GROQ_API_KEY:
            self.logger.warning("GROQ_API_KEY not found in environment variables")
        self.logger.info(f"Evaluation config validated: backend={self.config.backend_url}, model={self.config.model}")

    @retry(tries=3, delay=1, backoff=2, exceptions=(requests.RequestException,))
    def _upload_and_index_files(self, files: List[str], chat_uid: str) -> bool:
        """
        Upload and index files for evaluation using the RAG backend.

        Args:
            files (List[str]): List of file paths to upload and index.
            chat_uid (str): Unique identifier for the chat session.

        Returns:
            bool: True if indexing succeeds, False otherwise.

        Raises:
            Exception: If file upload or indexing fails after retries.
        """
        try:
            self.logger.info(f"Uploading and indexing {len(files)} files for session {chat_uid}")
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
                    self.logger.error(f"Indexing failed: {response.status_code}, {response.text}")
                    return False
        except Exception as e:
            self.logger.error(f"Error during file indexing: {e}")
            return False

    @retry(tries=3, delay=1, backoff=2, exceptions=(requests.RequestException,))
    def _query_rag_system(self, question: str, chat_uid: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Query the RAG system and retrieve response and context.

        Args:
            question (str): Query to send to the RAG system.
            chat_uid (str): Unique identifier for the chat session.

        Returns:
            Tuple[Optional[str], Optional[str]]: Generated response and context (placeholder).

        Raises:
            Exception: If querying fails after retries.
        """
        try:
            self.logger.info(f"Querying RAG system: {question[:100]}...")
            payload = {
                "query": question,
                "model": self.config.model,
                "chat_uid": chat_uid,
                "chatbot_name": self.config.chatbot_name
            }
            response = requests.post(
                f"{self.config.backend_url}/chat",
                json=payload,
                timeout=self.config.timeout,
                stream=True
            )

            if response.status_code != 200:
                self.logger.error(f"Chat request failed: {response.status_code}, {response.text}")
                return None, None

            accumulated_response = ""
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=False):
                if chunk:
                    try:
                        accumulated_response += chunk.decode("utf-8")
                    except UnicodeDecodeError:
                        accumulated_response += chunk.decode("latin-1", errors="ignore")

            if not accumulated_response.strip():
                self.logger.warning("Empty response from RAG system")
                return None, None

            self.logger.info(f"Generated response (length: {len(accumulated_response)})")
            return accumulated_response, "Context retrieval not yet implemented"
        except Exception as e:
            self.logger.error(f"Error querying RAG system: {e}")
            return None, None

    def _create_contradiction_template(self) -> Optional[Any]:
        """
        Create a contradiction detection template for LLM-based evaluation.

        Returns:
            BinaryClassificationPromptTemplate or None: Configured template or None if not available.
        """
        if BinaryClassificationPromptTemplate is None:
            self.logger.warning("BinaryClassificationPromptTemplate not available; skipping contradiction detection.")
            return None
        return BinaryClassificationPromptTemplate(
            criteria="""Label an ANSWER as **contradictory** only if it directly contradicts any part of the REFERENCE.
            Differences in length or wording are acceptable. It is acceptable if the ANSWER adds details, but not if it omits facts that cause contradiction.
            Compare factual consistency only, not completeness or style.

            REFERENCE:
            =====
            {reference}
            =====
            """,
            target_category="contradictory",
            non_target_category="non-contradictory",
            uncertainty="unknown",
            include_reasoning=True,
            pre_messages=[("system", "You are an expert evaluator comparing factual consistency.")]
        )

    def evaluate_rag_pipeline(
        self,
        questions: List[str],
        reference_answers: List[str],
        chat_uid: str,
        files_to_index: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the RAG pipeline performance against reference data.

        Args:
            questions (List[str]): Evaluation questions.
            reference_answers (List[str]): Reference answers for comparison.
            chat_uid (str): Unique identifier for the chat session.
            files_to_index (Optional[List[str]]): File paths to index.

        Returns:
            Dict[str, Any]: Evaluation results with question counts, responses, and metrics.

        Raises:
            RuntimeError: If evaluation fails.
        """
        try:
            self.logger.info(f"Starting evaluation for {len(questions)} questions")
            if files_to_index and not self._upload_and_index_files(files_to_index, chat_uid):
                raise RuntimeError("Failed to index files for evaluation")

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
                time.sleep(0.5)

            evaluation_data = pd.DataFrame({
                'question': questions,
                'reference_answer': reference_answers,
                'generated_answer': generated_responses,
                'context': retrieved_contexts
            })

            evaluation_results = self._run_evidently_evaluation(evaluation_data)
            results = {
                'chat_uid': chat_uid,
                'total_questions': len(questions),
                'successful_responses': len([r for r in generated_responses if not r.startswith("ERROR")]),
                'evaluation_data': evaluation_data.to_dict('records'),
                'evidently_results': evaluation_results,
                'timestamp': time.time()
            }
            self.logger.info(f"Evaluation completed: {results['successful_responses']}/{results['total_questions']} successful")
            return results
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise RuntimeError(f"Error during RAG pipeline evaluation: {e}")

    def _run_evidently_evaluation(self, evaluation_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run Evidently AI evaluation on the dataset.

        Args:
            evaluation_data (pd.DataFrame): DataFrame with questions, reference answers, generated answers, and contexts.

        Returns:
            Dict[str, Any]: Evaluation report, dataset size, timestamp, and contradiction analysis.

        Raises:
            Exception: If evaluation fails.
        """
        try:
            self.logger.info("Running Evidently AI evaluation...")
            dataset = Dataset.from_pandas(
                evaluation_data,
                data_definition=DataDefinition(),
                descriptors=[Sentiment("generated_answer"), TextLength("generated_answer")]
            )

            report = Report([TextEvals()] if TextEvals else [Sentiment(), TextLength()])
            evaluation_result = report.run(dataset, None)

            if self.config.project_id:
                try:
                    self.evidently_client.add_run(
                        self.config.project_id,
                        evaluation_result,
                        include_data=True
                    )
                    self.logger.info(f"Results added to Evidently project {self.config.project_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to upload to Evidently cloud: {e}")

            contradiction_results = self._run_groq_contradiction_detection(evaluation_data)
            return {
                'report': evaluation_result,
                'dataset_size': len(evaluation_data),
                'evaluation_timestamp': time.time(),
                'contradiction_analysis': contradiction_results
            }
        except Exception as e:
            self.logger.error(f"Evidently evaluation failed: {e}")
            return {'error': str(e), 'contradiction_analysis': {'status': 'failed', 'message': str(e)}}

    def _run_groq_contradiction_detection(self, evaluation_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run contradiction detection using Groq LLM.

        Args:
            evaluation_data (pd.DataFrame): DataFrame with reference and generated answers.

        Returns:
            Dict[str, Any]: Contradiction analysis results.

        Raises:
            Exception: If contradiction detection fails.
        """
        if not GROQ_API_KEY:
            self.logger.warning("GROQ_API_KEY not available for contradiction detection")
            return {'status': 'no_groq_key', 'message': 'GROQ_API_KEY required'}

        if BinaryClassificationPromptTemplate is None:
            self.logger.warning("Contradiction detection not available due to missing Evidently LLM modules")
            return {'status': 'no_llm_template', 'message': 'Evidently LLM template not available'}

        try:
            self.logger.info("Running Groq-based contradiction detection...")
            groq_client = self.llm_client.get_groq()
            contradiction_prompt = self._create_contradiction_template()
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
                    time.sleep(1)
                except Exception as e:
                    self.logger.warning(f"Contradiction analysis failed for question {idx}: {e}")
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
            self.logger.error(f"Groq contradiction detection failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_evaluation_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of evaluation results.

        Args:
            results (Dict[str, Any]): Evaluation results dictionary.

        Returns:
            str: Formatted summary string.

        Raises:
            Exception: If summary generation fails.
        """
        try:
            success_rate = (results.get('successful_responses', 0) / max(results.get('total_questions', 1), 1)) * 100
            summary = f"""
RAG Pipeline Evaluation Summary
==============================
Session ID: {results.get('chat_uid', 'N/A')}
Total Questions: {results.get('total_questions', 0)}
Successful Responses: {results.get('successful_responses', 0)}
Success Rate: {success_rate:.1f}%
Evaluation Timestamp: {pd.to_datetime(results.get('timestamp', 0), unit='s')}
Evidently Results: {'Available' if results.get('evidently_results') else 'Not Available'}
"""
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
                else:
                    summary += f"\nContradiction Analysis: {contra_analysis.get('message', 'Unknown error')}"
            return summary.strip()
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {e}"

    def get_contradiction_details(self, results: Dict[str, Any]) -> str:
        """
        Generate detailed contradiction analysis results.

        Args:
            results (Dict[str, Any]): Evaluation results dictionary.

        Returns:
            str: Formatted contradiction analysis details.

        Raises:
            Exception: If detail generation fails.
        """
        try:
            evidently_results = results.get('evidently_results', {})
            if not isinstance(evidently_results, dict) or 'contradiction_analysis' not in evidently_results:
                return "No contradiction analysis available"
            contra_analysis = evidently_results['contradiction_analysis']
            if contra_analysis.get('status') != 'success':
                return f"Contradiction analysis failed: {contra_analysis.get('message', 'Unknown error')}"
            details = f"""
Detailed Contradiction Analysis
--------- -------------
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
    """
    Create sample evaluation data for testing the RAG system.

    Returns:
        Tuple[List[str], List[str]]: Sample questions and reference answers.
    """
    sample_questions = [
        "What is the primary method of diagnosis for Trichuris infections?",
        "What are the common symptoms of light infections caused by Strongyloides stercoralis?",
        "What is the name of the disease caused by the parasites Wuchereria bancrofti and Brugia malayi?",
        "What is the typical shape and size of eggs laid by Enterobius vermicularis?",
        "What are the potential complications of heavy, chronic infections caused by Strongyloides stercoralis?"
    ]
    sample_answers = [
        "The primary method of diagnosis for Trichuris infections is based on symptoms and the presence of eggs in faeces.",
        "Light infections of Strongyloides stercoralis are often asymptomatic, but may include itching, red blotches, bronchial verminous pneumonia, epigastric pain, nausea, vomiting, and alternating diarrhea/constipation.",
        "The disease caused by Wuchereria bancrofti and Brugia malayi is Filariasis, also known as elephantiasis.",
        "Eggs of Enterobius vermicularis are ovoid, asymmetrically flat on one side, measuring 60 micrometers x 27 micrometers.",
        "Heavy, chronic infections of Strongyloides stercoralis can cause anemia, weight loss, chronic bloody dysentery, and secondary bacterial infections."
    ]
    return sample_questions, sample_answers

if __name__ == "__main__":
    """
    Entry point for running a sample evaluation.
    """
    config = EvaluationConfig(
        backend_url="http://localhost:5000",
        evidently_cloud_url="https://app.evidently.cloud/",
        evidently_api_key=EVIDENTLY_API_KEY,
        project_id="your-project-id-here"
    )
    evaluator = RAGEvaluator(config)
    questions, answers = create_sample_evaluation_data()
    results = evaluator.evaluate_rag_pipeline(
        questions=questions,
        reference_answers=answers,
        chat_uid="demo-eval-1"
    )
    print(evaluator.get_evaluation_summary(results))