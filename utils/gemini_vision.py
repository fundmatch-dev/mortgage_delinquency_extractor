"""Approach 1: Gemini Vision Direct PDF Extraction."""

import json
import time
from pathlib import Path
from typing import Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel

from .validators import ExtractedDocument, DelinquencyCategory


EXTRACTION_PROMPT = """You are an expert at extracting structured data from mortgage-backed securities trustee reports.

Analyze this PDF document and extract the following information:

1. **series_name**: The series identifier (e.g., "Series 2025-2", "GSAMP Trust 2006-S1", etc.) Usually this will be found at the start of the document. 
2. **report_date**: The date of this report (format as MM/DD/YYYY)
3. **beginning_balance**: The total beginning balance of the loan pool (dollar amount)
4. **ending_balance**: The total ending balance of the loan pool (dollar amount)
5. **delinquency**: A list of delinquency categories, each containing:
   - category: The delinquency status (e.g., "Current", "30-59 days delinquent", "60-89 days", "90+ days", "Foreclosure", "REO", "Bankruptcy") Note that if there are columns with the categories, only take data from the DELINQUENCY category. Do not sum across the rows.
   - count: The number of loans in this category (integer)
   - balance: The dollar balance for this category (number)

Return ONLY valid JSON in this exact format:
{
  "series_name": "string or null",
  "report_date": "MM/DD/YYYY or null",
  "beginning_balance": number or null,
  "ending_balance": number or null,
  "delinquency": [
    {"category": "Current", "count": 450, "balance": 9500000.00},
    {"category": "30-59 days", "count": 20, "balance": 250000.00}
  ]
}

Important guidelines:
- Extract ALL delinquency categories you can find in the document. 
- For balances, remove any currency symbols, commas, and convert to numbers
- For loan counts, extract only the integer value
- If a value cannot be found, use null
- Look carefully at tables for delinquency data - they often contain the breakdown
- The beginning and ending balances are typically found in the pool summary section
- Series name may appear in the header or title of the document

Return ONLY the JSON object, no additional text or markdown formatting.
"""


class GeminiVisionExtractor:
    """Extract mortgage data from PDFs using Gemini Vision API."""

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize the Gemini Vision extractor.

        Args:
            api_key: Google AI API key
            model_name: Gemini model to use (default: gemini-2.0-flash-exp)
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests

    def _rate_limit(self):
        """Implement basic rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((Exception,)),
    )
    def _call_api(self, pdf_path: Path) -> str:
        """Make API call with retry logic."""
        self._rate_limit()

        # Upload the PDF file
        uploaded_file = genai.upload_file(pdf_path, mime_type="application/pdf")

        # Generate content
        response = self.model.generate_content(
            [EXTRACTION_PROMPT, uploaded_file],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )

        return response.text

    def extract(self, pdf_path: str | Path) -> dict:
        """
        Extract data from a single PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with extracted data and metadata
        """
        pdf_path = Path(pdf_path)
        start_time = time.time()

        result = {
            "filename": pdf_path.name,
            "success": False,
            "data": None,
            "error": None,
            "processing_time": 0,
            "approach": "gemini_vision",
        }

        try:
            response_text = self._call_api(pdf_path)

            # Parse JSON from response
            # Handle potential markdown code blocks
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            data = json.loads(text)
            data["filename"] = pdf_path.name

            result["success"] = True
            result["data"] = data

        except json.JSONDecodeError as e:
            result["error"] = f"JSON parsing error: {str(e)}"
        except Exception as e:
            result["error"] = f"Extraction error: {str(e)}"

        result["processing_time"] = time.time() - start_time
        return result

    def extract_batch(self, pdf_paths: list[Path], progress_callback=None) -> list[dict]:
        """
        Extract data from multiple PDFs.

        Args:
            pdf_paths: List of PDF file paths
            progress_callback: Optional callback function(current, total, filename)

        Returns:
            List of extraction results
        """
        results = []
        total = len(pdf_paths)

        for idx, pdf_path in enumerate(pdf_paths):
            if progress_callback:
                progress_callback(idx + 1, total, pdf_path.name)

            result = self.extract(pdf_path)
            results.append(result)

        return results

    def estimate_cost(self, num_pages: int, num_documents: int) -> dict:
        """
        Estimate API cost for processing documents.

        Args:
            num_pages: Average pages per document
            num_documents: Number of documents to process

        Returns:
            Cost estimation dictionary
        """
        # Rough estimates for Gemini pricing (as of 2024)
        # These are approximations and should be updated based on current pricing
        input_tokens_per_page = 1500  # Approximate for PDF content
        output_tokens_per_doc = 500   # Approximate for JSON response

        total_input_tokens = num_pages * input_tokens_per_page * num_documents
        total_output_tokens = output_tokens_per_doc * num_documents

        # Gemini Flash pricing (approximate)
        input_cost_per_1k = 0.00001875  # Per 1K tokens
        output_cost_per_1k = 0.000075   # Per 1K tokens

        input_cost = (total_input_tokens / 1000) * input_cost_per_1k
        output_cost = (total_output_tokens / 1000) * output_cost_per_1k

        return {
            "estimated_input_tokens": total_input_tokens,
            "estimated_output_tokens": total_output_tokens,
            "estimated_input_cost": input_cost,
            "estimated_output_cost": output_cost,
            "estimated_total_cost": input_cost + output_cost,
            "cost_per_document": (input_cost + output_cost) / num_documents if num_documents > 0 else 0,
        }
