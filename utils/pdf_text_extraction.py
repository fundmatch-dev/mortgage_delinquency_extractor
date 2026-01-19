"""Approach 2: PDF Text Parsing + Gemini Pro Extraction."""

import json
import time
from pathlib import Path
from typing import Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# PDF parsing libraries
import pdfplumber
import pypdf
import fitz  # pymupdf


EXTRACTION_PROMPT = """You are an expert at extracting structured data from mortgage-backed securities trustee reports.

Below is the text content extracted from a PDF document. Analyze it and extract the following information:

1. **series_name**: The series identifier (e.g., "Series 2025-2", "GSAMP Trust 2006-S1", etc.)
2. **report_date**: The date of this report (format as MM/DD/YYYY)
3. **beginning_balance**: The total beginning balance of the loan pool (dollar amount)
4. **ending_balance**: The total ending balance of the loan pool (dollar amount)
5. **delinquency**: A list of delinquency categories, each containing:
   - category: The delinquency status (e.g., "Current", "30-59 days delinquent", "60-89 days", "90+ days", "Foreclosure", "REO", "Bankruptcy")
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
- Extract ALL delinquency categories you can find
- For balances, parse numbers removing currency symbols and commas
- For loan counts, extract only the integer value
- If a value cannot be found, use null
- Tables may appear as space-separated or tab-separated values
- Look for patterns like "Current", "30-59", "60-89", "90+" in delinquency data

DOCUMENT TEXT:
---
{text}
---

Return ONLY the JSON object, no additional text or markdown formatting.
"""


class PDFTextExtractor:
    """Extract mortgage data using PDF text parsing + Gemini Pro."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the PDF text extractor.

        Args:
            api_key: Google AI API key
            model_name: Gemini model to use for text analysis
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.last_request_time = 0
        self.min_request_interval = 1.0

    def _rate_limit(self):
        """Implement basic rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def extract_text_pdfplumber(self, pdf_path: Path) -> tuple[str, dict]:
        """
        Extract text from PDF using pdfplumber (best for tables).

        Returns:
            Tuple of (extracted_text, metadata)
        """
        text_parts = []
        metadata = {"pages": 0, "tables_found": 0, "method": "pdfplumber"}

        with pdfplumber.open(pdf_path) as pdf:
            metadata["pages"] = len(pdf.pages)

            for page in pdf.pages:
                # Extract tables first (better structure preservation)
                tables = page.extract_tables()
                if tables:
                    metadata["tables_found"] += len(tables)
                    for table in tables:
                        table_text = "\n".join(
                            "\t".join(str(cell) if cell else "" for cell in row)
                            for row in table
                        )
                        text_parts.append(f"[TABLE]\n{table_text}\n[/TABLE]")

                # Extract regular text
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n\n".join(text_parts), metadata

    def extract_text_pypdf(self, pdf_path: Path) -> tuple[str, dict]:
        """
        Extract text from PDF using pypdf (fallback method).

        Returns:
            Tuple of (extracted_text, metadata)
        """
        text_parts = []
        metadata = {"pages": 0, "method": "pypdf"}

        reader = pypdf.PdfReader(pdf_path)
        metadata["pages"] = len(reader.pages)

        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        return "\n\n".join(text_parts), metadata

    def extract_text_pymupdf(self, pdf_path: Path) -> tuple[str, dict]:
        """
        Extract text from PDF using pymupdf (fast and accurate).

        Returns:
            Tuple of (extracted_text, metadata)
        """
        text_parts = []
        metadata = {"pages": 0, "method": "pymupdf"}

        doc = fitz.open(pdf_path)
        metadata["pages"] = len(doc)

        for page in doc:
            text = page.get_text()
            if text:
                text_parts.append(text)

        doc.close()
        return "\n\n".join(text_parts), metadata

    def extract_text(self, pdf_path: Path, method: str = "pdfplumber") -> tuple[str, dict]:
        """
        Extract text from PDF using specified method.

        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('pdfplumber', 'pypdf', 'pymupdf')

        Returns:
            Tuple of (extracted_text, metadata)
        """
        methods = {
            "pdfplumber": self.extract_text_pdfplumber,
            "pypdf": self.extract_text_pypdf,
            "pymupdf": self.extract_text_pymupdf,
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")

        return methods[method](pdf_path)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((Exception,)),
    )
    def _call_api(self, text: str) -> str:
        """Make API call with retry logic."""
        self._rate_limit()

        prompt = EXTRACTION_PROMPT.format(text=text)

        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )

        return response.text

    def extract(self, pdf_path: str | Path, text_method: str = "pdfplumber") -> dict:
        """
        Extract data from a single PDF.

        Args:
            pdf_path: Path to the PDF file
            text_method: Method for text extraction

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
            "approach": "pdf_text_gemini",
            "text_extraction_metadata": None,
        }

        try:
            # Step 1: Extract text from PDF
            text, text_metadata = self.extract_text(pdf_path, method=text_method)
            result["text_extraction_metadata"] = text_metadata

            if not text.strip():
                result["error"] = "No text extracted from PDF"
                result["processing_time"] = time.time() - start_time
                return result

            # Step 2: Send to Gemini for structured extraction
            response_text = self._call_api(text)

            # Parse JSON from response
            text_response = response_text.strip()
            if text_response.startswith("```json"):
                text_response = text_response[7:]
            if text_response.startswith("```"):
                text_response = text_response[3:]
            if text_response.endswith("```"):
                text_response = text_response[:-3]
            text_response = text_response.strip()

            data = json.loads(text_response)
            data["filename"] = pdf_path.name

            result["success"] = True
            result["data"] = data

        except json.JSONDecodeError as e:
            result["error"] = f"JSON parsing error: {str(e)}"
        except Exception as e:
            result["error"] = f"Extraction error: {str(e)}"

        result["processing_time"] = time.time() - start_time
        return result

    def extract_batch(self, pdf_paths: list[Path], text_method: str = "pdfplumber", progress_callback=None) -> list[dict]:
        """
        Extract data from multiple PDFs.

        Args:
            pdf_paths: List of PDF file paths
            text_method: Method for text extraction
            progress_callback: Optional callback function(current, total, filename)

        Returns:
            List of extraction results
        """
        results = []
        total = len(pdf_paths)

        for idx, pdf_path in enumerate(pdf_paths):
            if progress_callback:
                progress_callback(idx + 1, total, pdf_path.name)

            result = self.extract(pdf_path, text_method=text_method)
            results.append(result)

        return results

    def estimate_cost(self, avg_text_length: int, num_documents: int) -> dict:
        """
        Estimate API cost for processing documents.

        Args:
            avg_text_length: Average character count of extracted text
            num_documents: Number of documents to process

        Returns:
            Cost estimation dictionary
        """
        # Approximate tokens (4 chars per token)
        tokens_per_doc = avg_text_length // 4
        output_tokens_per_doc = 500

        total_input_tokens = tokens_per_doc * num_documents
        total_output_tokens = output_tokens_per_doc * num_documents

        # Gemini Pro pricing (approximate)
        input_cost_per_1k = 0.00125   # Per 1K tokens
        output_cost_per_1k = 0.00375  # Per 1K tokens

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
