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

Below is text extracted from a PDF. Your task is to locate specific sections and extract data accurately.

STEP 1 - LOCATE THE DELINQUENCY TABLE:
Find the section with "DELINQUENCY", "DELINQUENT", or "DELINQUENT STATUS" in the header.
This table contains the loan performance breakdown. ONLY extract delinquency data from this section.

STEP 2 - EXTRACT FROM THE DELINQUENCY TABLE:
Common categories include:
- Current (loans that are up to date)
- 30-59 days delinquent
- 60-89 days delinquent
- 90-119 days delinquent
- 120+ days delinquent (or 90+ days)
- Foreclosure
- REO (Real Estate Owned)
- Bankruptcy

For EACH category, extract:
- category: The exact status name
- count: Number of loans (integer)
- balance: Dollar amount (number, no $ or commas)

STEP 3 - EXTRACT POOL BALANCES:
Look for Beginning and Ending pool balances in:
- Summary Table / Factor Information
- Collateral Performance / Principal Reconciliation section

Keywords for Beginning Balance: "Beginning Pool Balance", "Aggregate Beginning Balance", "Previous Period UPB"
Keywords for Ending Balance: "Ending Pool Balance", "Aggregate Ending Principal Balance", "Current Period UPB"

STEP 4 - EXTRACT METADATA:
- series_name: Found in document header/title (e.g., "Series 2025-2", "GSAMP Trust 2006-S1")
- report_date: The distribution/report date (format as MM/DD/YYYY)

DOCUMENT TEXT:
---
{text}
---

OUTPUT FORMAT - Return ONLY this JSON:
{{
  "series_name": "string or null",
  "report_date": "MM/DD/YYYY or null",
  "beginning_balance": number or null,
  "ending_balance": number or null,
  "delinquency": [
    {{"category": "Current", "count": 450, "balance": 9500000.00}},
    {{"category": "30-59 Days Delinquent", "count": 20, "balance": 250000.00}}
  ]
}}

CRITICAL RULES:
- Extract EVERY row from the delinquency table - do not skip any categories
- Match each count and balance to its correct category
- Convert all dollar amounts to plain numbers (no $, no commas)

Return ONLY the JSON object, no additional text or markdown.
"""


class PDFTextExtractor:
    """Extract mortgage data using PDF text parsing + Gemini."""

    # Max characters to send to API (roughly 100k tokens limit, ~4 chars/token)
    MAX_TEXT_LENGTH = 300000

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash"):
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

        # Truncate text if too long
        if len(text) > self.MAX_TEXT_LENGTH:
            text = text[:self.MAX_TEXT_LENGTH] + "\n\n[TEXT TRUNCATED]"

        prompt = EXTRACTION_PROMPT.format(text=text)

        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0,
                max_output_tokens=4096,
            ),
        )

        # Handle potential response issues
        if not response.candidates:
            raise ValueError("No response candidates returned from API")

        if response.candidates[0].finish_reason.name == "SAFETY":
            raise ValueError("Response blocked by safety filters")

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
