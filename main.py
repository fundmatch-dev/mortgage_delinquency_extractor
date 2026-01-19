"""
Main extraction script for Mortgage-Backed Securities PDF data extraction.

Implements three approaches:
1. Gemini Vision Direct Extraction
2. PDF Text Parsing + Gemini Pro
3. Hybrid Approach (Approach 3)

Usage:
    python main.py --approach 1 --api-key YOUR_API_KEY
    python main.py --approach 2 --api-key YOUR_API_KEY
    python main.py --approach 3 --api-key YOUR_API_KEY
    python main.py --all --api-key YOUR_API_KEY
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from utils.gemini_vision import GeminiVisionExtractor
from utils.pdf_text_extraction import PDFTextExtractor
from utils.validators import DataValidator
from utils.output_formatter import ExcelFormatter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("extraction.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
OUTPUT_DIR = DATA_DIR / "output"


def get_pdf_files(pdf_dir: Path = PDF_DIR) -> list[Path]:
    """Get all PDF files from the data directory."""
    if not pdf_dir.exists():
        logger.warning(f"PDF directory does not exist: {pdf_dir}")
        return []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    return pdf_files


def run_approach_1(api_key: str, pdf_files: list[Path], output_suffix: str = "") -> list[dict]:
    """
    Approach 1: Gemini Vision Direct Extraction.

    Sends entire PDF to Gemini Vision API for direct extraction.
    """
    logger.info("=" * 60)
    logger.info("APPROACH 1: Gemini Vision Direct Extraction")
    logger.info("=" * 60)

    extractor = GeminiVisionExtractor(api_key=api_key, model_name="gemini-2.0-flash-exp")
    validator = DataValidator()
    formatter = ExcelFormatter()

    results = []
    successful = 0
    failed = 0

    with tqdm(pdf_files, desc="Extracting (Vision)", unit="file") as pbar:
        for pdf_path in pbar:
            pbar.set_postfix(file=pdf_path.name[:20])

            result = extractor.extract(pdf_path)
            results.append(result)

            if result["success"]:
                # Validate extracted data
                is_valid, validated_doc = validator.validate_document(
                    result["data"], pdf_path.name
                )
                result["validation"] = validator.get_validation_summary()

                if validated_doc:
                    formatter.add_document_from_model(validated_doc)
                    successful += 1
                else:
                    logger.warning(f"Validation failed for {pdf_path.name}: {result['validation']['errors']}")
                    failed += 1
            else:
                logger.error(f"Extraction failed for {pdf_path.name}: {result['error']}")
                failed += 1

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"mortgage_data_approach1{output_suffix}.xlsx"
    formatter.save_to_excel(output_file)

    # Save raw results as JSON
    json_file = OUTPUT_DIR / f"results_approach1{output_suffix}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Approach 1 Complete: {successful} successful, {failed} failed")
    logger.info(f"Output saved to: {output_file}")

    return results


def run_approach_2(api_key: str, pdf_files: list[Path], output_suffix: str = "") -> list[dict]:
    """
    Approach 2: PDF Text Parsing + Gemini Pro.

    Extracts text from PDF first, then sends to Gemini Pro for structured extraction.
    """
    logger.info("=" * 60)
    logger.info("APPROACH 2: PDF Text Parsing + Gemini Pro")
    logger.info("=" * 60)

    extractor = PDFTextExtractor(api_key=api_key, model_name="gemini-1.5-pro")
    validator = DataValidator()
    formatter = ExcelFormatter()

    results = []
    successful = 0
    failed = 0

    with tqdm(pdf_files, desc="Extracting (Text+Pro)", unit="file") as pbar:
        for pdf_path in pbar:
            pbar.set_postfix(file=pdf_path.name[:20])

            result = extractor.extract(pdf_path, text_method="pdfplumber")
            results.append(result)

            if result["success"]:
                is_valid, validated_doc = validator.validate_document(
                    result["data"], pdf_path.name
                )
                result["validation"] = validator.get_validation_summary()

                if validated_doc:
                    formatter.add_document_from_model(validated_doc)
                    successful += 1
                else:
                    logger.warning(f"Validation failed for {pdf_path.name}: {result['validation']['errors']}")
                    failed += 1
            else:
                logger.error(f"Extraction failed for {pdf_path.name}: {result['error']}")
                failed += 1

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"mortgage_data_approach2{output_suffix}.xlsx"
    formatter.save_to_excel(output_file)

    json_file = OUTPUT_DIR / f"results_approach2{output_suffix}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Approach 2 Complete: {successful} successful, {failed} failed")
    logger.info(f"Output saved to: {output_file}")

    return results


def run_approach_3(api_key: str, pdf_files: list[Path], output_suffix: str = "") -> list[dict]:
    """
    Approach 3: Hybrid Smart Extraction.

    Strategy:
    1. First attempt text extraction with pdfplumber
    2. Analyze text quality (check for table structure, completeness)
    3. If text extraction quality is good, use Gemini Pro (cheaper/faster)
    4. If text extraction is poor (mangled tables), fall back to Gemini Vision

    This approach optimizes for cost while maintaining accuracy.
    """
    logger.info("=" * 60)
    logger.info("APPROACH 3: Hybrid Smart Extraction")
    logger.info("=" * 60)

    vision_extractor = GeminiVisionExtractor(api_key=api_key, model_name="gemini-2.0-flash-exp")
    text_extractor = PDFTextExtractor(api_key=api_key, model_name="gemini-1.5-flash")
    validator = DataValidator()
    formatter = ExcelFormatter()

    results = []
    successful = 0
    failed = 0
    used_vision = 0
    used_text = 0

    def assess_text_quality(text: str, metadata: dict) -> bool:
        """
        Assess if extracted text is good enough for text-based extraction.

        Returns True if text quality is sufficient, False if vision is needed.
        """
        if not text or len(text) < 500:
            return False

        # Check for table indicators
        has_tables = metadata.get("tables_found", 0) > 0

        # Check for common delinquency keywords
        delinquency_keywords = ["current", "delinquent", "30-59", "60-89", "90+", "foreclosure"]
        keyword_count = sum(1 for kw in delinquency_keywords if kw.lower() in text.lower())

        # Check for balance indicators
        has_balance_data = "$" in text or "balance" in text.lower()

        # Quality score
        quality_score = 0
        if has_tables:
            quality_score += 2
        if keyword_count >= 3:
            quality_score += 2
        if has_balance_data:
            quality_score += 1
        if len(text) > 2000:
            quality_score += 1

        return quality_score >= 4

    with tqdm(pdf_files, desc="Extracting (Hybrid)", unit="file") as pbar:
        for pdf_path in pbar:
            pbar.set_postfix(file=pdf_path.name[:20])

            # Step 1: Try text extraction first
            try:
                text, text_metadata = text_extractor.extract_text(pdf_path, method="pdfplumber")
                text_quality_ok = assess_text_quality(text, text_metadata)
            except Exception as e:
                logger.warning(f"Text extraction failed for {pdf_path.name}: {e}")
                text_quality_ok = False

            # Step 2: Choose extraction method based on quality
            if text_quality_ok:
                result = text_extractor.extract(pdf_path, text_method="pdfplumber")
                result["hybrid_method"] = "text"
                used_text += 1
            else:
                result = vision_extractor.extract(pdf_path)
                result["hybrid_method"] = "vision"
                used_vision += 1

            results.append(result)

            if result["success"]:
                is_valid, validated_doc = validator.validate_document(
                    result["data"], pdf_path.name
                )
                result["validation"] = validator.get_validation_summary()

                if validated_doc:
                    formatter.add_document_from_model(validated_doc)
                    successful += 1
                else:
                    # If text method failed validation, try vision as fallback
                    if result.get("hybrid_method") == "text":
                        logger.info(f"Text extraction validation failed for {pdf_path.name}, trying vision...")
                        vision_result = vision_extractor.extract(pdf_path)
                        if vision_result["success"]:
                            is_valid, validated_doc = validator.validate_document(
                                vision_result["data"], pdf_path.name
                            )
                            if validated_doc:
                                formatter.add_document_from_model(validated_doc)
                                successful += 1
                                result["fallback_to_vision"] = True
                                used_vision += 1
                                used_text -= 1
                                continue

                    logger.warning(f"Validation failed for {pdf_path.name}")
                    failed += 1
            else:
                logger.error(f"Extraction failed for {pdf_path.name}: {result['error']}")
                failed += 1

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"mortgage_data_approach3{output_suffix}.xlsx"
    formatter.save_to_excel(output_file)

    json_file = OUTPUT_DIR / f"results_approach3{output_suffix}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Approach 3 Complete: {successful} successful, {failed} failed")
    logger.info(f"Method breakdown: {used_text} text-based, {used_vision} vision-based")
    logger.info(f"Output saved to: {output_file}")

    return results


def run_all_approaches(api_key: str, pdf_files: list[Path]) -> dict:
    """Run all three approaches and return combined results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "approach_1": run_approach_1(api_key, pdf_files, f"_{timestamp}"),
        "approach_2": run_approach_2(api_key, pdf_files, f"_{timestamp}"),
        "approach_3": run_approach_3(api_key, pdf_files, f"_{timestamp}"),
    }

    # Save combined summary
    summary = {
        "timestamp": timestamp,
        "total_files": len(pdf_files),
        "approach_1": {
            "successful": sum(1 for r in results["approach_1"] if r["success"]),
            "failed": sum(1 for r in results["approach_1"] if not r["success"]),
            "avg_time": sum(r["processing_time"] for r in results["approach_1"]) / len(results["approach_1"]) if results["approach_1"] else 0,
        },
        "approach_2": {
            "successful": sum(1 for r in results["approach_2"] if r["success"]),
            "failed": sum(1 for r in results["approach_2"] if not r["success"]),
            "avg_time": sum(r["processing_time"] for r in results["approach_2"]) / len(results["approach_2"]) if results["approach_2"] else 0,
        },
        "approach_3": {
            "successful": sum(1 for r in results["approach_3"] if r["success"]),
            "failed": sum(1 for r in results["approach_3"] if not r["success"]),
            "avg_time": sum(r["processing_time"] for r in results["approach_3"]) / len(results["approach_3"]) if results["approach_3"] else 0,
        },
    }

    summary_file = OUTPUT_DIR / f"summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_file}")
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract mortgage data from PDF trustee reports"
    )
    parser.add_argument(
        "--approach",
        type=int,
        choices=[1, 2, 3],
        help="Which extraction approach to use (1, 2, or 3)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three approaches",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("GOOGLE_API_KEY"),
        help="Google AI API key (or set GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=str(PDF_DIR),
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of PDFs to process (for testing)",
    )

    args = parser.parse_args()

    # Validate API key
    if not args.api_key:
        logger.error("API key is required. Set GOOGLE_API_KEY env var or use --api-key")
        sys.exit(1)

    # Get PDF files
    pdf_dir = Path(args.pdf_dir)
    pdf_files = get_pdf_files(pdf_dir)

    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        logger.info("Please add PDF files to the data/pdfs/ directory")
        sys.exit(1)

    # Apply limit if specified
    if args.limit:
        pdf_files = pdf_files[: args.limit]
        logger.info(f"Limited to {len(pdf_files)} files")

    # Run extraction
    if args.all:
        run_all_approaches(args.api_key, pdf_files)
    elif args.approach == 1:
        run_approach_1(args.api_key, pdf_files)
    elif args.approach == 2:
        run_approach_2(args.api_key, pdf_files)
    elif args.approach == 3:
        run_approach_3(args.api_key, pdf_files)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
