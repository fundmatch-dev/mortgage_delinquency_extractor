# Mortgage-Backed Securities PDF Data Extraction

Extract structured data from MBS trustee report PDFs using three different approaches powered by Google Gemini AI.

## Overview

This project extracts the following data from mortgage-backed securities trustee reports:

- **Metadata**: filename, series name, report date
- **Balance Information**: beginning balance, ending balance
- **Delinquency Breakdown**: category, loan count, balance for each delinquency status

Output is a single Excel file with all data in long format (one row per delinquency category per document).

## Project Structure

```
mortgage-extraction/
├── main.py                      # Main extraction script (all 3 approaches)
├── analysis.ipynb               # Jupyter notebook for comparison experiments
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── utils/
│   ├── __init__.py
│   ├── gemini_vision.py         # Approach 1: Gemini Vision extraction
│   ├── pdf_text_extraction.py   # Approach 2: PDF text + Gemini Pro
│   ├── validators.py            # Data validation with Pydantic
│   └── output_formatter.py      # Excel formatting utilities
└── data/
    ├── pdfs/                    # Place PDF files here
    └── output/                  # Generated Excel files and results
```

## Setup & Installation

### 1. Prerequisites

- Python 3.10 or higher
- Google AI API key (for Gemini access)

### 2. Install Dependencies

```bash
cd mortgage-extraction
pip install -r requirements.txt
```

### 3. Set API Key

Option A: Environment variable
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Option B: Create `.env` file
```
GOOGLE_API_KEY=your-api-key-here
```

### 4. Add PDF Files

Place your PDF trustee reports in the `data/pdfs/` directory.

## Usage

### Run Extraction

```bash
# Run Approach 1: Gemini Vision Direct Extraction
python main.py --approach 1

# Run Approach 2: PDF Text Parsing + Gemini Pro
python main.py --approach 2

# Run Approach 3: Hybrid Smart Extraction
python main.py --approach 3

# Run all approaches for comparison
python main.py --all

# Limit to first N files (for testing)
python main.py --approach 1 --limit 5

# Specify custom PDF directory
python main.py --approach 1 --pdf-dir /path/to/pdfs
```

### Run Comparison Analysis

Open and run the Jupyter notebook:

```bash
jupyter notebook analysis.ipynb
```

The notebook will:
1. Run all three approaches on a sample of 5-10 PDFs
2. Calculate accuracy, speed, and cost metrics
3. Generate comparison visualizations
4. Provide a recommendation on which approach to use

## Extraction Approaches

### Approach 1: Gemini Vision Direct Extraction

- **Method**: Sends entire PDF directly to Gemini Vision API
- **Model**: `gemini-2.0-flash-exp`
- **Pros**: Handles complex tables, works with scanned PDFs
- **Cons**: Higher cost, slightly slower
- **Best for**: Documents with complex layouts or embedded images

### Approach 2: PDF Text Parsing + Gemini Pro

- **Method**: Extract text with pdfplumber, then send to Gemini Pro
- **Model**: `gemini-1.5-pro`
- **Pros**: Lower cost, faster processing
- **Cons**: May lose table structure in text extraction
- **Best for**: Clean, text-based PDFs with simple layouts

### Approach 3: Hybrid Smart Extraction

- **Method**: Assess text quality first, choose optimal method per document
- **Strategy**:
  1. Extract text with pdfplumber
  2. Assess quality (table detection, keyword presence)
  3. If quality is good: use text + Gemini Flash (faster/cheaper)
  4. If quality is poor: fall back to Gemini Vision
- **Pros**: Optimizes cost while maintaining accuracy
- **Best for**: Mixed document quality in batch processing

## Output Format

### Excel File Structure

Single sheet with columns:
| Column | Description |
|--------|-------------|
| filename | PDF filename (e.g., "1.pdf") |
| series_name | Series identifier (e.g., "Series 2025-2") |
| report_date | Document date (MM/DD/YYYY) |
| beginning_balance | Total beginning balance ($) |
| ending_balance | Total ending balance ($) |
| delinquency_category | Category name (e.g., "Current", "30-59 days") |
| loan_count | Number of loans in category |
| balance | Dollar balance for category |

Each document produces multiple rows (one per delinquency category).

## Validation Checks

The extractor performs these validation checks:
- Beginning balance >= Ending balance (warning if not)
- Delinquency balances sum reasonably
- Dates are valid and properly formatted
- All required fields are present
- Loan counts are positive integers
- Balances are positive numbers

Validation warnings are logged but don't block extraction.

## Comparison Results

*To be updated after running analysis.ipynb*

### Summary

| Approach | Success Rate | Avg Time | Est. Cost/Doc |
|----------|-------------|----------|---------------|
| Approach 1 (Vision) | TBD | TBD | TBD |
| Approach 2 (Text+Pro) | TBD | TBD | TBD |
| Approach 3 (Hybrid) | TBD | TBD | TBD |

### Recommendation

*To be determined after running comparison on your specific PDFs.*

## Troubleshooting

### Common Issues

**API Rate Limiting**
- The extractor includes 1-second delays between requests
- If you hit rate limits, increase `min_request_interval` in the extractor classes

**Text Extraction Issues**
- If pdfplumber fails, try `--text-method pymupdf` or `pypdf`
- Scanned PDFs require Approach 1 (Vision)

**JSON Parsing Errors**
- Usually indicates the model returned unexpected format
- Check logs for the raw response
- May need to adjust the extraction prompt

**Missing Delinquency Data**
- Check if the document actually contains delinquency tables
- Try Approach 1 (Vision) which handles tables better

### Logging

Extraction logs are saved to `extraction.log` in the project directory.

## Development

### Adding New Fields

1. Update `EXTRACTION_PROMPT` in both extractor files
2. Add field to `ExtractedDocument` model in `validators.py`
3. Update `ExcelFormatter.COLUMNS` in `output_formatter.py`

### Modifying Validation Rules

Edit `DataValidator` class in `utils/validators.py`.

## License

This project is for internal use. See your organization's data handling policies for PDF document processing.
# mortgage_delinquency_extractor
