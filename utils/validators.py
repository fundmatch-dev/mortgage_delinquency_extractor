"""Data validation functions for extracted mortgage data."""

import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class DelinquencyCategory(BaseModel):
    """Model for a single delinquency category."""

    category: str = Field(..., description="Delinquency category name")
    count: int = Field(..., ge=0, description="Number of loans in this category")
    balance: float = Field(..., ge=0, description="Dollar balance for this category")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Category name cannot be empty")
        return v.strip()


class ExtractedDocument(BaseModel):
    """Model for all extracted data from a single document."""

    filename: str = Field(..., description="PDF filename")
    series_name: Optional[str] = Field(None, description="Series identifier")
    report_date: Optional[str] = Field(None, description="Date of the document")
    beginning_balance: Optional[float] = Field(None, ge=0, description="Beginning balance")
    ending_balance: Optional[float] = Field(None, ge=0, description="Ending balance")
    delinquency: list[DelinquencyCategory] = Field(
        default_factory=list,
        description="List of delinquency categories"
    )

    @field_validator("report_date")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        date_formats = ["%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y", "%d/%m/%Y"]
        for fmt in date_formats:
            try:
                parsed = datetime.strptime(v.strip(), fmt)
                return parsed.strftime("%m/%d/%Y")
            except ValueError:
                continue
        return v.strip()

    @field_validator("series_name")
    @classmethod
    def validate_series_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return v.strip()


class DataValidator:
    """Validator class for extracted mortgage data."""

    # Tolerance for balance sum check (10% difference allowed)
    BALANCE_TOLERANCE = 0.10
    MIN_DELINQUENCY_CATEGORIES = 2

    def __init__(self):
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.checks_passed: list[str] = []
        self.checks_failed: list[str] = []

    def reset(self):
        self.warnings = []
        self.errors = []
        self.checks_passed = []
        self.checks_failed = []

    def validate_document(self, data: dict, filename: str) -> tuple[bool, ExtractedDocument | None]:
        self.reset()
        try:
            data["filename"] = filename
            doc = ExtractedDocument(**data)
            self._check_balance_relationship(doc)
            self._check_delinquency_totals(doc)
            self._check_required_fields(doc)
            self._check_balance_sum_sanity(doc)  # Key sanity check
            self._check_minimum_categories(doc)   # Key sanity check
            return len(self.errors) == 0, doc
        except Exception as e:
            self.errors.append(f"Validation failed: {str(e)}")
            return False, None

    def _check_balance_relationship(self, doc: ExtractedDocument):
        if doc.beginning_balance is not None and doc.ending_balance is not None:
            if doc.beginning_balance < doc.ending_balance:
                self.warnings.append(
                    f"Beginning balance ({doc.beginning_balance:,.2f}) < ending balance ({doc.ending_balance:,.2f})"
                )

    def _check_delinquency_totals(self, doc: ExtractedDocument):
        if not doc.delinquency:
            self.warnings.append("No delinquency categories found")
            return
        total_balance = sum(d.balance for d in doc.delinquency)
        total_count = sum(d.count for d in doc.delinquency)
        if total_count == 0:
            self.warnings.append("Total loan count is 0")
        if doc.ending_balance and total_balance > doc.ending_balance * 1.1:
            self.warnings.append(f"Delinquency total ({total_balance:,.2f}) exceeds ending balance")

    def _check_required_fields(self, doc: ExtractedDocument):
        if doc.series_name is None:
            self.warnings.append("Series name missing")
        if doc.report_date is None:
            self.warnings.append("Report date missing")
        if doc.beginning_balance is None:
            self.warnings.append("Beginning balance missing")
        if doc.ending_balance is None:
            self.warnings.append("Ending balance missing")

    def _check_balance_sum_sanity(self, doc: ExtractedDocument):
        """
        SANITY CHECK: Sum of delinquency balances should approximately equal ending balance.

        This validates that we extracted all categories and assigned correct values.
        Allows 10% tolerance for rounding differences.
        """
        if not doc.delinquency or doc.ending_balance is None:
            return

        delinquency_sum = sum(d.balance for d in doc.delinquency)

        if doc.ending_balance == 0:
            return

        # Calculate percentage difference
        pct_diff = abs(delinquency_sum - doc.ending_balance) / doc.ending_balance

        if pct_diff <= self.BALANCE_TOLERANCE:
            self.checks_passed.append(
                f"BALANCE SUM CHECK PASSED: Delinquency sum (${delinquency_sum:,.2f}) "
                f"is within {self.BALANCE_TOLERANCE*100:.0f}% of ending balance (${doc.ending_balance:,.2f})"
            )
        else:
            self.checks_failed.append(
                f"BALANCE SUM CHECK FAILED: Delinquency sum (${delinquency_sum:,.2f}) "
                f"differs by {pct_diff*100:.1f}% from ending balance (${doc.ending_balance:,.2f})"
            )
            self.warnings.append(f"Balance sum mismatch: {pct_diff*100:.1f}% difference")

    def _check_minimum_categories(self, doc: ExtractedDocument):
        """
        SANITY CHECK: Should have at least N delinquency categories.

        MBS reports typically have Current + multiple delinquency buckets.
        Having too few categories suggests incomplete extraction.
        """
        num_categories = len(doc.delinquency)

        if num_categories >= self.MIN_DELINQUENCY_CATEGORIES:
            self.checks_passed.append(
                f"CATEGORY COUNT CHECK PASSED: Found {num_categories} categories "
                f"(minimum: {self.MIN_DELINQUENCY_CATEGORIES})"
            )
        else:
            self.checks_failed.append(
                f"CATEGORY COUNT CHECK FAILED: Only {num_categories} categories found "
                f"(expected at least {self.MIN_DELINQUENCY_CATEGORIES})"
            )
            self.warnings.append(f"Only {num_categories} delinquency categories extracted")

    def get_confidence_score(self) -> int:
        """
        Calculate a simple confidence score (0-100) based on validation checks.

        Scoring:
        - Starts at 100
        - -20 for each failed sanity check
        - -10 for each warning
        - -30 for each error
        """
        score = 100
        score -= len(self.checks_failed) * 20
        score -= len(self.warnings) * 10
        score -= len(self.errors) * 30
        return max(0, min(100, score))

    def get_validation_summary(self) -> dict:
        return {
            "has_errors": len(self.errors) > 0,
            "has_warnings": len(self.warnings) > 0,
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
            "checks_passed": self.checks_passed.copy(),
            "checks_failed": self.checks_failed.copy(),
            "confidence_score": self.get_confidence_score(),
        }
