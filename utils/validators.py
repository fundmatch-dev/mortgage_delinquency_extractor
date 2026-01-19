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

    def __init__(self):
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def reset(self):
        self.warnings = []
        self.errors = []

    def validate_document(self, data: dict, filename: str) -> tuple[bool, ExtractedDocument | None]:
        self.reset()
        try:
            data["filename"] = filename
            doc = ExtractedDocument(**data)
            self._check_balance_relationship(doc)
            self._check_delinquency_totals(doc)
            self._check_required_fields(doc)
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

    def get_validation_summary(self) -> dict:
        return {
            "has_errors": len(self.errors) > 0,
            "has_warnings": len(self.warnings) > 0,
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
        }
