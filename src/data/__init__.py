"""Data simulation, ingestion, validation, feature engineering, and repository package."""

from src.data.features import FeatureEngineer
from src.data.generator import LocationProfile, ProductProfile, SyntheticDataGenerator
from src.data.repository import DataRepository
from src.data.split import SplitVerificationResult, TimeSeriesSplitter
from src.data.validation import DataValidator, TransactionSchema, ValidationResult

__all__ = [
    "SyntheticDataGenerator",
    "ProductProfile",
    "LocationProfile",
    "DataValidator",
    "ValidationResult",
    "TransactionSchema",
    "FeatureEngineer",
    "TimeSeriesSplitter",
    "SplitVerificationResult",
    "DataRepository",
]
