I'll provide a complete updated version with all the fixes integrated. Here's the enhanced code with the missing implementations and fixes:

```python
# elite_financial_analytics_platform_v3.py
# Enterprise-Grade Financial Analytics Platform - Complete Fixed Version

# --- 1. Core Imports and Setup ---
import asyncio
import concurrent.futures
import functools
import hashlib
import io
import json
import logging
import os
import pickle
import re
import sys
import threading
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Set, TypeVar, Generic, 
    Callable, Protocol, Type, cast, overload
)
from weakref import WeakValueDictionary

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

import bleach
from fuzzywuzzy import fuzz

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

# Configure logging with rotation
from logging.handlers import RotatingFileHandler

# Set up warnings
warnings.filterwarnings('ignore')

# --- Import Core Components (with fallback) ---
try:
    from financial_analytics_core import (
        ChartGenerator as CoreChartGenerator,
        FinancialRatioCalculator as CoreRatioCalculator,
        PenmanNissimAnalyzer as CorePenmanNissim,
        IndustryBenchmarks as CoreIndustryBenchmarks,
        DataProcessor as CoreDataProcessor,
        DataQualityMetrics,
        parse_html_content,
        parse_csv_content,
        process_and_merge_dataframes,
        REQUIRED_METRICS as CORE_REQUIRED_METRICS,
        YEAR_REGEX as CORE_YEAR_REGEX,
        MAX_FILE_SIZE_MB as CORE_MAX_FILE_SIZE,
        ALLOWED_FILE_TYPES as CORE_ALLOWED_TYPES,
        EPS
    )
    CORE_COMPONENTS_AVAILABLE = True
except ImportError:
    CORE_COMPONENTS_AVAILABLE = False
    # Define fallbacks
    CORE_REQUIRED_METRICS = {}
    CORE_YEAR_REGEX = re.compile(r'(20\d{2}|19\d{2}|FY\s?20\d{2}|FY\s?19\d{2})')
    CORE_MAX_FILE_SIZE = 10
    CORE_ALLOWED_TYPES = ['csv', 'html', 'htm', 'xls', 'xlsx']

# --- 2. Advanced Logging Configuration ---
class LoggerFactory:
    """Factory for creating configured loggers with context"""
    
    _loggers = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_logger(cls, name: str, level: int = logging.INFO) -> logging.Logger:
        """Get or create a logger with proper configuration"""
        with cls._lock:
            if name not in cls._loggers:
                logger = logging.getLogger(name)
                logger.setLevel(level)
                
                # Console handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(level)
                
                # File handler with rotation
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                file_handler = RotatingFileHandler(
                    log_dir / f"{name}.log",
                    maxBytes=10485760,  # 10MB
                    backupCount=5
                )
                file_handler.setLevel(level)
                
                # Formatter
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                console_handler.setFormatter(formatter)
                file_handler.setFormatter(formatter)
                
                logger.addHandler(console_handler)
                logger.addHandler(file_handler)
                
                cls._loggers[name] = logger
            
            return cls._loggers[name]

# --- 3. Advanced Error Handling ---
class ErrorContext:
    """Context manager for error handling with recovery"""
    
    def __init__(self, operation: str, logger: logging.Logger, 
                 fallback: Optional[Callable] = None,
                 max_retries: int = 3):
        self.operation = operation
        self.logger = logger
        self.fallback = fallback
        self.max_retries = max_retries
        self.attempts = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.attempts += 1
            self.logger.error(
                f"Error in {self.operation} (attempt {self.attempts}/{self.max_retries}): "
                f"{exc_type.__name__}: {exc_val}"
            )
            
            if self.attempts < self.max_retries:
                self.logger.info(f"Retrying {self.operation}...")
                return True  # Suppress exception for retry
            
            if self.fallback:
                self.logger.info(f"Executing fallback for {self.operation}")
                try:
                    self.fallback()
                except Exception as fallback_error:
                    self.logger.error(f"Fallback failed: {fallback_error}")
            
            # Log full traceback for debugging
            self.logger.debug(f"Full traceback:\n{''.join(traceback.format_tb(exc_tb))}")
            
        return False

# --- 4. Advanced Configuration Management ---
class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass

class Configuration:
    """Centralized configuration with validation and type safety"""
    
    # Type definitions
    class DisplayMode(Enum):
        FULL = auto()
        LITE = auto()
        MINIMAL = auto()
    
    class NumberFormat(Enum):
        INDIAN = auto()
        INTERNATIONAL = auto()
    
    # Default configurations
    DEFAULTS = {
        'app': {
            'version': '3.0.0',
            'name': 'Elite Financial Analytics Platform',
            'debug': False,
            'display_mode': DisplayMode.LITE,
            'max_file_size_mb': 10,
            'allowed_file_types': ['csv', 'html', 'htm', 'xls', 'xlsx'],
            'cache_ttl_seconds': 3600,
            'max_cache_size_mb': 100,
        },
        'processing': {
            'max_workers': 4,
            'chunk_size': 1000,
            'timeout_seconds': 30,
            'memory_limit_mb': 512,
            'enable_parallel': True,
        },
        'analysis': {
            'confidence_threshold': 0.6,
            'outlier_std_threshold': 3,
            'min_data_points': 3,
            'interpolation_method': 'linear',
            'number_format': NumberFormat.INDIAN,
        },
        'ai': {
            'enabled': True,
            'model_name': 'all-MiniLM-L6-v2',
            'batch_size': 32,
            'max_sequence_length': 512,
            'similarity_threshold': 0.6,
        },
        'ui': {
            'theme': 'light',
            'animations': True,
            'auto_save': True,
            'auto_save_interval': 60,
        }
    }
    
    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        self._config = self._deep_merge(self.DEFAULTS.copy(), custom_config or {})
        self._validate_config()
        self._logger = LoggerFactory.get_logger('Configuration')
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def _validate_config(self):
        """Validate configuration values"""
        # App validation
        if self._config['app']['max_file_size_mb'] <= 0:
            raise ConfigurationError("max_file_size_mb must be positive")
        
        if not self._config['app']['allowed_file_types']:
            raise ConfigurationError("allowed_file_types cannot be empty")
        
        # Processing validation
        if self._config['processing']['max_workers'] <= 0:
            raise ConfigurationError("max_workers must be positive")
        
        if self._config['processing']['timeout_seconds'] <= 0:
            raise ConfigurationError("timeout_seconds must be positive")
        
        # Analysis validation
        if not 0 < self._config['analysis']['confidence_threshold'] <= 1:
            raise ConfigurationError("confidence_threshold must be between 0 and 1")
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path"""
        try:
            value = self._config
            for key in path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            self._logger.warning(f"Configuration key '{path}' not found, using default: {default}")
            return default
    
    def set(self, path: str, value: Any):
        """Set configuration value by dot-separated path"""
        keys = path.split('.')
        target = self._config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
        self._logger.info(f"Configuration updated: {path} = {value}")
    
    @contextmanager
    def override(self, **overrides):
        """Temporarily override configuration values"""
        original = {}
        
        try:
            # Save original values and apply overrides
            for path, value in overrides.items():
                original[path] = self.get(path)
                self.set(path, value)
            
            yield self
            
        finally:
            # Restore original values
            for path, value in original.items():
                self.set(path, value)

# --- 5. Advanced Caching System ---
class CacheEntry:
    """Cache entry with metadata"""
    
    def __init__(self, value: Any, ttl: Optional[int] = None):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = time.time()
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> Any:
        """Access the value and update metadata"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value

class AdvancedCache:
    """Thread-safe cache with TTL, size limits, and statistics"""
    
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._default_ttl = default_ttl
        self._stats = defaultdict(int)
        self._logger = LoggerFactory.get_logger('Cache')
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            return 1024  # Default estimate
    
    def _evict_if_needed(self):
        """Evict entries if cache is too large"""
        current_size = sum(self._estimate_size(entry.value) for entry in self._cache.values())
        
        if current_size > self._max_size_bytes:
            # Evict least recently used entries
            entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            while current_size > self._max_size_bytes * 0.8 and entries:
                key, entry = entries.pop(0)
                current_size -= self._estimate_size(entry.value)
                del self._cache[key]
                self._stats['evictions'] += 1
                self._logger.debug(f"Evicted cache entry: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            self._stats['get_calls'] += 1
            
            if key in self._cache:
                entry = self._cache[key]
                
                if entry.is_expired():
                    del self._cache[key]
                    self._stats['expired'] += 1
                    return None
                
                self._stats['hits'] += 1
                return entry.access()
            
            self._stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        with self._lock:
            self._stats['set_calls'] += 1
            
            # Create entry
            entry = CacheEntry(value, ttl or self._default_ttl)
            self._cache[key] = entry
            
            # Evict if needed
            self._evict_if_needed()
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats['deletes'] += 1
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._stats['clears'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_calls = self._stats['get_calls']
            hit_rate = (self._stats['hits'] / total_calls * 100) if total_calls > 0 else 0
            
            return {
                'entries': len(self._cache),
                'size_bytes': sum(self._estimate_size(e.value) for e in self._cache.values()),
                'hit_rate': hit_rate,
                **self._stats
            }

# --- 6. Resource Management ---
class ResourceManager:
    """Manage computational resources and prevent overload"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self._semaphore = threading.Semaphore(config.get('processing.max_workers', 4))
        self._memory_limit = config.get('processing.memory_limit_mb', 512) * 1024 * 1024
        self._active_tasks = set()
        self._lock = threading.Lock()
        self._logger = LoggerFactory.get_logger('ResourceManager')
    
    @contextmanager
    def acquire_worker(self, task_name: str):
        """Acquire a worker slot for processing"""
        acquired = False
        start_time = time.time()
        
        try:
            # Try to acquire with timeout
            acquired = self._semaphore.acquire(
                timeout=self.config.get('processing.timeout_seconds', 30)
            )
            
            if not acquired:
                raise TimeoutError(f"Failed to acquire worker for {task_name}")
            
            with self._lock:
                self._active_tasks.add(task_name)
            
            self._logger.debug(f"Acquired worker for {task_name}")
            yield
            
        finally:
            if acquired:
                self._semaphore.release()
                
                with self._lock:
                    self._active_tasks.discard(task_name)
                
                elapsed = time.time() - start_time
                self._logger.debug(f"Released worker for {task_name} after {elapsed:.2f}s")
    
    def check_memory_available(self, estimated_size: int) -> bool:
        """Check if enough memory is available"""
        try:
            import psutil
            available = psutil.virtual_memory().available
            return available > estimated_size + self._memory_limit
        except ImportError:
            # Fallback if psutil not available
            return True
    
    def get_active_tasks(self) -> List[str]:
        """Get list of active tasks"""
        with self._lock:
            return list(self._active_tasks)

# --- 7. Advanced Data Validation ---
class ValidationResult:
    """Result of validation with detailed information"""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_error(self, message: str):
        """Add error message"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add warning message"""
        self.warnings.append(message)
    
    def add_info(self, message: str):
        """Add info message"""
        self.info.append(message)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result"""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        self.metadata.update(other.metadata)

class DataValidator:
    """Advanced data validation with comprehensive checks"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self._logger = LoggerFactory.get_logger('DataValidator')
    
    def validate_dataframe(self, df: pd.DataFrame, context: str = "data") -> ValidationResult:
        """Comprehensive dataframe validation"""
        result = ValidationResult()
        
        # Basic structure checks
        if df.empty:
            result.add_error(f"{context}: DataFrame is empty")
            return result
        
        if len(df.columns) == 0:
            result.add_error(f"{context}: No columns found")
            return result
        
        # Size checks
        if df.shape[0] > 1000000:
            result.add_warning(f"{context}: Large dataset ({df.shape[0]} rows), performance may be impacted")
        
        # Data type checks
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            result.add_warning(f"{context}: No numeric columns found")
        
        # Check for duplicates
        if df.index.duplicated().any():
            result.add_warning(f"{context}: Duplicate indices found")
        
        # Check for missing values
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 50:
            result.add_warning(f"{context}: High percentage of missing values ({missing_pct:.1f}%)")
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                result.add_info(f"{context}: Column '{col}' has constant value")
        
        # Statistical checks for numeric columns
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                # Check for outliers
                mean = series.mean()
                std = series.std()
                outlier_threshold = self.config.get('analysis.outlier_std_threshold', 3)
                
                outliers = series[(series < mean - outlier_threshold * std) | 
                                 (series > mean + outlier_threshold * std)]
                
                if len(outliers) > len(series) * 0.1:
                    result.add_warning(
                        f"{context}: Column '{col}' has many outliers ({len(outliers)} values)"
                    )
                
                # Check for negative values in typically positive metrics
                positive_keywords = ['assets', 'revenue', 'sales', 'income', 'cash']
                if any(keyword in str(col).lower() for keyword in positive_keywords):
                    if (series < 0).any():
                        result.add_warning(f"{context}: Column '{col}' contains negative values")
        
        # Add metadata
        result.metadata['shape'] = df.shape
        result.metadata['columns'] = list(df.columns)
        result.metadata['dtypes'] = df.dtypes.to_dict()
        result.metadata['memory_usage'] = df.memory_usage(deep=True).sum()
        
        return result
    
    def validate_financial_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate financial statement data"""
        result = self.validate_dataframe(df, "financial_data")
        
        # Additional financial-specific checks
        required_keywords = ['asset', 'liability', 'equity', 'revenue', 'expense']
        found_keywords = []
        
        for keyword in required_keywords:
            if any(keyword in str(idx).lower() for idx in df.index):
                found_keywords.append(keyword)
        
        if len(found_keywords) < 2:
            result.add_warning(
                "Limited financial keywords found in data. "
                "Please verify the data contains financial statements."
            )
        
        # Check accounting equation (Assets = Liabilities + Equity)
        asset_rows = [idx for idx in df.index if 'total asset' in str(idx).lower()]
        liability_rows = [idx for idx in df.index if 'total liabilit' in str(idx).lower()]
        equity_rows = [idx for idx in df.index if 'total equity' in str(idx).lower()]
        
        if asset_rows and liability_rows and equity_rows:
            for col in df.select_dtypes(include=[np.number]).columns:
                try:
                    assets = df.loc[asset_rows[0], col]
                    liabilities = df.loc[liability_rows[0], col]
                    equity = df.loc[equity_rows[0], col]
                    
                    if not pd.isna(assets) and not pd.isna(liabilities) and not pd.isna(equity):
                        diff = abs(assets - (liabilities + equity))
                        tolerance = assets * 0.01  # 1% tolerance
                        
                        if diff > tolerance:
                            result.add_warning(
                                f"Accounting equation imbalance in {col}: "
                                f"Assets ({assets:,.0f}) ≠ Liabilities ({liabilities:,.0f}) "
                                f"+ Equity ({equity:,.0f})"
                            )
                except Exception:
                    pass
        
        return result

# --- 8. Advanced Pattern Matching System ---
class PatternMatcher:
    """Advanced pattern matching for financial metrics"""
    
    def __init__(self):
        self._patterns = self._build_patterns()
        self._logger = LoggerFactory.get_logger('PatternMatcher')
    
    def _build_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Build comprehensive pattern library"""
        return {
            # Assets
            'total_assets': [
                re.compile(r'\btotal\s+assets?\b', re.IGNORECASE),
                re.compile(r'\bassets?\s+total\b', re.IGNORECASE),
                re.compile(r'\b(?:sum|total)\s+of\s+assets?\b', re.IGNORECASE),
            ],
            'current_assets': [
                re.compile(r'\bcurrent\s+assets?\b', re.IGNORECASE),
                re.compile(r'\bshort[\s-]?term\s+assets?\b', re.IGNORECASE),
                re.compile(r'\bliquid\s+assets?\b', re.IGNORECASE),
            ],
            'non_current_assets': [
                re.compile(r'\bnon[\s-]?current\s+assets?\b', re.IGNORECASE),
                re.compile(r'\blong[\s-]?term\s+assets?\b', re.IGNORECASE),
                re.compile(r'\bfixed\s+assets?\b', re.IGNORECASE),
            ],
            'cash': [
                re.compile(r'\bcash\b', re.IGNORECASE),
                re.compile(r'\bcash\s+(?:and|&)\s+cash\s+equivalents?\b', re.IGNORECASE),
                re.compile(r'\bcash\s+(?:and|&)\s+bank\b', re.IGNORECASE),
                re.compile(r'\bliquid\s+funds?\b', re.IGNORECASE),
            ],
            'inventory': [
                re.compile(r'\binventor(?:y|ies)\b', re.IGNORECASE),
                re.compile(r'\bstock\s+in\s+trade\b', re.IGNORECASE),
                re.compile(r'\bgoods?\b', re.IGNORECASE),
                re.compile(r'\bmaterials?\b', re.IGNORECASE),
            ],
            'receivables': [
                re.compile(r'\breceivables?\b', re.IGNORECASE),
                re.compile(r'\baccounts?\s+receivables?\b', re.IGNORECASE),
                re.compile(r'\btrade\s+receivables?\b', re.IGNORECASE),
                re.compile(r'\bdebtors?\b', re.IGNORECASE),
            ],
            
            # Liabilities
            'total_liabilities': [
                re.compile(r'\btotal\s+liabilit(?:y|ies)\b', re.IGNORECASE),
                re.compile(r'\bliabilit(?:y|ies)\s+total\b', re.IGNORECASE),
                re.compile(r'\b(?:sum|total)\s+of\s+liabilit(?:y|ies)\b', re.IGNORECASE),
            ],
            'current_liabilities': [
                re.compile(r'\bcurrent\s+liabilit(?:y|ies)\b', re.IGNORECASE),
                re.compile(r'\bshort[\s-]?term\s+liabilit(?:y|ies)\b', re.IGNORECASE),
            ],
            'non_current_liabilities': [
                re.compile(r'\bnon[\s-]?current\s+liabilit(?:y|ies)\b', re.IGNORECASE),
                re.compile(r'\blong[\s-]?term\s+liabilit(?:y|ies)\b', re.IGNORECASE),
            ],
            'debt': [
                re.compile(r'\bdebt\b', re.IGNORECASE),
                re.compile(r'\bborrowing\b', re.IGNORECASE),
                re.compile(r'\bloan\b', re.IGNORECASE),
                re.compile(r'\bdebenture\b', re.IGNORECASE),
            ],
            
            # Equity
            'total_equity': [
                re.compile(r'\btotal\s+equity\b', re.IGNORECASE),
                re.compile(r'\bshareholders?\s+equity\b', re.IGNORECASE),
                re.compile(r'\bstockholders?\s+equity\b', re.IGNORECASE),
                re.compile(r'\bnet\s+worth\b', re.IGNORECASE),
            ],
            'share_capital': [
                re.compile(r'\bshare\s+capital\b', re.IGNORECASE),
                re.compile(r'\bcapital\s+stock\b', re.IGNORECASE),
                re.compile(r'\bpaid[\s-]?up\s+capital\b', re.IGNORECASE),
                re.compile(r'\bequity\s+shares?\b', re.IGNORECASE),
            ],
            'retained_earnings': [
                re.compile(r'\bretained\s+earnings?\b', re.IGNORECASE),
                re.compile(r'\breserves?\s+(?:and|&)\s+surplus\b', re.IGNORECASE),
                re.compile(r'\baccumulated\s+profits?\b', re.IGNORECASE),
            ],
            
            # Income Statement
            'revenue': [
                re.compile(r'\brevenue\b', re.IGNORECASE),
                re.compile(r'\bsales?\b', re.IGNORECASE),
                re.compile(r'\bturnover\b', re.IGNORECASE),
                re.compile(r'\bincome\s+from\s+operations?\b', re.IGNORECASE),
                re.compile(r'\bnet\s+sales?\b', re.IGNORECASE),
            ],
            'cost_of_goods_sold': [
                re.compile(r'\bcost\s+of\s+goods?\s+sold\b', re.IGNORECASE),
                re.compile(r'\bcogs\b', re.IGNORECASE),
                re.compile(r'\bcost\s+of\s+sales?\b', re.IGNORECASE),
                re.compile(r'\bcost\s+of\s+revenue\b', re.IGNORECASE),
            ],
            'operating_expenses': [
                re.compile(r'\boperating\s+expenses?\b', re.IGNORECASE),
                re.compile(r'\bopex\b', re.IGNORECASE),
                re.compile(r'\badministrative\s+expenses?\b', re.IGNORECASE),
                re.compile(r'\bselling\s+(?:and|&)\s+distribution\b', re.IGNORECASE),
            ],
            'ebit': [
                re.compile(r'\bebit\b', re.IGNORECASE),
                re.compile(r'\bearnings?\s+before\s+interest\s+(?:and|&)\s+tax\b', re.IGNORECASE),
                re.compile(r'\boperating\s+(?:profit|income)\b', re.IGNORECASE),
            ],
            'net_income': [
                re.compile(r'\bnet\s+(?:income|profit|earnings?)\b', re.IGNORECASE),
                re.compile(r'\bprofit\s+after\s+tax\b', re.IGNORECASE),
                re.compile(r'\bpat\b', re.IGNORECASE),
                re.compile(r'\bprofit\s+for\s+the\s+(?:year|period)\b', re.IGNORECASE),
            ],
        }
    
    def find_matches(self, text: str, metric_type: str) -> List[Tuple[str, float]]:
        """Find pattern matches with confidence scores"""
        matches = []
        
        if metric_type not in self._patterns:
            return matches
        
        for pattern in self._patterns[metric_type]:
            if pattern.search(text):
                # Calculate confidence based on match quality
                match = pattern.search(text)
                confidence = self._calculate_confidence(text, match)
                matches.append((metric_type, confidence))
        
        return matches
    
    def _calculate_confidence(self, text: str, match: re.Match) -> float:
        """Calculate confidence score for a match"""
        # Base confidence
        confidence = 0.7
        
        # Exact match bonus
        if match.group(0).lower() == text.lower():
            confidence += 0.2
        
        # Position bonus (earlier matches are better)
        position_ratio = match.start() / len(text)
        confidence += (1 - position_ratio) * 0.1
        
        return min(confidence, 1.0)
    
    def classify_metric(self, text: str) -> Dict[str, float]:
        """Classify a metric into categories with confidence scores"""
        classifications = defaultdict(float)
        
        for metric_type in self._patterns:
            matches = self.find_matches(text, metric_type)
            for _, confidence in matches:
                classifications[metric_type] = max(
                    classifications[metric_type], 
                    confidence
                )
        
        return dict(classifications)

# --- 9. State Management System ---
class StateManager:
    """Advanced state management with persistence and recovery"""
    
    def __init__(self, namespace: str = "app"):
        self.namespace = namespace
        self._state = {}
        self._observers = defaultdict(list)
        self._lock = threading.RLock()
        self._logger = LoggerFactory.get_logger('StateManager')
        self._persistence_path = Path(f".state/{namespace}")
        self._persistence_path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get state value"""
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any, persist: bool = False):
        """Set state value"""
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value
            
            # Notify observers
            for observer in self._observers[key]:
                try:
                    observer(key, old_value, value)
                except Exception as e:
                    self._logger.error(f"Observer error for {key}: {e}")
            
            # Persist if requested
            if persist:
                self._persist_key(key, value)
    
    def update(self, updates: Dict[str, Any], persist: bool = False):
        """Batch update state"""
        with self._lock:
            for key, value in updates.items():
                self.set(key, value, persist)
    
    def observe(self, key: str, callback: Callable[[str, Any, Any], None]):
        """Add observer for state changes"""
        with self._lock:
            self._observers[key].append(callback)
    
    def _persist_key(self, key: str, value: Any):
        """Persist a key to disk"""
        try:
            filepath = self._persistence_path / f"{key}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            self._logger.error(f"Failed to persist {key}: {e}")
    
    def load_persisted(self):
        """Load all persisted state"""
        for filepath in self._persistence_path.glob("*.pkl"):
            key = filepath.stem
            try:
                with open(filepath, 'rb') as f:
                    value = pickle.load(f)
                    self._state[key] = value
                    self._logger.info(f"Loaded persisted state: {key}")
            except Exception as e:
                self._logger.error(f"Failed to load {filepath}: {e}")
    
    def create_checkpoint(self, name: str):
        """Create a state checkpoint"""
        checkpoint_path = self._persistence_path / f"checkpoint_{name}.pkl"
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(dict(self._state), f)
            self._logger.info(f"Created checkpoint: {name}")
        except Exception as e:
            self._logger.error(f"Failed to create checkpoint {name}: {e}")
    
    def restore_checkpoint(self, name: str) -> bool:
        """Restore from checkpoint"""
        checkpoint_path = self._persistence_path / f"checkpoint_{name}.pkl"
        try:
            with open(checkpoint_path, 'rb') as f:
                self._state = pickle.load(f)
            self._logger.info(f"Restored checkpoint: {name}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to restore checkpoint {name}: {e}")
            return False

# --- 10. Base Components with Dependency Injection ---
class Component(ABC):
    """Base component with lifecycle management"""
    
    def __init__(self, config: Configuration, state: StateManager):
        self.config = config
        self.state = state
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._initialized = False
    
    def initialize(self):
        """Initialize component"""
        if not self._initialized:
            self._logger.info(f"Initializing {self.__class__.__name__}")
            self._do_initialize()
            self._initialized = True
    
    @abstractmethod
    def _do_initialize(self):
        """Actual initialization logic"""
        pass
    
    def cleanup(self):
        """Cleanup component"""
        if self._initialized:
            self._logger.info(f"Cleaning up {self.__class__.__name__}")
            self._do_cleanup()
            self._initialized = False
    
    def _do_cleanup(self):
        """Actual cleanup logic"""
        pass

# --- 11. Enhanced Security Module ---
class SecurityModule(Component):
    """Enhanced security with comprehensive validation"""
    
    def __init__(self, config: Configuration, state: StateManager):
        super().__init__(config, state)
        self._sanitizer = None
        self._rate_limiter = defaultdict(deque)
        self._blocked_ips = set()
    
    def _do_initialize(self):
        """Initialize security components"""
        # Initialize HTML sanitizer
        self._allowed_tags = ['table', 'tr', 'td', 'th', 'tbody', 'thead', 'p', 'div', 'span', 'br']
        self._allowed_attributes = {
            '*': ['class', 'id'],
            'table': ['border', 'cellpadding', 'cellspacing'],
        }
    
    def validate_file_upload(self, file: UploadedFile) -> ValidationResult:
        """Comprehensive file validation"""
        result = ValidationResult()
        
        # Check file size
        max_size = self.config.get('app.max_file_size_mb', 10) * 1024 * 1024
        if file.size > max_size:
            result.add_error(f"File size ({file.size / 1024 / 1024:.1f}MB) exceeds limit ({max_size / 1024 / 1024}MB)")
            return result
        
        # Check file extension
        allowed_types = self.config.get('app.allowed_file_types', [])
        file_ext = Path(file.name).suffix.lower().lstrip('.')
        
        if file_ext not in allowed_types:
            result.add_error(f"File type '{file_ext}' not allowed. Allowed types: {', '.join(allowed_types)}")
            return result
        
        # Check file name for suspicious patterns
        suspicious_patterns = [
            r'\.\./', r'\.\.\\',  # Path traversal
            r'[<>:"|?*]',  # Invalid characters
            r'^\.',  # Hidden files
            r'\.(exe|bat|cmd|sh|ps1)$',  # Executable extensions
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, file.name, re.IGNORECASE):
                result.add_error(f"Suspicious file name pattern detected")
                return result
        
        # Content validation for HTML/XML files
        if file_ext in ['html', 'htm', 'xml']:
            content = self._read_file_safely(file)
            if content:
                validation = self._validate_html_content(content)
                result.merge(validation)
        
        return result
    
    def _read_file_safely(self, file: UploadedFile, max_bytes: int = 1024 * 1024) -> Optional[str]:
        """Safely read file content with size limit"""
        try:
            content = file.read(max_bytes)
            file.seek(0)  # Reset file pointer
            
            # Try to decode
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            return None
        except Exception as e:
            self._logger.error(f"Error reading file: {e}")
            return None
    
    def _validate_html_content(self, content: str) -> ValidationResult:
        """Validate HTML content for security issues"""
        result = ValidationResult()
        
        # Check for malicious patterns
        malicious_patterns = [
            (r'<script', 'JavaScript code detected'),
            (r'javascript:', 'JavaScript protocol detected'),
            (r'on\w+\s*=', 'Event handler detected'),
            (r'<iframe', 'IFrame detected'),
            (r'<object', 'Object tag detected'),
            (r'<embed', 'Embed tag detected'),
            (r'<link[^>]+href', 'External link detected'),
            (r'@import', 'CSS import detected'),
            (r'expression\s*\(', 'CSS expression detected'),
            (r'vbscript:', 'VBScript protocol detected'),
        ]
        
        content_lower = content.lower()
        for pattern, message in malicious_patterns:
            if re.search(pattern, content_lower):
                result.add_error(f"Security issue: {message}")
        
        # Check content size
        if len(content) > 10 * 1024 * 1024:  # 10MB
            result.add_warning("Large HTML content may impact performance")
        
        return result
    
    def sanitize_html(self, content: str) -> str:
        """Sanitize HTML content"""
        return bleach.clean(
            content,
            tags=self._allowed_tags,
            attributes=self._allowed_attributes,
            strip=True,
            strip_comments=True
        )
    
    def check_rate_limit(self, identifier: str, action: str, limit: int = 100, window: int = 60) -> bool:
        """Check rate limit for an action"""
        key = f"{identifier}:{action}"
        now = time.time()
        
        # Clean old entries
        self._rate_limiter[key] = deque(
            [t for t in self._rate_limiter[key] if now - t < window],
            maxlen=limit
        )
        
        # Check limit
        if len(self._rate_limiter[key]) >= limit:
            self._logger.warning(f"Rate limit exceeded for {key}")
            return False
        
        # Add current request
        self._rate_limiter[key].append(now)
        return True

# --- 12. Enhanced Data Processing Pipeline ---
class DataProcessor(Component):
    """Advanced data processing with pipeline architecture"""
    
    def __init__(self, config: Configuration, state: StateManager):
        super().__init__(config, state)
        self._transformers = []
        self._validators = []
        self.resource_manager = None
    
    def _do_initialize(self):
        """Initialize processor"""
        self.resource_manager = ResourceManager(self.config)
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup processing pipeline"""
        # Add transformers
        self._transformers = [
            self._clean_numeric_data,
            self._normalize_indices,
            self._interpolate_missing,
            self._detect_outliers,
        ]
        
        # Add validators
        validator = DataValidator(self.config)
        self._validators = [
            validator.validate_dataframe,
            validator.validate_financial_data,
        ]
    
    def process(self, df: pd.DataFrame, context: str = "data") -> Tuple[pd.DataFrame, ValidationResult]:
        """Process dataframe through pipeline"""
        result = ValidationResult()
        processed_df = df.copy()
        
        with self.resource_manager.acquire_worker(f"process_{context}"):
            # Validation phase
            for validator in self._validators:
                validation = validator(processed_df)
                result.merge(validation)
                
                if not validation.is_valid:
                    self._logger.warning(f"Validation failed in {context}")
                    break
            
            # Transformation phase
            if result.is_valid:
                for transformer in self._transformers:
                    try:
                        processed_df = transformer(processed_df)
                    except Exception as e:
                        result.add_error(f"Transformation error: {str(e)}")
                        self._logger.error(f"Error in transformer {transformer.__name__}: {e}")
                        break
        
        return processed_df, result
    
    def _clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric data with advanced techniques"""
        df_clean = df.copy()
        
        for col in df.select_dtypes(include=['object']).columns:
            # Try to convert to numeric
            df_clean[col] = pd.to_numeric(df[col], errors='coerce')
            
            # If mostly non-numeric, keep as string
            if df_clean[col].isna().sum() > len(df) * 0.5:
                df_clean[col] = df[col]
        
        return df_clean
    
    def _normalize_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize index names"""
        df_norm = df.copy()
        
        # Clean index names
        if isinstance(df.index, pd.Index):
            df_norm.index = df.index.map(lambda x: str(x).strip())
        
        # Remove duplicate indices
        if df_norm.index.duplicated().any():
            df_norm = df_norm[~df_norm.index.duplicated(keep='first')]
            self._logger.warning("Removed duplicate indices")
        
        return df_norm
    
    def _interpolate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing values intelligently"""
        df_interp = df.copy()
        method = self.config.get('analysis.interpolation_method', 'linear')
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                # Only interpolate if enough data points
                non_na_count = df[col].notna().sum()
                min_points = self.config.get('analysis.min_data_points', 3)
                
                if non_na_count >= min_points:
                    df_interp[col] = df[col].interpolate(method=method, limit_direction='both')
        
        return df_interp
    
    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and optionally handle outliers"""
        df_clean = df.copy()
        outlier_threshold = self.config.get('analysis.outlier_std_threshold', 3)
        
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) > 3:
                mean = series.mean()
                std = series.std()
                
                if std > 0:
                    # Mark outliers
                    lower_bound = mean - outlier_threshold * std
                    upper_bound = mean + outlier_threshold * std
                    
                    outliers = (series < lower_bound) | (series > upper_bound)
                    if outliers.any():
                        self._logger.info(f"Found {outliers.sum()} outliers in {col}")
                        
                        # Store outlier information in metadata
                        outlier_indices = series[outliers].index.tolist()
                        self.state.set(f"outliers_{col}", outlier_indices)
        
        return df_clean

# --- 13. Enhanced Financial Analysis Engine ---
class FinancialAnalysisEngine(Component):
    """Core financial analysis engine with advanced features"""
    
    def __init__(self, config: Configuration, state: StateManager):
        super().__init__(config, state)
        self.pattern_matcher = PatternMatcher()
        self.cache = AdvancedCache()
    
    def _do_initialize(self):
        """Initialize analysis components"""
        # Load any required models or data
        pass
    
    def analyze_financial_statements(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive financial statement analysis"""
        cache_key = self._generate_cache_key(df)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            self._logger.info("Returning cached analysis")
            return cached_result
        
        analysis = {
            'summary': self._generate_summary(df),
            'metrics': self._extract_key_metrics(df),
            'ratios': self._calculate_ratios(df),
            'trends': self._analyze_trends(df),
            'quality_score': self._calculate_quality_score(df),
            'insights': self._generate_insights(df)
        }
        
        # Cache the result
        self.cache.set(cache_key, analysis, ttl=3600)
        
        return analysis
    
    def _generate_cache_key(self, df: pd.DataFrame) -> str:
        """Generate cache key for dataframe"""
        # Use shape and sample of data for key
        key_parts = [
            str(df.shape),
            str(df.index[:5].tolist()),
            str(df.columns[:5].tolist()),
            str(df.iloc[:5, :5].values.tolist()) if df.shape[0] >= 5 and df.shape[1] >= 5 else ""
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'error': 'No numeric data found'}
        
        summary = {
            'total_metrics': len(df),
            'years_covered': len(numeric_df.columns),
            'year_range': f"{numeric_df.columns[0]} - {numeric_df.columns[-1]}" if len(numeric_df.columns) > 0 else "N/A",
            'completeness': (numeric_df.notna().sum().sum() / numeric_df.size) * 100,
            'key_statistics': {}
        }
        
        # Calculate key statistics
        for col in numeric_df.columns[-3:]:  # Last 3 years
            summary['key_statistics'][str(col)] = {
                'mean': numeric_df[col].mean(),
                'median': numeric_df[col].median(),
                'std': numeric_df[col].std(),
                'min': numeric_df[col].min(),
                'max': numeric_df[col].max()
            }
        
        return summary
    
    def _extract_key_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract and classify key financial metrics"""
        metrics = {}
        
        for idx in df.index:
            classifications = self.pattern_matcher.classify_metric(str(idx))
            if classifications:
                top_classification = max(classifications.items(), key=lambda x: x[1])
                metric_type, confidence = top_classification
                
                if confidence > 0.5:
                    if metric_type not in metrics:
                        metrics[metric_type] = []
                    
                    metrics[metric_type].append({
                        'name': str(idx),
                        'confidence': confidence,
                        'values': df.loc[idx].to_dict()
                    })
        
        return metrics
    
    def _calculate_ratios(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate financial ratios with error handling"""
        ratios = {}
        
        # Extract key metrics using pattern matching
        metrics = self._extract_key_metrics(df)
        
        # Liquidity Ratios
        try:
            liquidity = pd.DataFrame()
            
            # Current Ratio
            current_assets = self._get_metric_value(df, metrics, 'current_assets')
            current_liabilities = self._get_metric_value(df, metrics, 'current_liabilities')
            
            if current_assets is not None and current_liabilities is not None:
                liquidity['Current Ratio'] = current_assets / current_liabilities.replace(0, np.nan)
            
            # Quick Ratio
            inventory = self._get_metric_value(df, metrics, 'inventory')
            if current_assets is not None and inventory is not None and current_liabilities is not None:
                liquidity['Quick Ratio'] = (current_assets - inventory) / current_liabilities.replace(0, np.nan)
            
            if not liquidity.empty:
                ratios['Liquidity'] = liquidity.T
                
        except Exception as e:
            self._logger.error(f"Error calculating liquidity ratios: {e}")
        
        # Profitability Ratios
        try:
            profitability = pd.DataFrame()
            
            # Net Profit Margin
            net_income = self._get_metric_value(df, metrics, 'net_income')
            revenue = self._get_metric_value(df, metrics, 'revenue')
            
            if net_income is not None and revenue is not None:
                profitability['Net Profit Margin %'] = (net_income / revenue.replace(0, np.nan)) * 100
            
            # ROA
            total_assets = self._get_metric_value(df, metrics, 'total_assets')
            if net_income is not None and total_assets is not None:
                profitability['Return on Assets %'] = (net_income / total_assets.replace(0, np.nan)) * 100
            
            # ROE
            total_equity = self._get_metric_value(df, metrics, 'total_equity')
            if net_income is not None and total_equity is not None:
                profitability['Return on Equity %'] = (net_income / total_equity.replace(0, np.nan)) * 100
            
            if not profitability.empty:
                ratios['Profitability'] = profitability.T
                
        except Exception as e:
            self._logger.error(f"Error calculating profitability ratios: {e}")
        
        # Leverage Ratios
        try:
            leverage = pd.DataFrame()
            
            # Debt to Equity
            total_liabilities = self._get_metric_value(df, metrics, 'total_liabilities')
            if total_liabilities is not None and total_equity is not None:
                leverage['Debt to Equity'] = total_liabilities / total_equity.replace(0, np.nan)
            
            # Debt Ratio
            if total_liabilities is not None and total_assets is not None:
                leverage['Debt Ratio'] = total_liabilities / total_assets.replace(0, np.nan)
            
            if not leverage.empty:
                ratios['Leverage'] = leverage.T
                
        except Exception as e:
            self._logger.error(f"Error calculating leverage ratios: {e}")
        
        return ratios
    
    def _get_metric_value(self, df: pd.DataFrame, metrics: Dict, metric_type: str) -> Optional[pd.Series]:
        """Get metric value from dataframe with fallback"""
        if metric_type in metrics and metrics[metric_type]:
            # Get the highest confidence match
            best_match = max(metrics[metric_type], key=lambda x: x['confidence'])
            metric_name = best_match['name']
            
            if metric_name in df.index:
                return df.loc[metric_name]
        
        return None
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in financial data"""
        trends = {}
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        for idx in numeric_df.index:
            series = numeric_df.loc[idx].dropna()
            
            if len(series) >= 3:
                # Calculate trend metrics
                years = np.arange(len(series))
                values = series.values
                
                # Linear regression
                slope, intercept = np.polyfit(years, values, 1)
                
                # Compound Annual Growth Rate (CAGR)
                if series.iloc[0] > 0 and series.iloc[-1] > 0:
                    years_diff = len(series) - 1
                    cagr = ((series.iloc[-1] / series.iloc[0]) ** (1 / years_diff) - 1) * 100
                else:
                    cagr = None
                
                # Volatility
                volatility = series.pct_change().std() * 100
                
                trends[str(idx)] = {
                    'slope': slope,
                    'direction': 'increasing' if slope > 0 else 'decreasing',
                    'cagr': cagr,
                    'volatility': volatility,
                    'r_squared': self._calculate_r_squared(years, values, slope, intercept)
                }
        
        return trends
    
    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
        """Calculate R-squared for linear regression"""
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score"""
        scores = []
        
        # Completeness score
        completeness = (df.notna().sum().sum() / df.size) * 100
        scores.append(completeness)
        
        # Consistency score (check for reasonable values)
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            # Check for negative values in typically positive metrics
            positive_metrics = ['assets', 'revenue', 'equity']
            consistency_score = 100
            
            for idx in numeric_df.index:
                if any(keyword in str(idx).lower() for keyword in positive_metrics):
                    negative_count = (numeric_df.loc[idx] < 0).sum()
                    if negative_count > 0:
                        consistency_score -= (negative_count / len(numeric_df.columns)) * 20
            
            scores.append(max(0, consistency_score))
        
        # Temporal consistency (year-over-year changes)
        if len(numeric_df.columns) > 1:
            temporal_score = 100
            extreme_changes = 0
            
            for idx in numeric_df.index:
                series = numeric_df.loc[idx].dropna()
                if len(series) > 1:
                    pct_changes = series.pct_change().dropna()
                    # Flag changes over 200%
                    extreme_changes += (pct_changes.abs() > 2).sum()
            
            total_changes = len(numeric_df) * (len(numeric_df.columns) - 1)
            if total_changes > 0:
                temporal_score -= (extreme_changes / total_changes) * 50
            
            scores.append(max(0, temporal_score))
        
        return sum(scores) / len(scores) if scores else 0
    
    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable insights from analysis"""
        insights = []
        
        # Get calculated ratios
        ratios = self._calculate_ratios(df)
        
        # Liquidity insights
        if 'Liquidity' in ratios and 'Current Ratio' in ratios['Liquidity'].index:
            current_ratios = ratios['Liquidity'].loc['Current Ratio'].dropna()
            if len(current_ratios) > 0:
                latest_cr = current_ratios.iloc[-1]
                if latest_cr < 1:
                    insights.append(f"⚠️ Low current ratio ({latest_cr:.2f}) indicates potential liquidity issues")
                elif latest_cr > 3:
                    insights.append(f"💡 High current ratio ({latest_cr:.2f}) suggests excess idle assets")
        
        # Profitability insights
        if 'Profitability' in ratios and 'Net Profit Margin %' in ratios['Profitability'].index:
            npm = ratios['Profitability'].loc['Net Profit Margin %'].dropna()
            if len(npm) > 1:
                trend = 'improving' if npm.iloc[-1] > npm.iloc[0] else 'declining'
                insights.append(f"📊 Net profit margin is {trend} ({npm.iloc[0]:.1f}% → {npm.iloc[-1]:.1f}%)")
        
        # Leverage insights
        if 'Leverage' in ratios and 'Debt to Equity' in ratios['Leverage'].index:
            de_ratio = ratios['Leverage'].loc['Debt to Equity'].dropna()
            if len(de_ratio) > 0:
                latest_de = de_ratio.iloc[-1]
                if latest_de > 2:
                    insights.append(f"⚠️ High debt-to-equity ratio ({latest_de:.2f}) indicates high leverage")
        
        # Growth insights
        trends = self._analyze_trends(df)
        
        # Revenue growth
        revenue_trends = [v for k, v in trends.items() if 'revenue' in k.lower()]
        if revenue_trends and revenue_trends[0].get('cagr') is not None:
            cagr = revenue_trends[0]['cagr']
            if cagr > 20:
                insights.append(f"🚀 Strong revenue growth (CAGR: {cagr:.1f}%)")
            elif cagr < 0:
                insights.append(f"📉 Declining revenue (CAGR: {cagr:.1f}%)")
        
        # Data quality
        quality_score = self._calculate_quality_score(df)
        if quality_score < 70:
            insights.append(f"⚠️ Data quality score is low ({quality_score:.0f}%), results may be less reliable")
        
        return insights

# --- 14. AI-Enhanced Mapping System ---
class AIMapper(Component):
    """AI-powered mapping with fallback mechanisms"""
    
    def __init__(self, config: Configuration, state: StateManager):
        super().__init__(config, state)
        self.model = None
        self.embeddings_cache = AdvancedCache(max_size_mb=50)
        self.fallback_mapper = None
    
    def _do_initialize(self):
        """Initialize AI components"""
        if not self.config.get('ai.enabled', True):
            self._logger.info("AI mapping disabled in configuration")
            return
        
        try:
            # Only try to load if available
            if SENTENCE_TRANSFORMER_AVAILABLE:
                model_name = self.config.get('ai.model_name', 'all-MiniLM-L6-v2')
                self.model = SentenceTransformer(model_name)
                self._logger.info(f"Loaded AI model: {model_name}")
                
                # Pre-compute standard embeddings
                self._precompute_standard_embeddings()
            else:
                self._logger.warning("Sentence transformers not available, using fallback")
                
        except Exception as e:
            self._logger.error(f"Failed to initialize AI model: {e}")
        
        # Initialize fallback
        self.fallback_mapper = FuzzyMapper(self.config, self.state)
        self.fallback_mapper.initialize()
    
    def _precompute_standard_embeddings(self):
        """Pre-compute embeddings for standard metrics"""
        if not self.model:
            return
        
        standard_metrics = {
            'Total Assets': ['total assets', 'sum of assets', 'asset total'],
            'Total Liabilities': ['total liabilities', 'sum of liabilities', 'liability total'],
            'Total Equity': ['total equity', 'shareholders equity', 'net worth'],
            'Revenue': ['revenue', 'sales', 'turnover', 'income from operations'],
            'Net Income': ['net income', 'net profit', 'profit after tax', 'earnings'],
            'Current Assets': ['current assets', 'short term assets', 'liquid assets'],
            'Current Liabilities': ['current liabilities', 'short term liabilities'],
            'Cash': ['cash', 'cash and cash equivalents', 'liquid funds'],
            'Inventory': ['inventory', 'stock', 'goods'],
            'Receivables': ['receivables', 'accounts receivable', 'trade receivables', 'debtors'],
        }
        
        for metric, descriptions in standard_metrics.items():
            combined_text = ' '.join(descriptions)
            embedding = self._get_embedding(combined_text)
            if embedding is not None:
                self.state.set(f"standard_embedding_{metric}", embedding, persist=True)
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding with caching"""
        if not self.model:
            return None
        
        # Check cache
        cache_key = f"embedding_{hashlib.md5(text.encode()).hexdigest()}"
        cached = self.embeddings_cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            # Compute embedding
            embedding = self.model.encode(
                text, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Cache it
            self.embeddings_cache.set(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            self._logger.error(f"Error computing embedding: {e}")
            return None
    
    def map_metrics(self, source_metrics: List[str], 
                   target_metrics: Optional[List[str]] = None,
                   confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Map source metrics to target metrics"""
        
        if not self.model:
            # Use fallback
            return self.fallback_mapper.map_metrics(source_metrics, target_metrics)
        
        if confidence_threshold is None:
            confidence_threshold = self.config.get('ai.similarity_threshold', 0.6)
        
        # Default target metrics
        if target_metrics is None:
            target_metrics = self._get_standard_metrics()
        
        mappings = {}
        confidence_scores = {}
        suggestions = {}
        unmapped = []
        
        # Batch process for efficiency
        batch_size = self.config.get('ai.batch_size', 32)
        
        for i in range(0, len(source_metrics), batch_size):
            batch = source_metrics[i:i + batch_size]
            
            try:
                # Get embeddings for batch
                source_embeddings = []
                valid_sources = []
                
                for metric in batch:
                    embedding = self._get_embedding(str(metric).lower())
                    if embedding is not None:
                        source_embeddings.append(embedding)
                        valid_sources.append(metric)
                
                if not source_embeddings:
                    unmapped.extend(batch)
                    continue
                
                # Get target embeddings
                target_embeddings = []
                valid_targets = []
                
                for target in target_metrics:
                    # Check for pre-computed embedding
                    embedding = self.state.get(f"standard_embedding_{target}")
                    
                    if embedding is None:
                        embedding = self._get_embedding(target.lower())
                    
                    if embedding is not None:
                        target_embeddings.append(embedding)
                        valid_targets.append(target)
                
                if not target_embeddings:
                    unmapped.extend(valid_sources)
                    continue
                
                # Compute similarities
                source_matrix = np.vstack(source_embeddings)
                target_matrix = np.vstack(target_embeddings)
                
                similarities = cosine_similarity(source_matrix, target_matrix)
                
                # Process results
                for idx, source in enumerate(valid_sources):
                    sim_scores = similarities[idx]
                    top_indices = np.argsort(sim_scores)[::-1][:3]
                    
                    top_matches = [
                        (valid_targets[i], sim_scores[i]) 
                        for i in top_indices
                    ]
                    
                    best_target, best_score = top_matches[0]
                    
                    if best_score >= confidence_threshold:
                        mappings[source] = best_target
                        confidence_scores[source] = float(best_score)
                    else:
                        unmapped.append(source)
                    
                    suggestions[source] = [
                        (target, float(score)) 
                        for target, score in top_matches
                    ]
                    
            except Exception as e:
                self._logger.error(f"Error in batch mapping: {e}")
                unmapped.extend(batch)
        
        return {
            'mappings': mappings,
            'confidence_scores': confidence_scores,
            'suggestions': suggestions,
            'unmapped_metrics': unmapped,
            'method': 'ai' if self.model else 'fuzzy'
        }
    
    def _get_standard_metrics(self) -> List[str]:
        """Get list of standard financial metrics"""
        return [
            'Total Assets', 'Current Assets', 'Non-current Assets',
            'Cash and Cash Equivalents', 'Inventory', 'Trade Receivables',
            'Property Plant and Equipment', 'Total Liabilities',
            'Current Liabilities', 'Non-current Liabilities',
            'Total Equity', 'Share Capital', 'Retained Earnings',
            'Revenue', 'Cost of Goods Sold', 'Gross Profit',
            'Operating Expenses', 'Operating Income', 'Net Income',
            'Earnings Per Share', 'Operating Cash Flow',
            'Investing Cash Flow', 'Financing Cash Flow'
        ]

# --- 15. Fuzzy Mapping Fallback ---
class FuzzyMapper(Component):
    """Fuzzy string matching for metric mapping"""
    
    def _do_initialize(self):
        """Initialize fuzzy mapper"""
        pass
    
    def map_metrics(self, source_metrics: List[str], 
                   target_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Map metrics using fuzzy string matching"""
        
        if target_metrics is None:
            target_metrics = self._get_standard_metrics()
        
        mappings = {}
        confidence_scores = {}
        suggestions = {}
        unmapped = []
        
        for source in source_metrics:
            source_lower = str(source).lower()
            
            # Calculate fuzzy scores
            scores = []
            for target in target_metrics:
                score = fuzz.token_sort_ratio(source_lower, target.lower()) / 100.0
                scores.append((target, score))
            
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get best match
            if scores and scores[0][1] > 0.7:
                mappings[source] = scores[0][0]
                confidence_scores[source] = scores[0][1]
            else:
                unmapped.append(source)
            
            # Store top 3 suggestions
            suggestions[source] = scores[:3]
        
        return {
            'mappings': mappings,
            'confidence_scores': confidence_scores,
            'suggestions': suggestions,
            'unmapped_metrics': unmapped,
            'method': 'fuzzy'
        }
    
    def _get_standard_metrics(self) -> List[str]:
        """Get standard metrics list"""
        # Same as AIMapper
        return [
            'Total Assets', 'Current Assets', 'Non-current Assets',
            'Cash and Cash Equivalents', 'Inventory', 'Trade Receivables',
            'Property Plant and Equipment', 'Total Liabilities',
            'Current Liabilities', 'Non-current Liabilities',
            'Total Equity', 'Share Capital', 'Retained Earnings',
            'Revenue', 'Cost of Goods Sold', 'Gross Profit',
            'Operating Expenses', 'Operating Income', 'Net Income',
            'Earnings Per Share', 'Operating Cash Flow',
            'Investing Cash Flow', 'Financing Cash Flow'
        ]

# --- 16. Enhanced Penman-Nissim Analyzer (FIXED) ---
class EnhancedPenmanNissimAnalyzer:
    """Enhanced Penman-Nissim analyzer with flexible initialization"""
    
    def __init__(self, df: pd.DataFrame, mappings: Dict[str, str]):
        self.df = df
        self.mappings = mappings
        self.logger = LoggerFactory.get_logger('PenmanNissim')
        
        # Initialize core analyzer with proper handling
        self._initialize_core_analyzer()
    
    def _initialize_core_analyzer(self):
        """Initialize core analyzer with proper error handling"""
        if CORE_COMPONENTS_AVAILABLE:
            try:
                # Try different initialization patterns
                try:
                    # First try with both parameters
                    self.core_analyzer = CorePenmanNissim(self.df, self.mappings)
                    self.logger.info("Initialized CorePenmanNissim with df and mappings")
                except TypeError:
                    try:
                        # Try with just df
                        self.core_analyzer = CorePenmanNissim(self.df)
                        if hasattr(self.core_analyzer, 'set_mappings'):
                            self.core_analyzer.set_mappings(self.mappings)
                        elif hasattr(self.core_analyzer, 'mappings'):
                            self.core_analyzer.mappings = self.mappings
                        self.logger.info("Initialized CorePenmanNissim with df only")
                    except TypeError:
                        # Try with no parameters
                        self.core_analyzer = CorePenmanNissim()
                        if hasattr(self.core_analyzer, 'df'):
                            self.core_analyzer.df = self.df
                        if hasattr(self.core_analyzer, 'mappings'):
                            self.core_analyzer.mappings = self.mappings
                        self.logger.info("Initialized CorePenmanNissim with no parameters")
            except Exception as e:
                self.logger.warning(f"Could not initialize CorePenmanNissim: {e}")
                self.core_analyzer = None
        else:
            self.core_analyzer = None
    
    def calculate_all(self):
        """Calculate all Penman-Nissim metrics"""
        if self.core_analyzer and hasattr(self.core_analyzer, 'calculate_all'):
            try:
                return self.core_analyzer.calculate_all()
            except Exception as e:
                self.logger.error(f"Error in core calculate_all: {e}")
        
        # Fallback implementation
        return self._fallback_calculate_all()
    
    def _fallback_calculate_all(self):
        """Fallback implementation of Penman-Nissim calculations"""
        try:
            # Apply mappings
            mapped_df = self.df.rename(index=self.mappings)
            
            results = {
                'reformulated_balance_sheet': self._reformulate_balance_sheet(mapped_df),
                'reformulated_income_statement': self._reformulate_income_statement(mapped_df),
                'ratios': self._calculate_ratios(mapped_df),
                'free_cash_flow': self._calculate_free_cash_flow(mapped_df)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in fallback calculations: {e}")
            return {'error': str(e)}
    
    def _reformulate_balance_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reformulate balance sheet for Penman-Nissim analysis"""
        reformulated = pd.DataFrame(index=df.columns)
        
        # Operating assets
        operating_assets = ['Current Assets', 'Property Plant Equipment', 'Intangible Assets']
        operating_assets_sum = pd.Series(0, index=df.columns)
        for asset in operating_assets:
            if asset in df.index:
                operating_assets_sum += df.loc[asset].fillna(0)
        
        # Financial assets
        financial_assets = ['Cash', 'Short-term Investments', 'Long-term Investments']
        financial_assets_sum = pd.Series(0, index=df.columns)
        for asset in financial_assets:
            if asset in df.index:
                financial_assets_sum += df.loc[asset].fillna(0)
        
        # Operating liabilities
        operating_liabilities = ['Accounts Payable', 'Accrued Expenses', 'Deferred Revenue']
        operating_liabilities_sum = pd.Series(0, index=df.columns)
        for liab in operating_liabilities:
            if liab in df.index:
                operating_liabilities_sum += df.loc[liab].fillna(0)
        
        # Financial liabilities
        financial_liabilities = ['Short-term Debt', 'Long-term Debt', 'Bonds Payable']
        financial_liabilities_sum = pd.Series(0, index=df.columns)
        for liab in financial_liabilities:
            if liab in df.index:
                financial_liabilities_sum += df.loc[liab].fillna(0)
        
        # Net operating assets
        reformulated['Net Operating Assets'] = operating_assets_sum - operating_liabilities_sum
        reformulated['Net Financial Assets'] = financial_assets_sum - financial_liabilities_sum
        reformulated['Common Equity'] = reformulated['Net Operating Assets'] + reformulated['Net Financial Assets']
        
        return reformulated
    
    def _reformulate_income_statement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reformulate income statement for Penman-Nissim analysis"""
        reformulated = pd.DataFrame(index=df.columns)
        
        if 'Revenue' in df.index and 'Operating Income' in df.index:
            reformulated['Operating Income'] = df.loc['Operating Income']
            
            # Tax allocation
            if 'Tax Expense' in df.index and 'Income Before Tax' in df.index:
                income_before_tax = df.loc['Income Before Tax'].replace(0, np.nan)
                tax_rate = df.loc['Tax Expense'] / income_before_tax
                reformulated['Tax on Operating Income'] = reformulated['Operating Income'] * tax_rate
                reformulated['Operating Income After Tax'] = (
                    reformulated['Operating Income'] - reformulated['Tax on Operating Income']
                )
        
        # Financial items
        if 'Interest Expense' in df.index:
            reformulated['Net Financial Expense'] = df.loc['Interest Expense']
            if 'Interest Income' in df.index:
                reformulated['Net Financial Expense'] -= df.loc['Interest Income']
        
        return reformulated
    
    def _calculate_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Penman-Nissim ratios"""
        ratios = pd.DataFrame(index=df.columns)
        
        # Get reformulated statements
        ref_bs = self._reformulate_balance_sheet(df)
        ref_is = self._reformulate_income_statement(df)
        
        # RNOA (Return on Net Operating Assets)
        if 'Operating Income After Tax' in ref_is.columns and 'Net Operating Assets' in ref_bs.columns:
            noa = ref_bs['Net Operating Assets'].replace(0, np.nan)
            ratios['Return on Net Operating Assets (RNOA) %'] = (
                ref_is['Operating Income After Tax'] / noa
            ) * 100
        
        # FLEV (Financial Leverage)
        if 'Net Financial Assets' in ref_bs.columns and 'Common Equity' in ref_bs.columns:
            ce = ref_bs['Common Equity'].replace(0, np.nan)
            ratios['Financial Leverage (FLEV)'] = -ref_bs['Net Financial Assets'] / ce
        
        # NBC (Net Borrowing Cost)
        if 'Net Financial Expense' in ref_is.columns and 'Net Financial Assets' in ref_bs.columns:
            nfa = ref_bs['Net Financial Assets'].replace(0, np.nan)
            ratios['Net Borrowing Cost (NBC) %'] = (
                -ref_is['Net Financial Expense'] / nfa
            ) * 100
        
        # OPM (Operating Profit Margin)
        if 'Operating Income After Tax' in ref_is.columns and 'Revenue' in df.index:
            revenue = df.loc['Revenue'].replace(0, np.nan)
            ratios['Operating Profit Margin (OPM) %'] = (
                ref_is['Operating Income After Tax'] / revenue
            ) * 100
        
        # NOAT (Net Operating Asset Turnover)
        if 'Revenue' in df.index and 'Net Operating Assets' in ref_bs.columns:
            noa = ref_bs['Net Operating Assets'].replace(0, np.nan)
            ratios['Net Operating Asset Turnover (NOAT)'] = df.loc['Revenue'] / noa
        
        return ratios.T
    
    def _calculate_free_cash_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate free cash flow"""
        fcf = pd.DataFrame(index=df.columns)
        
        if 'Operating Cash Flow' in df.index:
            fcf['Operating Cash Flow'] = df.loc['Operating Cash Flow']
            
            if 'Capital Expenditure' in df.index:
                fcf['Free Cash Flow'] = fcf['Operating Cash Flow'] - df.loc['Capital Expenditure']
            else:
                fcf['Free Cash Flow'] = fcf['Operating Cash Flow']
        
        return fcf

# --- 17. Manual Mapping Interface Component ---
class ManualMappingInterface:
    """Manual mapping interface for metric mapping"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.source_metrics = [str(m) for m in df.index.tolist()]
        self.target_metrics = self._get_standard_target_metrics()
    
    def _get_standard_target_metrics(self) -> List[str]:
        """Get standard target metrics for mapping"""
        return [
            'Total Assets', 'Current Assets', 'Non-current Assets',
            'Cash and Cash Equivalents', 'Inventory', 'Trade Receivables',
            'Property Plant and Equipment', 'Total Liabilities',
            'Current Liabilities', 'Non-current Liabilities',
            'Total Equity', 'Share Capital', 'Retained Earnings',
            'Revenue', 'Cost of Goods Sold', 'Gross Profit',
            'Operating Expenses', 'Operating Income', 'Net Income',
            'Earnings Per Share', 'Operating Cash Flow',
            'Investing Cash Flow', 'Financing Cash Flow',
            'Interest Expense', 'Tax Expense', 'EBIT', 'EBITDA'
        ]
    
    def render(self) -> Dict[str, str]:
        """Render the manual mapping interface and return mappings"""
        st.subheader("📋 Manual Metric Mapping")
        st.info("Map your financial statement items to standard metrics for analysis")
        
        # Essential mappings for ratios
        essential_mappings = {
            'Balance Sheet': {
                'Total Assets': ['Total Assets', 'TOTAL ASSETS', 'Assets Total'],
                'Current Assets': ['Current Assets', 'CURRENT ASSETS', 'Short-term Assets'],
                'Current Liabilities': ['Current Liabilities', 'CURRENT LIABILITIES', 'Short-term Liabilities'],
                'Total Liabilities': ['Total Liabilities', 'TOTAL LIABILITIES', 'Liabilities Total'],
                'Total Equity': ['Total Equity', 'TOTAL EQUITY', 'Shareholders Equity', 'Net Worth'],
                'Cash and Cash Equivalents': ['Cash', 'Cash and Cash Equivalents', 'Cash & Bank'],
                'Inventory': ['Inventory', 'Inventories', 'Stock'],
                'Trade Receivables': ['Trade Receivables', 'Accounts Receivable', 'Debtors'],
            },
            'Income Statement': {
                'Revenue': ['Revenue', 'Total Income', 'Net Sales', 'Revenue from Operations', 'Sales'],
                'Cost of Goods Sold': ['Cost of Goods Sold', 'COGS', 'Cost of Sales', 'Cost of Materials'],
                'Operating Expenses': ['Operating Expenses', 'OPEX', 'Other Expenses'],
                'Net Income': ['Net Income', 'Net Profit', 'Profit for the Period', 'PAT'],
                'Interest Expense': ['Interest Expense', 'Finance Costs', 'Interest Costs'],
                'Tax Expense': ['Tax Expense', 'Income Tax', 'Tax'],
                'Operating Income': ['Operating Income', 'EBIT', 'Operating Profit'],
            },
            'Cash Flow': {
                'Operating Cash Flow': ['Operating Cash Flow', 'Cash from Operations', 'CFO'],
                'Investing Cash Flow': ['Investing Cash Flow', 'Cash from Investing', 'CFI'],
                'Financing Cash Flow': ['Financing Cash Flow', 'Cash from Financing', 'CFF'],
            }
        }
        
        mappings = {}
        
        # Create tabs for different statement types
        tabs = st.tabs(list(essential_mappings.keys()))
        
        for i, (statement_type, metrics) in enumerate(essential_mappings.items()):
            with tabs[i]:
                cols = st.columns(2)
                
                for j, (target, suggestions) in enumerate(metrics.items()):
                    col = cols[j % 2]
                    
                    with col:
                        # Find best match from source metrics
                        default_idx = 0
                        for k, source in enumerate(self.source_metrics):
                            if any(sug.lower() in source.lower() for sug in suggestions):
                                default_idx = k + 1
                                break
                        
                        selected = st.selectbox(
                            f"{target}:",
                            ['(Not mapped)'] + self.source_metrics,
                            index=default_idx,
                            key=f"map_{statement_type}_{target}",
                            help=f"Common names: {', '.join(suggestions[:3])}"
                        )
                        
                        if selected != '(Not mapped)':
                            mappings[selected] = target
        
        # Additional custom mappings
        with st.expander("➕ Add Custom Mappings"):
            col1, col2 = st.columns(2)
            
            with col1:
                custom_source = st.selectbox(
                    "Source Metric:",
                    [m for m in self.source_metrics if m not in mappings],
                    key="custom_source"
                )
            
            with col2:
                custom_target = st.selectbox(
                    "Target Metric:",
                    self.target_metrics,
                    key="custom_target"
                )
            
            if st.button("Add Mapping", key="add_custom_mapping"):
                if custom_source and custom_target:
                    mappings[custom_source] = custom_target
                    st.success(f"Added: {custom_source} → {custom_target}")
        
        # Show current mappings
        if mappings:
            with st.expander("📊 Current Mappings", expanded=True):
                mapping_df = pd.DataFrame(
                    [(k, v) for k, v in sorted(mappings.items(), key=lambda x: x[1])],
                    columns=['Source Metric', 'Target Metric']
                )
                st.dataframe(mapping_df, use_container_width=True)
        
        return mappings

# --- 18. UI Components Factory ---
class UIComponentFactory:
    """Factory for creating UI components with consistent styling"""
    
    @staticmethod
    def create_metric_card(title: str, value: Any, delta: Optional[float] = None, 
                          help_text: Optional[str] = None) -> None:
        """Create a metric card"""
        col = st.container()
        
        with col:
            if help_text:
                st.metric(title, value, delta, help=help_text)
            else:
                st.metric(title, value, delta)
    
    @staticmethod
    def create_progress_indicator(progress: float, text: str = "") -> None:
        """Create progress indicator"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Animate progress
        for i in range(int(progress * 100)):
            progress_bar.progress(i / 100)
            if text:
                status_text.text(f"{text} ({i}%)")
            time.sleep(0.01)
        
        progress_bar.progress(progress)
        if text:
            status_text.text(f"{text} ({int(progress * 100)}%)")
    
    @staticmethod
    def create_data_quality_badge(score: float) -> None:
        """Create data quality badge"""
        if score >= 80:
            color = "green"
            label = "High Quality"
        elif score >= 60:
            color = "orange"
            label = "Medium Quality"
        else:
            color = "red"
            label = "Low Quality"
        
        st.markdown(
            f'<span style="background-color: {color}; color: white; '
            f'padding: 5px 10px; border-radius: 5px; font-weight: bold;">'
            f'{label} ({score:.0f}%)</span>',
            unsafe_allow_html=True
        )
    
    @staticmethod
    def create_insight_card(insight: str, insight_type: str = "info") -> None:
        """Create insight card with appropriate styling"""
        icons = {
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'info': '💡'
        }
        
        colors = {
            'success': '#d4edda',
            'warning': '#fff3cd',
            'error': '#f8d7da',
            'info': '#d1ecf1'
        }
        
        icon = icons.get(insight_type, '💡')
        color = colors.get(insight_type, '#d1ecf1')
        
        st.markdown(
            f'<div style="background-color: {color}; padding: 10px; '
            f'border-radius: 5px; margin: 5px 0;">'
            f'{icon} {insight}</div>',
            unsafe_allow_html=True
        )

# --- 19. Sample Data Generator ---
class SampleDataGenerator:
    """Generate sample financial data for demonstration"""
    
    @staticmethod
    def generate_indian_tech_company() -> pd.DataFrame:
        """Generate sample data for Indian tech company"""
        years = ['2019', '2020', '2021', '2022', '2023']
        
        data = {
            # Balance Sheet Items
            'Total Assets': [45000, 52000, 61000, 72000, 85000],
            'Current Assets': [28000, 32000, 38000, 45000, 53000],
            'Non-current Assets': [17000, 20000, 23000, 27000, 32000],
            'Cash and Cash Equivalents': [12000, 14000, 17000, 21000, 25000],
            'Inventory': [2000, 2300, 2700, 3200, 3800],
            'Trade Receivables': [8000, 9200, 10800, 12700, 15000],
            'Property Plant and Equipment': [10000, 12000, 14000, 16500, 19500],
            
            'Total Liabilities': [18000, 20000, 22500, 25500, 29000],
            'Current Liabilities': [10000, 11000, 12500, 14000, 16000],
            'Non-current Liabilities': [8000, 9000, 10000, 11500, 13000],
            'Short-term Borrowings': [3000, 3300, 3700, 4200, 4800],
            'Long-term Debt': [6000, 6600, 7300, 8200, 9200],
            'Trade Payables': [4000, 4400, 4900, 5500, 6200],
            
            'Total Equity': [27000, 32000, 38500, 46500, 56000],
            'Share Capital': [10000, 10000, 10000, 10000, 10000],
            'Reserves and Surplus': [17000, 22000, 28500, 36500, 46000],
            
            # Income Statement Items
            'Revenue': [35000, 38000, 45000, 54000, 65000],
            'Cost of Goods Sold': [21000, 22000, 25200, 29700, 35100],
            'Gross Profit': [14000, 16000, 19800, 24300, 29900],
            'Operating Expenses': [8000, 8800, 10300, 12150, 14300],
            'Operating Income': [6000, 7200, 9500, 12150, 15600],
            'EBIT': [6000, 7200, 9500, 12150, 15600],
            'Interest Expense': [800, 880, 970, 1090, 1220],
            'Income Before Tax': [5200, 6320, 8530, 11060, 14380],
            'Tax Expense': [1560, 1896, 2559, 3318, 4314],
            'Net Income': [3640, 4424, 5971, 7742, 10066],
            
            # Cash Flow Items
            'Operating Cash Flow': [5500, 6600, 8800, 11000, 14000],
            'Investing Cash Flow': [-3000, -3500, -4200, -5000, -6000],
            'Financing Cash Flow': [-1500, -1800, -2200, -2700, -3300],
            'Capital Expenditure': [2800, 3200, 3800, 4500, 5300],
            'Free Cash Flow': [2700, 3400, 5000, 6500, 8700],
        }
        
        df = pd.DataFrame(data, index=list(data.keys()), columns=years)
        return df
    
    @staticmethod
    def generate_us_manufacturing() -> pd.DataFrame:
        """Generate sample data for US manufacturing company"""
        years = ['2019', '2020', '2021', '2022', '2023']
        
        data = {
            # Balance Sheet Items (in millions USD)
            'Total Assets': [120000, 115000, 125000, 135000, 145000],
            'Current Assets': [45000, 43000, 48000, 52000, 56000],
            'Non-current Assets': [75000, 72000, 77000, 83000, 89000],
            'Cash and Cash Equivalents': [8000, 7500, 9000, 10500, 12000],
            'Inventory': [18000, 17000, 19000, 21000, 23000],
            'Trade Receivables': [15000, 14500, 16000, 17500, 19000],
            'Property Plant and Equipment': [60000, 58000, 62000, 66000, 70000],
            
            'Total Liabilities': [72000, 69000, 74000, 79000, 84000],
            'Current Liabilities': [30000, 28000, 31000, 33000, 35000],
            'Non-current Liabilities': [42000, 41000, 43000, 46000, 49000],
            'Short-term Borrowings': [10000, 9000, 10500, 11500, 12500],
            'Long-term Debt': [35000, 34000, 35500, 38000, 40500],
            'Trade Payables': [12000, 11500, 12500, 13500, 14500],
            
            'Total Equity': [48000, 46000, 51000, 56000, 61000],
            'Share Capital': [20000, 20000, 20000, 20000, 20000],
            'Retained Earnings': [28000, 26000, 31000, 36000, 41000],
            
            # Income Statement Items
            'Revenue': [95000, 88000, 102000, 115000, 128000],
            'Cost of Goods Sold': [68000, 64000, 72000, 80000, 88000],
            'Gross Profit': [27000, 24000, 30000, 35000, 40000],
            'Operating Expenses': [18000, 17000, 19000, 21000, 23000],
            'Operating Income': [9000, 7000, 11000, 14000, 17000],
            'EBIT': [9000, 7000, 11000, 14000, 17000],
            'Interest Expense': [2800, 2700, 2850, 3050, 3250],
            'Income Before Tax': [6200, 4300, 8150, 10950, 13750],
            'Tax Expense': [1550, 1075, 2038, 2738, 3438],
            'Net Income': [4650, 3225, 6112, 8212, 10312],
            
            # Cash Flow Items
            'Operating Cash Flow': [11000, 9000, 13000, 16000, 19500],
            'Investing Cash Flow': [-8000, -6000, -9000, -11000, -13000],
            'Financing Cash Flow': [-2000, -2500, -3000, -3500, -4000],
            'Capital Expenditure': [7500, 5500, 8500, 10500, 12500],
            'Free Cash Flow': [3500, 3500, 4500, 5500, 7000],
        }
        
        df = pd.DataFrame(data, index=list(data.keys()), columns=years)
        return df

# --- 20. Main Application Class (Complete) ---
class FinancialAnalyticsPlatform:
    """Main application with advanced architecture"""
    
    def __init__(self):
        # Initialize configuration
        self.config = Configuration()
        
        # Initialize state manager
        self.state = StateManager("financial_analytics")
        self.state.load_persisted()
        
        # Initialize logger
        self.logger = LoggerFactory.get_logger('FinancialAnalyticsPlatform')
        
        # Initialize components
        self.components = self._initialize_components()
        
        # Initialize UI factory
        self.ui_factory = UIComponentFactory()
        
        # Initialize sample data generator
        self.sample_generator = SampleDataGenerator()
    
    def _initialize_components(self) -> Dict[str, Component]:
        """Initialize all components with dependency injection"""
        components = {
            'security': SecurityModule(self.config, self.state),
            'processor': DataProcessor(self.config, self.state),
            'analyzer': FinancialAnalysisEngine(self.config, self.state),
            'mapper': AIMapper(self.config, self.state),
        }
        
        # Initialize all components
        for name, component in components.items():
            try:
                component.initialize()
                self.logger.info(f"Initialized component: {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {name}: {e}")
        
        return components
    
    def run(self):
        """Main application entry point"""
        try:
            # Set page config
            st.set_page_config(
                page_title="Elite Financial Analytics Platform",
                page_icon="💹",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Apply custom CSS
            self._apply_custom_css()
            
            # Render header
            self._render_header()
            
            # Render sidebar
            self._render_sidebar()
            
            # Render main content
            self._render_main_content()
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            st.error("An unexpected error occurred. Please refresh the page.")
            
            if self.config.get('app.debug', False):
                st.exception(e)
    
    def _apply_custom_css(self):
        """Apply custom CSS styling"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }
        
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        .stMetric {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .insight-card {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0.5rem;
            border-left: 4px solid;
        }
        
        .quality-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-weight: 600;
            font-size: 0.875rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render application header"""
        st.markdown(
            '<h1 class="main-header">💹 Elite Financial Analytics Platform</h1>',
            unsafe_allow_html=True
        )
        
        # Show system status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            components_status = sum(1 for c in self.components.values() if c._initialized)
            self.ui_factory.create_metric_card(
                "Components", 
                f"{components_status}/{len(self.components)}",
                help="Active system components"
            )
        
        with col2:
            mode = self.config.get('app.display_mode', Configuration.DisplayMode.LITE)
            self.ui_factory.create_metric_card(
                "Mode", 
                mode.name,
                help="Current operating mode"
            )
        
        with col3:
            cache_stats = self.components['mapper'].embeddings_cache.get_stats()
            hit_rate = cache_stats.get('hit_rate', 0)
            self.ui_factory.create_metric_card(
                "Cache Hit Rate", 
                f"{hit_rate:.1f}%",
                help="AI cache performance"
            )
        
        with col4:
            version = self.config.get('app.version', 'Unknown')
            self.ui_factory.create_metric_card(
                "Version", 
                version,
                help="Platform version"
            )
    
    def _render_sidebar(self):
        """Render sidebar with configuration options"""
        st.sidebar.title("⚙️ Configuration")
        
        # Data input section
        st.sidebar.header("📥 Data Input")
        
        input_method = st.sidebar.radio(
            "Input Method",
            ["Upload Files", "Sample Data", "Manual Entry"],
            key="input_method"
        )
        
        if input_method == "Upload Files":
            self._render_file_upload()
        elif input_method == "Sample Data":
            self._render_sample_data_loader()
        else:
            st.sidebar.info("Use the main area for manual data entry")
        
        # Settings section
        st.sidebar.header("⚙️ Settings")
        
        # Performance mode
        mode_options = [m.name for m in Configuration.DisplayMode]
        current_mode = self.config.get('app.display_mode', Configuration.DisplayMode.LITE)
        
        selected_mode = st.sidebar.selectbox(
            "Performance Mode",
            mode_options,
            index=mode_options.index(current_mode.name),
            help="FULL: All features | LITE: Balanced | MINIMAL: Fast"
        )
        
        if selected_mode != current_mode.name:
            self.config.set('app.display_mode', Configuration.DisplayMode[selected_mode])
            st.rerun()
        
        # AI Settings
        if selected_mode != "MINIMAL":
            st.sidebar.subheader("🤖 AI Settings")
            
            ai_enabled = st.sidebar.checkbox(
                "Enable AI Mapping",
                value=self.config.get('ai.enabled', True),
                help="Use AI for intelligent metric mapping"
            )
            
            self.config.set('ai.enabled', ai_enabled)
            
            if ai_enabled:
                confidence_threshold = st.sidebar.slider(
                    "Confidence Threshold",
                    0.0, 1.0,
                    self.config.get('ai.similarity_threshold', 0.6),
                    0.05,
                    help="Minimum confidence for automatic mapping"
                )
                self.config.set('ai.similarity_threshold', confidence_threshold)
        
        # Number format
        st.sidebar.subheader("🔢 Number Format")
        
        format_option = st.sidebar.radio(
            "Display Format",
            ["Indian (₹ Lakhs/Crores)", "International ($ Millions)"],
            key="number_format"
        )
        
        self.state.set('number_format', 
                      Configuration.NumberFormat.INDIAN if "Indian" in format_option 
                      else Configuration.NumberFormat.INTERNATIONAL)
        
        # Advanced options
        with st.sidebar.expander("🔧 Advanced Options"):
            debug_mode = st.sidebar.checkbox(
                "Debug Mode",
                value=self.config.get('app.debug', False),
                help="Show detailed error information"
            )
            self.config.set('app.debug', debug_mode)
            
            if st.sidebar.button("Clear Cache"):
                self._clear_all_caches()
                st.success("Cache cleared!")
            
            if st.sidebar.button("Reset Configuration"):
                self._reset_configuration()
                st.rerun()
    
    def _render_file_upload(self):
        """Render file upload interface"""
        allowed_types = self.config.get('app.allowed_file_types', [])
        max_size = self.config.get('app.max_file_size_mb', 10)
        
        uploaded_files = st.sidebar.file_uploader(
            f"Upload Financial Statements (Max {max_size}MB each)",
            type=allowed_types,
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            st.sidebar.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Validate files
            all_valid = True
            for file in uploaded_files:
                result = self.components['security'].validate_file_upload(file)
                if not result.is_valid:
                    st.sidebar.error(f"❌ {file.name}: {result.errors[0]}")
                    all_valid = False
            
            if all_valid and st.sidebar.button("Process Files", type="primary"):
                self._process_uploaded_files(uploaded_files)
    
    def _render_sample_data_loader(self):
        """Render sample data loader"""
        sample_options = [
            "Indian Tech Company (IND-AS)",
            "US Manufacturing (GAAP)",
            "European Retail (IFRS)"
        ]
        
        selected_sample = st.sidebar.selectbox(
            "Select Sample Dataset",
            sample_options
        )
        
        if st.sidebar.button("Load Sample Data", type="primary"):
            self._load_sample_data(selected_sample)
    
    def _render_main_content(self):
        """Render main content area"""
        # Check if data is loaded
        if not self.state.get('analysis_data'):
            self._render_welcome_screen()
        else:
            self._render_analysis_interface()
    
    def _render_welcome_screen(self):
        """Render welcome screen"""
        st.header("Welcome to Elite Financial Analytics Platform")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            ### 📊 Advanced Analytics
            - Comprehensive ratio analysis
            - Trend detection & forecasting
            - Industry benchmarking
            - Custom metric creation
            """)
        
        with col2:
            st.success("""
            ### 🤖 AI-Powered Features
            - Intelligent metric mapping
            - Anomaly detection
            - Natural language insights
            - Pattern recognition
            """)
        
        with col3:
            st.warning("""
            ### 🔒 Enterprise Security
            - File validation & sanitization
            - Rate limiting protection
            - Audit trail logging
            - Data encryption
            """)
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.markdown("""
            1. **Upload Data**: Use the sidebar to upload financial statements (CSV, Excel, HTML)
            2. **Map Metrics**: AI will automatically map your data to standard financial metrics
            3. **Analyze**: View comprehensive analysis including ratios, trends, and insights
            4. **Export**: Download results in various formats for further analysis
            
            **Supported Formats**: IND-AS, US GAAP, IFRS
            """)
    
    def _render_analysis_interface(self):
        """Render main analysis interface"""
        data = self.state.get('analysis_data')
        
        if not data:
            st.error("No data available for analysis")
            return
        
        # Analysis tabs
        tabs = st.tabs([
            "📊 Overview",
            "📈 Financial Ratios", 
            "📉 Trends & Forecasting",
            "🎯 Penman-Nissim",
            "🏭 Industry Comparison",
            "🔍 Data Explorer",
            "📄 Reports"
        ])
        
        with tabs[0]:
            self._render_overview_tab(data)
        
        with tabs[1]:
            self._render_ratios_tab(data)
        
        with tabs[2]:
            self._render_trends_tab(data)
        
        with tabs[3]:
            self._render_penman_nissim_tab(data)
        
        with tabs[4]:
            self._render_industry_tab(data)
        
        with tabs[5]:
            self._render_data_explorer_tab(data)
        
        with tabs[6]:
            self._render_reports_tab(data)
    
    def _render_overview_tab(self, data: pd.DataFrame):
        """Render overview tab with key metrics and insights"""
        st.header("Financial Overview")
        
        # Analyze data
        analysis = self.components['analyzer'].analyze_financial_statements(data)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.ui_factory.create_metric_card(
                "Total Metrics",
                analysis['summary']['total_metrics']
            )
        
        with col2:
            self.ui_factory.create_metric_card(
                "Years Covered",
                analysis['summary']['years_covered']
            )
        
        with col3:
            self.ui_factory.create_metric_card(
                "Data Completeness",
                f"{analysis['summary']['completeness']:.1f}%"
            )
        
        with col4:
            quality_score = analysis['quality_score']
            self.ui_factory.create_data_quality_badge(quality_score)
        
        # Key insights
        st.subheader("Key Insights")
        
        insights = analysis.get('insights', [])
        if insights:
            for insight in insights[:5]:  # Show top 5 insights
                # Determine insight type based on content
                if "⚠️" in insight:
                    insight_type = "warning"
                elif "📉" in insight:
                    insight_type = "error"
                elif "🚀" in insight or "📊" in insight:
                    insight_type = "success"
                else:
                    insight_type = "info"
                
                self.ui_factory.create_insight_card(insight, insight_type)
        else:
            st.info("No specific insights available yet. Complete the analysis to see insights.")
        
        # Quick visualizations
        st.subheader("Quick Visualizations")
        
        # Extract key metrics for visualization
        metrics = analysis.get('metrics', {})
        
        if metrics:
            # Revenue trend
            revenue_data = metrics.get('revenue', [])
            if revenue_data:
                self._render_metric_chart(revenue_data[0], "Revenue Trend")
            
            # Profitability trend
            profit_data = metrics.get('net_income', [])
            if profit_data:
                self._render_metric_chart(profit_data[0], "Net Income Trend")
    
    def _render_metric_chart(self, metric_data: Dict, title: str):
        """Render a simple metric chart"""
        values = metric_data.get('values', {})
        
        if values:
            import plotly.graph_objects as go
            
            years = list(values.keys())
            amounts = list(values.values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years,
                y=amounts,
                mode='lines+markers',
                name=metric_data.get('name', 'Value'),
                line=dict(width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Year",
                yaxis_title="Amount",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_ratios_tab(self, data: pd.DataFrame):
        """Render financial ratios tab with manual mapping support"""
        st.header("📈 Financial Ratio Analysis")
        
        # Check if mappings exist
        if not self.state.get('metric_mappings'):
            st.warning("Please map metrics first to calculate ratios")
            
            # Show mapping options based on mode
            if self.config.get('app.display_mode') == Configuration.DisplayMode.MINIMAL or not self.config.get('ai.enabled', True):
                # Show manual mapping interface
                manual_mapper = ManualMappingInterface(data)
                mappings = manual_mapper.render()
                
                if st.button("✅ Apply Mappings", type="primary", key="apply_manual_mappings"):
                    self.state.set('metric_mappings', mappings)
                    st.success(f"Applied {len(mappings)} mappings!")
                    st.rerun()
            else:
                # Show both AI and manual options
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🤖 Auto-map with AI", type="primary", key="ai_map_ratios"):
                        self._perform_ai_mapping(data)
                
                with col2:
                    if st.button("✏️ Manual Mapping", key="manual_map_ratios"):
                        st.session_state['show_manual_mapping'] = True
                
                # Show manual mapping if requested
                if st.session_state.get('show_manual_mapping', False):
                    with st.expander("Manual Mapping", expanded=True):
                        manual_mapper = ManualMappingInterface(data)
                        mappings = manual_mapper.render()
                        
                        if st.button("✅ Apply Manual Mappings", type="primary", key="apply_manual_mappings_2"):
                            self.state.set('metric_mappings', mappings)
                            st.success(f"Applied {len(mappings)} mappings!")
                            st.session_state['show_manual_mapping'] = False
                            st.rerun()
            
            return
        
        # Apply mappings and calculate ratios
        mappings = self.state.get('metric_mappings')
        mapped_df = data.rename(index=mappings)
        
        # Calculate ratios
        with st.spinner("Calculating ratios..."):
            analysis = self.components['analyzer'].analyze_financial_statements(mapped_df)
            ratios = analysis.get('ratios', {})
        
        if not ratios:
            st.error("Unable to calculate ratios. Please check your mappings.")
            if st.button("🔄 Re-map Metrics"):
                self.state.set('metric_mappings', None)
                st.rerun()
            return
        
        # Display ratios by category
        for category, ratio_df in ratios.items():
            if isinstance(ratio_df, pd.DataFrame) and not ratio_df.empty:
                st.subheader(f"{category} Ratios")
                
                # Format based on number preference
                try:
                    st.dataframe(
                        ratio_df.style.format("{:,.2f}", na_rep="-"),
                        use_container_width=True
                    )
                except Exception as e:
                    self.logger.error(f"Error formatting ratios: {e}")
                    st.dataframe(ratio_df, use_container_width=True)
                
                # Visualization
                if st.checkbox(f"Visualize {category}", key=f"viz_{category}"):
                    metrics_to_plot = st.multiselect(
                        f"Select {category} metrics:",
                        ratio_df.index.tolist(),
                        default=ratio_df.index[:2].tolist() if len(ratio_df.index) >= 2 else ratio_df.index.tolist(),
                        key=f"select_{category}"
                    )
                    
                    if metrics_to_plot:
                        fig = go.Figure()
                        
                        for metric in metrics_to_plot:
                            fig.add_trace(go.Scatter(
                                x=ratio_df.columns,
                                y=ratio_df.loc[metric],
                                mode='lines+markers',
                                name=metric,
                                line=dict(width=2),
                                marker=dict(size=8)
                            ))
                        
                        fig.update_layout(
                            title=f"{category} Ratios Trend",
                            xaxis_title="Year",
                            yaxis_title="Value",
                            hovermode='x unified',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_trends_tab(self, data: pd.DataFrame):
        """Render trends and analysis tab"""
        st.header("📉 Trend Analysis")
        
        # Get trend analysis
        analysis = self.components['analyzer'].analyze_financial_statements(data)
        trends = analysis.get('trends', {})
        
        if not trends or 'error' in trends:
            st.error("Insufficient data for trend analysis. Need at least 2 years of data.")
            return
        
        # Trend summary
        st.subheader("Trend Summary")
        
        # Convert trends to DataFrame for display
        trend_data = []
        for metric, trend_info in trends.items():
            if isinstance(trend_info, dict) and 'direction' in trend_info:
                trend_data.append({
                    'Metric': metric,
                    'Direction': trend_info['direction'],
                    'CAGR %': trend_info.get('cagr', None),
                    'Volatility %': trend_info.get('volatility', None),
                    'R²': trend_info.get('r_squared', None)
                })
        
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            
            # Format and display
            st.dataframe(
                trend_df.style.format({
                    'CAGR %': '{:.1f}',
                    'Volatility %': '{:.1f}',
                    'R²': '{:.3f}'
                }, na_rep='-').background_gradient(subset=['CAGR %'], cmap='RdYlGn'),
                use_container_width=True
            )
        
        # Visualization
        st.subheader("Trend Visualization")
        
        # Select metrics to visualize
        numeric_metrics = data.select_dtypes(include=[np.number]).index.tolist()
        selected_metrics = st.multiselect(
            "Select metrics to visualize:",
            numeric_metrics,
            default=numeric_metrics[:3] if len(numeric_metrics) >= 3 else numeric_metrics
        )
        
        if selected_metrics:
            fig = go.Figure()
            
            for metric in selected_metrics:
                # Add actual values
                fig.add_trace(go.Scatter(
                    x=data.columns,
                    y=data.loc[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
                
                # Add trend line
                if metric in trends:
                    trend_info = trends[metric]
                    if 'slope' in trend_info and 'intercept' in trend_info:
                        x_numeric = np.arange(len(data.columns))
                        y_trend = trend_info['slope'] * x_numeric + trend_info['intercept']
                        
                        fig.add_trace(go.Scatter(
                            x=data.columns,
                            y=y_trend,
                            mode='lines',
                            name=f"{metric} (Trend)",
                            line=dict(width=2, dash='dash'),
                            opacity=0.7
                        ))
            
            fig.update_layout(
                title="Metric Trends with Linear Regression",
                xaxis_title="Year",
                yaxis_title="Value",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_penman_nissim_tab(self, data: pd.DataFrame):
        """Render Penman-Nissim analysis tab"""
        st.header("🎯 Penman-Nissim Analysis")
        
        # Check if mappings exist
        if not self.state.get('pn_mappings'):
            st.info("Configure Penman-Nissim mappings to proceed")
            
            # Mapping interface
            with st.expander("⚙️ Configure P-N Mappings", expanded=True):
                available_metrics = [''] + [str(m) for m in data.index.tolist()]
                
                # Essential mappings
                col1, col2, col3 = st.columns(3)
                
                mappings = {}
                
                with col1:
                    mappings['Total Assets'] = st.selectbox(
                        "Total Assets", 
                        available_metrics,
                        key="pn_total_assets"
                    )
                    mappings['Total Liabilities'] = st.selectbox(
                        "Total Liabilities",
                        available_metrics,
                        key="pn_total_liabilities"
                    )
                    mappings['Total Equity'] = st.selectbox(
                        "Total Equity",
                        available_metrics,
                        key="pn_total_equity"
                    )
                
                with col2:
                    mappings['Revenue'] = st.selectbox(
                        "Revenue",
                        available_metrics,
                        key="pn_revenue"
                    )
                    mappings['Operating Income'] = st.selectbox(
                        "Operating Income/EBIT",
                        available_metrics,
                        key="pn_operating_income"
                    )
                    mappings['Net Income'] = st.selectbox(
                        "Net Income",
                        available_metrics,
                        key="pn_net_income"
                    )
                
                with col3:
                    mappings['Operating Cash Flow'] = st.selectbox(
                        "Operating Cash Flow",
                        available_metrics,
                        key="pn_ocf"
                    )
                    mappings['Interest Expense'] = st.selectbox(
                        "Interest Expense",
                        available_metrics,
                        key="pn_interest"
                    )
                    mappings['Tax Expense'] = st.selectbox(
                        "Tax Expense",
                        available_metrics,
                        key="pn_tax"
                    )
                
                # Remove empty mappings
                mappings = {k: v for k, v in mappings.items() if v}
                
                if st.button("Apply P-N Mappings", type="primary"):
                    if len(mappings) >= 6:
                        self.state.set('pn_mappings', mappings)
                        st.success("Mappings applied successfully!")
                        st.rerun()
                    else:
                        st.error("Please provide at least 6 mappings for analysis")
            
            return
        
        # Run analysis
        if st.button("🚀 Run Penman-Nissim Analysis", type="primary"):
            mappings = self.state.get('pn_mappings')
            
            with st.spinner("Running Penman-Nissim analysis..."):
                try:
                    analyzer = EnhancedPenmanNissimAnalyzer(data, mappings)
                    results = analyzer.calculate_all()
                    
                    if 'error' in results:
                        st.error(f"Analysis failed: {results['error']}")
                        return
                    
                    self.state.set('pn_results', results)
                    st.success("Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    if self.config.get('app.debug', False):
                        st.exception(e)
                    return
        
        # Display results
        if self.state.get('pn_results'):
            results = self.state.get('pn_results')
            
            # Key metrics
            st.subheader("Key Metrics")
            
            if 'ratios' in results:
                ratios_df = results['ratios']
                
                # Display key ratios
                col1, col2, col3, col4 = st.columns(4)
                
                key_ratios = [
                    ('Return on Net Operating Assets (RNOA) %', 'RNOA'),
                    ('Financial Leverage (FLEV)', 'FLEV'),
                    ('Net Borrowing Cost (NBC) %', 'NBC'),
                    ('Operating Profit Margin (OPM) %', 'OPM')
                ]
                
                for i, (ratio_name, short_name) in enumerate(key_ratios):
                    if ratio_name in ratios_df.index:
                        col = [col1, col2, col3, col4][i]
                        with col:
                            latest_value = ratios_df.loc[ratio_name].iloc[-1]
                            self.ui_factory.create_metric_card(
                                short_name,
                                f"{latest_value:.2f}{'%' if '%' in ratio_name else 'x'}",
                                help=self._get_pn_ratio_help(short_name)
                            )
            
            # Reformulated statements
            col1, col2 = st.columns(2)
            
            with col1:
                if 'reformulated_balance_sheet' in results:
                    st.subheader("Reformulated Balance Sheet")
                    st.dataframe(
                        results['reformulated_balance_sheet'].style.format("{:,.0f}"),
                        use_container_width=True
                    )
            
            with col2:
                if 'reformulated_income_statement' in results:
                    st.subheader("Reformulated Income Statement")
                    st.dataframe(
                        results['reformulated_income_statement'].style.format("{:,.0f}"),
                        use_container_width=True
                    )
            
            # Free Cash Flow
            if 'free_cash_flow' in results:
                st.subheader("Free Cash Flow Analysis")
                fcf_df = results['free_cash_flow']
                
                # Visualization
                fig = go.Figure()
                
                for metric in fcf_df.index:
                    fig.add_trace(go.Bar(
                        x=fcf_df.columns,
                        y=fcf_df.loc[metric],
                        name=metric
                    ))
                
                fig.update_layout(
                    title="Cash Flow Components",
                    xaxis_title="Year",
                    yaxis_title="Amount",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _get_pn_ratio_help(self, ratio: str) -> str:
        """Get help text for Penman-Nissim ratios"""
        help_texts = {
            'RNOA': "Return on Net Operating Assets - measures operating efficiency",
            'FLEV': "Financial Leverage - ratio of financial obligations to equity",
            'NBC': "Net Borrowing Cost - effective interest rate on net debt",
            'OPM': "Operating Profit Margin - operating profitability",
            'NOAT': "Net Operating Asset Turnover - efficiency in using assets"
        }
        return help_texts.get(ratio, "Financial ratio")
    
    def _render_industry_tab(self, data: pd.DataFrame):
        """Render industry comparison tab"""
        st.header("🏭 Industry Comparison")
        
        # Industry selection
        industries = [
            "Technology", "Healthcare", "Financial Services", "Retail",
            "Manufacturing", "Energy", "Real Estate", "Consumer Goods"
        ]
        
        selected_industry = st.selectbox(
            "Select Industry for Comparison",
            industries,
            key="industry_selection"
        )
        
        # Mock industry benchmarks
        industry_benchmarks = {
            'Technology': {
                'Current Ratio': 2.5,
                'Quick Ratio': 2.2,
                'ROE %': 22.5,
                'ROA %': 12.8,
                'Net Profit Margin %': 15.2,
                'Debt to Equity': 0.45
            },
            'Manufacturing': {
                'Current Ratio': 1.8,
                'Quick Ratio': 1.2,
                'ROE %': 15.5,
                'ROA %': 7.2,
                'Net Profit Margin %': 8.5,
                'Debt to Equity': 0.85
            }
        }
        
        # Get company's latest ratios
        if self.state.get('metric_mappings'):
            mappings = self.state.get('metric_mappings')
            mapped_df = data.rename(index=mappings)
            
            analysis = self.components['analyzer'].analyze_financial_statements(mapped_df)
            company_ratios = {}
            
            for category, ratio_df in analysis.get('ratios', {}).items():
                if isinstance(ratio_df, pd.DataFrame) and not ratio_df.empty:
                    for ratio in ratio_df.index:
                        if len(ratio_df.columns) > 0:
                            company_ratios[ratio] = ratio_df.loc[ratio].iloc[-1]
        else:
            company_ratios = {}
        
        # Comparison visualization
        if selected_industry in industry_benchmarks and company_ratios:
            st.subheader("Performance vs Industry")
            
            benchmarks = industry_benchmarks[selected_industry]
            
            # Prepare data for radar chart
            categories = []
            company_values = []
            industry_values = []
            
            for metric, industry_value in benchmarks.items():
                if metric in company_ratios:
                    categories.append(metric)
                    company_values.append(company_ratios[metric])
                    industry_values.append(industry_value)
            
            if categories:
                # Create radar chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=company_values,
                    theta=categories,
                    fill='toself',
                    name='Your Company'
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=industry_values,
                    theta=categories,
                    fill='toself',
                    name=f'{selected_industry} Average'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(max(company_values), max(industry_values)) * 1.2]
                        )),
                    showlegend=True,
                    title=f"Company vs {selected_industry} Industry Benchmarks"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed comparison table
                st.subheader("Detailed Comparison")
                
                comparison_data = []
                for metric, industry_value in benchmarks.items():
                    if metric in company_ratios:
                        company_value = company_ratios[metric]
                        difference = company_value - industry_value
                        performance = "Above" if difference > 0 else "Below"
                        
                        comparison_data.append({
                            'Metric': metric,
                            'Your Company': f"{company_value:.2f}",
                            'Industry Avg': f"{industry_value:.2f}",
                            'Difference': f"{difference:+.2f}",
                            'Performance': performance
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Style the dataframe
                def highlight_performance(row):
                    if row['Performance'] == 'Above':
                        return ['background-color: #d4edda'] * len(row)
                    else:
                        return ['background-color: #f8d7da'] * len(row)
                
                st.dataframe(
                    comparison_df.style.apply(highlight_performance, axis=1),
                    use_container_width=True
                )
        else:
            st.info("Please calculate financial ratios first to enable industry comparison")
    
    def _render_data_explorer_tab(self, data: pd.DataFrame):
        """Render data explorer tab"""
        st.header("🔍 Data Explorer")
        
        # Data summary
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", len(data))
        with col2:
            st.metric("Total Columns", len(data.columns))
        with col3:
            st.metric("Data Points", data.size)
        
        # Data preview
        st.subheader("Data Preview")
        
        # Display options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_all = st.checkbox("Show all data", value=False)
        with col2:
            transpose = st.checkbox("Transpose", value=False)
        with col3:
            decimal_places = st.number_input("Decimal places", 0, 4, 2)
        
        # Display data
        display_df = data.T if transpose else data
        
        if not show_all:
            display_df = display_df.head(20)
        
        # Format numeric columns
        format_dict = {}
        for col in display_df.select_dtypes(include=[np.number]).columns:
            format_dict[col] = f"{{:,.{decimal_places}f}}"
        
        st.dataframe(
            display_df.style.format(format_dict, na_rep="-"),
            use_container_width=True
        )
        
        # Data filtering
        st.subheader("Data Filtering")
        
        # Metric filter
        selected_metrics = st.multiselect(
            "Select metrics to display:",
            data.index.tolist(),
            default=data.index[:10].tolist() if len(data) >= 10 else data.index.tolist()
        )
        
        # Year filter
        selected_years = st.multiselect(
            "Select years to display:",
            data.columns.tolist(),
            default=data.columns.tolist()
        )
        
        if selected_metrics and selected_years:
            filtered_df = data.loc[selected_metrics, selected_years]
            
            st.subheader("Filtered Data")
            st.dataframe(
                filtered_df.style.format(format_dict, na_rep="-"),
                use_container_width=True
            )
            
            # Export filtered data
            csv = filtered_df.to_csv()
            st.download_button(
                label="Download Filtered Data (CSV)",
                data=csv,
                file_name="filtered_financial_data.csv",
                mime="text/csv"
            )
    
    def _render_reports_tab(self, data: pd.DataFrame):
        """Render reports tab"""
        st.header("📄 Financial Reports")
        
        # Report type selection
        report_type = st.selectbox(
            "Select Report Type",
            [
                "Executive Summary",
                "Ratio Analysis Report",
                "Trend Analysis Report",
                "Penman-Nissim Report",
                "Complete Financial Analysis"
            ],
            key="report_type"
        )
        
        # Report configuration
        col1, col2 = st.columns(2)
        
        with col1:
            include_charts = st.checkbox("Include Charts", value=True)
            include_insights = st.checkbox("Include Insights", value=True)
        
        with col2:
            report_format = st.selectbox("Export Format", ["PDF", "Excel", "Word", "HTML"])
            include_raw_data = st.checkbox("Include Raw Data", value=False)
        
        # Generate report button
        if st.button("Generate Report", type="primary", key="generate_report"):
            with st.spinner(f"Generating {report_type}..."):
                # Generate report content based on type
                report_content = self._generate_report(data, report_type, include_charts, include_insights)
                
                st.success("Report generated successfully!")
                
                # Display report preview
                st.subheader("Report Preview")
                st.markdown(report_content['summary'])
                
                # For now, provide markdown download
                # In production, you would convert to actual PDF/Word/Excel
                st.download_button(
                    label=f"Download Report",
                    data=report_content['full_report'],
                    file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
    
    def _generate_report(self, data: pd.DataFrame, report_type: str, 
                        include_charts: bool, include_insights: bool) -> Dict[str, str]:
        """Generate report content"""
        # Get all analysis results
        analysis = self.components['analyzer'].analyze_financial_statements(data)
        
        # Build report
        report_lines = [
            f"# {report_type}",
            f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Company:** {self.state.get('company_name', 'Financial Analysis')}",
            "\n---\n"
        ]
        
        # Executive Summary
        report_lines.extend([
            "## Executive Summary\n",
            f"- **Total Metrics Analyzed:** {analysis['summary']['total_metrics']}",
            f"- **Period Covered:** {analysis['summary']['year_range']}",
            f"- **Data Quality Score:** {analysis['quality_score']:.1f}%",
            f"- **Data Completeness:** {analysis['summary']['completeness']:.1f}%",
            "\n"
        ])
        
        # Key Insights
        if include_insights and analysis.get('insights'):
            report_lines.extend([
                "## Key Insights\n"
            ])
            for insight in analysis['insights']:
                report_lines.append(f"- {insight}")
            report_lines.append("\n")
        
        # Financial Ratios
        if 'ratios' in analysis and report_type in ["Ratio Analysis Report", "Complete Financial Analysis"]:
            report_lines.extend([
                "## Financial Ratios\n"
            ])
            
            for category, ratio_df in analysis['ratios'].items():
                if isinstance(ratio_df, pd.DataFrame) and not ratio_df.empty:
                    report_lines.append(f"\n### {category} Ratios\n")
                    report_lines.append(ratio_df.to_markdown())
                    report_lines.append("\n")
        
        # Trend Analysis
        if 'trends' in analysis and report_type in ["Trend Analysis Report", "Complete Financial Analysis"]:
            report_lines.extend([
                "## Trend Analysis\n"
            ])
            
            significant_trends = []
            for metric, trend in analysis['trends'].items():
                if isinstance(trend, dict) and trend.get('cagr') is not None:
                    significant_trends.append({
                        'Metric': metric,
                        'Direction': trend['direction'],
                        'CAGR': trend['cagr'],
                        'R-squared': trend.get('r_squared', 0)
                    })
            
            if significant_trends:
                trend_df = pd.DataFrame(significant_trends)
                report_lines.append(trend_df.to_markdown())
                report_lines.append("\n")
        
        # Summary for preview
        summary = "\n".join(report_lines[:20]) + "\n\n*[Report continues...]*"
        
        return {
            'summary': summary,
            'full_report': "\n".join(report_lines)
        }
    
    def _perform_ai_mapping(self, data: pd.DataFrame):
        """Perform AI mapping for the data"""
        source_metrics = data.index.tolist()
        
        with st.spinner("Performing AI-powered metric mapping..."):
            try:
                result = self.components['mapper'].map_metrics(source_metrics)
                
                if result['mappings']:
                    self.state.set('metric_mappings', result['mappings'])
                    self.state.set('ai_mapping_result', result)
                    
                    st.success(f"✅ Successfully mapped {len(result['mappings'])} out of {len(source_metrics)} metrics")
                    
                    # Show mapping summary
                    with st.expander("View Mapping Details", expanded=True):
                        mapping_df = pd.DataFrame([
                            {
                                'Source': source,
                                'Target': target,
                                'Confidence': f"{result['confidence_scores'].get(source, 0):.2%}"
                            }
                            for source, target in result['mappings'].items()
                        ])
                        st.dataframe(mapping_df, use_container_width=True)
                    
                    if result['unmapped_metrics']:
                        st.warning(f"⚠️ {len(result['unmapped_metrics'])} metrics could not be mapped automatically")
                        with st.expander("Unmapped Metrics"):
                            st.write(result['unmapped_metrics'])
                else:
                    st.error("Could not map any metrics. Please use manual mapping.")
                    
            except Exception as e:
                self.logger.error(f"AI mapping failed: {e}")
                st.error("AI mapping failed. Please use manual mapping instead.")
    
    def _process_uploaded_files(self, files: List[UploadedFile]):
        """Process uploaded files"""
        try:
            # Process each file
            all_data = []
            
            for file in files:
                with ErrorContext(f"Processing {file.name}", self.logger):
                    # Read and parse file based on type
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file, index_col=0)
                    elif file.name.endswith(('.xls', '.xlsx')):
                        df = pd.read_excel(file, index_col=0)
                    else:
                        # HTML or other formats
                        if CORE_COMPONENTS_AVAILABLE:
                            content = file.read()
                            result = parse_html_content(content, file.name)
                            if result:
                                df = result['statement']
                            else:
                                df = None
                        else:
                            st.warning(f"Cannot parse {file.name} - core components not available")
                            df = None
                    
                    if df is not None:
                        all_data.append(df)
            
            if all_data:
                # Merge data if multiple files
                if len(all_data) == 1:
                    merged_data = all_data[0]
                else:
                    # Simple merge by columns
                    merged_data = pd.concat(all_data, axis=1)
                    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]
                
                # Process and validate
                processed_data, validation = self.components['processor'].process(merged_data)
                
                if validation.is_valid:
                    self.state.set('analysis_data', processed_data)
                    self.state.set('company_name', files[0].name.split('.')[0])
                    st.success("Files processed successfully!")
                    
                    # Auto-map if AI is enabled
                    if self.config.get('ai.enabled', True) and self.config.get('app.display_mode') != Configuration.DisplayMode.MINIMAL:
                        self._perform_ai_mapping(processed_data)
                    
                    st.rerun()
                else:
                    st.error("Validation failed:")
                    for error in validation.errors:
                        st.error(f"- {error}")
                    for warning in validation.warnings:
                        st.warning(f"- {warning}")
            else:
                st.error("No valid data found in uploaded files")
                
        except Exception as e:
            self.logger.error(f"Error processing files: {e}")
            st.error(f"Error processing files: {str(e)}")
            
            if self.config.get('app.debug', False):
                st.exception(e)
    
    def _load_sample_data(self, sample_name: str):
        """Load sample data"""
        try:
            if "Indian Tech" in sample_name:
                data = self.sample_generator.generate_indian_tech_company()
                company_name = "Indian Tech Company Ltd."
            elif "US Manufacturing" in sample_name:
                data = self.sample_generator.generate_us_manufacturing()
                company_name = "US Manufacturing Corp."
            else:
                # Default to Indian Tech
                data = self.sample_generator.generate_indian_tech_company()
                company_name = "Sample Company"
            
            # Process and validate
            processed_data, validation = self.components['processor'].process(data)
            
            if validation.is_valid:
                self.state.set('analysis_data', processed_data)
                self.state.set('company_name', company_name)
                st.success(f"Loaded sample data: {company_name}")
                
                # Auto-map if AI is enabled
                if self.config.get('ai.enabled', True) and self.config.get('app.display_mode') != Configuration.DisplayMode.MINIMAL:
                    self._perform_ai_mapping(processed_data)
                
                st.rerun()
            else:
                st.error("Sample data validation failed")
                
        except Exception as e:
            self.logger.error(f"Error loading sample data: {e}")
            st.error(f"Error loading sample data: {str(e)}")
    
    def _clear_all_caches(self):
        """Clear all caches"""
        # Clear component caches
        if 'mapper' in self.components:
            self.components['mapper'].embeddings_cache.clear()
        
        if 'analyzer' in self.components:
            self.components['analyzer'].cache.clear()
        
        # Clear streamlit cache
        st.cache_data.clear()
        st.cache_resource.clear()
        
        self.logger.info("All caches cleared")
    
    def _reset_configuration(self):
        """Reset configuration to defaults"""
        self.config = Configuration()
        self.state = StateManager("financial_analytics")
        self.logger.info("Configuration reset to defaults")

# --- 21. Application Entry Point ---
def main():
    """Main application entry point"""
    try:
        app = FinancialAnalyticsPlatform()
        app.run()
    except Exception as e:
        logger = LoggerFactory.get_logger('main')
        logger.critical(f"Fatal error: {e}", exc_info=True)
        
        st.error("A critical error occurred. Please refresh the page.")
        st.exception(e)

if __name__ == "__main__":
    main()
