# elite_financial_analytics_platform_v2.py
# Enterprise-Grade Financial Analytics Platform with Advanced Architecture

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

# Configure logging with rotation
from logging.handlers import RotatingFileHandler

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
            'version': '2.0.0',
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
            # Conditional import
            from sentence_transformers import SentenceTransformer
            
            model_name = self.config.get('ai.model_name', 'all-MiniLM-L6-v2')
            self.model = SentenceTransformer(model_name)
            self._logger.info(f"Loaded AI model: {model_name}")
            
            # Pre-compute standard embeddings
            self._precompute_standard_embeddings()
            
        except ImportError:
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

# --- 16. UI Components Factory ---
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

# --- 17. Main Application Class ---
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
    
    # ... Continue with other tab implementations ...
    
    def _process_uploaded_files(self, files: List[UploadedFile]):
        """Process uploaded files"""
        try:
            # Process each file
            all_data = []
            
            for file in files:
                with ErrorContext(f"Processing {file.name}", self.logger):
                    # Read and parse file based on type
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file)
                    elif file.name.endswith(('.xls', '.xlsx')):
                        df = pd.read_excel(file)
                    else:
                        # HTML or other formats
                        content = file.read()
                        # Parse HTML content (implement based on your needs)
                        df = self._parse_html_content(content)
                    
                    if df is not None:
                        all_data.append(df)
            
            if all_data:
                # Merge data if multiple files
                merged_data = self._merge_financial_data(all_data)
                
                # Process and validate
                processed_data, validation = self.components['processor'].process(merged_data)
                
                if validation.is_valid:
                    self.state.set('analysis_data', processed_data)
                    st.success("Files processed successfully!")
                    st.rerun()
                else:
                    st.error("Validation failed:")
                    for error in validation.errors:
                        st.error(f"- {error}")
            else:
                st.error("No valid data found in uploaded files")
                
        except Exception as e:
            self.logger.error(f"Error processing files: {e}")
            st.error(f"Error processing files: {str(e)}")
            
            if self.config.get('app.debug', False):
                st.exception(e)
    
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

# --- 18. Application Entry Point ---
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
