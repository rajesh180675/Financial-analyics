# elite_financial_analytics_platform_enhanced.py
# Enterprise-Grade Financial Analytics Platform - Enhanced Version with Robust Kaggle Integration

# --- 1. Core Imports and Setup ---
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import asyncio
import concurrent.futures
import functools
import hashlib
import importlib
import inspect
import io
import json
import logging
import os
import pickle
import re
import resource
import sys
import threading
import time
import traceback
import warnings
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict, deque, OrderedDict
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
from functools import lru_cache
import queue
import gc
import statistics

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
import zipfile
import tempfile
import shutil
import uuid

# Optional imports
try:
    import py7zr
    SEVEN_ZIP_AVAILABLE = True
except ImportError:
    SEVEN_ZIP_AVAILABLE = False
    py7zr = None

# Configure logging
warnings.filterwarnings('ignore')
from logging.handlers import RotatingFileHandler

# --- Constants ---
YEAR_REGEX = re.compile(r'(20\d{2}|19\d{2}|FY\s?20\d{2}|FY\s?19\d{2})')
MAX_FILE_SIZE_MB = 50
CACHE_TTL_SECONDS = 3600
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


# --- Enhanced Lazy Loading System ---
class LazyLoader:
    """Lazy loading for heavy modules with caching"""
    _cache = {}
    _lock = threading.Lock()

    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module = None

    def __getattr__(self, attr):
        if self._module is None:
            with self._lock:
                if self._module is None:
                    try:
                        if self.module_name not in self._cache:
                            self._cache[self.module_name] = importlib.import_module(self.module_name)
                        self._module = self._cache[self.module_name]
                    except ImportError:
                        raise ImportError(f"Optional module '{self.module_name}' not installed")
        return getattr(self._module, attr)


# Check module availability
def check_module_available(module_name: str) -> bool:
    """Check if a module is available"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# Lazy load optional dependencies
sentence_transformers = LazyLoader('sentence_transformers')
psutil = LazyLoader('psutil')
bs4 = LazyLoader('bs4')

# Check availability
SENTENCE_TRANSFORMER_AVAILABLE = check_module_available('sentence_transformers')
BEAUTIFULSOUP_AVAILABLE = check_module_available('bs4')
PSUTIL_AVAILABLE = check_module_available('psutil')

# --- Import Core Components ---
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
    CORE_REQUIRED_METRICS = {}
    CorePenmanNissim = None


# --- 2. Thread-Safe State Management (Consolidated) ---
class ThreadSafeState:
    """Thread-safe state management for Streamlit"""
    _lock = threading.RLock()
    _state_locks = {}

    @classmethod
    @contextmanager
    def lock(cls, key: Optional[str] = None):
        """Context manager for thread-safe state access"""
        if key:
            if key not in cls._state_locks:
                with cls._lock:
                    if key not in cls._state_locks:
                        cls._state_locks[key] = threading.RLock()
            lock = cls._state_locks[key]
        else:
            lock = cls._lock

        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Thread-safe get from session state"""
        with ThreadSafeState.lock(key):
            return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any):
        """Thread-safe set in session state"""
        with ThreadSafeState.lock(key):
            st.session_state[key] = value

    @staticmethod
    def update(updates: Dict[str, Any]):
        """Thread-safe batch update"""
        with ThreadSafeState.lock():
            for key, value in updates.items():
                st.session_state[key] = value

    @staticmethod
    def delete(key: str):
        """Thread-safe delete from session state"""
        with ThreadSafeState.lock(key):
            if key in st.session_state:
                del st.session_state[key]


# Single alias for consistency
SimpleState = ThreadSafeState


# --- 3. Enhanced Performance Monitoring System ---
class PerformanceMonitor:
    """Monitor and track performance metrics with API tracking"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self._lock = threading.Lock()
        self.logger = None
        self.api_metrics = defaultdict(lambda: {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'total_time': 0,
            'response_times': deque(maxlen=100),
            'error_types': defaultdict(int)
        })

    def _get_logger(self):
        """Lazy initialize logger"""
        if self.logger is None:
            self.logger = LoggerFactory.get_logger('PerformanceMonitor')
        return self.logger

    @contextmanager
    def measure(self, operation: str):
        """Measure operation performance"""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            memory_delta = self._get_memory_usage() - start_memory

            with self._lock:
                self.metrics[operation].append({
                    'duration': elapsed_time,
                    'memory_delta': memory_delta,
                    'timestamp': datetime.now()
                })

            if elapsed_time > 1.0:
                self._get_logger().warning(f"Slow operation '{operation}': {elapsed_time:.2f}s")

    def track_api_call(self, endpoint: str, success: bool, duration: float, error: Optional[str] = None):
        """Track API call metrics"""
        with self._lock:
            metrics = self.api_metrics[endpoint]
            metrics['requests'] += 1
            if success:
                metrics['successes'] += 1
            else:
                metrics['failures'] += 1
                if error:
                    metrics['error_types'][error] += 1
            metrics['total_time'] += duration
            metrics['response_times'].append(duration)

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        if PSUTIL_AVAILABLE:
            try:
                import psutil
                return psutil.Process().memory_info().rss
            except Exception:
                return 0
        return 0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for optimization"""
        with self._lock:
            summary = {}
            for operation, measurements in self.metrics.items():
                if measurements:
                    durations = [m['duration'] for m in measurements]
                    summary[operation] = {
                        'avg_duration': np.mean(durations),
                        'max_duration': np.max(durations),
                        'min_duration': np.min(durations),
                        'total_calls': len(measurements),
                        'total_time': sum(durations)
                    }
            return summary

    def get_api_summary(self) -> Dict[str, Any]:
        """Get API performance summary"""
        with self._lock:
            summary = {}
            for endpoint, metrics in self.api_metrics.items():
                if metrics['requests'] > 0:
                    response_times = list(metrics['response_times'])
                    summary[endpoint] = {
                        'total_requests': metrics['requests'],
                        'success_rate': metrics['successes'] / metrics['requests'],
                        'avg_response_time': statistics.mean(response_times) if response_times else 0,
                        'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
                        'error_distribution': dict(metrics['error_types'])
                    }
            return summary

    def clear_metrics(self):
        """Clear all metrics"""
        with self._lock:
            self.metrics.clear()
            self.api_metrics.clear()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# --- 4. Enhanced Logging Configuration ---
class LoggerFactory:
    """Factory for creating configured loggers with context"""

    _loggers = {}
    _lock = threading.Lock()
    _log_dir = Path("logs")
    _initialized = False

    @classmethod
    def _initialize(cls):
        """Initialize logging system"""
        if not cls._initialized:
            cls._log_dir.mkdir(exist_ok=True)
            cls._initialized = True

    @classmethod
    def get_logger(cls, name: str, level: int = logging.INFO) -> logging.Logger:
        """Get or create a logger with proper configuration"""
        cls._initialize()

        with cls._lock:
            if name not in cls._loggers:
                logger = logging.getLogger(name)
                logger.setLevel(level)

                # Remove existing handlers to avoid duplicates
                logger.handlers.clear()

                # Console handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(level)

                # File handler with rotation
                file_handler = RotatingFileHandler(
                    cls._log_dir / f"{name}.log",
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


# --- 5. Enhanced Error Context and Circuit Breaker ---
class CircuitBreaker:
    """Circuit breaker pattern for API calls"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == 'open':
                if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = 'half-open'
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == 'half-open':
                    self.state = 'closed'
                self.failure_count = 0
            return result
        except self.expected_exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'

            raise e

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        with self._lock:
            return {
                'state': self.state,
                'failure_count': self.failure_count,
                'last_failure': self.last_failure_time
            }


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
                return True

            if self.fallback:
                self.logger.info(f"Executing fallback for {self.operation}")
                try:
                    self.fallback()
                except Exception as fallback_error:
                    self.logger.error(f"Fallback failed: {fallback_error}")

            self.logger.debug(f"Full traceback:\n{''.join(traceback.format_tb(exc_tb))}")

        return False


def error_boundary(fallback_return=None):
    """Decorator to add error boundary to functions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = LoggerFactory.get_logger(func.__module__)
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)

                st.error(f"An error occurred in {func.__name__}. Please try again or contact support.")

                if callable(fallback_return):
                    return fallback_return()

                return fallback_return
        return wrapper
    return decorator


# --- 6. Configuration Management ---
class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass


class Configuration:
    """Centralized configuration with validation and type safety"""

    class DisplayMode(Enum):
        FULL = auto()
        LITE = auto()
        MINIMAL = auto()

    class NumberFormat(Enum):
        INDIAN = auto()
        INTERNATIONAL = auto()

    DEFAULTS = {
        'app': {
            'version': '5.1.0',
            'name': 'Elite Financial Analytics Platform',
            'debug': False,
            'display_mode': DisplayMode.LITE,
            'max_file_size_mb': 50,
            'allowed_file_types': ['csv', 'html', 'htm', 'xls', 'xlsx', 'zip', '7z'],
            'cache_ttl_seconds': 3600,
            'max_cache_size_mb': 100,
            'enable_telemetry': True,
            'enable_collaboration': True,
            'enable_ml_features': True,
            'enable_pn_debug_logging': True,  
            'pn_log_level': 'DEBUG',  
        },
        'processing': {
            'max_workers': 4,
            'chunk_size': 10000,
            'timeout_seconds': 30,
            'memory_limit_mb': 512,
            'enable_parallel': True,
            'batch_size': 5,
        },
        'analysis': {
            'confidence_threshold': 0.6,
            'outlier_std_threshold': 3,
            'min_data_points': 3,
            'interpolation_method': 'linear',
            'number_format': NumberFormat.INDIAN,
            'enable_auto_correction': True,
        },
        'ai': {
            'enabled': True,
            'model_name': 'all-MiniLM-L6-v2',
            'batch_size': 32,
            'max_sequence_length': 512,
            'similarity_threshold': 0.6,
            'confidence_levels': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            },
            'use_kaggle_api': False,
            'kaggle_api_url': '',
            'kaggle_api_timeout': 30,
            'kaggle_api_key': '',
            'kaggle_max_retries': 3,
            'kaggle_batch_size': 50,
            'kaggle_cache_results': True,
            'kaggle_fallback_to_local': True,
            'kaggle_health_check_interval': 60,
            'kaggle_circuit_breaker_threshold': 5,
            'kaggle_circuit_breaker_timeout': 300,
            'kaggle_request_pool_size': 10,
            'kaggle_connection_pool_size': 20,
            'kaggle_enable_compression': True,
            'kaggle_max_queue_size': 100,
            'kaggle_queue_timeout': 5,
            'kaggle_enable_request_coalescing': True,
            'kaggle_coalesce_window_ms': 100,
        },
        'ui': {
            'theme': 'light',
            'animations': True,
            'auto_save': True,
            'auto_save_interval': 60,
            'show_tutorial': True,
            'enable_skeleton_loading': True,
            'show_kaggle_status': True,
            'show_api_metrics': True,
            'enable_progress_tracking': True,
        },
        'security': {
            'rate_limit_requests': 100,
            'rate_limit_window': 60,
            'max_upload_size_mb': 50,
            'enable_sanitization': True,
            'allowed_html_tags': ['table', 'tr', 'td', 'th', 'tbody', 'thead', 'p', 'div', 'span', 'br'],
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

        # AI validation - Fixed
        if self._config['ai']['use_kaggle_api'] and not self._config['ai']['kaggle_api_url']:
            self._config['ai']['use_kaggle_api'] = False

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
            for path, value in overrides.items():
                original[path] = self.get(path)
                self.set(path, value)

            yield self

        finally:
            for path, value in original.items():
                self.set(path, value)


# --- 7. Number Formatting Functions ---
def format_indian_number(value: float) -> str:
    """Format number in Indian numbering system"""
    if pd.isna(value) or value is None:
        return "-"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 10000000:  # Crores
        return f"{sign}₹ {abs_value/10000000:.2f} Cr"
    elif abs_value >= 100000:  # Lakhs
        return f"{sign}₹ {abs_value/100000:.2f} L"
    elif abs_value >= 1000:  # Thousands
        return f"{sign}₹ {abs_value/1000:.1f} K"
    else:
        return f"{sign}₹ {abs_value:.0f}"


def format_international_number(value: float) -> str:
    """Format number in international system"""
    if pd.isna(value) or value is None:
        return "-"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1000000000:  # Billions
        return f"{sign}${abs_value/1000000000:.2f}B"
    elif abs_value >= 1000000:  # Millions
        return f"{sign}${abs_value/1000000:.2f}M"
    elif abs_value >= 1000:  # Thousands
        return f"{sign}${abs_value/1000:.1f}K"
    else:
        return f"{sign}${abs_value:.0f}"


# Cached formatter selection
@lru_cache(maxsize=2)
def get_number_formatter(format_type: str) -> Callable:
    """Get cached number formatter"""
    if format_type == 'Indian':
        return format_indian_number
    else:
        return format_international_number


# --- 8. Enhanced Caching System ---
class CacheEntry:
    """Cache entry with metadata and compression support"""

    def __init__(self, value: Any, ttl: Optional[int] = None, compressed: bool = False):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = time.time()
        self.compressed = compressed

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def access(self) -> Any:
        """Access the value and update metadata"""
        self.access_count += 1
        self.last_accessed = time.time()

        if self.compressed:
            return pickle.loads(zlib.decompress(self.value))
        return self.value


class AdvancedCache:
    """Thread-safe cache with TTL, size limits, compression, and statistics"""

    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._default_ttl = default_ttl
        self._stats = defaultdict(int)
        self._logger = LoggerFactory.get_logger('Cache')
        self._compression_threshold = 100 * 1024  # 100KB

    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            return 1024

    def _compress_value(self, value: Any) -> bytes:
        """Compress large values"""
        return zlib.compress(pickle.dumps(value), level=6)

    def _evict_if_needed(self):
        """Evict entries if cache is too large"""
        current_size = sum(self._estimate_size(entry.value) for entry in self._cache.values())

        if current_size > self._max_size_bytes:
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

    def set(self, key: str, value: Any, ttl: Optional[int] = None, compress: bool = None):
        """Set value in cache with optional compression"""
        with self._lock:
            self._stats['set_calls'] += 1

            if compress is None:
                compress = self._estimate_size(value) > self._compression_threshold

            if compress:
                compressed_value = self._compress_value(value)
                entry = CacheEntry(compressed_value, ttl or self._default_ttl, compressed=True)
            else:
                entry = CacheEntry(value, ttl or self._default_ttl, compressed=False)

            self._cache[key] = entry
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


# --- 9. Resource Management ---
class ResourceManager:
    """Manage computational resources and prevent overload"""

    def __init__(self, config: Configuration):
        self.config = config
        self._semaphore = threading.Semaphore(config.get('processing.max_workers', 4))
        self._memory_limit = config.get('processing.memory_limit_mb', 512) * 1024 * 1024
        self._active_tasks = set()
        self._lock = threading.Lock()
        self._logger = LoggerFactory.get_logger('ResourceManager')
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.get('processing.max_workers', 4)
        )

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

    @contextmanager
    def acquire_worker(self, task_name: str):
        """Acquire a worker slot for processing"""
        acquired = False
        start_time = time.time()

        try:
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

    def process_batch(self, items: List[Any], process_func: Callable,
                     batch_size: Optional[int] = None) -> List[Any]:
        """Process items in batches for better performance"""
        if batch_size is None:
            batch_size = self.config.get('processing.batch_size', 5)

        results = []
        futures = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            future = self._executor.submit(process_func, batch)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.extend(result if isinstance(result, list) else [result])
            except Exception as e:
                self._logger.error(f"Batch processing error: {e}")

        return results

    def check_memory_available(self, estimated_size: int) -> bool:
        """Check if enough memory is available"""
        if PSUTIL_AVAILABLE:
            try:
                import psutil
                available = psutil.virtual_memory().available
                return available > estimated_size + self._memory_limit
            except Exception as e:
                self._logger.warning(f"Memory check failed: {e}")
                return True
        else:
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                return soft == resource.RLIM_INFINITY or estimated_size < soft * 0.5
            except:
                return True

    def get_active_tasks(self) -> List[str]:
        """Get list of active tasks"""
        with self._lock:
            return list(self._active_tasks)

    def shutdown(self):
        """Shutdown the executor"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)


# --- 10. Data Validation ---
class ValidationResult:
    """Result of validation with detailed information"""

    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.corrections: List[str] = []

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

    def add_correction(self, message: str):
        """Add correction message"""
        self.corrections.append(message)

    def merge(self, other: 'ValidationResult'):
        """Merge another validation result"""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        self.corrections.extend(other.corrections)
        self.metadata.update(other.metadata)


class DataValidator:
    """Advanced data validation with comprehensive checks and auto-correction"""

    def __init__(self, config: Configuration):
        self.config = config
        self._logger = LoggerFactory.get_logger('DataValidator')
        self.enable_auto_correction = config.get('analysis.enable_auto_correction', True)

    @error_boundary((pd.DataFrame(), ValidationResult()))
    def validate_and_correct(self, df: pd.DataFrame, context: str = "data") -> Tuple[pd.DataFrame, ValidationResult]:
        """Validate and auto-correct dataframe"""
        result = self.validate_dataframe(df, context)

        if not self.enable_auto_correction:
            return df, result

        corrected_df = df.copy()
        corrections_made = []

        # Define items that CAN be negative
        negative_allowed_keywords = [
            'cash used', 'net cash used', 'purchased of', 'purchase of',
            'expenditure', 'expense', 'cost', 'depreciation', 'amortization',
            'loss', 'deficit', 'outflow', 'payment', 'dividend',
            'comprehensive income', 'other comprehensive'
        ]

        # Define items that should ALWAYS be positive
        always_positive_keywords = [
            'total assets', 'total equity', 'revenue from operations',
            'gross revenue', 'net revenue', 'total revenue'
        ]

        # Auto-corrections for positive metrics only
        for idx in corrected_df.index:
            idx_lower = str(idx).lower()

            # Skip if this metric is allowed to be negative
            if any(keyword in idx_lower for keyword in negative_allowed_keywords):
                continue

            # Only fix if it's in the always positive list
            if any(keyword in idx_lower for keyword in always_positive_keywords):
                row_data = corrected_df.loc[idx]

                try:
                    numeric_data = pd.to_numeric(row_data, errors='coerce')
                    negative_mask = numeric_data < 0
                    if negative_mask.any():
                        corrected_df.loc[idx][negative_mask] = abs(numeric_data[negative_mask])
                        corrections_made.append(f"Converted negative values to positive in {idx}")
                except Exception as e:
                    self._logger.warning(f"Error processing {idx}: {e}")

        # Update result
        if corrections_made:
            result.add_info(f"Applied {len(corrections_made)} auto-corrections")
            result.corrections = corrections_made

        return corrected_df, result

    def _fix_accounting_equation(self, df: pd.DataFrame, corrections_made: List[str]):
        """Fix violations of accounting equation (Assets = Liabilities + Equity)"""
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
                        expected_assets = liabilities + equity
                        diff = abs(assets - expected_assets)
                        tolerance = assets * 0.01

                        if diff > tolerance:
                            df.loc[equity_rows[0], col] = assets - liabilities
                            corrections_made.append(
                                f"Adjusted equity in {col} to balance accounting equation"
                            )
                except Exception:
                    pass

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

        # Check dataset size
        if df.shape[0] > 1000000:
            result.add_warning(f"{context}: Large dataset ({df.shape[0]} rows), performance may be impacted")

        # Calculate missing values correctly for numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            total_cells = numeric_df.shape[0] * numeric_df.shape[1]
            missing_cells = numeric_df.isnull().sum().sum()
            missing_pct = (missing_cells / total_cells) * 100

            # Report missing data with appropriate severity
            if missing_pct > 50:
                result.add_warning(f"{context}: High percentage of missing values ({missing_pct:.1f}%)")

                # Identify columns with most missing values
                missing_by_col = numeric_df.isnull().sum()
                worst_cols = missing_by_col.nlargest(3)
                for col, missing in worst_cols.items():
                    col_pct = (missing / len(numeric_df)) * 100
                    if col_pct > 50:
                        result.add_info(f"Column '{col}' is {col_pct:.1f}% empty")
            elif missing_pct > 20:
                result.add_info(f"{context}: Moderate missing values ({missing_pct:.1f}%)")
        else:
            result.add_warning(f"{context}: No numeric columns found")
            missing_pct = 0

        # Check for duplicate indices
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            result.add_warning(f"{context}: {dup_count} duplicate indices found")

            # Identify which indices are duplicated
            dup_indices = df.index[df.index.duplicated(keep=False)].unique()
            for idx in dup_indices[:5]:  # Show first 5 duplicates
                count = (df.index == idx).sum()
                result.add_info(f"Index '{idx}' appears {count} times")

            if len(dup_indices) > 5:
                result.add_info(f"... and {len(dup_indices) - 5} more duplicate indices")

        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                result.add_info(f"{context}: Column '{col}' has constant value")

        # Smart negative value checking for numeric columns
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) > 0 and (series < 0).any():
                negative_count = (series < 0).sum()
                negative_pct = (negative_count / len(series)) * 100

                col_lower = str(col).lower()

                # Define patterns for items that can legitimately be negative
                negative_allowed = [
                    'cash used', 'net cash used', 'cash flow from',
                    'purchased', 'purchase of', 'expenditure', 'expense',
                    'cost', 'depreciation', 'amortization', 'loss',
                    'deficit', 'outflow', 'payment', 'dividend',
                    'comprehensive income', 'other comprehensive',
                    'tax', 'interest', 'financing activities',
                    'investing activities'
                ]

                # Define patterns for items that should always be positive
                always_positive = [
                    'total assets', 'total equity', 'revenue from operations',
                    'gross revenue', 'net revenue', 'total revenue',
                    'sales', 'total liabilities'
                ]

                # Check context from both column name and row indices
                context_allows_negative = any(keyword in col_lower for keyword in negative_allowed)

                # Also check if any row index indicates cash flow context
                if not context_allows_negative and 'cash flow' in context.lower():
                    context_allows_negative = True

                # Determine if this is a problem
                is_always_positive = any(keyword in col_lower for keyword in always_positive)

                if is_always_positive:
                    result.add_warning(f"{context}: Column '{col}' contains {negative_count} negative values ({negative_pct:.1f}%)")
                elif context_allows_negative:
                    if negative_pct > 50:
                        result.add_info(f"{context}: Column '{col}' contains {negative_count} negative values (normal for this type)")
                else:
                    # For ambiguous cases, only warn if significant
                    if negative_pct > 10:
                        result.add_info(f"{context}: Column '{col}' contains {negative_count} negative values ({negative_pct:.1f}%)")

        # Check for outliers using IQR method
        outlier_info = {}
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) > 4:  # Need at least 4 values for IQR
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1

                if IQR > 0:
                    # Use standard 1.5*IQR for initial check
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    outlier_pct = (len(outliers) / len(series)) * 100

                    if outlier_pct > 20:
                        result.add_warning(f"{context}: Column '{col}' has high outlier percentage ({outlier_pct:.1f}%)")
                        outlier_info[col] = len(outliers)
                    elif outlier_pct > 10:
                        result.add_info(f"{context}: Column '{col}' has {len(outliers)} outliers ({outlier_pct:.1f}%)")
                        outlier_info[col] = len(outliers)

        # Check data types
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            result.add_info(f"{context}: {len(non_numeric_cols)} non-numeric columns: {list(non_numeric_cols)[:3]}...")

        # Store metadata
        result.metadata['shape'] = df.shape
        result.metadata['columns'] = list(df.columns)
        result.metadata['numeric_columns'] = list(numeric_df.columns)
        result.metadata['dtypes'] = df.dtypes.to_dict()
        result.metadata['memory_usage'] = df.memory_usage(deep=True).sum()
        result.metadata['missing_percentage'] = missing_pct
        result.metadata['duplicate_indices'] = list(df.index[df.index.duplicated()].unique())
        result.metadata['outlier_columns'] = outlier_info

        return result

    def validate_financial_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate financial statement data"""
        result = self.validate_dataframe(df, "financial_data")

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
                        tolerance = assets * 0.01

                        if diff > tolerance:
                            result.add_warning(
                                f"Accounting equation imbalance in {col}: "
                                f"Assets ({assets:,.0f}) ≠ Liabilities ({liabilities:,.0f}) "
                                f"+ Equity ({equity:,.0f})"
                            )
                except Exception:
                    pass

        return result


# --- 11. Pattern Matching System ---
class PatternMatcher:
    """Advanced pattern matching for financial metrics with compiled patterns"""

    # Class-level compiled patterns for efficiency
    _compiled_patterns = None
    _lock = threading.Lock()

    def __init__(self):
        self._logger = LoggerFactory.get_logger('PatternMatcher')
        self._patterns = self._get_compiled_patterns()

    @classmethod
    def _get_compiled_patterns(cls) -> Dict[str, List[re.Pattern]]:
        """Get compiled patterns with caching"""
        if cls._compiled_patterns is None:
            with cls._lock:
                if cls._compiled_patterns is None:
                    cls._compiled_patterns = cls._build_patterns()
        return cls._compiled_patterns

    @staticmethod
    def _build_patterns() -> Dict[str, List[re.Pattern]]:
        """Build comprehensive pattern library"""
        return {
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
            'interest_expense': [
                re.compile(r'\binterest\s+expense\b', re.IGNORECASE),
                re.compile(r'\bfinance\s+cost\b', re.IGNORECASE),
                re.compile(r'\binterest\s+cost\b', re.IGNORECASE),
            ],
        }

    def find_matches(self, text: str, metric_type: str) -> List[Tuple[str, float]]:
        """Find pattern matches with confidence scores"""
        matches = []

        if metric_type not in self._patterns:
            return matches

        for pattern in self._patterns[metric_type]:
            if pattern.search(text):
                match = pattern.search(text)
                confidence = self._calculate_confidence(text, match)
                matches.append((metric_type, confidence))

        return matches

    def _calculate_confidence(self, text: str, match: re.Match) -> float:
        """Calculate confidence score for a match"""
        confidence = 0.7

        if match.group(0).lower() == text.lower():
            confidence += 0.2

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


# --- 12. Base Component Class ---
class Component(ABC):
    """Base component with lifecycle management"""

    def __init__(self, config: Configuration):
        self.config = config
        self._logger = LoggerFactory.get_logger(self.__class__.__name__)
        self._initialized = False

    def initialize(self):
        """Initialize component"""
        if not self._initialized:
            self._logger.info(f"Initializing {self.__class__.__name__}")
            with performance_monitor.measure(f"init_{self.__class__.__name__}"):
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


# --- 13. Security Module ---
class SecurityModule(Component):
    """Enhanced security with comprehensive validation"""

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._sanitizer = None
        self._rate_limiter = defaultdict(deque)
        self._blocked_ips = set()

    def _do_initialize(self):
        """Initialize security components"""
        self._allowed_tags = self.config.get('security.allowed_html_tags', [])
        self._allowed_attributes = {
            '*': ['class', 'id'],
            'table': ['border', 'cellpadding', 'cellspacing'],
        }

    @error_boundary()
    def sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deep sanitization of dataframe content"""
        sanitized = df.copy()

        for col in sanitized.select_dtypes(include=['object']).columns:
            sanitized[col] = sanitized[col].apply(
                lambda x: bleach.clean(str(x)) if pd.notna(x) else x
            )

        for col in sanitized.select_dtypes(include=[np.number]).columns:
            max_val = sanitized[col].max()
            if pd.notna(max_val) and max_val > 1e15:
                self._logger.warning(f"Extremely large values detected in {col}")

        return sanitized

    def validate_file_upload(self, file: UploadedFile) -> ValidationResult:
        """Comprehensive file validation"""
        result = ValidationResult()

        max_size = self.config.get('security.max_upload_size_mb', 50) * 1024 * 1024
        if file.size > max_size:
            result.add_error(f"File size ({file.size / 1024 / 1024:.1f}MB) exceeds limit ({max_size / 1024 / 1024}MB)")
            return result

        allowed_types = self.config.get('app.allowed_file_types', [])
        file_ext = Path(file.name).suffix.lower().lstrip('.')

        if file_ext not in allowed_types:
            result.add_error(f"File type '{file_ext}' not allowed. Allowed types: {', '.join(allowed_types)}")
            return result

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
            file.seek(0)

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

        if len(content) > 10 * 1024 * 1024:
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

    def check_rate_limit(self, identifier: str, action: str,
                        limit: Optional[int] = None, window: Optional[int] = None) -> bool:
        """Check rate limit for an action"""
        if limit is None:
            limit = self.config.get('security.rate_limit_requests', 100)
        if window is None:
            window = self.config.get('security.rate_limit_window', 60)

        key = f"{identifier}:{action}"
        now = time.time()

        self._rate_limiter[key] = deque(
            [t for t in self._rate_limiter[key] if now - t < window],
            maxlen=limit
        )

        if len(self._rate_limiter[key]) >= limit:
            self._logger.warning(f"Rate limit exceeded for {key}")
            return False

        self._rate_limiter[key].append(now)
        return True


# --- 14. Compression Handler ---
class CompressionHandler:
    """Handle compressed file extraction with proper cleanup"""

    def __init__(self, logger):
        self.logger = logger
        self.temp_dirs = []

    def __del__(self):
        """Cleanup temporary directories"""
        self.cleanup()

    def cleanup(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    self.logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                self.logger.error(f"Error cleaning up temp dir: {e}")
        self.temp_dirs.clear()

    def extract_compressed_file(self, file: UploadedFile) -> List[Tuple[str, bytes]]:
        """Extract compressed file and return list of (filename, content) tuples"""
        extracted_files = []
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)

        try:
            temp_file = temp_dir / file.name
            with open(temp_file, 'wb') as f:
                f.write(file.getbuffer())

            if file.name.lower().endswith('.zip'):
                extracted_files = self._extract_zip(temp_file, temp_dir)
            elif file.name.lower().endswith('.7z'):
                if SEVEN_ZIP_AVAILABLE:
                    extracted_files = self._extract_7z(temp_file, temp_dir)
                else:
                    st.error("7z support not available. Please install 'py7zr' package: pip install py7zr")
                    return []

            self.logger.info(f"Extracted {len(extracted_files)} files from {file.name}")

        except Exception as e:
            self.logger.error(f"Error extracting {file.name}: {e}")
            st.error(f"Error extracting compressed file: {str(e)}")
        finally:
            # Ensure cleanup happens even on error
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass

        return extracted_files

    def _extract_zip(self, zip_path: Path, temp_dir: Path) -> List[Tuple[str, bytes]]:
        """Extract ZIP file"""
        extracted = []

        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            file_list = zip_file.namelist()
            supported_extensions = ['.csv', '.html', '.htm', '.xls', '.xlsx']

            for file_name in file_list:
                if file_name.endswith('/') or file_name.startswith('.') or '/' in file_name and file_name.split('/')[-1].startswith('.'):
                    continue

                if any(file_name.lower().endswith(ext) for ext in supported_extensions):
                    try:
                        content = zip_file.read(file_name)
                        clean_name = Path(file_name).name
                        extracted.append((clean_name, content))
                        self.logger.info(f"Extracted: {clean_name}")
                    except Exception as e:
                        self.logger.error(f"Error extracting {file_name}: {e}")

        return extracted

    def _extract_7z(self, seven_zip_path: Path, temp_dir: Path) -> List[Tuple[str, bytes]]:
        """Extract 7z file"""
        extracted = []

        with py7zr.SevenZipFile(seven_zip_path, mode='r') as seven_zip:
            seven_zip.extractall(path=temp_dir)
            supported_extensions = ['.csv', '.html', '.htm', '.xls', '.xlsx']

            for extracted_file in temp_dir.rglob('*'):
                if extracted_file.is_file() and not extracted_file.name.startswith('.'):
                    if any(extracted_file.name.lower().endswith(ext) for ext in supported_extensions):
                        try:
                            with open(extracted_file, 'rb') as f:
                                content = f.read()
                            extracted.append((extracted_file.name, content))
                            self.logger.info(f"Extracted: {extracted_file.name}")
                        except Exception as e:
                            self.logger.error(f"Error reading {extracted_file}: {e}")

        return extracted


# --- 15. Data Processing Pipeline ---
class DataProcessor(Component):
    """Advanced data processing with pipeline architecture and batch support"""

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._transformers = []
        self._validators = []
        self.resource_manager = None
        self.chunk_size = config.get('processing.chunk_size', 10000)

    def _do_initialize(self):
        """Initialize processor"""
        self.resource_manager = ResourceManager(self.config)
        self._setup_pipeline()

    def _do_cleanup(self):
        """Cleanup processor resources"""
        if self.resource_manager:
            self.resource_manager.shutdown()

    def _setup_pipeline(self):
        """Setup processing pipeline"""
        self._transformers = [
            self._clean_numeric_data,
            self._normalize_indices,
            self._interpolate_missing,
            self._detect_outliers,
        ]

        validator = DataValidator(self.config)
        self._validators = [
            validator.validate_dataframe,
            validator.validate_financial_data,
        ]

    @error_boundary()
    def process(self, df: pd.DataFrame, context: str = "data") -> Tuple[pd.DataFrame, ValidationResult]:
        """Process dataframe through pipeline"""
        with performance_monitor.measure(f"process_{context}"):
            if len(df) > self.chunk_size:
                return self._process_large_dataframe(df, context)
            else:
                return self._process_standard(df, context)

    def _process_standard(self, df: pd.DataFrame, context: str) -> Tuple[pd.DataFrame, ValidationResult]:
        """Standard processing for normal-sized dataframes"""
        result = ValidationResult()
        processed_df = df.copy()

        with self.resource_manager.acquire_worker(f"process_{context}"):
            for validator in self._validators:
                validation = validator(processed_df)
                result.merge(validation)

                if not validation.is_valid:
                    self._logger.warning(f"Validation failed in {context}")
                    break

            if self.config.get('analysis.enable_auto_correction', True):
                validator = DataValidator(self.config)
                try:
                    processed_df, correction_result = validator.validate_and_correct(processed_df, context)
                    result.merge(correction_result)
                except Exception as e:
                    self._logger.error(f"Auto-correction failed: {e}")

            if result.is_valid:
                for transformer in self._transformers:
                    try:
                        processed_df = transformer(processed_df)
                    except Exception as e:
                        result.add_error(f"Transformation error: {str(e)}")
                        self._logger.error(f"Error in transformer {transformer.__name__}: {e}")
                        break

        return processed_df, result

    def _process_large_dataframe(self, df: pd.DataFrame, context: str) -> Tuple[pd.DataFrame, ValidationResult]:
        """Process large dataframes in chunks to reduce memory usage"""
        self._logger.info(f"Processing large dataframe ({len(df)} rows) in chunks")

        result = ValidationResult()
        chunks = []

        for start in range(0, len(df), self.chunk_size):
            end = min(start + self.chunk_size, len(df))
            chunk = df.iloc[start:end]

            processed_chunk, chunk_result = self._process_standard(chunk, f"{context}_chunk_{start}")
            chunks.append(processed_chunk)
            result.merge(chunk_result)

            if not chunk_result.is_valid:
                break

        if chunks:
            processed_df = pd.concat(chunks)
        else:
            processed_df = df

        return processed_df, result

    def process_batch(self, dataframes: List[pd.DataFrame]) -> List[Tuple[pd.DataFrame, ValidationResult]]:
        """Process multiple dataframes in batch for efficiency"""
        return self.resource_manager.process_batch(
            dataframes,
            lambda batch: [self.process(df) for df in batch]
        )

    def _clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric data with advanced techniques"""
        df_clean = df.copy()

        for col in df.select_dtypes(include=['object']).columns:
            converted = pd.to_numeric(df[col], errors='coerce')

            if converted.notna().sum() > len(df) * 0.5:
                df_clean[col] = converted

        return df_clean

    def _normalize_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize index names"""
        df_norm = df.copy()

        if isinstance(df.index, pd.Index):
            df_norm.index = df.index.map(lambda x: str(x).strip())

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
                    lower_bound = mean - outlier_threshold * std
                    upper_bound = mean + outlier_threshold * std

                    outliers = (series < lower_bound) | (series > upper_bound)
                    if outliers.any():
                        self._logger.info(f"Found {outliers.sum()} outliers in {col}")

                        outlier_indices = series[outliers].index.tolist()
                        SimpleState.set(f"outliers_{col}", outlier_indices)

        return df_clean


# --- 16. Financial Analysis Engine ---
class FinancialAnalysisEngine(Component):
    """Core financial analysis engine with advanced features and caching"""

    def __init__(self, config: Configuration):
        super().__init__(config)
        self.pattern_matcher = PatternMatcher()
        self.cache = AdvancedCache()
        self._analysis_cache = {}
        # Import core components if available
        if CORE_COMPONENTS_AVAILABLE:
            self.chart_generator = CoreChartGenerator()
            self.ratio_calculator = CoreRatioCalculator()
            self.industry_benchmarks = CoreIndustryBenchmarks()

    def _do_initialize(self):
        """Initialize analysis components"""
        pass

    @error_boundary({})
    def analyze_financial_statements(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive financial statement analysis with caching"""
        # Generate cache key
        cache_key = self._generate_cache_key(df)

        # Check memory cache first (fastest)
        if cache_key in self._analysis_cache:
            self._logger.info("Returning analysis from memory cache")
            return self._analysis_cache[cache_key]

        # Check persistent cache
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self._logger.info("Returning cached analysis")
            self._analysis_cache[cache_key] = cached_result
            return cached_result

        # Perform analysis
        with performance_monitor.measure("analyze_financial_statements"):
            analysis = {
                'summary': self._generate_summary(df),
                'metrics': self._extract_key_metrics(df),
                'ratios': self._calculate_ratios(df),
                'trends': self._analyze_trends(df),
                'quality_score': self._calculate_quality_score(df),
                'insights': self._generate_insights(df),
                'anomalies': self._detect_anomalies(df)
            }

            # Cache the result
            self.cache.set(cache_key, analysis, ttl=3600)
            self._analysis_cache[cache_key] = analysis

            # Limit memory cache size
            if len(self._analysis_cache) > 10:
                # Remove oldest entry
                oldest_key = next(iter(self._analysis_cache))
                del self._analysis_cache[oldest_key]

            return analysis

    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Detect anomalies in financial data with better thresholds"""
        anomalies = {
            'value_anomalies': [],
            'trend_anomalies': [],
            'ratio_anomalies': []
        }

        numeric_df = df.select_dtypes(include=[np.number])

        # Value anomalies - use IQR method instead of z-score for financial data
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) > 4:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    # Use 3*IQR for financial data (more lenient)
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR

                    anomaly_mask = (series < lower_bound) | (series > upper_bound)
                    anomaly_indices = series[anomaly_mask].index

                    for idx in anomaly_indices:
                        anomalies['value_anomalies'].append({
                            'metric': str(idx),
                            'year': str(col),
                            'value': float(series[idx]),
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        })

        # Trend anomalies - only flag extreme changes
        for idx in df.index:
            series = numeric_df.loc[idx].dropna()
            if len(series) > 2:
                # Calculate year-over-year changes
                pct_changes = series.pct_change().dropna()

                # Only flag changes > 200% or < -66%
                extreme_changes = pct_changes[(pct_changes > 2.0) | (pct_changes < -0.66)]

                for year, change in extreme_changes.items():
                    anomalies['trend_anomalies'].append({
                        'metric': str(idx),
                        'year': str(year),
                        'change_pct': float(change * 100)
                    })

        return anomalies

    def _generate_cache_key(self, df: pd.DataFrame) -> str:
        """Generate cache key for dataframe"""
        key_parts = [
            str(df.shape),
            str(df.index[:5].tolist()),
            str(df.columns[:5].tolist()),
            str(df.iloc[:5, :5].values.tolist()) if df.shape[0] >= 5 and df.shape[1] >= 5 else ""
        ]

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics with better handling"""
        summary = {
            'total_metrics': len(df),
            'years_covered': 0,
            'year_range': "N/A",
            'completeness': 0,
            'key_statistics': {}
        }

        numeric_df = df.select_dtypes(include=[np.number])

        if not numeric_df.empty:
            # Calculate actual data completeness
            total_cells = numeric_df.shape[0] * numeric_df.shape[1]
            non_null_cells = numeric_df.notna().sum().sum()
            completeness = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0

            summary.update({
                'years_covered': len(numeric_df.columns),
                'year_range': f"{numeric_df.columns[0]} - {numeric_df.columns[-1]}" if len(numeric_df.columns) > 0 else "N/A",
                'completeness': completeness,
            })

            # Only include statistics for columns with sufficient data
            for col in numeric_df.columns[-3:]:
                col_data = numeric_df[col].dropna()
                if len(col_data) > 0:
                    summary['key_statistics'][str(col)] = {
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()) if len(col_data) > 1 else 0,
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'count': len(col_data)
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
        """Calculate financial ratios with proper error handling"""
        ratios = {}

        # Use core components if available
        if CORE_COMPONENTS_AVAILABLE and hasattr(self, 'ratio_calculator'):
            ratios['Liquidity'] = self.ratio_calculator.calculate_liquidity_ratios(df)
            ratios['Profitability'] = self.ratio_calculator.calculate_profitability_ratios(df)
            ratios['Leverage'] = self.ratio_calculator.calculate_leverage_ratios(df)
            ratios['Efficiency'] = self.ratio_calculator.calculate_efficiency_ratios(df)
        else:
            # Fallback implementation
            metrics = self._extract_key_metrics(df)

            metric_values = {}
            metric_keys = [
                'current_assets', 'current_liabilities', 'total_assets', 'total_liabilities',
                'total_equity', 'inventory', 'cash', 'net_income', 'revenue',
                'cost_of_goods_sold', 'ebit', 'interest_expense', 'receivables'
            ]

            for metric_key in metric_keys:
                metric_value = self._get_metric_value(df, metrics, metric_key)
                if metric_value is not None:
                    if isinstance(metric_value, pd.DataFrame):
                        metric_value = metric_value.iloc[0]
                    metric_values[metric_key] = metric_value
                else:
                    metric_values[metric_key] = None

            def safe_divide(numerator_key, denominator_key, multiplier=1):
                """Safely divide two metrics"""
                numerator = metric_values.get(numerator_key)
                denominator = metric_values.get(denominator_key)

                if numerator is None or denominator is None:
                    return None

                try:
                    if hasattr(denominator, 'replace'):
                        safe_denom = denominator.replace(0, np.nan)
                    else:
                        safe_denom = denominator if denominator != 0 else np.nan

                    result = (numerator / safe_denom) * multiplier
                    return result
                except Exception as e:
                    self._logger.warning(f"Division error: {e}")
                    return None

            # Liquidity Ratios
            try:
                liquidity_data = {}

                current_ratio = safe_divide('current_assets', 'current_liabilities')
                if current_ratio is not None:
                    liquidity_data['Current Ratio'] = current_ratio

                if metric_values['current_assets'] is not None and metric_values['inventory'] is not None:
                    quick_assets = metric_values['current_assets'] - metric_values['inventory']
                    if metric_values['current_liabilities'] is not None:
                        quick_ratio = quick_assets / metric_values['current_liabilities'].replace(0, np.nan)
                        liquidity_data['Quick Ratio'] = quick_ratio

                cash_ratio = safe_divide('cash', 'current_liabilities')
                if cash_ratio is not None:
                    liquidity_data['Cash Ratio'] = cash_ratio

                if liquidity_data:
                    liquidity_df = pd.DataFrame(liquidity_data)
                    ratios['Liquidity'] = liquidity_df.T

            except Exception as e:
                self._logger.error(f"Error calculating liquidity ratios: {e}")

            # Profitability Ratios
            try:
                profitability_data = {}

                npm = safe_divide('net_income', 'revenue', 100)
                if npm is not None:
                    profitability_data['Net Profit Margin %'] = npm

                if metric_values['revenue'] is not None and metric_values['cost_of_goods_sold'] is not None:
                    gross_profit = metric_values['revenue'] - metric_values['cost_of_goods_sold']
                    gpm = (gross_profit / metric_values['revenue'].replace(0, np.nan)) * 100
                    profitability_data['Gross Profit Margin %'] = gpm

                roa = safe_divide('net_income', 'total_assets', 100)
                if roa is not None:
                    profitability_data['Return on Assets %'] = roa

                roe = safe_divide('net_income', 'total_equity', 100)
                if roe is not None:
                    profitability_data['Return on Equity %'] = roe

                if metric_values['ebit'] is not None and metric_values['total_assets'] is not None and metric_values['current_liabilities'] is not None:
                    capital_employed = metric_values['total_assets'] - metric_values['current_liabilities']
                    roce = (metric_values['ebit'] / capital_employed.replace(0, np.nan)) * 100
                    profitability_data['ROCE %'] = roce

                if profitability_data:
                    profitability_df = pd.DataFrame(profitability_data)
                    ratios['Profitability'] = profitability_df.T

            except Exception as e:
                self._logger.error(f"Error calculating profitability ratios: {e}")

            # Leverage Ratios
            try:
                leverage_data = {}

                de_ratio = safe_divide('total_liabilities', 'total_equity')
                if de_ratio is not None:
                    leverage_data['Debt to Equity'] = de_ratio

                debt_ratio = safe_divide('total_liabilities', 'total_assets')
                if debt_ratio is not None:
                    leverage_data['Debt Ratio'] = debt_ratio

                icr = safe_divide('ebit', 'interest_expense')
                if icr is not None:
                    leverage_data['Interest Coverage'] = icr

                if leverage_data:
                    leverage_df = pd.DataFrame(leverage_data)
                    ratios['Leverage'] = leverage_df.T

            except Exception as e:
                self._logger.error(f"Error calculating leverage ratios: {e}")

            # Efficiency Ratios
            try:
                efficiency_data = {}

                asset_turnover = safe_divide('revenue', 'total_assets')
                if asset_turnover is not None:
                    efficiency_data['Asset Turnover'] = asset_turnover

                inv_turnover = safe_divide('cost_of_goods_sold', 'inventory')
                if inv_turnover is not None:
                    efficiency_data['Inventory Turnover'] = inv_turnover

                rec_turnover = safe_divide('revenue', 'receivables')
                if rec_turnover is not None:
                    efficiency_data['Receivables Turnover'] = rec_turnover
                    efficiency_data['Days Sales Outstanding'] = 365 / rec_turnover.replace(0, np.nan)

                if efficiency_data:
                    efficiency_df = pd.DataFrame(efficiency_data)
                    ratios['Efficiency'] = efficiency_df.T

            except Exception as e:
                self._logger.error(f"Error calculating efficiency ratios: {e}")

        return ratios
        
    def _get_metric_value(self, df: pd.DataFrame, metrics: Dict, metric_type: str) -> Optional[pd.Series]:
        """Get metric value from dataframe with fallback"""
        if metric_type in metrics and metrics[metric_type]:
            best_match = max(metrics[metric_type], key=lambda x: x['confidence'])
            metric_name = best_match['name']
            
            if metric_name in df.index:
                result = df.loc[metric_name]
                if isinstance(result, pd.DataFrame):
                    self._logger.warning(f"Multiple rows found for {metric_name}, taking first")
                    result = result.iloc[0]
                return result
        
        return None

    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in financial data"""
        trends = {}
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        for idx in numeric_df.index:
            series = numeric_df.loc[idx]
            
            if isinstance(series, pd.DataFrame):
                self._logger.warning(f"Multiple rows found for {idx}, taking first")
                series = series.iloc[0]
            
            series = series.dropna()
            
            if len(series) >= 3:
                years = np.arange(len(series))
                values = series.values
                
                coefficients = np.polyfit(years, values, 1)
                slope = float(coefficients[0])
                intercept = float(coefficients[1])
                
                # CAGR calculation
                try:
                    first_value = float(series.iloc[0])
                    last_value = float(series.iloc[-1])
                    
                    if first_value > 0 and last_value > 0:
                        years_diff = len(series) - 1
                        if years_diff > 0:
                            cagr = ((last_value / first_value) ** (1 / years_diff) - 1) * 100
                        else:
                            cagr = 0
                    else:
                        cagr = 0
                        
                except Exception as e:
                    self._logger.warning(f"Could not calculate CAGR for {idx}: {e}")
                    cagr = 0
                
                # Volatility
                try:
                    volatility = float(series.pct_change().std() * 100)
                    if pd.isna(volatility):
                        volatility = 0
                except Exception:
                    volatility = 0
                
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
        
        completeness = (df.notna().sum().sum() / df.size) * 100
        scores.append(completeness)
        
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            consistency_score = 100
            
            for idx in numeric_df.index:
                positive_metrics = ['assets', 'revenue', 'equity']
                if any(keyword in str(idx).lower() for keyword in positive_metrics):
                    row_data = numeric_df.loc[idx]
                    
                    if isinstance(row_data, pd.DataFrame):
                        row_data = row_data.iloc[0]
                    
                    negative_count = int((row_data < 0).sum())
                    
                    if negative_count > 0:
                        consistency_score -= (negative_count / len(numeric_df.columns)) * 20
            
            scores.append(max(0, consistency_score))
            
            if len(numeric_df.columns) > 1:
                temporal_score = 100
                extreme_changes = 0
                
                for idx in numeric_df.index:
                    series = numeric_df.loc[idx]
                    
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[0]
                    
                    series = series.dropna()
                    
                    if len(series) > 1:
                        pct_changes = series.pct_change().dropna()
                        extreme_count = int((pct_changes.abs() > 2).sum())
                        extreme_changes += extreme_count
                
                total_changes = len(numeric_df) * (len(numeric_df.columns) - 1)
                if total_changes > 0:
                    temporal_score -= (extreme_changes / total_changes) * 50
                
                scores.append(max(0, temporal_score))
        
        return sum(scores) / len(scores) if scores else 0
    
    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable insights from analysis"""
        insights = []
        
        ratios = self._calculate_ratios(df)
        
        if 'Liquidity' in ratios and 'Current Ratio' in ratios['Liquidity'].index:
            current_ratios = ratios['Liquidity'].loc['Current Ratio'].dropna()
            if len(current_ratios) > 0:
                latest_cr = current_ratios.iloc[-1]
                if latest_cr < 1:
                    insights.append(f"⚠️ Low current ratio ({latest_cr:.2f}) indicates potential liquidity issues")
                elif latest_cr > 3:
                    insights.append(f"💡 High current ratio ({latest_cr:.2f}) suggests excess idle assets")
                
                if len(current_ratios) > 1:
                    trend = 'improving' if current_ratios.iloc[-1] > current_ratios.iloc[0] else 'declining'
                    insights.append(f"📊 Current ratio is {trend} over the period")
        
        if 'Profitability' in ratios and 'Net Profit Margin %' in ratios['Profitability'].index:
            npm = ratios['Profitability'].loc['Net Profit Margin %'].dropna()
            if len(npm) > 1:
                trend = 'improving' if npm.iloc[-1] > npm.iloc[0] else 'declining'
                insights.append(f"📊 Net profit margin is {trend} ({npm.iloc[0]:.1f}% → {npm.iloc[-1]:.1f}%)")
            
            if npm.iloc[-1] < 5:
                insights.append(f"⚠️ Low profit margin may indicate competitive pressure or cost issues")
        
        if 'Leverage' in ratios and 'Debt to Equity' in ratios['Leverage'].index:
            de_ratio = ratios['Leverage'].loc['Debt to Equity'].dropna()
            if len(de_ratio) > 0:
                latest_de = de_ratio.iloc[-1]
                if latest_de > 2:
                    insights.append(f"⚠️ High debt-to-equity ratio ({latest_de:.2f}) indicates high leverage")
                elif latest_de < 0.3:
                    insights.append(f"💡 Low leverage ({latest_de:.2f}) - consider if debt could accelerate growth")
        
        if 'Efficiency' in ratios and 'Asset Turnover' in ratios['Efficiency'].index:
            asset_turnover = ratios['Efficiency'].loc['Asset Turnover'].dropna()
            if len(asset_turnover) > 0:
                latest_at = asset_turnover.iloc[-1]
                if latest_at < 0.5:
                    insights.append(f"⚠️ Low asset turnover ({latest_at:.2f}) suggests underutilized assets")
        
        trends = self._analyze_trends(df)
        
        revenue_trends = [v for k, v in trends.items() if 'revenue' in k.lower()]
        if revenue_trends and revenue_trends[0].get('cagr') is not None:
            cagr = revenue_trends[0]['cagr']
            if cagr > 20:
                insights.append(f"🚀 Strong revenue growth (CAGR: {cagr:.1f}%)")
            elif cagr < 0:
                insights.append(f"📉 Declining revenue (CAGR: {cagr:.1f}%)")
            elif 0 < cagr < 5:
                insights.append(f"🐌 Slow revenue growth (CAGR: {cagr:.1f}%) - explore growth strategies")
        
        profit_trends = [v for k, v in trends.items() if 'net income' in k.lower() or 'profit' in k.lower()]
        if revenue_trends and profit_trends:
            rev_cagr = revenue_trends[0].get('cagr', 0)
            prof_cagr = profit_trends[0].get('cagr', 0)
            if rev_cagr > 0 and prof_cagr < rev_cagr:
                insights.append(f"⚠️ Profit growing slower than revenue - check cost management")
        
        quality_score = self._calculate_quality_score(df)
        if quality_score < 70:
            insights.append(f"⚠️ Data quality score is low ({quality_score:.0f}%), results may be less reliable")
        
        anomalies = self._detect_anomalies(df)
        if anomalies['value_anomalies']:
            insights.append(f"🔍 Detected {len(anomalies['value_anomalies'])} unusual values - review for accuracy")
        
        return insights


from dataclasses import dataclass, field
from typing import Optional, Dict, Callable
import time
import queue
import threading


@dataclass
class APIRequest:
    """API request with metadata"""
    id: str
    endpoint: str
    method: str
    data: Optional[Dict] = None
    params: Optional[Dict] = None
    priority: int = 5
    timestamp: float = field(default_factory=time.time)
    retries: int = 0
    callback: Optional[Callable] = None


class RequestQueue:
    """Priority queue for API requests with coalescing"""
    
    def __init__(self, max_size: int = 100, coalesce_window_ms: int = 100):
        self.queue = queue.PriorityQueue(max_size)
        self.pending = {}
        self.coalesce_window = coalesce_window_ms / 1000
        self._lock = threading.Lock()
    
    def put(self, request: APIRequest, timeout: Optional[float] = None) -> bool:
        """Add request to queue with optional coalescing"""
        with self._lock:
            # Check for similar pending request
            coalesce_key = f"{request.endpoint}:{request.method}:{json.dumps(request.data, sort_keys=True)}"
            
            if coalesce_key in self.pending:
                # Coalesce with existing request
                existing = self.pending[coalesce_key]
                if time.time() - existing.timestamp < self.coalesce_window:
                    # Update priority to highest
                    existing.priority = min(existing.priority, request.priority)
                    return True
            
            self.pending[coalesce_key] = request
        
        try:
            self.queue.put((request.priority, request.id, request), timeout=timeout)
            return True
        except queue.Full:
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[APIRequest]:
        """Get highest priority request"""
        try:
            _, _, request = self.queue.get(timeout=timeout)
            
            with self._lock:
                coalesce_key = f"{request.endpoint}:{request.method}:{json.dumps(request.data, sort_keys=True)}"
                self.pending.pop(coalesce_key, None)
            
            return request
        except queue.Empty:
            return None


class EnhancedAPIClient:
    """Enhanced API client with advanced features for Kaggle integration"""
    
    def __init__(self, base_url: str, config: Configuration):
        self.base_url = base_url.rstrip('/')
        self.config = config
        self.timeout = config.get('ai.kaggle_api_timeout', 30)
        self.max_retries = config.get('ai.kaggle_max_retries', 3)
        self._session = None
        
        # Connection pool - MOVED BEFORE _setup_session()
        self.connection_pool_size = config.get('ai.kaggle_connection_pool_size', 20)
        
        # Now setup session AFTER connection_pool_size is defined
        self._setup_session()
        
        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        self.response_times = deque(maxlen=100)
        self._lock = threading.Lock()
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('ai.kaggle_circuit_breaker_threshold', 5),
            recovery_timeout=config.get('ai.kaggle_circuit_breaker_timeout', 300),
            expected_exception=requests.exceptions.RequestException
        )
        
        # Request queue
        self.request_queue = RequestQueue(
            max_size=config.get('ai.kaggle_max_queue_size', 100),
            coalesce_window_ms=config.get('ai.kaggle_coalesce_window_ms', 100)
        )
        
        # Response cache
        self.response_cache = AdvancedCache(max_size_mb=20, default_ttl=300)
        
        # Background processor
        self._processor_thread = None
        self._stop_processor = threading.Event()
        self._start_processor()
        
        self.logger = LoggerFactory.get_logger('EnhancedAPIClient')
    
    def _setup_session(self):
        """Setup requests session with optimized settings"""
        self._session = requests.Session()
        
        # Retry strategy with exponential backoff
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
            raise_on_status=False
        )
        
        # Connection pooling - Now self.connection_pool_size is defined
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.connection_pool_size,
            pool_maxsize=self.connection_pool_size * 2
        )
        
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        
        # Headers
        self._session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'EliteFinancialAnalytics/5.1',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'ngrok-skip-browser-warning': 'true'
        })
        
        # API key if configured
        api_key = self.config.get('ai.kaggle_api_key')
        if api_key:
            self._session.headers['Authorization'] = f'Bearer {api_key}'
    
    def _start_processor(self):
        """Start background request processor"""
        self._processor_thread = threading.Thread(target=self._process_requests, daemon=True)
        self._processor_thread.start()
    
    def _process_requests(self):
        """Process queued requests in background"""
        while not self._stop_processor.is_set():
            try:
                request = self.request_queue.get(timeout=0.1)
                if request:
                    self._execute_request(request)
            except Exception as e:
                self.logger.error(f"Request processor error: {e}")
    
    def _execute_request(self, api_request: APIRequest):
        """Execute a single API request"""
        try:
            response = self._make_raw_request(
                api_request.method,
                api_request.endpoint,
                api_request.data,
                api_request.params
            )
            
            if api_request.callback:
                api_request.callback(response)
                
        except Exception as e:
            self.logger.error(f"Failed to execute request {api_request.id}: {e}")
            if api_request.callback:
                api_request.callback({'error': str(e)})
    
    def _make_raw_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                         params: Optional[Dict] = None) -> Dict:
        """Make raw HTTP request with circuit breaker"""
        def _request():
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            
            # Check cache for GET requests
            if method == 'GET':
                cache_key = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
                cached = self.response_cache.get(cache_key)
                if cached:
                    self.logger.debug(f"Cache hit for {endpoint}")
                    return cached
            
            start_time = time.time()
            
            # Prepare headers for ngrok
            headers = {
                'ngrok-skip-browser-warning': 'true',
                'Content-Type': 'application/json'
            }
            
            # Prepare request kwargs
            request_kwargs = {
                'method': method,
                'url': url,
                'params': params,
                'timeout': self.timeout,
                'verify': False,  # Disable SSL verification for ngrok
                'headers': headers
            }
            
            # Handle data - FIX: Use the data parameter directly
            if data:
                if self.config.get('ai.kaggle_enable_compression', True) and len(json.dumps(data)) > 1024:
                    # Compression for large payloads
                    import gzip
                    self._session.headers['Content-Encoding'] = 'gzip'
                    data_bytes = json.dumps(data).encode('utf-8')
                    request_kwargs['data'] = gzip.compress(data_bytes)
                    request_kwargs['headers']['Content-Type'] = 'application/json'
                else:
                    # Remove compression header if present
                    self._session.headers.pop('Content-Encoding', None)
                    request_kwargs['json'] = data
            
            # Make request
            response = self._session.request(**request_kwargs)
            
            # Log response for debugging
            self.logger.debug(f"Response status: {response.status_code}")
            
            # Handle different response types
            if response.status_code == 200:
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    # Try to parse as text
                    result = {'response': response.text, 'status': 'ok'}
            else:
                self.logger.error(f"Request failed: {response.status_code} - {response.text}")
                response.raise_for_status()
            
            # Cache successful GET responses
            if method == 'GET' and response.status_code == 200:
                cache_key = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
                self.response_cache.set(cache_key, result)
            
            # Track metrics
            duration = time.time() - start_time
            performance_monitor.track_api_call(endpoint, True, duration)
            
            return result
        
        # Execute with circuit breaker
        return self.circuit_breaker.call(_request)
    
    def make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                    params: Optional[Dict] = None, timeout: Optional[int] = None,
                    priority: int = 5, async_callback: Optional[Callable] = None) -> Optional[Dict]:
        """Make HTTP request with queuing and priority"""
        request_id = f"{endpoint}_{time.time()}_{id(data)}"
        
        with self._lock:
            self.request_count += 1
            self.last_request_time = datetime.now()
        
        # Create request
        api_request = APIRequest(
            id=request_id,
            endpoint=endpoint,
            method=method,
            data=data,
            params=params,
            priority=priority,
            callback=async_callback
        )
        
        # Async request with callback
        if async_callback:
            success = self.request_queue.put(api_request, timeout=timeout)
            if not success:
                raise Exception("Request queue full")
            return None
        
        # Sync request
        try:
            return self._make_raw_request(method, endpoint, data, params)
        except Exception as e:
            with self._lock:
                self.error_count += 1
            
            performance_monitor.track_api_call(endpoint, False, 0, str(e))
            raise
    
    def batch_request(self, requests: List[Dict], endpoint: str = '/batch') -> List[Dict]:
        """Execute multiple requests in a single batch"""
        batch_size = self.config.get('ai.kaggle_batch_size', 50)
        results = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            response = self.make_request('POST', endpoint, {'requests': batch})
            
            if response and 'results' in response:
                results.extend(response['results'])
            else:
                # Fallback to individual requests
                for req in batch:
                    try:
                        result = self.make_request(req['method'], req['endpoint'], req.get('data'))
                        results.append(result)
                    except Exception as e:
                        results.append({'error': str(e)})
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check with detailed status"""
        try:
            # Ensure no double slashes
            url = f"{self.base_url.rstrip('/')}/health"
            self.logger.info(f"Health check URL: {url}")  # Add this for debugging
            
            response = self._session.get(
                url,
                timeout=5,
                verify=False,
                headers={'ngrok-skip-browser-warning': 'true'}
            )
            
            self.logger.info(f"Health response status: {response.status_code}")  # Debug log
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    self.logger.info(f"Health response data: {response_data}")  # Debug log
                    return {
                        'healthy': True,
                        'response': response_data,
                        'circuit_breaker': self.circuit_breaker.get_state(),
                        'queue_size': self.request_queue.queue.qsize(),
                        'cache_stats': self.response_cache.get_stats()
                    }
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Health response not JSON: {response.text[:200]} - {e}")
                    return {
                        'healthy': True,  # Assume healthy if we get 200 even if not JSON
                        'response': {'status': 'ok', 'message': response.text},
                        'circuit_breaker': self.circuit_breaker.get_state(),
                        'queue_size': self.request_queue.queue.qsize(),
                        'cache_stats': self.response_cache.get_stats()
                    }
            else:
                self.logger.error(f"Health check failed with status: {response.status_code} - {response.text[:200]}")
                return {
                    'healthy': False,
                    'error': f"Status code: {response.status_code}",
                    'circuit_breaker': self.circuit_breaker.get_state(),
                    'queue_size': self.request_queue.queue.qsize()
                }
                
        except Exception as e:
            self.logger.error(f"Health check exception: {str(e)}", exc_info=True)
            return {
                'healthy': False,
                'error': str(e),
                'circuit_breaker': self.circuit_breaker.get_state(),
                'queue_size': self.request_queue.queue.qsize()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed client statistics"""
        with self._lock:
            avg_response_time = np.mean(self.response_times) if self.response_times else 0
            p95_response_time = np.percentile(list(self.response_times), 95) if self.response_times else 0
            
            return {
                'total_requests': self.request_count,
                'total_errors': self.error_count,
                'error_rate': self.error_count / self.request_count if self.request_count > 0 else 0,
                'avg_response_time': avg_response_time,
                'p95_response_time': p95_response_time,
                'last_request': self.last_request_time,
                'circuit_breaker_state': self.circuit_breaker.get_state()['state'],
                'queue_size': self.request_queue.queue.qsize(),
                'cache_hit_rate': self.response_cache.get_stats()['hit_rate']
            }
    
    def close(self):
        """Close the session and cleanup"""
        self._stop_processor.set()
        if self._processor_thread:
            self._processor_thread.join(timeout=1)
        
        if self._session:
            self._session.close()


# --- 18. Enhanced AI Mapping System with Robust Kaggle Integration ---
class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self):
        self.operations = {}
        self._lock = threading.Lock()

    def start_operation(self, operation_id: str, total_items: int, description: str = ""):
        """Start tracking an operation"""
        with self._lock:
            self.operations[operation_id] = {
                'total': total_items,
                'completed': 0,
                'description': description,
                'start_time': time.time(),
                'status': 'running',
                'error': None
            }
    
    def update_progress(self, operation_id: str, completed: int):
        """Update operation progress"""
        with self._lock:
            if operation_id in self.operations:
                self.operations[operation_id]['completed'] = completed
    
    def complete_operation(self, operation_id: str, error: Optional[str] = None):
        """Mark operation as complete"""
        with self._lock:
            if operation_id in self.operations:
                self.operations[operation_id]['status'] = 'error' if error else 'completed'
                self.operations[operation_id]['error'] = error
                self.operations[operation_id]['end_time'] = time.time()
    
    def get_progress(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get operation progress"""
        with self._lock:
            return self.operations.get(operation_id)


class AIMapper(Component):
    """Enhanced AI-powered mapping with robust Kaggle GPU support"""
    
    def __init__(self, config: Configuration):
        super().__init__(config)
        self.model = None
        self.embeddings_cache = AdvancedCache(max_size_mb=50)
        self.fallback_mapper = None
        self._api_client = None
        self._kaggle_available = False
        self._kaggle_info = {}
        self._last_health_check = None
        self._health_check_lock = threading.Lock()
        self._batch_queue = queue.Queue()
        self._batch_processor = None
        self._session = None
        self.progress_tracker = ProgressTracker()
        self._embedding_buffer = OrderedDict()
        self._buffer_lock = threading.Lock()
        self._max_buffer_size = 1000

    def _do_initialize(self):
        """Initialize AI components with enhanced error handling"""
        if not self.config.get('ai.enabled', True):
            self._logger.info("AI mapping disabled in configuration")
            return
        
        self._session = requests.Session()
        self._session.headers.update({'Content-Type': 'application/json'})
        
        # Initialize Kaggle API if configured
        if self.config.get('ai.use_kaggle_api', False) and self.config.get('ai.kaggle_api_url'):
            self._initialize_kaggle_api()
        
        # Initialize local model if available and needed
        if not self._kaggle_available or self.config.get('ai.kaggle_fallback_to_local', True):
            self._initialize_local_model()
        
        # Initialize fallback fuzzy mapper
        self.fallback_mapper = FuzzyMapper(self.config)
        self.fallback_mapper.initialize()
        
        # Start batch processor if Kaggle is available
        if self._kaggle_available:
            self._start_batch_processor()
        
        # Start background health checker
        self._start_health_monitor()
    
    def _do_cleanup(self):
        """Cleanup resources"""
        if self._api_client:
            self._api_client.close()
        
        if self._session:
            self._session.close()
        
        if self._batch_processor and self._batch_processor.is_alive():
            # Signal shutdown
            self._batch_queue.put(None)
    
    def _initialize_kaggle_api(self):
        """Initialize Kaggle API connection with enhanced error handling"""
        try:
            api_url = self.config.get('ai.kaggle_api_url')
            
            # Validate URL format
            if not api_url.startswith(('http://', 'https://')):
                self._logger.error(f"Invalid API URL format: {api_url}")
                return
            
            self._api_client = EnhancedAPIClient(api_url, self.config)
            
            # Test connection with retries
            for attempt in range(3):
                if self._test_kaggle_connection():
                    self._kaggle_available = True
                    self._logger.info(f"Successfully connected to Kaggle API at {api_url}")
                    break
                
                if attempt < 2:
                    wait_time = (attempt + 1) * 2
                    self._logger.warning(f"Kaggle connection attempt {attempt + 1} failed, retrying in {wait_time}s")
                    time.sleep(wait_time)
            else:
                self._logger.warning("Failed to connect to Kaggle API after 3 attempts")
                
        except Exception as e:
            self._logger.error(f"Failed to initialize Kaggle API: {e}")
            self._kaggle_available = False
    
    def _test_kaggle_connection(self) -> bool:
        """Test Kaggle API connection with comprehensive checks"""
        try:
            self._logger.info(f"Testing connection to: {self.config.get('ai.kaggle_api_url')}")
            
            # First, test the embed endpoint since we know it works from diagnostics
            try:
                test_response = self._api_client.make_request(
                    'POST', '/embed',
                    {'texts': ['test']},
                    timeout=10
                )
                
                self._logger.info(f"Embed test response: {test_response}")
                
                if test_response and isinstance(test_response, dict) and 'embeddings' in test_response:
                    self._logger.info("Embed test successful - proceeding with connection")
                    
                    # Optionally test health for additional info
                    try:
                        health_status = self._api_client.health_check()
                        if health_status['healthy'] and health_status.get('response'):
                            response = health_status['response']
                            self._kaggle_info = response
                        else:
                            self._logger.warning("Health check failed, but embed works - ignoring")
                    except Exception as e:
                        self._logger.warning(f"Health check failed but ignored: {e}")
                    
                    self._last_health_check = time.time()
                    self._kaggle_info = self._kaggle_info or {'status': 'healthy (embed test)'}
                    
                    # Store GPU info if available
                    if 'system' in self._kaggle_info:
                        system_info = self._kaggle_info['system']
                        self._kaggle_info.update({
                            'gpu_name': system_info.get('gpu_name', 'Unknown'),
                            'gpu_available': system_info.get('gpu_available', False),
                            'model': system_info.get('model', 'Unknown'),
                            'device': system_info.get('device', 'Unknown')
                        })
                    
                    self._logger.info(f"Successfully connected to Kaggle API - GPU: {self._kaggle_info.get('gpu_name', 'Unknown')}")
                    return True
                else:
                    self._logger.error(f"Embed test failed: {test_response}")
                    return False
                    
            except Exception as e:
                self._logger.error(f"Embed test failed: {e}", exc_info=True)
                return False
                
        except Exception as e:
            self._logger.error(f"Kaggle connection test failed: {e}", exc_info=True)
            return False
    
    def _start_health_monitor(self):
        """Start background health monitoring thread"""
        def monitor():
            while self._initialized:
                try:
                    time.sleep(self.config.get('ai.kaggle_health_check_interval', 60))
                    if self._kaggle_available:
                        self._check_kaggle_health()
                except Exception as e:
                    self._logger.error(f"Health monitor error: {e}")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _initialize_local_model(self):
        """Initialize local sentence transformer model with memory optimization"""
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            self._logger.warning("Sentence transformers not available")
            return
            
        try:
            model_name = self.config.get('ai.model_name', 'all-MiniLM-L6-v2')
            from sentence_transformers import SentenceTransformer
            
            # Use half precision on CPU to reduce memory
            self.model = SentenceTransformer(model_name)
            
            # Optimize for inference
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            self._logger.info(f"Loaded local model: {model_name}")
            
            self._precompute_standard_embeddings()
            
        except Exception as e:
            self._logger.error(f"Failed to load local model: {e}")
    
    def _precompute_standard_embeddings(self):
        """Pre-compute embeddings for standard metrics with batching"""
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
            'EBIT': ['ebit', 'operating income', 'operating profit'],
            'Interest Expense': ['interest expense', 'finance costs', 'interest costs'],
            'Tax Expense': ['tax expense', 'income tax', 'tax'],
        }
        
        for metric, descriptions in standard_metrics.items():
            combined_text = ' '.join(descriptions)
            embedding = self._get_embedding(combined_text)
            if embedding is not None:
                SimpleState.set(f"standard_embedding_{metric}", embedding)
    
    def _start_batch_processor(self):
        """Start background thread for batch processing"""
        self._batch_processor = threading.Thread(target=self._process_batch_queue, daemon=True)
        self._batch_processor.start()
    
    def _process_batch_queue(self):
        """Process batched embedding requests with optimizations"""
        while True:
            try:
                batch = []
                deadline = time.time() + 0.1
                max_batch_size = self.config.get('ai.kaggle_batch_size', 50)
                
                # Collect batch
                while time.time() < deadline and len(batch) < max_batch_size:
                    try:
                        item = self._batch_queue.get(timeout=0.01)
                        if item is None:  # Shutdown signal
                            return
                        batch.append(item)
                    except queue.Empty:
                        break
                
                if batch:
                    self._process_embedding_batch(batch)
                    
            except Exception as e:
                self._logger.error(f"Batch processor error: {e}")
                time.sleep(1)
    
    def _process_embedding_batch(self, batch: List[Dict]):
        """Process a batch of embedding requests with error handling"""
        texts = [item['text'] for item in batch]
        futures = [item['future'] for item in batch]
        operation_id = f"batch_{time.time()}"
        
        try:
            # Track progress
            self.progress_tracker.start_operation(operation_id, len(texts), "Processing embeddings")
            
            embeddings = self._get_embeddings_kaggle_batch(texts, operation_id)
            
            if embeddings:
                for i, (future, embedding) in enumerate(zip(futures, embeddings)):
                    future.set_result(embedding)
                    self.progress_tracker.update_progress(operation_id, i + 1)
            else:
                # Fallback to local processing
                for i, (text, future) in enumerate(zip(texts, futures)):
                    embedding = self._get_embedding_local(text)
                    future.set_result(embedding)
                    self.progress_tracker.update_progress(operation_id, i + 1)
            
            self.progress_tracker.complete_operation(operation_id)
            
        except Exception as e:
            self.progress_tracker.complete_operation(operation_id, str(e))
            for future in futures:
                future.set_exception(e)
    
    def _check_kaggle_health(self) -> bool:
        """Periodic health check for Kaggle API with recovery"""
        with self._health_check_lock:
            current_time = time.time()
            check_interval = self.config.get('ai.kaggle_health_check_interval', 60)
            
            if (self._last_health_check is None or 
                current_time - self._last_health_check > check_interval):
                
                previous_state = self._kaggle_available
                self._kaggle_available = self._test_kaggle_connection()
                
                # Handle state changes
                if previous_state and not self._kaggle_available:
                    self._logger.warning("Kaggle API became unavailable")
                    SimpleState.set('kaggle_api_status', 'offline')
                elif not previous_state and self._kaggle_available:
                    self._logger.info("Kaggle API recovered")
                    SimpleState.set('kaggle_api_status', 'online')
        
        return self._kaggle_available
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding with multi-level caching and fallback"""
        if not text:
            return None
        
        # Check memory buffer first
        with self._buffer_lock:
            if text in self._embedding_buffer:
                # Move to end (LRU)
                self._embedding_buffer.move_to_end(text)
                return self._embedding_buffer[text]
        
        # Check persistent cache
        cache_key = f"embedding_{hashlib.md5(text.encode()).hexdigest()}"
        cached = self.embeddings_cache.get(cache_key)
        if cached is not None:
            # Add to memory buffer
            self._add_to_buffer(text, cached)
            return cached
        
        # Generate embedding
        embedding = None
        
        if self._kaggle_available and self._check_kaggle_health():
            embedding = self._get_embedding_kaggle(text)
        
        if embedding is None and self.model is not None:
            embedding = self._get_embedding_local(text)
        
        if embedding is not None:
            # Cache the result
            self.embeddings_cache.set(cache_key, embedding)
            self._add_to_buffer(text, embedding)
        
        return embedding
    
    def _add_to_buffer(self, text: str, embedding: np.ndarray):
        """Add embedding to memory buffer with size limit"""
        with self._buffer_lock:
            # Remove oldest if at capacity
            if len(self._embedding_buffer) >= self._max_buffer_size:
                self._embedding_buffer.popitem(last=False)
            
            self._embedding_buffer[text] = embedding
    
    def _get_embedding_kaggle(self, text: str) -> Optional[np.ndarray]:
        """Get single embedding from Kaggle API"""
        try:
            response = self._api_client.make_request(
                'POST', '/embed',
                {'texts': [text]},
                timeout=10,
                priority=5
            )
            
            if response and isinstance(response, dict):
                # Your API returns embeddings in the 'embeddings' key
                if 'embeddings' in response:
                    embeddings = response['embeddings']
                    if isinstance(embeddings, list) and len(embeddings) > 0:
                        # Get the first embedding (since we sent one text)
                        embedding = embeddings[0]
                        if isinstance(embedding, list):
                            return np.array(embedding)
                            
            return None
            
        except Exception as e:
            self._logger.error(f"Kaggle embedding error: {e}")
            return None
    
    def _get_embedding_local(self, text: str) -> Optional[np.ndarray]:
        """Get embedding using local model"""
        if self.model is None:
            return None
            
        try:
            return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            self._logger.error(f"Local embedding error: {e}")
            return None
    
    def _get_embeddings_kaggle_batch(self, texts: List[str], operation_id: str) -> Optional[List[np.ndarray]]:
        """Get batch embeddings from Kaggle API with progress tracking"""
        try:
            # Split into sub-batches if needed
            max_batch = self.config.get('ai.kaggle_batch_size', 50)
            all_embeddings = []
            
            for i in range(0, len(texts), max_batch):
                sub_batch = texts[i:i + max_batch]
                
                response = self._api_client.make_request(
                    'POST', '/embed',
                    {'texts': sub_batch},
                    priority=3
                )
                
                if response and isinstance(response, dict) and 'embeddings' in response:
                    embeddings = response['embeddings']
                    if isinstance(embeddings, list):
                        # Convert each embedding to numpy array
                        batch_embeddings = [np.array(emb) for emb in embeddings]
                        all_embeddings.extend(batch_embeddings)
                        self.progress_tracker.update_progress(operation_id, len(all_embeddings))
                    else:
                        self._logger.error("Unexpected embeddings format")
                        return None
                else:
                    self._logger.error(f"Invalid response format: {response}")
                    return None
            
            return all_embeddings
                
        except Exception as e:
            self._logger.error(f"Kaggle batch embedding error: {e}")
            return None
    
    @error_boundary({})
    def map_metrics_with_confidence_levels(self, source_metrics: List[str], 
                                         target_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Enhanced mapping with multiple confidence levels and progress tracking"""
        operation_id = f"mapping_{time.time()}"
        self.progress_tracker.start_operation(operation_id, len(source_metrics), "Mapping financial metrics")
        
        try:
            confidence_thresholds = self.config.get('ai.confidence_levels', {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            })
            
            # Check if we should use Kaggle
            use_kaggle = self._kaggle_available and self._check_kaggle_health()
            
            if use_kaggle:
                # Try Kaggle API first
                try:
                    response = self._api_client.make_request(
                        'POST', '/map_metrics_with_confidence',
                        {'source_metrics': source_metrics},
                        timeout=30
                    )
                    
                    if response and 'high_confidence' in response:
                        self.progress_tracker.complete_operation(operation_id)
                        return response
                        
                except Exception as e:
                    self._logger.warning(f"Kaggle mapping failed, falling back to local: {e}")
            
            # Fallback to local processing
            base_result = self.map_metrics(source_metrics, target_metrics)
            
            results = {
                'high_confidence': {},
                'medium_confidence': {},
                'low_confidence': {},
                'requires_manual': [],
                'suggestions': base_result.get('suggestions', {}),
                'method': base_result.get('method', 'unknown')
            }
            
            for i, source in enumerate(source_metrics):
                if source in base_result['mappings']:
                    confidence = base_result['confidence_scores'].get(source, 0)
                    target = base_result['mappings'][source]
                    
                    if confidence >= confidence_thresholds['high']:
                        results['high_confidence'][source] = {
                            'target': target,
                            'confidence': confidence
                        }
                    elif confidence >= confidence_thresholds['medium']:
                        results['medium_confidence'][source] = {
                            'target': target,
                            'confidence': confidence
                        }
                    elif confidence >= confidence_thresholds['low']:
                        results['low_confidence'][source] = {
                            'target': target,
                            'confidence': confidence
                        }
                    else:
                        results['requires_manual'].append(source)
                else:
                    results['requires_manual'].append(source)
                
                self.progress_tracker.update_progress(operation_id, i + 1)
            
            self.progress_tracker.complete_operation(operation_id)
            return results
            
        except Exception as e:
            self.progress_tracker.complete_operation(operation_id, str(e))
            raise
    
    def map_metrics(self, source_metrics: List[str], 
                   target_metrics: Optional[List[str]] = None,
                   confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Map source metrics to target metrics with hybrid approach"""
        
        if not source_metrics:
            return {
                'mappings': {},
                'confidence_scores': {},
                'suggestions': {},
                'unmapped_metrics': [],
                'method': 'none'
            }
        
        if confidence_threshold is None:
            confidence_threshold = self.config.get('ai.similarity_threshold', 0.6)
        
        if target_metrics is None:
            target_metrics = self._get_standard_metrics()
        
        # Try AI mapping if available
        if (self.model is not None or self._kaggle_available):
            return self._map_metrics_ai(source_metrics, target_metrics, confidence_threshold)
        else:
            # Fallback to fuzzy matching
            return self.fallback_mapper.map_metrics(source_metrics, target_metrics)
    
    def _map_metrics_ai(self, source_metrics: List[str], 
                       target_metrics: List[str],
                       confidence_threshold: float) -> Dict[str, Any]:
        """AI-based metric mapping with embeddings"""
        mappings = {}
        confidence_scores = {}
        suggestions = {}
        unmapped = []
        
        batch_size = self.config.get('ai.batch_size', 32)
        
        with performance_monitor.measure("ai_mapping"):
            # Process in batches
            for i in range(0, len(source_metrics), batch_size):
                batch = source_metrics[i:i + batch_size]
                
                try:
                    # Get embeddings for source metrics
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
                        # Check pre-computed embeddings first
                        embedding = SimpleState.get(f"standard_embedding_{target}")
                        
                        if embedding is None:
                            embedding = self._get_embedding(target.lower())
                        
                        if embedding is not None:
                            target_embeddings.append(embedding)
                            valid_targets.append(target)
                    
                    if not target_embeddings:
                        unmapped.extend(valid_sources)
                        continue
                    
                    # Calculate similarities
                    source_matrix = np.vstack(source_embeddings)
                    target_matrix = np.vstack(target_embeddings)
                    
                    similarities = cosine_similarity(source_matrix, target_matrix)
                    
                    # Create mappings
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
                            {'target': target, 'confidence': float(score)}
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
            'method': 'ai' if self._kaggle_available else 'local_ai'
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
            'Investing Cash Flow', 'Financing Cash Flow',
            'EBIT', 'EBITDA', 'Interest Expense', 'Tax Expense'
        ]
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get comprehensive API status with metrics"""
        status = {
            'kaggle_configured': bool(self.config.get('ai.kaggle_api_url')),
            'kaggle_available': self._kaggle_available,
            'local_model_available': self.model is not None,
            'cache_size': len(self.embeddings_cache._cache),
            'buffer_size': len(self._embedding_buffer),
            'api_info': self._kaggle_info,
            'api_stats': self._api_client.get_stats() if self._api_client else None,
            'api_metrics': performance_monitor.get_api_summary() if self._kaggle_available else None
        }
        
        # Add circuit breaker status
        if self._api_client and hasattr(self._api_client, 'circuit_breaker'):
            status['circuit_breaker'] = self._api_client.circuit_breaker.get_state()
        
        return status
    
    def process_natural_language_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process natural language queries about financial data"""
        if self._kaggle_available:
            try:
                response = self._api_client.make_request(
                    'POST', '/process_query',
                    {'query': query, 'context': context}
                )
                
                if response:
                    return response
                    
            except Exception as e:
                self._logger.warning(f"Kaggle NL query failed: {e}")
        
        # Fallback to simple keyword matching
        return self._process_query_local(query, context)
    
    def _process_query_local(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Local processing of natural language queries"""
        query_lower = query.lower()
        
        # Simple intent detection
        intents = {
            'growth': ['growth', 'increase', 'rise', 'expand'],
            'decline': ['decline', 'decrease', 'fall', 'drop'],
            'comparison': ['compare', 'versus', 'difference', 'better'],
            'trend': ['trend', 'pattern', 'direction', 'movement']
        }
        
        detected_intents = []
        for intent, keywords in intents.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        return {
            'query': query,
            'detected_intents': detected_intents,
            'context': context,
            'method': 'local'
        }


# --- 19. Fuzzy Mapping Fallback ---
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
            
            scores = []
            for target in target_metrics:
                score = fuzz.token_sort_ratio(source_lower, target.lower()) / 100.0
                scores.append((target, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            
            if scores and scores[0][1] > 0.7:
                mappings[source] = scores[0][0]
                confidence_scores[source] = scores[0][1]
            else:
                unmapped.append(source)
            
            suggestions[source] = [
                {'target': target, 'confidence': score}
                for target, score in scores[:3]
            ]
        
        return {
            'mappings': mappings,
            'confidence_scores': confidence_scores,
            'suggestions': suggestions,
            'unmapped_metrics': unmapped,
            'method': 'fuzzy'
        }
    
    def _get_standard_metrics(self) -> List[str]:
        """Get standard metrics list"""
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
            'EBIT', 'EBITDA', 'Interest Expense', 'Tax Expense'
        ]

#----20.class EnhancedPenmanNissimAnalyzer
class EnhancedPenmanNissimAnalyzer:
    """
    Enhanced Penman-Nissim analyzer with improved data flow, accuracy, and robustness
    Version 5.0 - Complete rewrite with all fixes and enhancements
    """
    
    def __init__(self, df: pd.DataFrame, mappings: Dict[str, str]):
        self.df = df
        self.mappings = mappings
        self.logger = LoggerFactory.get_logger('PenmanNissim')
        self.validation_results = {}
        self.calculation_metadata = {}
        self.calculation_cache = {}  # Cache for expensive calculations
        
        # CRITICAL: Restructure data ONCE and store it
        self._df_clean = None
        self._ensure_clean_data()
        
        # Store reformulated statements for reuse
        self._cached_bs = None
        self._cached_is = None
        
        self.core_analyzer = None  # Don't use core analyzer due to NotImplemented error
        self._validate_input_data()

    @staticmethod
    def safe_format(self, value, format_spec='.1f', default='N/A'):  # CHANGED: Default format_spec without leading ':'
        if isinstance(value, (int, float)) and not np.isnan(value):
            return f"{value:{format_spec}}"  # Correct: {value:{format_spec}} → e.g., {15.3:.1f} = '15.3'
        return default

    def _ensure_clean_data(self):
        """Ensure we have clean restructured data (only restructure once)"""
        if self._df_clean is None:
            self.logger.info("[PN-INIT] Restructuring data once for all calculations")
            self._df_clean = self._restructure_for_penman_nissim_v5(self.df)
            self.logger.info(f"[PN-INIT] Clean data shape: {self._df_clean.shape}")
            
            # Log sample of clean data for verification
            if len(self._df_clean.columns) > 0:
                sample_col = self._df_clean.columns[0]
                non_zero_count = (self._df_clean[sample_col] != 0).sum()
                self.logger.info(f"[PN-INIT] Sample column {sample_col} has {non_zero_count} non-zero values")

    # PASTE THIS CODE: Replace the _restructure_for_penman_nissim_v5 method with this enhanced version
    
    def _restructure_for_penman_nissim_v5(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Version 5: Ultimate restructuring with comprehensive debugging and data preservation
        """
        self.logger.info("\n" + "#"*80)
        self.logger.info("[PN-RESTRUCTURE-V5] Starting ULTIMATE data restructuring with debugging")
        self.logger.info(f"[PN-RESTRUCTURE-V5] Input shape: {df.shape}")
        self.logger.info("#"*80)
        
        # DEBUG: Log sample of original data
        self.logger.info("[PN-DEBUG] Sample of original data structure:")
        self.logger.info(f"  Columns: {list(df.columns)[:5]}...")
        self.logger.info(f"  Index sample: {list(df.index)[:10]}...")
        
        # Look for CapEx specifically in original data
        capex_items = [idx for idx in df.index if any(kw in str(idx).lower() 
                       for kw in ['capex', 'capital expenditure', 'purchase', 'fixed asset'])]
        self.logger.info(f"[PN-DEBUG] Found {len(capex_items)} potential CapEx items in original data:")
        for item in capex_items[:5]:
            self.logger.info(f"  - {item}")
        
        # 1. Enhanced year pattern matching with debugging
        year_patterns = [
            (re.compile(r'(\d{6})'), 'YYYYMM', lambda m: m.group(1), 1),
            (re.compile(r'(\d{4})(?!\d)'), 'YYYY', lambda m: m.group(1) + '03', 2),
            (re.compile(r'FY\s*(\d{4})'), 'FY YYYY', lambda m: m.group(1) + '03', 2),
            (re.compile(r'(\d{4})-(\d{2})'), 'YYYY-YY', lambda m: m.group(1) + '03', 3),
            (re.compile(r'Mar[- ](\d{2})'), 'Mar-YY', lambda m: '20' + m.group(1) + '03', 4),
            (re.compile(r'March[- ](\d{4})'), 'March YYYY', lambda m: m.group(1) + '03', 3),
        ]
        
        # Extract years with detailed logging
        all_years = set()
        year_to_columns = defaultdict(list)
        column_metadata = {}
        
        self.logger.info("[PN-DEBUG] Analyzing column structure:")
        for col in df.columns:
            col_str = str(col)
            self.logger.debug(f"  Processing column: {col_str}")
            
            best_match = None
            best_priority = 999
            
            for pattern, pattern_name, extractor, priority in year_patterns:
                match = pattern.search(col_str)
                if match and priority < best_priority:
                    try:
                        normalized_year = extractor(match)
                        year_int = int(normalized_year[:4])
                        if 2000 <= year_int <= 2099:
                            best_match = (normalized_year, pattern_name)
                            best_priority = priority
                            self.logger.debug(f"    Matched pattern {pattern_name}: {normalized_year}")
                    except Exception as e:
                        self.logger.debug(f"    Pattern {pattern_name} failed: {e}")
                        continue
            
            if best_match:
                normalized_year, pattern_name = best_match
                all_years.add(normalized_year)
                year_to_columns[normalized_year].append(col)
                column_metadata[col] = {
                    'year': normalized_year,
                    'pattern': pattern_name,
                    'priority': best_priority
                }
                self.logger.debug(f"    Final mapping: {col} -> {normalized_year}")
            else:
                self.logger.warning(f"    No year pattern found for column: {col}")
        
        final_columns = sorted(list(all_years))
        self.logger.info(f"[PN-RESTRUCTURE-V5] Extracted {len(final_columns)} unique years: {final_columns}")
        
        # 2. Create restructured DataFrame with comprehensive tracking
        restructured = pd.DataFrame(index=df.index, columns=final_columns, dtype=np.float64)
        
        # Track data transfer statistics
        transfer_stats = {
            'total_attempts': 0,
            'successful_transfers': 0,
            'failed_transfers': 0,
            'zero_values': 0,
            'null_values': 0,
            'capex_transfers': 0
        }
        
        # 3. Enhanced data extraction with preservation focus
        self.logger.info("[PN-DEBUG] Starting data transfer process...")
        
        for idx in df.index:
            idx_str = str(idx)
            is_capex = any(kw in idx_str.lower() for kw in ['capex', 'capital expenditure', 'purchase', 'fixed asset'])
            
            if is_capex:
                self.logger.info(f"[PN-CAPEX-TRANSFER] Processing CapEx item: {idx_str}")
            
            # Determine statement type
            statement_type = self._determine_statement_type_v2(idx_str)
            
            for year in final_columns:
                transfer_stats['total_attempts'] += 1
                
                # FIXED: Relaxed filter - prioritize columns with statement type, but fall back to any if none match
                source_columns = [col for col in year_to_columns[year] if statement_type.lower() in str(col).lower()]
                
                if not source_columns:
                    # Fallback: Use any column for this year if no statement match (prevents empty data)
                    source_columns = year_to_columns[year]
                    if source_columns:
                        self.logger.warning(f"[PN-TRANSFER] No statement-matching columns for {idx_str} ({statement_type}) in {year} - using fallback columns")
                    else:
                        self.logger.debug(f"[PN-TRANSFER] No columns at all for {idx_str} in {year}")
                        transfer_stats['null_values'] += 1
                        continue
                
                # Prioritize columns
                prioritized_columns = self._prioritize_columns_v2(
                    source_columns, statement_type, column_metadata
                )
                
                # Extract value with enhanced preservation
                original_values = []
                for col in prioritized_columns:
                    try:
                        val = df.loc[idx, col]
                        if pd.notna(val):
                            original_values.append((col, val))
                    except Exception as e:
                        self.logger.debug(f"Error accessing {idx}[{col}]: {e}")
                
                if is_capex and original_values:
                    self.logger.debug(f"[PN-CAPEX-TRANSFER] {idx_str} in {year}: found {len(original_values)} values: {original_values}")
                
                # Select best value with detailed logging
                if original_values:
                    # Use first non-zero value, or first value if all are zero
                    selected_value = None
                    selected_col = None
                    
                    for col, val in original_values:
                        try:
                            # Parse the value
                            if isinstance(val, str):
                                val_clean = (val.replace(',', '')
                                               .replace('(', '-')
                                               .replace(')', '')
                                               .replace('₹', '')
                                               .replace('$', '')
                                               .strip())
                                
                                if val_clean in ['-', '--', 'NA', 'N/A', 'nil', 'Nil']:
                                    continue
                                
                                numeric_val = float(val_clean)
                            else:
                                numeric_val = float(val)
                            
                            # Select first non-zero value, or first value if no non-zero found
                            if selected_value is None or (numeric_val != 0 and selected_value == 0):
                                selected_value = numeric_val
                                selected_col = col
                                
                        except Exception as e:
                            self.logger.debug(f"Failed to parse {val} from {col}: {e}")
                            continue
                    
                    if selected_value is not None:
                        # Apply data quality rules but preserve the value
                        cleaned_value, stats = self._apply_comprehensive_quality_rules(
                            selected_value, idx_str, year, statement_type
                        )
                        
                        if cleaned_value is not None:
                            restructured.loc[idx, year] = cleaned_value
                            transfer_stats['successful_transfers'] += 1
                            
                            if is_capex:
                                transfer_stats['capex_transfers'] += 1
                                self.logger.info(f"[PN-CAPEX-TRANSFER] Successfully transferred {idx_str}[{year}] = {cleaned_value} from {selected_col}")
                        else:
                            transfer_stats['failed_transfers'] += 1
                            if is_capex:
                                self.logger.warning(f"[PN-CAPEX-TRANSFER] Failed quality check for {idx_str}[{year}] = {selected_value}")
                    else:
                        transfer_stats['null_values'] += 1
                else:
                    transfer_stats['null_values'] += 1
                    if is_capex:
                        self.logger.warning(f"[PN-CAPEX-TRANSFER] No values found for {idx_str} in {year}")
        
        # 4. Post-processing with data preservation
        self.logger.info("[PN-DEBUG] Starting post-processing...")
        original_non_null = restructured.notna().sum().sum()
        
        restructured = self._conservative_post_processing(restructured, transfer_stats)
        
        final_non_null = restructured.notna().sum().sum()
        self.logger.info(f"[PN-DEBUG] Post-processing: {original_non_null} -> {final_non_null} non-null values")
        
        # 5. Comprehensive validation and reporting
        self._detailed_validation_report(restructured, transfer_stats, capex_items)
        
        self.logger.info(f"[PN-RESTRUCTURE-V5] Complete. Final shape: {restructured.shape}")
        self.logger.info(f"[PN-RESTRUCTURE-V5] Transfer success rate: {transfer_stats['successful_transfers']}/{transfer_stats['total_attempts']} ({transfer_stats['successful_transfers']/transfer_stats['total_attempts']*100:.1f}%)")
        self.logger.info(f"[PN-RESTRUCTURE-V5] CapEx transfers: {transfer_stats['capex_transfers']}")
        self.logger.info("#"*80 + "\n")
        
        return restructured
    
    def _conservative_post_processing(self, df: pd.DataFrame, stats: Dict) -> pd.DataFrame:
        """Conservative post-processing that preserves data"""
        processed = df.copy()
        
        # Only do minimal processing to preserve data integrity
        
        # 1. Forward fill balance sheet items ONLY if very few missing values
        bs_indices = [idx for idx in df.index if 'BalanceSheet::' in str(idx)]
        for idx in bs_indices:
            series = processed.loc[idx]
            null_pct = series.isna().sum() / len(series)
            
            # Only forward fill if less than 30% missing and there's a clear pattern
            if null_pct < 0.3 and series.notna().sum() >= 2:
                # Check if missing values are at the beginning or end
                first_valid = series.first_valid_index()
                last_valid = series.last_valid_index()
                
                if first_valid and last_valid:
                    # Only fill internal gaps
                    filled_series = series.copy()
                    filled_series.loc[first_valid:last_valid] = series.loc[first_valid:last_valid].fillna(method='ffill')
                    
                    # Only apply if we didn't fill too many values
                    filled_count = filled_series.notna().sum() - series.notna().sum()
                    if filled_count <= 2:  # Max 2 filled values
                        processed.loc[idx] = filled_series
                        self.logger.debug(f"Forward filled {filled_count} values for {idx}")
        
        return processed
    
    def _detailed_validation_report(self, df: pd.DataFrame, stats: Dict, original_capex_items: List):
        """Generate detailed validation report with CapEx focus"""
        self.logger.info("\n[PN-VALIDATION] Detailed Data Transfer Report:")
        
        # Overall statistics
        total_cells = df.size
        non_null_cells = df.notna().sum().sum()
        coverage = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
        
        self.logger.info(f"  - Overall Coverage: {coverage:.1f}% ({non_null_cells}/{total_cells})")
        self.logger.info(f"  - Successful Transfers: {stats['successful_transfers']}")
        self.logger.info(f"  - Failed Transfers: {stats['failed_transfers']}")
        self.logger.info(f"  - CapEx Transfers: {stats['capex_transfers']}")
        
        # CapEx specific validation
        self.logger.info(f"\n[PN-CAPEX-VALIDATION] Capital Expenditure Analysis:")
        self.logger.info(f"  - Original CapEx items found: {len(original_capex_items)}")
        
        capex_in_final = []
        for item in original_capex_items:
            if item in df.index:
                series = df.loc[item]
                non_null_count = series.notna().sum()
                if non_null_count > 0:
                    capex_in_final.append((item, non_null_count))
                    self.logger.info(f"  ✓ {item}: {non_null_count} values transferred")
                else:
                    self.logger.warning(f"  ✗ {item}: No values transferred")
            else:
                self.logger.error(f"  ✗ {item}: Not found in final data")
        
        self.logger.info(f"  - CapEx items in final data: {len(capex_in_final)}/{len(original_capex_items)}")
        
        # Per-year coverage
        year_coverage = df.notna().sum() / len(df) * 100
        self.logger.info(f"\n[PN-VALIDATION] Coverage by year:")
        for year, cov in year_coverage.items():
            self.logger.info(f"    {year}: {cov:.1f}%")
        
        # Show sample of final data
        if len(capex_in_final) > 0:
            sample_capex = capex_in_final[0][0]
            sample_data = df.loc[sample_capex].dropna()
            self.logger.info(f"\n[PN-VALIDATION] Sample CapEx data ({sample_capex}):")
            self.logger.info(f"    Values: {sample_data.to_dict()}")

    def _determine_statement_type_v2(self, idx_str: str) -> str:
        """Enhanced statement type detection with fuzzy matching"""
        idx_lower = idx_str.lower()
        
        # Direct prefix matching
        if 'profitloss::' in idx_lower or 'p&l::' in idx_lower:
            return 'ProfitLoss'
        elif 'balancesheet::' in idx_lower or 'bs::' in idx_lower:
            return 'BalanceSheet'
        elif 'cashflow::' in idx_lower or 'cf::' in idx_lower:
            return 'CashFlow'
        
        # Keyword-based detection with weights
        statement_scores = {
            'ProfitLoss': 0,
            'BalanceSheet': 0,
            'CashFlow': 0
        }
        
        # P&L keywords
        pl_keywords = {
            'revenue': 3, 'sales': 3, 'income': 2, 'expense': 2,
            'profit': 3, 'loss': 2, 'cost': 2, 'margin': 2,
            'ebit': 3, 'ebitda': 3, 'tax': 2, 'interest': 2
        }
        
        # Balance Sheet keywords
        bs_keywords = {
            'asset': 3, 'liability': 3, 'liabilities': 3, 'equity': 3,
            'cash': 2, 'receivable': 2, 'inventory': 2, 'payable': 2,
            'debt': 2, 'capital': 2, 'retained': 2, 'reserve': 2
        }
        
        # Cash Flow keywords
        cf_keywords = {
            'cash flow': 5, 'cash from': 4, 'cash used': 4,
            'operating activities': 4, 'investing activities': 4,
            'financing activities': 4, 'capex': 3, 'dividend': 2
        }
        
        # Score each statement type
        for keyword, weight in pl_keywords.items():
            if keyword in idx_lower:
                statement_scores['ProfitLoss'] += weight
        
        for keyword, weight in bs_keywords.items():
            if keyword in idx_lower:
                statement_scores['BalanceSheet'] += weight
        
        for keyword, weight in cf_keywords.items():
            if keyword in idx_lower:
                statement_scores['CashFlow'] += weight
        
        # Return highest scoring type
        max_score = max(statement_scores.values())
        if max_score > 0:
            return max(statement_scores, key=statement_scores.get)
        
        return 'Unknown'

    def _prioritize_columns_v2(self, columns: List[str], statement_type: str, 
                              column_metadata: Dict) -> List[str]:
        """Enhanced column prioritization with metadata awareness"""
        prioritized = []
        others = []
        
        for col in columns:
            col_lower = str(col).lower()
            score = 0
            
            # Statement type match
            if statement_type == 'ProfitLoss' and any(kw in col_lower for kw in ['profit', 'loss', 'p&l']):
                score += 10
            elif statement_type == 'BalanceSheet' and any(kw in col_lower for kw in ['balance', 'sheet']):
                score += 10
            elif statement_type == 'CashFlow' and any(kw in col_lower for kw in ['cash', 'flow']):
                score += 10
            
            # Pattern priority (lower is better)
            if col in column_metadata:
                score -= column_metadata[col]['priority']
            
            # Standalone indicator
            if 'standalone' in col_lower:
                score += 5
            elif 'consolidated' in col_lower:
                score -= 3
            
            if score > 0:
                prioritized.append((col, score))
            else:
                others.append((col, score))
        
        # Sort by score descending
        prioritized.sort(key=lambda x: x[1], reverse=True)
        others.sort(key=lambda x: x[1], reverse=True)
        
        return [col for col, _ in prioritized] + [col for col, _ in others]

    def _extract_best_value_v3(self, series: pd.Series, columns: List[str], 
                               metric: str, year: str) -> Optional[float]:
        """Ultimate value extraction with conflict resolution"""
        candidates = []
        
        for col in columns:
            try:
                val = series[col]
                if pd.notna(val):
                    # Parse value
                    if isinstance(val, str):
                        # Remove formatting
                        val_clean = (val.replace(',', '')
                                       .replace('(', '-')
                                       .replace(')', '')
                                       .replace('₹', '')
                                       .replace('$', '')
                                       .strip())
                        
                        # Handle special cases
                        if val_clean in ['-', '--', 'NA', 'N/A', 'nil', 'Nil']:
                            continue
                        
                        numeric_val = float(val_clean)
                    else:
                        numeric_val = float(val)
                    
                    # Skip exact zeros for selection (might be missing data)
                    if numeric_val != 0 or len(columns) == 1:
                        candidates.append({
                            'value': numeric_val,
                            'column': col,
                            'original': val
                        })
            except Exception as e:
                self.logger.debug(f"Failed to parse value from {col}: {e}")
                continue
        
        if not candidates:
            return None
        
        # Single candidate
        if len(candidates) == 1:
            return candidates[0]['value']
        
        # Multiple candidates - apply selection logic
        # Check if all values are similar (within 1%)
        values = [c['value'] for c in candidates]
        non_zero_values = [v for v in values if v != 0]
        
        if non_zero_values:
            base_val = non_zero_values[0]
            if all(abs((v - base_val) / base_val) < 0.01 for v in non_zero_values[1:]):
                return base_val
        
        # Values differ significantly - log and select based on priority
        self.logger.debug(f"Multiple different values for {metric} in {year}:")
        for candidate in candidates:
            self.logger.debug(f"  {candidate['column']}: {candidate['value']}")
        
        # Prefer non-zero values
        non_zero_candidates = [c for c in candidates if c['value'] != 0]
        if non_zero_candidates:
            return non_zero_candidates[0]['value']
        
        return candidates[0]['value']

    # PASTE THIS CODE: Replace the _apply_comprehensive_quality_rules method

    def _apply_comprehensive_quality_rules(self, value: float, metric: str, 
                                     year: str, statement_type: str) -> Tuple[Optional[float], Dict]:
        """Enhanced quality rules with better CapEx handling"""
        stats = {}
        metric_lower = metric.lower()
        
        # 1. Bounds checking - more lenient for CapEx
        if abs(value) > 1e15:
            self.logger.warning(f"Extremely large value for {metric} in {year}: {value}")
            stats['outlier_fixed'] = 1
            return None, stats
        
        # 2. Enhanced sign conventions for CapEx
        original_value = value
        
        # Capital Expenditure specific handling
        if any(kw in metric_lower for kw in ['capex', 'capital expenditure', 'purchase', 'fixed asset']):
            # CapEx should be positive for subtraction in FCF calculation
            if value < 0:
                self.logger.info(f"Converting negative CapEx to positive: {metric} in {year}: {value} -> {abs(value)}")
                value = abs(value)
                stats['sign_corrected'] = 1
            # Don't change positive values - they're correct as outflows
            
        # Cash flow items - general handling
        elif statement_type == 'CashFlow':
            if any(kw in metric_lower for kw in ['payment', 'expenditure', 'acquisition', 'dividend paid']):
                # These should be positive (outflows)
                if value < 0:
                    value = abs(value)
                    stats['sign_corrected'] = 1
            elif any(kw in metric_lower for kw in ['proceeds', 'received', 'inflow', 'from operating']):
                # These can be positive (inflows) - don't change
                pass
        
        # P&L items
        elif statement_type == 'ProfitLoss':
            # Revenue should be positive
            if any(kw in metric_lower for kw in ['revenue', 'sales', 'income from operations']):
                if value < 0:
                    self.logger.warning(f"Negative revenue for {metric} in {year}: {value}")
                    value = abs(value)
                    stats['sign_corrected'] = 1
        
        # Balance sheet items
        elif statement_type == 'BalanceSheet':
            # Assets should be positive
            if 'asset' in metric_lower and 'net' not in metric_lower:
                if value < 0:
                    self.logger.warning(f"Negative asset for {metric} in {year}: {value}")
                    value = abs(value)
                    stats['sign_corrected'] = 1
        
        # 3. Value reasonableness - more lenient
        if abs(value) < 1e-6:  # Very small values
            self.logger.debug(f"Very small value for {metric} in {year}: {value}")
        
        # 4. Don't reject values too aggressively
        return value, stats

    def _advanced_post_processing(self, df: pd.DataFrame, stats: Dict) -> pd.DataFrame:
        """Advanced post-processing with interpolation and outlier handling"""
        processed = df.copy()
        
        # 1. Forward fill balance sheet items (point-in-time data)
        bs_indices = [idx for idx in df.index if 'BalanceSheet::' in str(idx)]
        for idx in bs_indices:
            # Only forward fill if missing data is limited
            null_pct = processed.loc[idx].isna().sum() / len(processed.columns)
            if null_pct < 0.5:  # Less than 50% missing
                processed.loc[idx] = processed.loc[idx].fillna(method='ffill', limit=1)
                stats['forward_fills'] = stats.get('forward_fills', 0) + 1
        
        # 2. Interpolate flow items (P&L and Cash Flow)
        flow_indices = [idx for idx in df.index 
                       if any(prefix in str(idx) for prefix in ['ProfitLoss::', 'CashFlow::'])]
        
        for idx in flow_indices:
            series = processed.loc[idx]
            if series.notna().sum() >= 2:  # At least 2 data points
                # Linear interpolation for internal gaps only
                first_valid = series.first_valid_index()
                last_valid = series.last_valid_index()
                if first_valid and last_valid:
                    processed.loc[idx, first_valid:last_valid] = \
                        series.loc[first_valid:last_valid].interpolate(method='linear')
                    stats['interpolations'] = stats.get('interpolations', 0) + 1
        
        # 3. Detect and handle outliers using IQR method
        for idx in processed.index:
            series = processed.loc[idx].dropna()
            if len(series) >= 4:  # Need at least 4 points for IQR
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    # Use 3*IQR for financial data (more lenient)
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    if len(outliers) > 0:
                        self.logger.info(f"Outliers in {idx}: {outliers.to_dict()}")
                        
                        # Don't auto-remove outliers, just flag them
                        stats['outliers_detected'] = stats.get('outliers_detected', 0) + len(outliers)
        
        return processed

    def _comprehensive_validation(self, df: pd.DataFrame, stats: Dict):
        """Comprehensive validation with detailed reporting"""
        self.logger.info("\n[PN-VALIDATE] Data Quality Report:")
        
        # Coverage statistics
        total_cells = df.size
        non_null_cells = df.notna().sum().sum()
        coverage = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
        
        self.logger.info(f"  - Overall Coverage: {coverage:.1f}% ({non_null_cells}/{total_cells})")
        self.logger.info(f"  - Fill Rate: {(stats['filled_cells']/stats['total_cells']*100):.1f}%")
        self.logger.info(f"  - Sign Corrections: {stats.get('sign_corrections', 0)}")
        self.logger.info(f"  - Validation Failures: {stats.get('validation_failures', 0)}")
        self.logger.info(f"  - Outliers Detected: {stats.get('outliers_detected', 0)}")
        
        # Per-year coverage
        year_coverage = df.notna().sum() / len(df) * 100
        self.logger.info("  - Year Coverage:")
        for year, cov in year_coverage.items():
            self.logger.info(f"    {year}: {cov:.1f}%")
        
        # Required metrics check
        required_metrics = ['Revenue', 'Operating Income', 'Net Income', 
                           'Total Assets', 'Total Equity', 'Operating Cash Flow']
        
        missing_required = []
        for metric in required_metrics:
            found = False
            for idx in df.index:
                if self.mappings.get(idx) == metric:
                    if df.loc[idx].notna().sum() > 0:
                        found = True
                        break
            if not found:
                missing_required.append(metric)
        
        if missing_required:
            self.logger.warning(f"  - Missing Required Metrics: {missing_required}")
        else:
            self.logger.info("  - All required metrics present ✓")
        
        # Store validation results
        self.validation_results['data_quality_stats'] = stats
        self.validation_results['coverage'] = coverage
        self.validation_results['missing_required'] = missing_required

    def _validate_input_data(self):
        """Comprehensive validation of input data and mappings"""
        validation = {
            'data_quality_score': 0,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check for essential data availability
        essential_metrics = ['Total Assets', 'Total Equity', 'Revenue', 'Net Income']
        missing_essentials = []
        
        for metric in essential_metrics:
            source_metric = self._find_source_metric(metric)
            if not source_metric or source_metric not in self._df_clean.index:
                missing_essentials.append(metric)
        
        if missing_essentials:
            validation['issues'].append(f"Missing essential metrics: {', '.join(missing_essentials)}")
        
        # Check data quality in clean data
        numeric_df = self._df_clean.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            completeness = (numeric_df.notna().sum().sum() / numeric_df.size) * 100
        else:
            completeness = 0
        
        # Check for minimum years
        if len(self._df_clean.columns) < 2:
            validation['issues'].append("Insufficient time periods for meaningful analysis")
        
        # Calculate quality score
        total_mappings = len(self.mappings)
        essential_mappings = len([m for m in essential_metrics if self._find_source_metric(m)])
        
        validation['data_quality_score'] = (
            (essential_mappings / len(essential_metrics)) * 0.4 +
            (min(total_mappings, 20) / 20) * 0.3 +
            (completeness / 100) * 0.3
        ) * 100
        
        self.validation_results = validation
        
        if validation['data_quality_score'] < 50:
            self.logger.warning(f"Low data quality score: {validation['data_quality_score']:.1f}")

    def _find_source_metric(self, target_metric: str) -> Optional[str]:
        """Find source metric that maps to target"""
        for source, target in self.mappings.items():
            if target == target_metric:
                return source
        return None

    def _get_metric_series(self, target_metric: str) -> Optional[pd.Series]:
        """Get series for a target metric with error handling"""
        source_metric = self._find_source_metric(target_metric)
        if source_metric and source_metric in self._df_clean.index:
            return self._df_clean.loc[source_metric]
        return None

    def _log_metric_fetch(self, target_metric: str, source_metric: str, 
                         series: pd.Series, context: str = ""):
        """Log detailed information about metric fetching"""
        self.logger.debug(f"\n{'='*60}")
        self.logger.debug(f"[PN-TRACE] Fetching: {target_metric}")
        self.logger.debug(f"[PN-TRACE] Source: {source_metric}")
        self.logger.debug(f"[PN-TRACE] Context: {context}")
        
        if series is not None:
            values_dict = series.to_dict()
            self.logger.debug(f"[PN-TRACE] Values: {values_dict}")
            
            non_null_values = series.dropna()
            if len(non_null_values) > 0:
                self.logger.debug(f"[PN-TRACE] Stats: Count={len(non_null_values)}, "
                                f"Mean={non_null_values.mean():.2f}, "
                                f"Min={non_null_values.min():.2f}, "
                                f"Max={non_null_values.max():.2f}")
            else:
                self.logger.debug(f"[PN-TRACE] All values are null/empty")
        else:
            self.logger.debug(f"[PN-TRACE] Series is None - metric not found")
        
        self.logger.debug(f"{'='*60}\n")

    def _log_calculation(self, calc_name: str, formula: str, inputs: Dict[str, pd.Series], 
                        result: pd.Series, metadata: Dict[str, Any] = None):
        """Log detailed calculation steps"""
        self.logger.debug(f"\n{'*'*60}")
        self.logger.debug(f"[PN-CALC] Calculation: {calc_name}")
        self.logger.debug(f"[PN-CALC] Formula: {formula}")
        
        self.logger.debug(f"[PN-CALC] Inputs:")
        for name, series in inputs.items():
            if series is not None:
                self.logger.debug(f"  - {name}: {series.to_dict()}")
            else:
                self.logger.debug(f"  - {name}: None")
        
        self.logger.debug(f"[PN-CALC] Result: {result.to_dict() if result is not None else 'None'}")
        
        if metadata:
            self.logger.debug(f"[PN-CALC] Metadata: {metadata}")
        
        self.logger.debug(f"{'*'*60}\n")

    def _reformulate_balance_sheet_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced balance sheet reformulation - ALWAYS uses cached clean data"""
        # Use cached result if available
        if self._cached_bs is not None:
            return self._cached_bs
        
        # ALWAYS use clean data, ignore parameter
        df = self._df_clean
        
        self.logger.info("\n" + "="*80)
        self.logger.info("[PN-BS-START] Starting Balance Sheet Reformulation (V5 - Enhanced)")
        self.logger.info("="*80)
        
        reformulated = pd.DataFrame(index=df.columns)
        metadata = {}
        
        try:
            # Core items with validation
            total_assets = self._get_safe_series(df, 'Total Assets')
            self._log_metric_fetch('Total Assets', self._find_source_metric('Total Assets'), 
                                  total_assets, "Core BS Item")
            
            total_equity = self._get_safe_series(df, 'Total Equity')
            self._log_metric_fetch('Total Equity', self._find_source_metric('Total Equity'), 
                                  total_equity, "Core BS Item")
            
            # Total Liabilities - try to find explicit first
            total_liabilities = None
            try:
                total_liabilities = self._get_safe_series(df, 'Total Liabilities')
                self.logger.info("[PN-BS] Found explicit Total Liabilities")
                metadata['liabilities_source'] = 'explicit'
            except:
                # Calculate from accounting equation
                total_liabilities = total_assets - total_equity
                self.logger.info("[PN-BS] Calculated Total Liabilities = Assets - Equity")
                metadata['liabilities_source'] = 'calculated'
            
            # Handle NaN in liabilities
            total_liabilities = total_liabilities.fillna(0)
            
            # Current items
            current_assets = self._get_safe_series(df, 'Current Assets', default_zero=True)
            current_liabilities = self._get_safe_series(df, 'Current Liabilities', default_zero=True)
            
            # ===== ENHANCED FINANCIAL ASSETS IDENTIFICATION =====
            self.logger.info("\n[PN-BS] Starting Enhanced Financial Assets Identification")
            
            # 1. Cash and cash equivalents
            cash = pd.Series(0, index=df.columns)
            cash_items = ['Cash and Cash Equivalents', 'Cash & Cash Equivalents', 'Cash', 
                          'Cash and Equivalents', 'Cash & Equivalents']
            for item in cash_items:
                try:
                    cash_series = self._get_safe_series(df, item, default_zero=True)
                    if (cash_series > 0).any():
                        cash = cash_series
                        metadata['cash_source'] = item
                        self.logger.info(f"[PN-BS] Found cash from: {item}")
                        break
                except:
                    continue
            
            # 2. Bank balances (separate from cash)
            bank_balances = pd.Series(0, index=df.columns)
            bank_items = ['Bank Balances Other Than Cash and Cash Equivalents', 
                          'Bank Balances', 'Other Bank Balances',
                          'Bank Deposits', 'Term Deposits']
            for item in bank_items:
                try:
                    bank_series = self._get_safe_series(df, item, default_zero=True)
                    if (bank_series > 0).any():
                        bank_balances = bank_series
                        metadata['bank_balances_source'] = item
                        self.logger.info(f"[PN-BS] Found bank balances from: {item}")
                        break
                except:
                    continue
            
            # 3. Current investments
            current_investments = pd.Series(0, index=df.columns)
            curr_inv_items = ['Current Investments', 'Short-term Investments', 
                              'Marketable Securities', 'Short Term Investments',
                              'Temporary Investments', 'Trading Securities']
            for item in curr_inv_items:
                try:
                    inv_series = self._get_safe_series(df, item, default_zero=True)
                    if (inv_series > 0).any():
                        current_investments = inv_series
                        metadata['current_investments_source'] = item
                        self.logger.info(f"[PN-BS] Found current investments from: {item}")
                        break
                except:
                    continue
            
            # 4. Long-term investments
            long_term_investments = pd.Series(0, index=df.columns)
            lt_inv_items = ['Investments - Long-term', 'Long-term Investments', 
                            'Non-current Investments', 'Investment Securities',
                            'Long Term Investments', 'Available-for-Sale Securities',
                            'Held-to-Maturity Securities']
            for item in lt_inv_items:
                try:
                    inv_series = self._get_safe_series(df, item, default_zero=True)
                    if (inv_series > 0).any():
                        long_term_investments = inv_series
                        metadata['long_term_investments_source'] = item
                        self.logger.info(f"[PN-BS] Found long-term investments from: {item}")
                        break
                except:
                    continue
            
            # 5. Short-term loans (given by company)
            short_term_loans = pd.Series(0, index=df.columns)
            st_loan_items = ['Loans - Short-term', 'Short-term Loans', 'Current Loans',
                             'Loans and Advances - Short-term', 'Short Term Loans Given']
            for item in st_loan_items:
                try:
                    loan_series = self._get_safe_series(df, item, default_zero=True)
                    if (loan_series > 0).any():
                        short_term_loans = loan_series
                        metadata['short_term_loans_source'] = item
                        self.logger.info(f"[PN-BS] Found short-term loans from: {item}")
                        break
                except:
                    continue
            
            # 6. Long-term loans (given by company)
            long_term_loans = pd.Series(0, index=df.columns)
            lt_loan_items = ['Loans - Long - Term', 'Loans - Long-term', 'Long-term Loans',
                             'Non-current Loans', 'Loans and Advances - Long-term',
                             'Long Term Loans Given']
            for item in lt_loan_items:
                try:
                    loan_series = self._get_safe_series(df, item, default_zero=True)
                    if (loan_series > 0).any():
                        long_term_loans = loan_series
                        metadata['long_term_loans_source'] = item
                        self.logger.info(f"[PN-BS] Found long-term loans from: {item}")
                        break
                except:
                    continue
            
            # 7. Other financial assets - short term
            other_fin_assets_st = pd.Series(0, index=df.columns)
            other_st_items = ['Others Financial Assets - Short-term', 
                              'Other Financial Assets - Short-term',
                              'Other Current Financial Assets',
                              'Derivative Financial Assets - Current',
                              'Financial Instruments - Current']
            for item in other_st_items:
                try:
                    other_series = self._get_safe_series(df, item, default_zero=True)
                    if (other_series > 0).any():
                        other_fin_assets_st = other_series
                        metadata['other_fin_assets_st_source'] = item
                        self.logger.info(f"[PN-BS] Found other short-term financial assets from: {item}")
                        break
                except:
                    continue
            
            # 8. Other financial assets - long term
            other_fin_assets_lt = pd.Series(0, index=df.columns)
            other_lt_items = ['Others Financial Assets - Long-term',
                              'Other Financial Assets - Long-term',
                              'Other Non-current Financial Assets',
                              'Derivative Financial Assets - Non-current',
                              'Financial Instruments - Non-current']
            for item in other_lt_items:
                try:
                    other_series = self._get_safe_series(df, item, default_zero=True)
                    if (other_series > 0).any():
                        other_fin_assets_lt = other_series
                        metadata['other_fin_assets_lt_source'] = item
                        self.logger.info(f"[PN-BS] Found other long-term financial assets from: {item}")
                        break
                except:
                    continue
            
            # Total financial assets calculation
            financial_assets = (cash + bank_balances + current_investments + long_term_investments + 
                               short_term_loans + long_term_loans + other_fin_assets_st + other_fin_assets_lt)
            
            # Log financial assets breakdown
            self.logger.info("\n[PN-BS] Financial Assets Breakdown:")
            self.logger.info(f"  Cash and Equivalents: {cash.sum():,.0f}")
            self.logger.info(f"  Bank Balances: {bank_balances.sum():,.0f}")
            self.logger.info(f"  Current Investments: {current_investments.sum():,.0f}")
            self.logger.info(f"  Long-term Investments: {long_term_investments.sum():,.0f}")
            self.logger.info(f"  Short-term Loans: {short_term_loans.sum():,.0f}")
            self.logger.info(f"  Long-term Loans: {long_term_loans.sum():,.0f}")
            self.logger.info(f"  Other ST Financial Assets: {other_fin_assets_st.sum():,.0f}")
            self.logger.info(f"  Other LT Financial Assets: {other_fin_assets_lt.sum():,.0f}")
            self.logger.info(f"  Total Financial Assets: {financial_assets.sum():,.0f}")
            
            # ===== DEBT IDENTIFICATION =====
            self.logger.info("\n[PN-BS] Starting Debt Identification")
            
            # CRITICAL: Proper debt identification
            short_term_debt = pd.Series(0, index=df.columns)
            long_term_debt = pd.Series(0, index=df.columns)
            
            # Explicit debt items only
            debt_found = False
            
            # Short-term debt
            st_debt_items = [
                'Short-term Debt', 'Short Term Borrowings', 'Current Borrowings',
                'Short-term Borrowings', 'Current Debt', 'Short Term Debt',
                'Current Portion of Long-term Debt', 'Notes Payable',
                'Bank Overdrafts', 'Commercial Paper'
            ]
            
            for item in st_debt_items:
                try:
                    debt_series = self._get_safe_series(df, item, default_zero=True)
                    if (debt_series > 0).any():
                        short_term_debt = debt_series
                        metadata['short_term_debt_source'] = item
                        self.logger.info(f"[PN-BS] Found short-term debt from: {item}")
                        debt_found = True
                        break
                except:
                    continue
            
            # Long-term debt
            lt_debt_items = [
                'Long-term Debt', 'Long Term Borrowings', 'Non-current Borrowings',
                'Long-term Borrowings', 'Non-current Debt', 'Long Term Debt',
                'Bonds Payable', 'Debentures', 'Term Loans',
                'Finance Lease Obligations - Non-current'
            ]
            
            for item in lt_debt_items:
                try:
                    debt_series = self._get_safe_series(df, item, default_zero=True)
                    if (debt_series > 0).any():
                        long_term_debt = debt_series
                        metadata['long_term_debt_source'] = item
                        self.logger.info(f"[PN-BS] Found long-term debt from: {item}")
                        debt_found = True
                        break
                except:
                    continue
            
            # Total debt calculation
            total_debt = short_term_debt + long_term_debt
            
            if not debt_found:
                self.logger.warning("[PN-BS] No explicit debt found - company may be debt-free")
                metadata['debt_status'] = 'debt_free'
            else:
                metadata['debt_status'] = 'leveraged'
                self.logger.info(f"[PN-BS] Total Debt: {total_debt.sum():,.0f}")
            
            # Financial liabilities (only actual debt, not operational liabilities)
            financial_liabilities = total_debt
            
            # Net financial position
            net_financial_assets = financial_assets - financial_liabilities
            
            # Operating items (residual approach)
            operating_assets = total_assets - financial_assets
            operating_liabilities = total_liabilities - financial_liabilities
            
            # Ensure non-negative operating liabilities
            operating_liabilities = operating_liabilities.clip(lower=0)
            
            # Net Operating Assets (NOA) - key metric
            net_operating_assets = operating_assets - operating_liabilities
            
            # Common Equity (same as total equity for most companies)
            common_equity = total_equity
            
            # Build reformulated balance sheet
            reformulated['Total Assets'] = total_assets
            reformulated['Operating Assets'] = operating_assets
            reformulated['Financial Assets'] = financial_assets
            reformulated['Total Liabilities'] = total_liabilities
            reformulated['Operating Liabilities'] = operating_liabilities
            reformulated['Financial Liabilities'] = financial_liabilities
            reformulated['Net Operating Assets'] = net_operating_assets
            reformulated['Net Financial Assets'] = net_financial_assets
            reformulated['Common Equity'] = common_equity
            
            # Detailed financial assets breakdown
            reformulated['Cash and Equivalents'] = cash
            reformulated['Bank Balances'] = bank_balances
            reformulated['Current Investments'] = current_investments
            reformulated['Long-term Investments'] = long_term_investments
            reformulated['Short-term Loans'] = short_term_loans
            reformulated['Long-term Loans'] = long_term_loans
            reformulated['Other Financial Assets ST'] = other_fin_assets_st
            reformulated['Other Financial Assets LT'] = other_fin_assets_lt
            
            # Debt breakdown
            reformulated['Total Debt'] = total_debt
            reformulated['Short-term Debt'] = short_term_debt
            reformulated['Long-term Debt'] = long_term_debt
            
            # Validation check - handle NaN
            check = net_operating_assets + net_financial_assets - common_equity
            check = check.fillna(0)  # Fill NaN for check
            metadata['balance_check'] = check.abs().max()
            metadata['balance_check_pct'] = (check.abs() / common_equity.abs().replace(0, np.nan)).max() * 100 if (common_equity != 0).any() else 0
            
            self.logger.info(f"\n[PN-BS] Balance check: NOA + NFA - CE = {check.to_dict()}")
            self.logger.info(f"[PN-BS] Maximum absolute difference: {metadata['balance_check']:.2f}")
            self.logger.info(f"[PN-BS] Maximum percentage difference: {metadata['balance_check_pct']:.2f}%")
            
            # Additional validation - components breakdown
            self.logger.info("\n[PN-BS] Component Validation:")
            fa_ratio = (financial_assets / total_assets * 100).mean()
            oa_ratio = (operating_assets / total_assets * 100).mean()
            self.logger.info(f"  Financial Assets as % of Total Assets (avg): {fa_ratio:.1f}%")
            self.logger.info(f"  Operating Assets as % of Total Assets (avg): {oa_ratio:.1f}%")
            
            if financial_liabilities.sum() > 0:
                debt_equity_ratio = (total_debt / common_equity).mean()
                self.logger.info(f"  Debt-to-Equity Ratio (avg): {debt_equity_ratio:.2f}")
            
            # Summary statistics
            self.logger.info("\n[PN-BS-SUMMARY] Balance Sheet Reformulation Summary:")
            self.logger.info(f"  Total Assets: {total_assets.sum():,.0f}" if not total_assets.isna().all() else "N/A")
            self.logger.info(f"  - Operating Assets: {operating_assets.sum():,.0f}" if not operating_assets.isna().all() else "N/A")
            self.logger.info(f"  - Financial Assets: {financial_assets.sum():,.0f}" if not financial_assets.isna().all() else "N/A")
            self.logger.info(f"  Total Liabilities: {total_liabilities.sum():,.0f}" if not total_liabilities.isna().all() else "N/A")
            self.logger.info(f"  - Operating Liabilities: {operating_liabilities.sum():,.0f}" if not operating_liabilities.isna().all() else "N/A")
            self.logger.info(f"  - Financial Liabilities: {financial_liabilities.sum():,.0f}" if not financial_liabilities.isna().all() else "N/A")
            self.logger.info(f"  Total Equity: {total_equity.sum():,.0f}" if not total_equity.isna().all() else "N/A")
            self.logger.info(f"  Net Operating Assets (NOA): {net_operating_assets.sum():,.0f}" if not net_operating_assets.isna().all() else "N/A")
            self.logger.info(f"  Net Financial Assets (NFA): {net_financial_assets.sum():,.0f}" if not net_financial_assets.isna().all() else "N/A")
            self.logger.info(f"  Total Debt: {total_debt.sum():,.0f}" if not total_debt.isna().all() else "N/A")
            
            # Store metadata
            metadata['financial_assets_components'] = {
                'cash': cash.sum(),
                'bank_balances': bank_balances.sum(),
                'current_investments': current_investments.sum(),
                'long_term_investments': long_term_investments.sum(),
                'short_term_loans': short_term_loans.sum(),
                'long_term_loans': long_term_loans.sum(),
                'other_st': other_fin_assets_st.sum(),
                'other_lt': other_fin_assets_lt.sum(),
                'total': financial_assets.sum()
            }
            
        except Exception as e:
            self.logger.error(f"[PN-BS-ERROR] Balance sheet reformulation failed: {e}", exc_info=True)
            raise
        
        self.calculation_metadata['balance_sheet'] = metadata
        
        self.logger.info("\n[PN-BS-END] Balance Sheet Reformulation Complete")
        self.logger.info("="*80 + "\n")
        
        # Cache result
        self._cached_bs = reformulated.T
        return self._cached_bs

    def _reformulate_income_statement_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced Income Statement Reformulation with proper tax allocation
        Version 6.3 - Simplified exceptional items handling (always treated as non-operating)
        """
        # Use cached result if available
        if self._cached_is is not None:
            return self._cached_is
        
        # ALWAYS use clean data
        df = self._df_clean
        
        self.logger.info("\n" + "="*80)
        self.logger.info("[PN-IS-V6.3] Starting Enhanced Income Statement Reformulation")
        self.logger.info("="*80)
        
        reformulated = pd.DataFrame(index=df.columns)
        metadata = {}
        
        try:
            # 1. REVENUE - Core top line
            revenue = self._get_safe_series(df, 'Revenue')
            reformulated['Revenue'] = revenue
            
            if (revenue.dropna() <= 0).all():
                raise ValueError("Revenue is zero or negative for all periods!")
            
            self.logger.info(f"[PN-IS-V6.3] Revenue range: {revenue.min():,.0f} to {revenue.max():,.0f}")
            
            # 2. OPERATING INCOME (PBIT) - May include Other Income
            pbit_with_other_income = None
            
            # Try multiple sources for operating income
            op_income_attempts = [
                ('Operating Income', 'Direct mapping'),
                ('EBIT', 'EBIT as proxy'),
                ('Operating Profit', 'Operating Profit as proxy'),
                ('Profit Before Exceptional Items and Tax', 'PBIT before exceptional items'),
                ('Profit Before Interest and Tax', 'Standard PBIT')
            ]
            
            for attempt_metric, description in op_income_attempts:
                try:
                    temp_oi = self._get_safe_series(df, attempt_metric, default_zero=False)
                    if temp_oi is not None and (temp_oi != 0).any():
                        pbit_with_other_income = temp_oi
                        metadata['operating_income_source'] = f"{attempt_metric} ({description})"
                        self.logger.info(f"[PN-IS-V6.3] Using {attempt_metric} for PBIT (may include Other Income)")
                        break
                except:
                    continue
            
            if pbit_with_other_income is None:
                raise ValueError("Could not find Operating Income in any form!")
            
            # 2A. ISOLATE NON-OPERATING INCOME - FIXED LOGIC
            self.logger.info("[PN-IS-V6.3] Analyzing Other Income composition...")
            
            # Get Interest Income and Other Income separately
            interest_income = self._get_safe_series(df, 'Interest Income', default_zero=True)
            other_income = self._get_safe_series(df, 'Other Income', default_zero=True)
            
            # CRITICAL FIX: Check if values are identical (not just sources)
            values_are_identical = False
            if interest_income is not None and other_income is not None:
                # Compare the actual values, not just the sources
                values_are_identical = (interest_income.round(2) == other_income.round(2)).all()
                
                if values_are_identical:
                    self.logger.info("[PN-IS-V6.3] DETECTED: Interest Income and Other Income have identical values")
                    self.logger.info(f"[PN-IS-V6.3] Interest Income: {interest_income.head().to_dict()}")
                    self.logger.info(f"[PN-IS-V6.3] Other Income: {other_income.head().to_dict()}")
            
            # Determine total non-operating income based on whether they're the same
            if values_are_identical:
                # They're the same - use only one
                total_non_operating_income = interest_income  # or other_income, they're the same
                metadata['other_income_treatment'] = 'Identical to Interest Income - using single value'
                self.logger.info("[PN-IS-V6.3] Using single value to avoid double-counting")
            else:
                # They're different - need to analyze composition
                self.logger.info("[PN-IS-V6.3] Interest and Other Income are different")
                
                # Check if they're from same source (legacy check)
                interest_income_source = self._find_source_metric('Interest Income')
                other_income_source = self._find_source_metric('Other Income')
                
                if other_income_source and interest_income_source and (other_income_source == interest_income_source):
                    # Same source but different values? Unusual case
                    total_non_operating_income = other_income
                    metadata['other_income_treatment'] = 'Same source, different values - using Other Income only'
                    self.logger.warning("[PN-IS-V6.3] Unusual case: same source but different values")
                else:
                    # Different sources - analyze Other Income composition
                    if self._analyze_other_income_composition(other_income_source, df):
                        # Other Income likely contains operating items
                        total_non_operating_income = interest_income  # Only use explicit interest
                        metadata['other_income_treatment'] = 'Mixed content - only using Interest Income as non-operating'
                        self.logger.info("[PN-IS-V6.3] Other Income appears mixed - conservative approach")
                    else:
                        # Other Income appears to be purely non-operating
                        total_non_operating_income = interest_income + other_income
                        metadata['other_income_treatment'] = 'Purely non-operating - using sum of both'
                        self.logger.info("[PN-IS-V6.3] Treating all Other Income as non-operating")
            
            # Log the adjustment
            self.logger.info(f"[PN-IS-V6.3] Total non-operating income: {total_non_operating_income.to_dict()}")
            
            # 2B. CALCULATE PURE OPERATING INCOME
            pure_operating_income = pbit_with_other_income - total_non_operating_income
            self.logger.info(f"[PN-IS-V6.3] Pure Operating Income = PBIT - Non-operating")
            self.logger.info(f"[PN-IS-V6.3] {pure_operating_income.head().to_dict()} = {pbit_with_other_income.head().to_dict()} - {total_non_operating_income.head().to_dict()}")
            
            reformulated['Operating Income Before Tax'] = pure_operating_income
            
            # 2C. HANDLE EXCEPTIONAL ITEMS - Simplified: Always treat as non-operating, no subtraction
            self.logger.info("[PN-IS-V6.3] Analyzing Exceptional Items (always treated as non-operating)...")
            
            exceptional_items = pd.Series(0, index=df.columns)
            exceptional_found = False
            exceptional_location = 'not_found'
            
            # Check multiple possible locations
            exceptional_sources = [
                ('Exceptional Items', 'standard'),
                ('Exceptional Items Before Tax', 'explicit_pretax'),
                ('Exceptional and Extraordinary Items', 'combined'),
                ('Prior Period Items', 'prior_period'),
                ('Exceptional Items (Before Tax)', 'parenthetical')
            ]
            
            for source, location_type in exceptional_sources:
                try:
                    temp_exceptional = self._get_safe_series(df, source, default_zero=True)
                    if (temp_exceptional != 0).any():
                        exceptional_items = temp_exceptional
                        exceptional_found = True
                        exceptional_location = location_type
                        self.logger.info(f"[PN-IS-V6.3] Found exceptional items in: {source} ({location_type})")
                        metadata['exceptional_adjustment'] = 'Treated as non-operating - no subtraction from operating income'
                        self.logger.info("[PN-IS-V6.3] Exceptional items treated as non-operating (no subtraction)")
                        break
                except:
                    continue
            
            if not exceptional_found:
                self.logger.info("[PN-IS-V6.3] No exceptional items found")
                metadata['exceptional_items'] = 'None found'
            else:
                metadata['exceptional_items'] = f'Found in {exceptional_location}'
            
            # 3. TAX ALLOCATION - CRITICAL CHANGE: Use actual tax expense
            self.logger.info("[PN-IS-V6.3] Allocating actual tax expense...")
            
            # Get actual reported values
            tax_expense = self._get_safe_series(df, 'Tax Expense')
            income_before_tax = self._get_safe_series(df, 'Income Before Tax')
            interest_expense = self._get_safe_series(df, 'Interest Expense', default_zero=True)
            
            # Validate interest expense
            if (abs(interest_expense.dropna()) < 1).all():
                self.logger.info("[PN-IS-V6.3] Negligible interest expense detected, treating as zero")
                interest_expense = pd.Series(0, index=df.columns)
            
            # Calculate components for tax allocation
            # Total pre-tax income = Operating Income + Net Financial Income + Exceptional Items
            net_financial_income_before_tax = total_non_operating_income - interest_expense
            total_pretax_income_components = pure_operating_income + net_financial_income_before_tax + exceptional_items
            
            # Allocate tax expense proportionally based on income components
            # But only for positive income components (negative components get tax benefit)
            reformulated['Tax Expense (Actual)'] = tax_expense
            
            # Method 1: Proportional allocation based on positive income
            tax_on_operating = pd.Series(index=df.columns, dtype=float)
            tax_on_financial = pd.Series(index=df.columns, dtype=float)
            tax_on_exceptional = pd.Series(index=df.columns, dtype=float)
            
            for year in df.columns:
                try:
                    total_tax = tax_expense.get(year, 0)
                    op_income = pure_operating_income.get(year, 0)
                    fin_income = net_financial_income_before_tax.get(year, 0)
                    except_items = exceptional_items.get(year, 0)
                    
                    # Calculate total positive income for allocation base
                    positive_base = max(0, op_income) + max(0, fin_income) + max(0, except_items)
                    
                    if positive_base > 0 and pd.notna(total_tax):
                        # Allocate tax proportionally to positive income components
                        if op_income > 0:
                            tax_on_operating[year] = total_tax * (op_income / positive_base)
                        else:
                            tax_on_operating[year] = 0
                        
                        if fin_income > 0:
                            tax_on_financial[year] = total_tax * (fin_income / positive_base)
                        else:
                            tax_on_financial[year] = 0
                        
                        if except_items > 0:
                            tax_on_exceptional[year] = total_tax * (except_items / positive_base)
                        else:
                            tax_on_exceptional[year] = 0
                    else:
                        # If no positive income or no tax, allocate based on income before tax ratio
                        if pd.notna(income_before_tax.get(year)) and income_before_tax.get(year) != 0:
                            effective_rate = total_tax / income_before_tax.get(year)
                            tax_on_operating[year] = op_income * effective_rate
                            tax_on_financial[year] = fin_income * effective_rate
                            tax_on_exceptional[year] = except_items * effective_rate
                        else:
                            # Last resort: allocate all to operating
                            tax_on_operating[year] = total_tax
                            tax_on_financial[year] = 0
                            tax_on_exceptional[year] = 0
                    
                    self.logger.debug(f"[PN-IS-V6.3] Year {year} - Tax allocation: Op={tax_on_operating[year]:.2f}, Fin={tax_on_financial[year]:.2f}, Except={tax_on_exceptional[year]:.2f}")
                    
                except Exception as e:
                    self.logger.warning(f"[PN-IS-V6.3] Tax allocation failed for {year}: {e}")
                    # Fallback: assign all tax to operating
                    tax_on_operating[year] = tax_expense.get(year, 0)
                    tax_on_financial[year] = 0
                    tax_on_exceptional[year] = 0
            
            # Store tax allocations
            reformulated['Tax on Operating Income'] = tax_on_operating
            reformulated['Tax on Financial Income'] = tax_on_financial
            reformulated['Tax on Exceptional Items'] = tax_on_exceptional
            
            # Calculate after-tax amounts
            reformulated['Operating Income After Tax'] = pure_operating_income - tax_on_operating
            reformulated['Net Financial Income After Tax'] = net_financial_income_before_tax - tax_on_financial
            reformulated['Exceptional Items After Tax'] = exceptional_items - tax_on_exceptional
            
            # Store effective tax rate for information only
            effective_tax_rate = pd.Series(index=df.columns, dtype=float)
            mask = (income_before_tax != 0) & (income_before_tax.notna()) & (tax_expense.notna())
            effective_tax_rate[mask] = tax_expense[mask] / income_before_tax[mask]
            reformulated['Effective Tax Rate (Info Only)'] = effective_tax_rate
            
            # 4. FINANCIAL ITEMS
            self.logger.info("[PN-IS-V6.3] Processing financial items...")
            
            # Store components
            reformulated['Interest Expense'] = interest_expense
            reformulated['Interest Income'] = interest_income
            
            # CRITICAL FIX: Only store "Other Non-Operating Income" if it's actually different from Interest Income
            if not values_are_identical:
                reformulated['Other Non-Operating Income'] = other_income
            else:
                reformulated['Other Non-Operating Income'] = pd.Series(0, index=df.columns)
                
            reformulated['Total Non-Operating Income'] = total_non_operating_income
            reformulated['Net Financial Income Before Tax'] = net_financial_income_before_tax
            
            # CRITICAL: Create Net Financial Expense (negative of income) for NBC calculation
            reformulated['Net Financial Expense After Tax'] = -reformulated['Net Financial Income After Tax']
            
            # 5. NET INCOME RECONCILIATION
            self.logger.info("[PN-IS-V6.3] Performing net income reconciliation...")
            
            net_income_reported = self._get_safe_series(df, 'Net Income')
            
            # Calculate Net Income: NOPAT + Net Financial Income + Exceptional Items
            calculated_net_income = (
                reformulated['Operating Income After Tax'] + 
                reformulated['Net Financial Income After Tax'] +
                reformulated['Exceptional Items After Tax']
            )
            
            reformulated['Net Income (Reported)'] = net_income_reported
            reformulated['Net Income (Calculated)'] = calculated_net_income
            
            # Reconciliation check
            reconciliation_diff = abs(net_income_reported - calculated_net_income)
            max_diff = reconciliation_diff.max()
            
            if max_diff > 1:  # Allow 1 unit tolerance
                self.logger.warning(f"[PN-IS-V6.3] Net Income reconciliation has differences up to {max_diff:.2f}")
                metadata['reconciliation_status'] = f'Mismatch: max diff = {max_diff:.2f}'
                
                # Detailed year-by-year reconciliation
                for year in df.columns:
                    reported = net_income_reported.get(year)
                    calculated = calculated_net_income.get(year)
                    if pd.notna(reported) and pd.notna(calculated):
                        diff = abs(reported - calculated)
                        if diff > 1:
                            self.logger.warning(
                                f"[PN-IS-V6.3] Year {year}: "
                                f"Reported={reported:.2f}, Calculated={calculated:.2f}, Diff={diff:.2f}"
                            )
            else:
                self.logger.info("[PN-IS-V6.3] Net Income reconciliation successful")
                metadata['reconciliation_status'] = 'Success'
            
            # Additional logging for debugging
            self.logger.info("\n[PN-IS-V6.3] Reconciliation Components:")
            self.logger.info(f"Operating Income After Tax: {reformulated['Operating Income After Tax'].head().to_dict()}")
            self.logger.info(f"Net Financial Income After Tax: {reformulated['Net Financial Income After Tax'].head().to_dict()}")
            self.logger.info(f"Exceptional Items After Tax: {reformulated['Exceptional Items After Tax'].head().to_dict()}")
            self.logger.info(f"Calculated Net Income: {calculated_net_income.head().to_dict()}")
            self.logger.info(f"Reported Net Income: {net_income_reported.head().to_dict()}")
            
            # Store metadata
            metadata['tax_allocation_method'] = 'Proportional based on positive income components'
            
            # 6. ADDITIONAL METRICS FOR ANALYSIS
            
            # Gross Profit (if available)
            try:
                gross_profit = self._get_safe_series(df, 'Gross Profit', default_zero=True)
                if (gross_profit == 0).all():
                    # Try to calculate: Revenue - COGS
                    cogs = self._get_safe_series(df, 'Cost of Goods Sold', default_zero=True)
                    if (cogs != 0).any():
                        gross_profit = revenue - cogs
                reformulated['Gross Profit'] = gross_profit
            except:
                pass
            
            # EBITDA (if we have depreciation)
            try:
                depreciation = self._get_safe_series(df, 'Depreciation', default_zero=True)
                if (depreciation != 0).any():
                    ebitda = pure_operating_income + depreciation
                    reformulated['EBITDA'] = ebitda
            except:
                pass
            
            # 7. SUMMARY AND VALIDATION
            self.logger.info("\n[PN-IS-V6.3-SUMMARY] Income Statement Reformulation Summary:")
            self.logger.info(f"  Revenue range: {revenue.min():,.0f} to {revenue.max():,.0f}")
            self.logger.info(f"  Pure Operating Income range: {pure_operating_income.min():,.0f} to {pure_operating_income.max():,.0f}")
            self.logger.info(f"  Net Income range: {net_income_reported.min():,.0f} to {net_income_reported.max():,.0f}")
            self.logger.info(f"  Tax Expense range: {tax_expense.min():,.0f} to {tax_expense.max():,.0f}")
            self.logger.info(f"  Other Income treatment: {metadata.get('other_income_treatment', 'N/A')}")
            self.logger.info(f"  Exceptional items: {metadata.get('exceptional_items', 'N/A')}")
            self.logger.info(f"  Reconciliation: {metadata.get('reconciliation_status', 'N/A')}")
            
        except Exception as e:
            self.logger.error(f"[PN-IS-V6.3-ERROR] Income statement reformulation failed: {e}", exc_info=True)
            raise
        
        self.calculation_metadata['income_statement'] = metadata
        
        self.logger.info("\n[PN-IS-V6.3-END] Income Statement Reformulation Complete")
        self.logger.info("="*80 + "\n")
        
        self._cached_is = reformulated.T
        return self._cached_is
    
    def _analyze_other_income_composition(self, other_income_source: str, df: pd.DataFrame) -> bool:
        """
        Analyze whether Other Income likely contains operating items
        Returns True if Other Income appears to contain operating items
        """
        if not other_income_source:
            return False
        
        # Keywords suggesting operating other income
        operating_keywords = [
            'forex', 'foreign exchange', 'exchange difference',
            'scrap', 'sale of scrap', 'commission', 'operating',
            'rent', 'lease', 'service', 'miscellaneous receipts',
            'duty drawback', 'export incentive', 'subsidy'
        ]
        
        # Keywords suggesting non-operating other income
        non_operating_keywords = [
            'dividend', 'interest', 'investment', 'profit on sale',
            'gain on investment', 'treasury', 'mutual fund'
        ]
        
        source_lower = other_income_source.lower()
        
        # Check the metric name itself
        operating_score = sum(1 for keyword in operating_keywords if keyword in source_lower)
        non_operating_score = sum(1 for keyword in non_operating_keywords if keyword in source_lower)
        
        # Check nearby items in the dataframe for context
        try:
            # Get the position of Other Income
            all_indices = list(df.index)
            if other_income_source in all_indices:
                position = all_indices.index(other_income_source)
                
                # Check items around it (within 3 positions)
                start = max(0, position - 3)
                end = min(len(all_indices), position + 4)
                
                nearby_items = all_indices[start:end]
                nearby_text = ' '.join([str(item).lower() for item in nearby_items])
                
                # Update scores based on context
                operating_score += sum(1 for keyword in operating_keywords if keyword in nearby_text) * 0.5
                non_operating_score += sum(1 for keyword in non_operating_keywords if keyword in nearby_text) * 0.5
        except:
            pass
        
        # If no clear indication, check the magnitude relative to revenue
        if operating_score == 0 and non_operating_score == 0:
            try:
                revenue = self._get_safe_series(df, 'Revenue')
                other_income = df.loc[other_income_source]
                
                # If Other Income is > 5% of revenue, it likely contains operating items
                other_income_pct = (other_income / revenue * 100).mean()
                if other_income_pct > 5:
                    self.logger.info(f"[PN-IS-V6] Other Income is {other_income_pct:.1f}% of revenue - likely contains operating items")
                    return True
            except:
                pass
        
        # Decision based on scores
        if operating_score > non_operating_score:
            self.logger.info(f"[PN-IS-V6] Other Income analysis: Operating score={operating_score}, Non-operating score={non_operating_score}")
            return True
        
        return False
    
    def _check_exceptional_in_operating(self, exceptional_source: str, 
                                       pbit_with_other: pd.Series, 
                                       pure_operating: pd.Series) -> bool:
        """
        Check if exceptional items are already included in operating income
        """
        # If the pure operating income calculation already removed something significant,
        # exceptional items might have been in the original PBIT
        
        # Check if there's a significant difference that matches exceptional items magnitude
        try:
            diff = pbit_with_other - pure_operating
            exceptional = self._get_safe_series(self._df_clean, exceptional_source, default_zero=True)
            
            # If the difference closely matches exceptional items, they were likely included
            for year in diff.index:
                if pd.notna(diff[year]) and pd.notna(exceptional[year]):
                    if abs(diff[year] - exceptional[year]) < abs(exceptional[year]) * 0.1:  # Within 10%
                        self.logger.info(f"[PN-IS-V6] Exceptional items appear to be included in operating income for {year}")
                        return True
        except:
            pass
        
        # Check naming conventions
        exceptional_lower = exceptional_source.lower()
        
        # If it says "before tax" or "below operating", it's likely not included
        if any(phrase in exceptional_lower for phrase in ['before tax', 'below operating', 'after operating']):
            return False
        
        # Conservative default - assume not included
        return False
    
    def _calculate_ratios_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced ratio calculations with comprehensive error handling and edge case management
        Version 7.0 - Enhanced FLEV, NBC, Spread, and Gross Margin calculations
        """
        
        self.logger.info("[PN-RATIOS-START] Starting Enhanced Ratio Calculations (V7.0)")
        self.logger.info("="*80)
        
        # Get reformulated statements (already cached)
        ref_bs = self._reformulate_balance_sheet_enhanced(None)
        ref_is = self._reformulate_income_statement_enhanced(None)
        
        self.logger.info(f"[PN-RATIOS-DATA] Reformulated BS shape: {ref_bs.shape}")
        self.logger.info(f"[PN-RATIOS-DATA] Reformulated IS shape: {ref_is.shape}")
        
        # Initialize ratios DataFrame with years as index (will transpose at end)
        ratios = pd.DataFrame(index=ref_bs.columns)
        metadata = {}
        
        # Track calculation status
        calculation_status = {
            'successful': [],
            'failed': [],
            'warnings': []
        }
        
        try:
            # ========== 1. CORE PENMAN-NISSIM RATIOS ==========
            
            # --- RNOA (Return on Net Operating Assets) ---
            self.logger.info("\n[PN-RATIOS-RNOA] CALCULATING RNOA...")
            
            rnoa_calculated = False
            if 'Operating Income After Tax' in ref_is.index and 'Net Operating Assets' in ref_bs.index:
                try:
                    nopat = ref_is.loc['Operating Income After Tax']
                    noa = ref_bs.loc['Net Operating Assets']
                    
                    # Use average NOA for more accurate calculation
                    avg_noa = noa.rolling(window=2, min_periods=1).mean()
                    
                    # Handle division by zero
                    rnoa = pd.Series(index=ref_bs.columns, dtype=float)
                    mask = (avg_noa != 0) & avg_noa.notna()
                    rnoa[mask] = (nopat[mask] / avg_noa[mask]) * 100
                    rnoa[~mask] = np.nan
                    
                    ratios['Return on Net Operating Assets (RNOA) %'] = rnoa
                    rnoa_calculated = True
                    calculation_status['successful'].append('RNOA')
                    
                    self._log_calculation(
                        "RNOA",
                        "(NOPAT / Average NOA) × 100",
                        {"NOPAT": nopat, "Average NOA": avg_noa},
                        rnoa,
                        {"formula": "RNOA = (Operating Income After Tax / Average Net Operating Assets) × 100"}
                    )
                    
                    self.logger.info(f"[PN-RATIOS-RNOA] CALCULATED RNOA VALUES: {rnoa.to_dict()}")
                    
                    # --- RNOA Components: OPM and NOAT ---
                    if 'Revenue' in ref_is.index:
                        revenue = ref_is.loc['Revenue']
                        
                        # Operating Profit Margin (OPM)
                        opm = pd.Series(index=ref_bs.columns, dtype=float)
                        mask = (revenue != 0) & revenue.notna()
                        opm[mask] = (nopat[mask] / revenue[mask]) * 100
                        opm[~mask] = np.nan
                        ratios['Operating Profit Margin (OPM) %'] = opm
                        
                        # Net Operating Asset Turnover (NOAT)
                        noat = pd.Series(index=ref_bs.columns, dtype=float)
                        mask = (avg_noa != 0) & avg_noa.notna()
                        noat[mask] = revenue[mask] / avg_noa[mask]
                        noat[~mask] = np.nan
                        ratios['Net Operating Asset Turnover (NOAT)'] = noat
                        
                        # Verify RNOA = OPM × NOAT
                        calculated_rnoa = (opm * noat) / 100
                        decomposition_diff = (rnoa - calculated_rnoa).abs().max()
                        metadata['rnoa_decomposition_check'] = decomposition_diff
                        
                        if decomposition_diff > 0.1:
                            calculation_status['warnings'].append(f'RNOA decomposition mismatch: {decomposition_diff:.2f}')
                        
                        self.logger.info("\n[PN-RATIOS-RNOA-DECOMP] RNOA Decomposition:")
                        self.logger.info(f"  OPM: {opm.to_dict()}")
                        self.logger.info(f"  NOAT: {noat.to_dict()}")
                        self.logger.info(f"  Decomposition check diff: {decomposition_diff:.4f}")
                        
                except Exception as e:
                    self.logger.error(f"[PN-RATIOS-RNOA-ERROR] Failed to calculate RNOA: {e}")
                    calculation_status['failed'].append(f'RNOA: {str(e)}')
                    ratios['Return on Net Operating Assets (RNOA) %'] = pd.Series(np.nan, index=ref_bs.columns)
            else:
                self.logger.warning("[PN-RATIOS-RNOA] Missing data for RNOA calculation")
                ratios['Return on Net Operating Assets (RNOA) %'] = pd.Series(np.nan, index=ref_bs.columns)
                calculation_status['failed'].append('RNOA: Missing required data')
            
            # --- ENHANCED Financial Leverage (FLEV) ---
            self.logger.info("\n[PN-RATIOS-FLEV] Calculating Financial Leverage (Enhanced)...")
            
            flev_calculated = False
            if 'Net Financial Assets' in ref_bs.index and 'Common Equity' in ref_bs.index:
                try:
                    nfa = ref_bs.loc['Net Financial Assets']
                    ce = ref_bs.loc['Common Equity']
                    
                    # Use beginning-of-period equity for consistency
                    avg_ce = ce.rolling(window=2, min_periods=1).mean()
                    
                    # FLEV = -NFA/CE (negative NFA means net debt)
                    flev = pd.Series(index=ref_bs.columns, dtype=float)
                    mask = (avg_ce != 0) & avg_ce.notna()
                    flev[mask] = -nfa[mask] / avg_ce[mask]
                    flev[~mask] = np.nan
                    
                    # Enhanced: Also calculate using end-of-period values for comparison
                    flev_end = pd.Series(index=ref_bs.columns, dtype=float)
                    mask_end = (ce != 0) & ce.notna()
                    flev_end[mask_end] = -nfa[mask_end] / ce[mask_end]
                    
                    ratios['Financial Leverage (FLEV)'] = flev
                    ratios['Financial Leverage (End of Period)'] = flev_end
                    flev_calculated = True
                    calculation_status['successful'].append('FLEV')
                    
                    # Log leverage status with more detail
                    self.logger.info(f"[PN-RATIOS-FLEV] FLEV values (avg CE): {flev.to_dict()}")
                    self.logger.info(f"[PN-RATIOS-FLEV] FLEV values (end CE): {flev_end.to_dict()}")
                    
                    # Analyze leverage position for each period
                    for year in flev.index:
                        if pd.notna(flev[year]):
                            if flev[year] < -0.5:
                                self.logger.info(f"[PN-RATIOS-FLEV] {year}: Strong net cash position (FLEV={flev[year]:.2f})")
                            elif flev[year] < 0:
                                self.logger.info(f"[PN-RATIOS-FLEV] {year}: Net cash position (FLEV={flev[year]:.2f})")
                            elif flev[year] < 0.5:
                                self.logger.info(f"[PN-RATIOS-FLEV] {year}: Low leverage (FLEV={flev[year]:.2f})")
                            elif flev[year] < 1.0:
                                self.logger.info(f"[PN-RATIOS-FLEV] {year}: Moderate leverage (FLEV={flev[year]:.2f})")
                            else:
                                self.logger.warning(f"[PN-RATIOS-FLEV] {year}: High leverage (FLEV={flev[year]:.2f})")
                    
                    # Alternative leverage metrics
                    if 'Total Debt' in ref_bs.index:
                        total_debt = ref_bs.loc['Total Debt']
                        
                        # Debt to Equity
                        debt_to_equity = pd.Series(index=ref_bs.columns, dtype=float)
                        mask = (avg_ce != 0) & avg_ce.notna()
                        debt_to_equity[mask] = total_debt[mask] / avg_ce[mask]
                        debt_to_equity[~mask] = np.nan
                        ratios['Debt to Equity'] = debt_to_equity
                        
                        # Net Debt to Equity
                        if 'Cash and Equivalents' in ref_bs.index:
                            cash = ref_bs.loc['Cash and Equivalents']
                            net_debt = total_debt - cash
                            net_debt_to_equity = pd.Series(index=ref_bs.columns, dtype=float)
                            net_debt_to_equity[mask] = net_debt[mask] / avg_ce[mask]
                            net_debt_to_equity[~mask] = np.nan
                            ratios['Net Debt to Equity'] = net_debt_to_equity
                        
                except Exception as e:
                    self.logger.error(f"[PN-RATIOS-FLEV-ERROR] Failed to calculate FLEV: {e}")
                    calculation_status['failed'].append(f'FLEV: {str(e)}')
                    ratios['Financial Leverage (FLEV)'] = pd.Series(np.nan, index=ref_bs.columns)
            else:
                self.logger.warning("[PN-RATIOS-FLEV] Missing data for FLEV calculation")
                ratios['Financial Leverage (FLEV)'] = pd.Series(np.nan, index=ref_bs.columns)
                calculation_status['failed'].append('FLEV: Missing required data')
            
            # --- ENHANCED Net Borrowing Cost (NBC) ---
            self.logger.info("\n[PN-RATIOS-NBC] Calculating Net Borrowing Cost (Ultra-Enhanced)...")
            
            nbc = pd.Series(0.0, index=ref_bs.columns)
            nbc_calculated = False
            
            try:
                if 'Net Financial Expense After Tax' in ref_is.index and 'Net Financial Assets' in ref_bs.index:
                    nfe_after_tax = ref_is.loc['Net Financial Expense After Tax']
                    nfa = ref_bs.loc['Net Financial Assets']
                    
                    # Use beginning-of-period NFA for consistency
                    avg_nfa = nfa.rolling(window=2, min_periods=1).mean()
                    
                    # Net Financial Obligations = -Net Financial Assets
                    avg_net_financial_obligations = -avg_nfa
                    
                    # Enhanced NBC calculation with detailed logging
                    for year in ref_bs.columns:
                        if pd.notna(avg_net_financial_obligations[year]) and pd.notna(nfe_after_tax[year]):
                            if avg_net_financial_obligations[year] > 10:  # Small threshold to avoid division issues
                                # Company has net debt
                                nbc[year] = (nfe_after_tax[year] / avg_net_financial_obligations[year]) * 100
                                self.logger.info(f"[PN-RATIOS-NBC] {year}: Net debt position, NBC = {nbc[year]:.2f}%")
                            elif avg_net_financial_obligations[year] < -10:  # Company has significant net cash
                                # For companies with net cash, NBC represents return on financial assets
                                # Use negative of the ratio to show it as a benefit
                                nbc[year] = (nfe_after_tax[year] / avg_net_financial_obligations[year]) * 100
                                self.logger.info(f"[PN-RATIOS-NBC] {year}: Net cash position, NBC = {nbc[year]:.2f}% (negative = income)")
                            else:
                                # Near-zero net position
                                nbc[year] = 0.0
                                self.logger.info(f"[PN-RATIOS-NBC] {year}: Near-zero net financial position, NBC = 0%")
                    
                    # Cap NBC at reasonable bounds
                    nbc = nbc.clip(lower=-20, upper=30)
                    
                    nbc_calculated = True
                    calculation_status['successful'].append('NBC')
                    
                    self.logger.info(f"[PN-RATIOS-NBC] Final NBC values: {nbc.to_dict()}")
                    
                else:
                    self.logger.warning("[PN-RATIOS-NBC] Missing data for NBC calculation, using zero")
                    calculation_status['warnings'].append('NBC: Missing data, defaulting to zero')
                    
            except Exception as e:
                self.logger.error(f"[PN-RATIOS-NBC-ERROR] NBC calculation failed: {e}")
                calculation_status['failed'].append(f'NBC: {str(e)}')
            
            ratios['Net Borrowing Cost (NBC) %'] = nbc
            
            # --- Alternative NBC calculations for validation ---
            self.logger.info("\n[PN-RATIOS-NBC-ALT] Calculating alternative NBC metrics...")
            
            # Gross Borrowing Rate
            gross_rate = pd.Series(0.0, index=ref_bs.columns)
            
            try:
                if 'Interest Expense' in ref_is.index and 'Total Debt' in ref_bs.index:
                    interest_expense = ref_is.loc['Interest Expense']
                    total_debt = ref_bs.loc['Total Debt']
                    avg_debt = total_debt.rolling(window=2, min_periods=1).mean()
                    
                    mask = (avg_debt > 10) & avg_debt.notna()  # Only where there's material debt
                    
                    if mask.any():
                        gross_rate[mask] = (interest_expense[mask] / avg_debt[mask]) * 100
                        gross_rate = gross_rate.clip(lower=0, upper=30)
                        self.logger.info(f"[PN-RATIOS-GROSS] Gross borrowing rate: {gross_rate[mask].to_dict()}")
                        
                    calculation_status['successful'].append('Gross Borrowing Rate')
                    
            except Exception as e:
                self.logger.error(f"[PN-RATIOS-GROSS-ERROR] Gross rate calculation failed: {e}")
                calculation_status['failed'].append(f'Gross Borrowing Rate: {str(e)}')
            
            ratios['Gross Borrowing Rate %'] = gross_rate
            
            # Effective interest rate on financial assets (for net cash companies)
            if 'Financial Assets' in ref_bs.index and 'Interest Income' in ref_is.index:
                try:
                    fin_assets = ref_bs.loc['Financial Assets']
                    interest_income = ref_is.loc['Interest Income']
                    avg_fin_assets = fin_assets.rolling(window=2, min_periods=1).mean()
                    
                    fin_asset_return = pd.Series(0.0, index=ref_bs.columns)
                    mask = (avg_fin_assets > 10) & avg_fin_assets.notna()
                    
                    if mask.any():
                        fin_asset_return[mask] = (interest_income[mask] / avg_fin_assets[mask]) * 100
                        fin_asset_return = fin_asset_return.clip(lower=0, upper=20)
                        ratios['Return on Financial Assets %'] = fin_asset_return
                        self.logger.info(f"[PN-RATIOS-NBC-ALT] Return on financial assets: {fin_asset_return[mask].to_dict()}")
                        
                except Exception as e:
                    self.logger.warning(f"[PN-RATIOS-NBC-ALT] Could not calculate return on financial assets: {e}")
            
            # --- ENHANCED Spread Calculation ---
            self.logger.info("\n[PN-RATIOS-SPREAD] Calculating Spread (Enhanced)...")
            
            spread = pd.Series(0.0, index=ref_bs.columns)
            
            try:
                if rnoa_calculated:
                    # Spread = RNOA - NBC
                    spread = ratios['Return on Net Operating Assets (RNOA) %'] - nbc
                    
                    # Log spread analysis
                    for year in spread.index:
                        if pd.notna(spread[year]) and pd.notna(ratios['Financial Leverage (FLEV)'][year]):
                            flev_val = ratios['Financial Leverage (FLEV)'][year]
                            
                            if flev_val > 0:  # Company has net debt
                                if spread[year] > 0:
                                    self.logger.info(f"[PN-RATIOS-SPREAD] {year}: Positive spread ({spread[year]:.2f}%), leverage is beneficial")
                                else:
                                    self.logger.warning(f"[PN-RATIOS-SPREAD] {year}: Negative spread ({spread[year]:.2f}%), leverage is detrimental")
                            else:  # Company has net cash
                                self.logger.info(f"[PN-RATIOS-SPREAD] {year}: Net cash position, spread = {spread[year]:.2f}%")
                    
                    self.logger.info(f"[PN-RATIOS-SPREAD] Spread values: {spread.to_dict()}")
                    calculation_status['successful'].append('Spread')
                else:
                    self.logger.warning("[PN-RATIOS-SPREAD] Cannot calculate spread without RNOA")
                    calculation_status['warnings'].append('Spread: RNOA not available')
                    
            except Exception as e:
                self.logger.error(f"[PN-RATIOS-SPREAD-ERROR] Spread calculation failed: {e}")
                calculation_status['failed'].append(f'Spread: {str(e)}')
            
            ratios['Spread %'] = spread
            ratios['Leverage Spread %'] = spread  # Duplicate for compatibility
            
            # --- ENHANCED Gross Profit Margin Calculation ---
            self.logger.info("\n[PN-RATIOS-MARGINS] Calculating Enhanced Profitability Margins...")
            
            if 'Revenue' in ref_is.index:
                revenue = ref_is.loc['Revenue']
                
                # Gross Profit Margin - Multiple attempts
                gpm_calculated = False
                gpm = pd.Series(np.nan, index=ref_bs.columns)
                
                # Try direct gross profit first
                if 'Gross Profit' in ref_is.index:
                    try:
                        gross_profit = ref_is.loc['Gross Profit']
                        mask = (revenue != 0) & revenue.notna()
                        gpm[mask] = (gross_profit[mask] / revenue[mask]) * 100
                        gpm_calculated = True
                        metadata['gpm_source'] = 'gross_profit'
                        self.logger.info("[PN-RATIOS-MARGINS] Calculated GPM from Gross Profit")
                    except Exception as e:
                        self.logger.warning(f"[PN-RATIOS-MARGINS] GPM from Gross Profit failed: {e}")
                
                # Try calculating from Revenue - COGS
                if not gpm_calculated and 'Cost of Goods Sold' in ref_is.index:
                    try:
                        cogs = ref_is.loc['Cost of Goods Sold']
                        gross_profit_calc = revenue - cogs
                        mask = (revenue != 0) & revenue.notna()
                        gpm[mask] = (gross_profit_calc[mask] / revenue[mask]) * 100
                        gpm_calculated = True
                        metadata['gpm_source'] = 'revenue_minus_cogs'
                        self.logger.info("[PN-RATIOS-MARGINS] Calculated GPM from Revenue - COGS")
                    except Exception as e:
                        self.logger.warning(f"[PN-RATIOS-MARGINS] GPM from Revenue-COGS failed: {e}")
                
                # Try using operating margin as proxy
                if not gpm_calculated and 'Operating Income Before Tax' in ref_is.index:
                    try:
                        oper_income = ref_is.loc['Operating Income Before Tax']
                        # Add back operating expenses if available
                        if 'Operating Expenses' in ref_is.index:
                            oper_expenses = ref_is.loc['Operating Expenses']
                            gross_proxy = oper_income + oper_expenses
                        else:
                            gross_proxy = oper_income * 1.5  # Rough approximation
                        
                        mask = (revenue != 0) & revenue.notna()
                        gpm[mask] = (gross_proxy[mask] / revenue[mask]) * 100
                        gpm_calculated = True
                        metadata['gpm_source'] = 'operating_income_proxy'
                        self.logger.warning("[PN-RATIOS-MARGINS] Using operating income proxy for GPM")
                    except Exception as e:
                        self.logger.warning(f"[PN-RATIOS-MARGINS] GPM proxy calculation failed: {e}")
                
                ratios['Gross Profit Margin %'] = gpm
                
                if gpm_calculated:
                    calculation_status['successful'].append('Gross Profit Margin')
                    self.logger.info(f"[PN-RATIOS-MARGINS] GPM values: {gpm.to_dict()}")
                else:
                    calculation_status['failed'].append('Gross Profit Margin: Could not calculate')
                
                # EBITDA Margin
                if 'EBITDA' in ref_is.index:
                    try:
                        ebitda = ref_is.loc['EBITDA']
                        ebitda_margin = pd.Series(index=ref_bs.columns, dtype=float)
                        mask = (revenue != 0) & revenue.notna()
                        ebitda_margin[mask] = (ebitda[mask] / revenue[mask]) * 100
                        ebitda_margin[~mask] = np.nan
                        ratios['EBITDA Margin %'] = ebitda_margin
                        calculation_status['successful'].append('EBITDA Margin')
                    except Exception as e:
                        self.logger.warning(f"[PN-RATIOS-MARGINS] EBITDA margin failed: {e}")
                        ratios['EBITDA Margin %'] = pd.Series(np.nan, index=ref_bs.columns)
                
                # Net Profit Margin
                if 'Net Income (Reported)' in ref_is.index:
                    try:
                        net_income = ref_is.loc['Net Income (Reported)']
                        npm = pd.Series(index=ref_bs.columns, dtype=float)
                        mask = (revenue != 0) & revenue.notna()
                        npm[mask] = (net_income[mask] / revenue[mask]) * 100
                        npm[~mask] = np.nan
                        ratios['Net Profit Margin %'] = npm
                        calculation_status['successful'].append('Net Profit Margin')
                    except Exception as e:
                        self.logger.warning(f"[PN-RATIOS-MARGINS] Net margin failed: {e}")
                        ratios['Net Profit Margin %'] = pd.Series(np.nan, index=ref_bs.columns)
            
            # --- ROE and its decomposition ---
            self.logger.info("\n[PN-RATIOS-ROE] Calculating ROE and decomposition...")
            
            roe_calculated = False
            if 'Net Income (Reported)' in ref_is.index and 'Common Equity' in ref_bs.index:
                try:
                    net_income = ref_is.loc['Net Income (Reported)']
                    ce = ref_bs.loc['Common Equity']
                    avg_ce = ce.rolling(window=2, min_periods=1).mean()
                    
                    roe = pd.Series(index=ref_bs.columns, dtype=float)
                    mask = (avg_ce != 0) & avg_ce.notna()
                    roe[mask] = (net_income[mask] / avg_ce[mask]) * 100
                    roe[~mask] = np.nan
                    
                    ratios['Return on Equity (ROE) %'] = roe
                    roe_calculated = True
                    calculation_status['successful'].append('ROE')
                    
                    # ROE Decomposition: ROE = RNOA + (FLEV × Spread)
                    if rnoa_calculated and flev_calculated:
                        rnoa_values = ratios['Return on Net Operating Assets (RNOA) %']
                        flev_values = ratios['Financial Leverage (FLEV)']
                        spread_values = ratios['Spread %']
                        
                        calculated_roe = rnoa_values + (flev_values * spread_values)
                        ratios['ROE (Calculated) %'] = calculated_roe
                        
                        # Check decomposition accuracy
                        roe_diff = (roe - calculated_roe).abs()
                        metadata['roe_decomposition_diff'] = roe_diff.max()
                        
                        if roe_diff.max() > 1.0:
                            calculation_status['warnings'].append(f'ROE decomposition mismatch: {roe_diff.max():.2f}%')
                        
                        self.logger.info(f"[PN-RATIOS-ROE-DECOMP] ROE decomposition check:")
                        self.logger.info(f"  Reported ROE: {roe.to_dict()}")
                        self.logger.info(f"  Calculated ROE: {calculated_roe.to_dict()}")
                        self.logger.info(f"  Max difference: {roe_diff.max():.2f}%")
                        
                        # Log decomposition components
                        for year in roe.index:
                            if pd.notna(roe[year]) and pd.notna(calculated_roe[year]):
                                self.logger.info(f"[PN-RATIOS-ROE-DECOMP] {year}:")
                                self.logger.info(f"  ROE = {roe[year]:.2f}%")
                                self.logger.info(f"  = RNOA({rnoa_values[year]:.2f}%) + FLEV({flev_values[year]:.2f}) × Spread({spread_values[year]:.2f}%)")
                                self.logger.info(f"  = {rnoa_values[year]:.2f}% + {flev_values[year] * spread_values[year]:.2f}%")
                        
                except Exception as e:
                    self.logger.error(f"[PN-RATIOS-ROE-ERROR] ROE calculation failed: {e}")
                    calculation_status['failed'].append(f'ROE: {str(e)}')
                    ratios['Return on Equity (ROE) %'] = pd.Series(np.nan, index=ref_bs.columns)
            else:
                self.logger.warning("[PN-RATIOS-ROE] Missing data for ROE calculation")
                ratios['Return on Equity (ROE) %'] = pd.Series(np.nan, index=ref_bs.columns)
                calculation_status['failed'].append('ROE: Missing required data')
            
            # ========== 2. ADDITIONAL PERFORMANCE METRICS ==========
            
            # --- Return on Assets (ROA) ---
            if 'Total Assets' in ref_bs.index and 'Net Income (Reported)' in ref_is.index:
                try:
                    total_assets = ref_bs.loc['Total Assets']
                    net_income = ref_is.loc['Net Income (Reported)']
                    avg_assets = total_assets.rolling(window=2, min_periods=1).mean()
                    
                    roa = pd.Series(index=ref_bs.columns, dtype=float)
                    mask = (avg_assets != 0) & avg_assets.notna()
                    roa[mask] = (net_income[mask] / avg_assets[mask]) * 100
                    roa[~mask] = np.nan
                    
                    ratios['Return on Assets (ROA) %'] = roa
                    calculation_status['successful'].append('ROA')
                    
                except Exception as e:
                    self.logger.error(f"[PN-RATIOS-ROA-ERROR] ROA calculation failed: {e}")
                    ratios['Return on Assets (ROA) %'] = pd.Series(np.nan, index=ref_bs.columns)
            
            # --- Growth Metrics ---
            self.logger.info("\n[PN-RATIOS-GROWTH] Calculating growth metrics...")
            
            # Revenue Growth
            if 'Revenue' in ref_is.index:
                try:
                    revenue = ref_is.loc['Revenue']
                    revenue_growth = revenue.pct_change() * 100
                    ratios['Revenue Growth %'] = revenue_growth
                    
                    # CAGR
                    first_valid = revenue.first_valid_index()
                    last_valid = revenue.last_valid_index()
                    if first_valid and last_valid and first_valid != last_valid:
                        years = revenue.index.get_loc(last_valid) - revenue.index.get_loc(first_valid)
                        if years > 0 and revenue[first_valid] > 0:
                            cagr = ((revenue[last_valid] / revenue[first_valid]) ** (1/years) - 1) * 100
                            metadata['revenue_cagr'] = cagr
                            self.logger.info(f"[PN-RATIOS-GROWTH] Revenue CAGR: {cagr:.2f}%")
                            
                except Exception as e:
                    self.logger.error(f"[PN-RATIOS-GROWTH-ERROR] Growth calculation failed: {e}")
            
            # NOA Growth
            if 'Net Operating Assets' in ref_bs.index:
                try:
                    noa = ref_bs.loc['Net Operating Assets']
                    noa_growth = noa.pct_change() * 100
                    ratios['NOA Growth %'] = noa_growth
                except:
                    pass
            
            # Net Income Growth
            if 'Net Income (Reported)' in ref_is.index:
                try:
                    net_income = ref_is.loc['Net Income (Reported)']
                    ni_growth = net_income.pct_change() * 100
                    ratios['Net Income Growth %'] = ni_growth
                except:
                    pass
            
            # --- Efficiency Ratios ---
            self.logger.info("\n[PN-RATIOS-EFFICIENCY] Calculating efficiency ratios...")
            
            # Asset Turnover
            if 'Revenue' in ref_is.index and 'Total Assets' in ref_bs.index:
                try:
                    revenue = ref_is.loc['Revenue']
                    total_assets = ref_bs.loc['Total Assets']
                    avg_assets = total_assets.rolling(window=2, min_periods=1).mean()
                    
                    asset_turnover = pd.Series(index=ref_bs.columns, dtype=float)
                    mask = (avg_assets != 0) & avg_assets.notna()
                    asset_turnover[mask] = revenue[mask] / avg_assets[mask]
                    asset_turnover[~mask] = np.nan
                    
                    ratios['Asset Turnover'] = asset_turnover
                    calculation_status['successful'].append('Asset Turnover')
                    
                except Exception as e:
                    self.logger.error(f"[PN-RATIOS-EFFICIENCY-ERROR] Efficiency calculation failed: {e}")
            
            # Working Capital Turnover
            if 'Revenue' in ref_is.index and all(x in ref_bs.index for x in ['Current Assets', 'Current Liabilities']):
                try:
                    revenue = ref_is.loc['Revenue']
                    current_assets = ref_bs.loc['Current Assets']
                    current_liabilities = ref_bs.loc['Current Liabilities']
                    
                    working_capital = current_assets - current_liabilities
                    avg_wc = working_capital.rolling(window=2, min_periods=1).mean()
                    
                    wc_turnover = pd.Series(index=ref_bs.columns, dtype=float)
                    mask = (avg_wc != 0) & avg_wc.notna() & (avg_wc > 0)
                    wc_turnover[mask] = revenue[mask] / avg_wc[mask]
                    wc_turnover[~mask] = np.nan
                    
                    ratios['Working Capital Turnover'] = wc_turnover
                    
                except Exception as e:
                    self.logger.error(f"[PN-RATIOS-WC-ERROR] Working capital turnover failed: {e}")
            
            # --- Liquidity Ratios ---
            self.logger.info("\n[PN-RATIOS-LIQUIDITY] Calculating liquidity ratios...")
            
            if 'Current Assets' in ref_bs.index and 'Current Liabilities' in ref_bs.index:
                try:
                    current_assets = ref_bs.loc['Current Assets']
                    current_liabilities = ref_bs.loc['Current Liabilities']
                    
                    # Current Ratio
                    current_ratio = pd.Series(index=ref_bs.columns, dtype=float)
                    mask = (current_liabilities != 0) & current_liabilities.notna()
                    current_ratio[mask] = current_assets[mask] / current_liabilities[mask]
                    current_ratio[~mask] = np.nan
                    
                    ratios['Current Ratio'] = current_ratio
                    
                    # Quick Ratio (if inventory available)
                    inventory_source = self._find_source_metric('Inventory')
                    if inventory_source and inventory_source in self._df_clean.index:
                        inventory = self._df_clean.loc[inventory_source]
                        quick_assets = current_assets - inventory
                        
                        quick_ratio = pd.Series(index=ref_bs.columns, dtype=float)
                        mask = (current_liabilities != 0) & current_liabilities.notna()
                        quick_ratio[mask] = quick_assets[mask] / current_liabilities[mask]
                        quick_ratio[~mask] = np.nan
                        
                        ratios['Quick Ratio'] = quick_ratio
                        
                    # Cash Ratio
                    if 'Cash and Equivalents' in ref_bs.index:
                        cash = ref_bs.loc['Cash and Equivalents']
                        
                        cash_ratio = pd.Series(index=ref_bs.columns, dtype=float)
                        mask = (current_liabilities != 0) & current_liabilities.notna()
                        cash_ratio[mask] = cash[mask] / current_liabilities[mask]
                        cash_ratio[~mask] = np.nan
                        
                        ratios['Cash Ratio'] = cash_ratio
                        
                except Exception as e:
                    self.logger.error(f"[PN-RATIOS-LIQUIDITY-ERROR] Liquidity calculation failed: {e}")
            
            # --- Coverage Ratios ---
            self.logger.info("\n[PN-RATIOS-COVERAGE] Calculating coverage ratios...")
            
            # Interest Coverage Ratio
            if 'Operating Income Before Tax' in ref_is.index and 'Interest Expense' in ref_is.index:
                try:
                    ebit = ref_is.loc['Operating Income Before Tax']
                    interest_expense = ref_is.loc['Interest Expense']
                    
                    interest_coverage = pd.Series(index=ref_bs.columns, dtype=float)
                    mask = (interest_expense != 0) & interest_expense.notna() & (interest_expense > 0)
                    interest_coverage[mask] = ebit[mask] / interest_expense[mask]
                    interest_coverage[~mask] = np.nan
                    
                    # Cap at reasonable level for companies with minimal debt
                    interest_coverage = interest_coverage.clip(upper=100)
                    
                    ratios['Interest Coverage'] = interest_coverage
                    
                    # Log coverage status
                    for year in interest_coverage.index:
                        if pd.notna(interest_coverage[year]):
                            if interest_coverage[year] < 1.5:
                                self.logger.warning(f"[PN-RATIOS-COVERAGE] {year}: Low interest coverage ({interest_coverage[year]:.1f}x)")
                            elif interest_coverage[year] > 10:
                                self.logger.info(f"[PN-RATIOS-COVERAGE] {year}: Strong interest coverage ({interest_coverage[year]:.1f}x)")
                    
                except Exception as e:
                    self.logger.error(f"[PN-RATIOS-COVERAGE-ERROR] Coverage ratio failed: {e}")
            
            # ========== 3. QUALITY CHECKS AND VALIDATION ==========
            
            self.logger.info("\n[PN-RATIOS-VALIDATION] Performing quality checks...")
            
            # Check for ratios with all NaN values
            nan_ratios = []
            for ratio_name in ratios.columns:
                if ratios[ratio_name].isna().all():
                    nan_ratios.append(ratio_name)
            
            if nan_ratios:
                self.logger.warning(f"[PN-RATIOS-VALIDATION] Ratios with all NaN values: {nan_ratios}")
                calculation_status['warnings'].append(f"All-NaN ratios: {len(nan_ratios)}")
            
            # Check for extreme values
            for ratio_name in ratios.columns:
                series = ratios[ratio_name].dropna()
                if len(series) > 0:
                    if series.abs().max() > 1000:
                        self.logger.warning(f"[PN-RATIOS-VALIDATION] Extreme values in {ratio_name}: max={series.max():.2f}")
                        calculation_status['warnings'].append(f"Extreme values in {ratio_name}")
            
            # ========== 4. TRANSPOSE BEFORE ENSURING REQUIRED RATIOS ==========
            
            # Transpose the DataFrame NOW before the ensure section
            ratios = ratios.T  # Now ratios has ratio names as index, years as columns
            
            # ========== 5. ENSURE ALL REQUIRED RATIOS EXIST ==========
            
            self.logger.info("\n[PN-RATIOS-ENSURE] Ensuring all required ratios exist...")
            
            required_ratios = [
                # Core Penman-Nissim Ratios
                'Return on Net Operating Assets (RNOA) %',
                'Operating Profit Margin (OPM) %',
                'Net Operating Asset Turnover (NOAT)',
                'Financial Leverage (FLEV)',
                'Financial Leverage (End of Period)',
                'Net Borrowing Cost (NBC) %',
                'Spread %',
                'Leverage Spread %',
                'Return on Equity (ROE) %',
                'ROE (Calculated) %',
                
                # Additional Performance Ratios
                'Return on Assets (ROA) %',
                'Gross Borrowing Rate %',
                'Return on Financial Assets %',
                'Debt to Equity',
                'Net Debt to Equity',
                
                # Growth Ratios
                'Revenue Growth %',
                'NOA Growth %',
                'Net Income Growth %',
                
                # Efficiency Ratios
                'Asset Turnover',
                'Working Capital Turnover',
                
                # Liquidity Ratios
                'Current Ratio',
                'Quick Ratio',
                'Cash Ratio',
                
                # Coverage Ratios
                'Interest Coverage',
                
                # Profitability Margins
                'Gross Profit Margin %',
                'EBITDA Margin %',
                'Net Profit Margin %'
            ]
            
            for ratio_name in required_ratios:
                if ratio_name not in ratios.index:
                    self.logger.info(f"[PN-RATIOS-ENSURE] Adding missing ratio: {ratio_name}")
                    ratios.loc[ratio_name] = pd.Series(np.nan, index=ratios.columns)
                    calculation_status['warnings'].append(f"Added missing ratio: {ratio_name}")
            
            # ========== 6. CALCULATE RATIO TRENDS ==========
            
            self.logger.info("\n[PN-RATIOS-TRENDS] Analyzing ratio trends...")
            
            ratio_trends = self._analyze_ratio_trends(ratios)
            metadata['ratio_trends'] = ratio_trends
            
            # ========== 7. CALCULATE RATIO QUALITY SCORES ==========
            
            self.logger.info("\n[PN-RATIOS-QUALITY] Calculating ratio quality scores...")
            
            quality_scores = self._calculate_ratio_quality_scores(ratios)
            metadata['ratio_quality_scores'] = quality_scores
            
            # ========== 8. SUMMARY AND METADATA ==========
            
            # Store calculation status
            metadata['calculation_status'] = calculation_status
            metadata['total_ratios'] = len(ratios.index)
            metadata['successful_calculations'] = len(calculation_status['successful'])
            metadata['failed_calculations'] = len(calculation_status['failed'])
            metadata['warnings'] = len(calculation_status['warnings'])
            
            # Summary logging
            self.logger.info("\n[PN-RATIOS-SUMMARY] Ratio Calculation Summary:")
            self.logger.info(f"  Total ratios calculated: {len(ratios.index)}")
            self.logger.info(f"  Successful calculations: {len(calculation_status['successful'])}")
            self.logger.info(f"  Failed calculations: {len(calculation_status['failed'])}")
            self.logger.info(f"  Warnings: {len(calculation_status['warnings'])}")
            
            # Log key ratios with their latest values
            key_ratios = ['Return on Net Operating Assets (RNOA) %', 'Financial Leverage (FLEV)', 
                          'Net Borrowing Cost (NBC) %', 'Spread %', 'Return on Equity (ROE) %',
                          'Gross Profit Margin %']
            
            self.logger.info("\n[PN-RATIOS-KEY] Key Ratios (Latest Values):")
            for ratio_name in key_ratios:
                if ratio_name in ratios.index:
                    latest_value = ratios.loc[ratio_name].iloc[-1] if not ratios.loc[ratio_name].empty else np.nan
                    if pd.notna(latest_value):
                        self.logger.info(f"  {ratio_name}: {latest_value:.2f}")
                    else:
                        self.logger.info(f"  {ratio_name}: N/A")
            
            # Log any failures
            if calculation_status['failed']:
                self.logger.error("\n[PN-RATIOS-FAILURES] Failed Calculations:")
                for failure in calculation_status['failed']:
                    self.logger.error(f"  - {failure}")
            
            # Log any warnings
            if calculation_status['warnings']:
                self.logger.warning("\n[PN-RATIOS-WARNINGS] Calculation Warnings:")
                for warning in calculation_status['warnings']:
                    self.logger.warning(f"  - {warning}")
            
        except Exception as e:
            self.logger.error(f"[PN-RATIOS-ERROR] Critical error in ratio calculation: {e}", exc_info=True)
            # Ensure we return at least an empty DataFrame with required ratios
            ratios = pd.DataFrame(index=required_ratios, columns=ref_bs.columns)
            metadata['critical_error'] = str(e)
        
        finally:
            # Store metadata
            self.calculation_metadata['ratios'] = metadata
            
            self.logger.info("\n[PN-RATIOS-END] Ratio Calculations Complete")
            self.logger.info("="*80 + "\n")
        
        # Return DataFrame with years as columns, ratios as rows (already transposed)
        return ratios
    
    def _analyze_ratio_trends(self, ratios: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in key ratios"""
        trends = {}
        
        key_ratios = [
            'Return on Net Operating Assets (RNOA) %',
            'Spread %',
            'Return on Equity (ROE) %',
            'Revenue Growth %',
            'Net Profit Margin %'
        ]
        
        for ratio in key_ratios:
            if ratio in ratios.index:
                series = ratios.loc[ratio].dropna()
                if len(series) >= 3:
                    # Calculate trend
                    x = np.arange(len(series))
                    y = series.values
                    
                    # Linear regression
                    z = np.polyfit(x, y, 1)
                    slope = z[0]
                    
                    # Calculate R-squared
                    p = np.poly1d(z)
                    yhat = p(x)
                    ybar = np.mean(y)
                    ssreg = np.sum((yhat - ybar)**2)
                    sstot = np.sum((y - ybar)**2)
                    r_squared = ssreg / sstot if sstot > 0 else 0
                    
                    # Categorize trend
                    if abs(slope) < 0.5:
                        trend = 'stable'
                    elif slope > 0.5:
                        trend = 'improving'
                    else:
                        trend = 'declining'
                    
                    # Calculate volatility
                    volatility = series.std()
                    
                    trends[ratio] = {
                        'direction': trend,
                        'slope': slope,
                        'r_squared': r_squared,
                        'average': series.mean(),
                        'volatility': volatility,
                        'latest': series.iloc[-1],
                        'change_from_start': series.iloc[-1] - series.iloc[0]
                    }
        
        return trends
    
    def _calculate_ratio_quality_scores(self, ratios: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality scores for each ratio based on data availability and consistency"""
        quality_scores = {}
        
        for ratio_name in ratios.index:
            series = ratios.loc[ratio_name]
            
            # Factors for quality score
            # 1. Completeness (50% weight)
            completeness = series.notna().sum() / len(series) if len(series) > 0 else 0
            
            # 2. Consistency - no extreme jumps (30% weight)
            consistency = 1.0
            if series.notna().sum() > 1:
                pct_changes = series.pct_change().abs()
                # Penalize changes > 100%
                extreme_changes = (pct_changes > 1).sum()
                consistency = max(0, 1 - (extreme_changes / len(pct_changes)))
            
            # 3. Reasonable values (20% weight)
            reasonable = 1.0
            if series.notna().any():
                # Check if values are within reasonable bounds
                if 'Ratio' in ratio_name or '%' in ratio_name:
                    # For percentage ratios, check if between -100% and 200%
                    if series.min() < -100 or series.max() > 200:
                        reasonable = 0.5
                else:
                    # For other ratios, check for extreme values
                    if series.abs().max() > 1000:
                        reasonable = 0.5
            
            # Combined quality score
            quality_score = (completeness * 0.5 + consistency * 0.3 + reasonable * 0.2)
            quality_scores[ratio_name] = round(quality_score, 3)
        
        return quality_scores

    def get_ratio_insights(self) -> List[str]:
        """Generate insights based on calculated ratios"""
        insights = []
        
        if not hasattr(self, 'calculation_metadata') or 'ratios' not in self.calculation_metadata:
            return ["No ratio calculations available for insights"]
        
        metadata = self.calculation_metadata['ratios']
        
        # Check ratio trends
        if 'ratio_trends' in metadata:
            trends = metadata['ratio_trends']
            
            # RNOA trend
            if 'Return on Net Operating Assets (RNOA) %' in trends:
                rnoa_trend = trends['Return on Net Operating Assets (RNOA) %']
                if rnoa_trend['direction'] == 'improving':
                    insights.append(f"✅ RNOA is improving with a positive trend (slope: {rnoa_trend['slope']:.2f})")
                elif rnoa_trend['direction'] == 'declining':
                    insights.append(f"⚠️ RNOA is declining (slope: {rnoa_trend['slope']:.2f}) - investigate operational efficiency")
            
            # Spread trend
            if 'Spread %' in trends:
                spread_trend = trends['Spread %']
                if spread_trend['latest'] < 0:
                    insights.append(f"❌ Negative spread ({spread_trend['latest']:.1f}%) - leverage is destroying value")
                elif spread_trend['direction'] == 'declining':
                    insights.append(f"⚠️ Spread is declining - monitor borrowing costs and operational returns")
        
        # Check calculation quality
        if 'calculation_status' in metadata:
            status = metadata['calculation_status']
            if len(status['failed']) > 0:
                insights.append(f"⚠️ {len(status['failed'])} ratio calculations failed - check data quality")
            if len(status['warnings']) > 5:
                insights.append(f"💡 {len(status['warnings'])} warnings detected - review data for anomalies")
        
        return insights
    
    def export_ratio_analysis(self) -> pd.DataFrame:
        """Export comprehensive ratio analysis with all calculations"""
        if not hasattr(self, '_cached_ratios') or self._cached_ratios is None:
            # Calculate if not already done
            self._cached_ratios = self._calculate_ratios_enhanced(None)
        
        return self._cached_ratios
    
    def get_ratio_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for all calculated ratios"""
        if not hasattr(self, '_cached_ratios') or self._cached_ratios is None:
            return pd.DataFrame()
        
        ratios = self._cached_ratios.T  # Transpose to have ratios as rows
        
        summary = pd.DataFrame({
            'Mean': ratios.mean(axis=1),
            'Std Dev': ratios.std(axis=1),
            'Min': ratios.min(axis=1),
            'Max': ratios.max(axis=1),
            'Latest': ratios.iloc[:, -1] if ratios.shape[1] > 0 else np.nan,
            'Trend': ratios.apply(lambda x: 'Increasing' if x.iloc[-1] > x.iloc[0] else 'Decreasing' if x.iloc[-1] < x.iloc[0] else 'Stable', axis=1)
        })
        
        return summary
                                              
    def _calculate_free_cash_flow_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced free cash flow calculation"""
        # Always use clean data
        df = self._df_clean
        
        self.logger.info("\n" + "="*80)
        self.logger.info("[PN-FCF-START] Starting Free Cash Flow Calculation (V5)")
        self.logger.info("="*80)
        
        fcf = pd.DataFrame(index=df.columns)
        metadata = {}
        
        try:
            # Operating Cash Flow
            ocf = self._get_safe_series(df, 'Operating Cash Flow')
            fcf['Operating Cash Flow'] = ocf
            
            # Capital Expenditure (ensure positive for subtraction)
            capex = self._get_safe_series(df, 'Capital Expenditure', default_zero=True)
            capex = capex.abs()  # Ensure positive
            fcf['Capital Expenditure'] = capex
            
            # Free Cash Flow to Firm
            fcff = ocf - capex
            fcf['Free Cash Flow to Firm'] = fcff
            
            # Alternative FCF calculation for validation
            try:
                net_income = self._get_safe_series(df, 'Net Income')
                depreciation = self._get_safe_series(df, 'Depreciation', default_zero=True)
                
                # Working capital change
                if self._find_source_metric('Current Assets') and self._find_source_metric('Current Liabilities'):
                    current_assets = self._get_safe_series(df, 'Current Assets')
                    current_liabilities = self._get_safe_series(df, 'Current Liabilities')
                    
                    working_capital = current_assets - current_liabilities
                    change_in_wc = working_capital.diff()
                    
                    fcf['Working Capital'] = working_capital
                    fcf['Change in Working Capital'] = change_in_wc
                    
                    # Alternative FCF
                    alt_fcff = net_income + depreciation - change_in_wc.fillna(0) - capex
                    fcf['FCF (from Net Income)'] = alt_fcff
                    
                    # Log comparison
                    fcf_diff = (fcff - alt_fcff).abs().max()
                    metadata['fcf_calculation_diff'] = fcf_diff
                    
            except Exception as e:
                self.logger.warning(f"[PN-FCF-ALT] Alternative FCF calculation failed: {e}")
            
            # FCF to Equity
            try:
                ref_bs = self._reformulate_balance_sheet_enhanced(None)
                if 'Financial Liabilities' in ref_bs.index:
                    debt = ref_bs.loc['Financial Liabilities']
                    debt_change = debt.diff().fillna(0)
                    
                    fcfe = fcff + debt_change
                    fcf['Net Borrowing'] = debt_change
                    fcf['Free Cash Flow to Equity'] = fcfe
                    
            except Exception as e:
                self.logger.warning(f"[PN-FCF-FCFE] FCFE calculation failed: {e}")
            
            # FCF Yield
            if 'Total Assets' in ref_bs.index:
                total_assets = ref_bs.loc['Total Assets']
                fcf_yield = (fcff / total_assets.replace(0, np.nan)) * 100
                fcf['FCF Yield %'] = fcf_yield
            
            # Summary
            self.logger.info("\n[PN-FCF-SUMMARY] Free Cash Flow Summary:")
            self.logger.info(f"  Operating Cash Flow: {ocf.sum():,.0f}")
            self.logger.info(f"  Capital Expenditure: {capex.sum():,.0f}")
            self.logger.info(f"  Free Cash Flow: {fcff.sum():,.0f}")
            
        except Exception as e:
            self.logger.error(f"[PN-FCF-ERROR] FCF calculation failed: {e}", exc_info=True)
            raise
        
        self.calculation_metadata['free_cash_flow'] = metadata
        
        self.logger.info("\n[PN-FCF-END] Free Cash Flow Calculation Complete")
        self.logger.info("="*80 + "\n")
        
        return fcf.T

    def _calculate_value_drivers_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced value drivers calculation"""
        # Always use clean data
        df = self._df_clean
        
        self.logger.info("\n" + "="*80)
        self.logger.info("[PN-DRIVERS-START] Starting Value Drivers Calculation (V5)")
        self.logger.info("="*80)
        
        drivers = pd.DataFrame(index=df.columns)
        metadata = {}
        
        try:
            # Revenue drivers
            revenue = self._get_safe_series(df, 'Revenue')
            drivers['Revenue'] = revenue
            drivers['Revenue Growth %'] = revenue.pct_change() * 100
            
            # CAGR calculation with error handling
            if len(revenue.dropna()) > 1:
                valid_revenue = revenue.dropna()
                years = len(valid_revenue) - 1
                if years > 0 and valid_revenue.iloc[0] != 0:  # Avoid div by zero
                    cagr = ((valid_revenue.iloc[-1] / valid_revenue.iloc[0]) ** (1/years) - 1) * 100
                    metadata['revenue_cagr'] = cagr
                else:
                    metadata['revenue_cagr'] = 'N/A'
                    if years <= 0:
                        self.logger.warning("[PN-DRIVERS] Could not calculate Revenue CAGR: insufficient periods")
                    elif valid_revenue.iloc[0] == 0:
                        self.logger.warning("[PN-DRIVERS] Could not calculate Revenue CAGR: division by zero (first value is 0)")
                    else:
                        self.logger.warning("[PN-DRIVERS] Could not calculate Revenue CAGR: unknown issue")
            else:
                metadata['revenue_cagr'] = 'N/A'
                self.logger.warning("[PN-DRIVERS] Insufficient data for Revenue CAGR")
            
            # Get reformulated statements
            ref_is = self._reformulate_income_statement_enhanced(None)
            ref_bs = self._reformulate_balance_sheet_enhanced(None)
            
            # Profitability drivers
            if 'Operating Income After Tax' in ref_is.index:
                nopat = ref_is.loc['Operating Income After Tax']
                drivers['NOPAT'] = nopat
                drivers['NOPAT Growth %'] = nopat.pct_change() * 100
                
                # NOPAT Margin
                nopat_margin = (nopat / revenue.replace(0, np.nan)) * 100
                drivers['NOPAT Margin %'] = nopat_margin
                drivers['NOPAT Margin Change %'] = nopat_margin.diff()
            
            # Investment drivers
            if 'Net Operating Assets' in ref_bs.index:
                noa = ref_bs.loc['Net Operating Assets']
                drivers['Net Operating Assets'] = noa
                drivers['NOA Growth %'] = noa.pct_change() * 100
                
                # Investment Rate
                if 'NOPAT' in drivers.columns:
                    noa_change = noa.diff()
                    investment_rate = (noa_change / drivers['NOPAT'].replace(0, np.nan)) * 100
                    drivers['Investment Rate %'] = investment_rate
            
            # Working Capital drivers
            if self._find_source_metric('Current Assets') and self._find_source_metric('Current Liabilities'):
                current_assets = self._get_safe_series(df, 'Current Assets')
                current_liabilities = self._get_safe_series(df, 'Current Liabilities')
                
                working_capital = current_assets - current_liabilities
                drivers['Working Capital'] = working_capital
                
                if 'Revenue' in drivers.columns:
                    wc_to_revenue = (working_capital / revenue.replace(0, np.nan)) * 100
                    drivers['Working Capital % of Revenue'] = wc_to_revenue
            
            # Asset efficiency
            if 'Total Assets' in ref_bs.index:
                total_assets = ref_bs.loc['Total Assets']
                asset_turnover = revenue / total_assets.replace(0, np.nan)
                drivers['Asset Turnover'] = asset_turnover
            
            # Cash conversion
            fcf_df = self._calculate_free_cash_flow_enhanced(None)
            if 'Free Cash Flow to Firm' in fcf_df.index and 'NOPAT' in drivers.columns:
                fcf = fcf_df.loc['Free Cash Flow to Firm']
                cash_conversion = (fcf / drivers['NOPAT'].replace(0, np.nan)) * 100
                drivers['Cash Conversion %'] = cash_conversion
            
            # Summary with SAFE formatting
            self.logger.info("\n[PN-DRIVERS-SUMMARY] Value Drivers Summary:")
            
            cagr_value = metadata.get('revenue_cagr', 'N/A')
            self.logger.info(f"  Revenue CAGR: {self.safe_format(cagr_value, '.1f', 'N/A')}%")  # FIXED: Pass '.1f' (no leading ':')
            
            # Safe formatting for other metrics (example)
            if 'NOPAT Margin %' in drivers.columns:
                latest_nopat = drivers['NOPAT Margin %'].iloc[-1]
                self.logger.info(f"  Latest NOPAT Margin: {self.safe_format(latest_nopat, '.1f', 'N/A')}%")
            
            if 'Asset Turnover' in drivers.columns:
                latest_at = drivers['Asset Turnover'].iloc[-1]
                self.logger.info(f"  Latest Asset Turnover: {self.safe_format(latest_at, '.2f', 'N/A')}")
            
        except Exception as e:
            self.logger.error(f"[PN-DRIVERS-ERROR] Value drivers calculation failed: {e}", exc_info=True)
            raise
        
        self.calculation_metadata['value_drivers'] = metadata
        
        self.logger.info("\n[PN-DRIVERS-END] Value Drivers Calculation Complete")
        self.logger.info("="*80 + "\n")
        
        return drivers.T
    
    def calculate_all(self):
        """Calculate all Penman-Nissim metrics with optimized data flow"""
        try:
            results = {
                'reformulated_balance_sheet': self._reformulate_balance_sheet_enhanced(None),
                'reformulated_income_statement': self._reformulate_income_statement_enhanced(None),
                'ratios': self._calculate_ratios_enhanced(None),
                'free_cash_flow': self._calculate_free_cash_flow_enhanced(None),
                'value_drivers': self._calculate_value_drivers_enhanced(None),
                'validation_results': self.validation_results,
                'calculation_metadata': self.calculation_metadata
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Penman-Nissim calculations: {e}", exc_info=True)
            return {
                'error': str(e),
                'validation_results': self.validation_results,
                'calculation_metadata': self.calculation_metadata
            }

    # Helper methods
    # PASTE THIS CODE: Replace the _get_safe_series method in the EnhancedPenmanNissimAnalyzer class

    def _get_safe_series(self, df: pd.DataFrame, target_metric: str, default_zero: bool = False) -> pd.Series:
        """Enhanced safe series retrieval with STATEMENT TYPE VALIDATION and comprehensive detection"""
        self.logger.info(f"\n[PN-FETCH-START] Looking for: '{target_metric}'")
        
        # CRITICAL NEW SECTION: Define which metrics MUST come from which statements
        STATEMENT_REQUIREMENTS = {
            # P&L items MUST come from ProfitLoss::
            'Revenue': 'ProfitLoss',
            'Cost of Goods Sold': 'ProfitLoss',
            'Operating Income': 'ProfitLoss',
            'Operating Expenses': 'ProfitLoss',
            'EBIT': 'ProfitLoss',
            'Interest Expense': 'ProfitLoss',
            'Interest Income': 'ProfitLoss',
            'Tax Expense': 'ProfitLoss',
            'Income Before Tax': 'ProfitLoss',
            'Net Income': 'ProfitLoss',
            'Depreciation': 'ProfitLoss',
            'Other Income': 'ProfitLoss',
            'Gross Profit': 'ProfitLoss',
            
            # Balance Sheet items MUST come from BalanceSheet::
            'Total Assets': 'BalanceSheet',
            'Total Liabilities': 'BalanceSheet',
            'Total Equity': 'BalanceSheet',
            'Current Assets': 'BalanceSheet',
            'Current Liabilities': 'BalanceSheet',
            'Cash and Cash Equivalents': 'BalanceSheet',
            'Inventory': 'BalanceSheet',
            'Trade Receivables': 'BalanceSheet',
            'Property Plant Equipment': 'BalanceSheet',
            'Short-term Debt': 'BalanceSheet',
            'Long-term Debt': 'BalanceSheet',
            'Share Capital': 'BalanceSheet',
            'Retained Earnings': 'BalanceSheet',
            'Investments': 'BalanceSheet',
            'Short-term Investments': 'BalanceSheet',
            'Accounts Payable': 'BalanceSheet',
            'Accrued Expenses': 'BalanceSheet',
            'Deferred Revenue': 'BalanceSheet',
            
            # Cash Flow items MUST come from CashFlow::
            'Operating Cash Flow': 'CashFlow',
            'Capital Expenditure': 'CashFlow',
            'Investing Cash Flow': 'CashFlow',
            'Financing Cash Flow': 'CashFlow',
        }
        
        # Get the required statement type for this metric
        required_statement = STATEMENT_REQUIREMENTS.get(target_metric)
        
        # First, try the mapped source
        source_metric = self._find_source_metric(target_metric)
        
        if source_metric:
            self.logger.info(f"[PN-FETCH] Found mapping: '{target_metric}' -> '{source_metric}'")
            
            # CRITICAL: Validate the statement type
            if required_statement:
                statement_prefix = f"{required_statement}::"
                if not source_metric.startswith(statement_prefix):
                    self.logger.error(
                        f"[PN-FETCH-ERROR] WRONG STATEMENT TYPE! "
                        f"'{target_metric}' requires {required_statement} item, "
                        f"but mapped to '{source_metric}'"
                    )
                    
                    # Try to find the correct item
                    self.logger.info(f"[PN-FETCH] Searching for correct {required_statement} item...")
                    
                    # Look for items with the right prefix
                    correct_items = []
                    for idx in df.index:
                        if idx.startswith(statement_prefix):
                            idx_clean = idx.split('::')[-1].lower()
                            target_clean = target_metric.lower()
                            
                            # Check for matching keywords
                            if any(keyword in idx_clean for keyword in target_clean.split()):
                                correct_items.append(idx)
                    
                    if correct_items:
                        self.logger.warning(
                            f"[PN-FETCH] Found {len(correct_items)} correct items. "
                            f"Please fix your mapping! Suggestions:"
                        )
                        for item in correct_items[:5]:
                            self.logger.warning(f"  - {item}")
                        
                        # Don't use the wrong item - set source_metric to None to trigger pattern search
                        source_metric = None
                        self.logger.warning(f"[PN-FETCH] Ignoring wrong mapping, will search patterns instead")
                    else:
                        self.logger.error(f"[PN-FETCH] No {required_statement} items found for '{target_metric}'")
                        
                        # Show what's available
                        available = [idx for idx in df.index if idx.startswith(statement_prefix)]
                        if available:
                            self.logger.info(f"[PN-FETCH] Available {required_statement} items:")
                            for item in available[:10]:
                                self.logger.info(f"  - {item}")
                        
                        # Set source_metric to None to trigger pattern search
                        source_metric = None
            
            # If validation passed and we still have a valid source_metric, get the series
            if source_metric and source_metric in df.index:
                series = df.loc[source_metric].fillna(0 if default_zero else np.nan)
                self.logger.info(f"[PN-FETCH-SUCCESS] Retrieved '{target_metric}' from '{source_metric}'")
                self.logger.debug(f"[PN-FETCH] Values: {series.to_dict()}")
                self._log_metric_fetch(target_metric, source_metric, series, "Direct mapping (validated)")
                return series
        
        # Enhanced search patterns with ALL possible variations
        search_patterns = {
            'Capital Expenditure': [
                # Direct variations
                'CashFlow::Capital Expenditure',
                'CashFlow::Capital Expenditures', 
                'CashFlow::CAPEX',
                'CashFlow::Capex',
                'CashFlow::CapEx',
                
                # Purchase variations (most common)
                'CashFlow::Purchase of Fixed Assets',
                'CashFlow::Purchased of Fixed Assets',  # Common typo
                'CashFlow::Purchase of Property Plant and Equipment',
                'CashFlow::Purchase of Property, Plant and Equipment',
                'CashFlow::Purchases of Fixed Assets',
                'CashFlow::Purchase of PPE',
                'CashFlow::Purchase of Plant and Equipment',
                'CashFlow::Purchase of Tangible Assets',
                
                # Investment variations
                'CashFlow::Purchase of Investments',
                'CashFlow::Investment in Fixed Assets',
                'CashFlow::Investments in Fixed Assets',
                'CashFlow::Investment in Property Plant and Equipment',
                'CashFlow::Investment in PPE',
                
                # Addition variations
                'CashFlow::Additions to Fixed Assets',
                'CashFlow::Addition to Fixed Assets',
                'CashFlow::Additions to Property Plant and Equipment',
                'CashFlow::Additions to PPE',
                'CashFlow::Net Additions to Fixed Assets',
                
                # Acquisition variations
                'CashFlow::Acquisition of Fixed Assets',
                'CashFlow::Acquisition of Property Plant and Equipment',
                'CashFlow::Asset Acquisitions',
                'CashFlow::Fixed Asset Acquisitions',
                
                # Payment variations
                'CashFlow::Payments for Fixed Assets',
                'CashFlow::Payment for Purchase of Fixed Assets',
                'CashFlow::Payments for Property Plant and Equipment',
                'CashFlow::Cash Payments for Fixed Assets',
                
                # Expenditure variations
                'CashFlow::Expenditure on Fixed Assets',
                'CashFlow::Capital Expenditure on Fixed Assets',
                'CashFlow::Fixed Asset Expenditure',
                'CashFlow::Plant and Equipment Expenditure',
                
                # Other common variations
                'CashFlow::Fixed Assets Purchased',
                'CashFlow::PPE Purchased',
                'CashFlow::Tangible Assets Purchased',
                'CashFlow::Equipment Purchases',
                'CashFlow::Machinery and Equipment',
                'CashFlow::Plant and Machinery',
                
                # Indian specific variations
                'CashFlow::Purchase of Plant and Machinery',
                'CashFlow::Addition to Plant and Machinery',
                'CashFlow::Investment in Plant and Machinery',
                
                # Without CashFlow prefix (in case data doesn't have prefixes)
                'Capital Expenditure',
                'Capital Expenditures',
                'CAPEX',
                'Purchase of Fixed Assets',
                'Purchased of Fixed Assets',
                'Purchase of Property Plant and Equipment',
                'Additions to Fixed Assets',
                'Investment in Fixed Assets',
            ],
            
            'Depreciation': [
                'ProfitLoss::Depreciation and Amortisation Expenses',
                'ProfitLoss::Depreciation and Amortization Expenses',
                'ProfitLoss::Depreciation & Amortisation Expenses',
                'ProfitLoss::Depreciation & Amortization Expenses',
                'ProfitLoss::Depreciation and Amortisation',
                'ProfitLoss::Depreciation and Amortization',
                'ProfitLoss::Depreciation & Amortisation',
                'ProfitLoss::Depreciation & Amortization',
                'ProfitLoss::Depreciation',
                'ProfitLoss::Amortisation',
                'ProfitLoss::Amortization',
                'ProfitLoss::Depreciation Expense',
                'ProfitLoss::Depreciation Expenses',
                'ProfitLoss::Depreciation and Impairment',
                'Depreciation and Amortisation Expenses',
                'Depreciation and Amortization Expenses',
                'Depreciation',
            ],
            
            'Operating Cash Flow': [
                'CashFlow::Net Cash from Operating Activities',
                'CashFlow::Net CashFlow From Operating Activities',
                'CashFlow::Net Cash Flow From Operating Activities',
                'CashFlow::Operating Cash Flow',
                'CashFlow::Cash from Operating Activities',
                'CashFlow::Cash Flow from Operating Activities',
                'CashFlow::Net Cash Generated from Operating Activities',
                'CashFlow::Cash Generated from Operating Activities',
                'CashFlow::Operating Activities',
                'CashFlow::Cash from Operations',
                'Net Cash from Operating Activities',
                'Operating Cash Flow',
                'Cash from Operating Activities',
            ],
            
            'Revenue': [
                'ProfitLoss::Revenue From Operations',
                'ProfitLoss::Revenue From Operations(Net)',
                'ProfitLoss::Revenue from Operations (Net)',
                'ProfitLoss::Total Revenue',
                'ProfitLoss::Net Sales',
                'ProfitLoss::Sales',
                'ProfitLoss::Revenue',
                'ProfitLoss::Gross Revenue',
                'ProfitLoss::Total Income',
                'Revenue From Operations',
                'Revenue',
                'Sales',
                'Total Revenue',
            ],
            
            'Operating Income': [
                'ProfitLoss::Profit Before Exceptional Items and Tax',
                'ProfitLoss::Operating Profit',
                'ProfitLoss::EBIT',
                'ProfitLoss::Profit Before Interest and Tax',
                'ProfitLoss::Operating Income',
                'ProfitLoss::Earnings Before Interest and Tax',
                'ProfitLoss::Profit from Operations',
                'Operating Profit',
                'EBIT',
                'Operating Income',
            ],
            
            'Net Income': [
                'ProfitLoss::Profit After Tax',
                'ProfitLoss::Profit/Loss For The Period',
                'ProfitLoss::Net Profit',
                'ProfitLoss::PAT',
                'ProfitLoss::Net Income',
                'ProfitLoss::Profit for the Period',
                'ProfitLoss::Net Profit After Tax',
                'Profit After Tax',
                'Net Profit',
                'Net Income',
            ],
            
            'Tax Expense': [
                'ProfitLoss::Tax Expense',
                'ProfitLoss::Tax Expenses', 
                'ProfitLoss::Current Tax',
                'ProfitLoss::Total Tax Expense',
                'ProfitLoss::Income Tax',
                'ProfitLoss::Provision for Tax',
                'ProfitLoss::Tax Provision',
                'Tax Expense',
                'Current Tax',
                'Income Tax',
            ],
            
            'Interest Expense': [
                'ProfitLoss::Finance Cost',
                'ProfitLoss::Finance Costs',
                'ProfitLoss::Interest',
                'ProfitLoss::Interest Expense',
                'ProfitLoss::Interest and Finance Charges',
                'ProfitLoss::Financial Expenses',
                'ProfitLoss::Borrowing Costs',
                'Finance Cost',
                'Finance Costs',
                'Interest Expense',
                'Interest',
            ],
            
            'Income Before Tax': [
                'ProfitLoss::Profit Before Tax',
                'ProfitLoss::PBT',
                'ProfitLoss::Income Before Tax',
                'ProfitLoss::Earnings Before Tax',
                'Profit Before Tax',
                'PBT',
                'Income Before Tax',
            ],
            
            'Other Income': [
                'ProfitLoss::Other Income',
                'ProfitLoss::Other Operating Income',
                'ProfitLoss::Miscellaneous Income',
                'Other Income',
            ],
            
            'Interest Income': [
                'ProfitLoss::Interest Income',
                'ProfitLoss::Investment Income',
                'ProfitLoss::Income from Investments',
                'Interest Income',
            ],
        }
        
        # Try exact matches from patterns
        if target_metric in search_patterns:
            self.logger.info(f"[PN-FETCH-SEARCH] Searching {len(search_patterns[target_metric])} patterns for '{target_metric}'")
            
            for i, pattern in enumerate(search_patterns[target_metric]):
                if pattern in df.index:
                    # VALIDATE STATEMENT TYPE even for pattern matches
                    if required_statement:
                        if not pattern.startswith(f"{required_statement}::"):
                            self.logger.warning(f"[PN-FETCH-PATTERN] Pattern '{pattern}' has wrong statement type, skipping")
                            continue
                    
                    series = df.loc[pattern].fillna(0 if default_zero else np.nan)
                    self.logger.info(f"[PN-FETCH-FOUND] Pattern match #{i+1}: '{pattern}' for '{target_metric}'")
                    self._log_metric_fetch(target_metric, pattern, series, f"Pattern search #{i+1}")
                    return series
        
        # CRITICAL FIX: Total Liabilities
        if target_metric == "Total Liabilities":
            try:
                # First, try to find it with patterns
                patterns = ['Total Liabilities', 'TOTAL LIABILITIES']
                for pattern in patterns:
                    for idx in df.index:
                        if pattern.lower() in str(idx).lower() and 'BalanceSheet::' in str(idx):
                            self.logger.info(f"[PN-FETCH] Found '{target_metric}' via pattern search: '{idx}'")
                            return df.loc[idx].fillna(0 if default_zero else np.nan)
                
                # If not found, CALCULATE it
                assets = self._get_safe_series(df, "Total Assets")
                equity = self._get_safe_series(df, "Total Equity")
                liabilities = assets - equity
                self.logger.warning(f"[PN-FETCH] Calculated '{target_metric}' as Total Assets - Total Equity.")
                return liabilities
            except Exception as e:
                self.logger.error(f"Failed to find or calculate Total Liabilities: {e}")
                if default_zero: 
                    return pd.Series(0, index=df.columns)
                raise ValueError(f"Could not find or calculate '{target_metric}'")
        
        # Enhanced fuzzy matching for Capital Expenditure specifically
        if target_metric == 'Capital Expenditure':
            self.logger.info("[PN-FETCH-CAPEX] Performing enhanced CapEx fuzzy search...")
            
            # Keywords that indicate capital expenditure
            capex_keywords = [
                'purchase', 'purchased', 'acquisition', 'addition', 'additions',
                'investment', 'expenditure', 'capex', 'fixed asset', 'ppe',
                'property plant', 'plant and equipment', 'machinery', 'equipment'
            ]
            
            # Look for any index containing these keywords
            for idx in df.index:
                idx_lower = str(idx).lower()
                
                # Must be in cash flow section
                if 'cashflow::' in idx_lower or 'cash flow' in idx_lower:
                    # Check for capex keywords
                    if any(keyword in idx_lower for keyword in capex_keywords):
                        # Additional validation - should be an outflow
                        series = df.loc[idx]
                        if series.notna().any():
                            self.logger.info(f"[PN-FETCH-CAPEX-FUZZY] Found potential CapEx: '{idx}'")
                            
                            # Ensure it's treated as an outflow (positive for subtraction)
                            series = series.abs().fillna(0 if default_zero else np.nan)
                            self._log_metric_fetch(target_metric, idx, series, "CapEx fuzzy match")
                            return series
        
        # General fuzzy matching for other metrics
        target_lower = target_metric.lower()
        best_match = None
        best_score = 0
        
        for idx in df.index:
            # VALIDATE STATEMENT TYPE in fuzzy matching
            if required_statement:
                if not idx.startswith(f"{required_statement}::"):
                    continue  # Skip items from wrong statement
            
            idx_clean = str(idx).split('::')[-1].lower() if '::' in str(idx) else str(idx).lower()
            
            # Calculate similarity score
            score = self._calculate_similarity(target_lower, idx_clean)
            
            if score > best_score and score > 0.7:  # 70% threshold
                best_score = score
                best_match = idx
        
        if best_match:
            series = df.loc[best_match].fillna(0 if default_zero else np.nan)
            self.logger.info(f"[PN-FETCH-FUZZY] Found '{target_metric}' via fuzzy match: {best_match} (score: {best_score:.2f})")
            self._log_metric_fetch(target_metric, best_match, series, f"Fuzzy match (score: {best_score:.2f})")
            return series
        
        # List of optional metrics that can default to zero
        optional_metrics = [
            'Depreciation', 'Investments', 'Short-term Investments',
            'Interest Income', 'Accrued Expenses', 'Deferred Revenue',
            'Total Liabilities', 'Capital Expenditure', 'Other Income'
        ]
        
        if default_zero or target_metric in optional_metrics:
            self.logger.info(f"[PN-FETCH-DEFAULT] Optional metric '{target_metric}' not found, using zeros")
            series = pd.Series(0, index=df.columns)
            self._log_metric_fetch(target_metric, "DEFAULT_ZEROS", series, "Default to zero")
            return series
        
        # For required metrics, show helpful suggestions
        self.logger.error(f"[PN-FETCH-ERROR] Required metric '{target_metric}' not found")
        
        # Show available items based on required statement type
        if required_statement:
            self.logger.info(f"[PN-FETCH-HELP] Available {required_statement} items in your data:")
            statement_items = [idx for idx in df.index if idx.startswith(f"{required_statement}::")]
            for item in statement_items[:20]:  # Show first 20
                self.logger.info(f"  - {item}")
            if len(statement_items) > 20:
                self.logger.info(f"  ... and {len(statement_items) - 20} more {required_statement} items")
        else:
            # Show available cash flow items for Capital Expenditure
            if target_metric == 'Capital Expenditure':
                self.logger.info("[PN-FETCH-HELP] Available Cash Flow items in your data:")
                cashflow_items = [idx for idx in df.index if 'cashflow::' in str(idx).lower()]
                for item in cashflow_items[:20]:  # Show first 20
                    self.logger.info(f"  - {item}")
                if len(cashflow_items) > 20:
                    self.logger.info(f"  ... and {len(cashflow_items) - 20} more cash flow items")
        
        raise ValueError(f"Required metric '{target_metric}' not found")

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        if not str1 or not str2:
            return 0.0
        
        # Tokenize
        tokens1 = set(str1.lower().split())
        tokens2 = set(str2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0

    def _is_optional_metric(self, metric: str) -> bool:
        """Check if a metric is optional"""
        optional_metrics = [
            'Depreciation', 'Investments', 'Short-term Investments',
            'Interest Income', 'Accrued Expenses', 'Deferred Revenue',
            'Intangible Assets', 'Other Income', 'Other Expenses',
            'Total Liabilities'  # Can be calculated
        ]
        return metric in optional_metrics

    def _calculate_data_completeness(self) -> float:
        """Calculate overall data completeness"""
        mapped_metrics = [self._find_source_metric(target) for target in self.mappings.values()]
        mapped_metrics = [m for m in mapped_metrics if m and m in self._df_clean.index]
        
        if not mapped_metrics:
            return 0.0
        
        total_cells = len(mapped_metrics) * len(self._df_clean.columns)
        non_null_cells = sum(self._df_clean.loc[metric].notna().sum() for metric in mapped_metrics)
        
        return non_null_cells / total_cells if total_cells > 0 else 0.0

    # PASTE THIS CODE: Add this method to the EnhancedPenmanNissimAnalyzer class (continuation)
    
    def detect_capex_candidates(self) -> List[Tuple[str, float]]:
        """Detect potential Capital Expenditure candidates with confidence scores"""
        candidates = []
        
        # Keywords that strongly indicate CapEx
        strong_indicators = ['capex', 'capital expenditure', 'purchase of fixed assets', 'purchase of ppe']
        medium_indicators = ['purchase', 'acquisition', 'addition', 'investment in fixed']
        weak_indicators = ['fixed asset', 'property plant', 'machinery', 'equipment']
        
        for idx in self._df_clean.index:
            idx_str = str(idx)
            idx_lower = idx_str.lower()
            
            # Must be cash flow related
            if not ('cashflow::' in idx_lower or 'cash flow' in idx_lower):
                continue
            
            confidence = 0.0
            
            # Check for strong indicators
            for indicator in strong_indicators:
                if indicator in idx_lower:
                    confidence += 0.9
                    break
            
            # Check for medium indicators
            if confidence < 0.5:
                for indicator in medium_indicators:
                    if indicator in idx_lower:
                        confidence += 0.6
                        break
            
            # Check for weak indicators
            if confidence < 0.3:
                for indicator in weak_indicators:
                    if indicator in idx_lower:
                        confidence += 0.3
                        break
            
            # Boost confidence if it's clearly an outflow
            if confidence > 0:
                series = self._df_clean.loc[idx]
                if series.notna().any():
                    # Check if values are typically positive (indicating outflows in cash flow)
                    positive_ratio = (series > 0).sum() / series.notna().sum()
                    if positive_ratio > 0.7:
                        confidence += 0.2
            
            if confidence > 0.3:
                candidates.append((idx_str, min(confidence, 1.0)))
        
        # Sort by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"[PN-CAPEX-DETECT] Found {len(candidates)} CapEx candidates:")
        for idx, conf in candidates[:5]:  # Log top 5
            self.logger.info(f"  - {idx} (confidence: {conf:.2f})")
        
        return candidates
    
    def suggest_missing_mappings(self) -> Dict[str, List[Tuple[str, float]]]:
        """Suggest mappings for missing required metrics"""
        suggestions = {}
        
        required_metrics = [
            'Capital Expenditure', 'Depreciation', 'Operating Cash Flow',
            'Interest Income', 'Cost of Goods Sold', 'Operating Expenses'
        ]
        
        for metric in required_metrics:
            # Check if already mapped
            if self._find_source_metric(metric):
                continue
            
            # Find candidates
            if metric == 'Capital Expenditure':
                candidates = self.detect_capex_candidates()
            else:
                candidates = self._find_metric_candidates(metric)
            
            if candidates:
                suggestions[metric] = candidates[:3]  # Top 3 suggestions
        
        return suggestions
    
    def _find_metric_candidates(self, target_metric: str) -> List[Tuple[str, float]]:
        """Find candidates for a specific metric"""
        candidates = []
        target_lower = target_metric.lower()
        
        # Define keywords for each metric type
        metric_keywords = {
            'depreciation': ['depreciation', 'amortisation', 'amortization'],
            'operating cash flow': ['operating activities', 'cash from operating', 'operating cash'],
            'interest income': ['interest income', 'other income', 'investment income'],
            'cost of goods sold': ['cost of materials', 'cost of goods', 'cogs', 'cost of sales'],
            'operating expenses': ['operating expenses', 'other expenses', 'employee benefit']
        }
        
        keywords = metric_keywords.get(target_lower, target_lower.split())
        
        for idx in self._df_clean.index:
            idx_lower = str(idx).lower()
            
            confidence = 0.0
            for keyword in keywords:
                if keyword in idx_lower:
                    confidence += 0.7
            
            # Boost for exact matches
            if target_lower in idx_lower:
                confidence += 0.3
            
            if confidence > 0.4:
                candidates.append((str(idx), min(confidence, 1.0)))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
        
    def debug_data_transfer(self, target_item: str) -> Dict[str, Any]:
        """Debug data transfer for a specific item"""
        debug_info = {
            'original_data': {},
            'year_mapping': {},
            'transfer_results': {},
            'issues': []
        }
        
        if target_item not in self.df.index:
            debug_info['issues'].append(f"Item '{target_item}' not found in original data")
            return debug_info
        
        # Get original data
        original_series = self.df.loc[target_item]
        debug_info['original_data'] = original_series.to_dict()
        
        # Analyze year patterns in columns
        year_patterns = [
            (re.compile(r'(\d{6})'), lambda m: m.group(1)),
            (re.compile(r'(\d{4})(?!\d)'), lambda m: m.group(1) + '03'),
        ]
        
        for col in self.df.columns:
            col_str = str(col)
            for pattern, extractor in year_patterns:
                match = pattern.search(col_str)
                if match:
                    try:
                        normalized_year = extractor(match)
                        year_int = int(normalized_year[:4])
                        if 2000 <= year_int <= 2099:
                            debug_info['year_mapping'][col] = normalized_year
                            break
                    except:
                        continue
        
        # Check what happened during transfer
        if target_item in self._df_clean.index:
            clean_series = self._df_clean.loc[target_item]
            debug_info['transfer_results'] = clean_series.to_dict()
            
            # Compare original vs clean
            original_non_null = original_series.notna().sum()
            clean_non_null = clean_series.notna().sum()
            
            if clean_non_null < original_non_null:
                debug_info['issues'].append(f"Data loss: {original_non_null} -> {clean_non_null} values")
        else:
            debug_info['issues'].append("Item not found in clean data")
        
        return debug_info
       
# --- 21. Manual Mapping Interface ---
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
            'Interest Expense', 'Tax Expense', 'EBIT', 'EBITDA',
            'Capital Expenditure', 'Depreciation', 'Income Before Tax'
        ]
    
    def render(self) -> Dict[str, str]:
        """Render the manual mapping interface and return mappings"""
        st.subheader("📋 Manual Metric Mapping")
        st.info("Map your financial statement items to standard metrics for analysis")
        
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
                'Capital Expenditure': ['Capital Expenditure', 'CAPEX', 'Fixed Asset Purchases'],
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
                        
                        # Use unique key with all indices
                        selected = st.selectbox(
                            f"{target}:",
                            ['(Not mapped)'] + self.source_metrics,
                            index=default_idx,
                            key=f"map_{statement_type}_{target}_{i}_{j}_{id(self)}",
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
                    key=f"custom_source_mapping_{id(self)}"
                )
            
            with col2:
                custom_target = st.selectbox(
                    "Target Metric:",
                    self.target_metrics,
                    key=f"custom_target_mapping_{id(self)}"
                )
            
            if st.button("Add Mapping", key=f"add_custom_mapping_btn_{id(self)}"):
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


# --- 22. Machine Learning Forecasting Module ---
class MLForecaster:
    """Machine learning based financial forecasting"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.logger = LoggerFactory.get_logger('MLForecaster')
        self.models = {
            'linear': self._train_linear,
            'polynomial': self._train_polynomial,
            'exponential': self._train_exponential,
            'auto': self._train_auto
        }
    
    @error_boundary({'error': 'Forecasting failed'})
    def forecast_metrics(self, df: pd.DataFrame, periods: int = 3, 
                        model_type: str = 'auto', metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Forecast financial metrics using ML"""
        if not self.config.get('app.enable_ml_features', True):
            return {'error': 'ML features disabled'}
        
        with performance_monitor.measure("ml_forecast"):
            if model_type == 'auto':
                model_type = self._select_best_model(df)
            
            if metrics is None:
                metrics = self._select_key_metrics(df)
            
            forecasts = {}
            accuracy_metrics = {}
            
            for metric in metrics:
                if metric in df.index:
                    series = df.loc[metric].dropna()
                    if len(series) >= 3:
                        try:
                            model = self.models[model_type](series)
                            forecast = self._generate_forecast(model, series, periods)
                            accuracy = self._calculate_accuracy(model, series)
                            
                            forecasts[metric] = forecast
                            accuracy_metrics[metric] = accuracy
                        except Exception as e:
                            self.logger.error(f"Error forecasting {metric}: {e}")
            
            return {
                'forecasts': forecasts,
                'accuracy_metrics': accuracy_metrics,
                'model_type': model_type,
                'periods': periods,
                'confidence_intervals': self._calculate_confidence_intervals(forecasts)
            }
    
    def _select_best_model(self, df: pd.DataFrame) -> str:
        """Automatically select best model based on data characteristics"""
        return 'linear'
    
    def _select_key_metrics(self, df: pd.DataFrame) -> List[str]:
        """Select key metrics to forecast"""
        priority_metrics = ['Revenue', 'Net Income', 'Total Assets', 'Operating Cash Flow']
        available_metrics = []
        
        for metric in priority_metrics:
            matching = [idx for idx in df.index if metric.lower() in str(idx).lower()]
            if matching:
                available_metrics.append(matching[0])
        
        return available_metrics[:4]
    
    def _train_linear(self, series: pd.Series) -> Any:
        """Train linear regression model"""
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model
    
    def _train_polynomial(self, series: pd.Series, degree: int = 2) -> Any:
        """Train polynomial regression model"""
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        
        return model
    
    def _train_exponential(self, series: pd.Series) -> Any:
        """Train exponential growth model"""
        X = np.arange(len(series)).reshape(-1, 1)
        y = np.log(series.values + 1)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Return wrapper that exponentiates predictions
        class ExpModel:
            def __init__(self, base_model):
                self.base_model = base_model
            
            def predict(self, X):
                log_pred = self.base_model.predict(X)
                return np.exp(log_pred) - 1
        
        return ExpModel(model)
    
    def _train_auto(self, series: pd.Series) -> Any:
        """Automatically select and train best model"""
        models = {
            'linear': self._train_linear(series),
            'polynomial': self._train_polynomial(series),
        }
        
        test_size = max(1, len(series) // 5)
        train_size = len(series) - test_size
        
        best_model = None
        best_score = float('inf')
        
        for name, model in models.items():
            X_test = np.arange(train_size, len(series)).reshape(-1, 1)
            y_test = series.values[train_size:]
            y_pred = model.predict(X_test)
            
            mse = np.mean((y_test - y_pred) ** 2)
            if mse < best_score:
                best_score = mse
                best_model = model
        
        return best_model
    
    def _generate_forecast(self, model: Any, series: pd.Series, periods: int) -> Dict[str, Any]:
        """Generate forecast for future periods"""
        last_index = len(series)
        future_indices = np.arange(last_index, last_index + periods).reshape(-1, 1)
        
        predictions = model.predict(future_indices)
        
        # Generate future period labels
        if all(series.index.astype(str).str.match(r'^\d{4}$')):
            # Year format
            last_year = int(series.index[-1])
            future_periods = [str(last_year + i + 1) for i in range(periods)]
        else:
            future_periods = [f"Period {i+1}" for i in range(periods)]
        
        return {
            'periods': future_periods,
            'values': predictions.tolist(),
            'last_actual': series.iloc[-1]
        }
    
    def _calculate_accuracy(self, model: Any, series: pd.Series) -> Dict[str, float]:
        """Calculate model accuracy metrics"""
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        y_pred = model.predict(X)
        
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        mape = np.mean(np.abs((y - y_pred) / y)) * 100 if not (y == 0).any() else None
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'rmse': np.sqrt(mse)
        }
    
    def _calculate_confidence_intervals(self, forecasts: Dict[str, Any], 
                                      confidence: float = 0.95) -> Dict[str, Any]:
        """Calculate confidence intervals for forecasts"""
        intervals = {}
        
        for metric, forecast in forecasts.items():
            values = np.array(forecast['values'])
            std = values.std() if len(values) > 1 else values[0] * 0.1
            
            z_score = stats.norm.ppf((1 + confidence) / 2)
            margin = z_score * std
            
            intervals[metric] = {
                'lower': (values - margin).tolist(),
                'upper': (values + margin).tolist()
            }
        
        return intervals

#--- 23. Natural Language Query Processor ---
class NLQueryProcessor:
    """Process natural language queries about financial data"""
    def __init__(self, config: Configuration):
        self.config = config
        self.logger = LoggerFactory.get_logger('NLQueryProcessor')
        self.intents = {
            'growth_rate': ['growth', 'increase', 'decrease', 'change', 'trend'],
            'comparison': ['compare', 'versus', 'vs', 'difference', 'better', 'worse'],
            'ratio': ['ratio', 'margin', 'return', 'turnover'],
            'forecast': ['forecast', 'predict', 'future', 'will', 'expect'],
            'summary': ['summary', 'overview', 'key', 'main', 'important']
        }
    
    @error_boundary({'type': 'error', 'message': 'Query processing failed'})
    def process_query(self, query: str, data: pd.DataFrame, 
                     analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process natural language query and return results"""
        query_lower = query.lower()
        
        # Classify intent
        intent = self._classify_intent(query_lower)
        entities = self._extract_entities(query_lower, data)
        
        self.logger.info(f"Query: {query}, Intent: {intent}, Entities: {entities}")
        
        # Process based on intent
        if intent == 'growth_rate':
            return self._handle_growth_query(entities, data, analysis)
        elif intent == 'comparison':
            return self._handle_comparison_query(entities, data, analysis)
        elif intent == 'ratio':
            return self._handle_ratio_query(entities, data, analysis)
        elif intent == 'forecast':
            return self._handle_forecast_query(entities, data)
        elif intent == 'summary':
            return self._handle_summary_query(entities, analysis)
        else:
            return self._handle_general_query(query, data, analysis)
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent"""
        for intent, keywords in self.intents.items():
            if any(keyword in query for keyword in keywords):
                return intent
        return 'general'
    
    def _extract_entities(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract entities from query"""
        entities = {
            'metrics': [],
            'years': [],
            'periods': []
        }
        
        # Extract metrics
        for metric in data.index:
            if str(metric).lower() in query:
                entities['metrics'].append(metric)
        
        # Extract years
        for col in data.columns:
            if str(col) in query:
                entities['years'].append(col)
        
        # Extract time periods
        if 'last year' in query:
            entities['periods'].append('last_year')
        elif 'this year' in query:
            entities['periods'].append('current_year')
        
        return entities
    
    def _handle_growth_query(self, entities: Dict, data: pd.DataFrame, 
                           analysis: Dict) -> Dict[str, Any]:
        """Handle growth rate queries"""
        results = {
            'type': 'growth_analysis',
            'data': []
        }
        
        metrics = entities.get('metrics', [])
        if not metrics:
            # Default to revenue if no metric specified
            revenue_metrics = [idx for idx in data.index if 'revenue' in str(idx).lower()]
            metrics = revenue_metrics[:1] if revenue_metrics else []
        
        for metric in metrics:
            if metric in data.index:
                series = data.loc[metric].dropna()
                if len(series) > 1:
                    # Calculate growth
                    growth_rate = ((series.iloc[-1] / series.iloc[0]) ** (1/(len(series)-1)) - 1) * 100
                    yoy_change = ((series.iloc[-1] / series.iloc[-2]) - 1) * 100 if len(series) > 1 else 0
                    
                    results['data'].append({
                        'metric': str(metric),
                        'cagr': growth_rate,
                        'yoy_change': yoy_change,
                        'first_value': series.iloc[0],
                        'last_value': series.iloc[-1],
                        'period': f"{series.index[0]} to {series.index[-1]}"
                    })
        
        return results
    
    def _handle_comparison_query(self, entities: Dict, data: pd.DataFrame, 
                               analysis: Dict) -> Dict[str, Any]:
        """Handle comparison queries"""
        return {
            'type': 'comparison',
            'message': 'Comparison analysis feature coming soon'
        }
    
    def _handle_ratio_query(self, entities: Dict, data: pd.DataFrame, 
                          analysis: Dict) -> Dict[str, Any]:
        """Handle ratio queries"""
        ratios = analysis.get('ratios', {})
        results = {
            'type': 'ratio_analysis',
            'data': {}
        }
        
        # Return all ratios or filtered based on query
        for category, ratio_df in ratios.items():
            if isinstance(ratio_df, pd.DataFrame):
                results['data'][category] = ratio_df.to_dict()
        
        return results
    
    def _handle_forecast_query(self, entities: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """Handle forecast queries"""
        # Use ML forecaster
        forecaster = MLForecaster(self.config)
        metrics = entities.get('metrics', None)
        
        forecast_results = forecaster.forecast_metrics(data, periods=3, metrics=metrics)
        
        return {
            'type': 'forecast',
            'data': forecast_results
        }
    
    def _handle_summary_query(self, entities: Dict, analysis: Dict) -> Dict[str, Any]:
        """Handle summary queries"""
        return {
            'type': 'summary',
            'data': {
                'summary': analysis.get('summary', {}),
                'insights': analysis.get('insights', [])[:5],
                'quality_score': analysis.get('quality_score', 0)
            }
        }
    
    def _handle_general_query(self, query: str, data: pd.DataFrame, 
                            analysis: Dict) -> Dict[str, Any]:
        """Handle general queries"""
        return {
            'type': 'general',
            'message': f"I understood your query: '{query}', but I need more specific information to help you.",
            'suggestions': [
                "What was the revenue growth last year?",
                "Show me the profitability ratios",
                "Forecast revenue for next 3 years",
                "Give me a summary of the financial performance"
            ]
        }

#--- 24. Collaboration Manager ---
class CollaborationManager:
    """Manage collaborative analysis sessions with cleanup"""
    def __init__(self):
        self.active_sessions = {}
        self.shared_analyses = {}
        self.user_presence = defaultdict(dict)
        self._lock = threading.Lock()
        self.logger = LoggerFactory.get_logger('CollaborationManager')
        self._cleanup_thread = None
        self._start_cleanup_thread()
    
    def __del__(self):
        """Cleanup on destruction"""
        self._stop_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread for session cleanup"""
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self._cleanup_thread.start()
    
    def _stop_cleanup_thread(self):
        """Stop cleanup thread"""
        # Implement proper shutdown mechanism if needed
        pass
    
    def _cleanup_expired_sessions(self):
        """Periodically clean up expired sessions"""
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes
                
                with self._lock:
                    current_time = datetime.now()
                    expired_sessions = []
                    
                    for session_id, session in self.active_sessions.items():
                        # Remove sessions inactive for more than 1 hour
                        if (current_time - session['last_activity']).seconds > 3600:
                            expired_sessions.append(session_id)
                    
                    for session_id in expired_sessions:
                        del self.active_sessions[session_id]
                        if session_id in self.user_presence:
                            del self.user_presence[session_id]
                        self.logger.info(f"Cleaned up expired session: {session_id}")
                        
            except Exception as e:
                self.logger.error(f"Error in cleanup thread: {e}")
    
    def create_session(self, analysis_id: str, owner_id: str) -> str:
        """Create a new collaborative session"""
        with self._lock:
            session_id = hashlib.md5(f"{analysis_id}_{owner_id}_{time.time()}".encode()).hexdigest()[:8]
            
            self.active_sessions[session_id] = {
                'analysis_id': analysis_id,
                'owner': owner_id,
                'participants': [owner_id],
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'chat_history': [],
                'annotations': {}
            }
            
            self.logger.info(f"Created session {session_id} for analysis {analysis_id}")
            return session_id
    
    def join_session(self, session_id: str, user_id: str) -> bool:
        """Join an existing session"""
        with self._lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                if user_id not in session['participants']:
                    session['participants'].append(user_id)
                
                session['last_activity'] = datetime.now()
                
                self.user_presence[session_id][user_id] = {
                    'joined_at': datetime.now(),
                    'last_seen': datetime.now(),
                    'cursor_position': None
                }
                
                self.logger.info(f"User {user_id} joined session {session_id}")
                return True
            
            return False
    
    def share_analysis(self, analysis_data: Dict[str, Any], owner_id: str, 
                      permissions: List[str] = None) -> str:
        """Share analysis with other users"""
        if permissions is None:
            permissions = ['view', 'comment']
        
        share_token = hashlib.md5(f"{owner_id}_{time.time()}".encode()).hexdigest()[:12]
        
        with self._lock:
            self.shared_analyses[share_token] = {
                'data': analysis_data,
                'owner': owner_id,
                'permissions': permissions,
                'created_at': datetime.now(),
                'access_log': []
            }
        
        return share_token
    
    def get_shared_analysis(self, share_token: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get shared analysis by token"""
        with self._lock:
            if share_token in self.shared_analyses:
                analysis = self.shared_analyses[share_token]
                
                # Log access
                analysis['access_log'].append({
                    'user': user_id,
                    'timestamp': datetime.now()
                })
                
                return analysis
            
            return None
    
    def add_annotation(self, session_id: str, user_id: str, 
                      annotation: Dict[str, Any]) -> bool:
        """Add annotation to collaborative session"""
        with self._lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                annotation_id = f"{user_id}_{time.time()}"
                session['annotations'][annotation_id] = {
                    'user': user_id,
                    'timestamp': datetime.now(),
                    'type': annotation.get('type', 'comment'),
                    'content': annotation.get('content', ''),
                    'position': annotation.get('position', None)
                }
                
                session['last_activity'] = datetime.now()
                
                return True
            
            return False
    
    def get_session_activity(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session activity and participants"""
        with self._lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                presence = self.user_presence.get(session_id, {})
                
                return {
                    'participants': session['participants'],
                    'active_users': [
                        uid for uid, data in presence.items()
                        if (datetime.now() - data['last_seen']).seconds < 300  # 5 minutes
                    ],
                    'annotations': session['annotations'],
                    'chat_history': session['chat_history'][-50:]  # Last 50 messages
                }
            
            return None

# --- 25. Tutorial System ---
class TutorialSystem:
    """Interactive tutorial system for new users"""
    
    def __init__(self):
        self.steps = [
            {
                'id': 'welcome',
                'title': 'Welcome to Elite Financial Analytics',
                'content': 'This tutorial will guide you through the platform features.',
                'location': 'main',
                'action': None
            },
            {
                'id': 'upload',
                'title': 'Upload Financial Data',
                'content': 'Start by uploading your financial statements using the sidebar.',
                'location': 'sidebar',
                'action': 'highlight_upload'
            },
            {
                'id': 'mapping',
                'title': 'Map Your Metrics',
                'content': 'Our AI will automatically map your metrics, or you can do it manually.',
                'location': 'main',
                'action': 'show_mapping'
            },
            {
                'id': 'analysis',
                'title': 'Explore Analysis',
                'content': 'Navigate through different analysis tabs to explore ratios, trends, and insights.',
                'location': 'tabs',
                'action': 'highlight_tabs'
            },
            {
                'id': 'export',
                'title': 'Export Results',
                'content': 'Generate and download comprehensive reports in various formats.',
                'location': 'reports',
                'action': 'show_export'
            }
        ]
        self.completed_steps = set()

    def render(self):
        """Render tutorial interface"""
        if not SimpleState.get('show_tutorial', True):
            return

        current_step = SimpleState.get('tutorial_step', 0)
        
        if current_step >= len(self.steps):
            self._complete_tutorial()
            return
        
        step = self.steps[current_step]
        
        # Tutorial overlay
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col2:
                st.info(f"""
                ### Tutorial Step {current_step + 1}/{len(self.steps)}: {step['title']}
                
                {step['content']}
                """)
                
                col_prev, col_next, col_skip = st.columns(3)
                
                with col_prev:
                    if current_step > 0:
                        if st.button("← Previous", key="tutorial_prev"):
                            SimpleState.set('tutorial_step', current_step - 1)
                
                with col_next:
                    if st.button("Next →", key="tutorial_next", type="primary"):
                        self.completed_steps.add(step['id'])
                        SimpleState.set('tutorial_step', current_step + 1)
                
                with col_skip:
                    if st.button("Skip Tutorial", key="tutorial_skip"):
                        self._complete_tutorial()
        
        # Execute step action
        if step['action']:
            self._execute_action(step['action'])
    
    def _execute_action(self, action: str):
        """Execute tutorial action"""
        if action == 'highlight_upload':
            st.sidebar.markdown("⬆️ Upload your files here")
        elif action == 'highlight_tabs':
            st.markdown("⬆️ Explore different analysis tabs above")
        # Add more actions as needed
    
    def _complete_tutorial(self):
        """Mark tutorial as completed"""
        SimpleState.set('show_tutorial', False)
        SimpleState.set('tutorial_completed', True)
        st.success("Tutorial completed! You're ready to use the platform.")


# ==============================================================================
# 26. Export Manager
# ==============================================================================

class ExportManager:
    """Handle various export formats for analysis results"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.logger = LoggerFactory.get_logger('ExportManager')
    
    @error_boundary("Export failed")
    def export_to_excel(self, analysis: Dict[str, Any], filename: str = "analysis.xlsx") -> bytes:
        """Export analysis to Excel format"""
        output = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Summary sheet
                if 'summary' in analysis:
                    summary_df = pd.DataFrame([analysis['summary']])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Ratios sheets
                if 'ratios' in analysis:
                    for category, ratio_df in analysis['ratios'].items():
                        if isinstance(ratio_df, pd.DataFrame):
                            sheet_name = f'Ratios_{category}'[:31]
                            ratio_df.to_excel(writer, sheet_name=sheet_name)
                
                # Trends sheet
                if 'trends' in analysis:
                    trends_data = []
                    for metric, trend in analysis['trends'].items():
                        if isinstance(trend, dict):
                            trend_row = {'Metric': metric}
                            trend_row.update(trend)
                            trends_data.append(trend_row)
                    
                    if trends_data:
                        trends_df = pd.DataFrame(trends_data)
                        trends_df.to_excel(writer, sheet_name='Trends', index=False)
                
                # Insights sheet
                if 'insights' in analysis:
                    insights_df = pd.DataFrame({'Insights': analysis['insights']})
                    insights_df.to_excel(writer, sheet_name='Insights', index=False)
                
                # Filtered data if present
                if 'filtered_data' in analysis:
                    analysis['filtered_data'].to_excel(writer, sheet_name='Data')
        
        except Exception as e:
            self.logger.error(f"Excel export error: {e}")
            raise
        
        output.seek(0)
        return output.read()
        
    def export_to_pdf(self, analysis: Dict[str, Any], charts: List[Any] = None) -> bytes:
        """Export analysis to PDF format (placeholder)"""
        self.logger.info("PDF export requested - placeholder implementation")
        return b"PDF export not yet implemented. Please use Excel or Markdown export."
    
    def export_to_powerpoint(self, analysis: Dict[str, Any], template: str = 'default') -> bytes:
        """Export analysis to PowerPoint format (placeholder)"""
        self.logger.info("PowerPoint export requested - placeholder implementation")
        return b"PowerPoint export not yet implemented. Please use Excel or Markdown export."
    
    def export_to_markdown(self, analysis: Dict[str, Any]) -> str:
        """Export analysis to Markdown format"""
        lines = [
            f"# Financial Analysis Report",
            f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nCompany: {analysis.get('company_name', 'Financial Analysis')}",
            "\n---\n"
        ]
    
        # Summary section
        if 'summary' in analysis:
            summary = analysis['summary']
            lines.extend([
                "## Executive Summary\n",
                f"- **Total Metrics Analyzed:** {summary.get('total_metrics', 'N/A')}",
                f"- **Period Covered:** {summary.get('year_range', 'N/A')}",
            ])
            
            if 'quality_score' in analysis:
                lines.append(f"- **Data Quality Score:** {analysis['quality_score']:.1f}%")
            
            if 'completeness' in summary:
                lines.append(f"- **Data Completeness:** {summary['completeness']:.1f}%")
            
            lines.append("\n")
    
        # Key Insights
        if 'insights' in analysis and analysis['insights']:
            lines.extend([
                "## Key Insights\n"
            ])
            for insight in analysis['insights']:
                lines.append(f"- {insight}")
            lines.append("\n")
    
        # Financial Ratios
        if 'ratios' in analysis:
            lines.extend([
                "## Financial Ratios\n"
            ])
            
            for category, ratio_df in analysis['ratios'].items():
                if isinstance(ratio_df, pd.DataFrame) and not ratio_df.empty:
                    lines.append(f"\n### {category} Ratios\n")
                    lines.append(ratio_df.to_markdown())
                    lines.append("\n")
    
        # Trends
        if 'trends' in analysis:
            lines.extend([
                "## Trend Analysis\n"
            ])
            
            significant_trends = []
            for metric, trend in analysis['trends'].items():
                if isinstance(trend, dict) and trend.get('cagr') is not None:
                    significant_trends.append({
                        'Metric': metric,
                        'Direction': trend['direction'],
                        'CAGR %': f"{trend['cagr']:.1f}" if trend['cagr'] is not None else 'N/A',
                        'R-squared': f"{trend.get('r_squared', 0):.3f}"
                    })
            
            if significant_trends:
                trend_df = pd.DataFrame(significant_trends)
                lines.append(trend_df.to_markdown(index=False))
                lines.append("\n")
    
        return "\n".join(lines)


# --- 27. UI Components Factory ---
class UIComponentFactory:
    """Factory for creating UI components with consistent styling"""

    @staticmethod
    def create_metric_card(title: str, value: Any, delta: Optional[float] = None, 
                          help: Optional[str] = None, delta_color: str = "normal") -> None:
        """Create a metric card with optional delta"""
        if help:
            st.metric(title, value, delta, delta_color=delta_color, help=help)
        else:
            st.metric(title, value, delta, delta_color=delta_color)

    @staticmethod
    def create_progress_indicator(progress: float, text: str = "") -> None:
        """Create animated progress indicator"""
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
            f'border-radius: 5px; margin: 5px 0; border-left: 4px solid;">'
            f'{icon} {insight}</div>',
            unsafe_allow_html=True
        )

    @staticmethod
    def render_with_skeleton(render_func: Callable, loading_key: str):
        """Render with skeleton loading state"""
        if SimpleState.get(f'{loading_key}_loading', False):
            for _ in range(3):
                st.container().markdown(
                    """
                    <div class="skeleton"></div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            render_func()


# --- 28. Sample Data Generator ---
class SampleDataGenerator:
    """Generate sample financial data for demonstration"""

    @staticmethod
    def generate_indian_tech_company() -> pd.DataFrame:
        """Generate sample data for Indian tech company"""
        years = ['2019', '2020', '2021', '2022', '2023']

        data = {
            # Balance Sheet Items (in ₹ Crores)
            'Total Assets': [450, 520, 610, 720, 850],
            'Current Assets': [280, 320, 380, 450, 530],
            'Non-current Assets': [170, 200, 230, 270, 320],
            'Cash and Cash Equivalents': [120, 140, 170, 210, 250],
            'Inventory': [20, 23, 27, 32, 38],
            'Trade Receivables': [80, 92, 108, 127, 150],
            'Property Plant and Equipment': [100, 120, 140, 165, 195],
            
            'Total Liabilities': [180, 200, 225, 255, 290],
            'Current Liabilities': [100, 110, 125, 140, 160],
            'Non-current Liabilities': [80, 90, 100, 115, 130],
            'Short-term Borrowings': [30, 33, 37, 42, 48],
            'Long-term Debt': [60, 66, 73, 82, 92],
            'Trade Payables': [40, 44, 49, 55, 62],
            
            'Total Equity': [270, 320, 385, 465, 560],
            'Share Capital': [100, 100, 100, 100, 100],
            'Reserves and Surplus': [170, 220, 285, 365, 460],
            
            # Income Statement Items
            'Revenue': [350, 380, 450, 540, 650],
            'Cost of Goods Sold': [210, 220, 252, 297, 351],
            'Gross Profit': [140, 160, 198, 243, 299],
            'Operating Expenses': [80, 88, 103, 121.5, 143],
            'Operating Income': [60, 72, 95, 121.5, 156],
            'EBIT': [60, 72, 95, 121.5, 156],
            'Interest Expense': [8, 8.8, 9.7, 10.9, 12.2],
            'Income Before Tax': [52, 63.2, 85.3, 110.6, 143.8],
            'Tax Expense': [15.6, 18.96, 25.59, 33.18, 43.14],
            'Net Income': [36.4, 44.24, 59.71, 77.42, 100.66],
            
            # Cash Flow Items
            'Operating Cash Flow': [55, 66, 88, 110, 140],
            'Investing Cash Flow': [-30, -35, -42, -50, -60],
            'Financing Cash Flow': [-15, -18, -22, -27, -33],
            'Capital Expenditure': [28, 32, 38, 45, 53],
            'Free Cash Flow': [27, 34, 50, 65, 87],
            'Depreciation': [15, 18, 21, 25, 30],
        }

        df = pd.DataFrame(data, index=list(data.keys()), columns=years)
        return df

    @staticmethod
    def generate_us_manufacturing() -> pd.DataFrame:
        """Generate sample data for US manufacturing company"""
        years = ['2019', '2020', '2021', '2022', '2023']

        data = {
            # Balance Sheet Items (in millions USD)
            'Total Assets': [1200, 1150, 1250, 1350, 1450],
            'Current Assets': [450, 430, 480, 520, 560],
            'Non-current Assets': [750, 720, 770, 830, 890],
            'Cash and Cash Equivalents': [80, 75, 90, 105, 120],
            'Inventory': [180, 170, 190, 210, 230],
            'Trade Receivables': [150, 145, 160, 175, 190],
            'Property Plant and Equipment': [600, 580, 620, 660, 700],
            
            'Total Liabilities': [720, 690, 740, 790, 840],
            'Current Liabilities': [300, 280, 310, 330, 350],
            'Non-current Liabilities': [420, 410, 430, 460, 490],
            'Short-term Borrowings': [100, 90, 105, 115, 125],
            'Long-term Debt': [350, 340, 355, 380, 405],
            'Trade Payables': [120, 115, 125, 135, 145],
            
            'Total Equity': [480, 460, 510, 560, 610],
            'Share Capital': [200, 200, 200, 200, 200],
            'Retained Earnings': [280, 260, 310, 360, 410],
            
            # Income Statement Items
            'Revenue': [950, 880, 1020, 1150, 1280],
            'Cost of Goods Sold': [680, 640, 720, 800, 880],
            'Gross Profit': [270, 240, 300, 350, 400],
            'Operating Expenses': [180, 170, 190, 210, 230],
            'Operating Income': [90, 70, 110, 140, 170],
            'EBIT': [90, 70, 110, 140, 170],
            'Interest Expense': [28, 27, 28.5, 30.5, 32.5],
            'Income Before Tax': [62, 43, 81.5, 109.5, 137.5],
            'Tax Expense': [15.5, 10.75, 20.38, 27.38, 34.38],
            'Net Income': [46.5, 32.25, 61.12, 82.12, 103.12],
            
            # Cash Flow Items
            'Operating Cash Flow': [110, 90, 130, 160, 195],
            'Investing Cash Flow': [-80, -60, -90, -110, -130],
            'Financing Cash Flow': [-20, -25, -30, -35, -40],
            'Capital Expenditure': [75, 55, 85, 105, 125],
            'Free Cash Flow': [35, 35, 45, 55, 70],
            'Depreciation': [45, 48, 52, 58, 65],
        }

        df = pd.DataFrame(data, index=list(data.keys()), columns=years)
        return df

    @staticmethod
    def generate_european_retail() -> pd.DataFrame:
        """Generate sample data for European retail company"""
        years = ['2019', '2020', '2021', '2022', '2023']

        data = {
            # Balance Sheet Items (in millions EUR)
            'Total Assets': [800, 750, 820, 880, 950],
            'Current Assets': [400, 380, 420, 450, 490],
            'Non-current Assets': [400, 370, 400, 430, 460],
            'Cash and Cash Equivalents': [60, 55, 70, 85, 100],
            'Inventory': [200, 190, 210, 225, 245],
            'Trade Receivables': [80, 75, 85, 90, 95],
            'Property Plant and Equipment': [350, 325, 350, 375, 400],
            
            'Total Liabilities': [480, 450, 490, 520, 560],
            'Current Liabilities': [250, 235, 260, 275, 295],
            'Non-current Liabilities': [230, 215, 230, 245, 265],
            'Short-term Borrowings': [80, 75, 85, 90, 95],
            'Long-term Debt': [180, 170, 180, 190, 205],
            'Trade Payables': [130, 125, 135, 145, 155],
            
            'Total Equity': [320, 300, 330, 360, 390],
            'Share Capital': [150, 150, 150, 150, 150],
            'Retained Earnings': [170, 150, 180, 210, 240],
            
            # Income Statement Items
            'Revenue': [1200, 1050, 1300, 1450, 1600],
            'Cost of Goods Sold': [840, 750, 900, 1000, 1100],
            'Gross Profit': [360, 300, 400, 450, 500],
            'Operating Expenses': [280, 260, 300, 330, 360],
            'Operating Income': [80, 40, 100, 120, 140],
            'EBIT': [80, 40, 100, 120, 140],
            'Interest Expense': [18, 17, 18, 19, 20.5],
            'Income Before Tax': [62, 23, 82, 101, 119.5],
            'Tax Expense': [18.6, 6.9, 24.6, 30.3, 35.85],
            'Net Income': [43.4, 16.1, 57.4, 70.7, 83.65],
            
            # Cash Flow Items
            'Operating Cash Flow': [95, 60, 115, 135, 160],
            'Investing Cash Flow': [-50, -30, -60, -70, -85],
            'Financing Cash Flow': [-30, -25, -35, -40, -45],
            'Capital Expenditure': [45, 25, 55, 65, 80],
            'Free Cash Flow': [50, 35, 60, 70, 80],
            'Depreciation': [25, 27, 30, 33, 36],
        }

        df = pd.DataFrame(data, index=list(data.keys()), columns=years)
        return df


# --- 29. Error Recovery Mechanisms ---
class ErrorRecoveryManager:
    """Manage error recovery and fallback strategies"""

    def __init__(self):
        self.error_counts = defaultdict(int)
        self.recovery_strategies = {
            'kaggle_api_down': self._recover_kaggle_api,
            'model_load_failed': self._recover_model_load,
            'memory_exceeded': self._recover_memory
        }

    def handle_error(self, error_type: str, context: Dict[str, Any]) -> bool:
        """Handle error with appropriate recovery strategy"""
        self.error_counts[error_type] += 1

        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](context)

        return False

    def _recover_kaggle_api(self, context: Dict[str, Any]) -> bool:
        """Recover from Kaggle API failure"""
        wait_time = min(2 ** self.error_counts['kaggle_api_down'], 300)
        time.sleep(wait_time)

        return context.get('mapper', {}).get('_test_kaggle_connection', lambda: False)()

    def _recover_model_load(self, context: Dict[str, Any]) -> bool:
        """Recover from model loading failure"""
        alternative_models = ['all-MiniLM-L6-v2', 'paraphrase-MiniLM-L3-v2', 'distilbert-base-nli-mean-tokens']

        for model in alternative_models:
            try:
                # Attempt to load alternative model
                return True
            except:
                continue

        return False

    def _recover_memory(self, context: Dict[str, Any]) -> bool:
        """Recover from memory issues"""
        if 'mapper' in context:
            context['mapper'].embeddings_cache.clear()

        gc.collect()

        return True


def safe_state_access(func):
    """Decorator to ensure safe session state access with comprehensive error handling"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                return func(self, *args, **kwargs)
                
            except KeyError as e:
                retry_count += 1
                key_name = str(e).strip("'\"")
                
                # Log the error
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Session state key error in {func.__name__}: {e} (attempt {retry_count}/{max_retries})")
                
                # Try to fix the missing key
                try:
                    # Reinitialize session state
                    self._initialize_session_state()
                    
                    # If it's a specific key we can predict, set a safe default
                    safe_defaults = {
                        'simple_parse_mode': False,
                        'uploaded_files': [],
                        'ml_forecast_results': None,
                        'forecast_periods': 3,
                        'forecast_model_type': 'auto',
                        'selected_forecast_metrics': [],
                        'analysis_data': None,
                        'metric_mappings': None,
                        'show_manual_mapping': False,
                        'number_format_value': 'Indian',
                        'kaggle_api_enabled': False,
                        'api_metrics_visible': False,
                        'analysis_mode': 'Standalone Analysis',
                        'benchmark_company': 'ITC Ltd',
                        'show_tutorial': True,
                        'tutorial_step': 0,
                        'collaboration_session': None,
                        'query_history': [],
                        'pn_mappings': None,
                        'pn_results': None,
                    }
                    
                    if key_name in safe_defaults:
                        st.session_state[key_name] = safe_defaults[key_name]
                        if hasattr(self, 'logger'):
                            self.logger.info(f"Set safe default for missing key: {key_name}")
                    
                    # If this is the last retry, continue anyway
                    if retry_count >= max_retries:
                        if hasattr(self, 'logger'):
                            self.logger.error(f"Failed to fix session state after {max_retries} attempts in {func.__name__}")
                        # Try to continue with a fallback
                        try:
                            return func(self, *args, **kwargs)
                        except:
                            # Return a safe fallback
                            return None
                    
                except Exception as init_error:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Error during session state reinitialization: {init_error}")
                    
                    if retry_count >= max_retries:
                        raise e  # Re-raise original error
        
            except AttributeError as e:
                # Handle cases where self.logger or other attributes don't exist
                if "'FinancialAnalyticsPlatform' object has no attribute 'logger'" in str(e):
                    # Initialize logger if missing
                    try:
                        self.logger = LoggerFactory.get_logger('FinancialAnalyticsPlatform')
                        return func(self, *args, **kwargs)
                    except:
                        # Continue without logging
                        return func(self, *args, **kwargs)
                else:
                    raise e
        
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Unexpected error in {func.__name__}: {e}")
                
                # For non-KeyError exceptions, don't retry
                raise e

        # If we get here, all retries failed
        if hasattr(self, 'logger'):
            self.logger.critical(f"All retry attempts failed for {func.__name__}")
        return None
    return wrapper


def critical_method(func):
    """Decorator for critical methods that must not fail"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.critical(f"Critical method {func.__name__} failed: {e}")

            # Show user-friendly error
            st.error(f"A critical error occurred in {func.__name__}. Please refresh the page.")
            
            # Provide recovery options
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🔄 Retry {func.__name__}", key=f"retry_{func.__name__}"):
                    st.rerun()
            with col2:
                if st.button("🏠 Reset Application", key=f"reset_{func.__name__}"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
            return None
    return wrapper

#--mapping class
class PenmanNissimMappingTemplates:
    """Pre-defined mapping templates for common Indian financial statement formats"""
    
    INDIAN_GAAP_TEMPLATE = {
        # Balance Sheet Mappings
        'Total Assets': ['Total Assets', 'TOTAL ASSETS', 'Total Asset', 'Assets Total', 'Total Equity and Liabilities'],
        'Current Assets': ['Current Assets', 'Total Current Assets', 'CURRENT ASSETS'],
        'Cash and Cash Equivalents': ['Cash and Cash Equivalents', 'Cash And Cash Equivalents', 'Cash & Cash Equivalents', 'Cash and Bank Balances'],
        'Short-term Investments': ['Current Investments', 'Short Term Investments', 'Investments-Current', 'Other Current Assets'],
        'Trade Receivables': ['Trade Receivables', 'Sundry Debtors', 'Debtors', 'Accounts Receivable', 'Trade receivables'],
        'Inventory': ['Inventories', 'Inventory', 'Stock-in-Trade', 'Stock', 'Stores and Spares'],
        'Property Plant Equipment': ['Property, Plant and Equipment', 'Fixed Assets', 'Tangible Assets', 'Net Block', 'Property Plant and Equipment'],
        'Intangible Assets': ['Intangible Assets', 'Intangibles', 'Goodwill', 'Other Intangible Assets'],
        
        'Total Liabilities': ['Total Liabilities', 'TOTAL LIABILITIES', 'Total Non-Current Liabilities'],
        'Current Liabilities': ['Current Liabilities', 'Total Current Liabilities', 'CURRENT LIABILITIES'],
        'Accounts Payable': ['Trade Payables', 'Sundry Creditors', 'Creditors', 'Accounts Payable', 'Trade payables'],
        'Short-term Debt': ['Short Term Borrowings', 'Current Maturities of Long Term Debt', 'Short-term Borrowings', 'Other Current Liabilities'],
        'Long-term Debt': ['Long Term Borrowings', 'Long-term Borrowings', 'Term Loans', 'Debentures', 'Other Non-Current Liabilities'],
        'Accrued Expenses': ['Provisions', 'Other Current Liabilities', 'Accrued Expenses'],
        'Deferred Revenue': ['Deferred Revenue', 'Advance from Customers', 'Unearned Revenue'],
        
        'Total Equity': ['Total Equity', 'Shareholders Funds', 'Shareholder\'s Equity', 'Total Shareholders Funds', 'Net Worth', 'Equity'],
        'Share Capital': ['Share Capital', 'Equity Share Capital', 'Paid-up Capital'],
        'Retained Earnings': ['Reserves and Surplus', 'Retained Earnings', 'Other Equity'],
        
        # Income Statement Mappings
        'Revenue': ['Revenue From Operations', 'Revenue from Operations (Net)', 'Total Revenue', 'Net Sales', 'Sales', 'Revenue From Operations(Net)'],
        'Cost of Goods Sold': ['Cost of Materials Consumed', 'Cost of Goods Sold', 'Purchase of Stock-In-Trade', 'Changes in Inventory'],
        'Operating Expenses': ['Employee Benefit Expenses', 'Other Expenses', 'Operating Expenses'],
        'Operating Income': ['Operating Profit', 'EBIT', 'Operating Income', 'Profit Before Interest and Tax'],
        'EBIT': ['EBIT', 'Profit Before Interest and Tax', 'Operating Profit'],
        'Interest Expense': ['Finance Costs', 'Interest Expense', 'Interest and Finance Charges', 'Interest'],
        'Interest Income': ['Interest Income', 'Other Income', 'Investment Income'],
        'Tax Expense': ['Tax Expense', 'Total Tax Expense', 'Current Tax', 'Income Tax'],
        'Net Income': ['Net Profit', 'Profit After Tax', 'Net Income', 'PAT', 'Net Profit After Tax', 'Profit/Loss For The Period'],
        'Income Before Tax': ['Profit Before Tax', 'PBT', 'Income Before Tax', 'Profit before tax'],
        
        # Cash Flow Mappings
        'Operating Cash Flow': ['Cash Flow from Operating Activities', 'Net Cash from Operating Activities', 'Operating Cash Flow', 'Net CashFlow From Operating Activities'],
        'Investing Cash Flow': ['Cash Flow from Investing Activities', 'Net Cash Used in Investing Activities', 'Purchase of Investments'],
        'Financing Cash Flow': ['Cash Flow from Financing Activities', 'Net Cash Used in Financing Activities'],
        'Capital Expenditure': ['Purchase of Fixed Assets', 'Purchase of Property, Plant and Equipment', 'Capital Expenditure', 'Additions to Fixed Assets'],
        'Depreciation': ['Depreciation and Amortisation', 'Depreciation', 'Depreciation & Amortization', 'Depreciation and Amortisation Expenses']
    }
    
    @staticmethod
    def create_smart_mapping(source_metrics: List[str], template: Dict[str, List[str]]) -> Tuple[Dict[str, str], List[str]]:
        """Create mappings using template with fuzzy matching"""
        mappings = {}
        unmapped = []
        used_sources = set()
        
        # First pass: exact matches
        for source in source_metrics:
            source_clean = source.split('::')[-1] if '::' in source else source
            source_lower = source_clean.lower().strip()
            
            matched = False
            for target, patterns in template.items():
                for pattern in patterns:
                    if source_lower == pattern.lower():
                        if source not in used_sources:
                            mappings[source] = target
                            used_sources.add(source)
                            matched = True
                            break
                if matched:
                    break
        
        # Second pass: fuzzy matching for remaining items
        for source in source_metrics:
            if source in used_sources:
                continue
                
            source_clean = source.split('::')[-1] if '::' in source else source
            source_lower = source_clean.lower().strip()
            
            best_match = None
            best_score = 0
            
            for target, patterns in template.items():
                # Skip if target already has a mapping
                if target in mappings.values():
                    continue
                    
                for pattern in patterns:
                    score = fuzz.token_sort_ratio(source_lower, pattern.lower())
                    if score > best_score:
                        best_score = score
                        best_match = target
            
            if best_match and best_score >= 80:  # 80% threshold
                mappings[source] = best_match
                used_sources.add(source)
            else:
                unmapped.append(source)
        
        return mappings, unmapped

#--Add the Enhanced Mapping Classes
class EnhancedPenmanNissimMapper:
    """Enhanced mapper specifically for Penman-Nissim analysis"""
    
    def __init__(self):
        self.template = PenmanNissimMappingTemplates.INDIAN_GAAP_TEMPLATE
        self.required_mappings = {
            'essential': [
                'Total Assets', 'Total Equity', 'Revenue', 'Operating Income',
                'Net Income', 'Tax Expense', 'Interest Expense', 'Income Before Tax'
            ],
            'important': [
                'Current Assets', 'Current Liabilities', 'Cash and Cash Equivalents',
                'Operating Cash Flow', 'Capital Expenditure', 'Depreciation',
                'Short-term Debt', 'Long-term Debt', 'Share Capital'
            ],
            'optional': [
                'Trade Receivables', 'Inventory', 'Property Plant Equipment',
                'Accounts Payable', 'Interest Income', 'Cost of Goods Sold',
                'Operating Expenses', 'Accrued Expenses', 'Deferred Revenue'
            ]
        }
    
    def validate_mappings(self, mappings: Dict[str, str]) -> Dict[str, Any]:
        """Validate if mappings are sufficient for Penman-Nissim"""
        validation = {
            'is_valid': True,
            'completeness': 0,
            'missing_essential': [],
            'missing_important': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check essential mappings
        mapped_targets = set(mappings.values())
        
        for item in self.required_mappings['essential']:
            if item not in mapped_targets:
                validation['missing_essential'].append(item)
                validation['is_valid'] = False
        
        for item in self.required_mappings['important']:
            if item not in mapped_targets:
                validation['missing_important'].append(item)
        
        # Calculate completeness
        total_required = len(self.required_mappings['essential']) + len(self.required_mappings['important'])
        total_mapped = len([m for m in self.required_mappings['essential'] + self.required_mappings['important'] 
                           if m in mapped_targets])
        validation['completeness'] = (total_mapped / total_required) * 100 if total_required > 0 else 0
        
        # Add warnings and suggestions
        if validation['completeness'] < 60:
            validation['warnings'].append("Insufficient mappings for accurate Penman-Nissim analysis")
        
        if 'Operating Income' not in mapped_targets and 'EBIT' in mapped_targets:
            validation['suggestions'].append("Using EBIT as proxy for Operating Income")
        
        if 'Total Liabilities' not in mapped_targets:
            validation['suggestions'].append("Total Liabilities can be calculated from Total Assets - Total Equity")
        
        return validation

# 1. First, add this class before your FinancialAnalyticsPlatform class:

class EnhancedPenmanNissimValidator:
    """Enhanced validator specifically for Penman-Nissim calculations"""
    
    def __init__(self):
        self.required_relationships = {
            'RNOA': ['Operating Income', 'Net Operating Assets'],
            'FLEV': ['Net Financial Obligations', 'Common Equity'],
            'NBC': ['Net Financial Expense', 'Net Financial Obligations'],
            'ROE': ['Net Income', 'Common Equity']
        }
        
        self.accounting_relationships = {
            'total_assets': {
                'equals': ['total_liabilities_and_equity'],
                'sum_of': ['operating_assets', 'financial_assets']
            },
            'total_liabilities_and_equity': {
                'sum_of': ['total_liabilities', 'total_equity']
            },
            'net_operating_assets': {
                'equals': ['operating_assets', '-operating_liabilities']
            },
            'net_financial_obligations': {
                'equals': ['financial_liabilities', '-financial_assets']
            },
            'operating_income': {
                'equals': ['revenue', '-operating_expenses']
            },
            'net_income': {
                'equals': ['operating_income', '-financial_expenses', '-tax_expense']
            }
        }
        
        # FIXED: Separated direct mappings from calculated metrics
        self.essential_metrics = {
            'Balance Sheet': [
                'Total Assets',
                'Total Equity',
                # 'Total Liabilities' can be calculated if not present
                'Current Assets',
                'Current Liabilities',
                'Cash and Cash Equivalents'
                # REMOVED: Operating Assets, Operating Liabilities, Financial Assets, Financial Liabilities
                # These are CALCULATED, not mapped
            ],
            'Income Statement': [
                'Revenue',
                'Operating Income',
                'Interest Expense',
                'Tax Expense',
                'Net Income'
            ],
            'Cash Flow': [
                'Operating Cash Flow'
                # Made investing and financing optional
            ]
        }
        
        # Add component requirements for calculated metrics
        self.component_requirements = {
            'Operating Assets': ['Total Assets', 'Cash and Cash Equivalents'],
            's': ['Cash and Cash Equivalents'],
            'Operating Liabilities': ['Current Liabilities'],
            'Financial Liabilities': ['Short-term Debt', 'Long-term Debt', 'Short Term Borrowings', 'Long Term Borrowings']
        }
    
    def validate_mapping_for_pn(self, mappings: Dict[str, str], data: pd.DataFrame) -> Dict[str, Any]:
        """Validate mappings specifically for Penman-Nissim calculations"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'pn_metrics_validity': {},
            'missing_essential': [],
            'quality_score': 0
        }
        
        # Check essential metrics
        self._validate_essential_metrics(mappings, validation)
        
        # Check Penman-Nissim specific components
        pn_components = self._validate_pn_components(mappings, data)
        validation['pn_metrics_validity'].update(pn_components)
        
        # Validate accounting relationships
        if data is not None:
            accounting_validation = self._validate_accounting_relationships(mappings, data)
            for key, value in accounting_validation.items():
                if key in validation:
                    validation[key].extend(value)
                else:
                    validation[key] = value
        
        # Calculate quality score
        validation['quality_score'] = self._calculate_quality_score(validation)
        
        # Set overall validity - reduced threshold to 60
        validation['is_valid'] = (
            len(validation['errors']) == 0 and 
            len(validation['missing_essential']) == 0 and
            validation['quality_score'] >= 60  # Reduced from 70
        )
        
        return validation
    
    def _validate_essential_metrics(self, mappings: Dict[str, str], validation: Dict[str, Any]):
        """Check if all essential metrics are mapped"""
        mapped_targets = set(mappings.values())
        
        for statement, metrics in self.essential_metrics.items():
            for metric in metrics:
                if metric not in mapped_targets:
                    # Special handling for metrics that can be derived
                    if metric == 'Total Liabilities':
                        # Check if we can derive it
                        if 'Total Assets' in mapped_targets and 'Total Equity' in mapped_targets:
                            validation['suggestions'].append(
                                f"Total Liabilities will be calculated from Total Assets - Total Equity"
                            )
                        else:
                            validation['missing_essential'].append(metric)
                    else:
                        validation['missing_essential'].append(metric)
                        validation['suggestions'].append(f"Map {metric} from {statement}")
        
        # Check for debt items (needed for Financial Liabilities calculation)
        debt_items = ['Short-term Debt', 'Long-term Debt', 'Short Term Borrowings', 
                      'Long Term Borrowings', 'Total Debt']
        if not any(item in mapped_targets for item in debt_items):
            validation['warnings'].append("No debt items mapped - needed for leverage calculations")
    
    def _validate_pn_components(self, mappings: Dict[str, str], data: pd.DataFrame) -> Dict[str, bool]:
        """Validate core Penman-Nissim components"""
        components = {}
        
        # RNOA Components
        rnoa_valid = self._check_rnoa_components(mappings, data)
        components['RNOA'] = rnoa_valid
        
        # FLEV Components
        flev_valid = self._check_flev_components(mappings, data)
        components['FLEV'] = flev_valid
        
        # NBC Components
        nbc_valid = self._check_nbc_components(mappings, data)
        components['NBC'] = nbc_valid
        
        return components
    
    def _check_rnoa_components(self, mappings: Dict[str, str], data: pd.DataFrame) -> bool:
        """Check if RNOA can be calculated correctly"""
        mapped_targets = set(mappings.values())
        
        # For RNOA, we need:
        # 1. Operating Income (or equivalent)
        # 2. Components to calculate Net Operating Assets
        
        # Check for operating income variants
        operating_income_variants = ['Operating Income', 'EBIT', 'Operating Profit', 
                                    'Profit Before Exceptional Items and Tax']
        has_operating_income = any(item in mapped_targets for item in operating_income_variants)
        
        # Check for NOA components
        has_total_assets = 'Total Assets' in mapped_targets
        has_total_equity = 'Total Equity' in mapped_targets
        
        # We can calculate NOA if we have basic balance sheet items
        can_calculate_noa = has_total_assets and has_total_equity
        
        return has_operating_income and can_calculate_noa
    
    def _check_flev_components(self, mappings: Dict[str, str], data: pd.DataFrame) -> bool:
        """Check if Financial Leverage can be calculated correctly"""
        mapped_targets = set(mappings.values())
        
        # For FLEV, we need:
        # 1. Total Equity (for Common Equity)
        # 2. Some indication of financial position (debt or cash)
        
        has_equity = 'Total Equity' in mapped_targets
        
        # Check for financial items
        debt_items = ['Short-term Debt', 'Long-term Debt', 'Short Term Borrowings', 
                      'Long Term Borrowings', 'Total Debt']
        has_debt = any(item in mapped_targets for item in debt_items)
        
        has_cash = 'Cash and Cash Equivalents' in mapped_targets or 'Cash' in mapped_targets
        
        # Validate equity values if data is provided
        if data is not None and has_equity:
            equity_source = self._get_mapped_source(mappings, 'Total Equity')
            if equity_source and equity_source in data.index:
                if (data.loc[equity_source] <= 0).any():
                    return False  # Equity shouldn't be negative or zero
        
        return has_equity and (has_debt or has_cash)
    
    def _check_nbc_components(self, mappings: Dict[str, str], data: pd.DataFrame) -> bool:
        """Check if Net Borrowing Cost can be calculated correctly"""
        mapped_targets = set(mappings.values())
        
        # For NBC, we need:
        # 1. Interest Expense
        # 2. Some debt items
        
        has_interest = 'Interest Expense' in mapped_targets
        
        debt_items = ['Short-term Debt', 'Long-term Debt', 'Short Term Borrowings', 
                      'Long Term Borrowings', 'Total Debt']
        has_debt = any(item in mapped_targets for item in debt_items)
        
        # NBC can work with just Interest Expense and any debt item
        return has_interest and has_debt
    
    def _validate_accounting_relationships(self, mappings: Dict[str, str], data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate fundamental accounting relationships - handles Indian format"""
        validation = {
            'errors': [],
            'warnings': []
        }
        
        mapped_targets = set(mappings.values())
        
        # Check Balance Sheet equation
        if 'Total Assets' in mapped_targets and 'Total Equity' in mapped_targets:
            try:
                assets_source = self._get_mapped_source(mappings, 'Total Assets')
                equity_source = self._get_mapped_source(mappings, 'Total Equity')
                
                if assets_source and equity_source:
                    assets = data.loc[assets_source]
                    equity = data.loc[equity_source]
                    
                    # Check if this is Indian format (Total Equity and Liabilities = Total Assets)
                    if 'Total Equity and Liabilities' in assets_source:
                        # This is Indian format - no need to check equation
                        validation['warnings'].append(
                            "Using Indian accounting format (Total Equity and Liabilities). Equation inherently satisfied."
                        )
                    else:
                        # Traditional format - check equation if Total Liabilities exists
                        if 'Total Liabilities' in mapped_targets:
                            liab_source = self._get_mapped_source(mappings, 'Total Liabilities')
                            if liab_source and liab_source in data.index:
                                liabilities = data.loc[liab_source]
                                
                                # Check equation with tolerance
                                for year in data.columns:
                                    if all(pd.notna([assets[year], liabilities[year], equity[year]])):
                                        diff = abs(assets[year] - (liabilities[year] + equity[year]))
                                        tolerance = abs(assets[year]) * 0.05  # 5% tolerance
                                        
                                        if diff > tolerance:
                                            validation['warnings'].append(
                                                f"Balance sheet equation imbalance in {year}: "
                                                f"Assets={assets[year]:,.0f} != L+E={liabilities[year] + equity[year]:,.0f}"
                                            )
                        else:
                            # No explicit Total Liabilities - will be calculated
                            validation['warnings'].append(
                                "Total Liabilities will be calculated as Total Assets - Total Equity"
                            )
                            
            except Exception as e:
                validation['warnings'].append(f"Could not validate balance sheet equation: {str(e)}")
        
        return validation
    
    def _get_mapped_source(self, mappings: Dict[str, str], target: str) -> Optional[str]:
        """Get source item for a target mapping"""
        for source, t in mappings.items():
            if t == target:
                return source
        return None
    
    def _calculate_quality_score(self, validation: Dict[str, Any]) -> float:
        """Calculate overall quality score for the mapping"""
        score = 100
        
        # Deduct for missing essential metrics (less harsh now)
        score -= len(validation['missing_essential']) * 8  # Reduced from 10
        
        # Deduct for errors
        score -= len(validation['errors']) * 15
        
        # Deduct for warnings (very light penalty)
        score -= len(validation['warnings']) * 2  # Reduced from 5
        
        # Check PN metrics validity
        pn_metrics = validation['pn_metrics_validity']
        valid_count = sum(1 for is_valid in pn_metrics.values() if is_valid)
        total_metrics = len(pn_metrics)
        
        if total_metrics > 0:
            pn_score = (valid_count / total_metrics) * 100
            # Weight PN validity at 40% of total score
            score = score * 0.6 + pn_score * 0.4
        
        return max(0, min(100, score))
    
    def suggest_improvements(self, validation_result: Dict[str, Any]) -> List[str]:
        """Suggest improvements based on validation results"""
        suggestions = []
        
        if validation_result['missing_essential']:
            # Filter out informational messages
            real_missing = [m for m in validation_result['missing_essential'] 
                           if 'will be calculated' not in str(m)]
            if real_missing:
                suggestions.append(
                    "🔍 Missing essential metrics: " + 
                    ", ".join(real_missing)
                )
        
        if not validation_result['pn_metrics_validity'].get('RNOA', False):
            suggestions.append(
                "📊 RNOA calculation needs attention - ensure you have mapped Operating Income "
                "(or EBIT/Operating Profit) and basic balance sheet items"
            )
        
        if not validation_result['pn_metrics_validity'].get('FLEV', False):
            suggestions.append(
                "💰 Financial Leverage calculation needs review - ensure you have mapped "
                "Total Equity and at least one debt item"
            )
        
        if not validation_result['pn_metrics_validity'].get('NBC', False):
            suggestions.append(
                "💸 Net Borrowing Cost calculation needs Interest Expense and debt items"
            )
        
        # Only show warnings if they're actionable
        actionable_warnings = [w for w in validation_result.get('warnings', []) 
                              if 'calculated' not in w and 'implicitly' not in w]
        if actionable_warnings:
            suggestions.append(
                "⚠️ Address warnings to improve analysis quality: " +
                ", ".join(actionable_warnings)
            )
        
        # Add positive feedback if mostly complete
        if validation_result['quality_score'] >= 80:
            suggestions.insert(0, "✅ Your mappings are nearly complete! Analysis can proceed.")
        
        return suggestions



class MappingTemplateManager:
    """Manage saved mapping templates for Penman-Nissim analysis"""
    
    def __init__(self, save_dir: str = "mapping_templates"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.templates_file = self.save_dir / "pn_mapping_templates.json"
        self.logger = LoggerFactory.get_logger('MappingTemplateManager')
        
    def save_template(self, name: str, mappings: Dict[str, str], 
                     description: str = "", company: str = "", 
                     metadata: Dict[str, Any] = None) -> bool:
        """Save a mapping template"""
        try:
            # Load existing templates
            templates = self._load_all_templates()
            
            # Create new template
            template = {
                'name': name,
                'mappings': mappings,
                'description': description,
                'company': company,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'version': '1.0',
                'metrics_count': len(mappings),
                'metadata': metadata or {}
            }
            
            # Add or update template
            templates[name] = template
            
            # Save to file
            with open(self.templates_file, 'w') as f:
                json.dump(templates, f, indent=2)
            
            self.logger.info(f"Saved mapping template: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving template: {e}")
            return False
    
    def load_template(self, name: str) -> Optional[Dict[str, str]]:
        """Load a specific mapping template"""
        try:
            templates = self._load_all_templates()
            if name in templates:
                self.logger.info(f"Loaded mapping template: {name}")
                return templates[name]['mappings']
            return None
        except Exception as e:
            self.logger.error(f"Error loading template: {e}")
            return None
    
    def get_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get all saved templates with metadata"""
        return self._load_all_templates()
    
    def delete_template(self, name: str) -> bool:
        """Delete a mapping template"""
        try:
            templates = self._load_all_templates()
            if name in templates:
                del templates[name]
                with open(self.templates_file, 'w') as f:
                    json.dump(templates, f, indent=2)
                self.logger.info(f"Deleted mapping template: {name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting template: {e}")
            return False
    
    def _load_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load all templates from file"""
        if self.templates_file.exists():
            try:
                with open(self.templates_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading templates file: {e}")
                return {}
        return {}
    
    def export_template(self, name: str, export_path: str) -> bool:
        """Export a template to a separate file"""
        try:
            template = self._load_all_templates().get(name)
            if template:
                with open(export_path, 'w') as f:
                    json.dump(template, f, indent=2)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error exporting template: {e}")
            return False
    
    def import_template(self, import_path: str) -> Optional[str]:
        """Import a template from a file"""
        try:
            with open(import_path, 'r') as f:
                template = json.load(f)
            
            # Validate template structure
            if 'name' in template and 'mappings' in template:
                # Update timestamps
                template['imported_at'] = datetime.now().isoformat()
                template['updated_at'] = datetime.now().isoformat()
                
                # Save the template
                templates = self._load_all_templates()
                templates[template['name']] = template
                
                with open(self.templates_file, 'w') as f:
                    json.dump(templates, f, indent=2)
                
                return template['name']
            return None
        except Exception as e:
            self.logger.error(f"Error importing template: {e}")
            return None
        
# --- 30. Main Application Class ---
class FinancialAnalyticsPlatform:
    """Main application with advanced architecture and all integrations"""

    def __init__(self):
        # Initialize session state for persistent data
        if 'initialized' not in st.session_state:
            self._initialize_session_state()

        # Initialize configuration with session state overrides
        self.config = Configuration(st.session_state.get('config_overrides', {}))

        # Initialize logger
        self.logger = LoggerFactory.get_logger('FinancialAnalyticsPlatform')

        # Initialize components only once
        if 'components' not in st.session_state or st.session_state.components is None:
            try:
                components = self._initialize_components()
                st.session_state.components = components
            except Exception as e:
                self.logger.error(f"Failed to initialize components: {e}")
                # Create empty components dictionary as fallback
                st.session_state.components = {}

        # Always set self.components from session state
        self.components = st.session_state.get('components', {})

        # Validate components
        if not self.components:
            self.logger.warning("No components initialized, creating defaults")
            self.components = self._create_default_components()

        # Initialize managers
        self.ui_factory = UIComponentFactory()
        self.sample_generator = SampleDataGenerator()
        self.export_manager = ExportManager(self.config)
        self.collaboration_manager = CollaborationManager()
        self.tutorial_system = TutorialSystem()
        self.nl_processor = NLQueryProcessor(self.config)
        self.ml_forecaster = MLForecaster(self.config)
        self.error_recovery = ErrorRecoveryManager()

        # Initialize compression handler
        self.compression_handler = CompressionHandler(self.logger)

    def _create_default_components(self) -> Dict[str, Component]:
        """Create default components as fallback"""
        try:
            return {
                'security': SecurityModule(self.config),
                'processor': DataProcessor(self.config),
                'analyzer': FinancialAnalysisEngine(self.config),
                'mapper': AIMapper(self.config),
            }
        except Exception as e:
            self.logger.error(f"Failed to create default components: {e}")
            return {}

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'compression_handler'):
            self.compression_handler.cleanup()

    @critical_method
    def _initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            # Core system state
            'initialized': True,
            'components': None,
            'config_overrides': {},

            # Data and analysis state
            'analysis_data': None,
            'analysis_hash': None,
            'data_source': None,
            'company_name': None,
            'filtered_data': None,
            
            # File handling state
            'uploaded_files': [],
            'simple_parse_mode': False,
            'processed_files_info': [],
            
            # Analysis mode and benchmarking
            'analysis_mode': 'Standalone Analysis',
            'benchmark_company': 'ITC Ltd',
            'benchmark_data': None,
            
            # Mapping state
            'metric_mappings': None,
            'ai_mapping_result': None,
            'show_manual_mapping': False,
            'mapping_confidence_threshold': 0.6,
            
            # Penman-Nissim analysis
            'pn_mappings': None,
            'pn_results': None,
            
            # ML and forecasting
            'ml_forecast_results': None,
            'forecast_periods': 3,
            'forecast_model_type': 'auto',
            
            # Kaggle API configuration
            'kaggle_api_url': '',
            'kaggle_api_key': '',
            'kaggle_api_enabled': False,
            'kaggle_api_status': 'unknown',
            'kaggle_status': {},
            'show_kaggle_config': False,
            'api_metrics_visible': False,
            'kaggle_connection_tested': False,
            
            # UI and display settings
            'number_format_value': 'Indian',
            'display_mode': 'LITE',
            'show_tutorial': True,
            'tutorial_step': 0,
            'tutorial_completed': False,
            'show_debug_info': False,
            
            # Collaboration and sharing
            'collaboration_session': None,
            'shared_analysis_token': None,
            'user_id': 'default_user',
            
            # Query and interaction history
            'query_history': [],
            'last_query_result': None,
            'interaction_count': 0,
            
            # Chart and visualization state
            'selected_chart_metrics': [],
            'chart_type': 'line',
            'show_trend_lines': True,
            'normalize_values': False,
            
            # Export and reporting
            'report_format': 'Excel',
            'include_charts': True,
            'report_sections': {
                'overview': True,
                'ratios': True,
                'trends': True,
                'forecasts': False,
                'penman_nissim': False,
                'industry': False,
                'raw_data': False
            },
            
            # Error tracking and recovery
            'last_error': None,
            'error_count': 0,
            'recovery_attempts': 0,
            
            # Performance and caching
            'cache_hit_count': 0,
            'analysis_count': 0,
            'performance_stats': {},
            
            # Advanced features
            'enable_ai_insights': True,
            'enable_anomaly_detection': True,
            'enable_pattern_recognition': True,
            'confidence_threshold': 0.6,
            
            # Data quality and validation
            'validation_results': None,
            'data_quality_score': None,
            'outlier_detection_results': {},
            
            # Industry comparison
            'selected_industry': 'Technology',
            'comparison_year': None,
            'industry_benchmarks': {},
            
            # Custom analysis settings
            'custom_ratios': {},
            'custom_metrics': {},
            'analysis_notes': '',
            
            # Session management
            'session_start_time': time.time(),
            'last_activity_time': time.time(),
            'session_id': hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8],
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Log initialization
        self.logger = LoggerFactory.get_logger('FinancialAnalyticsPlatform')
        self.logger.info(f"Initialized {len(defaults)} session state variables")

    @critical_method
    def _initialize_components(self) -> Dict[str, Component]:
        """Initialize all components with dependency injection"""
        components = {}

        try:
            # Create components one by one with error handling
            try:
                components['security'] = SecurityModule(self.config)
                components['security'].initialize()
                self.logger.info("Initialized component: security")
            except Exception as e:
                self.logger.error(f"Failed to initialize security: {e}")
            
            try:
                components['processor'] = DataProcessor(self.config)
                components['processor'].initialize()
                self.logger.info("Initialized component: processor")
            except Exception as e:
                self.logger.error(f"Failed to initialize processor: {e}")
            
            try:
                components['analyzer'] = FinancialAnalysisEngine(self.config)
                components['analyzer'].initialize()
                self.logger.info("Initialized component: analyzer")
            except Exception as e:
                self.logger.error(f"Failed to initialize analyzer: {e}")
            
            try:
                components['mapper'] = AIMapper(self.config)
                components['mapper'].initialize()
                self.logger.info("Initialized component: mapper")
            except Exception as e:
                self.logger.error(f"Failed to initialize mapper: {e}")
                # Mapper might fail due to AI dependencies, but app should still work
            
            return components
            
        except Exception as e:
            self.logger.error(f"Critical error in component initialization: {e}")
            # Return at least empty dict instead of None
            return {}

    def _cleanup_session_state(self, force_cleanup: bool = False):
        """Clean up unnecessary session state entries with improved handling"""
        try:
            # Core essential keys that should never be deleted
            core_essentials = {
                'analysis_data', 'analysis_hash', 'components', 'initialized',
                'config_overrides', 'data_source', 'number_format_value'
            }

            # Business logic keys that should be preserved unless force_cleanup
            business_keys = {
                'metric_mappings', 'pn_mappings', 'pn_results', 'company_name',
                'ml_forecast_results'
            }

            # Essential keys combination
            essential_keys = core_essentials | (set() if force_cleanup else business_keys)

            # Group keys by prefix for better organization
            key_groups = {
                'outliers': [],
                'standard_embedding': [],
                'template': [],
                'pn': [],
                'other': []
            }

            # Categorize all keys
            for key in list(st.session_state.keys()):
                if key in essential_keys:
                    continue
                
                if key.startswith('outliers_'):
                    key_groups['outliers'].append(key)
                elif key.startswith('standard_embedding_'):
                    key_groups['standard_embedding'].append(key)
                elif key.startswith('template_'):
                    key_groups['template'].append(key)
                elif key.startswith('pn_'):
                    key_groups['pn'].append(key)
                else:
                    key_groups['other'].append(key)

            # Cleanup rules
            cleanup_rules = {
                'outliers': lambda keys: keys,  # Remove all outlier keys
                'standard_embedding': lambda keys: [k for k in keys if not any(metric in k for metric in ['Revenue', 'Total Assets', 'Net Income'])],  # Keep essential embeddings
                'template': lambda keys: keys if force_cleanup else [],  # Remove only on force cleanup
                'pn': lambda keys: keys if force_cleanup else [],  # Remove only on force cleanup
                'other': lambda keys: [k for k in keys if not k.endswith(('_button', '_radio', '_checkbox'))]  # Remove UI state
            }

            # Perform cleanup
            cleaned_count = 0
            for group, keys in key_groups.items():
                keys_to_remove = cleanup_rules[group](keys)
                for key in keys_to_remove:
                    try:
                        del st.session_state[key]
                        cleaned_count += 1
                    except Exception as e:
                        self.logger.warning(f"Could not delete key {key}: {e}")

            # Log cleanup results
            self.logger.info(f"Cleaned up {cleaned_count} session state entries")
            
            # Perform garbage collection after significant cleanup
            if cleaned_count > 10:
                gc.collect()

        except Exception as e:
            self.logger.error(f"Error during session state cleanup: {e}")

    def _trigger_cleanup(self, trigger_type: str):
        """Trigger cleanup based on specific events"""
        cleanup_triggers = {
            'new_upload': True,
            'analysis_complete': False,
            'page_reload': False,
            'error_recovery': True,
        }
        force_cleanup = cleanup_triggers.get(trigger_type, False)
        self._cleanup_session_state(force_cleanup)

    def _manage_analysis_state(self, df: pd.DataFrame):
        """Manage analysis state and caching"""
        current_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

        if ('analysis_hash' not in st.session_state or 
            st.session_state.get('analysis_hash') != current_hash):
            
            keys_to_clear = [
                'metric_mappings', 'pn_mappings', 'pn_results',
                'ml_forecast_results', 'ai_mapping_result'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.session_state['analysis_hash'] = current_hash

    # State helper methods
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from session state with safe fallback"""
        try:
            return SimpleState.get(key, default)
        except Exception as e:
            self.logger.warning(f"Error accessing session state key '{key}': {e}")
            return default

    def set_state(self, key: str, value: Any) -> bool:
        """Set value in session state with error handling"""
        try:
            SimpleState.set(key, value)
            return True
        except Exception as e:
            self.logger.error(f"Error setting session state key '{key}': {e}")
            return False

    def ensure_state_key(self, key: str, default_value: Any = None):
        """Ensure a session state key exists with a default value"""
        if key not in st.session_state:
            st.session_state[key] = default_value
            self.logger.info(f"Initialized missing session state key: {key}")

    def get_state_safe(self, key: str, default: Any = None, initialize: bool = True) -> Any:
        """Ultra-safe session state access with auto-initialization"""
        try:
            if key not in st.session_state:
                if initialize:
                    st.session_state[key] = default
                    self.logger.info(f"Auto-initialized session state key: {key} = {default}")
                return default
            return st.session_state[key]
        except Exception as e:
            self.logger.error(f"Critical error accessing session state key '{key}': {e}")
            return default

    def _clear_all_caches(self):
        """Clear all caches"""
        if 'analyzer' in self.components:
            self.components['analyzer'].cache.clear()
        if 'mapper' in self.components:
            self.components['mapper'].embeddings_cache.clear()

        performance_monitor.clear_metrics()

        gc.collect()

    def _reset_configuration(self):
        """Reset configuration to defaults"""
        self.set_state('config_overrides', {})
        self.config = Configuration()
        st.success("Configuration reset to defaults!")

    def _export_logs(self):
        """
        Prepares log data for download and stores it in session state.
        This method does NOT render the button itself.
        """
        try:
            log_dir = Path("logs")
            if not log_dir.exists() or not any(log_dir.iterdir()):
                st.warning("No log files found to export.")
                self.set_state('log_download_data', None) # Clear any old data
                return
    
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for log_file in log_dir.glob("*.log"):
                    zip_file.write(log_file, arcname=log_file.name)
            
            zip_buffer.seek(0)
            # Store the prepared data in session state
            self.set_state('log_download_data', zip_buffer.getvalue())
            self.logger.info("Log data prepared for download.")
    
        except Exception as e:
            st.error(f"Failed to prepare logs for export: {e}")
            self.logger.error(f"Log export preparation error: {e}", exc_info=True)
            self.set_state('log_download_data', None)

    
    @error_boundary()
    @critical_method
    def run(self):
        """Main application entry point"""
        try:
            # Check if app is already running to prevent duplicate renders
            # Use hasattr to safely check for session state keys
            if hasattr(st.session_state, 'app_running') and st.session_state.app_running:
                return
            
            # Set the running flag
            st.session_state.app_running = True
            
            if not hasattr(self, 'components') or self.components is None:
                self.logger.warning("Components not initialized, attempting recovery")
                self._auto_recovery_attempt()
            
            self._apply_custom_css()
            
            if hasattr(self, 'tutorial_system') and self.tutorial_system:
                self.tutorial_system.render()
            
            self._render_header()
            
            self._render_sidebar()
            
            self._render_main_content()
            
            if self.config.get('app.debug', False):
                self._render_debug_footer()
            
            # Clear the running flag at the end
            st.session_state.app_running = False
            
        except Exception as e:
            # Clear the running flag on error
            st.session_state.app_running = False
                
            self.logger.error(f"Application error: {e}")
            st.error("An unexpected error occurred. Please refresh the page.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Retry"):
                    st.rerun()
            
            with col2:
                if st.button("🔧 Auto Recovery"):
                    if self._auto_recovery_attempt():
                        st.success("Recovery successful!")
                        st.rerun()
                    else:
                        st.error("Recovery failed. Please refresh manually.")
            
            with col3:
                if st.button("🗑️ Clear All & Restart"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
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

        .skeleton {
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
            height: 20px;
            margin: 10px 0;
            border-radius: 4px;
        }

        @keyframes loading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        .collaboration-indicator {
            position: fixed;
            top: 10px;
            right: 10px;
            background: #4CAF50;
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            z-index: 1000;
        }

        .kaggle-status {
            position: fixed;
            top: 60px;
            right: 20px;
            background: white;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 10px 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
        }

        .kaggle-status.error {
            border-color: #f44336;
        }

        .kaggle-metric {
            display: inline-block;
            margin: 0 10px;
            font-size: 14px;
        }

        .api-health {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }

        .api-health.healthy {
            background-color: #4CAF50;
        }

        .api-health.unhealthy {
            background-color: #f44336;
        }

        .api-metrics-panel {
            position: fixed;
            top: 120px;
            right: 20px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            z-index: 999;
            max-width: 300px;
        }

        .progress-tracker {
            background: #f5f5f5;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }

        .progress-bar {
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            height: 8px;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        </style>
        """, unsafe_allow_html=True)

    def _render_header(self):
        """Render application header with enhanced status indicators"""
        st.markdown(
            '<h1 class="main-header">💹 Elite Financial Analytics Platform v5.1</h1>',
            unsafe_allow_html=True
        )

        if self.get_state('collaboration_session'):
            session_id = self.get_state('collaboration_session')
            if hasattr(self, 'collaboration_manager') and self.collaboration_manager:
                activity = self.collaboration_manager.get_session_activity(session_id)
                if activity:
                    st.markdown(
                        f'<div class="collaboration-indicator">👥 {len(activity["active_users"])} users online</div>',
                        unsafe_allow_html=True
                    )

        if self.config.get('ui.show_kaggle_status', True) and self.get_state('kaggle_api_enabled'):
            self._render_kaggle_status_badge()

        if self.config.get('ui.show_api_metrics', True) and self.get_state('api_metrics_visible', False):
            self._render_api_metrics_panel()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            try:
                if hasattr(self, 'components') and self.components:
                    components_status = sum(
                        1 for c in self.components.values() 
                        if hasattr(c, '_initialized') and c._initialized
                    )
                    total_components = len(self.components)
                else:
                    components_status = 0
                    total_components = 0
            except Exception as e:
                self.logger.warning(f"Error counting components: {e}")
                components_status = 0
                total_components = 0
            
            self.ui_factory.create_metric_card(
                "Components", 
                f"{components_status}/{total_components}",
                help="Active system components"
            )

        with col2:
            try:
                mode = self.config.get('app.display_mode', Configuration.DisplayMode.LITE)
                mode_name = mode.name if hasattr(mode, 'name') else str(mode)
            except Exception as e:
                self.logger.warning(f"Error getting display mode: {e}")
                mode_name = "LITE"
            
            self.ui_factory.create_metric_card(
                "Mode", 
                mode_name,
                help="Current operating mode"
            )

        with col3:
            try:
                if (hasattr(self, 'components') and 
                    self.components and 
                    'mapper' in self.components and 
                    hasattr(self.components['mapper'], 'embeddings_cache')):
                    
                    cache_stats = self.components['mapper'].embeddings_cache.get_stats()
                    hit_rate = cache_stats.get('hit_rate', 0)
                else:
                    hit_rate = 0
            except Exception as e:
                self.logger.warning(f"Error getting cache stats: {e}")
                hit_rate = 0
            
            self.ui_factory.create_metric_card(
                "Cache Hit Rate", 
                f"{hit_rate:.1f}%",
                help="AI cache performance"
            )

        with col4:
            try:
                version = self.config.get('app.version', 'Unknown')
            except Exception as e:
                self.logger.warning(f"Error getting version: {e}")
                version = "5.1.0"
            
            self.ui_factory.create_metric_card(
                "Version", 
                version,
                help="Platform version"
            )

        if self.config.get('app.debug', False):
            try:
                health = self._perform_health_check() if hasattr(self, '_perform_health_check') else None
                if health and 'overall' in health:
                    health_color = "🟢" if health['overall'] else "🔴"
                    st.markdown(
                        f'<div style="text-align: center; margin-top: 10px;">'
                        f'{health_color} System Health: {"Good" if health["overall"] else "Issues Detected"}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                self.logger.warning(f"Error performing health check: {e}")
        
    def _render_kaggle_status_badge(self):
        """Render floating Kaggle API status badge with enhanced metrics"""
        try:
            if 'mapper' in self.components and hasattr(self.components['mapper'], 'get_api_status'):
                status = self.components['mapper'].get_api_status()

                if status['kaggle_available']:
                    info = status.get('api_info', {})
                    stats = status.get('api_stats', {})
                    
                    gpu_name = info.get('gpu_name', info.get('system', {}).get('gpu_name', 'GPU'))
                    model = info.get('model', info.get('system', {}).get('model', 'Unknown'))
                    version = info.get('version', 'Unknown')
                    
                    status_html = f"""
                    <div class="kaggle-status">
                        <span class="api-health healthy"></span>
                        <strong>Kaggle GPU Active</strong>
                        <span class="kaggle-metric">🖥️ {gpu_name}</span>
                        <span class="kaggle-metric">🤖 {model}</span>
                        <span class="kaggle-metric">📊 v{version}</span>
                    </div>
                    """
                else:
                    status_html = """
                    <div class="kaggle-status error">
                        <span class="api-health unhealthy"></span>
                        <strong>Kaggle GPU Offline</strong>
                        <span class="kaggle-metric">Using local processing</span>
                    </div>
                    """
                
                st.markdown(status_html, unsafe_allow_html=True)
        except Exception as e:
            self.logger.warning(f"Error rendering Kaggle status badge: {e}")

    def _render_api_metrics_panel(self):
        """Render detailed API metrics panel"""
        try:
            if hasattr(self, 'components') and 'mapper' in self.components:
                api_summary = performance_monitor.get_api_summary()

                if api_summary:
                    metrics_html = """
                    <div class="api-metrics-panel">
                        <h4 style="margin-top: 0;">API Performance Metrics</h4>
                    """
                    
                    for endpoint, metrics in api_summary.items():
                        metrics_html += f"""
                        <div style="margin-bottom: 10px;">
                            <strong>{endpoint}</strong><br>
                            <small>
                            Requests: {metrics['total_requests']} | 
                            Success: {metrics['success_rate']:.1%} | 
                            Avg: {metrics['avg_response_time']:.2f}s | 
                            P95: {metrics['p95_response_time']:.2f}s
                            </small>
                        </div>
                        """
                    
                    metrics_html += """
                    </div>
                    """
                    
                    st.markdown(metrics_html, unsafe_allow_html=True)
        except Exception as e:
            self.logger.warning(f"Error rendering API metrics panel: {e}")
            
    @safe_state_access
    def _render_sidebar(self):
        """Render sidebar with enhanced Kaggle configuration"""
        st.sidebar.title("⚙️ Configuration")

        st.sidebar.header("📊 Analysis Mode")

        analysis_mode = st.sidebar.radio(
            "Select Analysis Mode",
            ["Standalone Analysis", "Benchmark Comparison"],
            index=0,
            help="Standalone analyzes only your data. Benchmark compares with industry standards."
        )

        self.set_state('analysis_mode', analysis_mode)

        if analysis_mode == "Benchmark Comparison":
            benchmark_company = st.sidebar.selectbox(
                "Benchmark Company",
                ["ITC Ltd", "Hindustan Unilever", "Nestle India"],
                index=0,
                help="Company to compare with"
            )
            self.set_state('benchmark_company', benchmark_company)
            
            if st.sidebar.button("Load Benchmark Data", type="primary"):
                self._load_benchmark_data(benchmark_company)
                st.sidebar.success(f"✅ Benchmark data for {benchmark_company} loaded")

        st.sidebar.header("🖥️ Kaggle GPU Configuration")

        kaggle_enabled = st.sidebar.checkbox(
            "Enable Kaggle GPU Acceleration",
            value=self.get_state('kaggle_api_enabled', False),
            help="Use remote GPU for faster processing"
        )

        if kaggle_enabled:
            with st.sidebar.expander("Kaggle API Settings", expanded=True):
                api_url = st.text_input(
                    "Ngrok URL",
                    value=self.get_state('kaggle_api_url', ''),
                    placeholder="https://xxxx.ngrok-free.app",
                    help="Paste the ngrok URL from your Kaggle notebook"
                )
                
                api_key = st.text_input(
                    "API Key (Optional)",
                    type="password",
                    help="Optional API key for authentication"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    timeout = st.number_input(
                        "Timeout (seconds)",
                        min_value=10,
                        max_value=300,
                        value=30,
                        help="Request timeout"
                    )
                
                with col2:
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=1,
                        max_value=100,
                        value=50,
                        help="Optimal batch size for GPU"
                    )
                
                with st.expander("Circuit Breaker Settings"):
                    cb_threshold = st.number_input(
                        "Failure Threshold",
                        min_value=1,
                        max_value=20,
                        value=5,
                        help="Failures before circuit opens"
                    )
                    
                    cb_timeout = st.number_input(
                        "Recovery Timeout (seconds)",
                        min_value=30,
                        max_value=600,
                        value=300,
                        help="Time before retry after circuit opens"
                    )
                
                if st.button("🔌 Test Connection", type="primary"):
                    if api_url:
                        self.config.set('ai.kaggle_api_url', api_url)
                        self.config.set('ai.kaggle_api_key', api_key)
                        self.config.set('ai.kaggle_api_timeout', timeout)
                        self.config.set('ai.kaggle_batch_size', batch_size)
                        self.config.set('ai.kaggle_circuit_breaker_threshold', cb_threshold)
                        self.config.set('ai.kaggle_circuit_breaker_timeout', cb_timeout)
                        self.config.set('ai.use_kaggle_api', True)
                        
                        with st.spinner("Testing Kaggle connection..."):
                            try:
                                self.components['mapper'].cleanup()
                                self.components['mapper'] = AIMapper(self.config)
                                self.components['mapper'].initialize()
                                
                                status = self.components['mapper'].get_api_status()
                                
                                if status['kaggle_available']:
                                    st.success("✅ Successfully connected to Kaggle GPU!")
                                    
                                    if status['api_info']:
                                        info = status['api_info']
                                        system_info = info.get('system', {})
                                        gpu_name = system_info.get('gpu_name', 'Unknown')
                                        device = system_info.get('device', 'Unknown')
                                        model = system_info.get('model', 'Unknown')
                                        
                                        st.info(f"""
                                        **GPU Info:**
                                        - Model: {model}
                                        - GPU: {gpu_name}
                                        - Device: {device}
                                        - Version: {info.get('version', 'Unknown')}
                                        """)
                                    
                                    self.set_state('kaggle_api_url', api_url)
                                    self.set_state('kaggle_api_enabled', True)
                                    self.set_state('kaggle_status', status)
                                    self.set_state('kaggle_api_status', 'online')
                                    
                                else:
                                    st.error("❌ Connection failed. Please check your URL and try again.")
                                    self.set_state('kaggle_api_status', 'offline')
                            except Exception as e:
                                st.error(f"❌ Connection error: {str(e)}")
                                self.set_state('kaggle_api_status', 'error')
                    else:
                        st.warning("Please enter a valid ngrok URL")
                
                if st.button("🧪 Test Embed Endpoint", type="secondary"):
                    if api_url:
                        with st.spinner("Testing embed endpoint..."):
                            try:
                                import requests
                                import urllib3
                                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                                
                                test_data = {'texts': ['test embedding']}
                                
                                response = requests.post(
                                    f"{api_url.rstrip('/')}/embed",
                                    json=test_data,
                                    headers={
                                        'Content-Type': 'application/json', 
                                        'ngrok-skip-browser-warning': 'true'
                                    },
                                    verify=False,
                                    timeout=10
                                )
                                
                                st.write(f"**Status Code:** {response.status_code}")
                                
                                if response.status_code == 200:
                                    st.success("✅ Embed endpoint is working!")
                                    try:
                                        result = response.json()
                                        st.write("**Response format:**")
                                        
                                        if isinstance(result, dict):
                                            st.write(f"- Type: Dictionary with keys: {list(result.keys())}")
                                            
                                            for key in ['embeddings', 'data', 'vectors', 'output']:
                                                if key in result:
                                                    st.write(f"- Found embeddings in key: '{key}'")
                                                    embed_data = result[key]
                                                    if isinstance(embed_data, list) and len(embed_data) > 0:
                                                        st.write(f"  - Number of embeddings: {len(embed_data)}")
                                                        st.write(f"  - Embedding dimension: {len(embed_data[0]) if isinstance(embed_data[0], list) else 'N/A'}")
                                                        st.write(f"  - First 5 values: {embed_data[0][:5] if isinstance(embed_data[0], list) else str(embed_data[0])[:50]}")
                                                    break
                                        elif isinstance(result, list):
                                            st.write(f"- Type: List with {len(result)} items")
                                            if len(result) > 0:
                                                st.write(f"- First item type: {type(result[0])}")
                                                st.write(f"- Dimension: {len(result[0]) if isinstance(result[0], list) else 'N/A'}")
                                        
                                        with st.expander("View full response"):
                                            st.json(result)
                                            
                                    except Exception as e:
                                        st.warning(f"Response is not JSON: {response.text[:200]}")
                                else:
                                    st.error(f"❌ Embed endpoint failed")
                                    st.text(f"Response: {response.text[:500]}")
                                    
                            except requests.exceptions.Timeout:
                                st.error("❌ Request timed out. The endpoint might be slow or unresponsive.")
                            except requests.exceptions.ConnectionError:
                                st.error("❌ Connection error. Check if the URL is correct and accessible.")
                            except Exception as e:
                                st.error(f"❌ Test failed: {str(e)}")
                                import traceback
                                with st.expander("Show error details"):
                                    st.code(traceback.format_exc())
                    else:
                        st.warning("Please enter a ngrok URL above")
                
                if st.button("🔍 Run Full Diagnostics", type="secondary"):
                    if api_url:
                        with st.expander("Diagnostic Results", expanded=True):
                            st.write(f"**Testing URL:** `{api_url}`")
                            
                            import requests
                            import urllib3
                            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                            
                            endpoints_to_test = [
                                ('/', 'Root'),
                                ('/health', 'Health'),
                                ('/embed', 'Embed (GET)'),
                                ('/api/health', 'API Health'),
                                ('/status', 'Status'),
                                ('/docs', 'Documentation'),
                                ('/version', 'Version')
                            ]
                            
                            st.write("\n**Endpoint Tests:**")
                            
                            for endpoint, name in endpoints_to_test:
                                try:
                                    url = f"{api_url.rstrip('/')}{endpoint}"
                                    response = requests.get(
                                        url, 
                                        timeout=5, 
                                        verify=False,
                                        headers={'ngrok-skip-browser-warning': 'true'}
                                    )
                                    
                                    if response.status_code == 200:
                                        st.success(f"✅ {name} ({endpoint}): {response.status_code}")
                                    elif response.status_code == 404:
                                        st.warning(f"⚠️ {name} ({endpoint}): Not found")
                                    else:
                                        st.error(f"❌ {name} ({endpoint}): {response.status_code}")
                                        
                                except Exception as e:
                                    st.error(f"❌ {name} ({endpoint}): {str(e)}")
                            
                            st.write("\n**Embed Endpoint Tests (POST):**")
                            
                            test_payloads = [
                                {'texts': ['test']},
                                {'text': 'test'},
                                {'inputs': ['test']},
                                {'data': ['test']}
                            ]
                            
                            for payload in test_payloads:
                                try:
                                    response = requests.post(
                                        f"{api_url.rstrip('/')}/embed",
                                        json=payload,
                                        headers={'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true'},
                                        verify=False,
                                        timeout=10
                                    )
                                    
                                    if response.status_code == 200:
                                        st.success(f"✅ Payload format `{list(payload.keys())[0]}`: Success")
                                        break
                                    else:
                                        st.warning(f"⚠️ Payload format `{list(payload.keys())[0]}`: {response.status_code}")
                                        
                                except Exception as e:
                                    st.error(f"❌ Payload format `{list(payload.keys())[0]}`: {str(e)}")
                    else:
                        st.warning("Please enter a ngrok URL above")
                
                with st.expander("📚 Setup Guide"):
                    st.markdown("""
                    **How to connect to Kaggle GPU:**
                    
                    1. **Run the Kaggle notebook** with the API server code
                    2. **Copy the ngrok URL** shown in the output
                    3. **Paste it above** and click Test Connection
                    
                    **Benefits:**
                    - 🚀 10-100x faster embedding generation
                    - 💾 Larger model support (GPU memory)
                    - 🔋 Reduced local CPU/memory usage
                    - 📊 Better accuracy with larger models
                    
                    **Advanced Features:**
                    - Circuit breaker for resilience
                    - Request batching and coalescing
                    - Response caching
                    - Connection pooling
                    
                    **Troubleshooting:**
                    - Ensure the Kaggle notebook is running
                    - Check that ngrok is not expired (8 hour limit)
                    - Verify the URL includes https://
                    """)
        else:
            if self.get_state('kaggle_api_enabled'):
                self.config.set('ai.use_kaggle_api', False)
                self.set_state('kaggle_api_enabled', False)
                self.set_state('kaggle_api_status', 'disabled')
            
            st.sidebar.info("Enable to use GPU-accelerated processing via Kaggle")

        if 'mapper' in self.components:
            status = self.components['mapper'].get_api_status()
            
            if status['kaggle_configured'] or status['local_model_available']:
                st.sidebar.subheader("🎯 Processing Status")
                
                processing_methods = []
                if status['kaggle_available']:
                    processing_methods.append("✅ Kaggle GPU")
                if status['local_model_available']:
                    processing_methods.append("✅ Local Model")
                if not processing_methods:
                    processing_methods.append("✅ Fuzzy Matching")
                
                for method in processing_methods:
                    st.sidebar.text(method)
                
                if st.sidebar.checkbox("Show API Metrics", value=self.get_state('api_metrics_visible', False)):
                    self.set_state('api_metrics_visible', True)
                else:
                    self.set_state('api_metrics_visible', False)
                
                st.sidebar.metric("Cache Size", status['cache_size'])
                st.sidebar.metric("Buffer Size", status.get('buffer_size', 0))

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

        st.sidebar.header("⚙️ Settings")

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
            self.set_state('config_overrides', {'app': {'display_mode': Configuration.DisplayMode[selected_mode]}})

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

        if self.config.get('app.enable_collaboration', True):
            st.sidebar.subheader("👥 Collaboration")
            
            if not self.get_state('collaboration_session'):
                if st.sidebar.button("Start Collaborative Session"):
                    session_id = self.collaboration_manager.create_session(
                        self.get_state('analysis_data_id', 'default'),
                        'current_user'
                    )
                    self.set_state('collaboration_session', session_id)
                    st.sidebar.success(f"Session created: {session_id}")
            else:
                session_id = self.get_state('collaboration_session')
                st.sidebar.info(f"Session: {session_id}")
                
                if st.sidebar.button("Leave Session"):
                    self.set_state('collaboration_session', None)
                    st.sidebar.info("Left collaborative session")

        st.sidebar.subheader("🔢 Number Format")

        current_format = self.get_state('number_format_value', 'Indian')

        format_option = st.sidebar.radio(
            "Display Format",
            ["Indian (₹ Lakhs/Crores)", "International ($ Millions)"],
            index=0 if current_format == 'Indian' else 1,
            key="number_format_radio"
        )

        self.set_state('number_format_value', 
                      'Indian' if "Indian" in format_option else 'International')

        with st.sidebar.expander("🔧 Advanced Options"):
            debug_mode = st.sidebar.checkbox(
                "Debug Mode",
                value=self.config.get('app.debug', False),
                help="Show detailed error information"
            )
            self.config.set('app.debug', debug_mode)
            
          # In _render_sidebar (or a debug section)
            if st.sidebar.button("🗑️ Clear Cache & Reset App", key="clear_cache_reset"):
                # Step 1: Clear all internal caches
                if hasattr(self, '_clear_all_caches'):
                    self._clear_all_caches()  # Your existing cache clear method
                
                # Step 2: Reset key session state variables to initial values
                reset_keys = [
                    'analysis_data', 'analysis_hash', 'metric_mappings', 'pn_mappings', 
                    'pn_results', 'ml_forecast_results', 'ai_mapping_result', 
                    'benchmark_data', 'validation_results', 'data_quality_score',
                    'outlier_detection_results', 'industry_benchmarks', 'custom_ratios',
                    'custom_metrics', 'pn_active_template', 'temp_pn_mappings', 'pn_unmapped'
                    # Add any other analysis-related keys here
                ]
                
                for key in reset_keys:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Step 3: Reset UI-related states
                st.session_state['show_manual_mapping'] = False
                st.session_state['show_tutorial'] = True  # Optional: Restart tutorial
                st.session_state['tutorial_step'] = 0
                st.session_state['data_source'] = None
                
                # Step 4: Clear Streamlit caches (for good measure)
                st.cache_data.clear()
                st.cache_resource.clear()
                
                # Force full rerun
                st.sidebar.success("✅ App fully reset! Ready for new analysis.")
                st.rerun()
            
            if st.sidebar.button("Reset Configuration"):
                self._reset_configuration()
            
             # --- CORRECTED LOG DOWNLOAD LOGIC ---
            # 1. The preparation button
            if st.sidebar.button("Prepare Logs for Download"):
                self._export_logs() # This now prepares data and saves to session state
    
            # 2. The download button, which appears only after data is prepared
            log_data = self.get_state('log_download_data')
            if log_data:
                st.sidebar.download_button(
                    label="📥 Download Log File",
                    data=log_data,
                    file_name=f"app_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    key="download_logs_final_btn",
                    # On click, Streamlit handles the download and we can clear the state
                    on_click=lambda: self.set_state('log_download_data', None)
                )
                st.sidebar.info("Click the button above to download the prepared log file.")
            
            
            
            if debug_mode:
                perf_summary = performance_monitor.get_performance_summary()
                if perf_summary:
                    st.sidebar.write("**Performance Summary:**")
                    for op, stats in list(perf_summary.items())[:5]:
                        st.sidebar.text(f"{op}: {stats['avg_duration']:.3f}s")
                        
    @safe_state_access
    def _render_file_upload(self):
        """Render file upload interface with safe state handling"""
        self.ensure_state_key('uploaded_files', [])
        self.ensure_state_key('simple_parse_mode', False)

        allowed_types = self.config.get('app.allowed_file_types', [])
        max_size = self.config.get('security.max_upload_size_mb', 50)

        temp_files = st.sidebar.file_uploader(
            f"Upload Financial Statements (Max {max_size}MB each)",
            type=allowed_types,
            accept_multiple_files=True,
            key="file_uploader",
            help="You can upload compressed files (.zip, .7z) containing multiple financial statements"
        )

        if temp_files:
            self.set_state('uploaded_files', temp_files)
            
            regular_files = [f for f in temp_files if not f.name.lower().endswith(('.zip', '.7z'))]
            compressed_files = [f for f in temp_files if f.name.lower().endswith(('.zip', '.7z'))]
            
            if compressed_files:
                st.sidebar.info(f"📦 {len(compressed_files)} compressed file(s) uploaded")
            if regular_files:
                st.sidebar.info(f"📄 {len(regular_files)} regular file(s) uploaded")

        uploaded_files = self.get_state_safe('uploaded_files', [], initialize=True)

        if uploaded_files:
            current_simple_mode = self.get_state_safe('simple_parse_mode', False, initialize=True)
            
            new_simple_mode = st.sidebar.checkbox(
                "Use simple parsing mode", 
                value=current_simple_mode,
                help="Try this if normal parsing fails",
                key="simple_parse_mode_checkbox"
            )
            
            self.set_state('simple_parse_mode', new_simple_mode)
            
            has_7z = any(f.name.lower().endswith('.7z') for f in uploaded_files)
            if has_7z and not SEVEN_ZIP_AVAILABLE:
                st.sidebar.warning("⚠️ 7z files detected but py7zr not installed")
                st.sidebar.code("pip install py7zr")
            
            all_valid = True
            for file in uploaded_files:
                if not file.name.lower().endswith(('.zip', '.7z')):
                    result = self.components['security'].validate_file_upload(file)
                    if not result.is_valid:
                        st.sidebar.error(f"❌ {file.name}: {result.errors[0]}")
                        all_valid = False
            
            if all_valid and st.sidebar.button("Process Files", type="primary"):
                self._process_uploaded_files(uploaded_files)

        with st.sidebar.expander("📋 File Format Guide"):
            st.info("""
            **Supported Financial Data Formats:**
            
            1. **Capitaline Exports**: Both .xls (HTML) and true Excel formats
            2. **Moneycontrol/BSE/NSE**: HTML exports with .xls extension
            3. **Standard CSV/Excel**: With metrics in rows and years in columns
            4. **Compressed Files**: 
               - ZIP files (.zip) containing multiple statements
               - 7Z files (.7z) for maximum compression
            
            **💡 Pro Tips**: 
            - Compress multiple files into a single ZIP/7Z for faster uploads
            - 7Z typically provides 50-70% better compression than ZIP
            - You can mix different file types in a single compressed file
            """)

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
        
    @safe_state_access
    def _process_uploaded_files(self, uploaded_files: List[UploadedFile]):
        """Process uploaded files including compressed files with progress tracking"""
        try:
            self._trigger_cleanup('new_upload')

            self._cleanup_session_state()
            
            self.set_state('metric_mappings', None)
            self.set_state('pn_mappings', None)
            self.set_state('pn_results', None)
            self.set_state('ml_forecast_results', None)
            self.set_state('ai_mapping_result', None)
            
            all_dataframes = []
            file_info = []
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            total_files = len(uploaded_files)
            
            for i, file in enumerate(uploaded_files):
                progress_text.text(f"Processing {file.name}...")
                progress_bar.progress((i + 1) / total_files)
                
                try:
                    if file.name.lower().endswith(('.zip', '.7z')):
                        extracted_files = self.compression_handler.extract_compressed_file(file)
                        
                        for extracted_name, extracted_content in extracted_files:
                            temp_file = io.BytesIO(extracted_content)
                            temp_file.name = extracted_name
                            
                            df = self._parse_single_file(temp_file)
                            
                            # <<<--- INTEGRATION POINT 1: Validate and repair parsed data --->>>
                            if df is not None:
                                df = self._validate_and_repair_parsed_data(df, temp_file)
                            
                            if df is not None and not df.empty:
                                df_before = df.copy()
                                df = self._clean_dataframe(df)
                                self._verify_cash_flow_preservation(df_before, df)
                                
                                self._inspect_dataframe(df, extracted_name)
                                all_dataframes.append(df)
                                file_info.append({
                                    'name': extracted_name,
                                    'source': f"{file.name} (compressed)",
                                    'shape': df.shape
                                })
                    else:
                        df = self._parse_single_file(file)
                        
                        # <<<--- INTEGRATION POINT 2: Validate and repair parsed data --->>>
                        if df is not None:
                            df = self._validate_and_repair_parsed_data(df, file)
                        
                        if df is not None and not df.empty:
                            df_before = df.copy()
                            df = self._clean_dataframe(df)
                            self._verify_cash_flow_preservation(df_before, df)

                            self._inspect_dataframe(df, file.name)
                            all_dataframes.append(df)
                            file_info.append({
                                'name': file.name,
                                'source': 'direct upload',
                                'shape': df.shape
                            })
                except Exception as e:
                    self.logger.error(f"Error processing {file.name}: {e}")
                    st.error(f"Error processing {file.name}: {str(e)}")
                    continue
            
            progress_text.empty()
            progress_bar.empty()
            
            if all_dataframes:
                final_df = all_dataframes[0] if len(all_dataframes) == 1 else self._merge_dataframes(all_dataframes)
                
                processed_df, validation_result = self.components['processor'].process(final_df, "uploaded_data")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_rows = len(processed_df)
                    total_cols = len(processed_df.columns)
                    st.metric("Data Points", f"{total_rows}×{total_cols}")
                
                with col2:
                    numeric_df = processed_df.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        missing_pct = (numeric_df.isnull().sum().sum() / (numeric_df.shape[0] * numeric_df.shape[1])) * 100
                    else:
                        missing_pct = 0.0
                    st.metric("Missing Data", f"{missing_pct:.1f}%")
                
                with col3:
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    if len(processed_df.columns) > 0:
                        numeric_pct = (len(numeric_cols) / len(processed_df.columns)) * 100
                    else:
                        numeric_pct = 0.0
                    st.metric("Numeric Columns", f"{numeric_pct:.1f}%")
                
                self._manage_analysis_state(processed_df)
                self.set_state('analysis_data', processed_df)
                self.set_state('data_source', 'uploaded_files')
                
                st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)")
                
                with st.expander("📊 Data Quality Report", expanded=True):
                    st.write("**Data Structure:**")
                    st.write(f"- Rows: {total_rows}")
                    st.write(f"- Columns: {total_cols}")
                    st.write(f"- Missing Data: {missing_pct:.1f}%")
                    
                    if validation_result.warnings:
                        st.write("\n**Warnings:**")
                        for warning in validation_result.warnings[:3]:
                            st.warning(warning)
                    
                    if validation_result.corrections:
                        st.write("\n**Auto-corrections Applied:**")
                        for correction in validation_result.corrections[:5]:
                            st.info(correction)
                        if len(validation_result.corrections) > 5:
                            st.write(f"... and {len(validation_result.corrections)-5} more corrections")
                    
                    if len(file_info) > 1:
                        st.write("\n**Processed Files:**")
                        info_df = pd.DataFrame(file_info)
                        st.dataframe(info_df, use_container_width=True)
                
            else:
                st.error("No valid financial data found in uploaded files")
                
        except Exception as e:
            self._trigger_cleanup('error_recovery')
            st.error(f"Error processing files: {str(e)}")
        finally:
            self.compression_handler.cleanup()
        
    def _parse_single_file(self, file) -> Optional[pd.DataFrame]:
        """
        Enhanced parser that correctly handles multi-statement financial files including Google Sheets XLSX
        """
        try:
            file_ext = Path(file.name).suffix.lower()
            self.logger.info(f"Attempting to parse file: {file.name} with extension: {file_ext}")
    
            # Read file content
            file.seek(0)
            
            # For HTML-based files (including .xls from Capitaline/Moneycontrol)
            if file_ext in ['.html', '.htm', '.xls']:
                # Try to read all tables
                try:
                    tables = pd.read_html(file, match='.+', header=None)
                    self.logger.info(f"Found {len(tables)} tables in {file.name}")
                    
                    if not tables:
                        raise ValueError("No tables found")
                    
                    # Process each table to identify statement type
                    processed_dfs = []
                    
                    for i, table in enumerate(tables):
                        self.logger.debug(f"Processing table {i+1} with shape {table.shape}")
                        
                        # Skip very small tables (likely headers/footers)
                        if table.shape[0] < 5 or table.shape[1] < 2:
                            continue
                        
                        # Try to identify the statement type
                        statement_type = self._identify_statement_type_from_table(table)
                        
                        if statement_type:
                            # Process the table based on its type
                            processed_df = self._process_financial_table(table, statement_type)
                            if processed_df is not None and not processed_df.empty:
                                processed_dfs.append(processed_df)
                                self.logger.info(f"Successfully processed {statement_type} with {len(processed_df)} rows")
                    
                    # Combine all processed tables
                    if processed_dfs:
                        combined_df = pd.concat(processed_dfs, axis=0)
                        self.logger.info(f"Combined {len(processed_dfs)} tables into final DataFrame with {len(combined_df)} rows")
                        return combined_df
                    else:
                        # Fallback to original method if table identification fails
                        return self._parse_single_table_fallback(file)
                        
                except Exception as e:
                    self.logger.warning(f"Multi-table parsing failed: {e}, trying fallback method")
                    return self._parse_single_table_fallback(file)
            
            # For CSV files
            elif file_ext == '.csv':
                return self._parse_csv_file(file)
            
            # For Excel files - ENHANCED WITH GOOGLE SHEETS SUPPORT
            elif file_ext == '.xlsx':
                # First check if it's a Google Sheets consolidated file
                file.seek(0)  # Reset file pointer
                if self._detect_google_sheets_xlsx(file):
                    self.logger.info("[PARSE] Detected Google Sheets XLSX format")
                    file.seek(0)  # Reset file pointer again
                    return self._parse_google_sheets_xlsx(file)
                else:
                    # Standard XLSX parsing
                    file.seek(0)  # Reset file pointer
                    return self._parse_excel_file(file)
            
            else:
                self.logger.error(f"Unsupported file type: {file_ext}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error parsing {file.name}: {e}")
            return None
            
    def _parse_csv_file(self, file) -> Optional[pd.DataFrame]:
        """Parse CSV file containing financial data"""
        try:
            self.logger.info("[PARSE-CSV] Starting CSV parsing")
            
            # Reset file pointer
            file.seek(0)
            
            # Try different CSV reading strategies
            strategies = [
                {'sep': ',', 'header': 0, 'index_col': 0},
                {'sep': ',', 'header': None, 'index_col': 0},
                {'sep': ';', 'header': 0, 'index_col': 0},
                {'sep': '\t', 'header': 0, 'index_col': 0},
                {'sep': None, 'header': 0, 'index_col': 0, 'engine': 'python'}
            ]
            
            df = None
            for i, strategy in enumerate(strategies):
                try:
                    file.seek(0)
                    df = pd.read_csv(file, **strategy)
                    
                    if df is not None and not df.empty and len(df.columns) > 0:
                        if len(df) > 2 and len(df.columns) > 1:
                            self.logger.info(f"[PARSE-CSV] Successfully parsed with strategy {i+1}")
                            break
                            
                except Exception as e:
                    self.logger.debug(f"[PARSE-CSV] Strategy {i+1} failed: {e}")
                    continue
            
            if df is None or df.empty:
                self.logger.error("[PARSE-CSV] All parsing strategies failed")
                return None
            
            # Clean and standardize the DataFrame
            df = self._clean_csv_data(df)
            df = self._add_statement_type_prefixes(df)
            
            self.logger.info(f"[PARSE-CSV] Successfully parsed CSV with shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"[PARSE-CSV] Error parsing CSV file: {e}", exc_info=True)
            return None
    
    def _clean_csv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize CSV data"""
        try:
            if df.index.name is None or 'Unnamed' in str(df.index.name):
                df.index.name = "Particulars"
            
            df.columns = [str(col).strip().replace('\n', ' ').replace('\t', ' ') for col in df.columns]
            
            # Handle year columns
            year_pattern = re.compile(r'(20\d{2}|19\d{2}|FY\s*\d{4})')
            new_columns = []
            
            for col in df.columns:
                col_str = str(col)
                year_match = year_pattern.search(col_str)
                if year_match:
                    year = year_match.group(1).replace('FY', '').strip()
                    new_columns.append(f"{year}03")  # Assuming March year-end
                else:
                    new_columns.append(col_str)
            
            df.columns = new_columns
            
            # Convert to numeric
            for col in df.columns:
                df[col] = self._convert_to_numeric(df[col])
            
            df = df.dropna(how='all').dropna(axis=1, how='all')
            df.index = [str(idx).strip() for idx in df.index]
            
            return df
            
        except Exception as e:
            self.logger.error(f"[PARSE-CSV] Error cleaning CSV data: {e}")
            return df
    
    def _convert_to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert series to numeric, handling various formats"""
        try:
            if series.dtype == 'object':
                series = series.astype(str)
                series = series.str.replace(',', '')
                series = series.str.replace('₹', '')
                series = series.str.replace('$', '')
                series = series.str.replace(r'KATEX_INLINE_OPEN(.*?)KATEX_INLINE_CLOSE', r'-\1', regex=True)
                series = series.replace({'-': np.nan, '--': np.nan, 'NA': np.nan, 'nil': 0, '': np.nan})
            
            series = pd.to_numeric(series, errors='coerce')
            return series
            
        except Exception as e:
            self.logger.debug(f"Error converting series to numeric: {e}")
            return series
    
    def _add_statement_type_prefixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statement type prefixes to index based on content analysis"""
        try:
            new_index = []
            
            for idx in df.index:
                idx_str = str(idx).lower()
                
                if any(kw in idx_str for kw in ['revenue', 'income', 'profit', 'loss', 'expense', 'cost', 'ebit', 'tax']):
                    statement_type = 'ProfitLoss'
                elif any(kw in idx_str for kw in ['assets', 'liabilities', 'equity', 'capital', 'borrowings']):
                    statement_type = 'BalanceSheet'
                elif any(kw in idx_str for kw in ['cash flow', 'operating activities', 'investing']):
                    statement_type = 'CashFlow'
                else:
                    statement_type = 'Financial'
                
                if not str(idx).startswith(f"{statement_type}::"):
                    new_index.append(f"{statement_type}::{str(idx).strip()}")
                else:
                    new_index.append(str(idx).strip())
            
            df.index = new_index
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding statement type prefixes: {e}")
            return df
        
    def _parse_excel_file(self, file) -> Optional[pd.DataFrame]:
        """Enhanced Excel file parser that preserves all columns"""
        try:
            self.logger.info("[PARSE-EXCEL] Starting enhanced Excel parsing")
            
            # Read all sheets
            excel_file = pd.ExcelFile(file)
            sheet_names = excel_file.sheet_names
            self.logger.info(f"[PARSE-EXCEL] Found {len(sheet_names)} sheets: {sheet_names}")
            
            all_dataframes = []
            
            for sheet_name in sheet_names:
                try:
                    # First, read the raw data to understand structure
                    df_raw = pd.read_excel(file, sheet_name=sheet_name, header=None)
                    
                    self.logger.info(f"[PARSE-EXCEL] Sheet '{sheet_name}' raw shape: {df_raw.shape}")
                    
                    if df_raw.empty:
                        continue
                    
                    # Log first few rows to understand structure
                    self.logger.debug(f"[PARSE-EXCEL] First 5 rows of {sheet_name}:")
                    for i in range(min(5, len(df_raw))):
                        self.logger.debug(f"Row {i}: {df_raw.iloc[i].tolist()[:5]}...")  # First 5 cols
                    
                    # Find header row
                    header_row = self._find_header_row_excel_enhanced(df_raw)
                    
                    if header_row is not None:
                        self.logger.info(f"[PARSE-EXCEL] Found header at row {header_row}")
                        
                        # Extract headers and data
                        headers = df_raw.iloc[header_row].tolist()
                        data_df = df_raw.iloc[header_row + 1:].copy()
                        
                        # Set the first column as index
                        data_df.columns = headers
                        
                        # Find the metric name column (usually first non-numeric column)
                        metric_col_idx = 0
                        for i, col in enumerate(headers):
                            if pd.isna(col) or str(col).strip() == '':
                                continue
                            # Check if this column contains mostly text
                            sample_values = df_raw.iloc[header_row + 1:header_row + 6, i]
                            if sample_values.dtype == 'object':
                                metric_col_idx = i
                                break
                        
                        # Set index
                        if metric_col_idx < len(data_df.columns):
                            data_df = data_df.set_index(data_df.columns[metric_col_idx])
                        
                        # Clean column names - preserve year information
                        clean_columns = []
                        for col in data_df.columns:
                            if pd.isna(col):
                                clean_columns.append(f"Column_{len(clean_columns)}")
                            else:
                                # Check if it's a date/year
                                col_str = str(col).strip()
                                if re.search(r'\d{4}', col_str):  # Contains a year
                                    # Extract just the year/date part
                                    year_match = re.search(r'(\d{6}|\d{4})', col_str)
                                    if year_match:
                                        clean_columns.append(year_match.group(1))
                                    else:
                                        clean_columns.append(col_str)
                                else:
                                    clean_columns.append(col_str)
                        
                        data_df.columns = clean_columns
                        
                        # Remove the index column from columns if it got duplicated
                        if data_df.index.name in data_df.columns:
                            data_df = data_df.drop(columns=[data_df.index.name])
                        
                    else:
                        # Fallback: use first row as header
                        self.logger.warning(f"[PARSE-EXCEL] No clear header found, using row 0")
                        data_df = pd.read_excel(file, sheet_name=sheet_name)
                    
                    # Detect statement type
                    statement_type = self._detect_statement_type_from_sheet(sheet_name, data_df)
                    
                    # Add statement type prefix to index
                    if not data_df.empty:
                        data_df.index = [f"{statement_type}::{str(idx).strip()}" for idx in data_df.index]
                        
                        # Clean numeric data
                        data_df = self._clean_excel_data(data_df)
                        
                        if not data_df.empty:
                            all_dataframes.append(data_df)
                            self.logger.info(f"[PARSE-EXCEL] Processed {len(data_df)} rows from sheet {sheet_name}")
                            self.logger.info(f"[PARSE-EXCEL] Columns: {list(data_df.columns)}")
                            
                except Exception as e:
                    self.logger.warning(f"[PARSE-EXCEL] Error processing sheet {sheet_name}: {e}")
                    continue
            
            # Combine all dataframes
            if all_dataframes:
                combined_df = pd.concat(all_dataframes, axis=0)
                
                # Remove duplicates
                if combined_df.index.duplicated().any():
                    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                
                self.logger.info(f"[PARSE-EXCEL] Successfully parsed {len(combined_df)} total rows")
                self.logger.info(f"[PARSE-EXCEL] Final columns: {list(combined_df.columns)}")
                
                return combined_df
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"[PARSE-EXCEL] Error parsing Excel file: {e}", exc_info=True)
            return None
    
    def _find_header_row_excel_enhanced(self, df: pd.DataFrame) -> Optional[int]:
        """Enhanced header detection for Excel files"""
        self.logger.debug("[PARSE-EXCEL] Searching for header row")
        
        for i in range(min(20, len(df))):
            row = df.iloc[i]
            
            # Count cells that look like dates/years
            date_count = 0
            for val in row:
                if pd.notna(val):
                    val_str = str(val).strip()
                    
                    # Check for year patterns
                    if re.search(r'(20\d{2}|19\d{2})', val_str):
                        date_count += 1
                        
                    # Check for date-like numbers (YYYYMM format)
                    try:
                        if isinstance(val, (int, float)) and len(str(int(val))) == 6:
                            val_int = int(val)
                            year = val_int // 100
                            month = val_int % 100
                            if 1900 <= year <= 2100 and 1 <= month <= 12:
                                date_count += 1
                    except:
                        pass
            
            if date_count >= 2:  # At least 2 date-like values
                self.logger.info(f"[PARSE-EXCEL] Found header row at index {i} with {date_count} date patterns")
                return i
        
        self.logger.warning("[PARSE-EXCEL] No clear header row found")
        return None

    def _find_header_row_excel(self, df: pd.DataFrame) -> Optional[int]:
        """Find header row in Excel data - enhanced version"""
        self.logger.debug("Searching for header row in Excel data")
        
        for i in range(min(15, len(df))):  # Check first 15 rows
            row = df.iloc[i]
            
            # Count year patterns in this row
            year_count = 0
            for val in row:
                if pd.notna(val):
                    val_str = str(val).strip()
                    # Look for various year formats
                    if re.search(r'(20\d{2}|19\d{2}|FY\s*\d{4}|\d{4}-\d{2})', val_str):
                        year_count += 1
            
            # Also check for month patterns (like "Mar-2023")
            month_year_count = 0
            for val in row:
                if pd.notna(val):
                    val_str = str(val).strip()
                    if re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-\s]*\d{4}', val_str, re.IGNORECASE):
                        month_year_count += 1
            
            total_date_patterns = year_count + month_year_count
            
            if total_date_patterns >= 2:
                self.logger.info(f"Found header row at index {i} with {total_date_patterns} date patterns")
                return i
        
        self.logger.warning("No clear header row found")
        return None
    
    def _clean_excel_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Excel data - convert to numeric"""
        for col in df.columns:
            if col != df.index.name:  # Don't convert the index column
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '').str.replace('(', '-').str.replace(')', ''),
                    errors='coerce'
                )
        
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        return df
    
    def _detect_statement_type_excel(self, df: pd.DataFrame) -> str:
        """Detect statement type from Excel content"""
        # Convert DataFrame to string for analysis
        content_str = ' '.join([str(val) for val in df.values.flatten()[:200] if pd.notna(val)]).lower()
        
        if any(term in content_str for term in ['cash flow', 'operating activities', 'investing']):
            return 'CashFlow'
        elif any(term in content_str for term in ['profit', 'loss', 'revenue', 'income statement']):
            return 'ProfitLoss'
        elif any(term in content_str for term in ['assets', 'liabilities', 'balance sheet']):
            return 'BalanceSheet'
        else:
            return 'Financial'

    def _validate_parsed_google_sheets_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the parsed Google Sheets data has the expected structure
        """
        if df.empty:
            self.logger.error("[VALIDATE] DataFrame is empty")
            return False
        
        # Check for required statement types
        statement_types = set()
        for idx in df.index:
            if '::' in str(idx):
                statement_type = str(idx).split('::')[0]
                statement_types.add(statement_type)
        
        self.logger.info(f"[VALIDATE] Found statement types: {statement_types}")
        
        # We should have at least 2 statement types
        if len(statement_types) < 2:
            self.logger.warning(f"[VALIDATE] Only found {len(statement_types)} statement types")
        
        # Check for reasonable number of metrics
        metrics_per_type = {}
        for st in statement_types:
            count = sum(1 for idx in df.index if str(idx).startswith(f"{st}::"))
            metrics_per_type[st] = count
        
        self.logger.info(f"[VALIDATE] Metrics per statement type: {metrics_per_type}")
        
        # Check for year columns
        year_pattern = re.compile(r'^\d{6}$')  # YYYYMM format
        year_columns = [col for col in df.columns if year_pattern.match(str(col))]
        
        if len(year_columns) < 2:
            self.logger.warning(f"[VALIDATE] Only found {len(year_columns)} year columns")
            return False
        
        self.logger.info(f"[VALIDATE] Found {len(year_columns)} year columns: {year_columns}")
        
        # Check for data completeness
        non_null_count = df.notna().sum().sum()
        total_cells = df.size
        completeness = (non_null_count / total_cells) * 100 if total_cells > 0 else 0
        
        self.logger.info(f"[VALIDATE] Data completeness: {completeness:.1f}%")
        
        return True
    
    def _identify_statement_type_from_table(self, table: pd.DataFrame) -> Optional[str]:
        """
        Identify the type of financial statement from table content
        """
        # Convert table to string for analysis
        table_str = table.astype(str).values.flatten()
        table_text = ' '.join(table_str).lower()
        
        # Check for statement indicators
        if any(term in table_text for term in ['profit', 'loss', 'revenue', 'income statement', 'p&l']):
            return 'ProfitLoss'
        elif any(term in table_text for term in ['balance sheet', 'assets', 'liabilities', 'equity']):
            return 'BalanceSheet'
        elif any(term in table_text for term in ['cash flow', 'cash from', 'operating activities', 'investing activities']):
            return 'CashFlow'
        
        return None
    
    def _process_financial_table(self, table: pd.DataFrame, statement_type: str) -> Optional[pd.DataFrame]:
        """
        Process a financial table based on its type
        """
        try:
            # Find the header row (usually contains years)
            header_row = None
            for i in range(min(10, len(table))):
                row = table.iloc[i].astype(str)
                # FIXED: Check for both 4-digit and 6-digit year patterns
                year_count = sum(1 for val in row if re.search(r'(20\d{2}|19\d{2}|20\d{4}|19\d{4})', str(val)))
                if year_count >= 2:
                    header_row = i
                    break
            
            if header_row is None:
                self.logger.warning(f"Could not find header row in {statement_type} table")
                return None
            
            # Set the header and clean the data
            df = table.iloc[header_row:].copy()
            df.columns = table.iloc[header_row]
            df = df.iloc[1:]  # Remove the header row from data
            
            # Set the first column as index (usually contains line items)
            if df.iloc[:, 0].dtype == 'object':
                df = df.set_index(df.columns[0])
            
            # Clean column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Remove empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Add statement type prefix to index
            df.index = [f"{statement_type}::{str(idx).strip()}" for idx in df.index]
            
            # Convert numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('(', '-').str.replace(')', ''), errors='coerce')
            
            # Special handling for Cash Flow to ensure detail items are captured
            if statement_type == 'CashFlow':
                df = self._ensure_cash_flow_details(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing {statement_type} table: {e}")
            return None
        
    def _ensure_cash_flow_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure cash flow statement includes detailed items, not just summaries
        """
        # Check if we have detailed items
        has_details = any('purchase' in str(idx).lower() or 'capital' in str(idx).lower() 
                         or 'acquisition' in str(idx).lower() for idx in df.index)
        
        if not has_details:
            self.logger.warning("Cash flow statement appears to have only summary items")
            
            # Look for investing activities section
            investing_idx = None
            for idx in df.index:
                if 'investing activities' in str(idx).lower():
                    investing_idx = idx
                    break
            
            if investing_idx:
                # Try to extract the breakdown if it exists in the original data
                # This is a placeholder - you might need to adjust based on your specific file format
                self.logger.info("Attempting to extract cash flow details from investing activities section")
        
        return df

    def _parse_single_table_fallback(self, file) -> Optional[pd.DataFrame]:
        """
        [ROBUST FALLBACK] Fallback method for parsing when the multi-table approach fails.
        This handles simpler, single-table HTML files or malformed files.
        """
        self.logger.warning("Executing single-table fallback parser.")
        try:
            file.seek(0)
            # Attempt to read all tables in the file
            tables = pd.read_html(file, flavor='lxml')
            
            if not tables:
                self.logger.error("Fallback failed: No tables found in the file.")
                return None
    
            # Heuristic: The main data table is usually the largest one.
            df = max(tables, key=len)
            self.logger.info(f"Fallback selected the largest table with shape: {df.shape}")
    
            # --- ADD DEBUG 1: Log initial columns ---
            self.logger.info(f"[DEBUG] Initial columns before any processing: {list(df.columns)}")
    
            # 1. Handle and collapse MultiIndex columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                self.logger.debug("Collapsing MultiIndex columns in fallback.")
                # Join levels with ' >> ', handling potential NaN/None levels gracefully
                df.columns = [' >> '.join(str(level).strip() for level in col if pd.notna(level) and str(level).strip()).strip(' >> ') 
                              for col in df.columns]
            
            # 2. General column name cleanup
            df.columns = [str(col).strip().replace('  ', ' ') for col in df.columns]
    
            # --- ADD DEBUG 2: Log columns after cleanup ---
            self.logger.info(f"[DEBUG] Columns after cleanup: {list(df.columns)}")
    
            # 3. Modified: Drop unnamed columns, but preserve those that look like years
            columns_to_keep = []
            for col in df.columns:
                col_str = str(col)
                # Keep if it's NOT unnamed OR if it contains a year pattern (4-digit or 6-digit)
                if not col_str.startswith('Unnamed') or re.search(r'(20\d{2}|19\d{2}|20\d{4}|19\d{4})', col_str):
                    columns_to_keep.append(col)
                else:
                    self.logger.info(f"[DEBUG] Dropping unnamed non-year column: {col}")
    
            df = df[columns_to_keep]
    
            # --- ADD DEBUG 3: Log columns after dropping unnamed ---
            self.logger.info(f"[DEBUG] Columns after dropping unnamed (preserving years): {list(df.columns)}")
            self.logger.info(f"[DEBUG] Number of columns after drop: {len(df.columns)}")
    
            # 4. Intelligently find and set the index column
            # This is often the first column containing descriptive text.
            potential_index_cols = ['Particulars', 'Description', 'Items', 'Metric']
            index_col_found = False
    
            for col_name in potential_index_cols:
                # Find columns that contain the potential index name
                matching_cols = [c for c in df.columns if col_name.lower() in str(c).lower()]
                if matching_cols:
                    df = df.set_index(matching_cols[0])
                    df.index.name = "Particulars"  # Standardize index name
                    index_col_found = True
                    self.logger.debug(f"Set index to column: {matching_cols[0]}")
                    break
            
            # If no named index column was found, use the first column as a last resort
            if not index_col_found and df.shape[1] > 0:
                if df.iloc[:, 0].dtype == 'object' and df.iloc[:, 0].nunique() > len(df) / 2:
                    self.logger.debug("Using the first column as the index as a last resort.")
                    df = df.set_index(df.columns[0])
                    df.index.name = "Particulars"
    
            # --- ADD DEBUG 4: Log after setting index ---
            self.logger.info(f"[DEBUG] Columns after setting index: {list(df.columns)}")
    
            # 5. Remove rows and columns that are entirely empty
            before_shape = df.shape
            df = df.dropna(how='all').dropna(axis=1, how='all')
            after_shape = df.shape
    
            # --- ADD DEBUG 5: Log after dropna ---
            self.logger.info(f"[DEBUG] Shape before dropna: {before_shape}")
            self.logger.info(f"[DEBUG] Shape after dropna: {after_shape}")
            self.logger.info(f"[DEBUG] Columns after dropna: {list(df.columns)}")
    
            if df.empty:
                self.logger.error("Fallback failed: DataFrame is empty after cleaning.")
                return None
    
            # --- ADD FINAL CHECK: Verify expected number of year columns ---
            year_columns = [col for col in df.columns if re.search(r'(20\d{2}|19\d{2}|20\d{4}|19\d{4})', str(col))]
            self.logger.info(f"[DEBUG] Detected year columns: {year_columns}")
            if len(year_columns) < 10:  # Assuming you expect 10 years
                self.logger.warning(f"[DEBUG] Expected at least 10 year columns, but found {len(year_columns)}. Possible data loss!")
    
            self.logger.info(f"Fallback parsing successful. Final shape: {df.shape}")
            return df
    
        except Exception as e:
            self.logger.error(f"Error during single-table fallback parsing: {e}", exc_info=True)
            return None
        
    def _validate_and_repair_parsed_data(self, df: pd.DataFrame, file_or_content: Union[UploadedFile, io.BytesIO]) -> pd.DataFrame:
        """
        [ROBUST IMPLEMENTATION] Validates the initial parsed DataFrame for common catastrophic
        parsing errors (e.g., uniform values, mixed-up columns) and triggers a more robust
        parsing attempt if issues are found.
        """
        self.logger.info("Validating initial parse quality...")
        
        # --- Define conditions that indicate a catastrophic parsing failure ---
        parsing_failed = False

        # Condition 1: A significant number of numeric columns contain only one or two unique, suspicious values.
        # This was the exact problem you observed in your logs.
        suspicious_values_found = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 2: # Only check if there's enough data to be suspicious
            for col in numeric_cols:
                unique_vals = df[col].dropna().unique()
                if 1 <= len(unique_vals) <= 2 and all(val in [0.0, 1.54, -1.54] for val in unique_vals):
                    suspicious_values_found += 1
            
            # If more than 50% of numeric columns are bad, it's a failed parse.
            if suspicious_values_found / len(numeric_cols) > 0.5:
                self.logger.error("PARSING FAILURE DETECTED: Most numeric columns contain only suspicious values (0.0 or 1.54).")
                parsing_failed = True

        # Condition 2: The DataFrame has an extremely wide shape with very few rows,
        # suggesting rows and columns have been inverted.
        if not parsing_failed and df.shape[1] > 20 and df.shape[0] < 10:
             self.logger.error(f"PARSING FAILURE DETECTED: Suspicious DataFrame shape ({df.shape}). Rows and columns may be inverted.")
             parsing_failed = True

        # Condition 3: A single column name contains keywords for multiple statement types.
        if not parsing_failed:
            for col in df.columns:
                col_str = str(col).lower()
                if ('profit' in col_str or 'loss' in col_str) and ('balance' in col_str or 'asset' in col_str):
                    self.logger.error(f"PARSING FAILURE DETECTED: Column '{col}' contains mixed statement types.")
                    parsing_failed = True
                    break

        # --- Re-parsing Logic ---
        if parsing_failed:
            st.warning("Initial data parsing seems incorrect. Attempting a more robust deep-parse...")
            self.logger.warning("Triggering robust re-parse due to validation failure.")
            
            # Reset the file pointer to the beginning before re-parsing
            if hasattr(file_or_content, 'seek'):
                file_or_content.seek(0)
            
            # Attempt to re-parse using the new, robust multi-table parser
            repaired_df = self._parse_single_file(file_or_content)
            
            if repaired_df is not None and not repaired_df.empty:
                self.logger.info("RECOVERY SUCCESS: Robust re-parsing yielded a valid DataFrame.")
                st.success("Successfully repaired the data structure!")
                return repaired_df
            else:
                self.logger.error("RECOVERY FAILED: Robust re-parsing also failed to produce a valid DataFrame.")
                st.error("Data structure could not be automatically repaired. Please check the source file.")
                # Return the original (bad) DataFrame to avoid crashing, but the error is noted.
                return df
        else:
            self.logger.info("✓ Initial parse validation passed.")
            return df
        
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare dataframe for analysis - preserve all data columns"""
        try:
            self.logger.info(f"[CLEAN] Starting with shape: {df.shape}, columns: {list(df.columns)[:5]}...")
            
            cleaned_df = self._standardize_dataframe(df)
            
            # Only remove truly empty unnamed columns
            cols_to_drop = []
            for col in cleaned_df.columns:
                if 'Unnamed' in str(col):
                    # Check if this column has any non-null data
                    if cleaned_df[col].notna().sum() == 0:
                        cols_to_drop.append(col)
                    else:
                        self.logger.info(f"[CLEAN] Keeping 'Unnamed' column {col} - contains data")
            
            if cols_to_drop:
                cleaned_df = cleaned_df.drop(columns=cols_to_drop)
                self.logger.info(f"[CLEAN] Removed {len(cols_to_drop)} empty unnamed columns")
            
            # Convert numeric columns
            for col in cleaned_df.columns:
                try:
                    cleaned_df[col] = self._convert_to_numeric(cleaned_df[col])
                except Exception as e:
                    self.logger.warning(f"Could not convert column {col} to numeric: {e}")
            
            self.logger.info(f"[CLEAN] Final shape: {cleaned_df.shape}, columns: {list(cleaned_df.columns)}")
            
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Error cleaning dataframe: {e}")
            return df
    
    def _attempt_data_recovery(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to recover data when all columns were removed during cleaning
        """
        self.logger.info("Attempting data recovery from original DataFrame")
        
        try:
            # Create a copy of the original data
            recovery_df = original_df.copy()
            
            # If all columns are unnamed, try to find a header row within the data
            if all('Unnamed' in str(col) for col in recovery_df.columns):
                self.logger.info("All columns are unnamed - searching for header row")
                
                # Look for a row that might contain years (header row)
                header_row_idx = None
                for idx in range(min(10, len(recovery_df))):
                    row_values = recovery_df.iloc[idx].astype(str)
                    year_count = sum(1 for val in row_values if re.search(r'(20\d{2}|19\d{2})', str(val)))
                    
                    if year_count >= 2:  # Found a row with at least 2 years
                        header_row_idx = idx
                        self.logger.info(f"Found potential header row at index {idx}")
                        break
                
                if header_row_idx is not None:
                    # Use this row as column headers
                    new_columns = recovery_df.iloc[header_row_idx].astype(str).tolist()
                    recovery_df.columns = new_columns
                    recovery_df = recovery_df.iloc[header_row_idx + 1:]  # Remove header row from data
                    
                    # Clean the new column names
                    recovery_df.columns = [str(col).strip() for col in recovery_df.columns]
                    
                    self.logger.info(f"Recovery successful - new columns: {list(recovery_df.columns)}")
                else:
                    # No header row found, create generic column names
                    recovery_df.columns = [f"Column_{i}" for i in range(len(recovery_df.columns))]
                    self.logger.info("No header row found - using generic column names")
            
            # Remove completely empty columns
            recovery_df = recovery_df.dropna(how='all', axis=1)
            
            return recovery_df
            
        except Exception as e:
            self.logger.error(f"Data recovery failed: {e}")
            # Return original data with generic column names as last resort
            recovery_df = original_df.copy()
            recovery_df.columns = [f"Column_{i}" for i in range(len(recovery_df.columns))]
            return recovery_df

    def _trace_cash_flow_items(self, df: pd.DataFrame, stage: str) -> None:
        """Debug helper to trace cash flow items through processing"""
        cash_flow_items = [idx for idx in df.index if 'cash' in str(idx).lower() or 'flow' in str(idx).lower()]
        capex_items = [idx for idx in df.index if any(kw in str(idx).lower() 
                       for kw in ['capex', 'capital expenditure', 'purchase', 'fixed asset'])]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[TRACE-{stage}] Cash Flow Item Tracking")
        self.logger.info(f"[TRACE-{stage}] Found {len(cash_flow_items)} cash flow items")
        self.logger.info(f"[TRACE-{stage}] Found {len(capex_items)} potential CapEx items")
        
        if capex_items:
            self.logger.info(f"[TRACE-{stage}] CapEx items:")
            for item in capex_items[:10]:
                series = df.loc[item]
                non_null = series.notna().sum()
                self.logger.info(f"  - {item} (non-null values: {non_null})")
        
        self.logger.info(f"{'='*60}\n")
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame structure specifically for financial data"""
        try:
            std_df = df.copy()
    
            # Clean column names
            if isinstance(std_df.columns, pd.Index):
                std_df.columns = [
                    str(col).strip()
                    .replace('Fice >>', 'Finance >>')
                    .replace('  ', ' ')
                    .replace('\n', ' ')
                    .replace('\t', ' ')
                    for col in std_df.columns
                ]
            
            # CRITICAL: Preserve cash flow items during duplicate handling
            if std_df.index.duplicated().any():
                self.logger.info("Found rows with similar names - making indices unique")
                
                new_index = []
                seen_indices = {}
                
                for idx in std_df.index:
                    idx_str = str(idx) if not pd.isna(idx) else "EmptyIndex"
                    
                    # Special handling for cash flow items - preserve original names
                    if 'cashflow::' in idx_str.lower() or 'cash flow' in idx_str.lower():
                        # For cash flow items, append file identifier instead of version
                        if idx_str in seen_indices:
                            seen_indices[idx_str] += 1
                            # Keep original name but log the duplicate
                            self.logger.warning(f"Duplicate cash flow item found: {idx_str}")
                            unique_idx = idx_str  # Keep original name
                        else:
                            seen_indices[idx_str] = 0
                            unique_idx = idx_str
                    else:
                        # Standard duplicate handling for non-cash flow items
                        if idx_str in seen_indices:
                            seen_indices[idx_str] += 1
                            unique_idx = f"{idx_str}_v{seen_indices[idx_str]}"
                        else:
                            seen_indices[idx_str] = 0
                            unique_idx = idx_str
                    
                    new_index.append(unique_idx)
                
                std_df.index = new_index
                self.logger.info(f"Made {sum(v for v in seen_indices.values() if v > 0)} indices unique")
            
            # Remove completely empty rows but preserve cash flow items
            before_count = len(std_df)
            
            # Identify cash flow items to preserve
            cash_flow_indices = [idx for idx in std_df.index 
                                if 'cashflow::' in str(idx).lower() or 'cash flow' in str(idx).lower()]
            
            # Remove only non-cash flow empty rows
            non_cash_flow_mask = ~std_df.index.isin(cash_flow_indices)
            empty_rows_mask = std_df.isnull().all(axis=1)
            rows_to_remove = non_cash_flow_mask & empty_rows_mask
            
            std_df = std_df[~rows_to_remove]
            after_count = len(std_df)
            
            if before_count != after_count:
                self.logger.info(f"Removed {before_count - after_count} completely empty rows (preserved cash flow items)")
            
            # Remove completely empty columns
            before_cols = len(std_df.columns)
            std_df = std_df.dropna(how='all', axis=1)
            after_cols = len(std_df.columns)
            
            if before_cols != after_cols:
                self.logger.info(f"Removed {before_cols - after_cols} completely empty columns")
            
            return std_df
    
        except Exception as e:
            self.logger.error(f"Error standardizing DataFrame: {e}")
            return df
        
    def _inspect_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """Debug helper to inspect DataFrame structure"""
        self.logger.info(f"\nInspecting DataFrame from {filename}:")
        self.logger.info(f"Shape: {df.shape}")
        self.logger.info(f"Columns: {df.columns.tolist()}")

        if df.index.duplicated().any():
            dup_indices = df.index[df.index.duplicated(keep=False)].unique()
            self.logger.warning(f"Duplicate indices found: {dup_indices[:5].tolist()}")

        self.logger.info(f"First few rows:\n{df.head()}")

    def _merge_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge dataframes by aligning years and preserving ALL statement types and rows"""
        # Step 1: Find all unique years across all DFs
        all_years = set()
        for df in dataframes:
            for col in df.columns:
                match = YEAR_REGEX.search(str(col))
                if match:
                    year = match.group(1)
                    if year.startswith('FY'):
                        year = year.replace('FY ', '20')
                    all_years.add(year)
        
        final_columns = sorted(all_years)
        
        # Step 2: Create master DF with all rows and normalized year columns
        master_index = set()
        merged_data = {year: {} for year in final_columns}  # Temp dict for values
        
        for df in dataframes:
            statement_type = self._detect_statement_type(df)
            
            # Prefix indices
            prefixed_index = [f"{statement_type}::{str(idx).strip()}" for idx in df.index]
            
            # Add to master index
            master_index.update(prefixed_index)
            
            # Normalize and transfer values
            for col in df.columns:
                match = YEAR_REGEX.search(str(col))
                if match:
                    year = match.group(1)
                    if year.startswith('FY'):
                        year = year.replace('FY ', '20')
                    
                    for i, idx in enumerate(df.index):
                        prefixed_idx = prefixed_index[i]
                        value = df.loc[idx, col]
                        if pd.notna(value):
                            if prefixed_idx not in merged_data[year]:
                                merged_data[year][prefixed_idx] = value
                            else:
                                # If conflict, prefer non-zero or log
                                if merged_data[year][prefixed_idx] == 0 and value != 0:
                                    merged_data[year][prefixed_idx] = value
                                self.logger.warning(f"Conflict for {prefixed_idx} in {year}: keeping {merged_data[year][prefixed_idx]}")
        
        # Step 3: Build final DF
        merged_df = pd.DataFrame(index=sorted(master_index), columns=final_columns)
        
        for year in final_columns:
            for idx, value in merged_data[year].items():
                merged_df.loc[idx, year] = value
        
        # Fill remaining NaNs with 0 (or np.nan if preferred)
        merged_df = merged_df.fillna(0)
        
        # Log debugging
        self.logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        self.logger.info(f"Merged columns (years): {merged_df.columns.tolist()}")
        
        # Verify row counts by type
        for stmt_type in ['ProfitLoss', 'BalanceSheet', 'CashFlow']:
            type_rows = [idx for idx in merged_df.index if idx.startswith(f"{stmt_type}::")]
            self.logger.info(f"Preserved {len(type_rows)} rows for {stmt_type}")
        
        return merged_df

    def _detect_google_sheets_xlsx(self, file) -> bool:
        """
        Detect if an XLSX file is a Google Sheets consolidated financial statement
        """
        try:
            # Read the first sheet to check structure
            df = pd.read_excel(file, sheet_name=0, nrows=10)
            
            # Google Sheets consolidated files typically have specific patterns
            # Check for multiple statement indicators in one sheet
            content_str = ' '.join([str(val) for val in df.values.flatten() if pd.notna(val)]).lower()
            
            # Look for indicators of consolidated statements
            has_balance_sheet = any(term in content_str for term in ['balance sheet', 'assets', 'liabilities'])
            has_profit_loss = any(term in content_str for term in ['profit', 'loss', 'income statement', 'p&l'])
            has_cash_flow = any(term in content_str for term in ['cash flow', 'operating activities'])
            
            # If we find indicators of multiple statements, it's likely a consolidated file
            statement_count = sum([has_balance_sheet, has_profit_loss, has_cash_flow])
            
            self.logger.info(f"[PARSE-DETECT] Google Sheets detection - Statements found: {statement_count}")
            
            return statement_count >= 2
            
        except Exception as e:
            self.logger.debug(f"Not a Google Sheets consolidated file: {e}")
            return False
    
    def _parse_google_sheets_xlsx(self, file) -> pd.DataFrame:
        """
        Parse Google Sheets XLSX file containing consolidated financial statements
        """
        self.logger.info("[PARSE-GSHEET] Starting Google Sheets XLSX parsing")
        
        try:
            # Read all sheets in the file
            excel_file = pd.ExcelFile(file)
            sheet_names = excel_file.sheet_names
            self.logger.info(f"[PARSE-GSHEET] Found {len(sheet_names)} sheets: {sheet_names}")
            
            all_dataframes = []
            
            # Process each sheet
            for sheet_name in sheet_names:
                self.logger.info(f"[PARSE-GSHEET] Processing sheet: {sheet_name}")
                
                # Read the entire sheet first to understand its structure
                df_raw = pd.read_excel(file, sheet_name=sheet_name, header=None)
                
                if df_raw.empty:
                    self.logger.warning(f"[PARSE-GSHEET] Sheet {sheet_name} is empty, skipping")
                    continue
                
                # Detect statement type from sheet name or content
                statement_type = self._detect_statement_type_from_sheet(sheet_name, df_raw)
                
                # Parse the sheet based on detected structure
                parsed_df = self._parse_google_sheet_structure(df_raw, statement_type, sheet_name)
                
                if parsed_df is not None and not parsed_df.empty:
                    all_dataframes.append(parsed_df)
                    self.logger.info(f"[PARSE-GSHEET] Successfully parsed {len(parsed_df)} rows from {sheet_name}")
            
            # Combine all dataframes
            if all_dataframes:
                combined_df = pd.concat(all_dataframes, axis=0)
                self.logger.info(f"[PARSE-GSHEET] Combined {len(all_dataframes)} sheets into {len(combined_df)} total rows")
                
                # Ensure no duplicate indices
                if combined_df.index.duplicated().any():
                    self.logger.warning("[PARSE-GSHEET] Found duplicate indices, making unique")
                    combined_df = self._make_indices_unique(combined_df)
                
                return combined_df
            else:
                self.logger.error("[PARSE-GSHEET] No valid data found in any sheet")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"[PARSE-GSHEET] Error parsing Google Sheets XLSX: {e}", exc_info=True)
            # Fallback to standard XLSX parsing
            return self._parse_excel_file(file)
    
    def _detect_statement_type_from_sheet(self, sheet_name: str, df: pd.DataFrame) -> str:
        """
        Detect financial statement type from sheet name or content
        """
        sheet_lower = sheet_name.lower()
        
        # First try sheet name
        if any(term in sheet_lower for term in ['balance', 'bs', 'position']):
            return 'BalanceSheet'
        elif any(term in sheet_lower for term in ['profit', 'loss', 'p&l', 'pl', 'income']):
            return 'ProfitLoss'
        elif any(term in sheet_lower for term in ['cash', 'flow', 'cf']):
            return 'CashFlow'
        
        # If sheet name is not clear, analyze content
        content_str = ' '.join([str(val) for val in df.values.flatten()[:100] if pd.notna(val)]).lower()
        
        # Count keywords for each statement type
        bs_score = sum(1 for term in ['assets', 'liabilities', 'equity', 'current assets', 'total assets'] 
                       if term in content_str)
        pl_score = sum(1 for term in ['revenue', 'profit', 'loss', 'income', 'expense', 'ebit'] 
                       if term in content_str)
        cf_score = sum(1 for term in ['cash flow', 'operating activities', 'investing', 'financing'] 
                       if term in content_str)
        
        # Return the type with highest score
        scores = {'BalanceSheet': bs_score, 'ProfitLoss': pl_score, 'CashFlow': cf_score}
        statement_type = max(scores, key=scores.get)
        
        self.logger.info(f"[PARSE-GSHEET] Detected {statement_type} from content analysis (scores: {scores})")
        
        return statement_type
    
    def _parse_google_sheet_structure(self, df: pd.DataFrame, statement_type: str, sheet_name: str) -> pd.DataFrame:
        """
        Parse the specific structure of a Google Sheets financial statement
        """
        self.logger.info(f"[PARSE-GSHEET-STRUCT] Parsing {statement_type} from sheet: {sheet_name}")
        
        # Step 1: Find the header row (contains years)
        header_row_idx = self._find_header_row_google_sheets(df)
        
        if header_row_idx is None:
            self.logger.error(f"[PARSE-GSHEET-STRUCT] Could not find header row in {sheet_name}")
            return None
        
        self.logger.info(f"[PARSE-GSHEET-STRUCT] Found header row at index {header_row_idx}")
        
        # Step 2: Extract year columns
        year_columns = self._extract_year_columns_google_sheets(df, header_row_idx)
        
        if not year_columns:
            self.logger.error(f"[PARSE-GSHEET-STRUCT] No year columns found in {sheet_name}")
            return None
        
        self.logger.info(f"[PARSE-GSHEET-STRUCT] Found {len(year_columns)} year columns: {list(year_columns.keys())}")
        
        # Step 3: Find the metrics column (usually the first column)
        metrics_col_idx = self._find_metrics_column(df, header_row_idx)
        
        if metrics_col_idx is None:
            self.logger.error(f"[PARSE-GSHEET-STRUCT] Could not find metrics column in {sheet_name}")
            return None
        
        # Step 4: Extract the data
        parsed_data = {}
        
        # Iterate through rows after header
        for row_idx in range(header_row_idx + 1, len(df)):
            # Get the metric name
            metric_name = df.iloc[row_idx, metrics_col_idx]
            
            # Skip empty rows or non-string metrics
            if pd.isna(metric_name) or not isinstance(metric_name, str):
                continue
            
            # Clean the metric name
            metric_name = str(metric_name).strip()
            
            # Skip if it's a section header or total line (you can customize this)
            if metric_name.upper() == metric_name or metric_name.startswith('---'):
                continue
            
            # Add statement type prefix
            full_metric_name = f"{statement_type}::{metric_name}"
            
            # Extract values for each year
            row_data = {}
            for year, col_idx in year_columns.items():
                try:
                    value = df.iloc[row_idx, col_idx]
                    
                    # Clean and convert the value
                    if pd.notna(value):
                        # Handle string values with commas, parentheses, etc.
                        if isinstance(value, str):
                            # Remove currency symbols, commas, spaces
                            value_clean = value.replace(',', '').replace('₹', '').replace('$', '').strip()
                            
                            # Handle parentheses (negative numbers)
                            if value_clean.startswith('(') and value_clean.endswith(')'):
                                value_clean = '-' + value_clean[1:-1]
                            
                            # Handle special cases
                            if value_clean in ['-', '--', 'NA', 'N/A', 'nil', 'Nil']:
                                value = 0
                            else:
                                try:
                                    value = float(value_clean)
                                except ValueError:
                                    value = 0
                        else:
                            value = float(value)
                    else:
                        value = 0
                    
                    row_data[year] = value
                    
                except Exception as e:
                    self.logger.debug(f"Error parsing value at row {row_idx}, col {col_idx}: {e}")
                    row_data[year] = 0
            
            # Add to parsed data if we have valid data
            if any(v != 0 for v in row_data.values()):
                parsed_data[full_metric_name] = row_data
        
        # Convert to DataFrame
        if parsed_data:
            result_df = pd.DataFrame.from_dict(parsed_data, orient='index')
            
            # Sort columns (years) chronologically
            result_df = result_df.reindex(sorted(result_df.columns), axis=1)
            
            self.logger.info(f"[PARSE-GSHEET-STRUCT] Successfully parsed {len(result_df)} metrics with {len(result_df.columns)} years")
            
            return result_df
        else:
            self.logger.warning(f"[PARSE-GSHEET-STRUCT] No valid data found in {sheet_name}")
            return pd.DataFrame()
    
    def _find_header_row_google_sheets(self, df: pd.DataFrame) -> Optional[int]:
        """
        Find the row containing year headers in Google Sheets format
        """
        # Look for rows containing year patterns
        for idx in range(min(20, len(df))):  # Check first 20 rows
            row = df.iloc[idx]
            year_count = 0
            
            for val in row:
                if pd.notna(val):
                    val_str = str(val)
                    # Check for year patterns (YYYY, YYYY-YY, FY YYYY, etc.)
                    if re.search(r'(20\d{2}|19\d{2}|FY\s*\d{4}|\d{4}-\d{2})', val_str):
                        year_count += 1
            
            # If we find at least 2 year patterns, this is likely the header row
            if year_count >= 2:
                self.logger.debug(f"[PARSE-GSHEET] Found {year_count} year patterns in row {idx}")
                return idx
        
        return None
    
    def _extract_year_columns_google_sheets(self, df: pd.DataFrame, header_row_idx: int) -> Dict[str, int]:
        """
        Extract year column mappings from the header row
        """
        year_columns = {}
        header_row = df.iloc[header_row_idx]
        
        for col_idx, val in enumerate(header_row):
            if pd.notna(val):
                val_str = str(val).strip()
                
                # Extract year using various patterns
                year_match = re.search(r'(20\d{2}|19\d{2})', val_str)
                
                if year_match:
                    year = year_match.group(1)
                    
                    # Handle fiscal year formats
                    if 'FY' in val_str.upper():
                        # Assume fiscal year ends in March (03)
                        year_key = f"{year}03"
                    else:
                        # For calendar year, assume year-end (12) or March (03) based on your data
                        year_key = f"{year}03"  # Adjust this based on your fiscal year convention
                    
                    year_columns[year_key] = col_idx
                    self.logger.debug(f"[PARSE-GSHEET] Mapped column {col_idx} to year {year_key}")
        
        return year_columns
    
    def _find_metrics_column(self, df: pd.DataFrame, header_row_idx: int) -> Optional[int]:
        """
        Find the column containing metric names (usually the first non-empty column)
        """
        # Check each column to find the one with text metrics
        for col_idx in range(min(5, len(df.columns))):  # Check first 5 columns
            # Count non-empty text values in this column (after header)
            text_count = 0
            
            for row_idx in range(header_row_idx + 1, min(header_row_idx + 10, len(df))):
                val = df.iloc[row_idx, col_idx]
                if pd.notna(val) and isinstance(val, str) and len(val.strip()) > 0:
                    text_count += 1
            
            # If we find multiple text values, this is likely the metrics column
            if text_count >= 3:
                self.logger.debug(f"[PARSE-GSHEET] Found metrics column at index {col_idx}")
                return col_idx
        
        # Default to first column if not found
        return 0
    
    def _make_indices_unique(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make duplicate indices unique by appending version numbers
        """
        if not df.index.duplicated().any():
            return df
        
        new_index = []
        seen_indices = {}
        
        for idx in df.index:
            if idx in seen_indices:
                seen_indices[idx] += 1
                new_idx = f"{idx}_v{seen_indices[idx]}"
            else:
                seen_indices[idx] = 0
                new_idx = idx
            new_index.append(new_idx)
        
        df.index = new_index
        return df

    def _verify_cash_flow_preservation(self, original_data: pd.DataFrame, processed_data: pd.DataFrame) -> bool:
        """Verify that cash flow items are preserved during processing"""
        original_cf_items = set(idx for idx in original_data.index 
                               if 'cash' in str(idx).lower() or 'flow' in str(idx).lower())
        
        processed_cf_items = set(idx for idx in processed_data.index 
                                if 'cash' in str(idx).lower() or 'flow' in str(idx).lower())
        
        missing_items = original_cf_items - processed_cf_items
        
        if missing_items:
            self.logger.error(f"CRITICAL: Lost {len(missing_items)} cash flow items during processing!")
            for item in list(missing_items)[:10]:
                self.logger.error(f"  - Lost: {item}")
            return False
        
        self.logger.info(f"✓ All {len(original_cf_items)} cash flow items preserved")
        return True

    def _merge_standalone_data(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Helper function to merge standalone data"""
        processed_dfs = []

        for df in dataframes:
            df_copy = df.copy()
            statement_type = self._detect_statement_type(df_copy)
            
            new_index = [f"{statement_type}::{str(idx).strip()}" for idx in df_copy.index]
            df_copy.index = new_index
            processed_dfs.append(df_copy)

        return pd.concat(processed_dfs, axis=0, sort=False)

    def _detect_statement_type(self, df: pd.DataFrame) -> str:
        """Detect the type of financial statement from column names"""
        if df.columns.empty:
            return "Financial"

        col_sample = str(df.columns[0]).lower()

        if any(keyword in col_sample for keyword in ['profit', 'loss', 'income', 'p&l']):
            return "ProfitLoss"
        elif any(keyword in col_sample for keyword in ['balance', 'sheet']):
            return "BalanceSheet"
        elif any(keyword in col_sample for keyword in ['cash', 'flow']):
            return "CashFlow"
        elif any(keyword in col_sample for keyword in ['equity', 'changes']):
            return "Equity"
        else:
            return "Financial"

    def _extract_company_info(self, df: pd.DataFrame) -> Dict[str, str]:
        """Extract company name and other info from DataFrame columns"""
        company_info = {}

        for col in df.columns:
            col_str = str(col)
            if '>>' in col_str:
                parts = col_str.split('>>')
                if len(parts) >= 3:
                    company_part = parts[2].split('(')[0].strip()
                    company_info['name'] = company_part
                    
                    if '(' in parts[2] and ')' in parts[2]:
                        currency = parts[2].split('(')[1].split(')')[0].strip()
                        company_info['currency'] = currency
                    
                    break

        return company_info

    def _load_sample_data(self, sample_name: str):
        """Load sample data"""
        try:
            with st.spinner(f"Loading {sample_name}..."):
                if "Indian Tech" in sample_name:
                    df = self.sample_generator.generate_indian_tech_company()
                    company_name = "TechCorp India Ltd."
                elif "US Manufacturing" in sample_name:
                    df = self.sample_generator.generate_us_manufacturing()
                    company_name = "ManufactureCorp USA"
                elif "European Retail" in sample_name:
                    df = self.sample_generator.generate_european_retail()
                    company_name = "RetailChain Europe"
                else:
                    st.error("Unknown sample dataset")
                    return

                processed_df, validation_result = self.components['processor'].process(df, "sample_data")
                
                self.set_state('analysis_data', processed_df)
                self.set_state('company_name', company_name)
                self.set_state('data_source', 'sample_data')
                
                st.success(f"✅ Loaded sample data: {sample_name}")
                
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            
    def _load_benchmark_data(self, company_name: str):
        """Load benchmark data for comparison"""
        if company_name == "ITC Ltd":
            try:
                benchmark_dir = Path("data/benchmarks")
                benchmark_dir.mkdir(parents=True, exist_ok=True)
                benchmark_path = benchmark_dir / "itc_ltd.pkl"
    
                if not benchmark_path.exists():
                    self.logger.info(f"Generating synthetic benchmark data for {company_name}")
                    
                    years = ['2019', '2020', '2021', '2022', '2023']
                    
                    data = {
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
                        'Share Capital': [1000, 1000, 1000, 1000, 1000],
                        'Reserves and Surplus': [26000, 31000, 37500, 45500, 55000],
                        
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
                        
                        'Operating Cash Flow': [5500, 6600, 8800, 11000, 14000],
                        'Investing Cash Flow': [-3000, -3500, -4200, -5000, -6000],
                        'Financing Cash Flow': [-1500, -1800, -2200, -2700, -3300],
                        'Capital Expenditure': [2800, 3200, 3800, 4500, 5300],
                        'Free Cash Flow': [2700, 3400, 5000, 6500, 8700],
                        'Depreciation': [1500, 1800, 2100, 2500, 3000],
                    }
                    
                    benchmark_df = pd.DataFrame(data, index=data.keys())
                    
                    labeled_columns = [f"Finance >>Balance Sheet (Standalone)>>{company_name}(Curr. in Crores) >> {year}" 
                                     for year in years]
                    benchmark_df.columns = labeled_columns
                    
                    benchmark_df.to_pickle(benchmark_path)
                    
                else:
                    benchmark_df = pd.read_pickle(benchmark_path)
                
                self.set_state('benchmark_data', benchmark_df)
                
            except Exception as e:
                self.logger.error(f"Error loading benchmark data: {e}")
                st.error(f"Failed to load benchmark data: {str(e)}")
    
        elif company_name == "Hindustan Unilever":
            st.warning("Benchmark data for Hindustan Unilever coming soon")
    
        elif company_name == "Nestle India":
            st.warning("Benchmark data for Nestle India coming soon")
            
    
    @safe_state_access
    def _render_main_content(self):
        """Render main content area"""
        # Use hasattr for safer session state checking
        if hasattr(st.session_state, 'main_content_rendered') and st.session_state.main_content_rendered:
            return
        
        st.session_state.main_content_rendered = True
        
        try:
            if self.config.get('app.enable_ml_features', True):
                self._render_query_bar()
            
            # Just call _render_analysis_interface - it handles both cases
            self._render_analysis_interface()
            
        finally:
            # Clear the flag after rendering
            st.session_state.main_content_rendered = False
    
    def _display_query_result(self, result: Dict[str, Any]):
        """Display natural language query result"""
        if result['type'] == 'growth_analysis':
            st.subheader("📈 Growth Analysis")
            for item in result['data']:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(item['metric'], f"{item['last_value']:,.0f}")
                with col2:
                    st.metric("CAGR", f"{item['cagr']:.1f}%")
                with col3:
                    st.metric("YoY Change", f"{item['yoy_change']:.1f}%")
    
        elif result['type'] == 'forecast':
            st.subheader("🔮 Forecast Results")
            forecasts = result['data'].get('forecasts', {})
            for metric, forecast in forecasts.items():
                st.write(f"**{metric}**")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=forecast['periods'],
                    y=forecast['values'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(dash='dash')
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
        elif result['type'] == 'summary':
            st.subheader("📊 Summary")
            summary = result['data']
            
            cols = st.columns(4)
            summary_data = summary.get('summary', {})
            
            if 'total_metrics' in summary_data:
                cols[0].metric("Total Metrics", summary_data['total_metrics'])
            if 'year_range' in summary_data:
                cols[1].metric("Period", summary_data['year_range'])
            if 'quality_score' in summary:
                cols[2].metric("Data Quality", f"{summary['quality_score']:.0f}%")
            
            if 'insights' in summary:
                st.write("**Key Insights:**")
                for insight in summary['insights']:
                    st.write(f"- {insight}")
    
        else:
            st.info(result.get('message', 'Query processed successfully'))
    
    def _render_welcome_screen(self):
        """Render welcome screen"""
        st.header("Welcome to Elite Financial Analytics Platform v5.1")
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.info("""
            ### 📊 Advanced Analytics
            - Comprehensive ratio analysis
            - ML-powered forecasting
            - Anomaly detection
            - Natural language queries
            """)
    
        with col2:
            st.success("""
            ### 🤖 AI-Powered Features
            - Intelligent metric mapping
            - Kaggle GPU acceleration
            - Pattern recognition
            - Automated insights
            """)
    
        with col3:
            st.warning("""
            ### 👥 Collaboration
            - Real-time collaboration
            - Share analyses
            - Export to multiple formats
            - Interactive tutorials
            """)
    
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.markdown("""
            1. **Upload Data**: Use the sidebar to upload financial statements
            2. **Configure Kaggle GPU**: Optional - for faster AI processing
            3. **AI Mapping**: Let AI automatically map your metrics or do it manually
            4. **Analyze**: Explore comprehensive analysis with ratios, trends, and forecasts
            5. **Query**: Ask questions in natural language about your data
            6. **Collaborate**: Share your analysis with team members
            7. **Export**: Generate professional reports in various formats
            
            **New in v5.1:**
            - 🖥️ **Enhanced Kaggle GPU Integration**: Circuit breaker, request coalescing
            - 🚀 Improved performance monitoring and API metrics
            - 💬 Better error recovery and fallback strategies
            - 👥 Enhanced collaboration features
            - 📈 Advanced caching with memory buffers
            - 🔒 Robust security features
            - 📦 Support for compressed files (ZIP/7Z)
            """)
    
        st.subheader("🎯 Try with Sample Data")
        col1, col2, col3 = st.columns(3)
    
        with col1:
            if st.button("Indian Tech Company", key="sample_indian"):
                self._load_sample_data("Indian Tech Company (IND-AS)")
    
        with col2:
            if st.button("US Manufacturing", key="sample_us"):
                self._load_sample_data("US Manufacturing (GAAP)")
    
        with col3:
            if st.button("European Retail", key="sample_eu"):
                self._load_sample_data("European Retail (IFRS)")
                
   
    @safe_state_access
    def _render_analysis_interface(self):
        """Render main analysis interface"""
        # Use hasattr for safer session state checking
        render_key = 'analysis_interface_rendered'
        if hasattr(st.session_state, render_key) and getattr(st.session_state, render_key):
            return
        
        setattr(st.session_state, render_key, True)
        
        try:
            data = self.get_state('analysis_data')
            
            if data is None:
                self._render_welcome_screen()
                return
            
            # Create a single container for all content
            with st.container():
                tabs = st.tabs([
                    "📊 Overview",
                    "📈 Financial Ratios", 
                    "📉 Trends & Forecasting",
                    "🎯 Penman-Nissim",
                    "🏭 Industry Comparison",
                    "🔍 Data Explorer",
                    "📄 Reports",
                    "🤖 ML Insights"
                ])
            
                with tabs[0]:
                    self._render_overview_tab(data)
            
                with tabs[1]:
                    self._render_ratios_tab(data)
            
                with tabs[2]:
                    self._render_trends_tab(data)
            
                with tabs[3]:
                    self._render_penman_nissim_tab_enhanced(data)
            
                with tabs[4]:
                    self._render_industry_tab(data)
            
                with tabs[5]:
                    self._render_data_explorer_tab(data)
            
                with tabs[6]:
                    self._render_reports_tab(data)
            
                with tabs[7]:
                    self._render_ml_insights_tab(data)
                    
        finally:
            # Clear the render flag
            setattr(st.session_state, render_key, False)
    
    @error_boundary()
    @safe_state_access
    def _render_overview_tab(self, data: pd.DataFrame):
        """Render overview tab with key metrics and insights"""
        st.header("Financial Overview")
    
        analysis_mode = self.get_state('analysis_mode', 'Standalone Analysis')
        if analysis_mode == "Benchmark Comparison":
            benchmark_company = self.get_state('benchmark_company', 'Unknown')
            st.info(f"📊 Benchmark Comparison Mode: Comparing with {benchmark_company}")
    
        if 'mapper' in self.components and hasattr(self.components['mapper'], 'progress_tracker'):
            self._render_progress_tracking()
    
        with performance_monitor.measure("overview_analysis"):
            analysis = self.components['analyzer'].analyze_financial_statements(data)
    
        col1, col2, col3, col4 = st.columns(4)
    
        with col1:
            self.ui_factory.create_metric_card(
                "Total Metrics",
                analysis.get('summary', {}).get('total_metrics', len(data))
            )
    
        with col2:
            self.ui_factory.create_metric_card(
                "Years Covered",
                analysis.get('summary', {}).get('years_covered', len(data.columns))
            )
    
        with col3:
            completeness = analysis.get('summary', {}).get('completeness', 0)
            self.ui_factory.create_metric_card(
                "Data Completeness",
                f"{completeness:.1f}%"
            )
    
        with col4:
            quality_score = analysis.get('quality_score', 0)
            self.ui_factory.create_data_quality_badge(quality_score)
    
        st.subheader("Key Insights")
    
        insights = analysis.get('insights', [])
        if insights:
            for i, insight in enumerate(insights[:8]):
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
    
        if 'anomalies' in analysis:
            anomalies = analysis['anomalies']
            total_anomalies = sum(len(v) for v in anomalies.values())
            
            if total_anomalies > 0:
                st.subheader("🔍 Anomaly Detection")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Value Anomalies", len(anomalies.get('value_anomalies', [])))
                
                with col2:
                    st.metric("Trend Anomalies", len(anomalies.get('trend_anomalies', [])))
                
                with col3:
                    st.metric("Ratio Anomalies", len(anomalies.get('ratio_anomalies', [])))
                
                with st.expander("View Anomaly Details"):
                    for anomaly_type, items in anomalies.items():
                        if items:
                            st.write(f"**{anomaly_type.replace('_', ' ').title()}:**")
                            anomaly_df = pd.DataFrame(items)
                            st.dataframe(anomaly_df, use_container_width=True)
    
        st.subheader("Quick Visualizations")
    
        metrics = analysis.get('metrics', {})
    
        if metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                revenue_data = metrics.get('revenue', [])
                if revenue_data:
                    self._render_metric_chart(revenue_data[0], "Revenue Trend")
            
            with col2:
                profit_data = metrics.get('net_income', [])
                if profit_data:
                    self._render_metric_chart(profit_data[0], "Net Income Trend")
    
        if analysis_mode == "Benchmark Comparison" and self.get_state('benchmark_data') is not None:
            st.subheader("📈 Benchmark Comparison")
            
            company_name = self.get_state('company_name', 'Your Company')
            benchmark_company = self.get_state('benchmark_company', 'Benchmark')
            
            self._render_comparison_charts(data, company_name, benchmark_company)
    
    def _render_progress_tracking(self):
        """Render progress tracking for long operations"""
        progress_tracker = self.components['mapper'].progress_tracker
    
        active_operations = []
        for op_id, op_data in progress_tracker.operations.items():
            if op_data['status'] == 'running':
                active_operations.append((op_id, op_data))
    
        if active_operations:
            st.markdown("""
            <div class="progress-tracker">
                <strong>Processing Operations:</strong>
            </div>
            """, unsafe_allow_html=True)
            
            for op_id, op_data in active_operations:
                progress = op_data['completed'] / op_data['total']
                st.markdown(f"""
                <div style="margin: 5px 0;">
                    <small>{op_data['description']}</small>
                    <div style="background: #e0e0e0; border-radius: 4px; height: 8px;">
                        <div class="progress-bar" style="width: {progress * 100}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_metric_chart(self, metric_data: Dict, title: str):
        """Render a simple metric chart"""
        values = metric_data.get('values', {})
    
        if values:
            years = list(values.keys())
            amounts = list(values.values())
            
            formatter = get_number_formatter(self.get_state('number_format_value'))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years,
                y=amounts,
                mode='lines+markers',
                name=metric_data.get('name', 'Value'),
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate='%{x}: %{y:,.0f}<extra></extra>'
            ))
            
            if len(years) > 2:
                z = np.polyfit(range(len(years)), amounts, 1)
                p = np.poly1d(z)
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=p(range(len(years))),
                    mode='lines',
                    name='Trend',
                    line=dict(dash='dash', width=2),
                    opacity=0.7
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Year",
                yaxis_title="Amount",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_comparison_charts(self, data: pd.DataFrame, company_name: str, benchmark_company: str):
        """Render comparison charts between company and benchmark"""
        key_metrics = ['Revenue', 'Net Income', 'Total Assets', 'Operating Cash Flow']
    
        for metric in key_metrics:
            company_rows = [row for row in data.index if metric.lower() in str(row).lower()]
            
            if company_rows:
                metric_row = company_rows[0]
                
                fig = go.Figure()
                
                company_data = []
                benchmark_data = []
                years = []
                
                for col in data.columns:
                    col_str = str(col)
                    if company_name in col_str:
                        if pd.notna(data.loc[metric_row, col]):
                            company_data.append(float(data.loc[metric_row, col]))
                            if '>>' in col_str:
                                parts = col_str.split('>>')
                                if len(parts) >= 4:
                                    year = parts[3].strip()
                                    years.append(year)
                    elif benchmark_company in col_str:
                        if pd.notna(data.loc[metric_row, col]):
                            benchmark_data.append(float(data.loc[metric_row, col]))
                
                if company_data and years:
                    fig.add_trace(go.Bar(
                        x=years,
                        y=company_data,
                        name=company_name,
                        marker_color='blue'
                    ))
                
                if benchmark_data and len(benchmark_data) == len(years):
                    fig.add_trace(go.Bar(
                        x=years,
                        y=benchmark_data,
                        name=benchmark_company,
                        marker_color='orange'
                    ))
                
                fig.update_layout(
                    title=f"{metric} Comparison",
                    xaxis_title="Year",
                    yaxis_title="Value",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    @error_boundary()
    @safe_state_access
    def _render_ratios_tab(self, data: pd.DataFrame):
        """Render financial ratios tab with manual mapping support"""
        st.header("📈 Financial Ratio Analysis")
    
        if not self.get_state('metric_mappings'):
            st.warning("Please map metrics first to calculate ratios")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🤖 Auto-map with AI", type="primary", key="ai_map_ratios"):
                    self._perform_ai_mapping(data)
            
            with col2:
                if st.button("✏️ Manual Mapping", key="manual_map_ratios"):
                    self.set_state('show_manual_mapping', True)
            
            if self.get_state('show_manual_mapping', False):
                manual_mapper = ManualMappingInterface(data)
                mappings = manual_mapper.render()
                
                if st.button("✅ Apply Mappings", type="primary", key="apply_manual_mappings"):
                    self.set_state('metric_mappings', mappings)
                    st.success(f"Applied {len(mappings)} mappings!")
                    self.set_state('show_manual_mapping', False)
            
            return
    
        mappings = self.get_state('metric_mappings')
        mapped_df = data.rename(index=mappings)
    
        with st.spinner("Calculating ratios..."):
            with performance_monitor.measure("ratio_calculation"):
                analysis = self.components['analyzer'].analyze_financial_statements(mapped_df)
                ratios = analysis.get('ratios', {})
    
        if not ratios:
            st.error("Unable to calculate ratios. Please check your mappings.")
            if st.button("🔄 Re-map Metrics"):
                self.set_state('metric_mappings', None)
            return
    
        formatter = get_number_formatter(self.get_state('number_format_value'))
    
        for category, ratio_df in ratios.items():
            if isinstance(ratio_df, pd.DataFrame) and not ratio_df.empty:
                st.subheader(f"{category} Ratios")
                
                format_str = "{:,.2f}"
                
                try:
                    st.dataframe(
                        ratio_df.style.format(format_str, na_rep="-")
                        .background_gradient(cmap='RdYlGn', axis=1),
                        use_container_width=True
                    )
                except Exception as e:
                    self.logger.error(f"Error formatting ratios: {e}")
                    st.dataframe(ratio_df, use_container_width=True)
                
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
                            height=400,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
    @safe_state_access
    def _perform_ai_mapping(self, data: pd.DataFrame):
        """Perform AI mapping of metrics with progress tracking"""
        try:
            with st.spinner("AI is mapping your metrics..."):
                source_metrics = [str(m) for m in data.index.tolist()]
    
                kaggle_available = False
                if 'mapper' in self.components:
                    status = self.components['mapper'].get_api_status()
                    kaggle_available = status['kaggle_available']
                
                if kaggle_available:
                    st.info("🚀 Using Kaggle GPU for faster mapping...")
                
                if 'mapper' in self.components:
                    mapping_result = self.components['mapper'].map_metrics_with_confidence_levels(
                        source_metrics
                    )
                    
                    self.set_state('ai_mapping_result', mapping_result)
                    
                    auto_mappings = mapping_result.get('high_confidence', {})
                    if auto_mappings:
                        final_mappings = {source: data['target'] for source, data in auto_mappings.items()}
                        self.set_state('metric_mappings', final_mappings)
                        
                        st.success(f"✅ AI mapped {len(final_mappings)} metrics with high confidence!")
                        st.info(f"Method: {mapping_result.get('method', 'unknown')}")
                        
                        medium_conf = mapping_result.get('medium_confidence', {})
                        low_conf = mapping_result.get('low_confidence', {})
                        
                        if medium_conf or low_conf:
                            st.info(f"Review {len(medium_conf) + len(low_conf)} additional suggested mappings below")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("High Confidence", len(auto_mappings))
                            with col2:
                                st.metric("Medium Confidence", len(medium_conf))
                            with col3:
                                st.metric("Low Confidence", len(low_conf))
                    else:
                        st.warning("No high-confidence mappings found. Please review suggestions or use manual mapping.")
                        
                else:
                    st.error("AI mapper not available")
                    
        except Exception as e:
            st.error(f"AI mapping failed: {str(e)}")
    
    @error_boundary()
    @safe_state_access
    def _render_trends_tab(self, data: pd.DataFrame):
        """Render trends and analysis tab with comprehensive ML forecasting and safe state handling"""
        st.header("📉 Trend Analysis & ML Forecasting")
    
        self.ensure_state_key('ml_forecast_results', None)
        self.ensure_state_key('forecast_periods', 3)
        self.ensure_state_key('forecast_model_type', 'auto')
        self.ensure_state_key('selected_forecast_metrics', [])
        self.ensure_state_key('forecast_confidence_level', 0.95)
        self.ensure_state_key('show_forecast_details', False)
        self.ensure_state_key('forecast_history', [])
    
        with st.spinner("Analyzing trends..."):
            analysis = self.components['analyzer'].analyze_financial_statements(data)
            trends = analysis.get('trends', {})
    
        if not trends or 'error' in trends:
            st.error("Insufficient data for trend analysis. Need at least 2 years of data.")
            return
    
        st.subheader("📊 Trend Summary")
    
        trend_data = []
        for metric, trend_info in trends.items():
            if isinstance(trend_info, dict) and 'direction' in trend_info:
                trend_data.append({
                    'Metric': metric,
                    'Direction': trend_info['direction'],
                    'CAGR %': trend_info.get('cagr', None),
                    'Volatility %': trend_info.get('volatility', None),
                    'R²': trend_info.get('r_squared', None),
                    'Trend Strength': 'Strong' if trend_info.get('r_squared', 0) > 0.8 else 'Moderate' if trend_info.get('r_squared', 0) > 0.5 else 'Weak'
                })
    
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            
            st.dataframe(
                trend_df.style.format({
                    'CAGR %': '{:.1f}',
                    'Volatility %': '{:.1f}',
                    'R²': '{:.3f}'
                }, na_rep='-')
                .background_gradient(subset=['CAGR %'], cmap='RdYlGn')
                .background_gradient(subset=['R²'], cmap='Blues'),
                use_container_width=True
            )
            
            with st.expander("🔍 Trend Insights"):
                positive_trends = len([t for t in trend_data if t['Direction'] == 'increasing'])
                negative_trends = len([t for t in trend_data if t['Direction'] == 'decreasing'])
                strong_trends = len([t for t in trend_data if t['Trend Strength'] == 'Strong'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive Trends", positive_trends)
                with col2:
                    st.metric("Negative Trends", negative_trends)
                with col3:
                    st.metric("Strong Trends", strong_trends)
                
                if positive_trends > negative_trends:
                    st.success("📈 Overall positive trend momentum detected")
                elif negative_trends > positive_trends:
                    st.warning("📉 Overall negative trend momentum detected")
                else:
                    st.info("⚖️ Mixed trend signals - detailed analysis recommended")
    
        st.subheader("🤖 ML-Powered Forecasting")
    
        if not self.config.get('app.enable_ml_features', True):
            st.warning("ML features are disabled. Enable them in the sidebar settings.")
            return
    
        with st.expander("⚙️ Forecasting Configuration", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                forecast_periods = st.selectbox(
                    "Forecast Periods",
                    [1, 2, 3, 4, 5, 6],
                    index=self.get_state_safe('forecast_periods', 3) - 1,
                    help="Number of future periods to forecast",
                    key="forecast_periods_select"
                )
                self.set_state('forecast_periods', forecast_periods)
            
            with col2:
                model_type = st.selectbox(
                    "Model Type",
                    ['auto', 'linear', 'polynomial', 'exponential'],
                    index=['auto', 'linear', 'polynomial', 'exponential'].index(
                        self.get_state_safe('forecast_model_type', 'auto')
                    ),
                    help="ML model for forecasting",
                    key="forecast_model_select"
                )
                self.set_state('forecast_model_type', model_type)
            
            with col3:
                confidence_level = st.selectbox(
                    "Confidence Level",
                    [0.90, 0.95, 0.99],
                    index=[0.90, 0.95, 0.99].index(
                        self.get_state_safe('forecast_confidence_level', 0.95)
                    ),
                    help="Statistical confidence level for intervals",
                    key="confidence_level_select"
                )
                self.set_state('forecast_confidence_level', confidence_level)
            
            with col4:
                show_details = st.checkbox(
                    "Show Details",
                    value=self.get_state_safe('show_forecast_details', False),
                    help="Show model accuracy and technical details",
                    key="show_forecast_details_check"
                )
                self.set_state('show_forecast_details', show_details)
    
        numeric_metrics = data.select_dtypes(include=[np.number]).index.tolist()
        key_metrics = []
    
        for metric in numeric_metrics:
            metric_lower = str(metric).lower()
            if any(keyword in metric_lower for keyword in ['revenue', 'income', 'profit', 'cash', 'assets', 'equity']):
                key_metrics.append(metric)
    
        default_metrics = key_metrics[:4] if len(key_metrics) >= 4 else numeric_metrics[:4]
    
        selected_metrics = st.multiselect(
            "Select metrics to forecast:",
            numeric_metrics,
            default=self.get_state_safe('selected_forecast_metrics', default_metrics),
            help="Choose which financial metrics to forecast",
            key="forecast_metrics_select"
        )
        self.set_state('selected_forecast_metrics', selected_metrics)
    
        col1, col2, col3 = st.columns([2, 1, 1])
    
        with col1:
            if st.button("🚀 Generate ML Forecast", type="primary", key="generate_forecast_btn"):
                if not selected_metrics:
                    st.error("Please select at least one metric to forecast")
                else:
                    with st.spinner("🧠 Training ML models and generating forecasts..."):
                        try:
                            if not hasattr(self, 'ml_forecaster') or self.ml_forecaster is None:
                                self.ml_forecaster = MLForecaster(self.config)
                            
                            forecast_results = self.ml_forecaster.forecast_metrics(
                                data, 
                                periods=forecast_periods,
                                model_type=model_type,
                                metrics=selected_metrics
                            )
                            
                            self.set_state('ml_forecast_results', forecast_results)
                            
                            forecast_history = self.get_state_safe('forecast_history', [])
                            forecast_entry = {
                                'timestamp': datetime.now().isoformat(),
                                'model_type': model_type,
                                'periods': forecast_periods,
                                'metrics': selected_metrics,
                                'success': 'error' not in forecast_results
                            }
                            forecast_history.append(forecast_entry)
                            self.set_state('forecast_history', forecast_history[-5:])
    
                            st.success("✅ Forecast generated successfully!")
    
                        except Exception as e:
                            st.error(f"❌ Forecasting failed: {str(e)}")
                            if self.config.get('app.debug', False):
                                st.exception(e)
    
        # Display forecast results if available
        forecast_results = self.get_state_safe('ml_forecast_results')
        if forecast_results and 'forecasts' in forecast_results and forecast_results['forecasts']:
            st.markdown("---")
            
            for metric, forecast in forecast_results['forecasts'].items():
                st.subheader(f"Forecast for: {metric}")
                
                # Prepare data for chart
                actual_series = data.loc[metric].dropna()
                history_x = actual_series.index.tolist()
                history_y = actual_series.values.tolist()
                
                forecast_x = forecast['periods']
                forecast_y = forecast['values']
                
                # Get confidence intervals
                intervals = forecast_results['confidence_intervals'].get(metric, {})
                lower_bound = intervals.get('lower', [])
                upper_bound = intervals.get('upper', [])
                
                # Create chart
                fig = go.Figure()
                
                # Actual data
                fig.add_trace(go.Scatter(
                    x=history_x, y=history_y, mode='lines+markers', name='Actual',
                    line=dict(color='blue')
                ))
                
                # Forecast data
                fig.add_trace(go.Scatter(
                    x=forecast_x, y=forecast_y, mode='lines+markers', name='Forecast',
                    line=dict(color='orange', dash='dash')
                ))
                
                # Confidence interval
                if lower_bound and upper_bound:
                    fig.add_trace(go.Scatter(
                        x=forecast_x, y=lower_bound, mode='lines', name='Lower Bound',
                        line=dict(width=0), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast_x, y=upper_bound, mode='lines', name='Upper Bound',
                        fill='tonexty', fillcolor='rgba(255,165,0,0.2)',
                        line=dict(width=0), showlegend=False
                    ))
                
                fig.update_layout(
                    title=f"{metric} Forecast",
                    xaxis_title="Period",
                    yaxis_title="Value",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show accuracy metrics if details are enabled
                if self.get_state_safe('show_forecast_details', False):
                    with st.expander("Model Accuracy Details"):
                        accuracy = forecast_results['accuracy_metrics'].get(metric, {})
                        if accuracy:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Model Used", forecast_results.get('model_type', 'N/A'))
                            with col2:
                                st.metric("RMSE", f"{accuracy.get('rmse', 0):,.2f}")
                            with col3:
                                st.metric("MAE", f"{accuracy.get('mae', 0):,.2f}")
                            with col4:
                                mape = accuracy.get('mape')
                                st.metric("MAPE", f"{mape:.2f}%" if mape else "N/A")
                        else:
                            st.info("Accuracy metrics not available.")
    
    @error_boundary()
    @safe_state_access
    def _render_penman_nissim_tab(self, data: pd.DataFrame):
        """Placeholder for the old Penman-Nissim tab, will call the enhanced one."""
        self._render_penman_nissim_tab_enhanced(data)

    @error_boundary()
    @safe_state_access
    def _render_penman_nissim_tab_enhanced(self, data: pd.DataFrame):
        """Render the enhanced Penman-Nissim tab."""
        st.header("🎯 Penman-Nissim Analysis (Enhanced)")
        
        # Initialize validator
        pn_validator = EnhancedPenmanNissimValidator()
        
        # Check if mappings exist
        mappings = self.get_state('pn_mappings')
        
        # Add COMPLETE emergency fix button BEFORE checking mappings
        if st.button("🚨 Fix All Issues: Apply Complete VST Mappings", type="primary", key="complete_fix_vst_mappings"):
            # Log all available data rows for debugging
            self.logger.info("Available data rows:")
            for idx in data.index[:50]:  # Log first 50 rows
                self.logger.info(f"  {idx}")
            
            # COMPLETE mappings for VST - including ALL P&L items
            complete_vst_mappings = {
                # Balance Sheet - Complete
                'BalanceSheet::Total Equity and Liabilities': 'Total Assets',
                'BalanceSheet::Total Assets': 'Total Assets',
                'BalanceSheet::Total Equity': 'Total Equity',
                'BalanceSheet::Equity': 'Total Equity',
                'BalanceSheet::Total Current Assets': 'Current Assets',
                'BalanceSheet::Total Current Liabilities': 'Current Liabilities',
                'BalanceSheet::Cash and Cash Equivalents': 'Cash and Cash Equivalents',
                'BalanceSheet::Trade receivables': 'Trade Receivables',
                'BalanceSheet::Trade Receivables': 'Trade Receivables',
                'BalanceSheet::Inventories': 'Inventory',
                'BalanceSheet::Fixed Assets': 'Property Plant Equipment',
                'BalanceSheet::Property Plant and Equipment': 'Property Plant Equipment',
                'BalanceSheet::Share Capital': 'Share Capital',
                'BalanceSheet::Other Equity': 'Retained Earnings',
                'BalanceSheet::Trade payables': 'Accounts Payable',
                'BalanceSheet::Trade Payables': 'Accounts Payable',
                'BalanceSheet::Short Term Borrowings': 'Short-term Debt',
                'BalanceSheet::Long Term Borrowings': 'Long-term Debt',
                'BalanceSheet::Other Current Liabilities': 'Accrued Expenses',
                'BalanceSheet::Other Non-Current Liabilities': 'Deferred Revenue',
                
                # P&L - CRITICAL - ALL VARIATIONS
                'ProfitLoss::Revenue From Operations': 'Revenue',
                'ProfitLoss::Revenue From Operations(Net)': 'Revenue',
                'ProfitLoss::Total Revenue': 'Revenue',
                'ProfitLoss::Net Sales': 'Revenue',
                
                'ProfitLoss::Cost of Materials Consumed': 'Cost of Goods Sold',
                'ProfitLoss::Cost of Goods Sold': 'Cost of Goods Sold',
                'ProfitLoss::COGS': 'Cost of Goods Sold',
                
                'ProfitLoss::Profit Before Exceptional Items and Tax': 'Operating Income',
                'ProfitLoss::Operating Profit': 'Operating Income',
                'ProfitLoss::EBIT': 'Operating Income',
                'ProfitLoss::Profit Before Interest and Tax': 'Operating Income',
                
                'ProfitLoss::Profit Before Tax': 'Income Before Tax',
                'ProfitLoss::PBT': 'Income Before Tax',
                
                'ProfitLoss::Tax Expense': 'Tax Expense',
                'ProfitLoss::Tax Expenses': 'Tax Expense',
                'ProfitLoss::Current Tax': 'Tax Expense',
                'ProfitLoss::Total Tax Expense': 'Tax Expense',
                
                'ProfitLoss::Profit After Tax': 'Net Income',
                'ProfitLoss::Profit/Loss For The Period': 'Net Income',
                'ProfitLoss::Net Profit': 'Net Income',
                'ProfitLoss::PAT': 'Net Income',
                
                'ProfitLoss::Finance Cost': 'Interest Expense',
                'ProfitLoss::Finance Costs': 'Interest Expense',
                'ProfitLoss::Interest': 'Interest Expense',
                'ProfitLoss::Interest Expense': 'Interest Expense',
                
                'ProfitLoss::Other Expenses': 'Operating Expenses',
                'ProfitLoss::Employee Benefit Expenses': 'Operating Expenses',
                'ProfitLoss::Operating Expenses': 'Operating Expenses',
                
                'ProfitLoss::Other Income': 'Interest Income',
                'ProfitLoss::Interest Income': 'Interest Income',
                
                # CRITICAL - Depreciation with ALL variations
                'ProfitLoss::Depreciation and Amortisation Expenses': 'Depreciation',
                'ProfitLoss::Depreciation and Amortization Expenses': 'Depreciation',
                'ProfitLoss::Depreciation': 'Depreciation',
                'ProfitLoss::Depreciation and Amortisation': 'Depreciation',
                'ProfitLoss::Depreciation & Amortization': 'Depreciation',
                
                # Cash Flow - Complete
                'CashFlow::Net Cash from Operating Activities': 'Operating Cash Flow',
                'CashFlow::Net CashFlow From Operating Activities': 'Operating Cash Flow',
                'CashFlow::Operating Cash Flow': 'Operating Cash Flow',
                'CashFlow::Cash from Operating Activities': 'Operating Cash Flow',
                
                'CashFlow::Purchase of Fixed Assets': 'Capital Expenditure',
                'CashFlow::Purchased of Fixed Assets': 'Capital Expenditure',
                'CashFlow::Purchase of Investments': 'Capital Expenditure',
                'CashFlow::Capital Expenditure': 'Capital Expenditure',
                'CashFlow::Additions to Fixed Assets': 'Capital Expenditure',
            }
            
            # Apply ALL mappings that exist in data
            valid_mappings = {}
            missing_critical = []
            
            # Track what we're mapping
            mapped_targets = set()
            
            for source, target in complete_vst_mappings.items():
                if source in data.index:
                    valid_mappings[source] = target
                    mapped_targets.add(target)
                    self.logger.info(f"Mapped: {target} <- {source}")
            
            # Check for critical missing mappings
            critical_metrics = [
                'Revenue', 'Operating Income', 'Net Income', 'Tax Expense', 
                'Interest Expense', 'Income Before Tax', 'Total Assets', 
                'Total Equity', 'Operating Cash Flow'
            ]
            
            for metric in critical_metrics:
                if metric not in mapped_targets:
                    missing_critical.append(metric)
                    self.logger.warning(f"Critical metric not mapped: {metric}")
            
            # Log summary
            self.logger.info(f"Total mappings applied: {len(valid_mappings)}")
            self.logger.info(f"Mapped targets: {sorted(mapped_targets)}")
            
            if missing_critical:
                st.warning(f"⚠️ Missing critical metrics: {', '.join(missing_critical)}")
            
            # Apply the mappings
            self.set_state('pn_mappings', valid_mappings)
            st.success(f"✅ Applied {len(valid_mappings)} complete mappings!")
            
            # Show what was mapped
            with st.expander("📋 Applied Mappings", expanded=True):
                # Group by target for better visibility
                mappings_by_target = {}
                for source, target in valid_mappings.items():
                    if target not in mappings_by_target:
                        mappings_by_target[target] = []
                    mappings_by_target[target].append(source)
                
                for target, sources in sorted(mappings_by_target.items()):
                    st.write(f"**{target}**:")
                    for source in sources:
                        st.write(f"  ← {source}")
            
            mappings = valid_mappings
            st.rerun()
        
        # If still no mappings, show the mapping interface
        if not mappings:
            # Use the enhanced mapping interface
            mappings = self._render_enhanced_penman_nissim_mapping(data)
            if not mappings:
                return  # User hasn't completed mapping yet
        
        # Validate mappings with enhanced validator
        validation_result = pn_validator.validate_mapping_for_pn(mappings, data)
        
        # Display validation status
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            quality_color = "green" if validation_result['quality_score'] >= 80 else "orange" if validation_result['quality_score'] >= 60 else "red"
            st.metric(
                "Mapping Quality", 
                f"{validation_result['quality_score']:.0f}%",
                delta=None,
                help="Overall quality score of the mappings"
            )
        
        with col2:
            missing_count = len(validation_result['missing_essential'])
            st.metric(
                "Missing Essential", 
                missing_count,
                delta=None,
                delta_color="inverse",
                help="Number of missing essential metrics"
            )
        
        with col3:
            warnings_count = len(validation_result.get('warnings', []))
            st.metric(
                "Warnings", 
                warnings_count,
                delta=None,
                delta_color="inverse",
                help="Number of potential issues detected"
            )
        
        with col4:
            status = "✅ Valid" if validation_result['is_valid'] else "⚠️ Needs Review"
            st.metric(
                "Status",
                status,
                delta=None,
                help="Overall validation status"
            )
        
        # Show validation details if there are issues
        if not validation_result['is_valid'] and validation_result['quality_score'] < 60:
            with st.expander("⚠️ Validation Details", expanded=True):
                if validation_result['missing_essential']:
                    st.error("Missing Essential Metrics:")
                    for metric in validation_result['missing_essential']:
                        st.write(f"- {metric}")
                
                if validation_result.get('warnings', []):
                    st.warning("Potential Issues:")
                    for warning in validation_result['warnings']:
                        st.write(f"- {warning}")
                
                suggestions = pn_validator.suggest_improvements(validation_result)
                if suggestions:
                    st.info("💡 Suggested Improvements:")
                    for suggestion in suggestions:
                        st.write(f"- {suggestion}")
                
                if st.button("🔧 Reconfigure Mappings", key="pn_reconfig_validation"):
                    self.set_state('pn_mappings', None)
                    st.rerun()
                return
        
        # Perform analysis with better error handling
        with st.spinner("Running enhanced Penman-Nissim analysis..."):
            try:
                analyzer = EnhancedPenmanNissimAnalyzer(data, mappings)
                results = analyzer.calculate_all()
                
                # Log what we got
                self.logger.info(f"Analysis results keys: {list(results.keys())}")
                
                if 'ratios' in results and isinstance(results['ratios'], pd.DataFrame):
                    self.logger.info(f"Ratios shape: {results['ratios'].shape}")
                    self.logger.info(f"Ratios index: {list(results['ratios'].index)[:10]}")
                else:
                    self.logger.warning("No ratios in results or not a DataFrame")
                
                self.set_state('pn_results', results)
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                self.logger.error(f"Penman-Nissim analysis failed: {e}", exc_info=True)
                
                # Show debug info
                with st.expander("🔍 Debug Information"):
                    st.write("**Current Mappings:**")
                    st.json(mappings)
                    st.write("**Error Details:**")
                    st.code(str(e))
                    st.write("**Traceback:**")
                    st.code(traceback.format_exc())
                
                return
        
        if 'error' in results:
            st.error(f"Analysis failed: {results['error']}")
            return
        
        # Add a quality score to the results based on validation
        results['quality_score'] = validation_result['quality_score']
        
        # Create tabs for different views
        tabs = st.tabs([
            "📊 Key Ratios",
            "📈 Trend Analysis", 
            "🔄 Comparison",
            "📑 Reformulated Statements",
            "💰 Cash Flow Analysis",
            "🎯 Value Drivers",
            "📉 Time Series",
            "🔍 Debug",
            "📋 Calculation Trace"  # Add this new tab
        ])
        
        with tabs[0]:
            # Key Ratios Tab
            if 'ratios' in results and isinstance(results['ratios'], pd.DataFrame) and not results['ratios'].empty:
                st.subheader("Penman-Nissim Key Ratios")
                
                # Display latest year metrics
                # Assuming 'results' is a dictionary available in this scope
                ratios_df = results['ratios']
                
                # Display latest year metrics only if the DataFrame is not empty
                # Display latest year metrics only if the DataFrame is not empty
                if not ratios_df.empty and len(ratios_df.columns) > 0:
                    latest_year = ratios_df.columns[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # Check for RNOA
                        if 'Return on Net Operating Assets (RNOA) %' in ratios_df.index:
                            rnoa = ratios_df.loc['Return on Net Operating Assets (RNOA) %', latest_year]
                            # --- NEW: Add a check for NaN ---
                            if pd.notna(rnoa):
                                st.metric("RNOA", f"{rnoa:.1f}%", help="Return on Net Operating Assets")
                            else:
                                st.metric("RNOA", "N/A", help="Value is Not a Number (NaN)")
                        else:
                            st.metric("RNOA", "N/A", help="Not calculated")
                    
                    with col2:
                        # Check for FLEV
                        if 'Financial Leverage (FLEV)' in ratios_df.index:
                            flev = ratios_df.loc['Financial Leverage (FLEV)', latest_year]
                            # --- NEW: Add a check for NaN ---
                            if pd.notna(flev):
                                st.metric("FLEV", f"{flev:.2f}", help="Financial Leverage")
                            else:
                                st.metric("FLEV", "N/A", help="Value is Not a Number (NaN)")
                        else:
                            st.metric("FLEV", "N/A", help="Not calculated")
                    
                    with col3:
                        # Check for NBC
                        if 'Net Borrowing Cost (NBC) %' in ratios_df.index:
                            nbc = ratios_df.loc['Net Borrowing Cost (NBC) %', latest_year]
                            # --- NEW: Add a check for NaN ---
                            if pd.notna(nbc):
                                st.metric("NBC", f"{nbc:.1f}%", help="Net Borrowing Cost")
                            else:
                                st.metric("NBC", "N/A", help="Value is Not a Number (NaN)")
                        else:
                            st.metric("NBC", "N/A", help="Not calculated")
                    
                    with col4:
                        # Check for Spread
                        if 'Spread %' in ratios_df.index:
                            spread = ratios_df.loc['Spread %', latest_year]
                            # --- NEW: Add a check for NaN ---
                            if pd.notna(spread):
                                delta_color = "normal" if spread > 0 else "inverse"
                                st.metric("Spread", f"{spread:.1f}%", delta_color=delta_color, help="RNOA - NBC")
                            else:
                                st.metric("Spread", "N/A", help="Value is Not a Number (NaN)")
                        else:
                            st.metric("Spread", "N/A", help="Not calculated")
                
                # --- NEW: ROE DECOMPOSITION ANALYSIS SECTION ---
                st.markdown("---")
                st.subheader("🔬 ROE Decomposition Analysis")
                st.info("This analysis breaks down Return on Equity (ROE) into its core drivers: Operating Performance (RNOA) and the effect of Financial Leverage.")
                
                # Check if all required components are available for decomposition
                required_components = ['Return on Equity (ROE) %', 'Return on Net Operating Assets (RNOA) %', 'Financial Leverage (FLEV)', 'Spread %']
                if all(comp in ratios_df.index for comp in required_components):
                    
                    # 1. Prepare data for chart and table
                    roe = ratios_df.loc['Return on Equity (ROE) %']
                    rnoa = ratios_df.loc['Return on Net Operating Assets (RNOA) %']
                    
                    # Calculate the leverage effect component
                    flev = ratios_df.loc['Financial Leverage (FLEV)']
                    spread = ratios_df.loc['Spread %']
                    leverage_effect = flev * spread
                    
                    # 2. Create the Decomposition Chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=ratios_df.columns,
                        y=rnoa,
                        name='Operating Return (RNOA)',
                        marker_color='royalblue'
                    ))
                    fig.add_trace(go.Bar(
                        x=ratios_df.columns,
                        y=leverage_effect,
                        name='Leverage Effect (FLEV x Spread)',
                        marker_color='lightslategray'
                    ))
                    # Add the total ROE as a line to show the sum
                    fig.add_trace(go.Scatter(
                        x=ratios_df.columns,
                        y=roe,
                        mode='lines+markers',
                        name='Total ROE',
                        line=dict(color='firebrick', width=3)
                    ))
                
                    fig.update_layout(
                        barmode='relative',  # Stacked bars (positive and negative)
                        title='<b>Drivers of Return on Equity (ROE)</b>',
                        xaxis_title='Year',
                        yaxis_title='Percentage (%)',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode='x unified',
                        height=450
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                    # 3. Display the Data Table
                    with st.expander("View Decomposition Data Table"):
                        decomp_data = {
                            'Total ROE (%)': roe,
                            'Operating Return (RNOA %)': rnoa,
                            'Leverage Effect (%)': leverage_effect,
                            'Financial Leverage (FLEV)': flev,
                            'Spread (%)': spread
                        }
                        decomp_df = pd.DataFrame(decomp_data).T
                        st.dataframe(
                            decomp_df.style.format("{:.2f}", na_rep="-").background_gradient(
                                cmap='RdYlGn', axis=1, subset=pd.IndexSlice[['Total ROE (%)', 'Spread (%)'], :]
                            ),
                            use_container_width=True
                        )
                    
                    # 4. Generate a dynamic key takeaway
                    latest_roe = roe.iloc[-1]
                    latest_rnoa = rnoa.iloc[-1]
                    latest_leverage_effect = leverage_effect.iloc[-1]
                    
                    st.subheader("Key Takeaway")
                    if abs(latest_rnoa) > abs(latest_leverage_effect):
                        primary_driver = "core operations (RNOA)"
                        secondary_driver = "financial leverage"
                    else:
                        primary_driver = "financial leverage"
                        secondary_driver = "core operations (RNOA)"
                    
                    leverage_text = "positively contributing" if latest_leverage_effect > 0 else "negatively impacting"
                    
                    st.success(
                        f"For the latest year, the ROE of **{latest_roe:.2f}%** was primarily driven by **{primary_driver}**, "
                        f"which contributed **{latest_rnoa:.2f}%**. The use of **{secondary_driver}** is "
                        f"**{leverage_text}** to the total return, adding **{latest_leverage_effect:.2f}%**."
                    )
                
                else:
                    st.warning("Could not perform ROE decomposition. Some required ratios (RNOA, FLEV, Spread) are missing.")
                
                # --- END OF NEW SECTION ---
                
                # Display full ratios table (This section is now correctly un-nested)
                st.markdown("### Detailed Ratios Analysis")
                st.dataframe(
                    ratios_df.style.format("{:.2f}", na_rep="-")
                    .background_gradient(cmap='RdYlGn', axis=1),
                    use_container_width=True
                )
                
                # Generate insights (This section is now correctly un-nested)
                insights = self._generate_pn_insights_enhanced(results)
                if insights:
                    st.subheader("💡 Key Insights")
                    for insight in insights:
                        if "✅" in insight or "🚀" in insight:
                            st.success(insight)
                        elif "⚠️" in insight or "❌" in insight:
                            st.warning(insight)
                        else:
                            st.info(insight)
            else:
                st.warning("No ratio data available. Please check your mappings.")
                
                # Show debug info
                with st.expander("🔍 Debug: Why no ratios?"):
                    st.write("**Possible reasons:**")
                    st.write("1. Missing essential P&L mappings (Revenue, Operating Income, etc.)")
                    st.write("2. Data quality issues (all zeros or NaN values)")
                    st.write("3. Calculation errors")
                    
                    if 'ratios' in results:
                        st.write(f"**Ratios object type:** {type(results['ratios'])}")
                        if hasattr(results['ratios'], 'shape'):
                            st.write(f"**Ratios shape:** {results['ratios'].shape}")
                    else:
                        st.write("**No 'ratios' key in results**")
        
        # Implement other tabs...
        with tabs[1]:
            self._render_pn_trend_analysis(results)
        
        with tabs[2]:
            self._render_pn_comparison(results)
        
        with tabs[3]:
            self._render_pn_reformulated_statements(results)
        
        with tabs[4]:
            self._render_pn_cash_flow_analysis(results)
        
        with tabs[5]:
            self._render_pn_value_drivers(results)
        
        with tabs[6]:
            self._render_pn_time_series(results)

        # Add debug tab implementation
        with tabs[7]:
            st.subheader("🔍 Debug Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Mappings Count:**", len(mappings))
                
                # Show mapped targets
                mapped_targets = set(mappings.values())
                st.write("**Mapped Target Metrics:**")
                for target in sorted(mapped_targets):
                    st.write(f"✓ {target}")
            
            with col2:
                # Show missing critical mappings
                critical_metrics = [
                    'Revenue', 'Cost of Goods Sold', 'Operating Income',
                    'Net Income', 'Tax Expense', 'Interest Expense',
                    'Depreciation', 'Income Before Tax'
                ]
                
                missing = [m for m in critical_metrics if m not in mapped_targets]
                if missing:
                    st.write("**Missing Critical Metrics:**")
                    for metric in missing:
                        st.write(f"❌ {metric}")
            
            # Show sample data rows
            with st.expander("Sample Data Rows"):
                st.write("**First 20 rows in your data:**")
                for i, idx in enumerate(data.index[:20]):
                    st.code(f"{i+1}. {idx}")
            
            # Show results structure
            with st.expander("Analysis Results Structure"):
                if 'ratios' in results:
                    st.write("**Ratios:**", type(results['ratios']))
                    if hasattr(results['ratios'], 'shape'):
                        st.write("Shape:", results['ratios'].shape)
                        if not results['ratios'].empty:
                            st.write("Index:", list(results['ratios'].index))
                            st.write("Columns:", list(results['ratios'].columns))

        # Add implementation for the new tab
        with tabs[8]:  # Calculation Trace tab
            st.subheader("📋 Penman-Nissim Calculation Trace")
            
            if st.checkbox("Show detailed calculation logs", key="show_pn_trace"):
                # Read recent logs
                log_file = Path("logs/PenmanNissim.log")
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        # Get last 1000 lines
                        lines = f.readlines()[-1000:]
                        
                        # Filter for PN-specific logs
                        pn_logs = [line for line in lines if any(tag in line for tag in ['[PN-', 'RNOA', 'NOA', 'NOPAT'])]
                        
                        # Display in expandable sections
                        with st.expander("Balance Sheet Reformulation Trace", expanded=False):
                            bs_logs = [log for log in pn_logs if '[PN-BS' in log]
                            st.code('\n'.join(bs_logs))
                        
                        with st.expander("Income Statement Reformulation Trace", expanded=False):
                            is_logs = [log for log in pn_logs if '[PN-IS' in log]
                            st.code('\n'.join(is_logs))
                        
                        with st.expander("Ratio Calculation Trace", expanded=False):
                            ratio_logs = [log for log in pn_logs if '[PN-RATIOS' in log]
                            st.code('\n'.join(ratio_logs))
                        
                        with st.expander("All Calculations", expanded=False):
                            calc_logs = [log for log in pn_logs if '[PN-CALC]' in log]
                            st.code('\n'.join(calc_logs))
                else:
                    st.info("No log file found. Run an analysis to generate logs.")
            
            # Download logs button
            if st.button("Download Full Trace Log", key="download_pn_trace"):
                log_file = Path("logs/PenmanNissim.log")
                if log_file.exists():
                    with open(log_file, 'rb') as f:
                        st.download_button(
                            label="📥 Download Log File",
                            data=f.read(),
                            file_name=f"penman_nissim_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                            mime="text/plain"
                        )
    
             # Show current RNOA calculation
            if 'pn_results' in st.session_state and st.session_state.pn_results:
                results = st.session_state.pn_results
                if 'ratios' in results and isinstance(results['ratios'], pd.DataFrame):
                    st.subheader("Current RNOA Values")
                    ratios_df = results['ratios']
                    if 'Return on Net Operating Assets (RNOA) %' in ratios_df.index:
                        rnoa_values = ratios_df.loc['Return on Net Operating Assets (RNOA) %']
                        
                        # Create comparison table
                        comparison_data = []
                        expected_rnoa = {
                            '201603': 49.54,
                            '201703': 11.72,
                            '201803': 29.13,
                            '201903': 35.85,
                            '202003': 50.51,
                            '202103': 51.43,
                            '202203': 45.96,
                            '202303': 31.70,
                            '202403': 22.41
                        }
                        
                        for year in rnoa_values.index:
                            year_str = str(year)[-6:]  # Extract year part
                            if year_str in expected_rnoa:
                                comparison_data.append({
                                    'Year': year_str,
                                    'Calculated RNOA': f"{rnoa_values[year]:.2f}%",
                                    'Expected RNOA': f"{expected_rnoa[year_str]:.2f}%",
                                    'Difference': f"{abs(rnoa_values[year] - expected_rnoa[year_str]):.2f}%"
                                })
                        
                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                            

    def _render_pn_trend_analysis(self, results):
        """Render trend analysis tab"""
        if 'ratios' in results and isinstance(results['ratios'], pd.DataFrame) and not results['ratios'].empty:
            st.subheader("Trend Analysis")
            
            ratios_df = results['ratios']
            
            # Select key ratios for trend analysis
            key_ratios = [
                'Return on Net Operating Assets (RNOA) %',
                'Financial Leverage (FLEV)',
                'Net Borrowing Cost (NBC) %',
                'Spread %',
                'Operating Profit Margin (OPM) %',
                'Net Operating Asset Turnover (NOAT)'
            ]
            
            for ratio in key_ratios:
                if ratio in ratios_df.index:
                    fig = go.Figure()
                    
                    values = ratios_df.loc[ratio]
                    fig.add_trace(go.Scatter(
                        x=ratios_df.columns,
                        y=values,
                        mode='lines+markers',
                        name=ratio,
                        line=dict(width=3),
                        marker=dict(size=10)
                    ))
                    
                    # Add trend line
                    if len(ratios_df.columns) > 2:
                        x_numeric = list(range(len(ratios_df.columns)))
                        z = np.polyfit(x_numeric, values.values, 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(
                            x=ratios_df.columns,
                            y=p(x_numeric),
                            mode='lines',
                            name='Trend',
                            line=dict(dash='dash', color='gray')
                        ))
                    
                    fig.update_layout(
                        title=ratio,
                        xaxis_title="Year",
                        yaxis_title="Value",
                        height=350,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trend data available")
    
    def _render_pn_comparison(self, results):
        """Render comparison tab"""
        if 'ratios' in results and isinstance(results['ratios'], pd.DataFrame) and not results['ratios'].empty:
            st.subheader("Value Driver Comparison")
            
            ratios_df = results['ratios']
            
            # RNOA decomposition
            if all(metric in ratios_df.index for metric in ['Return on Net Operating Assets (RNOA) %', 
                                                            'Operating Profit Margin (OPM) %',
                                                            'Net Operating Asset Turnover (NOAT)']):
                
                years = ratios_df.columns[-min(3, len(ratios_df.columns)):]
                
                fig = go.Figure()
                
                for year in years:
                    rnoa = ratios_df.loc['Return on Net Operating Assets (RNOA) %', year]
                    opm = ratios_df.loc['Operating Profit Margin (OPM) %', year]
                    noat = ratios_df.loc['Net Operating Asset Turnover (NOAT)', year]
                    
                    fig.add_trace(go.Bar(
                        name=str(year),
                        x=['RNOA %', 'OPM %', 'NOAT'],
                        y=[rnoa, opm, noat],
                        text=[f"{rnoa:.1f}", f"{opm:.1f}", f"{noat:.2f}"],
                        textposition='auto',
                    ))
                
                fig.update_layout(
                    title="RNOA Decomposition Comparison",
                    xaxis_title="Metric",
                    yaxis_title="Value",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No comparison data available")
    
    def _render_pn_reformulated_statements(self, results):
        """Render reformulated statements tab"""
        st.subheader("Reformulated Financial Statements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'reformulated_balance_sheet' in results and isinstance(results['reformulated_balance_sheet'], pd.DataFrame):
                st.write("**Reformulated Balance Sheet**")
                ref_bs = results['reformulated_balance_sheet']
                st.dataframe(
                    ref_bs.style.format("{:,.0f}", na_rep="-"),
                    use_container_width=True
                )
            else:
                st.info("No reformulated balance sheet available")
        
        with col2:
            if 'reformulated_income_statement' in results and isinstance(results['reformulated_income_statement'], pd.DataFrame):
                st.write("**Reformulated Income Statement**")
                ref_is = results['reformulated_income_statement']
                st.dataframe(
                    ref_is.style.format("{:,.0f}", na_rep="-"),
                    use_container_width=True
                )
            else:
                st.info("No reformulated income statement available")
    
    def _render_pn_cash_flow_analysis(self, results):
        """Render cash flow analysis tab"""
        if 'free_cash_flow' in results and isinstance(results['free_cash_flow'], pd.DataFrame):
            st.subheader("Free Cash Flow Analysis")
            
            fcf_df = results['free_cash_flow']
            
            # Display FCF metrics
            if not fcf_df.empty and len(fcf_df.columns) > 0:
                latest_year = fcf_df.columns[-1]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'Operating Cash Flow' in fcf_df.index:
                        ocf = fcf_df.loc['Operating Cash Flow', latest_year]
                        st.metric("Operating Cash Flow", format_indian_number(ocf))
                
                with col2:
                    if 'Free Cash Flow to Firm' in fcf_df.index:
                        fcf = fcf_df.loc['Free Cash Flow to Firm', latest_year]
                        st.metric("Free Cash Flow", format_indian_number(fcf))
                
                with col3:
                    if 'FCF Yield %' in fcf_df.index:
                        fcf_yield = fcf_df.loc['FCF Yield %', latest_year]
                        st.metric("FCF Yield", f"{fcf_yield:.1f}%")
            
            # Display full FCF table
            st.dataframe(
                fcf_df.style.format("{:,.0f}", na_rep="-"),
                use_container_width=True
            )
            
            # FCF trend chart
            if 'Free Cash Flow to Firm' in fcf_df.index:
                fig = go.Figure()
                
                if 'Operating Cash Flow' in fcf_df.index:
                    fig.add_trace(go.Bar(
                        x=fcf_df.columns,
                        y=fcf_df.loc['Operating Cash Flow'],
                        name='Operating Cash Flow',
                        marker_color='green'
                    ))
                
                if 'Capital Expenditure' in fcf_df.index:
                    fig.add_trace(go.Bar(
                        x=fcf_df.columns,
                        y=-fcf_df.loc['Capital Expenditure'],
                        name='Capital Expenditure',
                        marker_color='red'
                    ))
                
                fig.add_trace(go.Scatter(
                    x=fcf_df.columns,
                    y=fcf_df.loc['Free Cash Flow to Firm'],
                    mode='lines+markers',
                    name='Free Cash Flow',
                    line=dict(color='blue', width=3),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title="Free Cash Flow Analysis",
                    xaxis_title="Year",
                    yaxis_title="Cash Flow Components",
                    yaxis2=dict(
                        title="Free Cash Flow",
                        overlaying='y',
                        side='right'
                    ),
                    barmode='relative',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cash flow data available")
    
    def _render_pn_value_drivers(self, results):
        """Render value drivers tab"""
        if 'value_drivers' in results and isinstance(results['value_drivers'], pd.DataFrame):
            st.subheader("Value Drivers Analysis")
            
            drivers_df = results['value_drivers']
            
            # Display value drivers table
            st.dataframe(
                drivers_df.style.format("{:.2f}", na_rep="-")
                .background_gradient(cmap='RdYlGn', axis=1),
                use_container_width=True
            )
            
            # Revenue growth chart
            if 'Revenue Growth %' in drivers_df.index:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=drivers_df.columns,
                    y=drivers_df.loc['Revenue Growth %'],
                    name='Revenue Growth %',
                    marker_color='green',
                    text=[f"{v:.1f}%" for v in drivers_df.loc['Revenue Growth %']],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Revenue Growth Trend",
                    xaxis_title="Year",
                    yaxis_title="Growth %",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No value driver data available")
    
    def _render_pn_time_series(self, results):
        """Render time series analysis tab"""
        st.subheader("Time Series Analysis")
        
        if 'ratios' in results and isinstance(results['ratios'], pd.DataFrame) and not results['ratios'].empty:
            ratios_df = results['ratios']
            
            # Allow selection of multiple metrics
            available_metrics = ratios_df.index.tolist()
            
            selected_metrics = st.multiselect(
                "Select metrics to compare",
                available_metrics,
                default=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics,
                key="pn_ts_metrics_select"
            )
            
            if selected_metrics:
                # Normalize option
                normalize = st.checkbox("Normalize to base 100", key="pn_normalize_check")
                
                fig = go.Figure()
                
                for metric in selected_metrics:
                    values = ratios_df.loc[metric]
                    
                    if normalize:
                        base_value = values.iloc[0]
                        if base_value != 0:
                            normalized_values = (values / base_value) * 100
                        else:
                            normalized_values = values
                        
                        fig.add_trace(go.Scatter(
                            x=ratios_df.columns,
                            y=normalized_values,
                            mode='lines+markers',
                            name=metric
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=ratios_df.columns,
                            y=values,
                            mode='lines+markers',
                            name=metric
                        ))
                
                fig.update_layout(
                    title="Multi-Metric Time Series Comparison",
                    xaxis_title="Year",
                    yaxis_title="Value" + (" (Base 100)" if normalize else ""),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time series data available")
        
    def _debug_show_available_metrics(self, data: pd.DataFrame):
        """Debug helper to show all available metrics for mapping"""
        with st.expander("🔍 Debug: Available Metrics in Your Data", expanded=False):
            st.info("Use these exact names when mapping metrics")
            
            # Group by statement type
            balance_sheet = []
            profit_loss = []
            cash_flow = []
            other = []
            
            for idx in data.index:
                idx_str = str(idx)
                if 'BalanceSheet::' in idx_str:
                    balance_sheet.append(idx_str)
                elif 'ProfitLoss::' in idx_str:
                    profit_loss.append(idx_str)
                elif 'CashFlow::' in idx_str:
                    cash_flow.append(idx_str)
                else:
                    other.append(idx_str)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Balance Sheet Items:**")
                for item in sorted(balance_sheet)[:30]:
                    st.code(item, language=None)
                if len(balance_sheet) > 30:
                    st.text(f"... and {len(balance_sheet) - 30} more")
            
            with col2:
                st.markdown("**P&L Items:**")
                for item in sorted(profit_loss)[:30]:
                    st.code(item, language=None)
                if len(profit_loss) > 30:
                    st.text(f"... and {len(profit_loss) - 30} more")
            
            with col3:
                st.markdown("**Cash Flow Items:**")
                for item in sorted(cash_flow)[:30]:
                    st.code(item, language=None)
                if len(cash_flow) > 30:
                    st.text(f"... and {len(cash_flow) - 30} more")
            
            if other:
                st.markdown("**Other Items:**")
                for item in sorted(other)[:10]:
                    st.code(item, language=None)
                
    #---- render enhanced nissim mapping
    def _render_enhanced_penman_nissim_mapping(self, data: pd.DataFrame) -> Optional[Dict[str, str]]:
        """Enhanced mapping interface with save/load functionality"""
        st.subheader("🎯 Penman-Nissim Mapping Configuration")
        
        # Initialize template manager
        if 'template_manager' not in st.session_state:
            st.session_state.template_manager = MappingTemplateManager()
        
        template_manager = st.session_state.template_manager
        
        # Initialize enhanced mapper
        pn_mapper = EnhancedPenmanNissimMapper()
        
        # Get source metrics
        source_metrics = [str(m) for m in data.index.tolist()]
        # Add debug helper
        if st.checkbox("Show available metrics for debugging", key="show_debug_metrics"):
            self._debug_show_available_metrics(data)
        
        # Template Selection UI
        st.markdown("### 📋 Mapping Templates")
        
        # --- BUG FIX STARTS HERE: Corrected state management for template selection ---
        
        templates = template_manager.get_all_templates()
        
        # Add quick templates to the main list for a simpler UI
        vst_template_name = "VST Industries (Quick Template)"
        template_options = ["🆕 Create New Mapping", "🤖 Auto-Map (Default)", vst_template_name] + list(templates.keys())

        # Use session state to preserve the dropdown selection across reruns
        if 'pn_active_template' not in st.session_state:
            st.session_state.pn_active_template = "🆕 Create New Mapping"

        try:
            default_index = template_options.index(st.session_state.pn_active_template)
        except ValueError:
            default_index = 0 # Default to "New Mapping" if not found

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            selected_template = st.selectbox(
                "Select Mapping Template",
                template_options,
                index=default_index,
                key="pn_template_selector", # Use a static key
                help="Choose a saved template or create a new mapping"
            )

        # Check if the user has changed the selection
        if selected_template != st.session_state.pn_active_template:
            st.session_state.pn_active_template = selected_template
            
            # This is the new logic: Explicitly load the template when the dropdown changes
            if selected_template == "🆕 Create New Mapping":
                st.session_state.temp_pn_mappings = {}
            elif selected_template == "🤖 Auto-Map (Default)":
                mappings, _ = PenmanNissimMappingTemplates.create_smart_mapping(source_metrics, pn_mapper.template)
                st.session_state.temp_pn_mappings = mappings
            elif selected_template == vst_template_name:
                 vst_mappings = {
                    'BalanceSheet::Total Assets': 'Total Assets', 'BalanceSheet::Total Equity and Liabilities': 'Total Assets',
                    'BalanceSheet::Total Current Assets': 'Current Assets', 'BalanceSheet::Cash and Cash Equivalents': 'Cash and Cash Equivalents',
                    'BalanceSheet::Trade receivables': 'Trade Receivables', 'BalanceSheet::Inventories': 'Inventory',
                    'BalanceSheet::Property Plant and Equipment': 'Property Plant Equipment', 'BalanceSheet::Fixed Assets': 'Property Plant Equipment',
                    'BalanceSheet::Total Equity': 'Total Equity', 'BalanceSheet::Equity': 'Total Equity',
                    'BalanceSheet::Share Capital': 'Share Capital', 'BalanceSheet::Other Equity': 'Retained Earnings',
                    'BalanceSheet::Total Current Liabilities': 'Current Liabilities', 'BalanceSheet::Trade payables': 'Accounts Payable',
                    'BalanceSheet::Other Current Liabilities': 'Short-term Debt', 'BalanceSheet::Short Term Borrowings': 'Short-term Debt',
                    'BalanceSheet::Other Non-Current Liabilities': 'Long-term Debt', 'BalanceSheet::Long Term Borrowings': 'Long-term Debt',
                    'ProfitLoss::Revenue From Operations(Net)': 'Revenue', 'ProfitLoss::Revenue From Operations': 'Revenue',
                    'ProfitLoss::Profit Before Tax': 'Income Before Tax', 'ProfitLoss::Tax Expense': 'Tax Expense',
                    'ProfitLoss::Current Tax': 'Tax Expense', 'ProfitLoss::Profit/Loss For The Period': 'Net Income',
                    'ProfitLoss::Profit After Tax': 'Net Income', 'ProfitLoss::Finance Costs': 'Interest Expense',
                    'ProfitLoss::Finance Cost': 'Interest Expense', 'ProfitLoss::Employee Benefit Expenses': 'Operating Expenses',
                    'ProfitLoss::Other Expenses': 'Operating Expenses', 'ProfitLoss::Depreciation and Amortisation Expenses': 'Depreciation',
                    'ProfitLoss::Cost of Materials Consumed': 'Cost of Goods Sold', 'ProfitLoss::Profit Before Exceptional Items and Tax': 'Operating Income',
                    'CashFlow::Net CashFlow From Operating Activities': 'Operating Cash Flow', 'CashFlow::Net Cash from Operating Activities': 'Operating Cash Flow',
                    'CashFlow::Purchase of Investments': 'Capital Expenditure', 'CashFlow::Capital Expenditure': 'Capital Expenditure'
                 }
                 applied_mappings = {source: target for source, target in vst_mappings.items() if source in source_metrics}
                 st.session_state.temp_pn_mappings = applied_mappings
            else:
                loaded_mappings = template_manager.load_template(selected_template)
                st.session_state.temp_pn_mappings = {k: v for k, v in loaded_mappings.items() if k in source_metrics}

            st.rerun() # Rerun to ensure UI updates with the newly loaded template
        
        with col2:
            if st.button("💾 Save Current", key="save_mapping_template", 
                         disabled=('temp_pn_mappings' not in st.session_state)):
                # Show save dialog
                st.session_state.show_save_dialog = True
        
        with col3:
            if selected_template not in ["🆕 Create New Mapping", "🤖 Auto-Map (Default)"] and \
               st.button("🗑️ Delete Template", key="delete_template"):
                if template_manager.delete_template(selected_template):
                    st.success(f"Deleted template: {selected_template}")
                    st.rerun()
                else:
                    st.error("Failed to delete template")
        
        # Save Dialog
        if st.session_state.get('show_save_dialog', False):
            with st.form("save_template_form"):
                st.markdown("### 💾 Save Mapping Template")
                
                template_name = st.text_input(
                    "Template Name",
                    value=f"{self.get_state('company_name', 'Company')}_{datetime.now().strftime('%Y%m%d')}",
                    help="Give your mapping template a memorable name"
                )
                
                template_description = st.text_area(
                    "Description (Optional)",
                    help="Describe when to use this template"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("💾 Save", type="primary"):
                        if template_name and 'temp_pn_mappings' in st.session_state:
                            if template_manager.save_template(
                                name=template_name,
                                mappings=st.session_state.temp_pn_mappings,
                                description=template_description,
                                company=self.get_state('company_name', ''),
                                metadata={
                                    'source_count': len(source_metrics),
                                    'data_structure': str(data.index[:5].tolist())
                                }
                            ):
                                st.success(f"✅ Saved template: {template_name}")
                                st.session_state.show_save_dialog = False
                                st.rerun()
                            else:
                                st.error("Failed to save template")
                
                with col2:
                    if st.form_submit_button("Cancel"):
                        st.session_state.show_save_dialog = False
                        st.rerun()
        
        # Initialize mappings based on template selection
        if 'temp_pn_mappings' not in st.session_state or st.session_state.get('last_template') != selected_template:
            st.session_state.last_template = selected_template
            
            if selected_template == "🆕 Create New Mapping":
                # Start with empty mappings
                st.session_state.temp_pn_mappings = {}
                st.session_state.pn_unmapped = source_metrics
                
            elif selected_template == "🤖 Auto-Map (Default)":
                # Use auto-mapping
                template_mappings, unmapped = PenmanNissimMappingTemplates.create_smart_mapping(
                    source_metrics, 
                    pn_mapper.template
                )
                st.session_state.temp_pn_mappings = template_mappings
                st.session_state.pn_unmapped = unmapped
                
            else:
                # Load saved template
                loaded_mappings = template_manager.load_template(selected_template)
                if loaded_mappings:
                    # Filter mappings to only include current source metrics
                    valid_mappings = {k: v for k, v in loaded_mappings.items() if k in source_metrics}
                    st.session_state.temp_pn_mappings = valid_mappings
                    st.session_state.pn_unmapped = [m for m in source_metrics if m not in valid_mappings]
                    
                    # Show template info
                    template_info = templates[selected_template]
                    st.info(f"""
                    **Template:** {selected_template}  
                    **Created:** {template_info.get('created_at', 'Unknown')[:10]}  
                    **Metrics:** {template_info.get('metrics_count', 0)}  
                    **Description:** {template_info.get('description', 'No description')}
                    """)
                else:
                    st.error("Failed to load template")
                    st.session_state.temp_pn_mappings = {}
                    st.session_state.pn_unmapped = source_metrics
        
        current_mappings = st.session_state.temp_pn_mappings
        unmapped = st.session_state.pn_unmapped
        
        # Validate current mappings
        validation = pn_mapper.validate_mappings(current_mappings)
        
        # Display validation status
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Completeness", f"{validation['completeness']:.0f}%")
        with col2:
            st.metric("Mapped Items", len(current_mappings))
        with col3:
            essential_complete = len(validation['missing_essential']) == 0
            st.metric("Essential", "✅ Complete" if essential_complete else f"❌ {len(validation['missing_essential'])} missing")
        with col4:
            status = "✅ Ready" if validation['is_valid'] else "⚠️ Incomplete"
            st.metric("Status", status)
        
        # Show what's missing
        if validation['missing_essential']:
            st.error(f"**Missing essential items:** {', '.join(validation['missing_essential'])}")
        
        if validation['missing_important']:
            st.warning(f"**Missing important items:** {', '.join(validation['missing_important'])}")
        
        if validation['suggestions']:
            st.info(f"**Suggestions:** {'; '.join(validation['suggestions'])}")
        
        # Quick actions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🤖 Auto-Complete", key="pn_auto_complete", 
                         help="Auto-map remaining unmapped items"):
                # Auto-complete missing mappings
                remaining_mappings, _ = PenmanNissimMappingTemplates.create_smart_mapping(
                    unmapped, 
                    pn_mapper.template
                )
                current_mappings.update(remaining_mappings)
                st.session_state.temp_pn_mappings = current_mappings
                st.success("Auto-completed missing mappings!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Current", key="pn_reset_current"):
                st.session_state.temp_pn_mappings = {}
                st.session_state.pn_unmapped = source_metrics
                st.rerun()
        
        with col3:
            if st.button("📥 Import", key="import_template"):
                st.session_state.show_import_dialog = True
        
        with col4:
            if st.button("📤 Export", key="export_template", 
                         disabled=not current_mappings):
                # Export current mappings
                export_data = {
                    'name': f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'mappings': current_mappings,
                    'created_at': datetime.now().isoformat(),
                    'company': self.get_state('company_name', ''),
                    'version': '1.0'
                }
                
                st.download_button(
                    label="📥 Download Mapping Template",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"pn_mapping_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        # Import Dialog
        if st.session_state.get('show_import_dialog', False):
            uploaded_file = st.file_uploader(
                "Upload Mapping Template (JSON)",
                type=['json'],
                key="template_upload"
            )
            
            if uploaded_file:
                try:
                    template_data = json.loads(uploaded_file.read())
                    if 'mappings' in template_data:
                        # Apply the imported mappings
                        imported_mappings = template_data['mappings']
                        valid_mappings = {k: v for k, v in imported_mappings.items() if k in source_metrics}
                        st.session_state.temp_pn_mappings = valid_mappings
                        st.success(f"Imported {len(valid_mappings)} mappings!")
                        st.session_state.show_import_dialog = False
                        st.rerun()
                    else:
                        st.error("Invalid template format")
                except Exception as e:
                    st.error(f"Error importing template: {e}")
        
        # Helper function to safely calculate selectbox index
        def safe_selectbox_index(current_source, candidates, source_metrics, current_mappings):
            """Safely calculate selectbox index with bounds checking"""
            available_sources = [s for s in source_metrics if s not in current_mappings or s == current_source]
            options = ['(Not mapped)'] + candidates + [s for s in available_sources if s not in candidates]
            
            if current_source and current_source in options:
                return options, options.index(current_source)
            else:
                return options, 0
        
        # Mapping interface with categories
        st.markdown("### 🔧 Metric Mappings")
        tabs = st.tabs(["🔴 Essential Mappings", "🟡 Important Mappings", "🟢 Optional Mappings", "❓ Unmapped Items"])
        
        with tabs[0]:
            st.info("These mappings are **required** for basic Penman-Nissim analysis")
            
            for target in pn_mapper.required_mappings['essential']:
                # Find current mapping
                current_source = None
                for source, mapped_target in current_mappings.items():
                    if mapped_target == target:
                        current_source = source
                        break
                
                # Find best candidates
                candidates = []
                target_patterns = pn_mapper.template.get(target, [target])
                
                for source in source_metrics:
                    source_clean = source.split('::')[-1] if '::' in source else source
                    for pattern in target_patterns:
                        if pattern.lower() in source_clean.lower():
                            candidates.append(source)
                            break
                
                # Get options and index safely
                options, default_index = safe_selectbox_index(current_source, candidates, source_metrics, current_mappings)
                
                selected = st.selectbox(
                    f"**{target}**" + (" ⚠️" if target in validation['missing_essential'] else ""),
                    options,
                    index=default_index,
                    key=f"pn_map_essential_{target}",
                    help=f"Common patterns: {', '.join(target_patterns[:3])}"
                )
                
                # Update mappings immediately when changed
                if selected != '(Not mapped)':
                    # Remove any previous mapping for this target
                    for src, tgt in list(current_mappings.items()):
                        if tgt == target and src != selected:
                            del current_mappings[src]
                    current_mappings[selected] = target
                else:
                    # Remove mapping if user selects '(Not mapped)'
                    for src, tgt in list(current_mappings.items()):
                        if tgt == target:
                            del current_mappings[src]
                
                # Update session state immediately
                st.session_state.temp_pn_mappings = current_mappings
        
        with tabs[1]:
            st.info("These mappings **improve accuracy** but aren't strictly required")
            
            for target in pn_mapper.required_mappings['important']:
                # Find current mapping
                current_source = None
                for source, mapped_target in current_mappings.items():
                    if mapped_target == target:
                        current_source = source
                        break
                
                # Find best candidates
                candidates = []
                target_patterns = pn_mapper.template.get(target, [target])
                
                for source in source_metrics:
                    source_clean = source.split('::')[-1] if '::' in source else source
                    for pattern in target_patterns:
                        if pattern.lower() in source_clean.lower():
                            candidates.append(source)
                            break
                
                # Get options and index safely
                options, default_index = safe_selectbox_index(current_source, candidates, source_metrics, current_mappings)
                
                selected = st.selectbox(
                    f"**{target}**" + (" ⚠️" if target in validation['missing_important'] else ""),
                    options,
                    index=default_index,
                    key=f"pn_map_important_{target}",
                    help=f"Common patterns: {', '.join(target_patterns[:3])}"
                )
                
                # Update mappings immediately when changed
                if selected != '(Not mapped)':
                    # Remove any previous mapping for this target
                    for src, tgt in list(current_mappings.items()):
                        if tgt == target and src != selected:
                            del current_mappings[src]
                    current_mappings[selected] = target
                else:
                    # Remove mapping if user selects '(Not mapped)'
                    for src, tgt in list(current_mappings.items()):
                        if tgt == target:
                            del current_mappings[src]
                
                # Update session state immediately
                st.session_state.temp_pn_mappings = current_mappings
        
        with tabs[2]:
            st.info("These mappings provide **additional insights**")
            
            for target in pn_mapper.required_mappings['optional']:
                # Find current mapping
                current_source = None
                for source, mapped_target in current_mappings.items():
                    if mapped_target == target:
                        current_source = source
                        break
                
                # Find best candidates
                candidates = []
                target_patterns = pn_mapper.template.get(target, [target])
                
                for source in source_metrics:
                    source_clean = source.split('::')[-1] if '::' in source else source
                    for pattern in target_patterns:
                        if pattern.lower() in source_clean.lower():
                            candidates.append(source)
                            break
                
                # Limit candidates for optional mappings
                candidates = candidates[:10]
                
                # Get options and index safely
                options, default_index = safe_selectbox_index(current_source, candidates, source_metrics, current_mappings)
                
                selected = st.selectbox(
                    f"{target}",
                    options,
                    index=default_index,
                    key=f"pn_map_optional_{target}"
                )
                
                # Update mappings immediately when changed
                if selected != '(Not mapped)':
                    # Remove any previous mapping for this target
                    for src, tgt in list(current_mappings.items()):
                        if tgt == target and src != selected:
                            del current_mappings[src]
                    current_mappings[selected] = target
                else:
                    # Remove mapping if user selects '(Not mapped)'
                    for src, tgt in list(current_mappings.items()):
                        if tgt == target:
                            del current_mappings[src]
                
                # Update session state immediately
                st.session_state.temp_pn_mappings = current_mappings
        
        with tabs[3]:
            st.info("Items that couldn't be automatically mapped")
            
            # Show unmapped items that aren't already mapped
            truly_unmapped = [item for item in unmapped if item not in current_mappings]
            
            if truly_unmapped:
                st.write(f"**{len(truly_unmapped)} unmapped items**")
                
                # Group by type
                balance_sheet_items = [item for item in truly_unmapped if 'BalanceSheet::' in item]
                pl_items = [item for item in truly_unmapped if 'ProfitLoss::' in item]
                cf_items = [item for item in truly_unmapped if 'CashFlow::' in item]
                other_items = [item for item in truly_unmapped if item not in balance_sheet_items + pl_items + cf_items]
                
                if balance_sheet_items:
                    with st.expander(f"Balance Sheet Items ({len(balance_sheet_items)})"):
                        for item in balance_sheet_items[:20]:
                            st.text(f"• {item.split('::')[-1]}")
                        if len(balance_sheet_items) > 20:
                            st.text(f"... and {len(balance_sheet_items) - 20} more")
                
                if pl_items:
                    with st.expander(f"P&L Items ({len(pl_items)})"):
                        for item in pl_items[:20]:
                            st.text(f"• {item.split('::')[-1]}")
                        if len(pl_items) > 20:
                            st.text(f"... and {len(pl_items) - 20} more")
                
                if cf_items:
                    with st.expander(f"Cash Flow Items ({len(cf_items)})"):
                        for item in cf_items[:20]:
                            st.text(f"• {item.split('::')[-1]}")
                        if len(cf_items) > 20:
                            st.text(f"... and {len(cf_items) - 20} more")
                
                if other_items:
                    with st.expander(f"Other Items ({len(other_items)})"):
                        for item in other_items[:20]:
                            st.text(f"• {item}")
                        if len(other_items) > 20:
                            st.text(f"... and {len(other_items) - 20} more")
            else:
                st.success("All items have been mapped!")
        
        # Quick Template Buttons
        st.markdown("### ⚡ Quick Templates")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Replace the existing VST template code with this enhanced version
            if st.button("📋 Load VST Template", key="pn_vst_template_quick", 
                         help="Load pre-configured template for VST Industries"):
                
                # VST-specific mappings with CORRECT names from your data
                vst_correct_mappings = {
                    # Balance Sheet mappings (verified from your debug data)
                    'BalanceSheet::Total Equity and Liabilities': 'Total Assets',
                    'BalanceSheet::Total Equity': 'Total Equity',
                    'BalanceSheet::Equity': 'Total Equity',  # Alternative name
                    'BalanceSheet::Total Current Assets': 'Current Assets',
                    'BalanceSheet::Total Current Liabilities': 'Current Liabilities',
                    'BalanceSheet::Cash and Cash Equivalents': 'Cash and Cash Equivalents',
                    'BalanceSheet::Trade Receivables': 'Trade Receivables',
                    'BalanceSheet::Trade receivables': 'Trade Receivables',  # Alternative case
                    'BalanceSheet::Inventories': 'Inventory',
                    'BalanceSheet::Fixed Assets': 'Property Plant Equipment',
                    'BalanceSheet::Property Plant and Equipment': 'Property Plant Equipment',
                    'BalanceSheet::Share Capital': 'Share Capital',
                    'BalanceSheet::Other Equity': 'Retained Earnings',
                    'BalanceSheet::Trade Payables': 'Accounts Payable',
                    'BalanceSheet::Trade payables': 'Accounts Payable',  # Alternative case
                    'BalanceSheet::Short Term Borrowings': 'Short-term Debt',
                    'BalanceSheet::Long Term Borrowings': 'Long-term Debt',
                    'BalanceSheet::Other Current Liabilities': 'Accrued Expenses',
                    'BalanceSheet::Other Non-Current Liabilities': 'Deferred Revenue',
                    
                    # P&L mappings (verified from your debug data)
                    'ProfitLoss::Revenue From Operations': 'Revenue',
                    'ProfitLoss::Revenue From Operations(Net)': 'Revenue',  # Alternative
                    'ProfitLoss::Profit Before Exceptional Items and Tax': 'Operating Income',
                    'ProfitLoss::Profit Before Tax': 'Income Before Tax',
                    'ProfitLoss::Tax Expense': 'Tax Expense',  # Correct - NOT "Tax Expenses"
                    'ProfitLoss::Current Tax': 'Tax Expense',  # Alternative
                    'ProfitLoss::Profit After Tax': 'Net Income',
                    'ProfitLoss::Profit/Loss For The Period': 'Net Income',  # Alternative
                    'ProfitLoss::Finance Cost': 'Interest Expense',
                    'ProfitLoss::Finance Costs': 'Interest Expense',  # Alternative
                    'ProfitLoss::Interest': 'Interest Expense',  # Alternative
                    'ProfitLoss::Cost of Materials Consumed': 'Cost of Goods Sold',
                    'ProfitLoss::Employee Benefit Expenses': 'Operating Expenses',
                    'ProfitLoss::Other Expenses': 'Operating Expenses',
                    'ProfitLoss::Depreciation and Amortisation Expenses': 'Depreciation',
                    'ProfitLoss::Other Income': 'Interest Income',
                    
                    # Cash Flow mappings (verified from your debug data)
                    'CashFlow::Net Cash from Operating Activities': 'Operating Cash Flow',
                    'CashFlow::Net CashFlow From Operating Activities': 'Operating Cash Flow',  # Alternative
                    'CashFlow::Purchase of Fixed Assets': 'Capital Expenditure',  # CORRECT - not "Purchased"
                    'CashFlow::Purchased of Fixed Assets': 'Capital Expenditure',  # Handle typo
                    'CashFlow::Purchase of Investments': 'Capital Expenditure',  # Alternative
                    'CashFlow::Capital Expenditure': 'Capital Expenditure',  # Direct match
                }
                
                # Apply only mappings that match current data
                applied_mappings = {}
                for source in source_metrics:
                    # Try exact match first
                    if source in vst_correct_mappings:
                        applied_mappings[source] = vst_correct_mappings[source]
                    else:
                        # Try to find in mappings by partial match
                        source_clean = source.split('::')[-1] if '::' in source else source
                        for vst_key, target in vst_correct_mappings.items():
                            vst_clean = vst_key.split('::')[-1] if '::' in vst_key else vst_key
                            if source_clean.lower() == vst_clean.lower():
                                applied_mappings[source] = target
                                break
                
                st.session_state.temp_pn_mappings = applied_mappings
                st.success(f"Loaded VST Industries template with {len(applied_mappings)} mappings!")
                
                # Show what was mapped
                with st.expander("View Applied Mappings"):
                    for source, target in sorted(applied_mappings.items(), key=lambda x: x[1]):
                        st.text(f"{target} ← {source}")
                
                st.rerun()

        # Add this after the VST template button in _render_enhanced_penman_nissim_mapping:

        if st.button("🚨 Force Add Capital Expenditure", key="force_add_capex", type="primary"):
            """Emergency button to manually add Capital Expenditure mapping"""
            
            # Check if we have investing cash flow
            investing_items = [idx for idx in data.index if 'investing' in str(idx).lower()]
            
            if investing_items:
                # Use the Net Cash Used in Investing Activities as a proxy for CapEx
                investing_item = investing_items[0]
                
                # Add to mappings
                current_mappings = st.session_state.get('temp_pn_mappings', {})
                current_mappings[investing_item] = 'Capital Expenditure'
                st.session_state.temp_pn_mappings = current_mappings
                
                st.success(f"✅ Mapped '{investing_item}' to Capital Expenditure as a proxy")
                st.info("Note: This uses total investing cash flow as CapEx. For more accurate analysis, please upload detailed cash flow statements.")
                st.rerun()
            else:
                st.error("No investing activities found in the data")

        # <<<--- START: PASTE THE NEW RECOVERY BUTTON CODE HERE ---<<<
        if st.button("🔧 Recover Lost Cash Flow Items", key="recover_cf_items", type="secondary"):
            """Emergency recovery for lost cash flow items"""
            with st.spinner("Searching for cash flow items..."):
                # Get the original unprocessed data if available
                original_files = self.get_state('uploaded_files', [])
                
                if original_files:
                    st.info("Re-processing files to recover cash flow items...")
                    
                    # Reprocess with cash flow preservation
                    recovered_items = []
                    
                    for file in original_files:
                        try:
                            # Parse file again
                            df = self._parse_single_file(file)
                            if df is not None:
                                # Look for cash flow items
                                for idx in df.index:
                                    idx_lower = str(idx).lower()
                                    if any(kw in idx_lower for kw in ['cash', 'flow', 'capex', 'capital', 'purchase', 'fixed asset']):
                                        # Check if this item exists in current data
                                        prefixed_idx = f"CashFlow::{idx}"
                                        if prefixed_idx not in data.index and idx not in data.index:
                                            recovered_items.append((idx, df.loc[idx]))
                                            
                        except Exception as e:
                            self.logger.error(f"Error recovering from {file.name}: {e}")
                            continue
                    
                    if recovered_items:
                        st.success(f"Found {len(recovered_items)} missing cash flow items!")
                        
                        # Show recovered items
                        with st.expander("Recovered Cash Flow Items", expanded=True):
                            for item, series in recovered_items[:10]:
                                st.write(f"**{item}**")
                                non_null = series.notna().sum()
                                st.write(f"  Non-null values: {non_null}")
                                if non_null > 0:
                                    st.write(f"  Sample: {series.dropna().head(3).to_dict()}")
                                st.write("---")
                        
                        # Option to re-process with preservation
                        if st.button("Re-process with Cash Flow Preservation", key="reprocess_with_cf"):
                            # Clear existing data
                            self.set_state('analysis_data', None)
                            self.set_state('metric_mappings', None)
                            self.set_state('pn_mappings', None)
                            
                            # Re-process files
                            self._process_uploaded_files(original_files)
                            st.rerun()
                    else:
                        st.warning("No additional cash flow items found in source files")
                else:
                    st.error("No uploaded files found for recovery")
        # <<<--- END: PASTE THE NEW RECOVERY BUTTON CODE HERE ---<<<
        
        with col2:
            if st.button("💼 Clear All", key="pn_clear_all"):
                st.session_state.temp_pn_mappings = {}
                st.session_state.pn_unmapped = source_metrics
                st.success("Cleared all mappings!")
                st.rerun()
        
        with col3:
            if st.button("🔍 Validate", key="pn_validate"):
                final_validation = pn_mapper.validate_mappings(current_mappings)
                if final_validation['is_valid']:
                    st.success("✅ Mappings are valid and ready!")
                else:
                    st.error("❌ Mappings need attention")
        
        with col4:
            if st.button("❓ Help", key="pn_help"):
                with st.expander("Mapping Help", expanded=True):
                    st.markdown("""
                    **How to use this mapping interface:**
                    
                    1. **Select a Template**: Choose from saved templates or start fresh
                    2. **Map Essential Items**: These are required for analysis
                    3. **Map Important Items**: Improve accuracy
                    4. **Review Optional Items**: Add for extra insights
                    5. **Save Your Mapping**: Save for future use
                    6. **Apply**: Apply mappings to proceed with analysis
                    
                    **Tips:**
                    - Use Auto-Complete to quickly map remaining items
                    - Save templates for specific data sources
                    - Export/Import templates to share with others
                    """)
        
        # Apply button with save reminder
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col2:
            if st.button("✅ Apply Mappings", type="primary", key="pn_apply_mappings", 
                         use_container_width=True):
                # Final validation
                final_validation = pn_mapper.validate_mappings(current_mappings)
                
                if final_validation['is_valid']:
                    self.set_state('pn_mappings', current_mappings)
                    
                    # Ask if user wants to save as template
                    if selected_template in ["🆕 Create New Mapping", "🤖 Auto-Map (Default)"]:
                        st.info("💡 Tip: Click '💾 Save Current' to save this mapping for future use!")
                    
                    # Clean up temp state
                    if 'temp_pn_mappings' in st.session_state:
                        del st.session_state.temp_pn_mappings
                    if 'pn_unmapped' in st.session_state:
                        del st.session_state.pn_unmapped
                    if 'last_template' in st.session_state:
                        del st.session_state.last_template
                    if 'show_save_dialog' in st.session_state:
                        del st.session_state.show_save_dialog
                    if 'show_import_dialog' in st.session_state:
                        del st.session_state.show_import_dialog
                    
                    st.success(f"✅ Applied {len(current_mappings)} mappings successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Please complete all essential mappings before proceeding")
                    if final_validation['missing_essential']:
                        st.write("**Missing essential items:**")
                        for item in final_validation['missing_essential']:
                            st.write(f"• {item}")
        # PASTE THIS CODE: Add this as a temporary debug section in your mapping interface
        # PASTE THIS CODE: Add this to your _render_enhanced_penman_nissim_mapping method 
        # (insert after the "Apply Mappings" button section)
        
        # Auto-detection section
        if st.button("🔍 Auto-Detect Missing Mappings", key="auto_detect_mappings"):
            with st.spinner("Analyzing data for missing mappings..."):
                # Create temporary analyzer to get suggestions
                temp_analyzer = EnhancedPenmanNissimAnalyzer(data, current_mappings or {})
                suggestions = temp_analyzer.suggest_missing_mappings()
                
                if suggestions:
                    st.success(f"Found suggestions for {len(suggestions)} missing metrics!")
                    
                    with st.expander("📋 Mapping Suggestions", expanded=True):
                        for metric, candidates in suggestions.items():
                            st.write(f"**{metric}:**")
                            
                            if candidates:
                                # Create columns for each suggestion
                                cols = st.columns(min(len(candidates), 3))
                                
                                for i, (candidate, confidence) in enumerate(candidates):
                                    if i < 3:  # Show max 3 suggestions
                                        with cols[i]:
                                            confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🟠"
                                            st.write(f"{confidence_color} **{confidence:.0%}** confidence")
                                            st.code(candidate.split('::')[-1] if '::' in candidate else candidate)
                                            
                                            if st.button(f"Use This", key=f"use_{metric}_{i}"):
                                                # Add to mappings
                                                current_mappings[candidate] = metric
                                                st.session_state.temp_pn_mappings = current_mappings
                                                st.success(f"Added mapping: {metric}")
                                                st.rerun()
                            else:
                                st.info("No good candidates found")
                else:
                    st.info("All required mappings appear to be complete!")
        
        # Show current CapEx detection results
        if st.checkbox("🔧 Show CapEx Detection Details", key="show_capex_details"):
            with st.expander("Capital Expenditure Detection Analysis", expanded=True):
                temp_analyzer = EnhancedPenmanNissimAnalyzer(data, current_mappings or {})
                capex_candidates = temp_analyzer.detect_capex_candidates()
                
                if capex_candidates:
                    st.write("**Potential Capital Expenditure items found:**")
                    
                    capex_df = pd.DataFrame(capex_candidates, columns=['Item', 'Confidence'])
                    capex_df['Confidence'] = capex_df['Confidence'].apply(lambda x: f"{x:.0%}")
                    capex_df['Short Name'] = capex_df['Item'].apply(
                        lambda x: x.split('::')[-1] if '::' in x else x
                    )
                    
                    st.dataframe(
                        capex_df[['Short Name', 'Confidence', 'Item']], 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Quick mapping buttons
                    st.write("**Quick Actions:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if capex_candidates and st.button("✅ Use Highest Confidence", key="use_best_capex"):
                            best_candidate = capex_candidates[0][0]
                            current_mappings[best_candidate] = 'Capital Expenditure'
                            st.session_state.temp_pn_mappings = current_mappings
                            st.success(f"Mapped Capital Expenditure to: {best_candidate}")
                            st.rerun()
                    
                    with col2:
                        if st.button("📋 Show All Cash Flow Items", key="show_all_cf"):
                            cf_items = [idx for idx in data.index if 'cashflow::' in str(idx).lower()]
                            st.write("**All Cash Flow items in your data:**")
                            for item in cf_items:
                                st.code(item)
                else:
                    st.warning("No Capital Expenditure candidates found in cash flow data.")
                    
                    # Show what cash flow items are available
                    cf_items = [idx for idx in data.index if 'cashflow::' in str(idx).lower()]
                    if cf_items:
                        st.write("**Available Cash Flow items:**")
                        for item in cf_items[:10]:
                            st.code(item)
                        if len(cf_items) > 10:
                            st.write(f"... and {len(cf_items) - 10} more items")
                    else:
                        st.error("No cash flow items found in the data!")
        if st.checkbox("🐛 Debug: Show My Cash Flow Data", key="debug_cashflow"):
            st.write("**All items in your data that contain 'cash' or 'flow':**")
            
            cash_flow_items = []
            for idx in data.index:
                idx_lower = str(idx).lower()
                if any(keyword in idx_lower for keyword in ['cash', 'flow', 'purchase', 'capex', 'expenditure']):
                    cash_flow_items.append(idx)
            
            if cash_flow_items:
                for item in cash_flow_items:
                    # Show the item and a sample of its data
                    sample_data = data.loc[item].dropna()
                    if len(sample_data) > 0:
                        st.write(f"**{item}**")
                        st.write(f"Sample values: {sample_data.head(3).to_dict()}")
                        st.write("---")
            else:
                st.error("No cash flow related items found!")
                
                # Show ALL indices for debugging
                st.write("**First 50 items in your data:**")
                for i, idx in enumerate(data.index[:50]):
                    st.code(f"{i+1}. {idx}")

        # PASTE THIS CODE: Add this debugging section to your mapping interface

        # In _render_enhanced_penman_nissim_mapping method, replace the debug section:

        if st.checkbox("🔍 Debug Data Transfer Issues", key="debug_data_transfer"):
            st.subheader("Data Transfer Debugging")
            
            # FIX: Get current mappings from session state, not from undefined variable
            current_mappings_debug = st.session_state.get('temp_pn_mappings', {})
            
            # Show original vs restructured data comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Data Structure:**")
                st.write(f"Shape: {data.shape}")
                
                # Look for CapEx in original
                capex_items = [idx for idx in data.index if any(kw in str(idx).lower() 
                               for kw in ['capex', 'capital', 'purchase', 'fixed asset', 'expenditure'])]
                
                st.write(f"**Found {len(capex_items)} potential CapEx items:**")
                for item in capex_items[:10]:
                    st.code(item)
                    sample_data = data.loc[item].dropna()
                    if len(sample_data) > 0:
                        st.write(f"Sample values: {dict(list(sample_data.items())[:3])}")
                    else:
                        st.write("⚠️ No data found")
                    st.write("---")
            
            with col2:
                st.write("**After Restructuring:**")
                
                if st.button("🔄 Test Restructuring", key="test_restructure"):
                    with st.spinner("Testing data restructuring..."):
                        try:
                            # FIX: Use the current_mappings_debug variable
                            temp_analyzer = EnhancedPenmanNissimAnalyzer(data, current_mappings_debug)
                            clean_data = temp_analyzer._df_clean
                            
                            st.write(f"Clean data shape: {clean_data.shape}")
                             # Check what happened to CapEx items
                            capex_in_clean = []
                            for item in capex_items:
                                if item in clean_data.index:
                                    series = clean_data.loc[item]
                                    non_null_count = series.notna().sum()
                                    capex_in_clean.append((item, non_null_count))
                            
                            st.write(f"**CapEx items preserved: {len(capex_in_clean)}/{len(capex_items)}**")
                            
                            for item, count in capex_in_clean[:5]:
                                st.write(f"✅ {item}: {count} values")
                                if count > 0:
                                    sample_clean = clean_data.loc[item].dropna()
                                    st.write(f"   Values: {dict(list(sample_clean.items())[:3])}")
                            
                            # Show missing items
                            missing_items = [item for item in capex_items if item not in clean_data.index]
                            if missing_items:
                                st.write(f"**❌ Missing items ({len(missing_items)}):**")
                                for item in missing_items[:3]:
                                    st.code(item)
                            
                        except Exception as e:
                            st.error(f"Restructuring test failed: {e}")
                            st.code(traceback.format_exc())
        
            # Column structure analysis
            st.subheader("Column Structure Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Columns:**")
                for i, col in enumerate(data.columns[:10]):
                    st.code(f"{i+1}. {col}")
                if len(data.columns) > 10:
                    st.write(f"... and {len(data.columns) - 10} more columns")
            
            with col2:
                st.write("**Year Pattern Analysis:**")
                
                # Analyze year patterns in columns
                year_patterns = [
                    (re.compile(r'(\d{6})'), 'YYYYMM (e.g., 202003)'),
                    (re.compile(r'(\d{4})(?!\d)'), 'YYYY (e.g., 2020)'),
                    (re.compile(r'FY\s*(\d{4})'), 'FY YYYY (e.g., FY 2020)'),
                    (re.compile(r'(\d{4})-(\d{2})'), 'YYYY-YY (e.g., 2020-21)'),
                ]
                
                pattern_matches = {}
                for col in data.columns:
                    col_str = str(col)
                    for pattern, name in year_patterns:
                        if pattern.search(col_str):
                            if name not in pattern_matches:
                                pattern_matches[name] = []
                            pattern_matches[name].append(col)
                            break
                
                for pattern_name, matches in pattern_matches.items():
                    st.write(f"**{pattern_name}:** {len(matches)} columns")
                    for match in matches[:3]:
                        st.code(match)
                    if len(matches) > 3:
                        st.write(f"... and {len(matches) - 3} more")
        
            # Data preservation test
            st.subheader("Data Preservation Test")
            
            if st.button("🧪 Run Full Data Preservation Test", key="preservation_test"):
                with st.spinner("Running comprehensive data preservation test..."):
                    
                    # Test with a known CapEx item
                    if capex_items:
                        test_item = capex_items[0]
                        st.write(f"**Testing with: {test_item}**")
                        
                        # Original data
                        original_series = data.loc[test_item]
                        original_non_null = original_series.notna().sum()
                        
                        st.write(f"Original non-null values: {original_non_null}")
                        st.write(f"Original data: {original_series.dropna().to_dict()}")
                        
                        # After restructuring
                        try:
                            temp_analyzer = EnhancedPenmanNissimAnalyzer(data, mappings or {})
                            clean_data = temp_analyzer._df_clean
                            
                            if test_item in clean_data.index:
                                clean_series = clean_data.loc[test_item]
                                clean_non_null = clean_series.notna().sum()
                                
                                st.write(f"After restructuring non-null values: {clean_non_null}")
                                st.write(f"Clean data: {clean_series.dropna().to_dict()}")
                                
                                # Calculate preservation rate
                                preservation_rate = (clean_non_null / original_non_null * 100) if original_non_null > 0 else 0
                                
                                if preservation_rate >= 90:
                                    st.success(f"✅ Excellent preservation: {preservation_rate:.1f}%")
                                elif preservation_rate >= 70:
                                    st.warning(f"⚠️ Good preservation: {preservation_rate:.1f}%")
                                else:
                                    st.error(f"❌ Poor preservation: {preservation_rate:.1f}%")
                                    
                                    # Analyze what went wrong
                                    st.write("**Debugging data loss:**")
                                    
                                    # Check column mapping
                                    st.write("**Column Analysis:**")
                                    for col in data.columns:
                                        val = data.loc[test_item, col]
                                        if pd.notna(val):
                                            st.write(f"  {col}: {val}")
                            else:
                                st.error(f"❌ Item {test_item} completely lost during restructuring!")
                                
                        except Exception as e:
                            st.error(f"Test failed: {e}")
                            st.code(traceback.format_exc())
                    else:
                        st.warning("No CapEx items found to test with")
            # PASTE THIS CODE: Add this emergency fix button to your mapping interface

            st.subheader("🚨 Emergency Data Recovery")
            
            if st.button("🔧 Force Preserve All CapEx Data", key="force_preserve_capex", type="primary"):
                with st.spinner("Forcing data preservation..."):
                    
                    # Find all potential CapEx items
                    capex_candidates = []
                    for idx in data.index:
                        idx_lower = str(idx).lower()
                        if any(kw in idx_lower for kw in ['capex', 'capital expenditure', 'purchase', 'fixed asset', 'expenditure']):
                            capex_candidates.append(idx)
                    
                    if capex_candidates:
                        st.write(f"Found {len(capex_candidates)} CapEx candidates:")
                        
                        # Show them with data
                        for candidate in capex_candidates:
                            with st.expander(f"📊 {candidate}"):
                                series = data.loc[candidate]
                                non_null_data = series.dropna()
                                
                                if len(non_null_data) > 0:
                                    st.write(f"**Data points:** {len(non_null_data)}")
                                    st.write(f"**Values:** {non_null_data.to_dict()}")
                                    
                                    # Quick mapping button
                                    if st.button(f"Map as Capital Expenditure", key=f"map_capex_{candidate}"):
                                        current_mappings = st.session_state.get('temp_pn_mappings', {})
                                        current_mappings[candidate] = 'Capital Expenditure'
                                        st.session_state.temp_pn_mappings = current_mappings
                                        st.success(f"✅ Mapped {candidate} to Capital Expenditure")
                                        st.rerun()
                                else:
                                    st.warning("No data found in this item")
                    else:
                        st.error("No CapEx candidates found!")
                        
                        # Show all cash flow items as backup
                        cf_items = [idx for idx in data.index if 'cashflow::' in str(idx).lower()]
                        if cf_items:
                            st.write("**All Cash Flow items:**")
                            for item in cf_items:
                                st.code(item)
        
        # Return None to indicate mapping is not complete yet
        return None
                        
    def _generate_pn_insights_enhanced(self, results: Dict[str, Any]) -> List[str]:
        """Generate enhanced insights from Penman-Nissim analysis"""
        insights = []
        
        if 'ratios' not in results or results['ratios'].empty:
            return ["Unable to generate insights due to insufficient data."]
        
        ratios = results['ratios']
        
        # RNOA Analysis
        if 'Return on Net Operating Assets (RNOA) %' in ratios.index:
            rnoa_series = ratios.loc['Return on Net Operating Assets (RNOA) %']
            latest_rnoa = rnoa_series.iloc[-1]
            avg_rnoa = rnoa_series.mean()
            
            if latest_rnoa > 20:
                insights.append(f"✅ Excellent operating performance with RNOA of {latest_rnoa:.1f}% (Elite level)")
            elif latest_rnoa > 15:
                insights.append(f"✅ Strong operating performance with RNOA of {latest_rnoa:.1f}%")
            elif latest_rnoa > 10:
                insights.append(f"💡 Moderate operating performance with RNOA of {latest_rnoa:.1f}%")
            else:
                insights.append(f"⚠️ Low operating performance with RNOA of {latest_rnoa:.1f}%")
            
            # RNOA trend
            if len(rnoa_series) > 1:
                trend = "improving" if rnoa_series.iloc[-1] > rnoa_series.iloc[0] else "declining"
                insights.append(f"📊 RNOA trend is {trend} over the analysis period")
        
        # Spread Analysis
        if 'Spread %' in ratios.index:
            spread_series = ratios.loc['Spread %']
            latest_spread = spread_series.iloc[-1]
            
            if latest_spread > 5:
                insights.append(f"🚀 Strong positive spread ({latest_spread:.1f}%) - Financial leverage is creating significant value")
            elif latest_spread > 0:
                insights.append(f"✅ Positive spread ({latest_spread:.1f}%) - Financial leverage is value accretive")
            else:
                insights.append(f"❌ Negative spread ({latest_spread:.1f}%) - Financial leverage is destroying value")
        
        # Financial Leverage Analysis
        if 'Financial Leverage (FLEV)' in ratios.index:
            flev_series = ratios.loc['Financial Leverage (FLEV)']
            latest_flev = flev_series.iloc[-1]
            
            if latest_flev > 2:
                insights.append(f"⚠️ High financial leverage ({latest_flev:.2f}) indicates significant financial risk")
            elif latest_flev < 0:
                insights.append(f"💡 Negative leverage ({latest_flev:.2f}) indicates net financial assets position")
        
        # OPM and NOAT Analysis
        if 'Operating Profit Margin (OPM) %' in ratios.index and 'Net Operating Asset Turnover (NOAT)' in ratios.index:
            opm = ratios.loc['Operating Profit Margin (OPM) %'].iloc[-1]
            noat = ratios.loc['Net Operating Asset Turnover (NOAT)'].iloc[-1]
            
            if opm > 15 and noat > 2:
                insights.append(f"✅ Excellent combination of profitability (OPM: {opm:.1f}%) and efficiency (NOAT: {noat:.2f})")
            elif opm < 5:
                insights.append(f"⚠️ Low operating margin ({opm:.1f}%) suggests pricing or cost challenges")
            elif noat < 1:
                insights.append(f"⚠️ Low asset turnover ({noat:.2f}) indicates asset utilization issues")
        
        return insights
    
    #render side
    def _render_quick_template_buttons(self, source_metrics, current_mappings):
        """Render quick template buttons for common formats"""
        st.markdown("### ⚡ Quick Templates")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Capitaline", key="quick_capitaline"):
                # Capitaline-specific mappings
                capitaline_mappings = {
                    'BalanceSheet::Total Assets': 'Total Assets',
                    'BalanceSheet::Current Assets': 'Current Assets',
                    'BalanceSheet::Cash and Bank Balances': 'Cash and Cash Equivalents',
                    'BalanceSheet::Inventories': 'Inventory',
                    'BalanceSheet::Trade Receivables': 'Trade Receivables',
                    'BalanceSheet::Property Plant and Equipment': 'Property Plant Equipment',
                    'BalanceSheet::Total Equity': 'Total Equity',
                    'BalanceSheet::Share Capital': 'Share Capital',
                    'BalanceSheet::Reserves and Surplus': 'Retained Earnings',
                    'BalanceSheet::Total Current Liabilities': 'Current Liabilities',
                    'BalanceSheet::Trade Payables': 'Accounts Payable',
                    'BalanceSheet::Short Term Borrowings': 'Short-term Debt',
                    'BalanceSheet::Long Term Borrowings': 'Long-term Debt',
                    'ProfitLoss::Total Revenue': 'Revenue',
                    'ProfitLoss::Cost of Materials Consumed': 'Cost of Goods Sold',
                    'ProfitLoss::Operating Profit': 'Operating Income',
                    'ProfitLoss::Profit Before Tax': 'Income Before Tax',
                    'ProfitLoss::Tax Expense': 'Tax Expense',
                    'ProfitLoss::Net Profit': 'Net Income',
                    'ProfitLoss::Finance Costs': 'Interest Expense',
                    'ProfitLoss::Depreciation': 'Depreciation',
                    'CashFlow::Operating Cash Flow': 'Operating Cash Flow',
                    'CashFlow::Capital Expenditure': 'Capital Expenditure'
                }
                current_mappings.update(capitaline_mappings)
                st.success("Applied Capitaline template!")
        
        with col2:
            if st.button("💼 MoneyControl", key="quick_moneycontrol"):
                # MoneyControl-specific mappings
                moneycontrol_mappings = {
                    'BalanceSheet::TOTAL ASSETS': 'Total Assets',
                    'BalanceSheet::Current Assets': 'Current Assets',
                    'BalanceSheet::Cash And Cash Equivalents': 'Cash and Cash Equivalents',
                    'BalanceSheet::Inventory': 'Inventory',
                    'BalanceSheet::Trade Receivables': 'Trade Receivables',
                    'BalanceSheet::Fixed Assets': 'Property Plant Equipment',
                    'BalanceSheet::Total Shareholders Funds': 'Total Equity',
                    'BalanceSheet::Equity Share Capital': 'Share Capital',
                    'BalanceSheet::Reserves And Surplus': 'Retained Earnings',
                    'BalanceSheet::Current Liabilities': 'Current Liabilities',
                    'BalanceSheet::Trade Payables': 'Accounts Payable',
                    'BalanceSheet::Short Term Borrowings': 'Short-term Debt',
                    'BalanceSheet::Long Term Borrowings': 'Long-term Debt',
                    'ProfitLoss::Revenue From Operations': 'Revenue',
                    'ProfitLoss::Cost Of Materials Consumed': 'Cost of Goods Sold',
                    'ProfitLoss::Operating Profit': 'Operating Income',
                    'ProfitLoss::Profit Before Tax': 'Income Before Tax',
                    'ProfitLoss::Tax Expenses': 'Tax Expense',
                    'ProfitLoss::Profit After Tax': 'Net Income',
                    'ProfitLoss::Interest': 'Interest Expense',
                    'ProfitLoss::Depreciation And Amortisation': 'Depreciation',
                    'CashFlow::Cash From Operating Activities': 'Operating Cash Flow',
                    'CashFlow::Purchase Of Fixed Assets': 'Capital Expenditure'
                }
                current_mappings.update(moneycontrol_mappings)
                st.success("Applied MoneyControl template!")
        
        with col3:
            if st.button("🏢 NSE/BSE", key="quick_nse_bse"):
                # NSE/BSE-specific mappings
                nse_bse_mappings = {
                    'BalanceSheet::Total Assets': 'Total Assets',
                    'BalanceSheet::Current assets': 'Current Assets',
                    'BalanceSheet::Cash and bank balances': 'Cash and Cash Equivalents',
                    'BalanceSheet::Inventories': 'Inventory',
                    'BalanceSheet::Trade receivables': 'Trade Receivables',
                    'BalanceSheet::Property, plant and equipment': 'Property Plant Equipment',
                    'BalanceSheet::Total equity': 'Total Equity',
                    'BalanceSheet::Share capital': 'Share Capital',
                    'BalanceSheet::Other equity': 'Retained Earnings',
                    'BalanceSheet::Current liabilities': 'Current Liabilities',
                    'BalanceSheet::Trade payables': 'Accounts Payable',
                    'BalanceSheet::Current borrowings': 'Short-term Debt',
                    'BalanceSheet::Non-current borrowings': 'Long-term Debt',
                    'ProfitLoss::Revenue from operations': 'Revenue',
                    'ProfitLoss::Cost of materials consumed': 'Cost of Goods Sold',
                    'ProfitLoss::Operating profit': 'Operating Income',
                    'ProfitLoss::Profit before tax': 'Income Before Tax',
                    'ProfitLoss::Tax expense': 'Tax Expense',
                    'ProfitLoss::Profit for the period': 'Net Income',
                    'ProfitLoss::Finance costs': 'Interest Expense',
                    'ProfitLoss::Depreciation and amortisation': 'Depreciation',
                    'CashFlow::Net cash from operating activities': 'Operating Cash Flow',
                    'CashFlow::Purchase of fixed assets': 'Capital Expenditure'
                }
                current_mappings.update(nse_bse_mappings)
                st.success("Applied NSE/BSE template!")
        
        with col4:
            if st.button("🏦 Custom Bank", key="quick_bank"):
                # Bank-specific mappings
                bank_mappings = {
                    'BalanceSheet::Total Assets': 'Total Assets',
                    'BalanceSheet::Current Assets': 'Current Assets',
                    'BalanceSheet::Cash and Balances': 'Cash and Cash Equivalents',
                    'BalanceSheet::Loans and Advances': 'Trade Receivables',
                    'BalanceSheet::Fixed Assets': 'Property Plant Equipment',
                    'BalanceSheet::Total Shareholders Funds': 'Total Equity',
                    'BalanceSheet::Share Capital': 'Share Capital',
                    'BalanceSheet::Reserves': 'Retained Earnings',
                    'BalanceSheet::Current Liabilities': 'Current Liabilities',
                    'BalanceSheet::Deposits': 'Accounts Payable',
                    'BalanceSheet::Borrowings': 'Short-term Debt',
                    'BalanceSheet::Long Term Borrowings': 'Long-term Debt',
                    'ProfitLoss::Interest Income': 'Revenue',
                    'ProfitLoss::Interest Expended': 'Cost of Goods Sold',
                    'ProfitLoss::Operating Profit': 'Operating Income',
                    'ProfitLoss::Profit Before Tax': 'Income Before Tax',
                    'ProfitLoss::Tax': 'Tax Expense',
                    'ProfitLoss::Net Profit': 'Net Income',
                    'ProfitLoss::Interest Expense': 'Interest Expense',
                    'ProfitLoss::Depreciation': 'Depreciation',
                    'CashFlow::Operating Activities': 'Operating Cash Flow',
                    'CashFlow::Purchase of Fixed Assets': 'Capital Expenditure'
                }
                current_mappings.update(bank_mappings)
                st.success("Applied Bank template!")
                
    
    @error_boundary()
    @safe_state_access
    def _render_industry_tab(self, data: pd.DataFrame):
        """Render industry comparison tab"""
        
        # Create a single container to ensure single render
        industry_container = st.container()
        
        with industry_container:
            st.header("🏭 Industry Comparison")
            st.info("Compare your company's performance against industry benchmarks.")
            
            # Generate a unique suffix for this render
            unique_suffix = str(uuid.uuid4())[:8]
            
            col1, col2 = st.columns(2)
            with col1:
                selected_industry = st.selectbox(
                    "Select Industry",
                    list(CoreIndustryBenchmarks.BENCHMARKS.keys()),
                    index=0,
                    key=f"industry_comparison_industry_select_{unique_suffix}"
                )
    
            with col2:
                analysis_year = st.selectbox(
                    "Select Year for Analysis",
                    data.columns.tolist(),
                    index=len(data.columns)-1,
                    key=f"industry_comparison_year_select_{unique_suffix}"
                )
    
            # Calculate necessary ratios for comparison
            mappings = self.get_state('pn_mappings')
            if not mappings:
                st.warning("Please configure Penman-Nissim mappings first for accurate industry comparison.")
                return
                
            analyzer = EnhancedPenmanNissimAnalyzer(data, mappings)
            results = analyzer.calculate_all()
    
            if 'ratios' not in results or 'error' in results['ratios']:
                st.error("Could not calculate required ratios for comparison.")
                return
    
            ratios = results['ratios']
    
            # Get company's metrics for the selected year
            company_metrics = {
                'RNOA': ratios.loc['Return on Net Operating Assets (RNOA) %', analysis_year],
                'OPM': ratios.loc['Operating Profit Margin (OPM) %', analysis_year],
                'NOAT': ratios.loc['Net Operating Asset Turnover (NOAT)', analysis_year],
                'NBC': ratios.loc['Net Borrowing Cost (NBC) %', analysis_year],
                'FLEV': ratios.loc['Financial Leverage (FLEV)', analysis_year],
            }
    
            # Calculate composite score
            score_data = CoreIndustryBenchmarks.calculate_composite_score(company_metrics, selected_industry)
    
            st.subheader("Performance Scorecard")
    
            if 'error' in score_data:
                st.error(score_data['error'])
                return
    
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Composite Score", f"{score_data['composite_score']:.1f}/100")
            with col2:
                st.metric("Performance Level", score_data['interpretation'])
    
            # Detailed metric comparison
            st.subheader("Detailed Metric Comparison")
    
            for metric, value in company_metrics.items():
                if not pd.isna(value):
                    benchmark = CoreIndustryBenchmarks.BENCHMARKS[selected_industry][metric]
                    percentile = score_data['metric_scores'][metric]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**{metric}**")
                    
                    with col2:
                        st.metric("Company Value", f"{value:.2f}")
                    
                    with col3:
                        st.metric("Industry Average", f"{benchmark['mean']:.2f}")
                    
                    with col4:
                        st.metric("Industry Percentile", f"{percentile:.0f}th")
                    
                    # Gauge chart for percentile
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = percentile,
                        title = {'text': f"{metric} Percentile"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps' : [
                                {'range': [0, 25], 'color': "red"},
                                {'range': [25, 75], 'color': "yellow"},
                                {'range': [75, 100], 'color': "green"}],
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
    
    @error_boundary()
    @safe_state_access
    def _render_data_explorer_tab(self, data: pd.DataFrame):
        """Render data explorer tab"""
        st.header("🔍 Data Explorer")
        st.info("Explore the processed financial data.")
    
        # Data filtering options
        with st.expander("Filter Data"):
            selected_metrics = st.multiselect(
                "Select Metrics",
                data.index.tolist(),
                default=data.index[:10].tolist(),
                key="explorer_metric_select"
            )
            
            selected_years = st.multiselect(
                "Select Years",
                data.columns.tolist(),
                default=data.columns.tolist(),
                key="explorer_year_select"
            )
            
            if selected_metrics and selected_years:
                filtered_df = data.loc[selected_metrics, selected_years]
            else:
                filtered_df = data
    
        st.dataframe(filtered_df, use_container_width=True)
    
        # Data download
        csv = filtered_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='financial_data.csv',
            mime='text/csv',
        )
    
    @error_boundary()
    @safe_state_access
    def _render_reports_tab(self, data: pd.DataFrame):
        """Render reports tab"""
        st.header("📄 Reports & Export")
    
        analysis = self.components['analyzer'].analyze_financial_statements(data)
        analysis['company_name'] = self.get_state('company_name', 'Financial Analysis')
        analysis['filtered_data'] = data  # Add raw data for export
    
        col1, col2 = st.columns(2)
        with col1:
            report_format = st.selectbox(
                "Select Report Format",
                ["Excel", "Markdown"],
                key="report_format_select"
            )
    
        if report_format == "Excel":
            excel_data = self.export_manager.export_to_excel(analysis)
            st.download_button(
                label="Download Excel Report",
                data=excel_data,
                file_name=f"{analysis['company_name']}_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif report_format == "Markdown":
            md_data = self.export_manager.export_to_markdown(analysis)
            st.text_area("Markdown Report", md_data, height=400)
            st.download_button(
                label="Download Markdown Report",
                data=md_data,
                file_name=f"{analysis['company_name']}_analysis.md",
                mime="text/markdown"
            )
    
    @error_boundary()
    @safe_state_access
    def _render_ml_insights_tab(self, data: pd.DataFrame):
        """Render ML Insights tab"""
        st.header("🤖 ML-Powered Insights")
        st.info("This section provides advanced insights generated by our machine learning models.")
    
        # Placeholder for future ML insights
        st.warning("More advanced ML insights are coming soon!")
    
        # Example: Anomaly contribution analysis
        with st.expander("Anomaly Contribution Analysis (Example)"):
            st.write("This analysis identifies which metrics contributed most to anomalies detected in the data.")
            st.image("https://i.imgur.com/example.png", caption="Example Anomaly Contribution Chart")
    
    def _render_debug_footer(self):
        """Render debug footer"""
        st.markdown("---")
        st.subheader("🐛 Debug Information")
    
        with st.expander("Show Debug Info"):
            # Performance summary
            st.write("**Performance Summary:**")
            perf_summary = performance_monitor.get_performance_summary()
            if perf_summary:
                st.json(perf_summary)
            
            # API summary
            st.write("**API Summary:**")
            api_summary = performance_monitor.get_api_summary()
            if api_summary:
                st.json(api_summary)
                
            # Session state
            st.write("**Session State:**")
            st.json({k: str(v)[:200] for k, v in st.session_state.items()})
    
    def _auto_recovery_attempt(self) -> bool:
        """Attempt automatic recovery from a critical failure"""
        try:
            self.logger.info("Attempting automatic recovery...")
    
            # Re-initialize session state
            self._initialize_session_state()
            
            # Re-initialize components
            self.components = self._initialize_components()
            st.session_state['components'] = self.components
            
            self.logger.info("Recovery attempt completed.")
            return True
        except Exception as e:
            self.logger.error(f"Auto recovery failed: {e}")
            return False
    
    def _perform_health_check(self) -> Dict[str, Any]:
        """Perform a system-wide health check."""
        health_status = {'overall': True, 'checks': {}}
    
        # Check components
        if hasattr(self, 'components') and self.components:
            for name, comp in self.components.items():
                is_init = hasattr(comp, '_initialized') and comp._initialized
                health_status['checks'][f'component_{name}'] = is_init
                if not is_init:
                    health_status['overall'] = False
        else:
            health_status['checks']['components'] = False
            health_status['overall'] = False
            
        # Check Kaggle API if enabled
        if self.config.get('ai.use_kaggle_api'):
            if 'mapper' in self.components:
                api_status = self.components['mapper'].get_api_status()
                health_status['checks']['kaggle_api'] = api_status['kaggle_available']
                if not api_status['kaggle_available']:
                    health_status['overall'] = False
    
        return health_status
    
    def _render_trend_metrics(self, ratios_df, selected_years):
        """Helper to render trend metrics view."""
        st.info("Trend view for key ratios.")
        key_ratios = ['Return on Net Operating Assets (RNOA) %', 
                      'Financial Leverage (FLEV)', 
                      'Leverage Spread %']
        for ratio in key_ratios:
            if ratio in ratios_df.index:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=selected_years,
                    y=ratios_df.loc[ratio, selected_years],
                    mode='lines+markers',
                    name=ratio
                ))
                fig.update_layout(title=ratio, height=250, margin=dict(t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_comparison_metrics(self, ratios_df, selected_years):
        """Helper to render comparison metrics view."""
        st.info("Comparison of key value drivers over time.")
        rnoa = ratios_df.loc['Return on Net Operating Assets (RNOA) %', selected_years]
        spread = ratios_df.loc['Leverage Spread %', selected_years]
    
        fig = go.Figure()
        fig.add_trace(go.Bar(x=selected_years, y=rnoa, name='RNOA'))
        fig.add_trace(go.Bar(x=selected_years, y=spread, name='Leverage Spread'))
        fig.update_layout(title='RNOA vs. Leverage Spread', barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_time_series_analysis(self, results):
        """Helper to render time series analysis tab."""
        st.info("Detailed time-series charts for all calculated ratios.")
        # Add implementation here
        st.write("Time series analysis visualizations will be here.")
    
    def _render_reformulated_statements(self, results):
        """Helper to render reformulated statements tab."""
        if 'reformulated_balance_sheet' in results:
            st.subheader("Reformulated Balance Sheet")
            st.dataframe(results['reformulated_balance_sheet'])
        if 'reformulated_income_statement' in results:
            st.subheader("Reformulated Income Statement")
            st.dataframe(results['reformulated_income_statement'])
    
    def _render_cashflow_analysis(self, results):
        """Helper to render cash flow analysis tab."""
        if 'free_cash_flow' in results:
            st.subheader("Free Cash Flow Analysis")
            st.dataframe(results['free_cash_flow'])
    
            # Chart FCF
            fcf_df = results['free_cash_flow']
            if 'Free Cash Flow' in fcf_df.index:
                fig = px.bar(fcf_df.T, y='Free Cash Flow', title='Free Cash Flow Trend')
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_value_drivers_analysis(self, results):
        """Helper to render value drivers tab."""
        if 'value_drivers' in results:
            st.subheader("Value Drivers")
            st.dataframe(results['value_drivers'])
    
    def _generate_pn_insights_enhanced(self, results):
        """Generate enhanced Penman-Nissim insights."""
        insights = []
        ratios = results.get('ratios')
        if ratios is None or ratios.empty:
            return ["Ratios not calculated, cannot generate insights."]
    
        # RNOA analysis
        if 'Return on Net Operating Assets (RNOA) %' in ratios.index:
            rnoa_series = ratios.loc['Return on Net Operating Assets (RNOA) %']
            avg_rnoa = rnoa_series.mean()
            if avg_rnoa > 20:
                insights.append(f"✅ Elite operating performance with an average RNOA of {avg_rnoa:.1f}%.")
            elif avg_rnoa > 10:
                insights.append(f"💡 Solid operating performance with an average RNOA of {avg_rnoa:.1f}%.")
            else:
                insights.append(f"⚠️ Operating performance needs improvement, average RNOA is {avg_rnoa:.1f}%.")
    
        # Leverage Spread analysis
        if 'Leverage Spread %' in ratios.index:
            spread_series = ratios.loc['Leverage Spread %']
            avg_spread = spread_series.mean()
            if avg_spread > 2:
                insights.append(f"🚀 Financial leverage is effectively creating value (Avg Spread: {avg_spread:.1f}%).")
            elif avg_spread < 0:
                insights.append(f"❌ Financial leverage is destroying value (Avg Spread: {avg_spread:.1f}%). Consider deleveraging.")
    
        return insights
    
    def _render_categorized_insights(self, insights):
        """Render insights with categorization."""
        for insight in insights:
            if "✅" in insight or "🚀" in insight:
                st.success(insight)
            elif "💡" in insight:
                st.info(insight)
            elif "⚠️" in insight or "❌" in insight:
                st.warning(insight)
            else:
                st.write(insight)
    
# --- 31. Application Entry Point ---
def main():
    """Main application entry point with comprehensive error handling"""
    try:
        # Set page config only once at the very beginning
        st.set_page_config(
            page_title="Elite Financial Analytics Platform v5.1",
            page_icon="💹",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize app state
        if 'app_instance' not in st.session_state:
            st.session_state.app_instance = None
            
        # Create app instance if needed
        if st.session_state.app_instance is None:
            try:
                st.session_state.app_instance = FinancialAnalyticsPlatform()
            except Exception as e:
                st.error(f"Failed to initialize application: {str(e)}")
                if st.button("🔄 Retry Initialization"):
                    st.session_state.clear()
                    st.rerun()
                return
        
        # Run the app
        st.session_state.app_instance.run()
            
    except Exception as e:
        # Check for specific Streamlit errors
        error_msg = str(e)
        if "can only be called once per app" in error_msg:
            # Page config already set, ignore and continue
            if st.session_state.app_instance is None:
                st.session_state.app_instance = FinancialAnalyticsPlatform()
            st.session_state.app_instance.run()
        else:
            # Critical error handling
            logging.critical(f"Fatal application error: {e}", exc_info=True)
            
            st.error("🚨 A critical error occurred.")
            
            # Show debug info if available
            if st.session_state.get('show_debug_info', False):
                st.exception(e)
                
                with st.expander("🔧 Debug Information"):
                    st.write("**Error Details:**")
                    st.code(traceback.format_exc())
                    
                    st.write("**Session State Keys:**")
                    st.json(list(st.session_state.keys()))
            
            # Recovery options
            st.subheader("🔄 Recovery Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Refresh Page", key="refresh_page_btn"):
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Cache", key="clear_cache_btn"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Cache cleared!")
            
            with col3:
                if st.button("🏠 Reset Application", key="reset_app_btn"):
                    # Clear all session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

if __name__ == "__main__":
    # Configure Python path and environment
    import sys
    from pathlib import Path
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Run the application
    main()
    
                
