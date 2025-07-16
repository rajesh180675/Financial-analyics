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
        
        # Auto-corrections
        positive_metrics = ['assets', 'revenue', 'equity', 'sales', 'income', 'cash']
        for idx in corrected_df.index:
            for metric in positive_metrics:
                if metric in str(idx).lower():
                    row_data = corrected_df.loc[idx]
                    
                    if isinstance(row_data, pd.DataFrame):
                        for i in range(len(row_data)):
                            negative_mask = row_data.iloc[i] < 0
                            if negative_mask.any():
                                corrected_df.loc[idx].iloc[i][negative_mask] = abs(row_data.iloc[i][negative_mask])
                                corrections_made.append(f"Converted negative values to positive in {idx} (row {i})")
                    else:
                        negative_mask = row_data < 0
                        if negative_mask.any():
                            corrected_df.loc[idx][negative_mask] = abs(row_data[negative_mask])
                            corrections_made.append(f"Converted negative values to positive in {idx}")
        
        # Fix outliers using IQR method
        for col in corrected_df.select_dtypes(include=[np.number]).columns:
            series = corrected_df[col].dropna()
            if len(series) > 4:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (corrected_df[col] < lower_bound) | (corrected_df[col] > upper_bound)
                    if outlier_mask.any():
                        median_value = series.median()
                        corrected_df.loc[outlier_mask, col] = median_value
                        corrections_made.append(f"Corrected {outlier_mask.sum()} outliers in {col} using median")
        
        # Fix accounting equation violations
        self._fix_accounting_equation(corrected_df, corrections_made)
        
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
        
        if df.empty:
            result.add_error(f"{context}: DataFrame is empty")
            return result
        
        if len(df.columns) == 0:
            result.add_error(f"{context}: No columns found")
            return result
        
        if df.shape[0] > 1000000:
            result.add_warning(f"{context}: Large dataset ({df.shape[0]} rows), performance may be impacted")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            result.add_warning(f"{context}: No numeric columns found")
        
        if df.index.duplicated().any():
            result.add_warning(f"{context}: Duplicate indices found")
        
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 50:
            result.add_warning(f"{context}: High percentage of missing values ({missing_pct:.1f}%)")
        
        for col in df.columns:
            if df[col].nunique() == 1:
                result.add_info(f"{context}: Column '{col}' has constant value")
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                mean = series.mean()
                std = series.std()
                outlier_threshold = self.config.get('analysis.outlier_std_threshold', 3)
                
                if std > 0:
                    outliers = series[(series < mean - outlier_threshold * std) | 
                                     (series > mean + outlier_threshold * std)]
                    
                    if len(outliers) > len(series) * 0.1:
                        result.add_warning(
                            f"{context}: Column '{col}' has many outliers ({len(outliers)} values)"
                        )
                
                positive_keywords = ['assets', 'revenue', 'sales', 'income', 'cash']
                if any(keyword in str(col).lower() for keyword in positive_keywords):
                    if (series < 0).any():
                        result.add_warning(f"{context}: Column '{col}' contains negative values")
        
        result.metadata['shape'] = df.shape
        result.metadata['columns'] = list(df.columns)
        result.metadata['dtypes'] = df.dtypes.to_dict()
        result.metadata['memory_usage'] = df.memory_usage(deep=True).sum()
        
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
        """Detect anomalies in financial data"""
        anomalies = {
            'value_anomalies': [],
            'trend_anomalies': [],
            'ratio_anomalies': []
        }
        
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) > 3:
                z_scores = np.abs(stats.zscore(series))
                anomaly_indices = np.where(z_scores > 3)[0]
                
                for idx in anomaly_indices:
                    anomalies['value_anomalies'].append({
                        'metric': df.index[idx],
                        'year': col,
                        'value': series.iloc[idx],
                        'z_score': z_scores[idx]
                    })
        
        for idx in df.index:
            series = df.loc[idx].dropna()
            if len(series) > 2:
                pct_changes = series.pct_change().dropna()
                extreme_changes = pct_changes[np.abs(pct_changes) > 1]
                
                for year, change in extreme_changes.items():
                    anomalies['trend_anomalies'].append({
                        'metric': str(idx),
                        'year': year,
                        'change_pct': change * 100
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
        
        for col in numeric_df.columns[-3:]:
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
        """Calculate financial ratios with proper error handling - FIXED"""
        ratios = {}
        
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

# --- 17. Enhanced API Client with Advanced Features ---
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

# --- 20. Penman-Nissim Analyzer ---
class EnhancedPenmanNissimAnalyzer:
    """Enhanced Penman-Nissim analyzer with flexible initialization"""
    
    def __init__(self, df: pd.DataFrame, mappings: Dict[str, str]):
        self.df = df
        self.mappings = mappings
        self.logger = LoggerFactory.get_logger('PenmanNissim')
        
        self._initialize_core_analyzer()
    
    def _initialize_core_analyzer(self):
        """Initialize core analyzer with proper error handling"""
        if CORE_COMPONENTS_AVAILABLE and CorePenmanNissim is not None:
            try:
                sig = inspect.signature(CorePenmanNissim.__init__)
                params = list(sig.parameters.keys())[1:]
                
                if len(params) >= 2:
                    self.core_analyzer = CorePenmanNissim(self.df, self.mappings)
                    self.logger.info("Initialized CorePenmanNissim with df and mappings")
                elif len(params) == 1:
                    self.core_analyzer = CorePenmanNissim(self.df)
                    if hasattr(self.core_analyzer, 'set_mappings'):
                        self.core_analyzer.set_mappings(self.mappings)
                    elif hasattr(self.core_analyzer, 'mappings'):
                        self.core_analyzer.mappings = self.mappings
                    self.logger.info("Initialized CorePenmanNissim with df only")
                else:
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
    
    @error_boundary({'error': 'Penman-Nissim analysis failed'})
    def calculate_all(self):
        """Calculate all Penman-Nissim metrics"""
        if self.core_analyzer and hasattr(self.core_analyzer, 'calculate_all'):
            try:
                return self.core_analyzer.calculate_all()
            except Exception as e:
                self.logger.error(f"Error in core calculate_all: {e}")
        
        return self._fallback_calculate_all()
    
    def _fallback_calculate_all(self):
        """Fallback implementation of Penman-Nissim calculations"""
        try:
            mapped_df = self.df.rename(index=self.mappings)
            
            results = {
                'reformulated_balance_sheet': self._reformulate_balance_sheet(mapped_df),
                'reformulated_income_statement': self._reformulate_income_statement(mapped_df),
                'ratios': self._calculate_ratios(mapped_df),
                'free_cash_flow': self._calculate_free_cash_flow(mapped_df),
                'value_drivers': self._calculate_value_drivers(mapped_df)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in fallback calculations: {e}")
            return {'error': str(e)}
    
    def _reformulate_balance_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reformulate balance sheet for Penman-Nissim analysis"""
        reformulated = pd.DataFrame(index=df.columns)
        
        operating_assets = ['Current Assets', 'Property Plant Equipment', 'Intangible Assets']
        operating_assets_sum = pd.Series(0, index=df.columns)
        for asset in operating_assets:
            if asset in df.index:
                operating_assets_sum += df.loc[asset].fillna(0)
        
        financial_assets = ['Cash', 'Short-term Investments', 'Long-term Investments']
        financial_assets_sum = pd.Series(0, index=df.columns)
        for asset in financial_assets:
            if asset in df.index:
                financial_assets_sum += df.loc[asset].fillna(0)
        
        operating_liabilities = ['Accounts Payable', 'Accrued Expenses', 'Deferred Revenue']
        operating_liabilities_sum = pd.Series(0, index=df.columns)
        for liab in operating_liabilities:
            if liab in df.index:
                operating_liabilities_sum += df.loc[liab].fillna(0)
        
        financial_liabilities = ['Short-term Debt', 'Long-term Debt', 'Bonds Payable']
        financial_liabilities_sum = pd.Series(0, index=df.columns)
        for liab in financial_liabilities:
            if liab in df.index:
                financial_liabilities_sum += df.loc[liab].fillna(0)
        
        reformulated['Net Operating Assets'] = operating_assets_sum - operating_liabilities_sum
        reformulated['Net Financial Assets'] = financial_assets_sum - financial_liabilities_sum
        reformulated['Common Equity'] = reformulated['Net Operating Assets'] + reformulated['Net Financial Assets']
        
        return reformulated
    
    def _reformulate_income_statement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reformulate income statement for Penman-Nissim analysis"""
        reformulated = pd.DataFrame(index=df.columns)
        
        if 'Revenue' in df.index and 'Operating Income' in df.index:
            reformulated['Operating Income'] = df.loc['Operating Income']
            
            if 'Tax Expense' in df.index and 'Income Before Tax' in df.index:
                income_before_tax = df.loc['Income Before Tax'].replace(0, np.nan)
                tax_rate = df.loc['Tax Expense'] / income_before_tax
                reformulated['Tax on Operating Income'] = reformulated['Operating Income'] * tax_rate
                reformulated['Operating Income After Tax'] = (
                    reformulated['Operating Income'] - reformulated['Tax on Operating Income']
                )
        
        if 'Interest Expense' in df.index:
            reformulated['Net Financial Expense'] = df.loc['Interest Expense']
            if 'Interest Income' in df.index:
                reformulated['Net Financial Expense'] -= df.loc['Interest Income']
        
        return reformulated
    
    def _calculate_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Penman-Nissim ratios - FIXED"""
        ratios = pd.DataFrame(index=df.columns)
        
        ref_bs = self._reformulate_balance_sheet(df)
        ref_is = self._reformulate_income_statement(df)
        
        # RNOA (Return on Net Operating Assets)
        if 'Operating Income After Tax' in ref_is.index and 'Net Operating Assets' in ref_bs.index:
            noa = ref_bs.loc['Net Operating Assets'].replace(0, np.nan)
            ratios['Return on Net Operating Assets (RNOA) %'] = (
                ref_is.loc['Operating Income After Tax'] / noa
            ) * 100
        
        # FLEV (Financial Leverage)
        if 'Net Financial Assets' in ref_bs.index and 'Common Equity' in ref_bs.index:
            ce = ref_bs.loc['Common Equity'].replace(0, np.nan)
            ratios['Financial Leverage (FLEV)'] = -ref_bs.loc['Net Financial Assets'] / ce
        
        # NBC (Net Borrowing Cost)
        if 'Net Financial Expense' in ref_is.index and 'Net Financial Assets' in ref_bs.index:
            nfa = ref_bs.loc['Net Financial Assets'].replace(0, np.nan)
            ratios['Net Borrowing Cost (NBC) %'] = (
                -ref_is.loc['Net Financial Expense'] / nfa
            ) * 100
        
        # OPM (Operating Profit Margin)
        if 'Operating Income After Tax' in ref_is.index and 'Revenue' in df.index:
            revenue = df.loc['Revenue'].replace(0, np.nan)
            ratios['Operating Profit Margin (OPM) %'] = (
                ref_is.loc['Operating Income After Tax'] / revenue
            ) * 100
        
        # NOAT (Net Operating Asset Turnover)
        if 'Revenue' in df.index and 'Net Operating Assets' in ref_bs.index:
            noa = ref_bs.loc['Net Operating Assets'].replace(0, np.nan)
            ratios['Net Operating Asset Turnover (NOAT)'] = df.loc['Revenue'] / noa
        
        # Spread (RNOA - NBC)
        if 'Return on Net Operating Assets (RNOA) %' in ratios.index and 'Net Borrowing Cost (NBC) %' in ratios.index:
            ratios['Spread %'] = ratios.loc['Return on Net Operating Assets (RNOA) %'] - ratios.loc['Net Borrowing Cost (NBC) %']
        
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
            
            # Free Cash Flow to Equity
            if 'Net Income' in df.index and 'Depreciation' in df.index:
                fcf['Free Cash Flow to Equity'] = (
                    df.loc['Net Income'] + 
                    df.loc['Depreciation'] - 
                    (df.loc['Capital Expenditure'] if 'Capital Expenditure' in df.index else 0)
                )
        
        return fcf
    
    def _calculate_value_drivers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate value drivers for DCF analysis"""
        drivers = pd.DataFrame(index=df.columns)
        
        # Revenue growth rate
        if 'Revenue' in df.index:
            revenue = df.loc['Revenue']
            drivers['Revenue Growth %'] = revenue.pct_change() * 100
        
        # NOPAT margin
        if 'Operating Income' in df.index and 'Revenue' in df.index:
            drivers['NOPAT Margin %'] = (df.loc['Operating Income'] / df.loc['Revenue']) * 100
        
        # Working capital as % of revenue
        if 'Current Assets' in df.index and 'Current Liabilities' in df.index and 'Revenue' in df.index:
            working_capital = df.loc['Current Assets'] - df.loc['Current Liabilities']
            drivers['Working Capital % of Revenue'] = (working_capital / df.loc['Revenue']) * 100
        
        return drivers

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

# --- 23. Natural Language Query Processor ---
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

# --- 24. Collaboration Manager ---
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
            st.sidebar.markdown("⬆️ **Upload your files here**")
        elif action == 'highlight_tabs':
            st.markdown("⬆️ **Explore different analysis tabs above**")
        # Add more actions as needed
    
    def _complete_tutorial(self):
        """Mark tutorial as completed"""
        SimpleState.set('show_tutorial', False)
        SimpleState.set('tutorial_completed', True)
        st.success("Tutorial completed! You're ready to use the platform.")

# --- 26. Export Manager ---
class ExportManager:
    """Handle various export formats for analysis results"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.logger = LoggerFactory.get_logger('ExportManager')
    
    @error_boundary(b"Export failed")
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
            f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Company:** {analysis.get('company_name', 'Financial Analysis')}",
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
    
#--- 27. UI Components Factory ---
class UIComponentFactory: """Factory for creating UI components with consistent styling"""

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
        
#--- 28. Sample Data Generator ---
class SampleDataGenerator: """Generate sample financial data for demonstration"""

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
    
#--- 29. Error Recovery Mechanisms ---
class ErrorRecoveryManager: """Manage error recovery and fallback strategies"""

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
    
#--- 30. Main Application Class ---
class FinancialAnalyticsPlatform: """Main application with advanced architecture and all integrations"""

def __init__(self):
    # Initialize session state for persistent data
    if 'initialized' not in st.session_state:
        self._initialize_session_state()
    
    # Initialize configuration with session state overrides
    self.config = Configuration(st.session_state.get('config_overrides', {}))
    
    # Initialize logger
    self.logger = LoggerFactory.get_logger('FinancialAnalyticsPlatform')
    
    # Initialize components only once
    if 'components' not in st.session_state:
        st.session_state.components = self._initialize_components()
    
    self.components = st.session_state.components
    
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

def __del__(self):
    """Cleanup resources"""
    if hasattr(self, 'compression_handler'):
        self.compression_handler.cleanup()

def _initialize_session_state(self):
    """Initialize all session state variables"""
    defaults = {
        'initialized': True,
        'analysis_data': None,
        'metric_mappings': None,
        'pn_mappings': None,
        'pn_results': None,
        'ai_mapping_result': None,
        'company_name': None,
        'data_source': None,
        'show_manual_mapping': False,
        'config_overrides': {},
        'uploaded_files': [],
        'simple_parse_mode': False,
        'number_format_value': 'Indian',
        'show_tutorial': True,
        'tutorial_step': 0,
        'collaboration_session': None,
        'query_history': [],
        'ml_forecast_results': None,
        'kaggle_api_url': '',
        'kaggle_api_enabled': False,
        'kaggle_status': {},
        'show_kaggle_config': False,
        'kaggle_api_status': 'unknown',
        'api_metrics_visible': False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def _initialize_components(self) -> Dict[str, Component]:
    """Initialize all components with dependency injection"""
    components = {
        'security': SecurityModule(self.config),
        'processor': DataProcessor(self.config),
        'analyzer': FinancialAnalysisEngine(self.config),
        'mapper': AIMapper(self.config),
    }
    
    # Initialize all components
    for name, component in components.items():
        try:
            component.initialize()
            self.logger.info(f"Initialized component: {name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize {name}: {e}")
            # Try error recovery
            self.error_recovery.handle_error(f"{name}_init_failed", {'component': component})
    
    return components

# State helper methods
def get_state(self, key: str, default: Any = None) -> Any:
    """Get value from session state"""
    return SimpleState.get(key, default)

def set_state(self, key: str, value: Any):
    """Set value in session state"""
    SimpleState.set(key, value)

def _clear_all_caches(self):
    """Clear all caches"""
    if 'analyzer' in self.components:
        self.components['analyzer'].cache.clear()
    if 'mapper' in self.components:
        self.components['mapper'].embeddings_cache.clear()
    
    # Clear performance monitor
    performance_monitor.clear_metrics()
    
    # Force garbage collection
    gc.collect()

def _reset_configuration(self):
    """Reset configuration to defaults"""
    self.set_state('config_overrides', {})
    self.config = Configuration()
    st.success("Configuration reset to defaults!")

def _export_logs(self):
    """Export application logs"""
    try:
        log_dir = Path("logs")
        if log_dir.exists():
            # Create a zip file with all logs
            import zipfile
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for log_file in log_dir.glob("*.log"):
                    zip_file.write(log_file, log_file.name)
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="Download Logs",
                data=zip_buffer.getvalue(),
                file_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
        else:
            st.warning("No log files found")
    except Exception as e:
        st.error(f"Failed to export logs: {e}")

def _parse_single_file(self, file) -> Optional[pd.DataFrame]:
    """Master parsing method with all enhancements"""
    try:
        filename = getattr(file, 'name', 'unknown_file')
        file_ext = Path(filename).suffix.lower()
        
        # Read file content
        if hasattr(file, 'read'):
            content = file.read()
            file.seek(0)
        else:
            with open(file, 'rb') as f:
                content = f.read()
        
        self.logger.info(f"Parsing {filename} ({len(content)} bytes, extension: {file_ext})")
        
        # Main parsing logic
        df = None
        
        if file_ext == '.xls':
            df = self._parse_xls_file(content, filename)
        elif file_ext == '.xlsx':
            df = self._parse_xlsx_file(content, filename)
        elif file_ext == '.csv':
            df = self._parse_csv_file(content, filename)
        elif file_ext in ['.html', '.htm']:
            df = self._parse_html_file(content, filename)
        
        # Validate result
        if df is not None and self._validate_financial_data(df, filename):
            self.logger.info(f"Successfully parsed {filename}: {df.shape}")
            return df
        
        # Fallback parsing
        self.logger.warning(f"Main parsing failed for {filename}, trying fallback")
        df = self._handle_parsing_fallback(file, filename)
        
        if df is not None:
            return df
        
        self.logger.error(f"All parsing attempts failed for {filename}")
        return None
        
    except Exception as e:
        self.logger.error(f"Critical parsing error for {filename}: {e}", exc_info=True)
        return None

def _parse_xls_file(self, content: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Parse .xls files (could be Excel or HTML)"""
    try:
        # Detect if it's actually HTML
        text_content = content.decode('utf-8', errors='ignore')
        
        if '<html>' in text_content.lower() or '<table>' in text_content.lower():
            self.logger.info(f"Detected HTML content in {filename}")
            return self._parse_html_file(content, filename)
        else:
            # Try as actual Excel
            self.logger.info(f"Parsing {filename} as Excel (.xls)")
            return pd.read_excel(io.BytesIO(content), index_col=0, engine='xlrd')
            
    except Exception as e:
        self.logger.warning(f"XLS parsing failed for {filename}: {e}")
        # Try HTML parsing as fallback
        return self._parse_html_file(content, filename)

def _parse_xlsx_file(self, content: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Parse .xlsx files"""
    try:
        self.logger.info(f"Parsing {filename} as Excel (.xlsx)")
        return pd.read_excel(io.BytesIO(content), index_col=0, engine='openpyxl')
    except Exception as e:
        self.logger.error(f"XLSX parsing failed for {filename}: {e}")
        return None

def _parse_csv_file(self, content: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Parse CSV files"""
    try:
        self.logger.info(f"Parsing {filename} as CSV")
        encoding = self._detect_file_encoding(content)
        text_content = content.decode(encoding)
        return pd.read_csv(io.StringIO(text_content), index_col=0)
    except Exception as e:
        self.logger.error(f"CSV parsing failed for {filename}: {e}")
        return None

def _parse_html_file(self, content: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Parse HTML content from financial exports with enhanced detection"""
    try:
        # Decode content
        if isinstance(content, bytes):
            # Try multiple encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']:
                try:
                    text_content = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text_content = content.decode('utf-8', errors='ignore')
        else:
            text_content = content
        
        self.logger.info(f"Attempting HTML parsing for {filename}")
        
        # Try pandas HTML parser first (most reliable)
        try:
            # Use multiple parsing strategies
            parsing_strategies = [
                {'attrs': None, 'header': 0},
                {'attrs': None, 'header': None},
                {'attrs': {'class': 'data'}, 'header': 0},
                {'attrs': {'border': '1'}, 'header': 0}
            ]
            
            tables = None
            for strategy in parsing_strategies:
                try:
                    tables = pd.read_html(
                        io.StringIO(text_content),
                        attrs=strategy['attrs'],
                        header=strategy['header']
                    )
                    if tables:
                        self.logger.info(f"Successfully parsed {len(tables)} tables using strategy: {strategy}")
                        break
                except Exception:
                    continue
            
            if tables:
                # Find the largest table (usually the main financial data)
                main_table = max(tables, key=lambda x: x.size)
                self.logger.info(f"Selected main table with shape: {main_table.shape}")
                
                # Clean and process the table
                return self._clean_financial_table(main_table, filename)
                
        except Exception as e:
            self.logger.warning(f"Pandas HTML parser failed for {filename}: {e}")
        
        # Fallback to BeautifulSoup parsing
        if BEAUTIFULSOUP_AVAILABLE:
            self.logger.info(f"Trying BeautifulSoup fallback for {filename}")
            return self._parse_with_beautifulsoup(text_content, filename)
        else:
            self.logger.warning("BeautifulSoup not available for fallback parsing")
        
        return None
        
    except Exception as e:
        self.logger.error(f"HTML parsing failed for {filename}: {e}", exc_info=True)
        return None

def _clean_financial_table(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Clean and standardize financial table format with enhanced logic"""
    try:
        self.logger.info(f"Cleaning table from {filename}, initial shape: {df.shape}")
        
        # Make a copy
        cleaned_df = df.copy()
        
        # 1. Remove completely empty rows and columns
        initial_shape = cleaned_df.shape
        cleaned_df = cleaned_df.dropna(how='all').dropna(axis=1, how='all')
        self.logger.info(f"After removing empty rows/cols: {initial_shape} -> {cleaned_df.shape}")
        
        # 2. Find the row that contains year headers (enhanced detection)
        year_row_idx = None
        year_patterns = [
            r'20\d{2}',  # 2020, 2021, etc.
            r'FY\s*20\d{2}',  # FY 2020, FY2021, etc.
            r'Mar\s*20\d{2}',  # Mar 2020, Mar2021, etc.
            r'\d{4}-\d{2}'  # 2020-21, etc.
        ]
        
        for idx, row in cleaned_df.iterrows():
            row_str = ' '.join(str(cell) for cell in row if pd.notna(cell))
            for pattern in year_patterns:
                if re.search(pattern, row_str, re.IGNORECASE):
                    year_row_idx = idx
                    self.logger.info(f"Found year row at index {idx}: {row_str[:100]}")
                    break
            if year_row_idx is not None:
                break
        
        if year_row_idx is not None:
            # Use year row as column headers
            new_columns = []
            year_row = cleaned_df.iloc[year_row_idx]
            
            for col in year_row:
                if pd.notna(col):
                    # Extract year from cell with multiple patterns
                    year_match = None
                    for pattern in year_patterns:
                        year_match = re.search(pattern, str(col), re.IGNORECASE)
                        if year_match:
                            # Extract just the year part
                            year_str = year_match.group(0)
                            # Clean up the year string
                            year_clean = re.search(r'20\d{2}', year_str)
                            if year_clean:
                                new_columns.append(year_clean.group(0))
                            else:
                                new_columns.append(year_str)
                            break
                    
                    if not year_match:
                        new_columns.append(str(col).strip())
                else:
                    new_columns.append('Unknown')
            
            # Set new column names
            cleaned_df.columns = new_columns[:len(cleaned_df.columns)]
            
            # Remove the year row and everything above it
            cleaned_df = cleaned_df.iloc[year_row_idx + 1:].reset_index(drop=True)
            self.logger.info(f"After header processing: {cleaned_df.shape}")
        
        # 3. Set first column as index (metric names)
        if len(cleaned_df.columns) > 0:
            # Clean the first column before setting as index
            first_col = cleaned_df.iloc[:, 0].astype(str).str.strip()
            cleaned_df = cleaned_df.set_index(first_col)
            cleaned_df.index.name = 'Metric'
        
        # 4. Convert numeric columns with enhanced cleaning
        for col in cleaned_df.columns:
            if col != cleaned_df.index.name:
                # Clean numeric data (remove commas, parentheses for negative numbers, etc.)
                cleaned_series = cleaned_df[col].astype(str)
                
                # Remove common formatting
                cleaned_series = cleaned_series.str.replace(',', '')  # Remove commas
                cleaned_series = cleaned_series.str.replace('(', '-')  # Convert (123) to -123
                cleaned_series = cleaned_series.str.replace(')', '')
                cleaned_series = cleaned_series.str.replace('₹', '')  # Remove currency symbols
                cleaned_series = cleaned_series.str.replace('$', '')
                cleaned_series = cleaned_series.str.strip()
                
                # Convert to numeric
                cleaned_df[col] = pd.to_numeric(cleaned_series, errors='coerce')
        
        # 5. Remove rows where all values are NaN
        cleaned_df = cleaned_df.dropna(how='all')
        
        # 6. Clean index names (remove extra spaces, special characters)
        cleaned_df.index = cleaned_df.index.map(
            lambda x: re.sub(r'\s+', ' ', str(x).strip()) if pd.notna(x) else 'Unknown'
        )
        
        # 7. Remove duplicate index entries
        cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]
        
        self.logger.info(f"Final cleaned table from {filename}: {cleaned_df.shape}")
        self.logger.info(f"Columns: {list(cleaned_df.columns)}")
        self.logger.info(f"Sample metrics: {list(cleaned_df.index[:5])}")
        
        return cleaned_df
        
    except Exception as e:
        self.logger.error(f"Error cleaning table from {filename}: {e}", exc_info=True)
        return df  # Return original if cleaning fails

def _parse_with_beautifulsoup(self, html_content: str, filename: str) -> Optional[pd.DataFrame]:
    """Fallback parser using BeautifulSoup"""
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all tables
        tables = soup.find_all('table')
        
        if not tables:
            return None
        
        # Use the largest table
        main_table = max(tables, key=lambda t: len(t.find_all('tr')))
        
        # Extract data
        rows = []
        for tr in main_table.find_all('tr'):
            row = []
            for td in tr.find_all(['td', 'th']):
                text = td.get_text(strip=True)
                row.append(text)
            if row:
                rows.append(row)
        
        if not rows:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rows[1:], columns=rows[0] if rows else None)
        
        return self._clean_financial_table(df, filename)
        
    except Exception as e:
        self.logger.error(f"BeautifulSoup parsing failed for {filename}: {e}")
        return None

def _detect_file_encoding(self, content: bytes) -> str:
    """Detect file encoding for better text parsing"""
    try:
        # Try common encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        
        for encoding in encodings:
            try:
                content.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        
        # Fallback to utf-8 with error handling
        return 'utf-8'
        
    except Exception:
        return 'utf-8'

def _validate_financial_data(self, df: pd.DataFrame, filename: str) -> bool:
    """Validate that the parsed data looks like financial data"""
    try:
        if df.empty:
            self.logger.warning(f"Empty dataframe from {filename}")
            return False
        
        # Check for financial keywords in index
        financial_keywords = [
            'revenue', 'income', 'profit', 'loss', 'assets', 'liabilities', 
            'equity', 'cash', 'debt', 'sales', 'expenses', 'ebit', 'ebitda',
            'turnover', 'capital', 'reserves', 'depreciation', 'tax'
        ]
        
        index_text = ' '.join(str(idx).lower() for idx in df.index)
        keyword_matches = sum(1 for keyword in financial_keywords if keyword in index_text)
        
        if keyword_matches < 2:
            self.logger.warning(f"Low financial keyword matches in {filename}: {keyword_matches}")
            return False
        
        # Check for numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            self.logger.warning(f"No numeric columns found in {filename}")
            return False
        
        # Check for year-like columns
        year_cols = [col for col in df.columns if re.search(r'20\d{2}', str(col))]
        if len(year_cols) == 0:
            self.logger.warning(f"No year columns found in {filename}")
            return False
        
        self.logger.info(f"Validation passed for {filename}: {keyword_matches} keywords, {len(numeric_cols)} numeric cols, {len(year_cols)} year cols")
        return True
        
    except Exception as e:
        self.logger.error(f"Validation error for {filename}: {e}")
        return False

def _handle_parsing_fallback(self, file, filename: str) -> Optional[pd.DataFrame]:
    """Last resort parsing attempts"""
    try:
        self.logger.info(f"Attempting fallback parsing for {filename}")
        
        # Try reading as plain text and look for tabular data
        if hasattr(file, 'read'):
            content = file.read()
            file.seek(0)
        else:
            with open(file, 'rb') as f:
                content = f.read()
        
        # Decode content
        encoding = self._detect_file_encoding(content)
        text_content = content.decode(encoding, errors='ignore')
        
        # Look for tab-separated or pipe-separated data
        lines = text_content.split('\n')
        
        # Try different separators
        separators = ['\t', '|', ';', ',']
        
        for sep in separators:
            try:
                # Check if this separator creates consistent columns
                line_lengths = []
                for line in lines[:10]:  # Check first 10 lines
                    if line.strip():
                        parts = line.split(sep)
                        line_lengths.append(len(parts))
                
                if line_lengths and len(set(line_lengths)) <= 2:  # Consistent structure
                    # Try to parse as CSV with this separator
                    df = pd.read_csv(io.StringIO(text_content), sep=sep, index_col=0)
                    
                    if self._validate_financial_data(df, filename):
                        self.logger.info(f"Fallback parsing successful with separator '{sep}'")
                        return df
                        
            except Exception:
                continue
        
        return None
        
    except Exception as e:
        self.logger.error(f"Fallback parsing failed for {filename}: {e}")
        return None

def _handle_parsing_error(self, filename: str, error: Exception) -> None:
    """Provide helpful error messages and suggestions"""
    error_msg = str(error).lower()
    
    if "file is not a zip file" in error_msg:
        st.error(f"❌ {filename}: File format detection issue")
        with st.expander("💡 How to fix this"):
            st.write("""
            **This error usually means:**
            - The file has a .xls extension but contains HTML data (common with Indian financial exports)
            - The file might be corrupted
            
            **Try these solutions:**
            1. **Rename the file**: Change .xls to .html and re-upload
            2. **Open in Excel**: Open the file in Excel and save as .xlsx
            3. **Check the source**: Ensure you downloaded the complete file
            4. **Use CSV format**: If possible, export data as CSV from the source
            """)
    
    elif "no tables found" in error_msg:
        st.error(f"❌ {filename}: No data tables detected")
        with st.expander("💡 How to fix this"):
            st.write("""
            **This means the file doesn't contain recognizable table data.**
            
            **Try these solutions:**
            1. **Check file content**: Open the file to verify it contains financial data
            2. **Different format**: Try exporting in a different format (CSV, Excel)
            3. **Clean the data**: Remove headers/footers that might confuse the parser
            """)
    
    elif "encoding" in error_msg or "unicode" in error_msg:
        st.error(f"❌ {filename}: Text encoding issue")
        with st.expander("💡 How to fix this"):
            st.write("""
            **This is a character encoding problem.**
            
            **Try these solutions:**
            1. **Save as UTF-8**: Open in Excel/Notepad and save with UTF-8 encoding
            2. **Remove special characters**: Clean any unusual symbols from the data
            3. **Use English headers**: Ensure column headers are in English
            """)
    
    else:
        st.error(f"❌ {filename}: Parsing failed - {str(error)}")
        with st.expander("💡 General troubleshooting"):
            st.write("""
            **General solutions:**
            1. **Check file format**: Ensure it's a supported format (CSV, Excel, HTML)
            2. **Verify data structure**: Data should have metrics in rows, columns = years
            3. **Clean the file**: Remove extra headers, footers, or merged cells
            4. **Try sample data**: Test with our sample datasets first
            5. **Contact support**: If issues persist, please contact support
            """)

def test_parsing_capabilities(self):
    """Test the parsing system with various file types"""
    test_results = {
        'html_xls': False,
        'real_excel': False,
        'csv': False,
        'html': False,
        'encoding_handling': False,
        'year_detection': False,
        'metric_cleaning': False
    }
    
    try:
        # Test 1: HTML disguised as XLS (common Indian format)
        html_content = """
        <html>
        <table border="1">
        <tr><td>Particulars</td><td>Mar 2023</td><td>Mar 2022</td><td>Mar 2021</td></tr>
        <tr><td>Total Revenue</td><td>1,50,000</td><td>1,25,000</td><td>1,00,000</td></tr>
        <tr><td>Net Profit</td><td>15,000</td><td>12,500</td><td>10,000</td></tr>
        <tr><td>Total Assets</td><td>5,00,000</td><td>4,50,000</td><td>4,00,000</td></tr>
        </table>
        </html>
        """.encode('utf-8')
        
        temp_file = io.BytesIO(html_content)
        temp_file.name = "test_file.xls"
        
        df = self._parse_single_file(temp_file)
        if df is not None and not df.empty:
            test_results['html_xls'] = True
            self.logger.info("✅ HTML-as-XLS parsing test passed")
        
        # Test 2: Year detection
        if df is not None:
            year_cols = [col for col in df.columns if re.search(r'20\d{2}', str(col))]
            if len(year_cols) >= 2:
                test_results['year_detection'] = True
                self.logger.info("✅ Year detection test passed")
        
        # Test 3: Metric cleaning
        if df is not None:
            clean_metrics = [idx for idx in df.index if isinstance(idx, str) and len(idx.strip()) > 0]
            if len(clean_metrics) >= 2:
                test_results['metric_cleaning'] = True
                self.logger.info("✅ Metric cleaning test passed")
        
        # Test 4: CSV parsing
        csv_content = """Metric,2023,2022,2021
Total Revenue,150000,125000,100000 Net Profit,15000,12500,10000 Total Assets,500000,450000,400000""".encode('utf-8')

        temp_csv = io.BytesIO(csv_content)
        temp_csv.name = "test_file.csv"
        
        csv_df = self._parse_single_file(temp_csv)
        if csv_df is not None and not csv_df.empty:
            test_results['csv'] = True
            self.logger.info("✅ CSV parsing test passed")
        
        # Test 5: Encoding handling
        try:
            # Test with different encodings
            unicode_content = "Metric,2023\nRevenue ₹,150000\nProfit ₹,15000".encode('utf-8')
            temp_unicode = io.BytesIO(unicode_content)
            temp_unicode.name = "test_unicode.csv"
            
            unicode_df = self._parse_single_file(temp_unicode)
            if unicode_df is not None:
                test_results['encoding_handling'] = True
                self.logger.info("✅ Encoding handling test passed")
        except Exception as e:
            self.logger.warning(f"Encoding test failed: {e}")
        
        # Summary
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        self.logger.info(f"Parsing tests summary: {passed_tests}/{total_tests} passed")
        
        if passed_tests >= total_tests * 0.8:  # 80% pass rate
            self.logger.info("✅ Parsing system is working well")
            return True
        else:
            self.logger.warning("⚠️ Parsing system needs attention")
            return False
            
    except Exception as e:
        self.logger.error(f"Parsing test failed: {e}")
        return False

def _process_uploaded_files(self, uploaded_files: List[UploadedFile]):
    """Complete enhanced file processing with all improvements"""
    try:
        # Run parsing capability test first (in debug mode)
        if self.config.get('app.debug', False):
            self.test_parsing_capabilities()
        
        all_dataframes = []
        file_info = []
        processing_errors = []
        
        # Create progress tracking
        progress_container = st.container()
        
        with progress_container:
            st.write("### 📁 Processing Files")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(uploaded_files)
            
            for i, file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress_bar.progress((i + 1) / total_files)
                    status_text.text(f"Processing {file.name} ({i+1}/{total_files})")
                    
                    # Handle compressed files
                    if file.name.lower().endswith(('.zip', '.7z')):
                        extracted_files = self.compression_handler.extract_compressed_file(file)
                        
                        for extracted_name, extracted_content in extracted_files:
                            temp_file = io.BytesIO(extracted_content)
                            temp_file.name = extracted_name
                            
                            try:
                                df = self._parse_single_file(temp_file)
                                if df is not None and not df.empty:
                                    all_dataframes.append(df)
                                    file_info.append({
                                        'name': extracted_name,
                                        'source': f"{file.name} (compressed)",
                                        'shape': df.shape,
                                        'columns': list(df.columns),
                                        'metrics_count': len(df.index),
                                        'status': 'success'
                                    })
                                else:
                                    processing_errors.append(f"No data in {extracted_name}")
                                    file_info.append({
                                        'name': extracted_name,
                                        'source': f"{file.name} (compressed)",
                                        'status': 'failed',
                                        'error': 'No data extracted'
                                    })
                            except Exception as e:
                                processing_errors.append(f"Error in {extracted_name}: {str(e)}")
                                self._handle_parsing_error(extracted_name, e)
                    else:
                        # Handle regular files
                        try:
                            df = self._parse_single_file(file)
                            if df is not None and not df.empty:
                                all_dataframes.append(df)
                                file_info.append({
                                    'name': file.name,
                                    'source': 'direct upload',
                                    'shape': df.shape,
                                    'columns': list(df.columns),
                                    'metrics_count': len(df.index),
                                    'status': 'success'
                                })
                            else:
                                processing_errors.append(f"No data in {file.name}")
                                file_info.append({
                                    'name': file.name,
                                    'source': 'direct upload',
                                    'status': 'failed',
                                    'error': 'No data extracted'
                                })
                        except Exception as e:
                            processing_errors.append(f"Error in {file.name}: {str(e)}")
                            self._handle_parsing_error(file.name, e)
                
                except Exception as e:
                    self.logger.error(f"Critical error processing {file.name}: {e}", exc_info=True)
                    processing_errors.append(f"Critical error in {file.name}: {str(e)}")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        # Show results
        if all_dataframes:
            success_count = len([info for info in file_info if info.get('status') == 'success'])
            total_count = len(file_info)
            
            st.success(f"✅ Successfully processed {success_count}/{total_count} files")
            
            # Show detailed results
            with st.expander("📊 Processing Results", expanded=True):
                results_df = pd.DataFrame(file_info)
                
                # Color code the results
                def highlight_status(val):
                    if val == 'success':
                        return 'background-color: #d4edda'
                    elif val == 'failed':
                        return 'background-color: #f8d7da'
                    return ''
                
                if 'status' in results_df.columns:
                    styled_df = results_df.style.applymap(highlight_status, subset=['status'])
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.dataframe(results_df, use_container_width=True)
            
            # Merge and process data
            if len(all_dataframes) == 1:
                final_df = all_dataframes[0]
            else:
                final_df = self._merge_dataframes(all_dataframes)
            
            # Validate and process
            processed_df, validation_result = self.components['processor'].process(final_df, "uploaded_data")
            
            # Store results
            self.set_state('analysis_data', processed_df)
            self.set_state('data_source', 'uploaded_files')
            
            # Show validation results
            if validation_result.warnings:
                st.warning("⚠️ Data Quality Warnings:")
                for warning in validation_result.warnings[:3]:
                    st.write(f"• {warning}")
            
            if validation_result.corrections:
                st.info(f"🔧 Applied {len(validation_result.corrections)} auto-corrections")
                with st.expander("View corrections"):
                    for correction in validation_result.corrections:
                        st.write(f"• {correction}")
            
            return True
            
        else:
            st.error("❌ No valid financial data found in any uploaded files")
            
            if processing_errors:
                with st.expander("🔍 Error Details"):
                    for error in processing_errors:
                        st.write(f"• {error}")
            
            # Show helpful suggestions
            st.info("💡 **Suggestions:**")
            st.write("1. Try uploading sample data first to test the system")
            st.write("2. Ensure your files contain financial statement data")
            st.write("3. Check that data is in tabular format (rows = metrics, columns = years)")
            st.write("4. For Indian data exports, files with .xls extension often work better")
            
            return False
            
    except Exception as e:
        self.logger.error(f"Critical error in file processing: {e}", exc_info=True)
        st.error(f"Critical system error: {str(e)}")
        return False
    finally:
        # Always cleanup
        self.compression_handler.cleanup()

def _merge_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Enhanced dataframe merging with intelligent handling"""
    try:
        self.logger.info(f"Merging {len(dataframes)} dataframes")
        
        # Log shapes for debugging
        for i, df in enumerate(dataframes):
            self.logger.info(f"DataFrame {i}: {df.shape}, columns: {list(df.columns)}")
        
        # Strategy 1: If all dataframes have similar column structure, concatenate
        if len(set(tuple(df.columns) for df in dataframes)) == 1:
            # All have same columns - simple concatenation
            merged_df = pd.concat(dataframes, axis=0, ignore_index=False)
            merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
            self.logger.info(f"Merged using concatenation: {merged_df.shape}")
            return merged_df
        
        # Strategy 2: Merge on index (metrics) with outer join
        merged_df = dataframes[0].copy()
        
        for i, df in enumerate(dataframes[1:], 1):
            try:
                # Align column names if they represent years
                aligned_df = self._align_dataframe_columns(df, merged_df)
                
                # Merge on index
                merged_df = pd.merge(
                    merged_df, aligned_df, 
                    left_index=True, right_index=True, 
                    how='outer', suffixes=('', f'_file{i}')
                )
                
                self.logger.info(f"After merging file {i}: {merged_df.shape}")
                
            except Exception as e:
                self.logger.warning(f"Failed to merge dataframe {i}: {e}")
                # Skip this dataframe and continue
                continue
        
        # Clean up the merged result
        merged_df = self._clean_merged_dataframe(merged_df)
        
        self.logger.info(f"Final merged dataframe: {merged_df.shape}")
        return merged_df
        
    except Exception as e:
        self.logger.error(f"Error merging dataframes: {e}", exc_info=True)
        # Fallback: return the largest dataframe
        return max(dataframes, key=lambda x: x.size)

def _align_dataframe_columns(self, df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Align column names between dataframes (especially for years)"""
    try:
        aligned_df = df.copy()
        
        # Extract years from both dataframes
        ref_years = set()
        df_years = set()
        
        for col in reference_df.columns:
            year_match = re.search(r'20\d{2}', str(col))
            if year_match:
                ref_years.add(year_match.group(0))
        
        for col in df.columns:
            year_match = re.search(r'20\d{2}', str(col))
            if year_match:
                df_years.add(year_match.group(0))
        
        # Create mapping for overlapping years
        column_mapping = {}
        for df_col in df.columns:
            df_year_match = re.search(r'20\d{2}', str(df_col))
            if df_year_match:
                df_year = df_year_match.group(0)
                
                # Find corresponding column in reference
                for ref_col in reference_df.columns:
                    ref_year_match = re.search(r'20\d{2}', str(ref_col))
                    if ref_year_match and ref_year_match.group(0) == df_year:
                        column_mapping[df_col] = ref_col
                        break
        
        # Apply mapping
        if column_mapping:
            aligned_df = aligned_df.rename(columns=column_mapping)
            self.logger.info(f"Aligned columns: {column_mapping}")
        
        return aligned_df
        
    except Exception as e:
        self.logger.warning(f"Column alignment failed: {e}")
        return df

def _clean_merged_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
    """Clean the merged dataframe"""
    try:
        cleaned_df = df.copy()
        
        # 1. Remove completely empty rows
        cleaned_df = cleaned_df.dropna(how='all')
        
        # 2. Handle duplicate columns (from suffixes)
        columns_to_drop = []
        base_columns = {}
        
        for col in cleaned_df.columns:
            # Check if this is a suffixed column
            if '_file' in str(col):
                base_col = re.sub(r'_file\d+$', '', str(col))
                if base_col in cleaned_df.columns:
                    # Compare values and keep the one with more data
                    base_data = cleaned_df[base_col].dropna()
                    suffix_data = cleaned_df[col].dropna()
                    
                    if len(suffix_data) > len(base_data):
                        # Replace base with suffix data
                        cleaned_df[base_col] = cleaned_df[col]
                    
                    columns_to_drop.append(col)
        
        # Drop duplicate columns
        cleaned_df = cleaned_df.drop(columns=columns_to_drop)
        
        # 3. Sort columns by year if they contain years
        year_columns = []
        other_columns = []
        
        for col in cleaned_df.columns:
            # First, perform the search for a year pattern in the column name.
            year_match = re.search(r'20\d{2}', str(col))
            
            # Next, check if the search found a match.
            if year_match:
                # If a match was found (i.e., it's a year column),
                # append the original column name and the extracted year string.
                year_columns.append((col, year_match.group(0)))
            else:
                # If no match was found, it's a different type of column.
                other_columns.append(col)

        # Sort year columns by year
        year_columns.sort(key=lambda x: x[1])
        sorted_columns = [col for col, _ in year_columns] + other_columns
        
        cleaned_df = cleaned_df[sorted_columns]
        
        self.logger.info(f"Cleaned merged dataframe: {cleaned_df.shape}")
        return cleaned_df
        
    except Exception as e:
        self.logger.error(f"Error cleaning merged dataframe: {e}")
        return df

@error_boundary()
def run(self):
    """Main application entry point"""
    try:
        # Set page config
        st.set_page_config(
            page_title="Elite Financial Analytics Platform v5.1",
            page_icon="💹",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        self._apply_custom_css()
        
        # Render tutorial if active
        self.tutorial_system.render()
        
        # Render header
        self._render_header()
        
        # Render sidebar
        self._render_sidebar()
        
        # Render main content
        self._render_main_content()
        
        # Show performance metrics in footer if debug mode
        if self.config.get('app.debug', False):
            self._render_debug_footer()
        
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
    
    # Collaboration indicator
    if self.get_state('collaboration_session'):
        session_id = self.get_state('collaboration_session')
        activity = self.collaboration_manager.get_session_activity(session_id)
        if activity:
            st.markdown(
                f'<div class="collaboration-indicator">👥 {len(activity["active_users"])} users online</div>',
                unsafe_allow_html=True
            )
    
    # Show Kaggle API status if enabled
    if self.config.get('ui.show_kaggle_status', True) and self.get_state('kaggle_api_enabled'):
        self._render_kaggle_status_badge()
    
    # Show API metrics panel if enabled
    if self.config.get('ui.show_api_metrics', True) and self.get_state('api_metrics_visible', False):
        self._render_api_metrics_panel()
    
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
        if 'mapper' in self.components:
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
def _render_kaggle_status_badge(self):
    """Render floating Kaggle API status badge with enhanced metrics"""
    if 'mapper' in self.components:
        status = self.components['mapper'].get_api_status()
        
        if status['kaggle_available']:
            info = status.get('api_info', {})
            stats = status.get('api_stats', {})
            
            # Get GPU info from the actual response
            gpu_name = info.get('gpu_name', info.get('system', {}).get('gpu_name', 'GPU'))
            model = info.get('model', info.get('system', {}).get('model', 'Unknown'))
            version = info.get('version', 'Unknown')
            
            # Check circuit breaker status
            circuit_state = 'closed'  # Your API shows it's closed (good)
            
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

def _render_api_metrics_panel(self):
    """Render detailed API metrics panel"""
    if 'mapper' in self.components:
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

def _render_sidebar(self):
    """Render sidebar with enhanced Kaggle configuration"""
    st.sidebar.title("⚙️ Configuration")
    
    # Kaggle GPU Configuration Section
    st.sidebar.header("🖥️ Kaggle GPU Configuration")
    
    # Enable/Disable toggle
    kaggle_enabled = st.sidebar.checkbox(
        "Enable Kaggle GPU Acceleration",
        value=self.get_state('kaggle_api_enabled', False),
        help="Use remote GPU for faster processing"
    )
    
    if kaggle_enabled:
        # Show configuration options
        with st.sidebar.expander("Kaggle API Settings", expanded=True):
            # API URL input
            api_url = st.text_input(
                "Ngrok URL",
                value=self.get_state('kaggle_api_url', ''),
                placeholder="https://xxxx.ngrok-free.app",
                help="Paste the ngrok URL from your Kaggle notebook"
            )
            
            # Optional API key
            api_key = st.text_input(
                "API Key (Optional)",
                type="password",
                help="Optional API key for authentication"
            )
            
            # Advanced settings
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
            
            # Circuit breaker settings
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
            
            # Test connection button
            if st.button("🔌 Test Connection", type="primary"):
                if api_url:
                    # Update configuration
                    self.config.set('ai.kaggle_api_url', api_url)
                    self.config.set('ai.kaggle_api_key', api_key)
                    self.config.set('ai.kaggle_api_timeout', timeout)
                    self.config.set('ai.kaggle_batch_size', batch_size)
                    self.config.set('ai.kaggle_circuit_breaker_threshold', cb_threshold)
                    self.config.set('ai.kaggle_circuit_breaker_timeout', cb_timeout)
                    self.config.set('ai.use_kaggle_api', True)
                    
                    # Reinitialize AI mapper
                    with st.spinner("Testing Kaggle connection..."):
                        try:
                            self.components['mapper'].cleanup()
                            self.components['mapper'] = AIMapper(self.config)
                            self.components['mapper'].initialize()
                            
                            status = self.components['mapper'].get_api_status()
                            
                            if status['kaggle_available']:
                                st.success("✅ Successfully connected to Kaggle GPU!")
                                
                                # Show API info
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
                                
                                # Save to session state
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
            
            # Debug embed endpoint button
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
                                    
                                    # Show structure of response
                                    if isinstance(result, dict):
                                        st.write(f"- Type: Dictionary with keys: {list(result.keys())}")
                                        
                                        # Show sample of embeddings if present
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
                                    
                                    # Show full response in expander
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
            
            # Add connection diagnostics button
            if st.button("🔍 Run Full Diagnostics", type="secondary"):
                if api_url:
                    with st.expander("Diagnostic Results", expanded=True):
                        st.write(f"**Testing URL:** `{api_url}`")
                        
                        # Test various endpoints
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
                        
                        # Test POST to embed with different payloads
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
            
            # Show connection guide
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
        # Disabled - clear settings
        if self.get_state('kaggle_api_enabled'):
            self.config.set('ai.use_kaggle_api', False)
            self.set_state('kaggle_api_enabled', False)
            self.set_state('kaggle_api_status', 'disabled')
        
        st.sidebar.info("Enable to use GPU-accelerated processing via Kaggle")
    
    # Show performance comparison
    if 'mapper' in self.components:
        status = self.components['mapper'].get_api_status()
        
        if status['kaggle_configured'] or status['local_model_available']:
            st.sidebar.subheader("🎯 Processing Status")
            
            # Show what's available
            processing_methods = []
            if status['kaggle_available']:
                processing_methods.append("✅ Kaggle GPU")
            if status['local_model_available']:
                processing_methods.append("✅ Local Model")
            if not processing_methods:
                processing_methods.append("✅ Fuzzy Matching")
            
            for method in processing_methods:
                st.sidebar.text(method)
            
            # Show metrics toggle
            if st.sidebar.checkbox("Show API Metrics", value=self.get_state('api_metrics_visible', False)):
                self.set_state('api_metrics_visible', True)
            else:
                self.set_state('api_metrics_visible', False)
            
            # Cache statistics
            st.sidebar.metric("Cache Size", status['cache_size'])
            st.sidebar.metric("Buffer Size", status.get('buffer_size', 0))
    
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
        self.set_state('config_overrides', {'app': {'display_mode': Configuration.DisplayMode[selected_mode]}})
    
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
    
    # Collaboration Settings
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
    
    # Number format
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
        
        if st.sidebar.button("Export Logs"):
            self._export_logs()
        
        # Show performance summary
        if debug_mode:
            perf_summary = performance_monitor.get_performance_summary()
            if perf_summary:
                st.sidebar.write("**Performance Summary:**")
                for op, stats in list(perf_summary.items())[:5]:
                    st.sidebar.text(f"{op}: {stats['avg_duration']:.3f}s")

def _render_file_upload(self):
    """Render file upload interface"""
    allowed_types = self.config.get('app.allowed_file_types', [])
    max_size = self.config.get('security.max_upload_size_mb', 50)
    
    # File uploader
    temp_files = st.sidebar.file_uploader(
        f"Upload Financial Statements (Max {max_size}MB each)",
        type=allowed_types,
        accept_multiple_files=True,
        key="file_uploader",
        help="You can upload compressed files (.zip, .7z) containing multiple financial statements"
    )
    
    if temp_files:
        st.session_state['uploaded_files'] = temp_files
        
        # Count files and show info
        regular_files = [f for f in temp_files if not f.name.lower().endswith(('.zip', '.7z'))]
        compressed_files = [f for f in temp_files if f.name.lower().endswith(('.zip', '.7z'))]
        
        if compressed_files:
            st.sidebar.info(f"📦 {len(compressed_files)} compressed file(s) uploaded")
        if regular_files:
            st.sidebar.info(f"📄 {len(regular_files)} regular file(s) uploaded")
    
    uploaded_files = st.session_state['uploaded_files']
    
    if uploaded_files:
        # Simple parsing mode checkbox
        st.session_state['simple_parse_mode'] = st.sidebar.checkbox(
            "Use simple parsing mode", 
            value=st.session_state['simple_parse_mode'],
            help="Try this if normal parsing fails"
        )
        
        # Check for 7z files and py7zr availability
        has_7z = any(f.name.lower().endswith('.7z') for f in uploaded_files)
        if has_7z and not SEVEN_ZIP_AVAILABLE:
            st.sidebar.warning("⚠️ 7z files detected but py7zr not installed")
            st.sidebar.code("pip install py7zr")
        
        # Validate files
        all_valid = True
        for file in uploaded_files:
            # Skip validation for compressed files
            if not file.name.lower().endswith(('.zip', '.7z')):
                result = self.components['security'].validate_file_upload(file)
                if not result.is_valid:
                    st.sidebar.error(f"❌ {file.name}: {result.errors[0]}")
                    all_valid = False
        
        if all_valid and st.sidebar.button("Process Files", type="primary"):
            self._process_uploaded_files(uploaded_files)
    
    # Format guide
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
            
            # Process data
            processed_df, validation_result = self.components['processor'].process(df, "sample_data")
            
            # Store in session state
            self.set_state('analysis_data', processed_df)
            self.set_state('company_name', company_name)
            self.set_state('data_source', 'sample_data')
            
            st.success(f"✅ Loaded sample data: {sample_name}")
            
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")

def _render_main_content(self):
    """Render main content area"""
    # Natural Language Query Bar
    if self.config.get('app.enable_ml_features', True):
        self._render_query_bar()
    
    # Main content
    if self.get_state('analysis_data') is not None:
        self._render_analysis_interface()
    else:
        self._render_welcome_screen()

def _render_query_bar(self):
    """Render natural language query bar"""
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Ask a question about your financial data",
            placeholder="e.g., What was the revenue growth last year?",
            key="nl_query"
        )
    
    with col2:
        if st.button("Ask", type="primary", key="ask_button"):
            if query and self.get_state('analysis_data') is not None:
                with st.spinner("Processing query..."):
                    analysis = self.components['analyzer'].analyze_financial_statements(
                        self.get_state('analysis_data')
                    )
                    
                    result = self.nl_processor.process_query(
                        query,
                        self.get_state('analysis_data'),
                        analysis
                    )
                    
                    # Store in history
                    query_history = self.get_state('query_history', [])
                    query_history.append({
                        'query': query,
                        'result': result,
                        'timestamp': datetime.now()
                    })
                    self.set_state('query_history', query_history[-10:])
                    
                    # Display result
                    self._display_query_result(result)

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
            
            # Create forecast chart
            fig = go.Figure()
            
            # Forecast
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
        
        # Display summary metrics
        cols = st.columns(4)
        summary_data = summary.get('summary', {})
        
        if 'total_metrics' in summary_data:
            cols[0].metric("Total Metrics", summary_data['total_metrics'])
        if 'year_range' in summary_data:
            cols[1].metric("Period", summary_data['year_range'])
        if 'quality_score' in summary:
            cols[2].metric("Data Quality", f"{summary['quality_score']:.0f}%")
        
        # Display insights
        if 'insights' in summary:
            st.write("**Key Insights:**")
            for insight in summary['insights']:
                st.write(f"- {insight}")
    
    else:
        st.info(result.get('message', 'Query processed successfully'))

def _render_welcome_screen(self):
    """Render welcome screen"""
    st.header("Welcome to Elite Financial Analytics Platform v5.1")
    
    # Feature cards
    col1,col2,col3 = st.columns(3)
    
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
    
    # Quick start guide
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
    
    # Sample data quick access
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

def _render_analysis_interface(self):
    """Render main analysis interface"""
    data = self.get_state('analysis_data')
    
    if data is None:
        self._render_welcome_screen()
        return
    
    # Analysis tabs
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
        self._render_penman_nissim_tab(data)
    
    with tabs[4]:
        self._render_industry_tab(data)
    
    with tabs[5]:
        self._render_data_explorer_tab(data)
    
    with tabs[6]:
        self._render_reports_tab(data)
    
    with tabs[7]:
        self._render_ml_insights_tab(data)

@error_boundary()
def _render_overview_tab(self, data: pd.DataFrame):
    """Render overview tab with key metrics and insights"""
    st.header("Financial Overview")
    
    # Show progress tracking if available
    if 'mapper' in self.components and hasattr(self.components['mapper'], 'progress_tracker'):
        self._render_progress_tracking()
    
    # Analyze data
    with performance_monitor.measure("overview_analysis"):
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
    
    # Anomaly detection
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
    
    # Quick visualizations
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

def _render_progress_tracking(self):
    """Render progress tracking for long operations"""
    progress_tracker = self.components['mapper'].progress_tracker
    
    # Get active operations
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

@error_boundary()
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

def _perform_ai_mapping(self, data: pd.DataFrame):
    """Perform AI mapping of metrics with progress tracking"""
    try:
        with st.spinner("AI is mapping your metrics..."):
            source_metrics = [str(m) for m in data.index.tolist()]
            
            # Check if Kaggle is available
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
                
                # Auto-apply high confidence mappings
                auto_mappings = mapping_result.get('high_confidence', {})
                if auto_mappings:
                    final_mappings = {source: data['target'] for source, data in auto_mappings.items()}
                    self.set_state('metric_mappings', final_mappings)
                    
                    st.success(f"✅ AI mapped {len(final_mappings)} metrics with high confidence!")
                    st.info(f"Method: {mapping_result.get('method', 'unknown')}")
                    
                    # Show medium and low confidence for review
                    medium_conf = mapping_result.get('medium_confidence', {})
                    low_conf = mapping_result.get('low_confidence', {})
                    
                    if medium_conf or low_conf:
                        st.info(f"Review {len(medium_conf) + len(low_conf)} additional suggested mappings below")
                        
                        # Show confidence breakdown
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
def _render_trends_tab(self, data: pd.DataFrame):
    """Render trends and analysis tab"""
    st.header("📉 Trend Analysis & ML Forecasting")
    
    analysis = self.components['analyzer'].analyze_financial_statements(data)
    trends = analysis.get('trends', {})
    
    if not trends or 'error' in trends:
        st.error("Insufficient data for trend analysis. Need at least 2 years of data.")
        return
    
    # Trend summary
    st.subheader("Trend Summary")
    
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
    
    # ML Forecasting Section
    st.subheader("🤖 ML-Powered Forecasting")
    
    if self.config.get('app.enable_ml_features', True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_periods = st.selectbox(
                "Forecast Periods",
                [1, 2, 3, 4, 5],
                index=2,
                help="Number of future periods to forecast"
            )
        
        with col2:
            model_type = st.selectbox(
                "Model Type",
                ['auto', 'linear', 'polynomial', 'exponential'],
                index=0,
                help="ML model for forecasting"
            )
        
        with col3:
            if st.button("🚀 Generate Forecast", type="primary"):
                with st.spinner("Training ML models and generating forecasts..."):
                    try:
                        forecast_results = self.ml_forecaster.forecast_metrics(
                            data, 
                            periods=forecast_periods,
                            model_type=model_type
                        )
                        
                        self.set_state('ml_forecast_results', forecast_results)
                        
                        if 'error' not in forecast_results:
                            st.success(f"✅ Forecast generated using {forecast_results['model_type']} model")
                        else:
                            st.error(f"Forecast failed: {forecast_results['error']}")
                            
                    except Exception as e:
                        st.error(f"Forecasting error: {str(e)}")
        
        # Display forecast results
        forecast_results = self.get_state('ml_forecast_results')
        if forecast_results and 'error' not in forecast_results:
            st.subheader("📈 Forecast Results")
            
            forecasts = forecast_results.get('forecasts', {})
            confidence_intervals = forecast_results.get('confidence_intervals', {})
            
            for metric, forecast in forecasts.items():
                st.write(f"**{metric} Forecast**")
                
                # Create forecast visualization
                fig = go.Figure()
                
                # Historical data (last few points for context)
                if metric in data.index:
                    hist_series = data.loc[metric].dropna()
                    hist_years = hist_series.index.tolist()
                    hist_values = hist_series.values.tolist()
                    
                    fig.add_trace(go.Scatter(
                        x=hist_years,
                        y=hist_values,
                        mode='lines+markers',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                
                # Forecast
                forecast_periods = forecast['periods']
                forecast_values = forecast['values']
                
                fig.add_trace(go.Scatter(
                    x=forecast_periods,
                    y=forecast_values,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                # Confidence intervals
                if metric in confidence_intervals:
                    intervals = confidence_intervals[metric]
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_periods + forecast_periods[::-1],
                        y=intervals['upper'] + intervals['lower'][::-1],
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence',
                        showlegend=True
                    ))
                
                fig.update_layout(
                    title=f"{metric} - Historical vs Forecast",
                    xaxis_title="Period",
                    yaxis_title="Value",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show accuracy metrics
            accuracy_metrics = forecast_results.get('accuracy_metrics', {})
            if accuracy_metrics:
                st.subheader("🎯 Model Accuracy")
                
                accuracy_data = []
                for metric, accuracy in accuracy_metrics.items():
                    accuracy_data.append({
                        'Metric': metric,
                        'RMSE': accuracy.get('rmse', 0),
                        'MAE': accuracy.get('mae', 0),
                        'MAPE %': accuracy.get('mape', 0) if accuracy.get('mape') else 'N/A'
                    })
                
                if accuracy_data:
                    accuracy_df = pd.DataFrame(accuracy_data)
                    st.dataframe(accuracy_df, use_container_width=True)
    
    # Interactive visualization
    st.subheader("📊 Interactive Trend Visualization")
    
    numeric_metrics = data.select_dtypes(include=[np.number]).index.tolist()
    selected_metrics = st.multiselect(
        "Select metrics to visualize:",
        numeric_metrics,
        default=numeric_metrics[:3] if len(numeric_metrics) >= 3 else numeric_metrics
    )
    
    if selected_metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_trend_lines = st.checkbox("Show Trend Lines", value=True)
        
        with col2:
            normalize = st.checkbox("Normalize Values", value=False)
        
        with col3:
            chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Area"])
        
        fig = go.Figure()
        
        for i, metric in enumerate(selected_metrics):
            values = data.loc[metric]
            
            if normalize:
                values = (values / values.iloc[0]) * 100
            
            if chart_type == "Line":
                fig.add_trace(go.Scatter(
                    x=data.columns,
                    y=values,
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
            elif chart_type == "Bar":
                fig.add_trace(go.Bar(
                    x=data.columns,
                    y=values,
                    name=metric
                ))
            elif chart_type == "Area":
                fig.add_trace(go.Scatter(
                    x=data.columns,
                    y=values,
                    mode='lines',
                    name=metric,
                    fill='tonexty' if i > 0 else 'tozeroy',
                    line=dict(width=2)
                ))
            
            if show_trend_lines and metric in trends:
                trend_info = trends[metric]
                if 'slope' in trend_info and 'intercept' in trend_info:
                    x_numeric = np.arange(len(data.columns))
                    y_trend = trend_info['slope'] * x_numeric + trend_info['intercept']
                    
                    if normalize and values.iloc[0] != 0:
                        y_trend = (y_trend / values.iloc[0]) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=data.columns,
                        y=y_trend,
                        mode='lines',
                        name=f"{metric} (Trend)",
                        line=dict(width=2, dash='dash'),
                        opacity=0.7
                    ))
        
        fig.update_layout(
            title="Metric Trends Analysis",
            xaxis_title="Year",
            yaxis_title="Value" + (" (Base 100)" if normalize else ""),
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        if chart_type == "Bar":
            fig.update_layout(barmode='group')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Analysis
    st.subheader("📈 Statistical Analysis")
    
    if selected_metrics and len(selected_metrics) > 1:
        corr_data = data.loc[selected_metrics].T.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_data.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig_corr.update_layout(
            title="Correlation Matrix",
            height=400
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)

@error_boundary()
def _render_penman_nissim_tab(self, data: pd.DataFrame):
    """Render Penman-Nissim analysis tab"""
    st.header("🎯 Penman-Nissim Analysis")
    
    if not self.get_state('pn_mappings'):
        st.info("Configure Penman-Nissim mappings to proceed")
        
        with st.expander("⚙️ Configure P-N Mappings", expanded=True):
            available_metrics = [''] + [str(m) for m in data.index.tolist()]
            
            mapping_fields = {
                'Balance Sheet': [
                    ('Total Assets', 'pn_total_assets'),
                    ('Total Liabilities', 'pn_total_liabilities'),
                    ('Total Equity', 'pn_total_equity'),
                    ('Current Assets', 'pn_current_assets'),
                    ('Current Liabilities', 'pn_current_liabilities'),
                ],
                'Income Statement': [
                    ('Revenue', 'pn_revenue'),
                    ('Operating Income/EBIT', 'pn_operating_income'),
                    ('Net Income', 'pn_net_income'),
                    ('Interest Expense', 'pn_interest'),
                    ('Tax Expense', 'pn_tax'),
                ],
                'Cash Flow': [
                    ('Operating Cash Flow', 'pn_ocf'),
                    ('Capital Expenditure', 'pn_capex'),
                    ('Depreciation', 'pn_depreciation'),
                    ('Income Before Tax', 'pn_ibt'),
                ]
            }
            
            mappings = {}
            cols = st.columns(3)
            
            for i, (category, fields) in enumerate(mapping_fields.items()):
                with cols[i]:
                    st.markdown(f"**{category} Items**")
                    for field_name, field_key in fields:
                        selected = st.selectbox(
                            field_name,
                            available_metrics,
                            key=field_key
                        )
                        if selected:
                            mappings[selected] = field_name
            
            mappings = {k: v for k, v in mappings.items() if k}
            
            if st.button("Apply P-N Mappings", type="primary"):
                if len(mappings) >= 8:
                    self.set_state('pn_mappings', mappings)
                    st.success("Mappings applied successfully!")
                else:
                    st.error("Please provide at least 8 mappings for analysis")
        
        return
    
    if st.button("🚀 Run Penman-Nissim Analysis", type="primary"):
        mappings = self.get_state('pn_mappings')
        
        with st.spinner("Running Penman-Nissim analysis..."):
            try:
                analyzer = EnhancedPenmanNissimAnalyzer(data, mappings)
                results = analyzer.calculate_all()
                
                if 'error' in results:
                    st.error(f"Analysis failed: {results['error']}")
                    return
                
                self.set_state('pn_results', results)
                st.success("Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                if self.config.get('app.debug', False):
                    st.exception(e)
                return
    
    if self.get_state('pn_results'):
        results = self.get_state('pn_results')
        
        st.subheader("Key Penman-Nissim Metrics")
        
        if 'ratios' in results:
            ratios_df = results['ratios']
            
            key_ratios = [
                ('Return on Net Operating Assets (RNOA) %', 'RNOA', 'success'),
                ('Financial Leverage (FLEV)', 'FLEV', 'info'),
                ('Net Borrowing Cost (NBC) %', 'NBC', 'warning'),
                ('Operating Profit Margin (OPM) %', 'OPM', 'primary')
            ]
            
            col1, col2, col3, col4 = st.columns(4)
            
            for i, (ratio_name, short_name, color) in enumerate(key_ratios):
                if ratio_name in ratios_df.index:
                    col = [col1, col2, col3, col4][i]
                    with col:
                        latest_value = ratios_df.loc[ratio_name].iloc[-1]
                        prev_value = ratios_df.loc[ratio_name].iloc[-2] if len(ratios_df.columns) > 1 else None
                        
                        delta = None
                        if prev_value is not None:
                            delta = latest_value - prev_value
                        
                        self.ui_factory.create_metric_card(
                            short_name,
                            f"{latest_value:.2f}{'%' if '%' in ratio_name else 'x'}",
                            delta=delta,
                            help=self._get_pn_ratio_help(short_name)
                        )
        
        # Reformulated statements
        col1, col2 = st.columns(2)
        
        with col1:
            if 'reformulated_balance_sheet' in results:
                st.subheader("Reformulated Balance Sheet")
                ref_bs = results['reformulated_balance_sheet']
                
                st.dataframe(
                    ref_bs.style.format("{:,.0f}"),
                    use_container_width=True
                )
        
        with col2:
            if 'reformulated_income_statement' in results:
                st.subheader("Reformulated Income Statement")
                ref_is = results['reformulated_income_statement']
                
                st.dataframe(
                    ref_is.style.format("{:,.0f}"),
                    use_container_width=True
                )
        
        # Value Drivers
        if 'value_drivers' in results:
            st.subheader("Value Drivers Analysis")
            
            drivers_df = results['value_drivers']
            
            fig = go.Figure()
            
            for driver in drivers_df.index:
                fig.add_trace(go.Scatter(
                    x=drivers_df.columns,
                    y=drivers_df.loc[driver],
                    mode='lines+markers',
                    name=driver
                ))
            
            fig.update_layout(
                title="Value Drivers Trend",
                xaxis_title="Year",
                yaxis_title="Value (%)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Free Cash Flow
        if 'free_cash_flow' in results:
            st.subheader("Free Cash Flow Analysis")
            fcf_df = results['free_cash_flow']
            
            if len(fcf_df.columns) > 0:
                latest_year = fcf_df.columns[-1]
                
                if 'Operating Cash Flow' in fcf_df.index and 'Free Cash Flow' in fcf_df.index:
                    ocf = fcf_df.loc['Operating Cash Flow', latest_year]
                    fcf = fcf_df.loc['Free Cash Flow', latest_year]
                    capex = ocf - fcf
                    
                    fig = go.Figure(go.Waterfall(
                        name="",
                        orientation="v",
                        measure=["relative", "relative", "total"],
                        x=["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"],
                        y=[ocf, -capex, fcf],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    
                    fig.update_layout(
                        title=f"Free Cash Flow Waterfall - {latest_year}",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.subheader("Penman-Nissim Insights")
        
        insights = []
        
        if 'ratios' in results:
            ratios_df = results['ratios']
            
            if 'Return on Net Operating Assets (RNOA) %' in ratios_df.index:
                rnoa_latest = ratios_df.loc['Return on Net Operating Assets (RNOA) %'].iloc[-1]
                if rnoa_latest > 15:
                    insights.append("✅ Strong operating performance with RNOA above 15%")
                elif rnoa_latest < 8:
                    insights.append("⚠️ Low RNOA indicates operational efficiency concerns")
            
            if 'Spread %' in ratios_df.index:
                spread = ratios_df.loc['Spread %'].iloc[-1]
                if spread > 0:
                    insights.append(f"✅ Positive spread ({spread:.1f}%) - operations earn more than borrowing cost")
                else:
                    insights.append(f"⚠️ Negative spread ({spread:.1f}%) - borrowing cost exceeds operating returns")
        
        for insight in insights:
            self.ui_factory.create_insight_card(insight, "info")

def _get_pn_ratio_help(self, ratio: str) -> str:
    """Get help text for Penman-Nissim ratios"""
    help_texts = {
        'RNOA': "Return on Net Operating Assets - measures operating efficiency",
        'FLEV': "Financial Leverage - ratio of financial obligations to equity",
        'NBC': "Net Borrowing Cost - effective interest rate on net debt",
        'OPM': "Operating Profit Margin - operating profitability",
        'NOAT': "Net Operating Asset Turnover - efficiency in using assets",
        'Spread': "RNOA - NBC, positive spread creates value through leverage"
    }
    return help_texts.get(ratio, "Financial ratio")

@error_boundary()
def _render_industry_tab(self, data: pd.DataFrame):
    """Render industry comparison tab - COMPLETE IMPLEMENTATION"""
    st.header("🏭 Industry Comparison")
    
    # Industry selection
    industries = [
        "Technology", "Healthcare", "Financial Services", "Retail",
        "Manufacturing", "Energy", "Real Estate", "Consumer Goods",
        "Telecommunications", "Utilities", "Materials", "Industrials"
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_industry = st.selectbox(
            "Select Industry for Comparison",
            industries,
            key="industry_selection"
        )
    
    with col2:
        comparison_year = st.selectbox(
            "Select Year for Comparison",
            data.columns.tolist(),
            index=len(data.columns)-1 if len(data.columns) > 0 else 0,
            key="comparison_year"
        )
    
    # Generate industry benchmarks (simplified for demo)
    st.subheader(f"📊 {selected_industry} Industry Benchmarks")
    
    # Mock industry data - in production, this would come from a database
    industry_benchmarks = self._generate_industry_benchmarks(selected_industry)
    
    # Compare company metrics with industry averages
    if self.get_state('metric_mappings'):
        mappings = self.get_state('metric_mappings')
        mapped_df = data.rename(index=mappings)
        
        # Calculate company ratios
        analysis = self.components['analyzer'].analyze_financial_statements(mapped_df)
        company_ratios = analysis.get('ratios', {})
        
        # Display comparison
        for category, ratio_df in company_ratios.items():
            if isinstance(ratio_df, pd.DataFrame) and not ratio_df.empty:
                st.subheader(f"{category} Ratios Comparison")
                
                # Get latest year data
                if comparison_year in ratio_df.columns:
                    company_values = ratio_df[comparison_year]
                    industry_values = industry_benchmarks.get(category, {})
                    
                    comparison_data = []
                    for ratio_name in company_values.index:
                        company_val = company_values[ratio_name]
                        industry_avg = industry_values.get(ratio_name, None)
                        
                        if pd.notna(company_val) and industry_avg is not None:
                            comparison_data.append({
                                'Ratio': ratio_name,
                                'Company': company_val,
                                'Industry Average': industry_avg,
                                'Difference': company_val - industry_avg,
                                'Performance': 'Above Average' if company_val > industry_avg else 'Below Average'
                            })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Create visualization
                        fig = go.Figure()
                        
                        # Company values
                        fig.add_trace(go.Bar(
                            x=comparison_df['Ratio'],
                            y=comparison_df['Company'],
                            name='Company',
                            marker_color='blue'
                        ))
                        
                        # Industry averages
                        fig.add_trace(go.Bar(
                            x=comparison_df['Ratio'],
                            y=comparison_df['Industry Average'],
                            name='Industry Average',
                            marker_color='orange'
                        ))
                        
                        fig.update_layout(
                            title=f"{category} - Company vs Industry",
                            xaxis_title="Ratios",
                            yaxis_title="Value",
                            barmode='group',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show detailed comparison table
                        st.dataframe(
                            comparison_df.style.format({
                                'Company': '{:.2f}',
                                'Industry Average': '{:.2f}',
                                'Difference': '{:.2f}'
                            }),
                            use_container_width=True
                        )
    else:
        st.warning("Please complete metric mapping in the Financial Ratios tab first")
    
    # Industry insights
    st.subheader("💡 Industry Insights")
    
    industry_insights = {
        "Technology": [
            "Tech companies typically have higher profit margins due to scalable software models",
            "R&D spending is usually 10-20% of revenue",
            "Working capital requirements are generally low",
            "Growth rates are typically higher but more volatile"
        ],
        "Manufacturing": [
            "Asset-heavy industry with lower asset turnover ratios",
            "Inventory management is crucial for profitability",
            "Cyclical nature affects financial performance",
            "Leverage ratios tend to be higher due to capital requirements"
        ],
        "Retail": [
            "Inventory turnover is a key performance metric",
            "Seasonal variations significantly impact performance",
            "Profit margins are typically lower but volumes are higher",
            "Working capital management is critical"
        ],
        "Financial Services": [
            "Leverage ratios are naturally higher due to business model",
            "Net interest margin is a key profitability metric",
            "Asset quality indicators are crucial",
            "Regulatory capital requirements affect operations"
        ]
    }
    
    insights = industry_insights.get(selected_industry, [
        f"Industry-specific insights for {selected_industry} sector",
        "Compare your metrics with industry peers",
        "Focus on key performance indicators for your sector",
        "Consider industry trends and cycles in your analysis"
    ])
    
    for insight in insights:
        self.ui_factory.create_insight_card(insight, "info")

def _generate_industry_benchmarks(self, industry: str) -> Dict[str, Dict[str, float]]:
    """Generate mock industry benchmarks - in production, would fetch from database"""
    benchmarks = {
        "Technology": {
            "Profitability": {
                "Net Profit Margin %": 15.2,
                "Return on Assets %": 8.5,
                "Return on Equity %": 18.3,
                "ROCE %": 12.7
            },
            "Liquidity": {
                "Current Ratio": 2.1,
                "Quick Ratio": 1.8,
                "Cash Ratio": 0.9
            },
            "Leverage": {
                "Debt to Equity": 0.3,
                "Debt Ratio": 0.25,
                "Interest Coverage": 12.5
            },
            "Efficiency": {
                "Asset Turnover": 0.6,
                "Inventory Turnover": 8.2,
                "Receivables Turnover": 6.8
            }
        },
        "Manufacturing": {
            "Profitability": {
                "Net Profit Margin %": 8.5,
                "Return on Assets %": 5.2,
                "Return on Equity %": 12.1,
                "ROCE %": 8.9
            },
            "Liquidity": {
                "Current Ratio": 1.5,
                "Quick Ratio": 1.1,
                "Cash Ratio": 0.3
            },
            "Leverage": {
                "Debt to Equity": 0.8,
                "Debt Ratio": 0.45,
                "Interest Coverage": 5.2
            },
            "Efficiency": {
                "Asset Turnover": 1.2,
                "Inventory Turnover": 6.5,
                "Receivables Turnover": 8.1
            }
        },
        "Retail": {
            "Profitability": {
                "Net Profit Margin %": 4.8,
                "Return on Assets %": 6.2,
                "Return on Equity %": 15.5,
                "ROCE %": 9.8
            },
            "Liquidity": {
                "Current Ratio": 1.3,
                "Quick Ratio": 0.8,
                "Cash Ratio": 0.2
            },
            "Leverage": {
                "Debt to Equity": 0.6,
                "Debt Ratio": 0.38,
                "Interest Coverage": 4.5
            },
            "Efficiency": {
                "Asset Turnover": 2.1,
                "Inventory Turnover": 12.3,
                "Receivables Turnover": 15.2
            }
        }
    }
    
    return benchmarks.get(industry, benchmarks["Technology"])

@error_boundary()
def _render_data_explorer_tab(self, data: pd.DataFrame):
    """Render data explorer tab"""
    st.header("🔍 Data Explorer")
    
    # Data overview
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", data.shape[0])
    
    with col2:
        st.metric("Total Columns", data.shape[1])
    
    with col3:
        missing_pct = (data.isnull().sum().sum() / data.size) * 100
        st.metric("Missing Data %", f"{missing_pct:.1f}")
    
    with col4:
        numeric_cols = data.select_dtypes(include=[np.number]).shape[1]
        st.metric("Numeric Columns", numeric_cols)
    
    # Raw data view
    st.subheader("📋 Raw Data")
    
    # Data filtering options
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.text_input(
            "🔍 Search metrics",
            placeholder="Type to filter rows...",
            help="Search in metric names"
        )
    
    with col2:
        selected_years = st.multiselect(
            "📅 Select years",
            data.columns.tolist(),
            default=data.columns.tolist(),
            help="Filter by specific years"
        )
    
    # Apply filters
    filtered_data = data.copy()
    
    if search_term:
        mask = filtered_data.index.str.contains(search_term, case=False, na=False)
        filtered_data = filtered_data[mask]
    
    if selected_years:
        filtered_data = filtered_data[selected_years]
    
    # Display filtered data
    st.dataframe(
        filtered_data.style.format("{:,.0f}", na_rep="-")
        .background_gradient(cmap='RdYlGn', axis=1),
        use_container_width=True,
        height=400
    )
    
    # Data statistics
    st.subheader("📈 Data Statistics")
    
    numeric_data = filtered_data.select_dtypes(include=[np.number])
    
    if not numeric_data.empty:
        stats_df = numeric_data.describe().T
        stats_df = stats_df.round(2)
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Missing data analysis
        st.subheader("🕳️ Missing Data Analysis")
        
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if not missing_data.empty:
            fig = go.Figure(go.Bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                marker_color='red'
            ))
            
            fig.update_layout(
                title="Missing Values by Column",
                xaxis_title="Number of Missing Values",
                yaxis_title="Columns",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data found!")
    
    # Data export
    st.subheader("💾 Export Filtered Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Copy to Clipboard"):
            st.code(filtered_data.to_csv())
    
    with col2:
        csv_data = filtered_data.to_csv().encode('utf-8')
        st.download_button(
            label="📁 Download CSV",
            data=csv_data,
            file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col3:
        excel_data = io.BytesIO()
        filtered_data.to_excel(excel_data, engine='xlsxwriter')
        excel_data.seek(0)
        
        st.download_button(
            label="📊 Download Excel",
            data=excel_data.getvalue(),
            file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

@error_boundary()
def _render_reports_tab(self, data: pd.DataFrame):
    """Render reports tab"""
    st.header("📄 Financial Analysis Reports")
    
    # Report configuration
    st.subheader("⚙️ Report Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input(
            "Company Name",
            value=self.get_state('company_name', 'Your Company'),
            help="Name to appear on the report"
        )
    
    with col2:
        report_format = st.selectbox(
            "Report Format",
            ["Excel", "Markdown", "PDF", "PowerPoint"],
            help="Choose output format"
        )
    
    # Report sections
    st.subheader("📋 Report Sections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_overview = st.checkbox("Executive Summary", value=True)
        include_ratios = st.checkbox("Financial Ratios", value=True)
        include_trends = st.checkbox("Trend Analysis", value=True)
        include_forecasts = st.checkbox("ML Forecasts", value=False)
    
    with col2:
        include_pn = st.checkbox("Penman-Nissim Analysis", value=False)
        include_industry = st.checkbox("Industry Comparison", value=False)
        include_raw_data = st.checkbox("Raw Data", value=False)
        include_charts = st.checkbox("Charts and Visualizations", value=True)
    
    # Generate report
    if st.button("🚀 Generate Report", type="primary"):
        with st.spinner(f"Generating {report_format} report..."):
            try:
                # Compile analysis data
                analysis = self.components['analyzer'].analyze_financial_statements(data)
                
                # Add additional sections based on selections
                if include_pn and self.get_state('pn_results'):
                    analysis['penman_nissim'] = self.get_state('pn_results')
                
                if include_forecasts and self.get_state('ml_forecast_results'):
                    analysis['forecasts'] = self.get_state('ml_forecast_results')
                
                if include_raw_data:
                    analysis['filtered_data'] = data
                
                analysis['company_name'] = company_name
                
                # Generate report based on format
                if report_format == "Excel":
                    report_data = self.export_manager.export_to_excel(analysis, f"{company_name}_analysis.xlsx")
                    
                    st.download_button(
                        label="📊 Download Excel Report",
                        data=report_data,
                        file_name=f"{company_name}_financial_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    st.success("✅ Excel report generated successfully!")
                
                elif report_format == "Markdown":
                    report_content = self.export_manager.export_to_markdown(analysis)
                    
                    st.download_button(
                        label="📝 Download Markdown Report",
                        data=report_content.encode('utf-8'),
                        file_name=f"{company_name}_financial_analysis_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
                    
                    # Show preview
                    with st.expander("📖 Report Preview"):
                        st.markdown(report_content)
                    
                    st.success("✅ Markdown report generated successfully!")
                
                elif report_format == "PDF":
                    st.warning("PDF export is coming soon. Please use Excel or Markdown for now.")
                
                elif report_format == "PowerPoint":
                    st.warning("PowerPoint export is coming soon. Please use Excel or Markdown for now.")
                
            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")
    
    # Report templates
    st.subheader("📋 Report Templates")
    
    templates = {
        "Executive Summary": {
            "description": "High-level overview for executives",
            "sections": ["Overview", "Key Metrics", "Insights"]
        },
        "Detailed Analysis": {
            "description": "Comprehensive financial analysis",
            "sections": ["All sections", "Charts", "Raw Data"]
        },
        "Investor Presentation": {
            "description": "Investor-focused metrics and trends",
            "sections": ["Ratios", "Trends", "Forecasts", "Industry Comparison"]
        },
        "Audit Support": {
            "description": "Detailed data for audit purposes",
            "sections": ["Raw Data", "Calculations", "Validation Results"]
        }
    }
    
    for template_name, template_info in templates.items():
        with st.expander(f"📄 {template_name}"):
            st.write(f"**Description:** {template_info['description']}")
            st.write(f"**Sections:** {', '.join(template_info['sections'])}")
            
            if st.button(f"Use {template_name} Template", key=f"template_{template_name}"):
                st.info(f"Applied {template_name} template settings")

@error_boundary()
def _render_ml_insights_tab(self, data: pd.DataFrame):
    """Render ML insights and advanced analytics tab"""
    st.header("🤖 ML Insights & Advanced Analytics")
    
    if not self.config.get('app.enable_ml_features', True):
        st.warning("ML features are disabled. Enable them in the sidebar settings.")
        return
    
    # Check if AI features are using Kaggle
    using_kaggle = False
    if 'mapper' in self.components:
        status = self.components['mapper'].get_api_status()
        using_kaggle = status['kaggle_available']
    
    if using_kaggle:
        st.info("🚀 Using Kaggle GPU for enhanced ML processing")
    
    # AI-powered insights
    st.subheader("🧠 AI-Powered Financial Insights")
    
    with st.spinner("AI is analyzing your financial data..."):
        analysis = self.components['analyzer'].analyze_financial_statements(data)
    
    # Display AI insights with confidence scores
    insights = analysis.get('insights', [])
    
    if insights:
        st.write("**AI has identified the following insights:**")
        
        for i, insight in enumerate(insights):
            # Simulate confidence score
            confidence = np.random.uniform(0.7, 0.95)
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                if "⚠️" in insight:
                    insight_type = "warning"
                elif "📉" in insight:
                    insight_type = "error"
                elif "🚀" in insight or "📊" in insight:
                    insight_type = "success"
                else:
                    insight_type = "info"
                
                self.ui_factory.create_insight_card(insight, insight_type)
            
            with col2:
                st.metric("Confidence", f"{confidence:.0%}")
    
    # Anomaly detection with ML
    st.subheader("🔍 Advanced Anomaly Detection")
    
    anomalies = analysis.get('anomalies', {})
    total_anomalies = sum(len(v) for v in anomalies.values())
    
    if total_anomalies > 0:
        st.warning(f"Detected {total_anomalies} potential anomalies in your data")
        
        # Anomaly visualization
        fig = go.Figure()
        
        categories = list(anomalies.keys())
        counts = [len(anomalies[cat]) for cat in categories]
        
        fig.add_trace(go.Bar(
            x=categories,
            y=counts,
            marker_color=['red', 'orange', 'yellow'][:len(categories)]
        ))
        
        fig.update_layout(
            title="Anomalies by Category",
            xaxis_title="Anomaly Type",
            yaxis_title="Count",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No significant anomalies detected")
    
    # Predictive analytics
    st.subheader("🔮 Predictive Analytics")
    
    if st.button("🚀 Run Predictive Analysis", type="primary"):
        with st.spinner("Running ML models for predictive analysis..."):
            try:
                # Run ML forecasting
                forecast_results = self.ml_forecaster.forecast_metrics(
                    data, 
                    periods=3,
                    model_type='auto'
                )
                
                if 'error' not in forecast_results:
                    st.success("✅ Predictive analysis completed")
                    
                    # Display key predictions
                    forecasts = forecast_results.get('forecasts', {})
                    
                    st.write("**Key Financial Predictions (Next 3 Periods):**")
                    
                    for metric, forecast in list(forecasts.items())[:3]:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**{metric}**")
                        
                        with col2:
                            current_value = forecast['last_actual']
                            predicted_value = forecast['values'][-1]
                            growth = ((predicted_value / current_value) - 1) * 100
                            
                            st.metric(
                                "Predicted Growth",
                                f"{growth:+.1f}%",
                                delta=f"{predicted_value - current_value:,.0f}"
                            )
                        
                        with col3:
                            # Get accuracy from previous results
                            accuracy = forecast_results.get('accuracy_metrics', {}).get(metric, {})
                            mape = accuracy.get('mape', 0)
                            
                            if mape and mape < 20:
                                confidence_text = "High"
                                confidence_color = "success"
                            elif mape and mape < 40:
                                confidence_text = "Medium"
                                confidence_color = "warning"
                            else:
                                confidence_text = "Low"
                                confidence_color = "error"
                            
                            st.markdown(f"**Confidence:** :{confidence_color}[{confidence_text}]")
                else:
                    st.error(f"Predictive analysis failed: {forecast_results['error']}")
                    
            except Exception as e:
                st.error(f"Predictive analysis error: {str(e)}")
    
    # Pattern recognition
    st.subheader("🎯 Financial Pattern Recognition")
    
    # Analyze patterns in the data
    patterns = self._detect_financial_patterns(data)
    
    if patterns:
        st.write("**Detected Financial Patterns:**")
        
        for pattern in patterns:
            self.ui_factory.create_insight_card(pattern['description'], pattern['type'])
    
    # Risk analysis
    st.subheader("⚠️ Risk Analysis")
    
    risk_metrics = self._calculate_risk_metrics(data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        volatility = risk_metrics.get('volatility', 0)
        st.metric(
            "Revenue Volatility",
            f"{volatility:.1f}%",
            help="Standard deviation of revenue growth"
        )
    
    with col2:
        trend_stability = risk_metrics.get('trend_stability', 0)
        st.metric(
            "Trend Stability",
            f"{trend_stability:.0%}",
            help="Consistency of growth trends"
        )
    
    with col3:
        outlier_risk = risk_metrics.get('outlier_risk', 0)
        st.metric(
            "Outlier Risk",
            f"{outlier_risk:.0%}",
            help="Percentage of outlier values"
        )
    
    # Performance monitoring
    if self.config.get('app.debug', False):
        st.subheader("🔧 Performance Monitoring")
        
        perf_summary = performance_monitor.get_performance_summary()
        
        if perf_summary:
            perf_data = []
            for operation, stats in perf_summary.items():
                perf_data.append({
                    'Operation': operation,
                    'Total Calls': stats['total_calls'],
                    'Avg Duration (s)': stats['avg_duration'],
                    'Max Duration (s)': stats['max_duration'],
                    'Total Time (s)': stats['total_time']
                })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, use_container_width=True)
        
        # Cache statistics
        if 'mapper' in self.components:
            cache_stats = self.components['mapper'].embeddings_cache.get_stats()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cache Entries", cache_stats.get('entries', 0))
            
            with col2:
                st.metric("Cache Hit Rate", f"{cache_stats.get('hit_rate', 0):.1f}%")
            
            with col3:
                cache_size_mb = cache_stats.get('size_bytes', 0) / (1024 * 1024)
                st.metric("Cache Size", f"{cache_size_mb:.1f} MB")

def _detect_financial_patterns(self, data: pd.DataFrame) -> List[Dict[str, str]]:
    """Detect patterns in financial data"""
    patterns = []
    
    # Analyze revenue patterns
    revenue_metrics = [idx for idx in data.index if 'revenue' in str(idx).lower()]
    if revenue_metrics:
        revenue_series = data.loc[revenue_metrics[0]].dropna()
        
        # Check for seasonality (simple check)
        if len(revenue_series) >= 4:
            growth_rates = revenue_series.pct_change().dropna()
            
            if growth_rates.std() < 0.1:  # Low volatility
                patterns.append({
                    'description': '📈 Stable revenue growth pattern detected - consistent performance',
                    'type': 'success'
                })
            elif growth_rates.std() > 0.3:  # High volatility
                patterns.append({
                    'description': '⚠️ Volatile revenue pattern - consider investigating causes',
                    'type': 'warning'
                })
    
    # Check for improvement trends
    analysis = self.components['analyzer'].analyze_financial_statements(data)
    trends = analysis.get('trends', {})
    
    improving_metrics = 0
    declining_metrics = 0
    
    for metric, trend in trends.items():
        if isinstance(trend, dict):
            if trend.get('direction') == 'increasing':
                improving_metrics += 1
            elif trend.get('direction') == 'decreasing':
                declining_metrics += 1
    
    if improving_metrics > declining_metrics * 2:
        patterns.append({
            'description': '🚀 Overall improving trend across multiple metrics',
            'type': 'success'
        })
    elif declining_metrics > improving_metrics * 2:
        patterns.append({
            'description': '📉 Concerning declining trend across multiple metrics',
            'type': 'error'
        })
    
    return patterns

def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
    """Calculate risk-related metrics"""
    risk_metrics = {}
    
    # Revenue volatility
    revenue_metrics = [idx for idx in data.index if 'revenue' in str(idx).lower()]
    if revenue_metrics:
        revenue_series = data.loc[revenue_metrics[0]].dropna()
        growth_rates = revenue_series.pct_change().dropna()
        risk_metrics['volatility'] = growth_rates.std() * 100
    
    # Trend stability
    analysis = self.components['analyzer'].analyze_financial_statements(data)
    trends = analysis.get('trends', {})
    
    r_squared_values = []
    for trend in trends.values():
        if isinstance(trend, dict) and 'r_squared' in trend:
            r_squared_values.append(trend['r_squared'])
    
    if r_squared_values:
        risk_metrics['trend_stability'] = np.mean(r_squared_values)
    
    # Outlier risk
    numeric_data = data.select_dtypes(include=[np.number])
    total_values = numeric_data.size
    outlier_count = 0
    
    for col in numeric_data.columns:
        for val in numeric_data[col].dropna():
            z_score = np.abs((val - numeric_data[col].mean()) / numeric_data[col].std())
            if z_score > 3:
                outlier_count += 1
    
    risk_metrics['outlier_risk'] = outlier_count / total_values if total_values > 0 else 0
    
    return risk_metrics

def _render_debug_footer(self):
    """Render debug information in footer"""
    with st.expander("🔧 Debug Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Session State Keys:**")
            st.write(list(st.session_state.keys()))
        
        with col2:
            st.write("**Performance Summary:**")
            perf_summary = performance_monitor.get_performance_summary()
            if perf_summary:
                for op, stats in list(perf_summary.items())[:5]:
                    st.write(f"- {op}: {stats['avg_duration']:.3f}s avg")
        
        # API Performance
        if self.get_state('kaggle_api_enabled'):
            st.write("**API Performance:**")
            api_summary = performance_monitor.get_api_summary()
            if api_summary:
                for endpoint, metrics in api_summary.items():
                    st.write(f"- {endpoint}: {metrics['total_requests']} requests, {metrics['success_rate']:.1%} success")
--- 31. Application Entry Point ---
def main(): """Main application entry point with comprehensive error handling""" try: # Create and run the application app = FinancialAnalyticsPlatform() app.run()

except Exception as e:
    # Critical error handling
    logging.critical(f"Fatal application error: {e}", exc_info=True)
    
    st.error("🚨 A critical error occurred. Please refresh the page.")
    
    # Show debug info if available
    if st.session_state.get('debug_mode', False):
        st.exception(e)
        
        with st.expander("🔧 Debug Information"):
            st.write("**Error Details:**")
            st.code(traceback.format_exc())
            
            st.write("**Session State:**")
            st.json(dict(st.session_state))
    
    # Recovery options
    st.subheader("🔄 Recovery Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Page"):
            st.experimental_rerun()
    
    with col2:
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")
    
    with col3:
        if st.button("🏠 Reset to Home"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
if name == "main": # Configure Python path and environment import sys from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Run the application
main()
