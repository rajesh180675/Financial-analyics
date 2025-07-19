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

--- 18. Enhanced AI Mapping System with Robust Kaggle Integration ---
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

--- 19. Fuzzy Mapping Fallback ---
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

--- 20. Enhanced Penman-Nissim Analyzer ---
class EnhancedPenmanNissimAnalyzer:
    """Enhanced Penman-Nissim analyzer with advanced features and robustness"""
    def __init__(self, df: pd.DataFrame, mappings: Dict[str, str]):
        self.df = df
        self.mappings = mappings
        self.logger = LoggerFactory.get_logger('PenmanNissim')
        self.validation_results = {}
        self.calculation_metadata = {}
        
        self._initialize_core_analyzer()
        self._validate_input_data()
    
    def _validate_input_data(self):
        """Comprehensive validation of input data"""
        validation = {
            'data_quality_score': 0,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check for essential data availability
        essential_metrics = ['Total Assets', 'Total Liabilities', 'Total Equity', 'Revenue']
        missing_essentials = []
        
        for metric in essential_metrics:
            source_metric = self._find_source_metric(metric)
            if not source_metric or source_metric not in self.df.index:
                missing_essentials.append(metric)
        
        if missing_essentials:
            validation['issues'].append(f"Missing essential metrics: {', '.join(missing_essentials)}")
        
        # Check accounting equation
        total_assets = self._get_metric_series('Total Assets')
        total_liabilities = self._get_metric_series('Total Liabilities')
        total_equity = self._get_metric_series('Total Equity')
        
        if all(x is not None for x in [total_assets, total_liabilities, total_equity]):
            for year in self.df.columns:
                if year in total_assets.index and year in total_liabilities.index and year in total_equity.index:
                    assets = total_assets[year]
                    liabilities = total_liabilities[year]
                    equity = total_equity[year]
                    
                    if all(pd.notna([assets, liabilities, equity])):
                        difference = abs(assets - (liabilities + equity))
                        tolerance = assets * 0.05  # 5% tolerance
                        
                        if difference > tolerance:
                            validation['warnings'].append(f"Accounting equation imbalance in {year}: {difference:,.0f}")
        
        # Calculate quality score
        total_mappings = len(self.mappings)
        essential_mappings = len([m for m in essential_metrics if self._find_source_metric(m)])
        data_completeness = self._calculate_data_completeness()
        
        validation['data_quality_score'] = (
            (essential_mappings / len(essential_metrics)) * 0.4 +
            (total_mappings / 15) * 0.3 +  # Assume 15 optimal mappings
            data_completeness * 0.3
        ) * 100
        
        self.validation_results = validation
    
    def _find_source_metric(self, target_metric: str) -> Optional[str]:
        """Find source metric that maps to target"""
        for source, target in self.mappings.items():
            if target == target_metric:
                return source
        return None
    
    def _get_metric_series(self, target_metric: str) -> Optional[pd.Series]:
        """Get series for a target metric with error handling"""
        source_metric = self._find_source_metric(target_metric)
        if source_metric and source_metric in self.df.index:
            return self.df.loc[source_metric]
        return None
    
    def _calculate_data_completeness(self) -> float:
        """Calculate overall data completeness"""
        mapped_metrics = [self._find_source_metric(target) for target in self.mappings.values()]
        mapped_metrics = [m for m in mapped_metrics if m and m in self.df.index]
        
        if not mapped_metrics:
            return 0.0
        
        total_cells = len(mapped_metrics) * len(self.df.columns)
        non_null_cells = sum(self.df.loc[metric].notna().sum() for metric in mapped_metrics)
        
        return non_null_cells / total_cells if total_cells > 0 else 0.0
    
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
                results = self.core_analyzer.calculate_all()
                # Add our validation results
                results['validation_results'] = self.validation_results
                results['calculation_metadata'] = self.calculation_metadata
                return results
            except Exception as e:
                self.logger.error(f"Error in core calculate_all: {e}")
        
        return self._fallback_calculate_all()
    
    def _fallback_calculate_all(self):
        """Fallback implementation of Penman-Nissim calculations"""
        try:
            mapped_df = self.df.rename(index=self.mappings)
            
            results = {
                'reformulated_balance_sheet': self._reformulate_balance_sheet_enhanced(mapped_df),
                'reformulated_income_statement': self._reformulate_income_statement(mapped_df),
                'ratios': self._calculate_ratios_enhanced(mapped_df),
                'free_cash_flow': self._calculate_free_cash_flow(mapped_df),
                'value_drivers': self._calculate_value_drivers(mapped_df),
                'validation_results': self.validation_results,
                'calculation_metadata': self.calculation_metadata
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in fallback calculations: {e}")
            return {'error': str(e)}
    
    def _reformulate_balance_sheet_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced balance sheet reformulation with multiple calculation methods"""
        reformulated = pd.DataFrame(index=df.columns)
        metadata = {}
        
        # Method 1: Direct mapping approach
        try:
            total_assets = self._get_safe_series(df, 'Total Assets')
            cash = self._get_safe_series(df, 'Cash and Cash Equivalents', default_zero=True)
            st_debt = self._get_safe_series(df, 'Short-term Debt', default_zero=True)
            lt_debt = self._get_safe_series(df, 'Long-term Debt', default_zero=True)
            current_liab = self._get_safe_series(df, 'Current Liabilities')
            total_equity = self._get_safe_series(df, 'Total Equity')
            
            # Calculate NOA (Net Operating Assets)
            financial_assets = cash
            financial_liabilities = st_debt + lt_debt
            
            reformulated['Financial Assets'] = financial_assets
            reformulated['Financial Liabilities'] = financial_liabilities
            reformulated['Net Financial Assets'] = financial_assets - financial_liabilities
            reformulated['Net Operating Assets'] = total_assets - reformulated['Net Financial Assets']
            reformulated['Common Equity'] = total_equity
            
            # Validation: NOA + NFA should equal Common Equity
            validation_check = reformulated['Net Operating Assets'] + reformulated['Net Financial Assets'] - reformulated['Common Equity']
            metadata['balance_check'] = validation_check.abs().max()
            
        except Exception as e:
            self.logger.error(f"Enhanced BS reformulation failed: {e}")
            # Fallback to simple method
            reformulated = self._reformulate_balance_sheet(df)
        
        self.calculation_metadata['balance_sheet'] = metadata
        return reformulated
    
    def _get_safe_series(self, df: pd.DataFrame, target_metric: str, default_zero: bool = False) -> pd.Series:
        """Safely get a series with fallback options"""
        source_metric = self._find_source_metric(target_metric)
        
        if source_metric and source_metric in df.index:
            series = df.loc[source_metric].fillna(0 if default_zero else np.nan)
            return series
        elif default_zero:
            return pd.Series(0, index=df.columns)
        else:
            raise ValueError(f"Required metric '{target_metric}' not found")
    
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
        
        if 'Operating Income' in df.index:
            reformulated['Operating Income'] = df.loc['Operating Income']
        elif 'EBIT' in df.index:
            reformulated['Operating Income'] = df.loc['EBIT']
        
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
    
    def _calculate_ratios_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced ratio calculation with alternative methods and validation"""
        ratios = pd.DataFrame(index=df.columns)
        metadata = {}
        
        try:
            ref_bs = self._reformulate_balance_sheet_enhanced(df)
            ref_is = self._reformulate_income_statement(df)
            
            # Enhanced RNOA calculation with alternatives
            if 'Operating Income After Tax' in ref_is.index and 'Net Operating Assets' in ref_bs.index:
                oiat = ref_is.loc['Operating Income After Tax']
                noa = ref_bs.loc['Net Operating Assets']
                
                # Use average NOA for more accurate calculation
                avg_noa = noa.rolling(window=2).mean()
                avg_noa.iloc[0] = noa.iloc[0]  # Fill first value
                
                ratios['Return on Net Operating Assets (RNOA) %'] = (oiat / avg_noa.replace(0, np.nan)) * 100
                metadata['rnoa_method'] = 'average_noa'
            
            # Enhanced FLEV with market value consideration
            if 'Net Financial Assets' in ref_bs.index and 'Common Equity' in ref_bs.index:
                nfa = ref_bs.loc['Net Financial Assets']
                ce = ref_bs.loc['Common Equity']
                
                avg_ce = ce.rolling(window=2).mean()
                avg_ce.iloc[0] = ce.iloc[0]
                
                ratios['Financial Leverage (FLEV)'] = -nfa / avg_ce.replace(0, np.nan)
                metadata['flev_method'] = 'average_equity'
            
            # Enhanced NBC with tax adjustment
            if 'Net Financial Expense' in ref_is.index and 'Net Financial Assets' in ref_bs.index:
                nfe = ref_is.loc['Net Financial Expense']
                nfa = ref_bs.loc['Net Financial Assets']
                
                # Tax-adjusted NBC
                tax_rate = self._estimate_tax_rate(df)
                if tax_rate is not None:
                    after_tax_nfe = nfe * (1 - tax_rate)
                    ratios['Net Borrowing Cost (NBC) %'] = (-after_tax_nfe / nfa.replace(0, np.nan)) * 100
                    metadata['nbc_tax_adjusted'] = True
                else:
                    ratios['Net Borrowing Cost (NBC) %'] = (-nfe / nfa.replace(0, np.nan)) * 100
                    metadata['nbc_tax_adjusted'] = False
            
            # Additional advanced ratios
            self._calculate_advanced_ratios(ratios, ref_bs, ref_is, df)
            
            # Cross-validation
            self._validate_ratios(ratios)
            
        except Exception as e:
            self.logger.error(f"Enhanced ratio calculation failed: {e}")
            ratios = self._calculate_ratios(df)
        
        self.calculation_metadata['ratios'] = metadata
        return ratios.T
    
    def _estimate_tax_rate(self, df: pd.DataFrame) -> Optional[float]:
        """Estimate effective tax rate"""
        try:
            tax_expense = self._get_safe_series(df, 'Tax Expense')
            income_before_tax = self._get_safe_series(df, 'Income Before Tax')
            
            if tax_expense is not None and income_before_tax is not None:
                tax_rate = (tax_expense / income_before_tax.replace(0, np.nan)).median()
                return max(0, min(1, tax_rate))  # Constrain between 0 and 1
        except Exception:
            pass
        return None
    
    def _calculate_advanced_ratios(self, ratios: pd.DataFrame, ref_bs: pd.DataFrame, 
                                 ref_is: pd.DataFrame, df: pd.DataFrame):
        """Calculate additional advanced ratios"""
        try:
            # Operating Asset Turnover (more granular)
            if 'Revenue' in df.index and 'Net Operating Assets' in ref_bs.index:
                revenue = df.loc[self._find_source_metric('Revenue')]
                noa = ref_bs.loc['Net Operating Assets']
                ratios['Operating Asset Turnover'] = revenue / noa.replace(0, np.nan)
            
            # Financial Leverage Spread
            if 'Return on Net Operating Assets (RNOA) %' in ratios.index and 'Net Borrowing Cost (NBC) %' in ratios.index:
                ratios['Leverage Spread %'] = ratios['Return on Net Operating Assets (RNOA) %'] - ratios['Net Borrowing Cost (NBC) %']
            
            # Operating Margin Stability (coefficient of variation)
            if 'Operating Profit Margin (OPM) %' in ratios.index:
                opm_series = ratios['Operating Profit Margin (OPM) %']
                if len(opm_series) > 1:
                    cv = opm_series.std() / opm_series.mean() if opm_series.mean() != 0 else np.nan
                    ratios['OPM Stability (CV)'] = pd.Series(cv, index=ratios.index)
        
        except Exception as e:
            self.logger.warning(f"Advanced ratios calculation warning: {e}")
    
    def _validate_ratios(self, ratios: pd.DataFrame):
        """Validate calculated ratios for reasonableness"""
        validation_issues = []
        
        # Check for extreme values
        for ratio_name in ratios.index:
            series = ratios.loc[ratio_name]
            
            # Check for extreme outliers (beyond 3 standard deviations)
            if len(series) > 2:
                z_scores = np.abs((series - series.mean()) / series.std())
                outliers = z_scores > 3
                if outliers.any():
                    validation_issues.append(f"Extreme values detected in {ratio_name}")
        
        # Industry reasonableness checks
        if 'Return on Net Operating Assets (RNOA) %' in ratios.index:
            rnoa = ratios.loc['Return on Net Operating Assets (RNOA) %']
            if (rnoa > 100).any():
                validation_issues.append("RNOA > 100% detected - check data quality")
            if (rnoa < -50).any():
                validation_issues.append("Very negative RNOA detected - check for data errors")
        
        if validation_issues:
            self.validation_results['ratio_warnings'] = validation_issues
    
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
        
        return fcf.T
    
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
        
        return drivers.T

--- 21. Manual Mapping Interface ---
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

--- 22. Machine Learning Forecasting Module ---
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

--- 23. Natural Language Query Processor ---
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

--- 24. Collaboration Manager ---
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

--- 25. Tutorial System ---
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
                    st.experimental_rerun()
            with col2:
                if st.button("🏠 Reset Application", key=f"reset_{func.__name__}"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.experimental_rerun()
            
            return None
    return wrapper

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
        """Export application logs"""
        try:
            log_dir = Path("logs")
            if log_dir.exists():
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

    @error_boundary()
    @critical_method
    def run(self):
        """Main application entry point"""
        try:
            st.set_page_config(
                page_title="Elite Financial Analytics Platform v5.1",
                page_icon="💹",
                layout="wide",
                initial_sidebar_state="expanded"
            )

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
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            st.error("An unexpected error occurred. Please refresh the page.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Retry"):
                    st.experimental_rerun()
            
            with col2:
                if st.button("🔧 Auto Recovery"):
                    if self._auto_recovery_attempt():
                        st.success("Recovery successful!")
                        st.experimental_rerun()
                    else:
                        st.error("Recovery failed. Please refresh manually.")
            
            with col3:
                if st.button("🗑️ Clear All & Restart"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.experimental_rerun()
            
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
            
            if st.sidebar.button("Clear Cache"):
                self._clear_all_caches()
                st.success("Cache cleared!")
            
            if st.sidebar.button("Reset Configuration"):
                self._reset_configuration()
            
            if st.sidebar.button("Export Logs"):
                self._export_logs()
            
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
                            if df is not None and not df.empty:
                                df = self._clean_dataframe(df)
                                self._inspect_dataframe(df, extracted_name)
                                all_dataframes.append(df)
                                file_info.append({
                                    'name': extracted_name,
                                    'source': f"{file.name} (compressed)",
                                    'shape': df.shape
                                })
                    else:
                        df = self._parse_single_file(file)
                        if df is not None and not df.empty:
                            df = self._clean_dataframe(df)
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
        """Parse a single file and return DataFrame"""
        try:
            file_ext = Path(file.name).suffix.lower()
            self.logger.info(f"Attempting to parse file: {file.name} with extension: {file_ext}")

            sample = file.read(1024).decode('utf-8', errors='ignore')
            file.seek(0)
            
            self.logger.info(f"File content starts with: {sample[:100]}")
            
            if '<html' in sample.lower() or '<table' in sample.lower():
                self.logger.info(f"{file.name} appears to be HTML content")
                try:
                    tables = pd.read_html(file)
                    if tables:
                        df = max(tables, key=len)
                        
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [' >> '.join(str(level).strip() for level in col if str(level) != 'nan').strip(' >> ') 
                                         for col in df.columns]
                        
                        df.columns = [str(col).strip().replace('  ', ' ').replace('Fice >>', 'Finance >>') for col in df.columns]
                        
                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                        
                        potential_index_cols = ['Particulars', 'Description', 'Items', 'Metric', 'Item']
                        index_col_found = False
                        
                        for col in potential_index_cols:
                            matching_cols = [c for c in df.columns if col.lower() in c.lower()]
                            if matching_cols:
                                df = df.set_index(matching_cols[0])
                                index_col_found = True
                                break
                        
                        if not index_col_found and len(df.columns) > 0:
                            first_col = df.columns[0]
                            if df[first_col].dtype == object:
                                df = df.set_index(first_col)
                        
                        return df
                except Exception as e:
                    self.logger.error(f"HTML parsing failed for {file.name}: {e}")
                    return None
                    
            elif file_ext == '.csv':
                try:
                    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    for encoding in encodings:
                        try:
                            file.seek(0)
                            df = pd.read_csv(file, encoding=encoding, index_col=0)
                            self.logger.info(f"Successfully parsed CSV with {encoding} encoding")
                            return df
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            self.logger.warning(f"CSV parsing failed with {encoding}: {e}")
                            continue
                    
                    file.seek(0)
                    df = pd.read_csv(file, encoding='utf-8', index_col=None)
                    if df.iloc[:, 0].dtype == object:
                        df = df.set_index(df.columns[0])
                    return df
                    
                except Exception as e:
                    self.logger.error(f"CSV parsing failed for {file.name}: {e}")
                    return None
                    
            elif file_ext in ['.xls', '.xlsx']:
                for engine in ['openpyxl', 'xlrd']:
                    try:
                        file.seek(0)
                        self.logger.info(f"Trying {engine} engine for {file.name}")
                        
                        try:
                            df = pd.read_excel(file, index_col=0, engine=engine)
                            if not df.empty:
                                return df
                        except Exception:
                            file.seek(0)
                            df = pd.read_excel(file, index_col=None, engine=engine)
                            if not df.empty and df.iloc[:, 0].dtype == object:
                                df = df.set_index(df.columns[0])
                                return df
                            
                    except Exception as e:
                        self.logger.warning(f"{engine} engine failed for {file.name}: {e}")
                        continue
                
                try:
                    file.seek(0)
                    tables = pd.read_html(file)
                    if tables:
                        df = max(tables, key=len)
                        
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [' >> '.join(str(level).strip() for level in col if str(level) != 'nan').strip(' >> ') 
                                         for col in df.columns]
                        
                        df.columns = [str(col).strip().replace('  ', ' ').replace('Fice >>', 'Finance >>') for col in df.columns]
                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                        
                        potential_index_cols = ['Particulars', 'Description', 'Items', 'Metric', 'Item']
                        for col in potential_index_cols:
                            matching_cols = [c for c in df.columns if col.lower() in c.lower()]
                            if matching_cols:
                                df = df.set_index(matching_cols[0])
                                break
                        else:
                            if len(df.columns) > 0 and df[df.columns[0]].dtype == object:
                                df = df.set_index(df.columns[0])
                        
                        self.logger.info(f"Successfully parsed {file.name} as HTML table")
                        return df
                        
                except Exception as e:
                    self.logger.error(f"HTML fallback failed for {file.name}: {e}")
                    return None
            
            elif file_ext in ['.html', '.htm']:
                try:
                    tables = pd.read_html(file)
                    if tables:
                        df = max(tables, key=len)
                        
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [' >> '.join(str(level).strip() for level in col if str(level) != 'nan').strip(' >> ') 
                                         for col in df.columns]
                        
                        df.columns = [str(col).strip().replace('  ', ' ').replace('Fice >>', 'Finance >>') for col in df.columns]
                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                        
                        potential_index_cols = ['Particulars', 'Description', 'Items', 'Metric', 'Item']
                        for col in potential_index_cols:
                            matching_cols = [c for c in df.columns if col.lower() in c.lower()]
                            if matching_cols:
                                df = df.set_index(matching_cols[0])
                                break
                        else:
                            if len(df.columns) > 0 and df[df.columns[0]].dtype == object:
                                df = df.set_index(df.columns[0])
                        
                        return df
                except Exception as e:
                    self.logger.error(f"HTML parsing failed for {file.name}: {e}")
                    return None
            
            else:
                self.logger.warning(f"Unsupported file type: {file_ext}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error parsing {file.name}: {e}")
            return None
        
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare dataframe for analysis"""
        try:
            cleaned_df = self._standardize_dataframe(df)

            unnamed_cols = [col for col in cleaned_df.columns if 'Unnamed' in str(col)]
            if unnamed_cols:
                cleaned_df = cleaned_df.drop(columns=unnamed_cols)
                self.logger.info(f"Removed {len(unnamed_cols)} unnamed columns")
            
            for col in cleaned_df.columns:
                try:
                    cleaned_df[col] = cleaned_df[col].replace({
                        '-': np.nan,
                        '': np.nan,
                        'NA': np.nan,
                        'None': np.nan
                    })
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Could not convert column {col} to numeric: {e}")
            
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Error cleaning dataframe: {e}")
            return df
        
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame structure specifically for financial data"""
        try:
            std_df = df.copy()

            if isinstance(std_df.columns, pd.Index):
                std_df.columns = [
                    str(col).strip()
                    .replace('Fice >>', 'Finance >>')
                    .replace('  ', ' ')
                    .replace('\n', ' ')
                    .replace('\t', ' ')
                    for col in std_df.columns
                ]
            
            if std_df.index.duplicated().any():
                self.logger.info("Found rows with similar names - making indices unique")
                
                new_index = []
                seen_indices = {}
                
                for idx in std_df.index:
                    idx_str = str(idx) if not pd.isna(idx) else "EmptyIndex"
                    
                    if idx_str in seen_indices:
                        seen_indices[idx_str] += 1
                        unique_idx = f"{idx_str}_v{seen_indices[idx_str]}"
                    else:
                        seen_indices[idx_str] = 0
                        unique_idx = idx_str
                    
                    new_index.append(unique_idx)
                
                std_df.index = new_index
                self.logger.info(f"Made {sum(v for v in seen_indices.values() if v > 0)} indices unique")
            
            before_count = len(std_df)
            std_df = std_df.dropna(how='all')
            after_count = len(std_df)
            
            if before_count != after_count:
                self.logger.info(f"Removed {before_count - after_count} completely empty rows")
            
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
        """Merge dataframes preserving all financial line items"""
        try:
            analysis_mode = self.get_state('analysis_mode', 'Standalone Analysis')

            if analysis_mode == "Standalone Analysis":
                self.logger.info("Using standalone analysis mode - preserving all financial data")
                
                processed_dfs = []
                
                for i, df in enumerate(dataframes):
                    df_copy = df.copy()
                    
                    statement_type = self._detect_statement_type(df_copy)
                    self.logger.info(f"Processing {statement_type} with {len(df_copy)} line items")
                    
                    new_index = []
                    for idx in df_copy.index:
                        if pd.isna(idx) or str(idx).strip() == '':
                            new_idx = f"{statement_type}_EmptyRow_{df_copy.index.get_loc(idx)}"
                        else:
                            clean_idx = str(idx).strip()
                            new_idx = f"{statement_type}::{clean_idx}"
                        new_index.append(new_idx)
                    
                    df_copy.index = new_index
                    processed_dfs.append(df_copy)
                
                merged_df = pd.concat(processed_dfs, axis=0, sort=False)
                
                total_original = sum(len(df) for df in dataframes)
                self.logger.info(f"Successfully preserved all {len(merged_df)} line items (original: {total_original})")
                
                company_info = self._extract_company_info(merged_df)
                if company_info:
                    self.logger.info(f"Analyzing data for: {company_info['name']}")
                    self.set_state('company_name', company_info['name'])
                
                return merged_df
                
            elif analysis_mode == "Benchmark Comparison":
                self.logger.info("Using benchmark comparison mode")
                
                merged_df = self._merge_standalone_data(dataframes)
                
                benchmark_data = self.get_state('benchmark_data')
                if benchmark_data is not None:
                    benchmark_copy = benchmark_data.copy()
                    benchmark_copy.index = [f"Benchmark::{idx}" for idx in benchmark_copy.index]
                    merged_df = pd.concat([merged_df, benchmark_copy], axis=0)
                
                return merged_df
            
            return pd.concat(dataframes, axis=0)
            
        except Exception as e:
            self.logger.error(f"Error merging dataframes: {e}")
            return dataframes[0] if dataframes else pd.DataFrame()

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
        if self.config.get('app.enable_ml_features', True):
            self._render_query_bar()
    
        if self.get_state('analysis_data') is not None:
            self._render_analysis_interface()
        else:
            self._render_welcome_screen()
            
    @safe_state_access
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
                        
                        query_history = self.get_state('query_history', [])
                        query_history.append({
                            'query': query,
                            'result': result,
                            'timestamp': datetime.now()
                        })
                        self.set_state('query_history', query_history[-10:])
                        
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
        data = self.get_state('analysis_data')
    
        if data is None:
            self._render_welcome_screen()
            return
    
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
            # Using the new enhanced method
            self._render_penman_nissim_tab_enhanced(data)
    
        with tabs[4]:
            self._render_industry_tab(data)
    
        with tabs[5]:
            self._render_data_explorer_tab(data)
    
        with tabs[6]:
            self._render_reports_tab(data)
    
        with tabs[7]:
            self._render_ml_insights_tab(data)
    
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
    def _render_industry_tab(self, data: pd.DataFrame):
        """Render industry comparison tab"""
        st.header("🏭 Industry Comparison")
        st.info("Compare your company's performance against industry benchmarks.")
    
        col1, col2 = st.columns(2)
        with col1:
            selected_industry = st.selectbox(
                "Select Industry",
                list(CoreIndustryBenchmarks.BENCHMARKS.keys()),
                index=0,
                key="industry_select"
            )
    
        with col2:
            analysis_year = st.selectbox(
                "Select Year for Analysis",
                data.columns.tolist(),
                index=len(data.columns)-1,
                key="industry_year_select"
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
    
    if __name__ == "__main__":
        try:
            app = FinancialAnalyticsPlatform()
            app.run()
        except Exception as e:
            # Fallback error display if main app fails catastrophically
            st.error("A critical application error occurred during startup.")
            st.exception(e)
    
                
