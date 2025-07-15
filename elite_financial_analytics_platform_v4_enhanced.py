# elite_financial_analytics_platform_enhanced.py
# Enterprise-Grade Financial Analytics Platform - Enhanced Version with Kaggle GPU Integration

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
from functools import lru_cache
import queue
import gc

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

# --- 3. Performance Monitoring System ---
class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self._lock = threading.Lock()
        self.logger = None
    
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
    
    def clear_metrics(self):
        """Clear all metrics"""
        with self._lock:
            self.metrics.clear()

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

# --- 5. Error Context and Boundary ---
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
            'version': '5.0.0',
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
        },
        'ui': {
            'theme': 'light',
            'animations': True,
            'auto_save': True,
            'auto_save_interval': 60,
            'show_tutorial': True,
            'enable_skeleton_loading': True,
            'show_kaggle_status': True,
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

# --- 17. Enhanced API Client ---
class EnhancedAPIClient:
    """Enhanced API client with retry logic, connection pooling, and monitoring"""
    
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = None
        self._setup_session()
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        self.response_times = deque(maxlen=100)
        self._lock = threading.Lock()
        
    def _setup_session(self):
        """Setup requests session with retry and connection pooling"""
        self._session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        
        self._session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'EliteFinancialAnalytics/5.0',
            'ngrok-skip-browser-warning': 'true'
        })
    
    def make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                    params: Optional[Dict] = None, timeout: Optional[int] = None) -> Optional[Dict]:
        """Make HTTP request with error handling and monitoring"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        timeout = timeout or self.timeout
        
        start_time = time.time()
        
        try:
            with self._lock:
                self.request_count += 1
                self.last_request_time = datetime.now()
            
            response = self._session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.Timeout:
            with self._lock:
                self.error_count += 1
            raise TimeoutError(f"Request timed out after {timeout}s")
            
        except requests.exceptions.ConnectionError:
            with self._lock:
                self.error_count += 1
            raise ConnectionError("Could not connect to API")
            
        except requests.exceptions.HTTPError as e:
            with self._lock:
                self.error_count += 1
            
            if e.response.status_code == 429:
                raise Exception("Rate limit exceeded")
            elif e.response.status_code >= 500:
                raise Exception(f"Server error: {e.response.status_code}")
            else:
                raise Exception(f"HTTP error: {e.response.status_code} - {e.response.text}")
                
        except Exception as e:
            with self._lock:
                self.error_count += 1
            raise Exception(f"Unexpected error: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        with self._lock:
            avg_response_time = np.mean(self.response_times) if self.response_times else 0
            return {
                'total_requests': self.request_count,
                'total_errors': self.error_count,
                'error_rate': self.error_count / self.request_count if self.request_count > 0 else 0,
                'avg_response_time': avg_response_time,
                'last_request': self.last_request_time
            }
    
    def close(self):
        """Close the session"""
        if self._session:
            self._session.close()

# --- 18. ENHANCED AI Mapping System with Kaggle Integration ---
class AIMapper(Component):
    """AI-powered mapping with Kaggle GPU support and comprehensive error handling"""
    
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
        self._gpu_stats = {}
        self._fallback_reasons = []
        self._connection_retries = 0
        self._max_retries = 3
    
    def _do_initialize(self):
        """Initialize AI components"""
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
    
    def _initialize_kaggle_api(self):
        """Initialize Kaggle API connection with validation"""
        try:
            api_url = self.config.get('ai.kaggle_api_url')
            api_key = self.config.get('ai.kaggle_api_key', '')
            timeout = self.config.get('ai.kaggle_api_timeout', 30)
            max_retries = self.config.get('ai.kaggle_max_retries', 3)
            
            self._api_client = EnhancedAPIClient(api_url, timeout, max_retries)
            
            # Set API key if provided
            if api_key:
                self._api_client._session.headers.update({
                    'Authorization': f'Bearer {api_key}'
                })
            
            if self._validate_kaggle_connection():
                self._kaggle_available = True
                self._logger.info(f"Successfully connected to Kaggle API at {api_url}")
                self._start_gpu_monitoring()
            else:
                self._logger.warning("Kaggle API validation failed, will use fallback")
                self._fallback_reasons.append("API validation failed")
                
        except Exception as e:
            self._logger.error(f"Failed to initialize Kaggle API: {e}")
            self._kaggle_available = False
            self._fallback_reasons.append(f"Initialization error: {str(e)}")
    
    def _validate_kaggle_connection(self) -> bool:
        """Comprehensive Kaggle API connection validation"""
        try:
            # Test basic connectivity
            response = self._api_client.make_request('GET', '/health', timeout=10)
            
            if not response:
                self._fallback_reasons.append("No response from health endpoint")
                return False
            
            # Validate required fields
            required_fields = ['status', 'model']
            if not all(field in response for field in required_fields):
                self._fallback_reasons.append("Missing required fields in health response")
                return False
            
            if response.get('status') != 'healthy':
                self._fallback_reasons.append(f"API status: {response.get('status')}")
                return False
            
            # Test embedding endpoint
            test_response = self._api_client.make_request(
                'POST', 
                '/embed',
                {'texts': ['test']},
                timeout=15
            )
            
            if not test_response or 'embeddings' not in test_response:
                self._fallback_reasons.append("Embedding endpoint test failed")
                return False
            
            # Store API info
            self._kaggle_info = response
            self._last_health_check = time.time()
            self._connection_retries = 0
            
            self._logger.info("Kaggle API validation successful")
            return True
            
        except ConnectionError:
            self._fallback_reasons.append("Connection error")
            return False
        except TimeoutError:
            self._fallback_reasons.append("Connection timeout")
            return False
        except Exception as e:
            self._fallback_reasons.append(f"Validation error: {str(e)}")
            return False
    
    def _start_gpu_monitoring(self):
        """Start real-time GPU monitoring"""
        def monitor_gpu():
            while self._kaggle_available:
                try:
                    stats = self._api_client.make_request('GET', '/gpu_stats', timeout=5)
                    if stats:
                        self._gpu_stats = {
                            'gpu_utilization': stats.get('gpu_util', 0),
                            'memory_used': stats.get('memory_used', 0),
                            'memory_total': stats.get('memory_total', 0),
                            'temperature': stats.get('temperature', 0),
                            'last_updated': time.time()
                        }
                except Exception as e:
                    self._logger.debug(f"GPU monitoring error: {e}")
                
                time.sleep(30)  # Update every 30 seconds
        
        monitoring_thread = threading.Thread(target=monitor_gpu, daemon=True)
        monitoring_thread.start()
    
    def _smart_fallback_decision(self, operation: str, data_size: int = 0) -> str:
        """Intelligent fallback decision based on multiple factors"""
        # Check Kaggle availability
        if not self._kaggle_available:
            return 'local_model' if self.model else 'fuzzy'
        
        # Check recent performance
        api_stats = self._api_client.get_stats()
        if api_stats['error_rate'] > 0.2:  # High error rate
            return 'local_model' if self.model else 'fuzzy'
        
        # Check response time
        if api_stats['avg_response_time'] > 10:  # Slow responses
            return 'local_model' if self.model else 'fuzzy'
        
        # Check data size
        if data_size > 1000:  # Large dataset
            return 'kaggle_gpu'  # GPU can handle better
        elif data_size < 10:  # Small dataset
            return 'local_model' if self.model else 'fuzzy'
        
        return 'kaggle_gpu'
    
    def map_metrics_with_confidence_levels(self, source_metrics: List[str], 
                                         target_metrics: Optional[List[str]] = None,
                                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Enhanced mapping with HTTP API integration and progress tracking"""
        
        if progress_callback:
            progress_callback(0.1, "Initializing mapping...")
        
        # Smart fallback decision
        fallback_strategy = self._smart_fallback_decision('mapping', len(source_metrics))
        
        if progress_callback:
            progress_callback(0.2, f"Using {fallback_strategy} for processing...")
        
        # Try Kaggle API first if available
        if fallback_strategy == 'kaggle_gpu' and self._check_kaggle_health():
            try:
                if progress_callback:
                    progress_callback(0.3, "Sending request to Kaggle GPU...")
                
                response = self._api_client.make_request(
                    'POST',
                    '/map_metrics_with_confidence',
                    {
                        'source_metrics': source_metrics,
                        'target_metrics': target_metrics,
                        'confidence_threshold': self.config.get('ai.similarity_threshold', 0.6)
                    },
                    timeout=self.config.get('ai.kaggle_api_timeout', 30)
                )
                
                if progress_callback:
                    progress_callback(0.9, "Processing Kaggle GPU response...")
                
                if response and 'high_confidence' in response:
                    response['method'] = 'kaggle_gpu'
                    response['fallback_reasons'] = []
                    
                    if progress_callback:
                        progress_callback(1.0, "Kaggle GPU mapping completed!")
                    
                    return response
                else:
                    raise Exception("Invalid response format from Kaggle API")
                    
            except ConnectionError as e:
                self._logger.warning(f"Kaggle connection error: {e}")
                self._fallback_reasons.append("Connection lost during mapping")
                self._kaggle_available = False
                
            except TimeoutError as e:
                self._logger.warning(f"Kaggle timeout: {e}")
                self._fallback_reasons.append("Request timeout")
                
            except Exception as e:
                self._logger.error(f"Kaggle API error: {e}")
                self._fallback_reasons.append(f"API error: {str(e)}")
        
        # Fallback to local processing
        if progress_callback:
            progress_callback(0.4, f"Falling back to {fallback_strategy}...")
        
        if fallback_strategy == 'local_model' and self.model:
            try:
                if progress_callback:
                    progress_callback(0.6, "Processing with local AI model...")
                
                result = self._local_mapping_with_confidence(source_metrics, target_metrics)
                result['method'] = 'local_ai'
                result['fallback_reasons'] = self._fallback_reasons.copy()
                
                if progress_callback:
                    progress_callback(1.0, "Local AI mapping completed!")
                
                return result
                
            except Exception as e:
                self._logger.error(f"Local model error: {e}")
                self._fallback_reasons.append(f"Local model error: {str(e)}")
        
        # Final fallback to fuzzy matching
        if progress_callback:
            progress_callback(0.7, "Using fuzzy matching fallback...")
        
        result = self.fallback_mapper.map_metrics(source_metrics, target_metrics)
        
        # Convert to confidence level format
        confidence_result = self._convert_to_confidence_format(result)
        confidence_result['method'] = 'fuzzy'
        confidence_result['fallback_reasons'] = self._fallback_reasons.copy()
        
        if progress_callback:
            progress_callback(1.0, "Fuzzy matching completed!")
        
        return confidence_result
    
    def _local_mapping_with_confidence(self, source_metrics: List[str], 
                                     target_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Local AI mapping with confidence levels"""
        base_result = self.map_metrics(source_metrics, target_metrics)
        return self._convert_to_confidence_format(base_result)
    
    def _convert_to_confidence_format(self, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert basic mapping result to confidence level format"""
        confidence_thresholds = self.config.get('ai.confidence_levels', {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        })
        
        result = {
            'high_confidence': {},
            'medium_confidence': {},
            'low_confidence': {},
            'requires_manual': [],
            'suggestions': base_result.get('suggestions', {}),
            'method': base_result.get('method', 'unknown')
        }
        
        mappings = base_result.get('mappings', {})
        confidence_scores = base_result.get('confidence_scores', {})
        
        for source in mappings:
            confidence = confidence_scores.get(source, 0)
            target = mappings[source]
            
            if confidence >= confidence_thresholds['high']:
                result['high_confidence'][source] = {
                    'target': target,
                    'confidence': confidence
                }
            elif confidence >= confidence_thresholds['medium']:
                result['medium_confidence'][source] = {
                    'target': target,
                    'confidence': confidence
                }
            elif confidence >= confidence_thresholds['low']:
                result['low_confidence'][source] = {
                    'target': target,
                    'confidence': confidence
                }
            else:
                result['requires_manual'].append(source)
        
        # Add unmapped metrics
        result['requires_manual'].extend(base_result.get('unmapped_metrics', []))
        
        return result
    
    def _check_kaggle_health(self) -> bool:
        """Enhanced health check with retry logic"""
        with self._health_check_lock:
            current_time = time.time()
            check_interval = self.config.get('ai.kaggle_health_check_interval', 60)
            
            # Skip if recently checked
            if (self._last_health_check and 
                current_time - self._last_health_check < check_interval):
                return self._kaggle_available
            
            # Perform health check with retries
            for attempt in range(self._max_retries):
                try:
                    response = self._api_client.make_request('GET', '/health', timeout=10)
                    
                    if response and response.get('status') == 'healthy':
                        self._kaggle_available = True
                        self._last_health_check = current_time
                        self._connection_retries = 0
                        return True
                        
                except Exception as e:
                    self._logger.debug(f"Health check attempt {attempt + 1} failed: {e}")
                    
                time.sleep(2 ** attempt)  # Exponential backoff
            
            # All retries failed
            self._kaggle_available = False
            self._connection_retries += 1
            self._fallback_reasons.append(f"Health check failed after {self._max_retries} attempts")
            
            return False
    
    def _initialize_local_model(self):
        """Initialize local sentence transformer model"""
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            self._logger.warning("Sentence transformers not available")
            return
            
        try:
            model_name = self.config.get('ai.model_name', 'all-MiniLM-L6-v2')
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(model_name)
            self._logger.info(f"Loaded local model: {model_name}")
            
            self._precompute_standard_embeddings()
            
        except Exception as e:
            self._logger.error(f"Failed to load local model: {e}")
    
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
            'EBIT': ['ebit', 'operating income', 'operating profit'],
            'Interest Expense': ['interest expense', 'finance costs', 'interest costs'],
            'Tax Expense': ['tax expense', 'income tax', 'tax'],
        }
        
        for metric, descriptions in standard_metrics.items():
            combined_text = ' '.join(descriptions)
            embedding = self._get_embedding(combined_text)
            if embedding is not None:
                SimpleState.set(f"standard_embedding_{metric}", embedding)
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding with caching"""
        if not self.model:
            return None
        
        cache_key = f"embedding_{hashlib.md5(text.encode()).hexdigest()}"
        cached = self.embeddings_cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            embedding = self.model.encode(
                text, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            self.embeddings_cache.set(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            self._logger.error(f"Error computing embedding: {e}")
            return None
    
    def map_metrics(self, source_metrics: List[str], 
                   target_metrics: Optional[List[str]] = None,
                   confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Map source metrics to target metrics using semantic similarity"""
        
        if not self.model:
            return self.fallback_mapper.map_metrics(source_metrics, target_metrics)
        
        if confidence_threshold is None:
            confidence_threshold = self.config.get('ai.similarity_threshold', 0.6)
        
        if target_metrics is None:
            target_metrics = self._get_standard_metrics()
        
        mappings = {}
        confidence_scores = {}
        suggestions = {}
        unmapped = []
        
        batch_size = self.config.get('ai.batch_size', 32)
        
        with performance_monitor.measure("ai_mapping"):
            for i in range(0, len(source_metrics), batch_size):
                batch = source_metrics[i:i + batch_size]
                
                try:
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
                    
                    target_embeddings = []
                    valid_targets = []
                    
                    for target in target_metrics:
                        embedding = SimpleState.get(f"standard_embedding_{target}")
                        
                        if embedding is None:
                            embedding = self._get_embedding(target.lower())
                        
                        if embedding is not None:
                            target_embeddings.append(embedding)
                            valid_targets.append(target)
                    
                    if not target_embeddings:
                        unmapped.extend(valid_sources)
                        continue
                    
                    source_matrix = np.vstack(source_embeddings)
                    target_matrix = np.vstack(target_embeddings)
                    
                    similarities = cosine_similarity(source_matrix, target_matrix)
                    
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
            'Investing Cash Flow', 'Financing Cash Flow',
            'EBIT', 'EBITDA', 'Interest Expense', 'Tax Expense'
        ]
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get comprehensive API status with GPU monitoring"""
        base_status = {
            'kaggle_configured': bool(self.config.get('ai.kaggle_api_url')),
            'kaggle_available': self._kaggle_available,
            'local_model_available': self.model is not None,
            'cache_size': len(self.embeddings_cache._cache),
            'api_info': self._kaggle_info,
            'api_stats': self._api_client.get_stats() if self._api_client else None,
            'fallback_reasons': self._fallback_reasons,
            'connection_retries': self._connection_retries
        }
        
        # Add GPU stats if available
        if self._gpu_stats:
            base_status['gpu_stats'] = self._gpu_stats.copy()
            
            # Calculate GPU utilization percentage
            if self._gpu_stats.get('memory_total', 0) > 0:
                memory_usage_pct = (self._gpu_stats.get('memory_used', 0) / 
                                   self._gpu_stats.get('memory_total', 1)) * 100
                base_status['gpu_stats']['memory_usage_pct'] = memory_usage_pct
        
        return base_status
    
    def clear_fallback_reasons(self):
        """Clear accumulated fallback reasons"""
        self._fallback_reasons.clear()
    
    def _start_batch_processor(self):
        """Start background thread for batch processing"""
        self._batch_processor = threading.Thread(target=self._process_batch_queue, daemon=True)
        self._batch_processor.start()
    
    def _process_batch_queue(self):
        """Process batched embedding requests"""
        while True:
            try:
                batch = []
                deadline = time.time() + 0.1
                
                while time.time() < deadline and len(batch) < self.config.get('ai.kaggle_batch_size', 50):
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
        """Process a batch of embedding requests"""
        texts = [item['text'] for item in batch]
        futures = [item['future'] for item in batch]
        
        try:
            embeddings = self._get_embeddings_kaggle_batch(texts)
            
            if embeddings:
                for future, embedding in zip(futures, embeddings):
                    future.set_result(embedding)
            else:
                for i, (text, future) in enumerate(zip(texts, futures)):
                    embedding = self._get_embedding_local(text)
                    future.set_result(embedding)
                    
        except Exception as e:
            for future in futures:
                future.set_exception(e)
    
    def _get_embedding_local(self, text: str) -> Optional[np.ndarray]:
        """Get embedding using local model"""
        if self.model is None:
            return None
            
        try:
            return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            self._logger.error(f"Local embedding error: {e}")
            return None
    
    def _get_embeddings_kaggle_batch(self, texts: List[str]) -> Optional[List[np.ndarray]]:
        """Get batch embeddings from Kaggle API"""
        try:
            response = self._api_client.make_request('POST', '/embed', {
                'texts': texts,
                'batch_size': len(texts)
            })
            
            if response and 'embeddings' in response:
                return [np.array(emb) for emb in response['embeddings']]
                
        except Exception as e:
            self._logger.error(f"Kaggle batch embedding error: {e}")
            
        return None
    
    def _do_cleanup(self):
        """Cleanup resources"""
        if self._api_client:
            self._api_client.close()
        
        if self._session:
            self._session.close()
        
        if self._batch_processor and self._batch_processor.is_alive():
            # Signal shutdown
            self._batch_queue.put(None)

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
    
    def _calculate_accuracy(self, model: Any, series: pd.Series) -> Dict[str,
    
