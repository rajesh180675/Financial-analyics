# elite_financial_analytics_platform_v4_debugged.py
# Enterprise-Grade Financial Analytics Platform - Debugged and Enhanced Version

# --- 1. Core Imports and Setup ---
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

# Set up warnings
warnings.filterwarnings('ignore')

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

# Try to import sentence transformers for type checking
if SENTENCE_TRANSFORMER_AVAILABLE:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = None

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
    CorePenmanNissim = None

# --- 2. Thread-Safe State Management ---
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

# Use ThreadSafeState as SimpleState
SimpleState = ThreadSafeState

# --- 3. Performance Monitoring System ---
class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self._lock = threading.Lock()
        self.logger = None  # Will be initialized later
    
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
            
            if elapsed_time > 1.0:  # Log slow operations
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

# --- 5. Error Context with Recovery ---
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

# --- 6. Error Boundary Decorator ---
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
                
                # Show user-friendly error in Streamlit
                st.error(f"An error occurred in {func.__name__}. Please try again or contact support.")
                
                # If fallback_return is callable, call it
                if callable(fallback_return):
                    return fallback_return()
                    
                return fallback_return
        return wrapper
    return decorator

# --- 7. Configuration Management ---
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
            'version': '4.0.0',
            'name': 'Elite Financial Analytics Platform',
            'debug': False,
            'display_mode': DisplayMode.LITE,
            'max_file_size_mb': 50,
            'allowed_file_types': ['csv', 'html', 'htm', 'xls', 'xlsx'],
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
            }
        },
        'ui': {
            'theme': 'light',
            'animations': True,
            'auto_save': True,
            'auto_save_interval': 60,
            'show_tutorial': True,
            'enable_skeleton_loading': True,
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

# --- 8. Number Formatting Functions ---
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

# --- 9. Enhanced Caching System ---
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
            return 1024  # Default estimate
    
    def _compress_value(self, value: Any) -> bytes:
        """Compress large values"""
        return zlib.compress(pickle.dumps(value), level=6)
    
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
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, compress: bool = None):
        """Set value in cache with optional compression"""
        with self._lock:
            self._stats['set_calls'] += 1
            
            # Determine if compression should be used
            if compress is None:
                compress = self._estimate_size(value) > self._compression_threshold
            
            # Create entry
            if compress:
                compressed_value = self._compress_value(value)
                entry = CacheEntry(compressed_value, ttl or self._default_ttl, compressed=True)
            else:
                entry = CacheEntry(value, ttl or self._default_ttl, compressed=False)
            
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

# --- 10. Resource Management ---
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
            # Conservative estimate if psutil not available
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                # Assume we can use 50% of available memory
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

# --- 11. Data Validation ---
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
    
    @error_boundary((pd.DataFrame(), ValidationResult()))  # Return proper tuple as default
    def validate_and_correct(self, df: pd.DataFrame, context: str = "data") -> Tuple[pd.DataFrame, ValidationResult]:
        """Validate and auto-correct dataframe"""
        result = self.validate_dataframe(df, context)
        
        if not self.enable_auto_correction:
            return df, result
        
        corrected_df = df.copy()
        
        # Auto-corrections
        corrections_made = []
        
        # Fix negative values in typically positive metrics
        positive_metrics = ['assets', 'revenue', 'equity', 'sales', 'income', 'cash']
        for idx in corrected_df.index:
            for metric in positive_metrics:
                if metric in str(idx).lower():
                    # Get the row data
                    row_data = corrected_df.loc[idx]
                    
                    # Handle case where loc returns a DataFrame (duplicate indices)
                    if isinstance(row_data, pd.DataFrame):
                        # Process each duplicate row
                        for i in range(len(row_data)):
                            negative_mask = row_data.iloc[i] < 0
                            if negative_mask.any():
                                corrected_df.loc[idx].iloc[i][negative_mask] = abs(row_data.iloc[i][negative_mask])
                                corrections_made.append(f"Converted negative values to positive in {idx} (row {i})")
                    else:
                        # Single row - process normally
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
        
        # Update result with corrections
        if corrections_made:
            result.add_info(f"Applied {len(corrections_made)} auto-corrections")
            result.corrections = corrections_made
        
        return corrected_df, result
    
    def _fix_accounting_equation(self, df: pd.DataFrame, corrections_made: List[str]):
        """Fix violations of accounting equation (Assets = Liabilities + Equity)"""
        # Find relevant rows
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
                        tolerance = assets * 0.01  # 1% tolerance
                        
                        if diff > tolerance:
                            # Adjust equity to balance
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
                
                if std > 0:
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

# --- 12. Pattern Matching System ---
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

# --- 13. Base Component Class ---
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

# --- 14. Security Module ---
class SecurityModule(Component):
    """Enhanced security with comprehensive validation"""
    
    def __init__(self, config: Configuration):
        super().__init__(config)
        self._sanitizer = None
        self._rate_limiter = defaultdict(deque)
        self._blocked_ips = set()
    
    def _do_initialize(self):
        """Initialize security components"""
        # Initialize HTML sanitizer
        self._allowed_tags = self.config.get('security.allowed_html_tags', [])
        self._allowed_attributes = {
            '*': ['class', 'id'],
            'table': ['border', 'cellpadding', 'cellspacing'],
        }
    
    @error_boundary()
    def sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deep sanitization of dataframe content"""
        sanitized = df.copy()
        
        # Sanitize string columns
        for col in sanitized.select_dtypes(include=['object']).columns:
            sanitized[col] = sanitized[col].apply(
                lambda x: bleach.clean(str(x)) if pd.notna(x) else x
            )
        
        # Validate numeric ranges
        for col in sanitized.select_dtypes(include=[np.number]).columns:
            # Check for unrealistic values
            max_val = sanitized[col].max()
            if pd.notna(max_val) and max_val > 1e15:  # Trillion+
                self._logger.warning(f"Extremely large values detected in {col}")
        
        return sanitized
    
    def validate_file_upload(self, file: UploadedFile) -> ValidationResult:
        """Comprehensive file validation"""
        result = ValidationResult()
        
        # Check file size
        max_size = self.config.get('security.max_upload_size_mb', 50) * 1024 * 1024
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
    
    def check_rate_limit(self, identifier: str, action: str, 
                        limit: Optional[int] = None, window: Optional[int] = None) -> bool:
        """Check rate limit for an action"""
        if limit is None:
            limit = self.config.get('security.rate_limit_requests', 100)
        if window is None:
            window = self.config.get('security.rate_limit_window', 60)
        
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
    
    @error_boundary()
    def process(self, df: pd.DataFrame, context: str = "data") -> Tuple[pd.DataFrame, ValidationResult]:
        """Process dataframe through pipeline"""
        with performance_monitor.measure(f"process_{context}"):
            # Check if we need chunk processing
            if len(df) > self.chunk_size:
                return self._process_large_dataframe(df, context)
            else:
                return self._process_standard(df, context)
    
    def _process_standard(self, df: pd.DataFrame, context: str) -> Tuple[pd.DataFrame, ValidationResult]:
        """Standard processing for normal-sized dataframes"""
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
            
            # Auto-correction if enabled
            if self.config.get('analysis.enable_auto_correction', True):
                validator = DataValidator(self.config)
                try:
                    processed_df, correction_result = validator.validate_and_correct(processed_df, context)
                    result.merge(correction_result)
                except Exception as e:
                    self._logger.error(f"Auto-correction failed: {e}")
                    # Continue without auto-correction
            
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
    
    def _process_large_dataframe(self, df: pd.DataFrame, context: str) -> Tuple[pd.DataFrame, ValidationResult]:
        """Process large dataframes in chunks to reduce memory usage"""
        self._logger.info(f"Processing large dataframe ({len(df)} rows) in chunks")
        
        result = ValidationResult()
        chunks = []
        
        # Process in chunks
        for start in range(0, len(df), self.chunk_size):
            end = min(start + self.chunk_size, len(df))
            chunk = df.iloc[start:end]
            
            processed_chunk, chunk_result = self._process_standard(chunk, f"{context}_chunk_{start}")
            chunks.append(processed_chunk)
            result.merge(chunk_result)
            
            if not chunk_result.is_valid:
                break
        
        # Combine chunks
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
            # Try to convert to numeric
            converted = pd.to_numeric(df[col], errors='coerce')
            
            # If mostly numeric (>50%), convert the column
            if converted.notna().sum() > len(df) * 0.5:
                df_clean[col] = converted
        
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
                        
                        # Store outlier information in session state
                        outlier_indices = series[outliers].index.tolist()
                        SimpleState.set(f"outliers_{col}", outlier_indices)
        
        return df_clean

# --- 16. Financial Analysis Engine ---
class FinancialAnalysisEngine(Component):
    """Core financial analysis engine with advanced features"""
    
    def __init__(self, config: Configuration):
        super().__init__(config)
        self.pattern_matcher = PatternMatcher()
        self.cache = AdvancedCache()
    
    def _do_initialize(self):
        """Initialize analysis components"""
        # Load any required models or data
        pass
    
    @error_boundary({})
    def analyze_financial_statements(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive financial statement analysis"""
        with performance_monitor.measure("analyze_financial_statements"):
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
                'insights': self._generate_insights(df),
                'anomalies': self._detect_anomalies(df)
            }
            
            # Cache the result
            self.cache.set(cache_key, analysis, ttl=3600)
            
            return analysis
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Detect anomalies in financial data"""
        anomalies = {
            'value_anomalies': [],
            'trend_anomalies': [],
            'ratio_anomalies': []
        }
        
        # Value anomalies - extreme values
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
        
        # Trend anomalies - sudden changes
        for idx in df.index:
            series = df.loc[idx].dropna()
            if len(series) > 2:
                pct_changes = series.pct_change().dropna()
                extreme_changes = pct_changes[np.abs(pct_changes) > 1]  # >100% change
                
                for year, change in extreme_changes.items():
                    anomalies['trend_anomalies'].append({
                        'metric': str(idx),
                        'year': year,
                        'change_pct': change * 100
                    })
        
        return anomalies
    
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
        """Calculate financial ratios with proper error handling"""
        ratios = {}
        
        # Extract key metrics using pattern matching
        metrics = self._extract_key_metrics(df)
        
        # Initialize all variables
        metric_values = {}
        metric_keys = [
            'current_assets', 'current_liabilities', 'total_assets', 'total_liabilities',
            'total_equity', 'inventory', 'cash', 'net_income', 'revenue',
            'cost_of_goods_sold', 'ebit', 'interest_expense', 'receivables'
        ]
        
        # Get all metrics first
        for metric_key in metric_keys:
            metric_value = self._get_metric_value(df, metrics, metric_key)
            if metric_value is not None:
                # Ensure it's a Series
                if isinstance(metric_value, pd.DataFrame):
                    metric_value = metric_value.iloc[0]
                metric_values[metric_key] = metric_value
            else:
                metric_values[metric_key] = None
        
        # Helper function with null checking
        def safe_divide(numerator_key, denominator_key, multiplier=1):
            """Safely divide two metrics"""
            numerator = metric_values.get(numerator_key)
            denominator = metric_values.get(denominator_key)
            
            if numerator is None or denominator is None:
                return None
            
            try:
                # Handle both Series and scalar values
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
            
            # Current Ratio
            current_ratio = safe_divide('current_assets', 'current_liabilities')
            if current_ratio is not None:
                liquidity_data['Current Ratio'] = current_ratio
            
            # Quick Ratio
            if metric_values['current_assets'] is not None and metric_values['inventory'] is not None:
                quick_assets = metric_values['current_assets'] - metric_values['inventory']
                quick_ratio = safe_divide(None, 'current_liabilities')
                if metric_values['current_liabilities'] is not None:
                    quick_ratio = quick_assets / metric_values['current_liabilities'].replace(0, np.nan)
                    liquidity_data['Quick Ratio'] = quick_ratio
            
            # Cash Ratio
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
            
            # Net Profit Margin
            npm = safe_divide('net_income', 'revenue', 100)
            if npm is not None:
                profitability_data['Net Profit Margin %'] = npm
            
            # Gross Profit Margin
            if metric_values['revenue'] is not None and metric_values['cost_of_goods_sold'] is not None:
                gross_profit = metric_values['revenue'] - metric_values['cost_of_goods_sold']
                gpm = (gross_profit / metric_values['revenue'].replace(0, np.nan)) * 100
                profitability_data['Gross Profit Margin %'] = gpm
            
            # ROA
            roa = safe_divide('net_income', 'total_assets', 100)
            if roa is not None:
                profitability_data['Return on Assets %'] = roa
            
            # ROE
            roe = safe_divide('net_income', 'total_equity', 100)
            if roe is not None:
                profitability_data['Return on Equity %'] = roe
            
            # ROCE
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
            
            # Debt to Equity
            de_ratio = safe_divide('total_liabilities', 'total_equity')
            if de_ratio is not None:
                leverage_data['Debt to Equity'] = de_ratio
            
            # Debt Ratio
            debt_ratio = safe_divide('total_liabilities', 'total_assets')
            if debt_ratio is not None:
                leverage_data['Debt Ratio'] = debt_ratio
            
            # Interest Coverage Ratio
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
            
            # Asset Turnover
            asset_turnover = safe_divide('revenue', 'total_assets')
            if asset_turnover is not None:
                efficiency_data['Asset Turnover'] = asset_turnover
            
            # Inventory Turnover
            inv_turnover = safe_divide('cost_of_goods_sold', 'inventory')
            if inv_turnover is not None:
                efficiency_data['Inventory Turnover'] = inv_turnover
            
            # Receivables Turnover
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
            # Get the highest confidence match
            best_match = max(metrics[metric_type], key=lambda x: x['confidence'])
            metric_name = best_match['name']
            
            if metric_name in df.index:
                result = df.loc[metric_name]
                # If multiple rows match (DataFrame), take the first one
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
            
            # Handle case where loc returns a DataFrame (duplicate indices)
            if isinstance(series, pd.DataFrame):
                self._logger.warning(f"Multiple rows found for {idx}, taking first")
                series = series.iloc[0]
            
            series = series.dropna()
            
            if len(series) >= 3:
                # Calculate trend metrics
                years = np.arange(len(series))
                values = series.values
                
                # Linear regression - polyfit returns coefficients
                coefficients = np.polyfit(years, values, 1)
                slope = coefficients[0]  # First coefficient is slope
                intercept = coefficients[1]  # Second coefficient is intercept
                
                # Compound Annual Growth Rate (CAGR)
                try:
                    first_value = series.iloc[0]
                    last_value = series.iloc[-1]
                    
                    # Convert to scalar properly
                    if hasattr(first_value, 'item'):
                        first_value = first_value.item()
                    elif isinstance(first_value, np.ndarray):
                        first_value = first_value.flat[0]
                    else:
                        first_value = float(first_value)
                        
                    if hasattr(last_value, 'item'):
                        last_value = last_value.item()
                    elif isinstance(last_value, np.ndarray):
                        last_value = last_value.flat[0]
                    else:
                        last_value = float(last_value)
                    
                    if first_value > 0 and last_value > 0:
                        years_diff = len(series) - 1
                        cagr = ((last_value / first_value) ** (1 / years_diff) - 1) * 100
                    else:
                        cagr = None
                        
                except Exception as e:
                    self._logger.warning(f"Could not calculate CAGR for {idx}: {e}")
                    cagr = None
                
                # Volatility
                try:
                    volatility = series.pct_change().std() * 100
                    if pd.isna(volatility):
                        volatility = 0
                    else:
                        # Ensure it's a scalar
                        if hasattr(volatility, 'item'):
                            volatility = volatility.item()
                        elif isinstance(volatility, np.ndarray):
                            volatility = volatility.flat[0]
                        else:
                            volatility = float(volatility)
                except Exception:
                    volatility = 0
                
                # Ensure slope and intercept are scalars
                if hasattr(slope, 'item'):
                    slope = slope.item()
                elif isinstance(slope, np.ndarray):
                    slope = slope.flat[0]
                else:
                    slope = float(slope)
                    
                if hasattr(intercept, 'item'):
                    intercept = intercept.item()
                elif isinstance(intercept, np.ndarray):
                    intercept = intercept.flat[0]
                else:
                    intercept = float(intercept)
                
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
                    # Get the row data
                    row_data = numeric_df.loc[idx]
                    
                    # Handle case where loc returns a DataFrame (duplicate indices)
                    if isinstance(row_data, pd.DataFrame):
                        row_data = row_data.iloc[0]
                    
                    # Count negative values
                    negative_count = (row_data < 0).sum()
                    
                    # Ensure negative_count is a scalar
                    if hasattr(negative_count, 'item'):
                        negative_count = negative_count.item()
                    elif isinstance(negative_count, np.ndarray):
                        negative_count = int(negative_count)
                    else:
                        negative_count = int(negative_count)
                    
                    if negative_count > 0:
                        consistency_score -= (negative_count / len(numeric_df.columns)) * 20
            
            scores.append(max(0, consistency_score))
        
        # Temporal consistency (year-over-year changes)
        if len(numeric_df.columns) > 1:
            temporal_score = 100
            extreme_changes = 0
            
            for idx in numeric_df.index:
                series = numeric_df.loc[idx]
                
                # Handle case where loc returns a DataFrame (duplicate indices)
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[0]
                
                series = series.dropna()
                
                if len(series) > 1:
                    pct_changes = series.pct_change().dropna()
                    # Flag changes over 200%
                    extreme_count = (pct_changes.abs() > 2).sum()
                    
                    # Ensure extreme_count is a scalar
                    if hasattr(extreme_count, 'item'):
                        extreme_count = extreme_count.item()
                    elif isinstance(extreme_count, np.ndarray):
                        extreme_count = int(extreme_count)
                    else:
                        extreme_count = int(extreme_count)
                    
                    extreme_changes += extreme_count
            
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
                
                # Trend analysis
                if len(current_ratios) > 1:
                    trend = 'improving' if current_ratios.iloc[-1] > current_ratios.iloc[0] else 'declining'
                    insights.append(f"📊 Current ratio is {trend} over the period")
        
        # Profitability insights
        if 'Profitability' in ratios and 'Net Profit Margin %' in ratios['Profitability'].index:
            npm = ratios['Profitability'].loc['Net Profit Margin %'].dropna()
            if len(npm) > 1:
                trend = 'improving' if npm.iloc[-1] > npm.iloc[0] else 'declining'
                insights.append(f"📊 Net profit margin is {trend} ({npm.iloc[0]:.1f}% → {npm.iloc[-1]:.1f}%)")
                
                # Industry comparison hint
                if npm.iloc[-1] < 5:
                    insights.append(f"⚠️ Low profit margin may indicate competitive pressure or cost issues")
        
        # Leverage insights
        if 'Leverage' in ratios and 'Debt to Equity' in ratios['Leverage'].index:
            de_ratio = ratios['Leverage'].loc['Debt to Equity'].dropna()
            if len(de_ratio) > 0:
                latest_de = de_ratio.iloc[-1]
                if latest_de > 2:
                    insights.append(f"⚠️ High debt-to-equity ratio ({latest_de:.2f}) indicates high leverage")
                elif latest_de < 0.3:
                    insights.append(f"💡 Low leverage ({latest_de:.2f}) - consider if debt could accelerate growth")
        
        # Efficiency insights
        if 'Efficiency' in ratios and 'Asset Turnover' in ratios['Efficiency'].index:
            asset_turnover = ratios['Efficiency'].loc['Asset Turnover'].dropna()
            if len(asset_turnover) > 0:
                latest_at = asset_turnover.iloc[-1]
                if latest_at < 0.5:
                    insights.append(f"⚠️ Low asset turnover ({latest_at:.2f}) suggests underutilized assets")
        
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
            elif 0 < cagr < 5:
                insights.append(f"🐌 Slow revenue growth (CAGR: {cagr:.1f}%) - explore growth strategies")
        
        # Profit growth vs revenue growth
        profit_trends = [v for k, v in trends.items() if 'net income' in k.lower() or 'profit' in k.lower()]
        if revenue_trends and profit_trends:
            rev_cagr = revenue_trends[0].get('cagr', 0)
            prof_cagr = profit_trends[0].get('cagr', 0)
            if rev_cagr > 0 and prof_cagr < rev_cagr:
                insights.append(f"⚠️ Profit growing slower than revenue - check cost management")
        
        # Data quality
        quality_score = self._calculate_quality_score(df)
        if quality_score < 70:
            insights.append(f"⚠️ Data quality score is low ({quality_score:.0f}%), results may be less reliable")
        
        # Anomaly insights
        anomalies = self._detect_anomalies(df)
        if anomalies['value_anomalies']:
            insights.append(f"🔍 Detected {len(anomalies['value_anomalies'])} unusual values - review for accuracy")
        
        return insights

# Continue in next message due to length...
# --- 17. AI Mapping System ---
class AIMapper(Component):
    """AI-powered mapping with fallback mechanisms and confidence levels"""
    
    def __init__(self, config: Configuration):
        super().__init__(config)
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
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self._logger.info(f"Loaded AI model: {model_name}")
                
                # Pre-compute standard embeddings
                self._precompute_standard_embeddings()
            else:
                self._logger.warning("Sentence transformers not available, using fallback")
                
        except Exception as e:
            self._logger.error(f"Failed to initialize AI model: {e}")
        
        # Initialize fallback
        self.fallback_mapper = FuzzyMapper(self.config)
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
    
    @error_boundary({})
    def map_metrics_with_confidence_levels(self, source_metrics: List[str], 
                                         target_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Enhanced mapping with multiple confidence levels"""
        confidence_thresholds = self.config.get('ai.confidence_levels', {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        })
        
        # Get base mappings
        base_result = self.map_metrics(source_metrics, target_metrics)
        
        # Categorize by confidence level
        results = {
            'high_confidence': {},
            'medium_confidence': {},
            'low_confidence': {},
            'requires_manual': [],
            'suggestions': base_result.get('suggestions', {}),
            'method': base_result.get('method', 'unknown')
        }
        
        for source in source_metrics:
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
        
        return results
    
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
        
        with performance_monitor.measure("ai_mapping"):
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
                        embedding = SimpleState.get(f"standard_embedding_{target}")
                        
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

# --- 18. Fuzzy Mapping Fallback ---
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

# --- 19. Penman-Nissim Analyzer ---
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
        if CORE_COMPONENTS_AVAILABLE and CorePenmanNissim is not None:
            try:
                # Use inspect to check constructor signature
                sig = inspect.signature(CorePenmanNissim.__init__)
                params = list(sig.parameters.keys())[1:]  # Skip 'self'
                
                if len(params) >= 2:
                    # Expects both df and mappings
                    self.core_analyzer = CorePenmanNissim(self.df, self.mappings)
                    self.logger.info("Initialized CorePenmanNissim with df and mappings")
                elif len(params) == 1:
                    # Only expects df
                    self.core_analyzer = CorePenmanNissim(self.df)
                    if hasattr(self.core_analyzer, 'set_mappings'):
                        self.core_analyzer.set_mappings(self.mappings)
                    elif hasattr(self.core_analyzer, 'mappings'):
                        self.core_analyzer.mappings = self.mappings
                    self.logger.info("Initialized CorePenmanNissim with df only")
                else:
                    # No parameters expected
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
        
        # Spread (RNOA - NBC)
        if 'Return on Net Operating Assets (RNOA) %' in ratios.index and 'Net Borrowing Cost (NBC) %' in ratios.index:
            ratios['Spread %'] = ratios['Return on Net Operating Assets (RNOA) %'] - ratios['Net Borrowing Cost (NBC) %']
        
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
                    df.loc['Capital Expenditure'] if 'Capital Expenditure' in df.index else 0
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

# --- 20. Manual Mapping Interface ---
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
                        
                        selected = st.selectbox(
                            f"{target}:",
                            ['(Not mapped)'] + self.source_metrics,
                            index=default_idx,
                            key=f"map_{statement_type}_{target}_{i}_{j}",  # Unique key
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
                    key="custom_source_mapping"
                )
            
            with col2:
                custom_target = st.selectbox(
                    "Target Metric:",
                    self.target_metrics,
                    key="custom_target_mapping"
                )
            
            if st.button("Add Mapping", key="add_custom_mapping_btn"):
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

# --- 21. Machine Learning Forecasting Module ---
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
                # Select key metrics to forecast
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
        # For now, use linear as default
        # In production, would test multiple models and select based on cross-validation
        return 'linear'
    
    def _select_key_metrics(self, df: pd.DataFrame) -> List[str]:
        """Select key metrics to forecast"""
        priority_metrics = ['Revenue', 'Net Income', 'Total Assets', 'Operating Cash Flow']
        available_metrics = []
        
        for metric in priority_metrics:
            matching = [idx for idx in df.index if metric.lower() in str(idx).lower()]
            if matching:
                available_metrics.append(matching[0])
        
        return available_metrics[:4]  # Limit to 4 metrics
    
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
        # Log transform for exponential
        X = np.arange(len(series)).reshape(-1, 1)
        y = np.log(series.values + 1)  # Add 1 to handle zeros
        
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
        # Test different models and select best based on validation
        models = {
            'linear': self._train_linear(series),
            'polynomial': self._train_polynomial(series),
        }
        
        # Simple validation: use last 20% for testing
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
            # Simple confidence interval based on historical volatility
            # In production, would use proper statistical methods
            values = np.array(forecast['values'])
            std = values.std() if len(values) > 1 else values[0] * 0.1
            
            z_score = stats.norm.ppf((1 + confidence) / 2)
            margin = z_score * std
            
            intervals[metric] = {
                'lower': (values - margin).tolist(),
                'upper': (values + margin).tolist()
            }
        
        return intervals

# --- 22. Natural Language Query Processor ---
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
        # Implementation for comparison queries
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

# --- 23. Collaboration Manager ---
class CollaborationManager:
    """Manage collaborative analysis sessions"""
    
    def __init__(self):
        self.active_sessions = {}
        self.shared_analyses = {}
        self.user_presence = defaultdict(dict)
        self._lock = threading.Lock()
        self.logger = LoggerFactory.get_logger('CollaborationManager')
    
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

# --- 24. Tutorial System ---
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

# --- 25. Export Manager ---
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
                            # Ensure sheet name is valid (Excel limit is 31 chars)
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
        """Export analysis to PDF format (placeholder - requires additional libraries)"""
        # This would require libraries like reportlab or weasyprint
        # For now, return a placeholder
        self.logger.info("PDF export requested - placeholder implementation")
        return b"PDF export not yet implemented. Please use Excel or Markdown export."
    
    def export_to_powerpoint(self, analysis: Dict[str, Any], template: str = 'default') -> bytes:
        """Export analysis to PowerPoint format (placeholder - requires python-pptx)"""
        # This would require python-pptx library
        # For now, return a placeholder
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

# --- 26. UI Components Factory ---
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
            # Show skeleton
            for _ in range(3):
                st.container().markdown(
                    """
                    <div class="skeleton"></div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            render_func()

# --- 27. Sample Data Generator ---
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

# Continue in next message with the main application class...
# --- 28. Main Application Class ---
class FinancialAnalyticsPlatform:
    """Main application with advanced architecture and all integrations"""
    
    def __init__(self):
        # Initialize session state for persistent data
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.analysis_data = None
            st.session_state.metric_mappings = None
            st.session_state.pn_mappings = None
            st.session_state.pn_results = None
            st.session_state.ai_mapping_result = None
            st.session_state.company_name = None
            st.session_state.data_source = None
            st.session_state.show_manual_mapping = False
            st.session_state.config_overrides = {}
            st.session_state.uploaded_files = []
            st.session_state.simple_parse_mode = False
            st.session_state.number_format_value = 'Indian'
            st.session_state.show_tutorial = True
            st.session_state.tutorial_step = 0
            st.session_state.collaboration_session = None
            st.session_state.query_history = []
            st.session_state.ml_forecast_results = None
            
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
        
        return components
    
    # State helper methods
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from session state"""
        return SimpleState.get(key, default)
    
    def set_state(self, key: str, value: Any):
        """Set value in session state"""
        SimpleState.set(key, value)
    
    @error_boundary()
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
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render application header"""
        st.markdown(
            '<h1 class="main-header">💹 Elite Financial Analytics Platform</h1>',
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
    
    def _render_file_upload(self):
        """Render file upload interface"""
        allowed_types = self.config.get('app.allowed_file_types', [])
        max_size = self.config.get('security.max_upload_size_mb', 50)
        
        # File uploader
        temp_files = st.sidebar.file_uploader(
            f"Upload Financial Statements (Max {max_size}MB each)",
            type=allowed_types,
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if temp_files:
            st.session_state['uploaded_files'] = temp_files
            st.sidebar.success(f"✅ {len(temp_files)} file(s) uploaded")
        
        uploaded_files = st.session_state['uploaded_files']
        
        if uploaded_files:
            # Simple parsing mode checkbox
            st.session_state['simple_parse_mode'] = st.sidebar.checkbox(
                "Use simple parsing mode", 
                value=st.session_state['simple_parse_mode'],
                help="Try this if normal parsing fails"
            )
            
            # Validate files
            all_valid = True
            for file in uploaded_files:
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
            
            **💡 Pro Tip**: If you're downloading from Capitaline, both "Export to Excel" 
            and "Download as Excel" options will work with this tool.
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
                        self.set_state('query_history', query_history[-10:])  # Keep last 10
                        
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
                
                # Historical data
                hist_periods = list(range(len(forecast['periods'])))
                hist_values = [forecast['last_actual']] * len(hist_periods)
                
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
        st.header("Welcome to Elite Financial Analytics Platform v4.0")
        
        # Feature cards
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
            - Pattern recognition
            - Automated insights
            - Confidence-based recommendations
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
            2. **AI Mapping**: Let AI automatically map your metrics or do it manually
            3. **Analyze**: Explore comprehensive analysis with ratios, trends, and forecasts
            4. **Query**: Ask questions in natural language about your data
            5. **Collaborate**: Share your analysis with team members
            6. **Export**: Generate professional reports in various formats
            
            **New in v4.0:**
            - 🚀 ML-powered forecasting
            - 💬 Natural language queries
            - 👥 Real-time collaboration
            - 📈 Enhanced performance
            - 🔒 Advanced security features
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
            # Group insights by type
            for i, insight in enumerate(insights[:8]):  # Show top 8 insights
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
                
                # Show details
                with st.expander("View Anomaly Details"):
                    for anomaly_type, items in anomalies.items():
                        if items:
                            st.write(f"**{anomaly_type.replace('_', ' ').title()}:**")
                            anomaly_df = pd.DataFrame(items)
                            st.dataframe(anomaly_df, use_container_width=True)
        
        # Quick visualizations
        st.subheader("Quick Visualizations")
        
        # Extract key metrics for visualization
        metrics = analysis.get('metrics', {})
        
        if metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue trend
                revenue_data = metrics.get('revenue', [])
                if revenue_data:
                    self._render_metric_chart(revenue_data[0], "Revenue Trend")
            
            with col2:
                # Profitability trend
                profit_data = metrics.get('net_income', [])
                if profit_data:
                    self._render_metric_chart(profit_data[0], "Net Income Trend")
    
    def _render_metric_chart(self, metric_data: Dict, title: str):
        """Render a simple metric chart"""
        values = metric_data.get('values', {})
        
        if values:
            years = list(values.keys())
            amounts = list(values.values())
            
            # Get formatter based on preference
            if self.get_state('number_format_value') == 'Indian':
                formatter = format_indian_number
            else:
                formatter = format_international_number
            
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
            
            # Add trend line
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
        
        # Check if mappings exist
        if not self.get_state('metric_mappings'):
            st.warning("Please map metrics first to calculate ratios")
            
            # Show mapping options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🤖 Auto-map with AI", type="primary", key="ai_map_ratios"):
                    self._perform_ai_mapping(data)
            
            with col2:
                if st.button("✏️ Manual Mapping", key="manual_map_ratios"):
                    self.set_state('show_manual_mapping', True)
            
            # Show manual mapping if requested
            if self.get_state('show_manual_mapping', False):
                manual_mapper = ManualMappingInterface(data)
                mappings = manual_mapper.render()
                
                if st.button("✅ Apply Mappings", type="primary", key="apply_manual_mappings"):
                    self.set_state('metric_mappings', mappings)
                    st.success(f"Applied {len(mappings)} mappings!")
                    self.set_state('show_manual_mapping', False)
            
            return
        
        # Apply mappings and calculate ratios
        mappings = self.get_state('metric_mappings')
        mapped_df = data.rename(index=mappings)
        
        # Calculate ratios with performance monitoring
        with st.spinner("Calculating ratios..."):
            with performance_monitor.measure("ratio_calculation"):
                analysis = self.components['analyzer'].analyze_financial_statements(mapped_df)
                ratios = analysis.get('ratios', {})
        
        if not ratios:
            st.error("Unable to calculate ratios. Please check your mappings.")
            if st.button("🔄 Re-map Metrics"):
                self.set_state('metric_mappings', None)
            return
        
        # Get formatter
        if self.get_state('number_format_value') == 'Indian':
            formatter = format_indian_number
        else:
            formatter = format_international_number
        
        # Display ratios by category
        for category, ratio_df in ratios.items():
            if isinstance(ratio_df, pd.DataFrame) and not ratio_df.empty:
                st.subheader(f"{category} Ratios")
                
                # Format based on number format preference
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
    
    @error_boundary()
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
                    'R²': trend_info.get('r_squared', None),
                    'Trend Strength': 'Strong' if trend_info.get('r_squared', 0) > 0.8 else 'Moderate' if trend_info.get('r_squared', 0) > 0.5 else 'Weak'
                })
        
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            
            # Format and display with color coding
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
        
        # Interactive visualization
        st.subheader("Trend Visualization")
        
        # Select metrics to visualize
        numeric_metrics = data.select_dtypes(include=[np.number]).index.tolist()
        selected_metrics = st.multiselect(
            "Select metrics to visualize:",
            numeric_metrics,
            default=numeric_metrics[:3] if len(numeric_metrics) >= 3 else numeric_metrics
        )
        
        if selected_metrics:
            # Visualization options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_trend_lines = st.checkbox("Show Trend Lines", value=True)
            
            with col2:
                normalize = st.checkbox("Normalize Values", value=False)
            
            with col3:
                chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Area"])
            
            # Create visualization
            fig = go.Figure()
            
            for i, metric in enumerate(selected_metrics):
                values = data.loc[metric]
                
                if normalize:
                    # Normalize to base 100
                    values = (values / values.iloc[0]) * 100
                
                # Add main trace
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
                
                # Add trend line if requested
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
        st.subheader("Statistical Analysis")
        
        if selected_metrics:
            # Correlation matrix
            if len(selected_metrics) > 1:
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
        
        # Check if mappings exist
        if not self.get_state('pn_mappings'):
            st.info("Configure Penman-Nissim mappings to proceed")
            
            # Mapping interface
            with st.expander("⚙️ Configure P-N Mappings", expanded=True):
                available_metrics = [''] + [str(m) for m in data.index.tolist()]
                
                # Essential mappings
                col1, col2, col3 = st.columns(3)
                
                mappings = {}
                
                # Define mapping fields
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
                
                # Create columns for each category
                cols = [col1, col2, col3]
                
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
                
                # Remove empty mappings
                mappings = {k: v for k, v in mappings.items() if k}
                
                if st.button("Apply P-N Mappings", type="primary"):
                    if len(mappings) >= 8:
                        self.set_state('pn_mappings', mappings)
                        st.success("Mappings applied successfully!")
                    else:
                        st.error("Please provide at least 8 mappings for analysis")
            
            return
        
        # Run analysis
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
        
        # Display results
        if self.get_state('pn_results'):
            results = self.get_state('pn_results')
            
            # Key metrics
            st.subheader("Key Penman-Nissim Metrics")
            
            if 'ratios' in results:
                ratios_df = results['ratios']
                
                # Display key ratios with explanations
                col1, col2, col3, col4 = st.columns(4)
                
                key_ratios = [
                    ('Return on Net Operating Assets (RNOA) %', 'RNOA', 'success'),
                    ('Financial Leverage (FLEV)', 'FLEV', 'info'),
                    ('Net Borrowing Cost (NBC) %', 'NBC', 'warning'),
                    ('Operating Profit Margin (OPM) %', 'OPM', 'primary')
                ]
                
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
                    
                    # Format numbers
                    if self.get_state('number_format_value') == 'Indian':
                        formatter = format_indian_number
                    else:
                        formatter = format_international_number
                    
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
                
                # Create value driver visualization
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
                
                # Waterfall chart for latest year
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
            
            # Generate insights based on results
            insights = []
            
            if 'ratios' in results:
                ratios_df = results['ratios']
                
                # RNOA insight
                if 'Return on Net Operating Assets (RNOA) %' in ratios_df.index:
                    rnoa_latest = ratios_df.loc['Return on Net Operating Assets (RNOA) %'].iloc[-1]
                    if rnoa_latest > 15:
                        insights.append("✅ Strong operating performance with RNOA above 15%")
                    elif rnoa_latest < 8:
                        insights.append("⚠️ Low RNOA indicates operational efficiency concerns")
                
                # Spread insight
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
        """Render industry comparison tab"""
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
        
        # Enhanced industry benchmarks
        industry_benchmarks = {
            'Technology': {
                'Current Ratio': 2.5,
                'Quick Ratio': 2.2,
                'ROE %': 22.5,
                'ROA %': 12.8,
                'Net Profit Margin %': 15.2,
                'Debt to Equity': 0.45,
                'Asset Turnover': 0.85,
                'Revenue Growth %': 18.5
            },
            'Healthcare': {
                'Current Ratio': 1.8,
                'Quick Ratio': 1.5,
                'ROE %': 18.5,
                'ROA %': 9.2,
                'Net Profit Margin %': 12.5,
                'Debt to Equity': 0.65,
                'Asset Turnover': 0.72,
                'Revenue Growth %': 12.3
            },
            'Financial Services': {
                'Current Ratio': 1.2,
                'Quick Ratio': 1.1,
                'ROE %': 12.5,
                'ROA %': 1.2,
                'Net Profit Margin %': 18.5,
                'Debt to Equity': 4.5,
                'Asset Turnover': 0.08,
                'Revenue Growth %': 8.5
            },
            'Manufacturing': {
                'Current Ratio': 1.8,
                'Quick Ratio': 1.2,
                'ROE %': 15.5,
                'ROA %': 7.2,
                'Net Profit Margin %': 8.5,
                'Debt to Equity': 0.85,
                'Asset Turnover': 1.2,
                'Revenue Growth %': 6.5
            },
            'Retail': {
                'Current Ratio': 1.5,
                'Quick Ratio': 0.8,
                'ROE %': 20.5,
                'ROA %': 8.5,
                'Net Profit Margin %': 4.5,
                'Debt to Equity': 0.95,
                'Asset Turnover': 2.2,
                'Revenue Growth %': 9.5
            }
        }
        
        # Add remaining industries with default values
        for industry in industries:
            if industry not in industry_benchmarks:
                industry_benchmarks[industry] = {
                    'Current Ratio': 1.5,
                    'Quick Ratio': 1.2,
                    'ROE %': 15.0,
                    'ROA %': 8.0,
                    'Net Profit Margin %': 10.0,
                    'Debt to Equity': 0.8,
                    'Asset Turnover': 1.0,
                    'Revenue Growth %': 10.0
                }
        
        # Get company's ratios
        company_ratios = {}
        
        if self.get_state('metric_mappings'):
            mappings = self.get_state('metric_mappings')
            mapped_df = data.rename(index=mappings)
            
            analysis = self.components['analyzer'].analyze_financial_statements(mapped_df)
            
            # Extract ratios for selected year
            for category, ratio_df in analysis.get('ratios', {}).items():
                if isinstance(ratio_df, pd.DataFrame) and comparison_year in ratio_df.columns:
                    for ratio in ratio_df.index:
                        # Standardize ratio names for comparison
                        ratio_name = ratio.replace('Return on Equity %', 'ROE %')\
                                          .replace('Return on Assets %', 'ROA %')
                        company_ratios[ratio_name] = ratio_df.loc[ratio, comparison_year]
            
            # Add growth metrics
            trends = analysis.get('trends', {})
            revenue_trends = [v for k, v in trends.items() if 'revenue' in k.lower()]
            if revenue_trends and revenue_trends[0].get('cagr') is not None:
                company_ratios['Revenue Growth %'] = revenue_trends[0]['cagr']
        
        # Comparison visualization
        if selected_industry in industry_benchmarks and company_ratios:
            st.subheader(f"Performance vs {selected_industry} Industry")
            
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
                    name='Your Company',
                    line=dict(color='rgb(31, 119, 180)', width=2),
                    fillcolor='rgba(31, 119, 180, 0.2)'
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=industry_values,
                    theta=categories,
                    fill='toself',
                    name=f'{selected_industry} Average',
                    line=dict(color='rgb(255, 127, 14)', width=2),
                    fillcolor='rgba(255, 127, 14, 0.2)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(max(company_values), max(industry_values)) * 1.2]
                        )),
                    showlegend=True,
                    title=f"Company vs {selected_industry} Industry Benchmarks",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed comparison table
                st.subheader("Detailed Comparison")
                
                comparison_data = []
                for metric, industry_value in benchmarks.items():
                    if metric in company_ratios:
                        company_value = company_ratios[metric]
                        difference = company_value - industry_value
                        performance_pct = ((company_value / industry_value) - 1) * 100 if industry_value != 0 else 0
                        
                        comparison_data.append({
                            'Metric': metric,
                            'Your Company': f"{company_value:.2f}",
                            'Industry Avg': f"{industry_value:.2f}",
                            'Difference': f"{difference:+.2f}",
                            'Performance': f"{performance_pct:+.1f}%"
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Style the dataframe
                def highlight_performance(val):
                    if isinstance(val, str) and '%' in val:
                        num_val = float(val.replace('%', '').replace('+', ''))
                        if num_val > 10:
                            return 'background-color: #d4edda; color: #155724'
                        elif num_val < -10:
                            return 'background-color: #f8d7da; color: #721c24'
                        else:
                            return 'background-color: #fff3cd; color: #856404'
                    return ''
                
                st.dataframe(
                    comparison_df.style.applymap(highlight_performance, subset=['Performance']),
                    use_container_width=True
                )
                
                # Industry insights
                st.subheader("Industry Insights")
                
                outperforming = []
                underperforming = []
                
                for item in comparison_data:
                    perf = float(item['Performance'].replace('%', '').replace('+', ''))
                    if perf > 10:
                        outperforming.append((item['Metric'], perf))
                    elif perf < -10:
                        underperforming.append((item['Metric'], perf))
                
                if outperforming:
                    st.success(f"**Outperforming in {len(outperforming)} areas:**")
                    for metric, perf in outperforming[:3]:
                        st.write(f"• {metric}: {perf:+.1f}% above industry average")
                
                if underperforming:
                    st.warning(f"**Underperforming in {len(underperforming)} areas:**")
                    for metric, perf in underperforming[:3]:
                        st.write(f"• {metric}: {perf:.1f}% below industry average")
                
                # Peer comparison
                st.subheader("Peer Company Comparison")
                
                # Mock peer data (in production, this would come from a database)
                peer_companies = {
                    'Technology': ['TechCorp A', 'TechCorp B', 'TechCorp C'],
                    'Healthcare': ['HealthCo A', 'HealthCo B', 'HealthCo C'],
                    'Manufacturing': ['ManuCo A', 'ManuCo B', 'ManuCo C'],
                    'Retail': ['RetailCo A', 'RetailCo B', 'RetailCo C'],
                    'Financial Services': ['FinCo A', 'FinCo B', 'FinCo C']
                }
                
                if selected_industry in peer_companies:
                    peers = peer_companies[selected_industry]
                    
                    # Generate mock peer data
                    peer_data = []
                    for peer in peers:
                        peer_metrics = {}
                        for metric, industry_val in benchmarks.items():
                            # Generate random variation around industry average
                            variation = np.random.uniform(0.8, 1.2)
                            peer_metrics[metric] = industry_val * variation
                        peer_data.append(peer_metrics)
                    
                    # Create peer comparison chart
                    fig_peers = go.Figure()
                    
                    # Add company data
                    fig_peers.add_trace(go.Bar(
                        name='Your Company',
                        x=list(company_ratios.keys()),
                        y=list(company_ratios.values()),
                        marker_color='rgb(31, 119, 180)'
                    ))
                    
                    # Add peer data
                    for i, (peer, peer_metrics) in enumerate(zip(peers, peer_data)):
                        peer_values = [peer_metrics.get(metric, 0) for metric in company_ratios.keys()]
                        fig_peers.add_trace(go.Bar(
                            name=peer,
                            x=list(company_ratios.keys()),
                            y=peer_values
                        ))
                    
                    fig_peers.update_layout(
                        title='Peer Company Comparison',
                        barmode='group',
                        xaxis_tickangle=-45,
                        height=500
                    )
                    
                    st.plotly_chart(fig_peers, use_container_width=True)
        
        else:
            st.info("Please calculate financial ratios first to enable industry comparison")
            
            # Show sample industry metrics
            st.subheader("Industry Benchmark Reference")
            
            benchmark_df = pd.DataFrame(industry_benchmarks).T
            st.dataframe(
                benchmark_df.style.format("{:.2f}"),
                use_container_width=True
            )
    
    @error_boundary()
    def _render_data_explorer_tab(self, data: pd.DataFrame):
        """Render data explorer tab"""
        st.header("🔍 Data Explorer")
        
        # Data summary
        st.subheader("Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(data))
        with col2:
            st.metric("Total Columns", len(data.columns))
        with col3:
            st.metric("Data Points", data.size)
        with col4:
            missing_pct = (data.isnull().sum().sum() / data.size) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        # Interactive data viewer
        st.subheader("Interactive Data Viewer")
        
        # Display options
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            show_all = st.checkbox("Show all data", value=False)
        with col2:
            transpose = st.checkbox("Transpose", value=False)
        with col3:
            decimal_places = st.number_input("Decimal places", 0, 4, 2)
        with col4:
            highlight_negatives = st.checkbox("Highlight negatives", value=True)
        
        # Display data
        display_df = data.T if transpose else data
        
        if not show_all:
            display_df = display_df.head(20)
        
        # Format numeric columns
        format_dict = {}
        for col in display_df.select_dtypes(include=[np.number]).columns:
            format_dict[col] = f"{{:,.{decimal_places}f}}"
        
        # Apply styling
        styled_df = display_df.style.format(format_dict, na_rep="-")
        
        if highlight_negatives:
            styled_df = styled_df.applymap(
                lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else ''
            )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Advanced filtering
        st.subheader("Advanced Filtering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Metric filter
            selected_metrics = st.multiselect(
                "Select metrics to display:",
                data.index.tolist(),
                default=data.index[:10].tolist() if len(data) >= 10 else data.index.tolist()
            )
        
        with col2:
            # Year filter
            selected_years = st.multiselect(
                "Select years to display:",
                data.columns.tolist(),
                default=data.columns.tolist()
            )
        
        # Apply filters
        if selected_metrics and selected_years:
            filtered_df = data.loc[selected_metrics, selected_years]
            
            st.subheader("Filtered Data")
            st.dataframe(
                filtered_df.style.format(format_dict, na_rep="-"),
                use_container_width=True
            )
            
            # Quick statistics for filtered data
            st.subheader("Filtered Data Statistics")
            
            stats_df = pd.DataFrame({
                'Mean': filtered_df.mean(axis=1),
                'Std': filtered_df.std(axis=1),
                'Min': filtered_df.min(axis=1),
                'Max': filtered_df.max(axis=1),
                'Growth %': ((filtered_df.iloc[:, -1] / filtered_df.iloc[:, 0]) - 1) * 100 if len(filtered_df.columns) > 1 else 0
            })
            
            st.dataframe(
                stats_df.style.format("{:,.2f}"),
                use_container_width=True
            )
            
            # Export options
            st.subheader("Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV export
                csv = filtered_df.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"financial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel export
                # Store filtered data in analysis for export
                analysis_with_data = {'filtered_data': filtered_df}
                excel_buffer = self.export_manager.export_to_excel(analysis_with_data)
                st.download_button(
                    label="Download Excel",
                    data=excel_buffer,
                    file_name=f"financial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col3:
                # JSON export
                json_data = filtered_df.to_json(orient='split', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"financial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Data quality checks
        st.subheader("Data Quality Checks")
        
        validator = DataValidator(self.config)
        validation_result = validator.validate_financial_data(data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Errors", len(validation_result.errors), 
                     delta=-len(validation_result.errors) if validation_result.errors else None,
                     delta_color="inverse")
        
        with col2:
            st.metric("Warnings", len(validation_result.warnings))
        
        with col3:
            st.metric("Info Messages", len(validation_result.info))
        
        # Show validation details
        if validation_result.errors or validation_result.warnings:
            with st.expander("View Validation Details"):
                if validation_result.errors:
                    st.error("**Errors:**")
                    for error in validation_result.errors:
                        st.write(f"• {error}")
                
                if validation_result.warnings:
                    st.warning("**Warnings:**")
                    for warning in validation_result.warnings:
                        st.write(f"• {warning}")
                
                if validation_result.info:
                    st.info("**Information:**")
                    for info in validation_result.info:
                        st.write(f"• {info}")
        
        # Data transformations
        st.subheader("Data Transformations")
        
        transform_options = st.multiselect(
            "Apply transformations:",
            ["Log Transform", "Percentage Change", "Normalize (0-1)", "Standardize (Z-score)", "Moving Average"]
        )
        
        if transform_options and selected_metrics and selected_years:
            transformed_df = filtered_df.copy()
            
            for transform in transform_options:
                if transform == "Log Transform":
                    transformed_df = np.log1p(transformed_df.replace(0, np.nan))
                elif transform == "Percentage Change":
                    transformed_df = transformed_df.pct_change(axis=1) * 100
                elif transform == "Normalize (0-1)":
                    min_vals = transformed_df.min(axis=1)
                    max_vals = transformed_df.max(axis=1)
                    transformed_df = transformed_df.sub(min_vals, axis=0).div(max_vals - min_vals, axis=0)
                elif transform == "Standardize (Z-score)":
                    mean_vals = transformed_df.mean(axis=1)
                    std_vals = transformed_df.std(axis=1)
                    transformed_df = transformed_df.sub(mean_vals, axis=0).div(std_vals, axis=0)
                elif transform == "Moving Average":
                    window_size = st.slider("Window size", 2, len(transformed_df.columns), 3)
                    transformed_df = transformed_df.rolling(window=window_size, axis=1).mean()
            
            st.subheader("Transformed Data")
            st.dataframe(
                transformed_df.style.format("{:,.3f}", na_rep="-"),
                use_container_width=True
            )
    
    @error_boundary()
    def _render_reports_tab(self, data: pd.DataFrame):
        """Render reports tab"""
        st.header("📄 Financial Reports")
        
        # Report configuration
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Select Report Type",
                [
                    "Executive Summary",
                    "Comprehensive Financial Analysis",
                    "Ratio Analysis Report",
                    "Trend Analysis Report",
                    "Penman-Nissim Report",
                    "Industry Comparison Report",
                    "Custom Report"
                ],
                key="report_type"
            )
        
        with col2:
            report_format = st.selectbox(
                "Export Format",
                ["PDF", "Excel", "Word", "PowerPoint", "HTML", "Markdown"],
                key="report_format"
            )
        
        # Report options
        st.subheader("Report Options")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            include_charts = st.checkbox("Include Charts", value=True)
        with col2:
            include_insights = st.checkbox("Include Insights", value=True)
        with col3:
            include_raw_data = st.checkbox("Include Raw Data", value=False)
        with col4:
            include_benchmarks = st.checkbox("Include Benchmarks", value=True)
        
        # Period selection
        if len(data.columns) > 1:
            period_selection = st.select_slider(
                "Report Period",
                options=["Latest Year", "Last 3 Years", "Last 5 Years", "All Years"],
                value="Last 3 Years"
            )
        else:
            period_selection = "All Years"
        
        # Generate report button
        if st.button("🚀 Generate Report", type="primary", key="generate_report"):
            with st.spinner(f"Generating {report_type}..."):
                # Get analysis results
                analysis = self.components['analyzer'].analyze_financial_statements(data)
                
                # Add additional data
                analysis['company_name'] = self.get_state('company_name', 'Financial Analysis')
                analysis['report_date'] = datetime.now()
                analysis['report_type'] = report_type
                analysis['include_charts'] = include_charts
                analysis['include_insights'] = include_insights
                
                # Generate report based on format
                if report_format == "Excel":
                    report_data = self.export_manager.export_to_excel(analysis)
                    file_extension = "xlsx"
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif report_format == "Markdown":
                    report_data = self.export_manager.export_to_markdown(analysis)
                    file_extension = "md"
                    mime_type = "text/markdown"
                else:
                    # For other formats, use markdown as placeholder
                    report_data = self.export_manager.export_to_markdown(analysis)
                    file_extension = "md"
                    mime_type = "text/markdown"
                    st.info(f"{report_format} export will be available in the full version")
                
                st.success("Report generated successfully!")
                
                # Display preview
                with st.expander("Report Preview", expanded=True):
                    if report_format == "Markdown":
                        st.markdown(report_data[:2000] + "\n\n*[Report continues...]*")
                    else:
                        st.info(f"Preview not available for {report_format} format")
                
                # Download button
                st.download_button(
                    label=f"📥 Download {report_format} Report",
                    data=report_data if isinstance(report_data, bytes) else report_data.encode(),
                    file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}",
                    mime=mime_type
                )
        
        # Report templates
        st.subheader("Report Templates")
        
        templates = {
            "Investor Presentation": "Professional presentation for investors with key metrics and growth story",
            "Board Report": "Detailed analysis for board meetings with strategic insights",
            "Audit Report": "Comprehensive report for audit purposes with all calculations",
            "Quick Summary": "One-page executive summary with key highlights",
            "Quarterly Review": "Standard quarterly financial review format"
        }
        
        selected_template = st.selectbox(
            "Use Template",
            ["None"] + list(templates.keys())
        )
        
        if selected_template != "None":
            st.info(f"**{selected_template}**: {templates[selected_template]}")
        
        # Report scheduler
        st.subheader("Report Scheduler")
        
        enable_scheduling = st.checkbox("Enable Automated Reports")
        
        if enable_scheduling:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                frequency = st.selectbox(
                    "Frequency",
                    ["Daily", "Weekly", "Monthly", "Quarterly"]
                )
            
            with col2:
                email = st.text_input("Email Address", placeholder="email@example.com")
            
            with col3:
                if st.button("Schedule Report"):
                    st.success(f"Report scheduled: {frequency} delivery to {email}")
                    st.info("Note: Email functionality requires server configuration")
    
    @error_boundary()
    def _render_ml_insights_tab(self, data: pd.DataFrame):
        """Render ML insights tab with forecasting and advanced analytics"""
        st.header("🤖 Machine Learning Insights")
        
        if not self.config.get('app.enable_ml_features', True):
            st.warning("ML features are disabled. Enable them in settings to use this feature.")
            return
        
        # Forecasting section
        st.subheader("📈 Financial Forecasting")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_periods = st.number_input(
                "Forecast Periods",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of periods to forecast"
            )
        
        with col2:
            model_type = st.selectbox(
                "Model Type",
                ["auto", "linear", "polynomial", "exponential"],
                help="Forecasting model to use"
            )
        
        with col3:
            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Generating forecasts..."):
                    forecast_results = self.ml_forecaster.forecast_metrics(
                        data,
                        periods=forecast_periods,
                        model_type=model_type
                    )
                    
                    self.set_state('ml_forecast_results', forecast_results)
        
        # Display forecast results
        forecast_results = self.get_state('ml_forecast_results')
        
        if forecast_results and 'forecasts' in forecast_results:
            st.subheader("Forecast Results")
            
            for metric, forecast in forecast_results['forecasts'].items():
                with st.expander(f"📊 {metric}", expanded=True):
                    # Create forecast visualization
                    fig = go.Figure()
                    
                    # Historical data
                    historical_values = data.loc[metric].dropna()
                    
                    # Add historical line
                    fig.add_trace(go.Scatter(
                        x=list(historical_values.index),
                        y=list(historical_values.values),
                        mode='lines+markers',
                        name='Historical',
                        line=dict(width=3)
                    ))
                    
                    # Add forecast line
                    forecast_x = forecast['periods']
                    forecast_y = forecast['values']
                    
                    # Connect to last historical point
                    forecast_x = [historical_values.index[-1]] + forecast_x
                    forecast_y = [historical_values.iloc[-1]] + forecast_y
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_x,
                        y=forecast_y,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(dash='dash', width=3)
                    ))
                    
                    # Add confidence intervals if available
                    if 'confidence_intervals' in forecast_results:
                        ci = forecast_results['confidence_intervals'].get(metric, {})
                        if 'lower' in ci and 'upper' in ci:
                            ci_x = forecast['periods']
                            
                            fig.add_trace(go.Scatter(
                                x=ci_x + ci_x[::-1],
                                y=ci['upper'] + ci['lower'][::-1],
                                fill='toself',
                                fillcolor='rgba(0,100,200,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='95% CI',
                                showlegend=True
                            ))
                    
                    fig.update_layout(
                        title=f"{metric} Forecast",
                        xaxis_title="Period",
                        yaxis_title="Value",
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show accuracy metrics
                    if metric in forecast_results.get('accuracy_metrics', {}):
                        accuracy = forecast_results['accuracy_metrics'][metric]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("RMSE", f"{accuracy.get('rmse', 0):.2f}")
                        
                        with col2:
                            if accuracy.get('mape') is not None:
                                st.metric("MAPE", f"{accuracy['mape']:.1f}%")
                        
                        with col3:
                            st.metric("MAE", f"{accuracy.get('mae', 0):.2f}")
        
        # Pattern Recognition
        st.subheader("🔍 Pattern Recognition")
        
        analysis = self.components['analyzer'].analyze_financial_statements(data)
        trends = analysis.get('trends', {})
        
        # Identify significant patterns
        patterns = []
        
        for metric, trend in trends.items():
            if isinstance(trend, dict):
                # Strong growth pattern
                if trend.get('cagr', 0) > 15:
                    patterns.append({
                        'type': 'Strong Growth',
                        'metric': metric,
                        'value': f"{trend['cagr']:.1f}% CAGR",
                        'confidence': trend.get('r_squared', 0)
                    })
                
                # Declining pattern
                elif trend.get('cagr', 0) < -5:
                    patterns.append({
                        'type': 'Declining Trend',
                        'metric': metric,
                        'value': f"{trend['cagr']:.1f}% CAGR",
                        'confidence': trend.get('r_squared', 0)
                    })
                
                # High volatility pattern
                if trend.get('volatility', 0) > 30:
                    patterns.append({
                        'type': 'High Volatility',
                        'metric': metric,
                        'value': f"{trend['volatility']:.1f}% volatility",
                        'confidence': 0.8
                    })
        
        if patterns:
            pattern_df = pd.DataFrame(patterns)
            st.dataframe(
                pattern_df.style.background_gradient(subset=['confidence'], cmap='Greens'),
                use_container_width=True
            )
        else:
            st.info("No significant patterns detected in the data")
        
        # Anomaly Detection Results
        if 'anomalies' in analysis:
            st.subheader("🚨 Anomaly Detection")
            
            anomalies = analysis['anomalies']
            
            # Create anomaly visualization
            if anomalies['value_anomalies']:
                st.write("**Value Anomalies Detected:**")
                
                anomaly_df = pd.DataFrame(anomalies['value_anomalies'])
                fig = px.scatter(anomaly_df, x='year', y='value', color='z_score',
                               hover_data=['metric'], 
                               title="Detected Value Anomalies",
                               color_continuous_scale='Reds')
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_debug_footer(self):
        """Render debug information in footer"""
        with st.expander("🐛 Debug Information"):
            # Performance metrics
            st.subheader("Performance Metrics")
            perf_summary = performance_monitor.get_performance_summary()
            
            if perf_summary:
                perf_df = pd.DataFrame(perf_summary).T
                st.dataframe(perf_df, use_container_width=True)
            
            # Cache statistics
            st.subheader("Cache Statistics")
            
            cache_stats = {}
            if 'mapper' in self.components:
                cache_stats['AI Embeddings Cache'] = self.components['mapper'].embeddings_cache.get_stats()
            if 'analyzer' in self.components:
                cache_stats['Analysis Cache'] = self.components['analyzer'].cache.get_stats()
            
            for cache_name, stats in cache_stats.items():
                st.write(f"**{cache_name}:**")
                stats_df = pd.DataFrame([stats])
                st.dataframe(stats_df, use_container_width=True)
            
            # Session state
            st.subheader("Session State")
            state_data = {k: type(v).__name__ for k, v in st.session_state.items()}
            st.json(state_data)
            
            # Component status
            st.subheader("Component Status")
            component_status = {
                name: "Initialized" if comp._initialized else "Not Initialized"
                for name, comp in self.components.items()
            }
            st.json(component_status)
    
    @error_boundary()
    def _perform_ai_mapping(self, data: pd.DataFrame):
        """Perform AI mapping for the data"""
        source_metrics = data.index.tolist()
        
        with st.spinner("Performing AI-powered metric mapping..."):
            try:
                # Get enhanced mapping with confidence levels
                result = self.components['mapper'].map_metrics_with_confidence_levels(source_metrics)
                
                # Combine all mappings
                all_mappings = {}
                all_mappings.update(result['high_confidence'])
                all_mappings.update(result['medium_confidence'])
                all_mappings.update(result['low_confidence'])
                
                # Extract just the target values for the final mapping
                final_mappings = {k: v['target'] for k, v in all_mappings.items()}
                
                if final_mappings:
                    self.set_state('metric_mappings', final_mappings)
                    self.set_state('ai_mapping_result', result)
                    
                    # Show results
                    st.success(f"✅ AI Mapping Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("High Confidence", len(result['high_confidence']))
                    with col2:
                        st.metric("Medium Confidence", len(result['medium_confidence']))
                    with col3:
                        st.metric("Low Confidence", len(result['low_confidence']))
                    
                    # Show mapping details
                    with st.expander("View Mapping Details", expanded=True):
                        # High confidence mappings
                        if result['high_confidence']:
                            st.success("**High Confidence Mappings (>80%)**")
                            high_conf_df = pd.DataFrame([
                                {
                                    'Source': source,
                                    'Target': details['target'],
                                    'Confidence': f"{details['confidence']:.0%}"
                                }
                                for source, details in result['high_confidence'].items()
                            ])
                            st.dataframe(high_conf_df, use_container_width=True)
                        
                        # Medium confidence mappings
                        if result['medium_confidence']:
                            st.warning("**Medium Confidence Mappings (60-80%)**")
                            med_conf_df = pd.DataFrame([
                                {
                                    'Source': source,
                                    'Target': details['target'],
                                    'Confidence': f"{details['confidence']:.0%}"
                                }
                                for source, details in result['medium_confidence'].items()
                            ])
                            st.dataframe(med_conf_df, use_container_width=True)
                        
                        # Low confidence mappings
                        if result['low_confidence']:
                            st.info("**Low Confidence Mappings (40-60%) - Review Recommended**")
                            low_conf_df = pd.DataFrame([
                                {
                                    'Source': source,
                                    'Target': details['target'],
                                    'Confidence': f"{details['confidence']:.0%}"
                                }
                                for source, details in result['low_confidence'].items()
                            ])
                            st.dataframe(low_conf_df, use_container_width=True)
                    
                    if result['requires_manual']:
                        st.warning(f"⚠️ {len(result['requires_manual'])} metrics require manual mapping")
                        with st.expander("Unmapped Metrics"):
                            for metric in result['requires_manual']:
                                st.write(f"• {metric}")
                else:
                    st.error("Could not map any metrics. Please use manual mapping.")
                    
            except Exception as e:
                self.logger.error(f"AI mapping failed: {e}")
                st.error("AI mapping failed. Please use manual mapping instead.")
    
    @error_boundary()
    def _process_uploaded_files(self, files: List[UploadedFile]):
        """Process uploaded files with enhanced parsing"""
        try:
            all_data = []
            
            for file in files:
                self.logger.info(f"Processing file: {file.name}, size: {file.size} bytes")
                
                # Detect file source
                file_source = self._detect_file_source(file.name)
                if file_source:
                    st.info(f"📊 Detected source: {file_source}")
                
                try:
                    if file.name.endswith('.csv'):
                        df = self._process_csv_file(file)
                    
                    elif file.name.endswith(('.xls', '.xlsx')):
                        df = self._process_excel_or_html_file(file, file_source)
                    
                    elif file.name.endswith(('.html', '.htm')):
                        df = self._process_html_file(file, file_source)
                    
                    else:
                        st.warning(f"Unsupported file type: {file.name}")
                        df = None
                    
                    # Post-processing
                    if df is not None and file_source:
                        df = self._apply_source_specific_cleaning(df, file_source)
                    
                    # Validation
                    if df is not None:
                        df = self._validate_and_clean_dataframe(df, file.name)
                        if df is not None:
                            # Apply security sanitization
                            df = self.components['security'].sanitize_dataframe(df)
                            all_data.append(df)
                            st.success(f"✅ Successfully parsed {file.name}")
                            
                except Exception as e:
                    self.logger.error(f"Error processing {file.name}: {e}")
                    st.error(f"Error processing {file.name}: {str(e)}")
                    if self.config.get('app.debug', False):
                        st.exception(e)
            
            # Merge and finalize data
            if all_data:
                self._merge_and_finalize_data(all_data, files)
            else:
                self._show_no_data_message()
                
        except Exception as e:
            self.logger.error(f"Error processing files: {e}", exc_info=True)
            st.error(f"Error processing files: {str(e)}")
    
    def _detect_file_source(self, filename: str) -> Optional[str]:
        """Detect the source of the file based on naming patterns"""
        filename_lower = filename.lower()
        
        # Common patterns from financial data providers
        patterns = {
            'Capitaline': ['capitaline', 'capline', 'cap_', 'cashflow_'],
            'Moneycontrol': ['moneycontrol', 'mc_', 'mcontrol'],
            'BSE': ['bse', 'bseindia', 'bombay_stock'],
            'NSE': ['nse', 'nseindia', 'national_stock'],
            'Screener': ['screener', 'screener.in'],
            'MCA': ['mca', 'ministry_corporate'],
            'Bloomberg': ['bloomberg', 'bbg'],
            'Reuters': ['reuters', 'thomsonreuters'],
            'CapitalIQ': ['capitaliq', 'capiq', 's&p'],
            'Yahoo Finance': ['yahoo', 'yfinance']
        }
        
        for source, keywords in patterns.items():
            if any(keyword in filename_lower for keyword in keywords):
                return source
        
        return None
    
    def _process_csv_file(self, file: UploadedFile) -> Optional[pd.DataFrame]:
        """Process CSV file"""
        try:
            # First try with index_col=0
            df = pd.read_csv(file, index_col=0)
            self.logger.info(f"Read CSV with index_col=0, shape: {df.shape}")
        except Exception as e1:
            self.logger.warning(f"Failed with index_col=0: {e1}")
            file.seek(0)  # Reset file pointer
            try:
                # Try without index_col
                df = pd.read_csv(file)
                # Use first column as index if it looks like metric names
                if df.iloc[:, 0].dtype == 'object':
                    df = df.set_index(df.columns[0])
                self.logger.info(f"Read CSV without index_col, shape: {df.shape}")
            except Exception as e2:
                self.logger.error(f"Failed to read CSV: {e2}")
                st.error(f"Error reading {file.name}: {str(e2)}")
                return None
        
        return df
    
    def _process_excel_or_html_file(self, file: UploadedFile, source: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Process files that might be Excel or HTML masquerading as Excel"""
        engine = 'xlrd' if file.name.endswith('.xls') else 'openpyxl'
        
        # First, check if it's actually HTML by peeking at the content
        file.seek(0)
        first_bytes = file.read(1024)  # Read first 1KB
        file.seek(0)  # Reset
        
        # Check for HTML signatures
        is_likely_html = any(marker in first_bytes.lower() for marker in [
            b'<html', b'<!doctype', b'<table', b'<head', b'<?xml'
        ])
        
        if is_likely_html:
            self.logger.info(f"{file.name} detected as HTML disguised as Excel")
            st.info(f"📄 {file.name} appears to be an HTML export. Using specialized parser...")
            return self._process_html_financial_export(file, source)
        
        # Try standard Excel parsing
        try:
            df = pd.read_excel(file, index_col=0, engine=engine)
            self.logger.info(f"Successfully read {file.name} as standard Excel")
            return df
        except Exception as e:
            error_msg = str(e)
            
            # If it fails with Excel-specific errors, try HTML
            if any(err in error_msg for err in [
                "Expected BOF record", "not a valid", "Unsupported format",
                "corrupt", "Can't find workbook", "found b'<html"
            ]):
                self.logger.warning(f"Excel parsing failed, attempting HTML fallback: {error_msg}")
                return self._process_html_financial_export(file, source)
            else:
                # Try without index_col as fallback
                try:
                    file.seek(0)
                    df = pd.read_excel(file, engine=engine)
                    if df.iloc[:, 0].dtype == 'object':
                        df = df.set_index(df.columns[0])
                    return df
                except Exception as e2:
                    st.error(f"Could not read {file.name}: {e2}")
                    return None
    
    def _process_html_file(self, file: UploadedFile, source: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Process HTML file"""
        return self._process_html_financial_export(file, source)
    
    def _process_html_financial_export(self, file: UploadedFile, source: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Process HTML exports from financial data providers"""
        try:
            file.seek(0)
            content = file.read()
            
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    content_str = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                content_str = content.decode('utf-8', errors='ignore')
            
            # Clean HTML for better parsing
            content_str = self._preprocess_financial_html(content_str, source)
            
            # Parse HTML tables
            df = None
            error_messages = []
            
            # Strategy 1: Simplest possible parsing
            try:
                tables = pd.read_html(
                    io.StringIO(content_str),
                    header=None,
                    index_col=None,
                    thousands=',',
                    na_values=['', '-', 'NA', 'N/A'],
                    keep_default_na=True
                )
                
                if tables:
                    # Take the largest table
                    df = max(tables, key=lambda x: x.size)
                    self.logger.info(f"Simple parsing successful, got table with shape {df.shape}")
                    
            except Exception as e:
                error_messages.append(f"Simple parsing: {str(e)}")
                self.logger.error(f"Simple parsing failed: {e}")
            
            if df is None:
                st.error("Could not parse the HTML file")
                return None
            
            # Post-process the dataframe
            st.info(f"Successfully parsed table with shape {df.shape}")
            
            # Clean up the dataframe
            df = self._clean_parsed_html_table(df, source)
            
            return df
            
        except Exception as e:
            self.logger.error(f"HTML parsing failed: {e}", exc_info=True)
            st.error(f"Failed to parse HTML data: {e}")
            return None
    
    def _clean_parsed_html_table(self, df: pd.DataFrame, source: Optional[str] = None) -> pd.DataFrame:
        """Clean a parsed HTML table"""
        try:
            self.logger.info(f"Cleaning table with shape {df.shape}")
            
            # Step 1: Remove completely empty rows and columns
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            
            # Step 2: Try to identify the structure
            # Look for a column that might be metric names
            metric_col_idx = None
            for i, col in enumerate(df.columns):
                if df.iloc[:, i].dtype == 'object':
                    # Check if this column has financial terms
                    sample = df.iloc[:, i].dropna().astype(str).str.lower()
                    if any('cash' in s or 'revenue' in s or 'asset' in s or 'income' in s for s in sample.head(20)):
                        metric_col_idx = i
                        break
            
            # Step 3: Restructure if we found a metric column
            if metric_col_idx is not None:
                # Set the metric column as index
                df = df.set_index(df.columns[metric_col_idx])
                df.index.name = 'Metrics'
                
                # Clean the index
                df.index = df.index.astype(str).str.strip()
                df = df[df.index != '']
            
            # Step 4: Clean column names (potential years)
            new_columns = []
            for col in df.columns:
                col_str = str(col).strip()
                # Remove common prefixes/suffixes
                col_str = col_str.replace('Unnamed:', '').strip()
                new_columns.append(col_str if col_str else f'Col_{len(new_columns)}')
            
            df.columns = new_columns
            
            # Step 5: Convert numeric columns
            for col in df.columns:
                try:
                    # Only convert if it looks numeric
                    sample = df[col].dropna().astype(str).head(5)
                    if any(any(c.isdigit() for c in str(s)) for s in sample):
                        df[col] = pd.to_numeric(
                            df[col].astype(str).str.replace(',', '').str.replace('(', '-').str.replace(')', ''),
                            errors='coerce'
                        )
                except:
                    continue
            
            # Step 6: Final cleanup
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            
            self.logger.info(f"Cleaning complete. Final shape: {df.shape}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning table: {e}")
            return df
    
    def _preprocess_financial_html(self, html_content: str, source: Optional[str] = None) -> str:
        """Preprocess HTML content based on source-specific quirks"""
        import re
        
        # Remove scripts and styles
        html_content = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Source-specific preprocessing
        if source == 'Capitaline':
            # Capitaline often has merged cells and complex headers
            html_content = re.sub(r'rowspan=[\"\']?\d+[\"\']?', '', html_content)
            html_content = re.sub(r'colspan=[\"\']?\d+[\"\']?', '', html_content)
            # Remove footnote markers
            html_content = re.sub(r'<sup>.*?</sup>', '', html_content)
            
        elif source == 'Moneycontrol':
            # Moneycontrol uses specific class names
            html_content = re.sub(r'class=[\"\']?.*?[\"\']?', '', html_content)
            
        elif source in ['BSE', 'NSE']:
            # Exchange data often has extra formatting
            html_content = re.sub(r'&nbsp;', ' ', html_content)
            html_content = re.sub(r'\s+', ' ', html_content)
        
        return html_content
    
    def _apply_source_specific_cleaning(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Apply source-specific data cleaning rules"""
        if source == 'Capitaline':
            # Capitaline specific cleaning
            # Remove footnote rows
            df = df[~df.index.str.contains('Note:|Source:', case=False, na=False)]
            
            # Standardize metric names
            rename_map = {
                'PBDIT': 'EBITDA',
                'PBDT': 'EBIT',
                'PAT': 'Net Income',
                'Total Income': 'Revenue'
            }
            df.index = df.index.to_series().replace(rename_map)
            
        elif source == 'Moneycontrol':
            # Remove percentage columns
            df = df.loc[:, ~df.columns.str.contains('%', na=False)]
        
        return df
    
    def _validate_and_clean_dataframe(self, df: pd.DataFrame, filename: str) -> Optional[pd.DataFrame]:
        """Validate and clean the dataframe"""
        try:
            # Log initial state
            self.logger.info(f"Validating {filename}: shape={df.shape if df is not None else None}")
            
            # Basic validation
            if df is None:
                st.error(f"{filename}: Dataframe is None")
                return None
                
            if df.empty:
                st.error(f"{filename}: Dataframe is empty")
                return None
            
            if df.shape[0] < 2:
                st.error(f"{filename}: Insufficient rows ({df.shape[0]} rows found, minimum 2 required)")
                return None
                
            if df.shape[1] < 2:
                st.error(f"{filename}: Insufficient columns ({df.shape[1]} columns found, minimum 2 required)")
                return None
            
            # Check for numeric data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.logger.info(f"Found {len(numeric_cols)} numeric columns")
            
            if len(numeric_cols) == 0:
                st.warning(f"{filename}: No numeric columns found. Data preview:")
                st.dataframe(df.head())
                return None
            
            # Remove completely empty rows and columns
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            
            # Final validation
            if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
                st.error(f"{filename}: Insufficient valid data after cleaning")
                return None
            
            st.success(f"✅ {filename}: Valid data found - {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error validating {filename}: {e}", exc_info=True)
            st.error(f"Validation error for {filename}: {str(e)}")
            return None
    
    def _merge_and_finalize_data(self, all_data: List[pd.DataFrame], files: List[UploadedFile]):
        """Merge multiple dataframes and finalize processing"""
        st.info(f"Successfully parsed {len(all_data)} file(s)")
        
        if len(all_data) == 1:
            merged_data = all_data[0]
        else:
            try:
                # More robust merging logic
                st.info("Attempting to merge multiple files...")
                
                # First, check if files have overlapping columns (years)
                all_columns = [set(df.columns) for df in all_data]
                common_columns = set.intersection(*all_columns) if all_columns else set()
                
                if common_columns:
                    # Files have common years - likely different metrics for same years
                    st.info(f"Found common columns: {common_columns}")
                    
                    # Merge by combining rows (metrics)
                    merged_data = pd.concat(all_data, axis=0)
                    
                    # Remove duplicate rows based on index
                    merged_data = merged_data[~merged_data.index.duplicated(keep='first')]
                    st.success("Merged files by combining metrics")
                    
                else:
                    # No common columns - likely same metrics for different years
                    st.info("No common columns found - attempting to merge by years")
                    
                    # Check if indices are similar
                    index_similarity = sum(
                        len(set(df.index) & set(all_data[0].index)) / len(df.index) 
                        for df in all_data[1:]
                    ) / (len(all_data) - 1)
                    
                    if index_similarity > 0.7:  # 70% similarity threshold
                        # Merge by columns (years)
                        merged_data = pd.concat(all_data, axis=1)
                        
                        # Remove duplicate columns
                        merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]
                        st.success("Merged files by combining years")
                    else:
                        # Different structure - use first file and warn
                        st.warning("Files have different structures. Using first file only.")
                        st.info("For best results, ensure all files have similar metric names.")
                        merged_data = all_data[0]
                        
            except Exception as e:
                self.logger.error(f"Error merging data: {e}")
                st.warning(f"Could not merge files automatically: {str(e)}")
                st.info("Using first file only. You can process files separately if needed.")
                merged_data = all_data[0]
        
        # Show data preview
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(merged_data.head(10))
            st.write(f"Shape: {merged_data.shape}")
            
            # Show column information
            st.write("**Columns (Years/Periods):**", list(merged_data.columns))
            st.write("**Sample Metrics:**", list(merged_data.index[:5]))
        
        # Process and validate
        try:
            with performance_monitor.measure("data_processing"):
                processed_data, validation = self.components['processor'].process(merged_data)
            
            if validation.is_valid:
                # Store in session state
                self.set_state('analysis_data', processed_data)
                self.set_state('company_name', files[0].name.split('.')[0])
                self.set_state('data_source', self._detect_file_source(files[0].name))
                st.success("✅ Files processed successfully!")
                
                # Show corrections if any
                if validation.corrections:
                    with st.expander("📝 Data Corrections Applied"):
                        for correction in validation.corrections:
                            st.write(f"• {correction}")
                
                # Auto-map if AI is enabled
                if self.config.get('ai.enabled', True) and self.config.get('app.display_mode') != Configuration.DisplayMode.MINIMAL:
                    self._perform_ai_mapping(processed_data)
                
            else:
                st.error("Validation failed:")
                for error in validation.errors[:5]:  # Show max 5 errors
                    st.error(f"• {error}")
                if len(validation.errors) > 5:
                    st.error(f"... and {len(validation.errors) - 5} more errors")
                    
                for warning in validation.warnings[:3]:  # Show max 3 warnings
                    st.warning(f"• {warning}")
                    
        except Exception as e:
            self.logger.error(f"Error processing merged data: {e}")
            st.error(f"Error processing data: {str(e)}")
            
            # Show data structure for debugging
            with st.expander("Debug Information"):
                st.write("**Data Shape:**", merged_data.shape)
                st.write("**Data Types:**")
                st.write(merged_data.dtypes)
                st.write("**Sample Data:**")
                st.write(merged_data.head())
    
    def _show_no_data_message(self):
        """Show message when no valid data is found"""
        st.error("No valid data found in uploaded files")
        st.info("""
        **Troubleshooting tips:**
        1. Ensure your file has financial metrics in rows and years in columns
        2. The first column should contain metric names
        3. Other columns should contain years or periods
        4. Try enabling 'Use simple parsing mode' in the sidebar
        
        **Example structure:**
        """)
        
        example_df = pd.DataFrame({
            'Metric': ['Revenue', 'Total Assets', 'Net Income'],
            '2021': [100000, 500000, 20000],
            '2022': [120000, 550000, 25000],
            '2023': [140000, 600000, 30000]
        }).set_index('Metric')
        
        st.dataframe(example_df)
    
    @error_boundary()
    def _load_sample_data(self, sample_name: str):
        """Load sample data"""
        try:
            if "Indian Tech" in sample_name:
                data = self.sample_generator.generate_indian_tech_company()
                company_name = "TechCorp India Ltd."
            elif "US Manufacturing" in sample_name:
                data = self.sample_generator.generate_us_manufacturing()
                company_name = "AmeriManufacturing Corp."
            elif "European Retail" in sample_name:
                data = self.sample_generator.generate_european_retail()
                company_name = "EuroRetail AG"
            else:
                data = self.sample_generator.generate_indian_tech_company()
                company_name = "Sample Company"
            
            # Process and validate
            processed_data, validation = self.components['processor'].process(data)
            
            if validation.is_valid:
                self.set_state('analysis_data', processed_data)
                self.set_state('company_name', company_name)
                st.success(f"✅ Loaded sample data: {company_name}")
                
                # Auto-map if AI is enabled
                if self.config.get('ai.enabled', True) and self.config.get('app.display_mode') != Configuration.DisplayMode.MINIMAL:
                    self._perform_ai_mapping(processed_data)
                
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
        
        # Clear performance metrics
        performance_monitor.clear_metrics()
        
        self.logger.info("All caches cleared")
    
    def _reset_configuration(self):
        """Reset configuration to defaults"""
        # Clear session state
        for key in list(st.session_state.keys()):
            if key not in ['initialized']:
                del st.session_state[key]
        
        # Reinitialize
        st.session_state.initialized = True
        st.session_state.analysis_data = None
        st.session_state.metric_mappings = None
        st.session_state.pn_mappings = None
        st.session_state.pn_results = None
        st.session_state.ai_mapping_result = None
        st.session_state.company_name = None
        st.session_state.data_source = None
        st.session_state.show_manual_mapping = False
        st.session_state.config_overrides = {}
        st.session_state.uploaded_files = []
        st.session_state.simple_parse_mode = False
        st.session_state.number_format_value = 'Indian'
        st.session_state.show_tutorial = True
        st.session_state.tutorial_step = 0
        st.session_state.collaboration_session = None
        st.session_state.query_history = []
        st.session_state.ml_forecast_results = None
        
        self.logger.info("Configuration reset to defaults")
        st.success("Configuration reset successfully!")
    
    def _export_logs(self):
        """Export application logs"""
        try:
            log_dir = Path("logs")
            if log_dir.exists():
                # Create zip file with all logs
                import zipfile
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for log_file in log_dir.glob("*.log"):
                        zip_file.write(log_file, log_file.name)
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Logs",
                    data=zip_buffer.read(),
                    file_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
            else:
                st.warning("No logs found")
                
        except Exception as e:
            self.logger.error(f"Error exporting logs: {e}")
            st.error(f"Error exporting logs: {str(e)}")

# --- 29. Application Entry Point ---
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
