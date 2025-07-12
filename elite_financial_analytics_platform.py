# elite_financial_analytics_platform.py
# Enhanced Financial Analytics Platform with IND-AS Support and AI Features
# Now imports from financial_analytics_core.py

# --- 1. Imports and Dependencies ---
import io
import os
import re
import sys
import json
import pickle
import hashlib
import logging
import warnings
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from datetime import datetime
from pathlib import Path
from functools import lru_cache, wraps
import time

# Scientific and Data Processing
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning and AI
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Text Processing and Security
import bleach
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer

# --- IMPORT FROM CORE FINANCIAL ANALYTICS ---
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
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported core financial analytics components")
except ImportError as e:
    CORE_COMPONENTS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import core components: {e}")
    st.error("❌ Core financial analytics components not found!")
    st.error("Please ensure 'financial_analytics_core.py' is in the same directory")
    st.stop()

# --- 2. Configuration and Constants ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Application Constants
APP_VERSION = "2.0.0"
MAX_FILE_SIZE_MB = CORE_MAX_FILE_SIZE
ALLOWED_FILE_TYPES = CORE_ALLOWED_TYPES
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Use core constants
YEAR_REGEX = CORE_YEAR_REGEX

# Merge required metrics with IND-AS specific ones
REQUIRED_METRICS = CORE_REQUIRED_METRICS.copy()
REQUIRED_METRICS.update({
    'IND-AS': ['CSR Expense', 'Related Party Transactions', 'Deferred Tax Assets', 'Deferred Tax Liabilities'],
    'Indian_Specific': ['Dividend Distribution Tax', 'Securities Transaction Tax', 'GST Payable']
})

# IND-AS Specific Constants
INDIAN_NUMBER_REGEX = re.compile(r'₹?\s*([\d,]+\.?\d*)\s*(crores?|lakhs?|lacs?|millions?|mn|cr|l)?', re.IGNORECASE)

INDAS_DATE_FORMATS = [
    r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
    r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{2,4})',
    r'(March|Mar|September|Sept?)\s+(\d{1,2}),?\s+(\d{4})'
]

# Financial Metric Categories
METRIC_CATEGORIES = {
    'balance_sheet': {
        'assets': ['Current Assets', 'Non-current Assets', 'Total Assets', 'Cash and Cash Equivalents', 
                  'Inventory', 'Trade Receivables', 'Property Plant and Equipment', 'Intangible Assets'],
        'liabilities': ['Current Liabilities', 'Non-current Liabilities', 'Total Liabilities', 
                       'Short-term Borrowings', 'Long-term Debt', 'Trade Payables'],
        'equity': ['Total Equity', 'Share Capital', 'Reserves and Surplus', 'Retained Earnings']
    },
    'income_statement': {
        'revenue': ['Revenue', 'Net Sales', 'Total Income', 'Operating Revenue'],
        'expenses': ['Cost of Goods Sold', 'Operating Expenses', 'Finance Costs', 'Tax Expense'],
        'profits': ['Gross Profit', 'Operating Profit', 'EBIT', 'Net Profit', 'EPS']
    },
    'cash_flow': {
        'operating': ['Operating Cash Flow', 'Cash from Operations'],
        'investing': ['Investing Cash Flow', 'Capital Expenditure'],
        'financing': ['Financing Cash Flow', 'Dividends Paid']
    }
}

# Sample IND-AS Data
SAMPLE_BS_TEXT = """
Balance Sheet as at March 31, 2023
(All amounts in Rs. Lakhs unless otherwise stated)

ASSETS
Non-current Assets
Property, Plant and Equipment                    45,678.50
Intangible Assets                               12,345.00
Financial Assets                                 8,900.25
Total Non-current Assets                        66,923.75

Current Assets
Inventories                                     23,456.78
Trade Receivables                               34,567.89
Cash and Cash Equivalents                       15,678.90
Other Current Assets                             8,901.23
Total Current Assets                            82,604.80

TOTAL ASSETS                                   149,528.55

EQUITY AND LIABILITIES
Equity
Share Capital                                   10,000.00
Reserves and Surplus                            65,432.10
Total Equity                                    75,432.10

Non-current Liabilities
Long-term Borrowings                            25,678.90
Deferred Tax Liabilities                         5,432.10
Total Non-current Liabilities                   31,111.00

Current Liabilities
Short-term Borrowings                           15,432.10
Trade Payables                                  20,123.45
Other Current Liabilities                        7,429.90
Total Current Liabilities                       42,985.45

TOTAL EQUITY AND LIABILITIES                   149,528.55
"""

SAMPLE_PL_TEXT = """
Statement of Profit and Loss for the year ended March 31, 2023
(All amounts in Rs. Lakhs unless otherwise stated)

Revenue from Operations                         234,567.89
Other Income                                      3,456.78
Total Income                                    238,024.67

Expenses:
Cost of Materials Consumed                      123,456.78
Employee Benefits Expense                        34,567.89
Finance Costs                                     4,567.89
Depreciation and Amortization                     8,901.23
Other Expenses                                   23,456.78
Total Expenses                                  194,950.57

Profit Before Tax                                43,074.10
Tax Expense                                      10,768.53
Profit for the Period                            32,305.57

Earnings per Share (Basic)                          32.31
"""

# --- 3. Security and Validation Module ---
class SecurityValidator:
    """Centralized security validation and sanitization"""
    
    @staticmethod
    def sanitize_html_content(content: str) -> str:
        """Sanitize HTML content before processing"""
        allowed_tags = ['table', 'tr', 'td', 'th', 'tbody', 'thead', 'p', 'div', 'span']
        allowed_attributes = {'class': ['*'], 'id': ['*']}
        return bleach.clean(content, tags=allowed_tags, attributes=allowed_attributes, strip=True)
    
    @staticmethod
    def validate_file_upload(file: UploadedFile) -> bool:
        """Validate uploaded file for security"""
        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
        
        file_extension = file.name.split('.')[-1].lower()
        if file_extension not in ALLOWED_FILE_TYPES:
            raise ValueError(f"File type '{file_extension}' not allowed")
        
        if file_extension in ['html', 'htm']:
            content = file.read()
            file.seek(0)
            
            try:
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
            except UnicodeDecodeError:
                content = content.decode('latin-1')
            
            malicious_patterns = ['<script', 'javascript:', 'onerror=', 'onclick=', 'onload=']
            content_lower = content.lower()
            for pattern in malicious_patterns:
                if pattern in content_lower:
                    raise ValueError(f"File contains potentially malicious content: {pattern}")
        
        return True
    
    @staticmethod
    def sanitize_metric_name(name: str) -> str:
        """Sanitize metric names to prevent injection"""
        sanitized = re.sub(r'[;<>\"\'`]', '', name)
        return sanitized[:100]
    
    @staticmethod
    def validate_numeric_value(value: Any, min_val: float = -1e12, max_val: float = 1e12) -> float:
        """Validate and bound numeric values"""
        try:
            num_val = float(value)
            if np.isnan(num_val) or np.isinf(num_val):
                return 0.0
            return np.clip(num_val, min_val, max_val)
        except:
            return 0.0

# --- 4. Enhanced Data Structures ---
@dataclass
class ParsedFinancialData:
    """Structure for parsed financial data"""
    company_name: str
    statements: Dict[str, pd.DataFrame]
    year_columns: List[str]
    source_type: str
    parsing_notes: List[str] = field(default_factory=list)
    data_quality: Optional[Dict[str, Any]] = None
    detected_standard: str = "IND-AS"

@dataclass
class MappingResult:
    """Result of metric mapping operation"""
    mappings: Dict[str, str]
    confidence_scores: Dict[str, float]
    suggestions: Dict[str, List[Tuple[str, float]]]
    unmapped_metrics: List[str]

# --- 5. Enhanced Cache Management ---
class EmbeddingsCache:
    """Efficient caching for sentence embeddings"""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "embeddings_cache.pkl"
        self.memory_cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.memory_cache = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {e}")
                self.memory_cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.memory_cache, f)
        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {e}")
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        key = hashlib.md5(text.encode()).hexdigest()
        return self.memory_cache.get(key)
    
    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache"""
        key = hashlib.md5(text.encode()).hexdigest()
        self.memory_cache[key] = embedding
        if len(self.memory_cache) % 100 == 0:
            self._save_cache()
    
    def clear(self):
        """Clear cache"""
        self.memory_cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()

# --- 6. IND-AS Parser ---
class IndASParser:
    """Advanced parser for IND-AS format financial statements"""
    
    def __init__(self):
        self.number_converter = IndianNumberConverter()
        self.metric_patterns = self._build_metric_patterns()
    
    def _build_metric_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Build regex patterns for common financial metrics"""
        patterns = {
            'revenue': [
                re.compile(r'revenue\s+from\s+operations?', re.IGNORECASE),
                re.compile(r'net\s+sales?', re.IGNORECASE),
                re.compile(r'total\s+income', re.IGNORECASE)
            ],
            'net_profit': [
                re.compile(r'profit\s+for\s+the\s+(period|year)', re.IGNORECASE),
                re.compile(r'net\s+profit', re.IGNORECASE),
                re.compile(r'profit\s+after\s+tax', re.IGNORECASE)
            ],
            'total_assets': [
                re.compile(r'total\s+assets?', re.IGNORECASE)
            ],
            'total_equity': [
                re.compile(r'total\s+equity', re.IGNORECASE),
                re.compile(r'shareholders?\s+equity', re.IGNORECASE)
            ]
        }
        return patterns
    
    def parse_statements(self, text: str) -> ParsedFinancialData:
        """Parse IND-AS format financial statements from text"""
        lines = text.strip().split('\n')
        
        statement_type = self._detect_statement_type(text)
        years = self._extract_years(text)
        
        parsed_data = {}
        current_section = None
        parsing_notes = []
        
        for line in lines:
            line = line.strip()
            if not line or self._is_header_line(line):
                continue
            
            section = self._detect_section(line)
            if section:
                current_section = section
                continue
            
            metric_data = self._parse_metric_line(line, years)
            if metric_data:
                metric_name = metric_data['metric']
                values = metric_data['values']
                
                metric_name = SecurityValidator.sanitize_metric_name(metric_name)
                
                for year, value in values.items():
                    if year not in parsed_data:
                        parsed_data[year] = {}
                    parsed_data[year][metric_name] = value
                
                if metric_data.get('note'):
                    parsing_notes.append(metric_data['note'])
        
        df = pd.DataFrame(parsed_data)
        year_columns = sorted([col for col in df.columns if str(col).isdigit()], key=int)
        
        result = ParsedFinancialData(
            company_name=self._extract_company_name(text),
            statements={'parsed': df},
            year_columns=year_columns,
            source_type='text',
            parsing_notes=parsing_notes,
            detected_standard='IND-AS'
        )
        
        return result
    
    def _detect_statement_type(self, text: str) -> str:
        """Detect the type of financial statement"""
        text_lower = text.lower()
        if 'balance sheet' in text_lower:
            return 'balance_sheet'
        elif 'profit' in text_lower and 'loss' in text_lower:
            return 'income_statement'
        elif 'cash flow' in text_lower:
            return 'cash_flow'
        return 'unknown'
    
    def _extract_years(self, text: str) -> List[str]:
        """Extract years from statement text"""
        years = set()
        
        year_matches = re.findall(r'\b(20[1-2]\d|19[89]\d)\b', text)
        years.update(year_matches)
        
        date_patterns = [
            r'March\s+31,?\s+(20[1-2]\d)',
            r'December\s+31,?\s+(20[1-2]\d)',
            r'31\s+March\s+(20[1-2]\d)',
            r'31/03/(20[1-2]\d)'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            years.update(matches)
        
        return sorted(list(years))
    
    def _is_header_line(self, line: str) -> bool:
        """Check if line is a header or separator"""
        if re.match(r'^[-=]+$', line):
            return True
        if re.match(r'^\(.*\)$', line):
            return True
        if 'particulars' in line.lower() and len(line.split()) < 5:
            return True
        return False
    
    def _detect_section(self, line: str) -> Optional[str]:
        """Detect section headers in financial statements"""
        line_lower = line.lower()
        
        section_keywords = {
            'assets': ['assets', 'asset'],
            'liabilities': ['liabilities', 'liability'],
            'equity': ['equity', 'shareholders', 'capital'],
            'revenue': ['revenue', 'income'],
            'expenses': ['expenses', 'costs']
        }
        
        for section, keywords in section_keywords.items():
            if any(keyword in line_lower for keyword in keywords) and len(line.split()) < 5:
                return section
        
        return None
    
    def _parse_metric_line(self, line: str, years: List[str]) -> Optional[Dict[str, Any]]:
        """Parse a line containing metric and values"""
        parts = re.split(r'\s{2,}|\t+', line)
        
        if len(parts) < 2:
            return None
        
        metric_name = parts[0].strip()
        
        if len(metric_name) < 3 or metric_name.replace('.', '').replace(',', '').isdigit():
            return None
        
        values = {}
        note = None
        
        for i, part in enumerate(parts[1:]):
            parsed_value = self.number_converter.parse_indian_number(part)
            
            if parsed_value is not None:
                if i < len(years):
                    values[years[i]] = parsed_value
                else:
                    note = f"Extra value found for {metric_name}: {part}"
        
        if not values:
            return None
        
        return {
            'metric': metric_name,
            'values': values,
            'note': note
        }
    
    def _extract_company_name(self, text: str) -> str:
        """Extract company name from statement text"""
        patterns = [
            r'(?:Statement of|Balance Sheet of)\s+([A-Z][A-Za-z\s&]+(?:Limited|Ltd|Inc|Corp))',
            r'^([A-Z][A-Za-z\s&]+(?:Limited|Ltd|Inc|Corp))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return "Unknown Company"

# --- 7. Indian Number Converter ---
class IndianNumberConverter:
    """Convert Indian number formats to standard numeric values"""
    
    def __init__(self):
        self.multipliers = {
            'crore': 10000000,
            'crores': 10000000,
            'cr': 10000000,
            'lakh': 100000,
            'lakhs': 100000,
            'lacs': 100000,
            'lac': 100000,
            'l': 100000,
            'million': 1000000,
            'millions': 1000000,
            'mn': 1000000,
            'm': 1000000,
            'billion': 1000000000,
            'billions': 1000000000,
            'bn': 1000000000,
            'b': 1000000000,
            'thousand': 1000,
            'thousands': 1000,
            'k': 1000
        }
    
    def parse_indian_number(self, text: str) -> Optional[float]:
        """Parse Indian format numbers with lakhs/crores notation"""
        if not text or not isinstance(text, str):
            return None
        
        text = text.strip()
        text = re.sub(r'[₹$€£¥]', '', text)
        
        is_negative = False
        if text.startswith('(') and text.endswith(')'):
            is_negative = True
            text = text[1:-1]
        
        match = re.match(r'([-+]?\s*[\d,]+\.?\d*)\s*([a-zA-Z]+)?', text)
        
        if not match:
            return None
        
        number_str = match.group(1).replace(',', '').replace(' ', '')
        multiplier_str = match.group(2)
        
        try:
            number = float(number_str)
        except ValueError:
            return None
        
        if multiplier_str:
            multiplier_lower = multiplier_str.lower()
            if multiplier_lower in self.multipliers:
                number *= self.multipliers[multiplier_lower]
        
        if is_negative:
            number = -number
        
        return number
    
    def format_to_indian(self, number: float, use_lakhs: bool = True) -> str:
        """Format number to Indian notation"""
        if pd.isna(number):
            return "-"
        
        abs_num = abs(number)
        sign = '-' if number < 0 else ''
        
        if use_lakhs:
            if abs_num >= 10000000:
                return f"{sign}₹{abs_num/10000000:.2f} Cr"
            elif abs_num >= 100000:
                return f"{sign}₹{abs_num/100000:.2f} L"
            else:
                return f"{sign}₹{abs_num:,.0f}"
        else:
            if abs_num >= 1000000000:
                return f"{sign}${abs_num/1000000000:.2f}B"
            elif abs_num >= 1000000:
                return f"{sign}${abs_num/1000000:.2f}M"
            else:
                return f"{sign}${abs_num:,.0f}"

# --- 8. AI-Powered Financial Mapper (OPTIMIZED) ---

# Cache the model at module level
@st.cache_resource
def load_sentence_transformer_model():
    """Load and cache the sentence transformer model"""
    if st.session_state.get('lite_mode', False):
        return None
    
    try:
        with st.spinner("Loading AI model (one-time download)..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model
    except Exception as e:
        logger.error(f"Failed to initialize AI model: {e}")
        return None

# Pre-compute and cache standard embeddings
@st.cache_data
def compute_standard_embeddings_cached(_model):
    """Pre-compute embeddings for all standard financial metrics"""
    if _model is None:
        return None
    
    standard_metrics = {
        # Balance Sheet items
        'Total Assets': 'total assets sum of all assets',
        'Current Assets': 'current assets short term assets liquid assets',
        'Non-current Assets': 'non current assets long term assets fixed assets',
        'Cash and Cash Equivalents': 'cash bank balance liquid funds',
        'Inventory': 'inventory stock goods materials',
        'Trade Receivables': 'trade receivables accounts receivable debtors',
        'Property Plant and Equipment': 'property plant equipment PPE fixed assets',
        'Total Liabilities': 'total liabilities sum of all liabilities',
        'Current Liabilities': 'current liabilities short term liabilities',
        'Non-current Liabilities': 'non current liabilities long term liabilities',
        'Total Equity': 'total equity shareholders equity net worth',
        'Share Capital': 'share capital paid up capital equity shares',
        'Reserves and Surplus': 'reserves surplus retained earnings',
        
        # Income Statement items
        'Revenue': 'revenue sales turnover income from operations',
        'Cost of Goods Sold': 'cost of goods sold COGS cost of sales',
        'Gross Profit': 'gross profit gross margin',
        'Operating Expenses': 'operating expenses opex administrative expenses',
        'EBIT': 'EBIT earnings before interest tax operating profit',
        'Interest Expense': 'interest expense finance cost borrowing cost',
        'Net Profit': 'net profit net income profit after tax PAT',
        'EPS': 'earnings per share EPS',
        
        # Cash Flow items
        'Operating Cash Flow': 'operating cash flow cash from operations CFO',
        'Investing Cash Flow': 'investing cash flow capital expenditure capex',
        'Financing Cash Flow': 'financing cash flow debt repayment dividends'
    }
    
    embeddings = {}
    # Batch encode for efficiency
    texts = list(standard_metrics.values())
    keys = list(standard_metrics.keys())
    
    try:
        # Encode all at once - much faster than one by one
        all_embeddings = _model.encode(texts, batch_size=32, show_progress_bar=False)
        for key, embedding in zip(keys, all_embeddings):
            embeddings[key] = embedding
    except Exception as e:
        logger.error(f"Failed to compute standard embeddings: {e}")
        return None
    
    return embeddings

class IntelligentFinancialMapper:
    """AI-powered metric mapping using sentence transformers (OPTIMIZED)"""
    
    def __init__(self):
        self.model = None
        self.embeddings_cache = EmbeddingsCache()
        self.standard_embeddings = None
        self._initialize_model()
        self._batch_cache = {}  # Cache for batch operations
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        self.model = load_sentence_transformer_model()
        if self.model is None:
            st.warning("AI model not available. Using fuzzy matching instead.")
        else:
            # Use cached standard embeddings
            self.standard_embeddings = compute_standard_embeddings_cached(self.model)
    
    def _get_embeddings_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings for multiple texts efficiently"""
        if self.model is None:
            return {}
        
        # Check cache first
        uncached_texts = []
        cached_results = {}
        
        for text in texts:
            cached = self.embeddings_cache.get(text)
            if cached is not None:
                cached_results[text] = cached
            else:
                uncached_texts.append(text)
        
        # Compute uncached embeddings in batch
        if uncached_texts:
            try:
                # Batch encoding is much faster
                new_embeddings = self.model.encode(
                    uncached_texts, 
                    batch_size=min(32, len(uncached_texts)),
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                # Cache the results
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.embeddings_cache.set(text, embedding)
                    cached_results[text] = embedding
                    
            except Exception as e:
                logger.error(f"Failed to compute batch embeddings: {e}")
        
        return cached_results
    
    def map_metrics(self, source_metrics: List[str], target_metrics: Optional[List[str]] = None) -> MappingResult:
        """Map source metrics to standard financial metrics using AI (OPTIMIZED)"""
        
        # Quick fallback for no AI
        if self.model is None or self.standard_embeddings is None:
            return self._fuzzy_map_metrics(source_metrics, target_metrics)
        
        # Use standard metrics if no target specified
        if target_metrics is None:
            target_metrics = list(self.standard_embeddings.keys())
        
        # Limit processing for performance
        MAX_METRICS = 100  # Process max 100 metrics at once
        if len(source_metrics) > MAX_METRICS:
            st.warning(f"Processing first {MAX_METRICS} metrics for performance. Consider smaller batches.")
            source_metrics = source_metrics[:MAX_METRICS]
        
        # Prepare texts for batch processing
        source_texts = [metric.lower() for metric in source_metrics]
        
        # Get all embeddings in batch (much faster)
        with st.spinner("Computing metric similarities..."):
            source_embeddings_dict = self._get_embeddings_batch(source_texts)
        
        # Prepare target embeddings
        target_embeddings_list = []
        target_metrics_filtered = []
        
        for target_metric in target_metrics:
            if target_metric in self.standard_embeddings:
                target_embeddings_list.append(self.standard_embeddings[target_metric])
                target_metrics_filtered.append(target_metric)
        
        if not target_embeddings_list:
            return self._fuzzy_map_metrics(source_metrics, target_metrics)
        
        # Stack target embeddings for efficient computation
        target_embeddings_matrix = np.vstack(target_embeddings_list)
        
        mappings = {}
        confidence_scores = {}
        suggestions = {}
        unmapped = []
        
        # Process each source metric
        for i, source_metric in enumerate(source_metrics):
            source_text = source_texts[i]
            
            if source_text not in source_embeddings_dict:
                unmapped.append(source_metric)
                continue
            
            source_embedding = source_embeddings_dict[source_text]
            
            # Compute all similarities at once
            similarities = cosine_similarity(
                source_embedding.reshape(1, -1),
                target_embeddings_matrix
            )[0]
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:3]
            top_matches = [(target_metrics_filtered[idx], similarities[idx]) 
                          for idx in top_indices]
            
            best_match, best_score = top_matches[0]
            
            if best_score > 0.6:  # Threshold for accepting match
                mappings[source_metric] = best_match
                confidence_scores[source_metric] = float(best_score)
            else:
                unmapped.append(source_metric)
            
            suggestions[source_metric] = [(m, float(s)) for m, s in top_matches]
        
        return MappingResult(
            mappings=mappings,
            confidence_scores=confidence_scores,
            suggestions=suggestions,
            unmapped_metrics=unmapped
        )
    
    def _fuzzy_map_metrics(self, source_metrics: List[str], target_metrics: Optional[List[str]] = None) -> MappingResult:
        """Fallback fuzzy matching when AI is not available (OPTIMIZED)"""
        if target_metrics is None:
            target_metrics = []
            for category in METRIC_CATEGORIES.values():
                for subcategory in category.values():
                    target_metrics.extend(subcategory)
        
        # Limit for performance
        MAX_COMPARISONS = 50
        if len(source_metrics) > MAX_COMPARISONS:
            source_metrics = source_metrics[:MAX_COMPARISONS]
        
        mappings = {}
        confidence_scores = {}
        suggestions = {}
        unmapped = []
        
        # Pre-compute lowercase versions
        target_lower = [t.lower() for t in target_metrics]
        
        for source_metric in source_metrics:
            source_lower = source_metric.lower()
            
            # Quick exact match check first
            if source_lower in target_lower:
                idx = target_lower.index(source_lower)
                mappings[source_metric] = target_metrics[idx]
                confidence_scores[source_metric] = 1.0
                suggestions[source_metric] = [(target_metrics[idx], 1.0)]
                continue
            
            # Fuzzy matching for non-exact matches
            scores = []
            for i, target_metric in enumerate(target_metrics[:30]):  # Limit comparisons
                score = fuzz.token_sort_ratio(source_lower, target_lower[i])
                scores.append((target_metric, score / 100.0))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            
            if scores and scores[0][1] > 0.7:
                mappings[source_metric] = scores[0][0]
                confidence_scores[source_metric] = scores[0][1]
            else:
                unmapped.append(source_metric)
            
            suggestions[source_metric] = scores[:3]
        
        return MappingResult(
            mappings=mappings,
            confidence_scores=confidence_scores,
            suggestions=suggestions,
            unmapped_metrics=unmapped
        )

# --- 9. Configuration Management ---
class ConfigurationManager:
    """Manage mapping configurations and settings"""
    
    def __init__(self):
        self.config_dir = CACHE_DIR / "configs"
        self.config_dir.mkdir(exist_ok=True)
    
    def save_configuration(self, name: str, config: Dict[str, Any]) -> str:
        """Save configuration to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"config_{name}_{timestamp}.json"
        filepath = self.config_dir / filename
        
        config_data = {
            'name': name,
            'timestamp': timestamp,
            'version': APP_VERSION,
            'config': config
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return str(filepath)
    
    def load_configuration(self, filepath: str) -> Dict[str, Any]:
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        if config_data.get('version') != APP_VERSION:
            logger.warning(f"Configuration version mismatch: {config_data.get('version')} vs {APP_VERSION}")
        
        return config_data['config']
    
    def list_configurations(self) -> List[Dict[str, Any]]:
        """List all saved configurations"""
        configs = []
        
        for filepath in self.config_dir.glob("config_*.json"):
            try:
                with open(filepath, 'r') as f:
                    config_data = json.load(f)
                
                configs.append({
                    'name': config_data['name'],
                    'timestamp': config_data['timestamp'],
                    'filepath': str(filepath)
                })
            except Exception as e:
                logger.error(f"Failed to read config {filepath}: {e}")
        
        configs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return configs

# --- 10. Enhanced Wrapper Classes (FIXED) ---

class EnhancedChartGenerator(CoreChartGenerator):
    """Enhanced chart generator with Indian formatting (FIXED)"""
    
    def __init__(self):
        super().__init__()
        self.indian_converter = None
    
    def set_indian_converter(self, converter):
        """Set Indian number converter for formatting"""
        self.indian_converter = converter
    
    def create_chart_with_indian_format(self, df, metrics, title, chart_type="line", 
                                      use_indian_format=True, **kwargs):
        """Create charts with Indian number formatting (FIXED)"""
        try:
            # Filter valid metrics that exist in the dataframe
            valid_metrics = [m for m in metrics if m in df.index]
            if not valid_metrics:
                st.warning("No valid metrics found in the data")
                return None
            
            # Create the figure based on chart type
            fig = None
            
            # Try to use parent class methods if available
            if chart_type == "line":
                # Check for various possible method names in parent class
                if hasattr(super(), 'create_line_chart'):
                    fig = super().create_line_chart(df, valid_metrics, title, **kwargs)
                elif hasattr(super(), 'create_trend_chart'):
                    fig = super().create_trend_chart(df, valid_metrics, title, **kwargs)
                elif hasattr(super(), 'generate_line_chart'):
                    fig = super().generate_line_chart(df, valid_metrics, title, **kwargs)
                else:
                    # Use direct plotly creation as fallback
                    fig = self._create_line_chart_direct(df, valid_metrics, title)
                    
            elif chart_type == "bar":
                # Check for various possible method names in parent class
                if hasattr(super(), 'create_bar_chart'):
                    fig = super().create_bar_chart(df, valid_metrics, title, **kwargs)
                elif hasattr(super(), 'create_comparison_chart'):
                    fig = super().create_comparison_chart(df, valid_metrics, title, **kwargs)
                elif hasattr(super(), 'generate_bar_chart'):
                    fig = super().generate_bar_chart(df, valid_metrics, title, **kwargs)
                else:
                    # Use direct plotly creation as fallback
                    fig = self._create_bar_chart_direct(df, valid_metrics, title)
            else:
                # For any other chart type, create a line chart
                fig = self._create_line_chart_direct(df, valid_metrics, title)
            
            # Apply Indian formatting if requested and figure was created
            if fig and use_indian_format and self.indian_converter:
                self._apply_indian_formatting(fig)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            # Return a basic chart as fallback
            return self._create_error_chart(str(e))
    
    def _create_line_chart_direct(self, df, metrics, title):
        """Create line chart using plotly directly"""
        try:
            fig = go.Figure()
            
            # Get numeric columns only
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_cols:
                return self._create_error_chart("No numeric columns found")
            
            for metric in metrics:
                if metric in df.index:
                    y_values = df.loc[metric, numeric_cols].values
                    x_values = numeric_cols
                    
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode='lines+markers',
                        name=metric,
                        line=dict(width=2),
                        marker=dict(size=8),
                        connectgaps=True
                    ))
            
            fig.update_layout(
                title=dict(text=title, font=dict(size=16)),
                xaxis_title="Year",
                yaxis_title="Value",
                hovermode='x unified',
                template='plotly_white',
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error in _create_line_chart_direct: {e}")
            return self._create_error_chart(str(e))
    
    def _create_bar_chart_direct(self, df, metrics, title):
        """Create bar chart using plotly directly"""
        try:
            fig = go.Figure()
            
            # Get numeric columns only
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_cols:
                return self._create_error_chart("No numeric columns found")
            
            for metric in metrics:
                if metric in df.index:
                    y_values = df.loc[metric, numeric_cols].values
                    x_values = numeric_cols
                    
                    fig.add_trace(go.Bar(
                        x=x_values,
                        y=y_values,
                        name=metric,
                        text=[f'{v:,.0f}' if pd.notna(v) else '' for v in y_values],
                        textposition='auto'
                    ))
            
            fig.update_layout(
                title=dict(text=title, font=dict(size=16)),
                xaxis_title="Year",
                yaxis_title="Value",
                barmode='group',
                template='plotly_white',
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error in _create_bar_chart_direct: {e}")
            return self._create_error_chart(str(e))
    
    def _create_error_chart(self, error_message):
        """Create an error placeholder chart"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart could not be created: {error_message}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            height=300,
            template='plotly_white'
        )
        return fig
    
    def _apply_indian_formatting(self, fig):
        """Apply Indian number formatting to plotly figure"""
        if not self.indian_converter:
            return
        
        try:
            # Update hover template for all traces
            for trace in fig.data:
                if hasattr(trace, 'y') and trace.y is not None:
                    # Create custom hover text
                    hover_texts = []
                    for i, val in enumerate(trace.y):
                        if pd.notna(val) and isinstance(val, (int, float)):
                            formatted = self.indian_converter.format_to_indian(float(val))
                            hover_texts.append(formatted)
                        else:
                            hover_texts.append("N/A")
                    
                    # Update trace with custom hover
                    trace.customdata = hover_texts
                    trace.hovertemplate = '%{x}<br>%{customdata}<br><b>%{fullData.name}</b><extra></extra>'
            
            # Update y-axis tick format
            fig.update_yaxis(
                tickformat=',.0f',
                tickprefix='₹'
            )
            
        except Exception as e:
            logger.warning(f"Could not apply Indian formatting: {e}")

class EnhancedFinancialRatioCalculator(CoreRatioCalculator):
    """Enhanced ratio calculator with IND-AS specific ratios"""
    
    def __init__(self):
        super().__init__()
        self.indian_converter = None
    
    def calculate_indas_specific_ratios(self, df):
        """Calculate IND-AS specific ratios"""
        ratios = self.calculate_all_ratios(df)
        
        indas_ratios = pd.DataFrame(index=df.columns)
        
        if 'CSR Expense' in df.index and 'Net Profit' in df.index:
            indas_ratios['CSR Ratio %'] = self.safe_divide(
                df.loc['CSR Expense'], 
                df.loc['Net Profit'], 
                True
            )
        
        if 'Related Party Transactions' in df.index and 'Revenue' in df.index:
            indas_ratios['Related Party Ratio %'] = self.safe_divide(
                df.loc['Related Party Transactions'],
                df.loc['Revenue'],
                True
            )
        
        if not indas_ratios.empty:
            ratios['IND-AS Specific'] = indas_ratios.T
        
        return ratios

class EnhancedPenmanNissimAnalyzer(CorePenmanNissim):
    """Enhanced Penman-Nissim analyzer with IND-AS adjustments"""
    
    def __init__(self, df, mappings):
        super().__init__(df, mappings)
        self.indas_adjustments = {}
    
    def apply_indas_adjustments(self):
        """Apply IND-AS specific adjustments before analysis"""
        if 'Deferred Tax Assets' in self.df.index and 'Deferred Tax Liabilities' in self.df.index:
            net_deferred_tax = self.df.loc['Deferred Tax Assets'] - self.df.loc['Deferred Tax Liabilities']
            self.indas_adjustments['Net Deferred Tax'] = net_deferred_tax
        
        if 'Lease Liabilities' in self.df.index:
            self.indas_adjustments['Lease Adjustment'] = self.df.loc['Lease Liabilities']
        
        return self.indas_adjustments
    
    def calculate_with_indas(self):
        """Calculate Penman-Nissim with IND-AS adjustments"""
        self.apply_indas_adjustments()
        results = self.calculate_all()
        
        if 'indas_metrics' not in results:
            results['indas_metrics'] = pd.DataFrame(self.indas_adjustments)
        
        return results

class IntegratedIndustryBenchmarks(CoreIndustryBenchmarks):
    """Integrated industry benchmarks with Indian market data"""
    
    INDIAN_BENCHMARKS = {
        'Indian IT Services': {
            'RNOA': {'mean': 25.0, 'std': 7.0, 'quartiles': [18.0, 25.0, 32.0]},
            'OPM': {'mean': 24.0, 'std': 6.0, 'quartiles': [18.0, 24.0, 30.0]},
            'NOAT': {'mean': 1.8, 'std': 0.5, 'quartiles': [1.3, 1.8, 2.3]},
            'Employee Cost Ratio': {'mean': 55.0, 'std': 8.0, 'quartiles': [47.0, 55.0, 63.0]},
            'DSO': {'mean': 60, 'std': 15, 'quartiles': [45, 60, 75]},
            'Beta': 1.1,
            'Cost_of_Equity': 0.14
        },
        'Indian Pharma': {
            'RNOA': {'mean': 18.0, 'std': 6.0, 'quartiles': [12.0, 18.0, 24.0]},
            'OPM': {'mean': 20.0, 'std': 7.0, 'quartiles': [13.0, 20.0, 27.0]},
            'R&D Ratio': {'mean': 8.0, 'std': 3.0, 'quartiles': [5.0, 8.0, 11.0]},
            'Export Revenue Ratio': {'mean': 50.0, 'std': 20.0, 'quartiles': [30.0, 50.0, 70.0]},
            'Beta': 0.85,
            'Cost_of_Equity': 0.12
        },
        'Indian FMCG': {
            'RNOA': {'mean': 35.0, 'std': 10.0, 'quartiles': [25.0, 35.0, 45.0]},
            'OPM': {'mean': 15.0, 'std': 5.0, 'quartiles': [10.0, 15.0, 20.0]},
            'NOAT': {'mean': 3.5, 'std': 1.0, 'quartiles': [2.5, 3.5, 4.5]},
            'Distribution Cost Ratio': {'mean': 6.0, 'std': 2.0, 'quartiles': [4.0, 6.0, 8.0]},
            'Working Capital Cycle': {'mean': 20, 'std': 10, 'quartiles': [10, 20, 30]},
            'Beta': 0.75,
            'Cost_of_Equity': 0.11
        }
    }
    
    def __init__(self):
        super().__init__()
        self.BENCHMARKS.update(self.INDIAN_BENCHMARKS)
    
    def get_indian_peer_comparison(self, company_metrics, industry):
        """Get comparison with Indian peers"""
        if industry not in self.INDIAN_BENCHMARKS:
            return None
        
        comparison = {}
        benchmarks = self.INDIAN_BENCHMARKS[industry]
        
        for metric, value in company_metrics.items():
            if metric in benchmarks and not np.isnan(value):
                benchmark_data = benchmarks[metric]
                if isinstance(benchmark_data, dict) and 'mean' in benchmark_data:
                    percentile = self.get_percentile_rank(value, benchmark_data)
                    comparison[metric] = {
                        'value': value,
                        'industry_mean': benchmark_data['mean'],
                        'percentile': percentile,
                        'quartiles': benchmark_data.get('quartiles', [])
                    }
        
        return comparison

# --- 11. File Processing Functions ---

def parse_html_xls_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    """Parse HTML/XLS files from Streamlit upload"""
    try:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB.")
        
        file_content = uploaded_file.getvalue()
        return parse_html_content(file_content, uploaded_file.name)
    except Exception as e:
        logger.error(f"Failed to parse {uploaded_file.name}: {e}")
        return None

def parse_csv_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    """Parse CSV files from Streamlit upload"""
    try:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB.")
        
        file_content = uploaded_file.getvalue()
        return parse_csv_content(file_content, uploaded_file.name)
    except Exception as e:
        logger.error(f"Failed to parse {uploaded_file.name}: {e}")
        return None

def parse_single_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    """Parse a single uploaded file"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            parsed_data = parse_csv_file(uploaded_file)
        elif file_extension in ['html', 'htm', 'xls', 'xlsx']:
            parsed_data = parse_html_xls_file(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
        
        if parsed_data is None:
            st.warning(f"Could not parse '{uploaded_file.name}'.")
            return None
        
        df = parsed_data["statement"]
        
        year_cols_map = {}
        for col in df.columns:
            match = YEAR_REGEX.search(str(col))
            if match:
                year = match.group(0).replace('FY', '')
                year_cols_map[col] = year
            else:
                try:
                    year = int(col)
                    if 1980 <= year <= 2050:
                        year_cols_map[col] = str(year)
                except:
                    pass
        
        df = df.rename(columns=year_cols_map)
        valid_years = sorted([y for y in df.columns if str(y).isdigit()], key=int)
        
        if not valid_years:
            st.warning(f"No valid year columns found in '{uploaded_file.name}'.")
            return None
        
        df_proc = CoreDataProcessor.clean_numeric_data(df[valid_years].copy())
        df_proc = df_proc.dropna(how='all')
        
        parsed_data["statement"] = df_proc
        parsed_data["year_columns"] = valid_years
        
        return parsed_data
    except Exception as e:
        logger.error(f"Error parsing file {uploaded_file.name}: {e}")
        st.error(f"Error parsing '{uploaded_file.name}': {str(e)}")
        return None

@st.cache_data(show_spinner="Processing and merging files...")
def process_and_merge_files(_uploaded_files: List[UploadedFile]) -> Optional[Dict[str, Any]]:
    """Process and merge multiple uploaded files"""
    if not _uploaded_files:
        return None
    
    parsed_data_list = []
    
    with st.spinner("Parsing and merging files..."):
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(_uploaded_files):
            progress_bar.progress((idx + 1) / len(_uploaded_files))
            
            parsed = parse_single_file(file)
            if parsed:
                parsed_data_list.append(parsed)
        
        progress_bar.empty()
    
    if not parsed_data_list:
        st.error("None of the uploaded files could be parsed successfully.")
        return None
    
    return process_and_merge_dataframes(parsed_data_list)

# --- 12. Page Configuration and Styling ---
st.set_page_config(
    page_title="Elite Financial Analytics Platform",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1f77b4, #17a2b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .ai-badge {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .quality-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    .quality-high {
        background: #d4edda;
        color: #155724;
    }
    
    .quality-medium {
        background: #fff3cd;
        color: #856404;
    }
    
    .quality-low {
        background: #f8d7da;
        color: #721c24;
    }
    
    .indas-note {
        background: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 13. Main Application Class ---
class EnhancedFinancialAnalyticsPlatform:
    """Main application class with IND-AS and AI support"""
    
    def __init__(self):
        self._initialize_state()
        self._initialize_components()
    
    def _initialize_state(self):
        """Initialize session state variables"""
        defaults = {
            "analysis_data": None,
            "input_mode": "file_upload",
            "metric_mappings": {},
            "ai_mappings": {},
            "pn_results": None,
            "pn_mappings": {},
            "selected_industry": "Technology",
            "use_ai_mapping": True,
            "number_format": "indian",
            "current_config": None,
            "lite_mode": True,  # Default to True for better performance
            "debug_mode": False
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def _initialize_components(self):
        """Initialize platform components using enhanced classes"""
        self.security_validator = SecurityValidator()
        self.indas_parser = IndASParser()
        self.number_converter = IndianNumberConverter()
        self.config_manager = ConfigurationManager()
        
        # Lazy load AI mapper
        self.ai_mapper = None
        
        # Use enhanced components
        self.chart_generator = EnhancedChartGenerator()
        self.chart_generator.set_indian_converter(self.number_converter)
        
        self.ratio_calculator = EnhancedFinancialRatioCalculator()
        self.ratio_calculator.indian_converter = self.number_converter
        
        self.pn_analyzer = EnhancedPenmanNissimAnalyzer
        self.industry_benchmarks = IntegratedIndustryBenchmarks()
        self.data_processor = CoreDataProcessor
    
    def should_load_ai(self) -> bool:
        """Check if AI components should be loaded"""
        if st.session_state.get('lite_mode', False):
            return False
        if not st.session_state.get('use_ai_mapping', True):
            return False
        return True
    
    def get_ai_mapper(self):
        """Get AI mapper with lazy initialization"""
        if self.ai_mapper is None and self.should_load_ai():
            with st.spinner("Initializing AI components..."):
                self.ai_mapper = IntelligentFinancialMapper()
        return self.ai_mapper
    
    def run(self):
        """Main application entry point"""
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
    
    def _render_header(self):
        """Render application header"""
        st.markdown("<div class='main-header'>💹 Elite Financial Analytics Platform</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Advanced IND-AS Compliant Analysis with AI-Powered Intelligence</div>", unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render enhanced sidebar with all options"""
        st.sidebar.title("📊 Configuration")
        
        # Component status
        if CORE_COMPONENTS_AVAILABLE:
            st.sidebar.success("✅ All components loaded")
        else:
            st.sidebar.error("❌ Limited functionality")
        
        # Input mode selection
        st.sidebar.markdown("### 📥 Data Input")
        input_mode = st.sidebar.radio(
            "Choose input method:",
            ["file_upload", "text_paste", "sample_data"],
            format_func=lambda x: {
                "file_upload": "📁 Upload Files",
                "text_paste": "📝 Paste Text",
                "sample_data": "🔍 Sample Data"
            }[x]
        )
        st.session_state.input_mode = input_mode
        
        # Input interface
        st.sidebar.divider()
        if input_mode == "file_upload":
            self._render_file_upload()
        elif input_mode == "text_paste":
            st.sidebar.info("Paste financial statements in the main area")
        else:
            if st.sidebar.button("📥 Load IND-AS Sample", type="primary"):
                self._load_sample_data()
        
        # Settings
        st.sidebar.divider()
        st.sidebar.markdown("### ⚙️ Settings")
        
        # Performance options
        st.sidebar.markdown("#### 🚀 Performance")
        st.session_state.lite_mode = st.sidebar.checkbox(
            "Lite Mode (Faster)", 
            st.session_state.lite_mode,
            help="Disable AI features for faster performance"
        )
        
        if not st.session_state.lite_mode:
            st.session_state.use_ai_mapping = st.sidebar.checkbox(
                "🤖 AI-Powered Mapping",
                st.session_state.use_ai_mapping
            )
        
        st.session_state.number_format = st.sidebar.radio(
            "Number Format:",
            ["indian", "international"],
            format_func=lambda x: "₹ Lakhs/Crores" if x == "indian" else "$ Millions"
        )
        
        # Industry selection
        industries = list(self.industry_benchmarks.BENCHMARKS.keys())
        st.session_state.selected_industry = st.sidebar.selectbox(
            "Industry:",
            industries,
            index=industries.index(st.session_state.selected_industry) if st.session_state.selected_industry in industries else 0
        )
        
        # Debug options
        st.sidebar.markdown("#### 🛠️ Advanced")
        st.session_state.debug_mode = st.sidebar.checkbox(
            "Debug Mode",
            st.session_state.debug_mode
        )
        
        # Reset
        if st.sidebar.button("🔄 Reset All", type="secondary"):
            st.session_state.clear()
            st.rerun()
        
        # Debug info
        if st.session_state.debug_mode:
            self._debug_data_state()
    
    def _debug_data_state(self):
        """Debug helper to check data state"""
        st.sidebar.markdown("### 🐛 Debug Info")
        
        if st.session_state.analysis_data:
            data = st.session_state.analysis_data
            st.sidebar.write(f"Company: {data.company_name}")
            st.sidebar.write(f"Source: {data.source_type}")
            st.sidebar.write(f"Years: {data.year_columns}")
            st.sidebar.write(f"Statements keys: {list(data.statements.keys())}")
            
            for key, df in data.statements.items():
                if isinstance(df, pd.DataFrame):
                    st.sidebar.write(f"{key} shape: {df.shape}")
        else:
            st.sidebar.write("No data loaded")
    
    def _render_file_upload(self):
        """Render file upload interface (IMPROVED)"""
        files = st.sidebar.file_uploader(
            "Upload files",
            type=ALLOWED_FILE_TYPES,
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if files:
            st.sidebar.success(f"✅ {len(files)} file(s) uploaded")
            
            # Show file details
            for file in files:
                st.sidebar.text(f"📄 {file.name} ({file.size/1024:.1f}KB)")
            
            # Process button - make it prominent
            st.sidebar.markdown("---")
            if st.sidebar.button("📊 Process Files", type="primary", use_container_width=True):
                self._process_files(files)
        else:
            st.sidebar.info("Please upload financial statement files")
    
    def _render_main_content(self):
        """Render main content area"""
        if st.session_state.analysis_data is None:
            if st.session_state.input_mode == "text_paste":
                self._render_text_input()
            else:
                self._render_welcome()
        else:
            self._render_analysis()
    
    def _render_welcome(self):
        """Render welcome screen"""
        st.info("👋 Welcome! Choose an input method from the sidebar to begin.")
        
        # Feature cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='feature-card'>
            <h3>📊 Core + Enhanced</h3>
            <p>Your original financial analysis components enhanced with IND-AS support and AI</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='feature-card'>
            <h3>🇮🇳 Indian Market Focus</h3>
            <p>Built-in Indian accounting standards, number formats, and market benchmarks</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='feature-card'>
            <h3>🤖 Intelligent Mapping</h3>
            <p>AI-powered metric mapping with confidence scores and suggestions</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_text_input(self):
        """Render text input interface"""
        st.header("📝 Paste Financial Statements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Balance Sheet")
            bs_text = st.text_area(
                "Paste Balance Sheet",
                height=400,
                placeholder=SAMPLE_BS_TEXT[:500] + "...",
                key="bs_input"
            )
        
        with col2:
            st.subheader("Income Statement")
            pl_text = st.text_area(
                "Paste P&L Statement",
                height=400,
                placeholder=SAMPLE_PL_TEXT[:500] + "...",
                key="pl_input"
            )
        
        if st.button("🚀 Process Statements", type="primary"):
            if bs_text or pl_text:
                self._process_text_input(bs_text, pl_text)
            else:
                st.error("Please paste at least one statement")
    
    def _render_analysis(self):
        """Render analysis interface (FIXED)"""
        data = st.session_state.analysis_data
        
        # Check for data in both possible keys
        df = None
        if 'parsed' in data.statements:
            df = data.statements['parsed']
        elif 'merged' in data.statements:
            df = data.statements['merged']
        else:
            # Try to get any available dataframe
            for key, value in data.statements.items():
                if isinstance(value, pd.DataFrame) and not value.empty:
                    df = value
                    break
        
        if df is None or df.empty:
            st.error("No processed data available")
            if st.session_state.debug_mode:
                st.write("Debug Information:")
                st.write(f"Available statement keys: {list(data.statements.keys())}")
                st.write(f"Year columns: {data.year_columns}")
            return
        
        # Summary
        st.subheader(f"📊 {data.company_name}")
        
        # Quality badges
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Years", len(data.year_columns))
        with col2:
            st.metric("Metrics", len(df))
        with col3:
            st.metric("Standard", data.detected_standard)
        with col4:
            if st.session_state.use_ai_mapping and not st.session_state.lite_mode:
                mapped = len(st.session_state.ai_mappings)
                st.metric("AI Mapped", f"{mapped}/{len(df)}")
        
        # Analysis tabs
        tab_list = ["📊 Visualizations", "📈 Financial Ratios", "🔍 Penman-Nissim", 
                    "🏭 Industry Comparison", "📄 Data Table"]
        
        tabs = st.tabs(tab_list)
        
        # Use tabs properly
        with tabs[0]:
            self._render_visualizations_tab(df)
        
        with tabs[1]:
            self._render_ratios_tab(df)
        
        with tabs[2]:
            self._render_penman_nissim_tab(df)
        
        with tabs[3]:
            self._render_industry_tab(df)
        
        with tabs[4]:
            self._render_data_table_tab(df)
    
    def _render_visualizations_tab(self, df):
        """Render visualizations using enhanced chart generator (FIXED)"""
        st.header("📊 Financial Visualizations")
        
        # Controls
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # Filter to show only numeric data
            numeric_metrics = []
            for metric in df.index:
                try:
                    # Check if the row has at least one numeric value
                    if any(pd.api.types.is_numeric_dtype(type(v)) for v in df.loc[metric].values):
                        numeric_metrics.append(metric)
                except:
                    pass
            
            if not numeric_metrics:
                st.error("No numeric metrics found in data")
                return
            
            default = numeric_metrics[:min(3, len(numeric_metrics))]
            selected = st.multiselect(
                "Select metrics:",
                numeric_metrics,
                default=default,
                key="viz_metrics"
            )
        
        with col2:
            chart_type = st.selectbox(
                "Chart type:",
                ["line", "bar"],
                key="chart_type_selector"
            )
        
        with col3:
            show_grid = st.checkbox("Show grid", value=True)
        
        # Create visualization
        if selected:
            try:
                # Create the chart
                fig = self.chart_generator.create_chart_with_indian_format(
                    df, 
                    selected, 
                    title="Financial Metrics Analysis",
                    chart_type=chart_type,
                    use_indian_format=(st.session_state.number_format == "indian")
                )
                
                if fig:
                    # Add grid if requested
                    if show_grid:
                        fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='LightGray')
                        fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='LightGray')
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not create chart. Please try different metrics.")
                    # Show data table as fallback
                    st.subheader("Data Table (Fallback)")
                    display_df = df.loc[selected] if selected else df
                    st.dataframe(display_df.T)
                    
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                if st.session_state.debug_mode:
                    import traceback
                    st.code(traceback.format_exc())
                
                # Show data table as fallback
                st.subheader("Data Table (Fallback)")
                display_df = df.loc[selected] if selected else df
                st.dataframe(display_df.T)
        
        # Statistics section
        if selected:
            with st.expander("📊 Statistical Summary", expanded=False):
                stats_data = []
                for metric in selected:
                    if metric in df.index:
                        series = pd.to_numeric(df.loc[metric], errors='coerce')
                        series = series.dropna()
                        
                        if len(series) > 0:
                            stats_data.append({
                                'Metric': metric,
                                'Mean': series.mean(),
                                'Std Dev': series.std(),
                                'Min': series.min(),
                                'Max': series.max(),
                                'Count': len(series),
                                'Growth %': ((series.iloc[-1] / series.iloc[0] - 1) * 100) if len(series) > 1 and series.iloc[0] != 0 else 0
                            })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df.style.format({
                        'Mean': '{:,.2f}',
                        'Std Dev': '{:,.2f}',
                        'Min': '{:,.2f}',
                        'Max': '{:,.2f}',
                        'Count': '{:,.0f}',
                        'Growth %': '{:.1f}%'
                    }))
    
    def _render_ratios_tab(self, df):
        """Render financial ratios using enhanced calculator"""
        st.header("📈 Financial Ratio Analysis")
        
        # Check mappings
        if not st.session_state.metric_mappings:
            st.warning("Please map metrics first")
            if st.button("🤖 Auto-map with AI"):
                self._perform_ai_mapping()
            return
        
        # Apply mappings
        mapped_df = df.rename(index=st.session_state.metric_mappings)
        
        # Calculate ratios
        with st.spinner("Calculating ratios..."):
            ratios = self.ratio_calculator.calculate_indas_specific_ratios(mapped_df)
        
        if not ratios:
            st.error("Unable to calculate ratios")
            return
        
        # Display ratios by category
        for category, ratio_df in ratios.items():
            if not ratio_df.empty:
                st.subheader(f"{category} Ratios")
                
                # Format based on number preference
                if st.session_state.number_format == "indian":
                    formatted_df = ratio_df.style.format("{:,.2f}", na_rep="-")
                else:
                    formatted_df = ratio_df.style.format("{:,.2f}", na_rep="-")
                
                st.dataframe(formatted_df, use_container_width=True)
                
                # Visualization
                if st.checkbox(f"Visualize {category}", key=f"viz_{category}"):
                    metrics_to_plot = st.multiselect(
                        f"Select {category} metrics:",
                        ratio_df.index.tolist(),
                        default=ratio_df.index[:2].tolist() if len(ratio_df.index) >= 2 else ratio_df.index.tolist(),
                        key=f"select_{category}"
                    )
                    
                    if metrics_to_plot:
                        fig = self.chart_generator.create_chart_with_indian_format(
                            ratio_df, metrics_to_plot,
                            title=f"{category} Ratios Trend",
                            chart_type="line",
                            use_indian_format=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_penman_nissim_tab(self, df):
        """Render Penman-Nissim analysis"""
        st.header("🔍 Penman-Nissim Analysis")
        
        # Check if mappings exist
        if not st.session_state.pn_mappings:
            st.info("Configure Penman-Nissim mappings to proceed")
            
            # Auto-mapping interface
            with st.expander("⚙️ Configure P-N Mappings", expanded=True):
                self._render_pn_mapping_interface(df)
            
            return
        
        # Run analysis button
        if st.button("🚀 Run Penman-Nissim Analysis", type="primary"):
            with st.spinner("Running advanced analysis..."):
                try:
                    analyzer = self.pn_analyzer(df, st.session_state.pn_mappings)
                    results = analyzer.calculate_with_indas()
                    st.session_state.pn_results = results
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    if st.session_state.debug_mode:
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
        
        # Display results
        if st.session_state.pn_results:
            results = st.session_state.pn_results
            
            if "error" in results:
                st.error(f"Analysis error: {results['error']}")
                return
            
            # Key metrics summary
            st.subheader("Key Results")
            if 'ratios' in results and not results['ratios'].empty:
                ratios = results['ratios']
                if len(ratios.columns) > 0:
                    latest_year = ratios.columns[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        rnoa = ratios.loc['Return on Net Operating Assets (RNOA) %', latest_year] if 'Return on Net Operating Assets (RNOA) %' in ratios.index else 0
                        st.metric("RNOA", f"{rnoa:.2f}%")
                    
                    with col2:
                        opm = ratios.loc['Operating Profit Margin (OPM) %', latest_year] if 'Operating Profit Margin (OPM) %' in ratios.index else 0
                        st.metric("OPM", f"{opm:.2f}%")
                    
                    with col3:
                        noat = ratios.loc['Net Operating Asset Turnover (NOAT)', latest_year] if 'Net Operating Asset Turnover (NOAT)' in ratios.index else 0
                        st.metric("NOAT", f"{noat:.2f}x")
                    
                    with col4:
                        flev = ratios.loc['Financial Leverage (FLEV)', latest_year] if 'Financial Leverage (FLEV)' in ratios.index else 0
                        st.metric("FLEV", f"{flev:.2f}x")
            
            # Detailed results
            with st.expander("📊 Detailed Results", expanded=False):
                if 'reformulated_bs' in results and not results['reformulated_bs'].empty:
                    st.subheader("Reformulated Balance Sheet")
                    st.dataframe(results['reformulated_bs'].style.format("{:,.2f}"))
                
                if 'reformulated_is' in results and not results['reformulated_is'].empty:
                    st.subheader("Reformulated Income Statement")
                    st.dataframe(results['reformulated_is'].style.format("{:,.2f}"))
                
                if 'ratios' in results and not results['ratios'].empty:
                    st.subheader("All Ratios")
                    st.dataframe(results['ratios'].style.format("{:,.2f}"))
    
    def _render_industry_tab(self, df):
        """Render industry comparison"""
        st.header("🏭 Industry Comparison")
        
        if not st.session_state.pn_results:
            st.warning("Run Penman-Nissim analysis first")
            return
        
        results = st.session_state.pn_results
        if 'ratios' in results and not results['ratios'].empty:
            ratios = results['ratios']
            if len(ratios.columns) > 0:
                latest_year = ratios.columns[-1]
                
                # Get latest metrics
                latest_metrics = {}
                ratio_mapping = {
                    'Return on Net Operating Assets (RNOA) %': 'RNOA',
                    'Operating Profit Margin (OPM) %': 'OPM',
                    'Net Operating Asset Turnover (NOAT)': 'NOAT',
                    'Net Borrowing Cost (NBC) %': 'NBC',
                    'Financial Leverage (FLEV)': 'FLEV'
                }
                
                for pn_name, bench_name in ratio_mapping.items():
                    if pn_name in ratios.index:
                        value = ratios.loc[pn_name, latest_year]
                        if not pd.isna(value):
                            latest_metrics[bench_name] = float(value)
                
                if latest_metrics:
                    comparison = self.industry_benchmarks.calculate_composite_score(
                        latest_metrics, 
                        st.session_state.selected_industry
                    )
                    
                    if "error" not in comparison:
                        score = comparison['composite_score']
                        interpretation = comparison['interpretation']
                        
                        # Display score
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if score >= 80:
                                color = "green"
                            elif score >= 60:
                                color = "blue"
                            elif score >= 40:
                                color = "orange"
                            else:
                                color = "red"
                            
                            st.markdown(
                                f"<div style='text-align: center; padding: 20px; background-color: #f0f0f0; border-radius: 10px;'>"
                                f"<h3 style='color: {color};'>Industry Score: {score:.1f}/100</h3>"
                                f"<p style='font-size: 18px;'>{interpretation} vs {st.session_state.selected_industry}</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        
                        # Detailed comparison
                        if 'metric_scores' in comparison:
                            st.subheader("Metric-wise Comparison")
                            
                            comp_data = []
                            for metric, percentile in comparison['metric_scores'].items():
                                if metric in latest_metrics:
                                    comp_data.append({
                                        'Metric': metric,
                                        'Company Value': f"{latest_metrics[metric]:.2f}",
                                        'Percentile': f"{percentile:.0f}%",
                                        'Performance': self._get_performance_label(percentile)
                                    })
                            
                            if comp_data:
                                comp_df = pd.DataFrame(comp_data)
                                st.dataframe(comp_df, use_container_width=True)
    
    def _render_data_table_tab(self, df):
        """Render data table"""
        st.header("📄 Financial Data")
        
        # Format options
        col1, col2 = st.columns(2)
        with col1:
            decimal_places = st.number_input("Decimal places:", 0, 4, 2)
        with col2:
            highlight_negative = st.checkbox("Highlight negative values", True)
        
        # Format dataframe
        if st.session_state.number_format == "indian":
            formatted_df = df.copy()
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    formatted_df[col] = df[col].apply(
                        lambda x: self.number_converter.format_to_indian(x, use_lakhs=True)
                    )
            st.dataframe(formatted_df, use_container_width=True)
        else:
            format_dict = {col: f"{{:,.{decimal_places}f}}" for col in df.columns}
            styled_df = df.style.format(format_dict, na_rep="-")
            
            if highlight_negative:
                styled_df = styled_df.applymap(
                    lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else ''
                )
            
            st.dataframe(styled_df, use_container_width=True)
        
        # Export options
        self._render_export_options(df)
    
    def _render_pn_mapping_interface(self, df):
        """Render Penman-Nissim mapping interface"""
        available = [''] + df.index.tolist()
        
        # Essential mappings
        st.markdown("### Essential Mappings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.pn_mappings['Total Assets'] = st.selectbox(
                "Total Assets", available,
                index=self._find_index(available, 'Total Assets')
            )
            st.session_state.pn_mappings['Total Equity'] = st.selectbox(
                "Total Equity", available,
                index=self._find_index(available, 'Total Equity')
            )
        
        with col2:
            st.session_state.pn_mappings['Total Liabilities'] = st.selectbox(
                "Total Liabilities", available,
                index=self._find_index(available, 'Total Liabilities')
            )
            st.session_state.pn_mappings['Revenue'] = st.selectbox(
                "Revenue", available,
                index=self._find_index(available, 'Revenue')
            )
        
        with col3:
            st.session_state.pn_mappings['Operating Income'] = st.selectbox(
                "Operating Income/EBIT", available,
                index=self._find_index(available, 'EBIT')
            )
            st.session_state.pn_mappings['Net Income'] = st.selectbox(
                "Net Income", available,
                index=self._find_index(available, 'Net Profit')
            )
        
        # Financial items
        st.markdown("### Financial Items")
        
        st.session_state.pn_mappings['Financial Assets'] = st.multiselect(
            "Financial Assets (Cash, Investments, etc.)",
            df.index.tolist(),
            default=self._find_financial_items(df.index, 'assets')
        )
        
        st.session_state.pn_mappings['Financial Liabilities'] = st.multiselect(
            "Financial Liabilities (Debt, Loans, etc.)",
            df.index.tolist(),
            default=self._find_financial_items(df.index, 'liabilities')
        )
        
        st.session_state.pn_mappings['Net Financial Expense'] = st.selectbox(
            "Net Financial Expense/Interest",
            available,
            index=self._find_index(available, 'Interest', 'Finance Cost')
        )
    
    # --- Helper Methods ---
    
    def _process_files(self, files):
        """Process uploaded files (FIXED)"""
        try:
            # Validate files
            for file in files:
                self.security_validator.validate_file_upload(file)
            
            # Process files
            data = process_and_merge_files(files)
            if data:
                # Fix: Ensure consistent data structure
                parsed_data = ParsedFinancialData(
                    company_name=data.get('company_name', 'Unknown'),
                    statements={'parsed': data['statement']},  # Changed from 'merged' to 'parsed'
                    year_columns=data.get('year_columns', []),
                    source_type='file',
                    data_quality=data.get('data_quality', {})
                )
                
                st.session_state.analysis_data = parsed_data
                
                # Auto-map if enabled
                if st.session_state.use_ai_mapping and not st.session_state.lite_mode:
                    self._perform_ai_mapping()
                
                st.success("Files processed successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            if st.session_state.debug_mode:
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
    
    def _process_text_input(self, bs_text, pl_text):
        """Process text input"""
        try:
            combined = f"{bs_text}\n\n{pl_text}"
            parsed_data = self.indas_parser.parse_statements(combined)
            
            # Clean data
            df = parsed_data.statements['parsed']
            df_cleaned = self.data_processor.clean_numeric_data(df)
            
            parsed_data.statements['parsed'] = df_cleaned
            st.session_state.analysis_data = parsed_data
            
            # Auto-map if enabled
            if st.session_state.use_ai_mapping and not st.session_state.lite_mode:
                self._perform_ai_mapping()
            
            st.success("Text processed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing text: {str(e)}")
    
    def _load_sample_data(self):
        """Load sample data"""
        try:
            combined = f"{SAMPLE_BS_TEXT}\n\n{SAMPLE_PL_TEXT}"
            parsed_data = self.indas_parser.parse_statements(combined)
            parsed_data.company_name = "Sample Indian Company Ltd."
            
            df = parsed_data.statements['parsed']
            df_cleaned = self.data_processor.clean_numeric_data(df)
            
            parsed_data.statements['parsed'] = df_cleaned
            st.session_state.analysis_data = parsed_data
            
            st.success("Sample data loaded!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error loading sample: {str(e)}")
    
    def _perform_ai_mapping(self):
        """Perform AI mapping with progress indication"""
        if st.session_state.analysis_data:
            # Get the correct dataframe
            df = None
            data = st.session_state.analysis_data
            if 'parsed' in data.statements:
                df = data.statements['parsed']
            elif 'merged' in data.statements:
                df = data.statements['merged']
            else:
                for key, value in data.statements.items():
                    if isinstance(value, pd.DataFrame) and not value.empty:
                        df = value
                        break
            
            if df is None or df.empty:
                st.error("No data available for mapping")
                return
            
            source_metrics = df.index.tolist()
            
            # Show warning for large datasets
            if len(source_metrics) > 50:
                st.warning(f"Found {len(source_metrics)} metrics. This may take a moment...")
            
            # Progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            try:
                progress_text.text("Initializing AI mapper...")
                progress_bar.progress(0.3)
                
                # Get AI mapper
                mapper = self.get_ai_mapper()
                if mapper is None:
                    progress_bar.empty()
                    progress_text.empty()
                    st.warning("AI mapping not available in Lite Mode")
                    return
                
                # Get target metrics
                target_metrics = []
                for category in METRIC_CATEGORIES.values():
                    for subcategory in category.values():
                        target_metrics.extend(subcategory)
                
                progress_text.text("Computing similarities...")
                progress_bar.progress(0.6)
                
                # Perform mapping
                result = mapper.map_metrics(source_metrics, target_metrics)
                
                progress_text.text("Finalizing mappings...")
                progress_bar.progress(0.9)
                
                st.session_state.ai_mappings = result.mappings
                st.session_state.metric_mappings = result.mappings
                
                # Clear progress
                progress_bar.empty()
                progress_text.empty()
                
                # Show summary
                st.success(f"✅ AI mapped {len(result.mappings)} out of {len(source_metrics)} metrics")
                
            except Exception as e:
                progress_bar.empty()
                progress_text.empty()
                st.error(f"AI mapping failed: {str(e)}")
                st.info("Using fallback fuzzy matching instead")
    
    def _render_export_options(self, df):
        """Render export options"""
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv()
            st.download_button(
                "📄 CSV",
                csv,
                "financial_data.csv",
                "text/csv"
            )
        
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Financial Data')
            
            st.download_button(
                "📊 Excel",
                buffer.getvalue(),
                "financial_data.xlsx",
                "application/vnd.ms-excel"
            )
        
        with col3:
            json_str = df.to_json(orient='split', indent=2)
            st.download_button(
                "🔧 JSON",
                json_str,
                "financial_data.json",
                "application/json"
            )
    
    def _find_index(self, options, *keywords):
        """Find index of matching option (FIXED to handle non-string values)"""
        for keyword in keywords:
            for i, option in enumerate(options):
                # Convert option to string to safely use .lower()
                option_str = str(option) if option else ""
                if keyword.lower() in option_str.lower():
                    return i
        return 0
    
    def _find_financial_items(self, metrics, item_type):
        """Find financial items in metrics (FIXED to handle non-string values)"""
        financial_keywords = {
            'assets': ['cash', 'bank', 'investment', 'securities', 'deposits'],
            'liabilities': ['debt', 'loan', 'borrowing', 'debenture', 'bonds']
        }
        
        keywords = financial_keywords.get(item_type, [])
        matches = []
        
        for metric in metrics:
            # Convert metric to string to safely use .lower()
            metric_str = str(metric) if metric else ""
            metric_lower = metric_str.lower()
            if any(keyword in metric_lower for keyword in keywords):
                matches.append(metric)
        
        return matches
    
    def _get_performance_label(self, percentile):
        """Get performance label based on percentile"""
        if percentile >= 80:
            return "Excellent"
        elif percentile >= 60:
            return "Good"
        elif percentile >= 40:
            return "Average"
        elif percentile >= 20:
            return "Below Average"
        else:
            return "Poor"

# --- 14. Application Entry Point ---

def main():
    """Main application entry point"""
    app = EnhancedFinancialAnalyticsPlatform()
    app.run()

if __name__ == "__main__":
    main()
