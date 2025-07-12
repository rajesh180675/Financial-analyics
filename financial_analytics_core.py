# financial_analytics_core.py
# Core financial analysis components without Streamlit dependencies

import io
import logging
import re
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fuzzywuzzy import fuzz
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Configuration
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_TYPES = ['html', 'htm', 'xls', 'xlsx', 'csv']
YEAR_REGEX = re.compile(r'\b(19[8-9]\d|20\d\d|FY\d{4})\b')

REQUIRED_METRICS = {
    'Profitability': ['Revenue', 'Gross Profit', 'EBIT', 'Net Profit', 'Total Equity', 'Total Assets', 'Current Liabilities'],
    'Liquidity': ['Current Assets', 'Current Liabilities', 'Inventory', 'Cash and Cash Equivalents'],
    'Leverage': ['Total Debt', 'Total Equity', 'Total Assets', 'EBIT', 'Interest Expense'],
    'DuPont': ['Net Profit', 'Revenue', 'Total Assets', 'Total Equity'],
    'Cash Flow': ['Operating Cash Flow', 'Capital Expenditure']
}

# Financial constants
RISK_FREE_RATE = 0.045
MARKET_RISK_PREMIUM = 0.065
DEFAULT_WACC = 0.10
TERMINAL_GROWTH_RATE = 0.025
TAX_RATE_BOUNDS = (0.15, 0.35)
EPS = 1e-10

# Data structures
@dataclass
class DataQualityMetrics:
    total_rows: int
    missing_values: int
    missing_percentage: float
    duplicate_rows: int
    quality_score: str = field(init=False)

    def __post_init__(self):
        if self.missing_percentage < 5:
            self.quality_score = "High"
        elif self.missing_percentage < 20:
            self.quality_score = "Medium"
        else:
            self.quality_score = "Low"

# All the core classes without Streamlit dependencies
class IndustryBenchmarks:
    """Comprehensive industry benchmarks based on academic research and market data"""
    
    BENCHMARKS = {
        'Technology': {
            'RNOA': {'mean': 18.5, 'std': 6.2, 'quartiles': [12.0, 18.5, 25.0]},
            'OPM': {'mean': 22.0, 'std': 8.5, 'quartiles': [15.0, 22.0, 30.0]},
            'NOAT': {'mean': 1.2, 'std': 0.4, 'quartiles': [0.8, 1.2, 1.6]},
            'NBC': {'mean': 3.5, 'std': 1.2, 'quartiles': [2.5, 3.5, 4.5]},
            'FLEV': {'mean': 0.3, 'std': 0.2, 'quartiles': [0.1, 0.3, 0.5]},
            'Beta': 1.25,
            'Cost_of_Equity': 0.125
        },
        'Retail': {
            'RNOA': {'mean': 14.0, 'std': 4.5, 'quartiles': [10.0, 14.0, 18.0]},
            'OPM': {'mean': 8.0, 'std': 3.0, 'quartiles': [5.0, 8.0, 11.0]},
            'NOAT': {'mean': 2.8, 'std': 0.8, 'quartiles': [2.0, 2.8, 3.6]},
            'NBC': {'mean': 4.0, 'std': 1.5, 'quartiles': [2.5, 4.0, 5.5]},
            'FLEV': {'mean': 0.5, 'std': 0.3, 'quartiles': [0.2, 0.5, 0.8]},
            'Beta': 1.1,
            'Cost_of_Equity': 0.115
        },
        'Manufacturing': {
            'RNOA': {'mean': 12.0, 'std': 3.8, 'quartiles': [8.0, 12.0, 16.0]},
            'OPM': {'mean': 10.0, 'std': 3.5, 'quartiles': [7.0, 10.0, 13.0]},
            'NOAT': {'mean': 1.5, 'std': 0.5, 'quartiles': [1.0, 1.5, 2.0]},
            'NBC': {'mean': 3.8, 'std': 1.3, 'quartiles': [2.5, 3.8, 5.0]},
            'FLEV': {'mean': 0.6, 'std': 0.3, 'quartiles': [0.3, 0.6, 0.9]},
            'Beta': 1.0,
            'Cost_of_Equity': 0.11
        },
        'Financial Services': {
            'RNOA': {'mean': 10.0, 'std': 3.0, 'quartiles': [7.0, 10.0, 13.0]},
            'OPM': {'mean': 35.0, 'std': 10.0, 'quartiles': [25.0, 35.0, 45.0]},
            'NOAT': {'mean': 0.15, 'std': 0.05, 'quartiles': [0.1, 0.15, 0.2]},
            'NBC': {'mean': 2.5, 'std': 1.0, 'quartiles': [1.5, 2.5, 3.5]},
            'FLEV': {'mean': 2.5, 'std': 1.0, 'quartiles': [1.5, 2.5, 3.5]},
            'Beta': 1.3,
            'Cost_of_Equity': 0.13
        },
        'Healthcare': {
            'RNOA': {'mean': 16.0, 'std': 5.0, 'quartiles': [11.0, 16.0, 21.0]},
            'OPM': {'mean': 15.0, 'std': 5.0, 'quartiles': [10.0, 15.0, 20.0]},
            'NOAT': {'mean': 1.3, 'std': 0.4, 'quartiles': [0.9, 1.3, 1.7]},
            'NBC': {'mean': 3.2, 'std': 1.1, 'quartiles': [2.0, 3.2, 4.3]},
            'FLEV': {'mean': 0.4, 'std': 0.2, 'quartiles': [0.2, 0.4, 0.6]},
            'Beta': 0.9,
            'Cost_of_Equity': 0.10
        }
    }
    
    @staticmethod
    def get_percentile_rank(value: float, benchmark_data: Dict) -> float:
        """Calculate percentile rank using normal distribution approximation"""
        mean = benchmark_data['mean']
        std = benchmark_data['std']
        if std == 0 or np.isnan(value) or np.isnan(mean) or np.isnan(std):
            return 50.0
        z_score = (value - mean) / std
        percentile = stats.norm.cdf(z_score) * 100
        return np.clip(percentile, 0, 100)
    
    @staticmethod
    def calculate_composite_score(metrics: Dict[str, float], industry: str) -> Dict[str, Any]:
        """Calculate comprehensive performance score vs industry"""
        if industry not in IndustryBenchmarks.BENCHMARKS:
            return {"error": "Industry not found"}
        
        benchmarks = IndustryBenchmarks.BENCHMARKS[industry]
        scores = {}
        weights = {'RNOA': 0.35, 'OPM': 0.25, 'NOAT': 0.20, 'NBC': -0.10, 'FLEV': -0.10}
        
        weighted_score = 0
        total_weight = 0
        for metric, weight in weights.items():
            if metric in metrics and metric in benchmarks and not np.isnan(metrics[metric]):
                percentile = IndustryBenchmarks.get_percentile_rank(
                    metrics[metric], benchmarks[metric]
                )
                if weight < 0:
                    percentile = 100 - percentile
                scores[metric] = percentile
                weighted_score += abs(weight) * percentile
                total_weight += abs(weight)
        
        final_score = weighted_score / total_weight if total_weight > 0 else 50
        
        return {
            'composite_score': final_score,
            'metric_scores': scores,
            'interpretation': IndustryBenchmarks._interpret_score(final_score)
        }
    
    @staticmethod
    def _interpret_score(score: float) -> str:
        if score >= 80:
            return "Elite Performer"
        elif score >= 60:
            return "Above Average"
        elif score >= 40:
            return "Average"
        elif score >= 20:
            return "Below Average"
        else:
            return "Underperformer"

class DataProcessor:
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data to numeric format"""
        try:
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]):
                    df[col] = df[col].astype(str).str.replace(r'[,\(\)₹$€£]|Rs\.', '', regex=True)
                    df[col] = df[col].str.replace(r'^\s*-\s*$', 'NaN', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            logger.error(f"Error in clean_numeric_data: {e}")
            return df

    @staticmethod
    def calculate_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
        """Calculate data quality metrics"""
        try:
            total = df.size
            if total == 0:
                return DataQualityMetrics(0, 0, 0.0, 0)
            missing = int(df.isnull().sum().sum())
            duplicate_rows = int(df.duplicated().sum())
            missing_pct = (missing / total) * 100 if total > 0 else 0.0
            return DataQualityMetrics(len(df), missing, missing_pct, duplicate_rows)
        except Exception as e:
            logger.error(f"Error calculating data quality: {e}")
            return DataQualityMetrics(0, 0, 0.0, 0)

    @staticmethod
    def normalize_to_100(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """Normalize metrics to base 100"""
        try:
            df_scaled = df.loc[metrics].copy()
            for metric in metrics:
                if metric in df_scaled.index:
                    series = df_scaled.loc[metric].dropna()
                    if not series.empty and abs(series.iloc[0]) > EPS:
                        df_scaled.loc[metric] = (df_scaled.loc[metric] / series.iloc[0]) * 100
                    else:
                        df_scaled.loc[metric] = np.nan
            return df_scaled
        except Exception as e:
            logger.error(f"Error in normalize_to_100: {e}")
            return pd.DataFrame()

    @staticmethod
    def detect_outliers(df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect outliers using IQR method"""
        outliers = {}
        try:
            numeric_df = df.select_dtypes(include=np.number)
            for col in numeric_df.columns:
                data = numeric_df[col].dropna()
                if len(data) > 3:
                    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > EPS:
                        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                        outlier_mask = (numeric_df[col] < lower) | (numeric_df[col] > upper)
                        outlier_indices = numeric_df[outlier_mask].index.tolist()
                        if outlier_indices:
                            outliers[col] = outlier_indices
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
        return outliers

# Include all other core classes (ChartGenerator, FinancialRatioCalculator, PenmanNissimAnalyzer)
# Copy the class definitions exactly as they are, just remove any st.* references

class ChartGenerator:
    # Copy the entire ChartGenerator class here without modifications
    # (Already no Streamlit dependencies in this class)
    pass

class FinancialRatioCalculator:
    # Copy the entire FinancialRatioCalculator class here
    pass

class PenmanNissimAnalyzer:
    # Copy the entire PenmanNissimAnalyzer class here
    pass

# File parsing functions that work with raw data instead of UploadedFile
def parse_html_content(file_content: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Parse HTML/XLS content"""
    try:
        # Similar to parse_html_xls_file but works with bytes instead of UploadedFile
        dfs = pd.read_html(io.BytesIO(file_content), header=[0, 1])
        if dfs:
            df = dfs[0]
            company_name = "Unknown Company"
            if hasattr(df.columns, 'levels') and len(df.columns.levels) > 0:
                first_col = str(df.columns[0][0]) if isinstance(df.columns[0], tuple) else str(df.columns[0])
                if ">>" in first_col:
                    company_name = first_col.split(">>")[2].split("(")[0].strip()
            
            df.columns = [str(c[1]) if isinstance(c, tuple) else str(c) for c in df.columns]
            
            # Rest of processing logic...
            return {
                "statement": df,
                "company_name": company_name,
                "source": filename
            }
    except Exception as e:
        logger.error(f"Failed to parse HTML content: {e}")
        return None

def parse_csv_content(file_content: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Parse CSV content"""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        # Rest of processing logic...
        return {
            "statement": df,
            "company_name": "From CSV",
            "source": filename
        }
    except Exception as e:
        logger.error(f"Failed to parse CSV content: {e}")
        return None

# Add a function to process and merge dataframes without Streamlit dependencies
def process_and_merge_dataframes(parsed_data_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Process and merge multiple parsed data dictionaries"""
    if not parsed_data_list:
        return None
    
    all_dfs = []
    company_name = "Multiple Sources"
    sources = {}
    
    for parsed in parsed_data_list:
        if parsed:
            df = parsed["statement"]
            source = parsed["source"]
            
            for metric in df.index:
                if metric not in sources:
                    sources[metric] = source
            
            all_dfs.append(df)
            
            if parsed["company_name"] not in ["Unknown Company", "From CSV"]:
                company_name = parsed["company_name"]
    
    if not all_dfs:
        return None
    
    if len(all_dfs) == 1:
        merged_df = all_dfs[0]
    else:
        merged_df = pd.concat(all_dfs, axis=0, join='outer')
        merged_df = merged_df.groupby(level=0).first()
    
    # Process year columns
    all_years = set()
    for df in all_dfs:
        all_years.update(df.columns)
    
    year_columns = sorted([y for y in all_years if str(y).isdigit()], key=int)
    merged_df = merged_df.reindex(columns=year_columns, fill_value=np.nan)
    
    data_quality = asdict(DataProcessor.calculate_data_quality(merged_df))
    outliers = DataProcessor.detect_outliers(merged_df)
    
    return {
        "statement": merged_df,
        "company_name": company_name,
        "data_quality": data_quality,
        "outliers": outliers,
        "year_columns": year_columns,
        "sources": sources
    }
