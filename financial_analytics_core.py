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

# Core Classes
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

class ChartGenerator:
    """Generate various financial charts and visualizations"""
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame, metrics: List[str], title: str, 
                         y_title: str = "Value", normalize: bool = False) -> go.Figure:
        """Create line chart for selected metrics"""
        fig = go.Figure()
        
        for metric in metrics:
            if metric in df.index:
                y_values = df.loc[metric]
                if normalize and len(y_values) > 0 and y_values.iloc[0] != 0:
                    y_values = (y_values / y_values.iloc[0]) * 100
                
                fig.add_trace(go.Scatter(
                    x=df.columns,
                    y=y_values,
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=2),
                    marker=dict(size=8),
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title=y_title,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        return fig

    @staticmethod
    def create_waterfall_chart(data: Dict[str, float], title: str) -> go.Figure:
        """Create waterfall chart"""
        labels = list(data.keys())
        values = list(data.values())
        
        fig = go.Figure(go.Waterfall(
            name="", orientation="v",
            measure=["relative"] * (len(labels) - 1) + ["total"],
            x=labels,
            y=values,
            text=[f"{v:,.0f}" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(title=title, showlegend=False, height=400)
        return fig

    @staticmethod
    def create_ratio_comparison_chart(ratios: pd.DataFrame, selected_ratios: List[str], 
                                    company_name: str) -> go.Figure:
        """Create comparison chart for financial ratios"""
        fig = make_subplots(
            rows=len(selected_ratios), cols=1,
            subplot_titles=selected_ratios,
            shared_xaxes=True
        )
        
        for i, ratio in enumerate(selected_ratios):
            if ratio in ratios.index:
                fig.add_trace(
                    go.Scatter(
                        x=ratios.columns,
                        y=ratios.loc[ratio],
                        mode='lines+markers',
                        name=ratio,
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
                
                # Add average line
                avg_value = ratios.loc[ratio].mean()
                fig.add_hline(
                    y=avg_value, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text=f"Avg: {avg_value:.2f}",
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title=f"{company_name} - Financial Ratios Analysis",
            height=200 * len(selected_ratios),
            showlegend=False
        )
        
        return fig

class FinancialRatioCalculator:
    """Calculate comprehensive financial ratios"""
    
    @staticmethod
    def calculate_liquidity_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity ratios"""
        ratios = pd.DataFrame(index=df.columns)
        
        # Current Ratio
        if 'Current Assets' in df.index and 'Current Liabilities' in df.index:
            ratios['Current Ratio'] = df.loc['Current Assets'] / df.loc['Current Liabilities'].replace(0, np.nan)
        
        # Quick Ratio
        if all(metric in df.index for metric in ['Current Assets', 'Inventory', 'Current Liabilities']):
            quick_assets = df.loc['Current Assets'] - df.loc['Inventory']
            ratios['Quick Ratio'] = quick_assets / df.loc['Current Liabilities'].replace(0, np.nan)
        
        # Cash Ratio
        if 'Cash and Cash Equivalents' in df.index and 'Current Liabilities' in df.index:
            ratios['Cash Ratio'] = df.loc['Cash and Cash Equivalents'] / df.loc['Current Liabilities'].replace(0, np.nan)
        
        return ratios.T

    @staticmethod
    def calculate_profitability_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate profitability ratios"""
        ratios = pd.DataFrame(index=df.columns)
        
        # Net Profit Margin
        if 'Net Profit' in df.index and 'Revenue' in df.index:
            ratios['Net Profit Margin %'] = (df.loc['Net Profit'] / df.loc['Revenue'].replace(0, np.nan)) * 100
        
        # Gross Profit Margin
        if 'Gross Profit' in df.index and 'Revenue' in df.index:
            ratios['Gross Profit Margin %'] = (df.loc['Gross Profit'] / df.loc['Revenue'].replace(0, np.nan)) * 100
        
        # Operating Profit Margin
        if 'EBIT' in df.index and 'Revenue' in df.index:
            ratios['Operating Profit Margin %'] = (df.loc['EBIT'] / df.loc['Revenue'].replace(0, np.nan)) * 100
        
        # Return on Assets
        if 'Net Profit' in df.index and 'Total Assets' in df.index:
            ratios['Return on Assets %'] = (df.loc['Net Profit'] / df.loc['Total Assets'].replace(0, np.nan)) * 100
        
        # Return on Equity
        if 'Net Profit' in df.index and 'Total Equity' in df.index:
            ratios['Return on Equity %'] = (df.loc['Net Profit'] / df.loc['Total Equity'].replace(0, np.nan)) * 100
        
        # ROCE
        if all(metric in df.index for metric in ['EBIT', 'Total Assets', 'Current Liabilities']):
            capital_employed = df.loc['Total Assets'] - df.loc['Current Liabilities']
            ratios['ROCE %'] = (df.loc['EBIT'] / capital_employed.replace(0, np.nan)) * 100
        
        return ratios.T

    @staticmethod
    def calculate_leverage_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate leverage ratios"""
        ratios = pd.DataFrame(index=df.columns)
        
        # Debt to Equity
        if 'Total Debt' in df.index and 'Total Equity' in df.index:
            ratios['Debt to Equity'] = df.loc['Total Debt'] / df.loc['Total Equity'].replace(0, np.nan)
        
        # Debt Ratio
        if 'Total Debt' in df.index and 'Total Assets' in df.index:
            ratios['Debt Ratio'] = df.loc['Total Debt'] / df.loc['Total Assets'].replace(0, np.nan)
        
        # Interest Coverage
        if 'EBIT' in df.index and 'Interest Expense' in df.index:
            ratios['Interest Coverage'] = df.loc['EBIT'] / df.loc['Interest Expense'].replace(0, np.nan)
        
        # Equity Multiplier
        if 'Total Assets' in df.index and 'Total Equity' in df.index:
            ratios['Equity Multiplier'] = df.loc['Total Assets'] / df.loc['Total Equity'].replace(0, np.nan)
        
        return ratios.T

    @staticmethod
    def calculate_efficiency_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate efficiency ratios"""
        ratios = pd.DataFrame(index=df.columns)
        
        # Asset Turnover
        if 'Revenue' in df.index and 'Total Assets' in df.index:
            ratios['Asset Turnover'] = df.loc['Revenue'] / df.loc['Total Assets'].replace(0, np.nan)
        
        # Inventory Turnover
        if 'Cost of Goods Sold' in df.index and 'Inventory' in df.index:
            ratios['Inventory Turnover'] = df.loc['Cost of Goods Sold'] / df.loc['Inventory'].replace(0, np.nan)
        
        # Receivables Turnover
        if 'Revenue' in df.index and 'Trade Receivables' in df.index:
            ratios['Receivables Turnover'] = df.loc['Revenue'] / df.loc['Trade Receivables'].replace(0, np.nan)
        
        # Working Capital Turnover
        if all(metric in df.index for metric in ['Revenue', 'Current Assets', 'Current Liabilities']):
            working_capital = df.loc['Current Assets'] - df.loc['Current Liabilities']
            ratios['Working Capital Turnover'] = df.loc['Revenue'] / working_capital.replace(0, np.nan)
        
        return ratios.T

    @staticmethod
    def calculate_market_ratios(df: pd.DataFrame, market_price: float = None, 
                              shares_outstanding: float = None) -> pd.DataFrame:
        """Calculate market-based ratios"""
        ratios = pd.DataFrame(index=df.columns)
        
        if market_price and shares_outstanding:
            market_cap = market_price * shares_outstanding
            
            # P/E Ratio
            if 'Net Profit' in df.index:
                eps = df.loc['Net Profit'] / shares_outstanding
                ratios['P/E Ratio'] = market_price / eps.replace(0, np.nan)
            
            # P/B Ratio
            if 'Total Equity' in df.index:
                book_value_per_share = df.loc['Total Equity'] / shares_outstanding
                ratios['P/B Ratio'] = market_price / book_value_per_share.replace(0, np.nan)
            
            # EV/EBITDA
            if all(metric in df.index for metric in ['EBIT', 'Depreciation', 'Total Debt', 'Cash and Cash Equivalents']):
                ebitda = df.loc['EBIT'] + df.loc['Depreciation']
                enterprise_value = market_cap + df.loc['Total Debt'] - df.loc['Cash and Cash Equivalents']
                ratios['EV/EBITDA'] = enterprise_value / ebitda.replace(0, np.nan)
        
        return ratios.T

class PenmanNissimAnalyzer:
    """Penman-Nissim financial analysis framework"""
    
    def __init__(self, df: pd.DataFrame, mappings: Dict[str, str] = None):
        self.df = df
        self.mappings = mappings or {}
        self.logger = logging.getLogger(__name__)
    
    def calculate_all(self) -> Dict[str, pd.DataFrame]:
        """Calculate all Penman-Nissim components"""
        try:
            results = {
                'reformulated_balance_sheet': self._reformulate_balance_sheet(),
                'reformulated_income_statement': self._reformulate_income_statement(),
                'ratios': self._calculate_pn_ratios(),
                'free_cash_flow': self._calculate_free_cash_flow(),
                'value_drivers': self._calculate_value_drivers()
            }
            
            # Add ROE decomposition
            if 'ratios' in results and not results['ratios'].empty:
                results['roe_decomposition'] = self._decompose_roe(results['ratios'])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Penman-Nissim analysis: {e}")
            return {'error': str(e)}
    
    def _reformulate_balance_sheet(self) -> pd.DataFrame:
        """Reformulate balance sheet separating operating and financial activities"""
        reformulated = pd.DataFrame(index=self.df.columns)
        
        # Map metrics if mappings provided
        df = self.df.rename(index=self.mappings) if self.mappings else self.df
        
        # Operating Assets
        operating_assets = ['Current Assets', 'Property Plant Equipment', 'Intangible Assets']
        operating_assets_sum = pd.Series(0, index=df.columns)
        for asset in operating_assets:
            if asset in df.index:
                operating_assets_sum += df.loc[asset].fillna(0)
        
        # Financial Assets
        financial_assets = ['Cash', 'Short-term Investments', 'Long-term Investments']
        financial_assets_sum = pd.Series(0, index=df.columns)
        for asset in financial_assets:
            if asset in df.index:
                financial_assets_sum += df.loc[asset].fillna(0)
        
        # Operating Liabilities
        operating_liabilities = ['Accounts Payable', 'Accrued Expenses', 'Deferred Revenue']
        operating_liabilities_sum = pd.Series(0, index=df.columns)
        for liab in operating_liabilities:
            if liab in df.index:
                operating_liabilities_sum += df.loc[liab].fillna(0)
        
        # Financial Liabilities
        financial_liabilities = ['Short-term Debt', 'Long-term Debt', 'Bonds Payable']
        financial_liabilities_sum = pd.Series(0, index=df.columns)
        for liab in financial_liabilities:
            if liab in df.index:
                financial_liabilities_sum += df.loc[liab].fillna(0)
        
        # Calculate reformulated components
        reformulated['Operating Assets'] = operating_assets_sum
        reformulated['Operating Liabilities'] = operating_liabilities_sum
        reformulated['Net Operating Assets'] = operating_assets_sum - operating_liabilities_sum
        
        reformulated['Financial Assets'] = financial_assets_sum
        reformulated['Financial Liabilities'] = financial_liabilities_sum
        reformulated['Net Financial Assets'] = financial_assets_sum - financial_liabilities_sum
        
        # Common Equity should equal NOA + NFA
        if 'Total Equity' in df.index:
            reformulated['Common Equity'] = df.loc['Total Equity']
        else:
            reformulated['Common Equity'] = reformulated['Net Operating Assets'] + reformulated['Net Financial Assets']
        
        return reformulated
    
    def _reformulate_income_statement(self) -> pd.DataFrame:
        """Reformulate income statement separating operating and financial activities"""
        reformulated = pd.DataFrame(index=self.df.columns)
        
        df = self.df.rename(index=self.mappings) if self.mappings else self.df
        
        # Operating Income
        if 'Operating Income' in df.index:
            reformulated['Operating Income'] = df.loc['Operating Income']
        elif 'EBIT' in df.index:
            reformulated['Operating Income'] = df.loc['EBIT']
        
        # Tax allocation
        if all(metric in df.index for metric in ['Tax Expense', 'Income Before Tax']):
            income_before_tax = df.loc['Income Before Tax'].replace(0, np.nan)
            tax_rate = df.loc['Tax Expense'] / income_before_tax
            
            if 'Operating Income' in reformulated.index:
                reformulated['Tax on Operating Income'] = reformulated['Operating Income'] * tax_rate
                reformulated['Operating Income After Tax'] = reformulated['Operating Income'] - reformulated['Tax on Operating Income']
        
        # Financial Expense
        if 'Interest Expense' in df.index:
            reformulated['Net Financial Expense'] = df.loc['Interest Expense']
            if 'Interest Income' in df.index:
                reformulated['Net Financial Expense'] -= df.loc['Interest Income']
            
            # Tax benefit on financial expense
            if 'tax_rate' in locals():
                reformulated['Tax Benefit on Financial Expense'] = reformulated['Net Financial Expense'] * tax_rate
                reformulated['Net Financial Expense After Tax'] = reformulated['Net Financial Expense'] - reformulated['Tax Benefit on Financial Expense']
        
        # Comprehensive Income
        if 'Net Income' in df.index:
            reformulated['Comprehensive Income'] = df.loc['Net Income']
        
        return reformulated
    
    def _calculate_pn_ratios(self) -> pd.DataFrame:
        """Calculate Penman-Nissim financial ratios"""
        ratios = pd.DataFrame(index=self.df.columns)
        
        ref_bs = self._reformulate_balance_sheet()
        ref_is = self._reformulate_income_statement()
        
        # RNOA (Return on Net Operating Assets)
        if 'Operating Income After Tax' in ref_is.index and 'Net Operating Assets' in ref_bs.index:
            avg_noa = ref_bs.loc['Net Operating Assets'].rolling(window=2).mean()
            avg_noa.iloc[0] = ref_bs.loc['Net Operating Assets'].iloc[0]
            ratios['Return on Net Operating Assets (RNOA) %'] = (ref_is.loc['Operating Income After Tax'] / avg_noa.replace(0, np.nan)) * 100
        
        # FLEV (Financial Leverage)
        
        if 'Net Financial Assets' in ref_bs.index and 'Common Equity' in ref_bs.index:
            avg_ce = ref_bs.loc['Common Equity'].rolling(window=2).mean()
            avg_ce.iloc[0] = ref_bs.loc['Common Equity'].iloc[0]
            ratios['Financial Leverage (FLEV)'] = -ref_bs.loc['Net Financial Assets'] / avg_ce.replace(0, np.nan)
        
        # NBC (Net Borrowing Cost)
        if 'Net Financial Expense' in ref_is.index and 'Net Financial Assets' in ref_bs.index:
            avg_nfa = ref_bs.loc['Net Financial Assets'].rolling(window=2).mean()
            avg_nfa.iloc[0] = ref_bs.loc['Net Financial Assets'].iloc[0]
            ratios['Net Borrowing Cost (NBC) %'] = (-ref_is.loc['Net Financial Expense'] / avg_nfa.replace(0, np.nan)) * 100
        
        # OPM (Operating Profit Margin)
        if 'Operating Income After Tax' in ref_is.index and 'Revenue' in self.df.index:
            revenue = self.df.loc['Revenue'] if 'Revenue' in self.df.index else self.df.loc[self.mappings.get('Revenue', 'Revenue')]
            ratios['Operating Profit Margin (OPM) %'] = (ref_is.loc['Operating Income After Tax'] / revenue.replace(0, np.nan)) * 100
        
        # NOAT (Net Operating Asset Turnover)
        if 'Revenue' in self.df.index and 'Net Operating Assets' in ref_bs.index:
            revenue = self.df.loc['Revenue'] if 'Revenue' in self.df.index else self.df.loc[self.mappings.get('Revenue', 'Revenue')]
            avg_noa = ref_bs.loc['Net Operating Assets'].rolling(window=2).mean()
            avg_noa.iloc[0] = ref_bs.loc['Net Operating Assets'].iloc[0]
            ratios['Net Operating Asset Turnover (NOAT)'] = revenue / avg_noa.replace(0, np.nan)
        
        # Spread (RNOA - NBC)
        if 'Return on Net Operating Assets (RNOA) %' in ratios.index and 'Net Borrowing Cost (NBC) %' in ratios.index:
            ratios['Spread %'] = ratios.loc['Return on Net Operating Assets (RNOA) %'] - ratios.loc['Net Borrowing Cost (NBC) %']
        
        # ROE (Return on Equity) - using Penman-Nissim formula
        if all(metric in ratios.index for metric in ['Return on Net Operating Assets (RNOA) %', 'Financial Leverage (FLEV)', 'Spread %']):
            ratios['Return on Equity (ROE) %'] = ratios.loc['Return on Net Operating Assets (RNOA) %'] + (ratios.loc['Financial Leverage (FLEV)'] * ratios.loc['Spread %'])
        
        return ratios.T
    
    def _calculate_free_cash_flow(self) -> pd.DataFrame:
        """Calculate free cash flow components"""
        fcf = pd.DataFrame(index=self.df.columns)
        
        df = self.df.rename(index=self.mappings) if self.mappings else self.df
        
        if 'Operating Cash Flow' in df.index:
            fcf['Operating Cash Flow'] = df.loc['Operating Cash Flow']
            
            if 'Capital Expenditure' in df.index:
                fcf['Capital Expenditure'] = df.loc['Capital Expenditure']
                fcf['Free Cash Flow'] = fcf['Operating Cash Flow'] - fcf['Capital Expenditure']
            else:
                fcf['Free Cash Flow'] = fcf['Operating Cash Flow']
            
            # Free Cash Flow to Equity
            if all(metric in df.index for metric in ['Net Income', 'Depreciation']):
                fcf['Free Cash Flow to Equity'] = (
                    df.loc['Net Income'] + 
                    df.loc['Depreciation'] - 
                    (df.loc['Capital Expenditure'] if 'Capital Expenditure' in df.index else 0)
                )
            
            # Free Cash Flow Yield
            if 'Free Cash Flow' in fcf.index and 'Total Assets' in df.index:
                fcf['FCF Yield %'] = (fcf['Free Cash Flow'] / df.loc['Total Assets'].replace(0, np.nan)) * 100
        
        return fcf.T
    
    def _calculate_value_drivers(self) -> pd.DataFrame:
        """Calculate value drivers for DCF analysis"""
        drivers = pd.DataFrame(index=self.df.columns)
        
        df = self.df.rename(index=self.mappings) if self.mappings else self.df
        
        # Revenue growth rate
        if 'Revenue' in df.index:
            revenue = df.loc['Revenue']
            drivers['Revenue Growth %'] = revenue.pct_change() * 100
            
            # CAGR
            if len(revenue) > 1:
                years = len(revenue) - 1
                cagr = ((revenue.iloc[-1] / revenue.iloc[0]) ** (1/years) - 1) * 100
                drivers['Revenue CAGR %'] = cagr
        
        # NOPAT margin
        ref_is = self._reformulate_income_statement()
        if 'Operating Income After Tax' in ref_is.index and 'Revenue' in df.index:
            drivers['NOPAT Margin %'] = (ref_is.loc['Operating Income After Tax'] / df.loc['Revenue'].replace(0, np.nan)) * 100
        
        # Working capital as % of revenue
        if all(metric in df.index for metric in ['Current Assets', 'Current Liabilities', 'Revenue']):
            working_capital = df.loc['Current Assets'] - df.loc['Current Liabilities']
            drivers['Working Capital % of Revenue'] = (working_capital / df.loc['Revenue'].replace(0, np.nan)) * 100
        
        # Capital intensity
        if 'Capital Expenditure' in df.index and 'Revenue' in df.index:
            drivers['Capital Intensity %'] = (df.loc['Capital Expenditure'] / df.loc['Revenue'].replace(0, np.nan)) * 100
        
        # Asset efficiency
        if 'Revenue' in df.index and 'Total Assets' in df.index:
            drivers['Asset Turnover'] = df.loc['Revenue'] / df.loc['Total Assets'].replace(0, np.nan)
        
        return drivers.T
    
    def _decompose_roe(self, ratios: pd.DataFrame) -> pd.DataFrame:
        """Decompose ROE using Penman-Nissim framework"""
        decomposition = pd.DataFrame(index=ratios.columns)
        
        if 'Return on Net Operating Assets (RNOA) %' in ratios.index:
            decomposition['Operating Component'] = ratios.loc['Return on Net Operating Assets (RNOA) %']
        
        if all(metric in ratios.index for metric in ['Financial Leverage (FLEV)', 'Spread %']):
            decomposition['Leverage Effect'] = ratios.loc['Financial Leverage (FLEV)'] * ratios.loc['Spread %']
        
        if 'Operating Component' in decomposition.columns and 'Leverage Effect' in decomposition.columns:
            decomposition['Total ROE'] = decomposition['Operating Component'] + decomposition['Leverage Effect']
        
        return decomposition

# File parsing functions
def parse_html_content(file_content: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Parse HTML/XLS content"""
    try:
        dfs = pd.read_html(io.BytesIO(file_content), header=[0, 1])
        if dfs:
            df = dfs[0]
            company_name = "Unknown Company"
            if hasattr(df.columns, 'levels') and len(df.columns.levels) > 0:
                first_col = str(df.columns[0][0]) if isinstance(df.columns[0], tuple) else str(df.columns[0])
                if ">>" in first_col:
                    company_name = first_col.split(">>")[2].split("(")[0].strip()
            
            df.columns = [str(c[1]) if isinstance(c, tuple) else str(c) for c in df.columns]
            
            # Process the dataframe
            df = DataProcessor.clean_numeric_data(df)
            
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
        df = DataProcessor.clean_numeric_data(df)
        
        return {
            "statement": df,
            "company_name": "From CSV",
            "source": filename
        }
    except Exception as e:
        logger.error(f"Failed to parse CSV content: {e}")
        return None

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
