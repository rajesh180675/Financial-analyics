# financial_analytics_ui.py
# Streamlit UI components that use the core analytics

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import io
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from financial_analytics_core import (
    DataProcessor, ChartGenerator, FinancialRatioCalculator,
    PenmanNissimAnalyzer, IndustryBenchmarks, DataQualityMetrics,
    parse_html_content, parse_csv_content, process_and_merge_dataframes,
    REQUIRED_METRICS, ALLOWED_FILE_TYPES, MAX_FILE_SIZE_MB, YEAR_REGEX,
    asdict
)

# Wrapper functions to work with Streamlit UploadedFile
def parse_html_xls_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    """Parse HTML/XLS files from Streamlit upload"""
    try:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB.")
        
        file_content = uploaded_file.getvalue()
        return parse_html_content(file_content, uploaded_file.name)
    except Exception as e:
        st.error(f"Failed to parse {uploaded_file.name}: {e}")
        return None

def parse_csv_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    """Parse CSV files from Streamlit upload"""
    try:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB.")
        
        file_content = uploaded_file.getvalue()
        return parse_csv_content(file_content, uploaded_file.name)
    except Exception as e:
        st.error(f"Failed to parse {uploaded_file.name}: {e}")
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
        
        # Process the parsed data
        df = parsed_data["statement"]
        
        # Year column processing
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
        
        df_proc = DataProcessor.clean_numeric_data(df[valid_years].copy())
        df_proc = df_proc.dropna(how='all')
        
        parsed_data["statement"] = df_proc
        parsed_data["year_columns"] = valid_years
        
        return parsed_data
    except Exception as e:
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

# Copy the DashboardApp class here with all its UI methods
class DashboardApp:
    # Copy the entire DashboardApp class from the original code
    pass
