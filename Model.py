# \\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\


# -- SELECT COUNT(*) FROM vibepay_portal.vibepay_payin_report where `Date Time` LIKE '2025-07-08%';

# SELECT COUNT(*) FROM vibepay_portal.vibepay_payin_wallet where `Date` LIKE '2025-07-08%';

# SELECT COUNT(*) FROM vibepay_portal.vibepay_payout_report where `Date Time` LIKE '2025-07-08%';

# SELECT COUNT(*) FROM vibepay_portal.vibepay_payout_wallet where `Date` LIKE '2025-07-08%';

# SELECT COUNT(*) FROM vibepay_portal.vibepayin_add_fund_report where `Date` LIKE '2025-07-08%';


# SELECT COUNT(*) FROM sabpaisa.payin where `Transaction Date` LIKE '2025-07-08%';

# SELECT COUNT(*) FROM sabpaisa.settelment_report where `TRANSACTION DATE` LIKE '2025-07-08%';



# SELECT count(*) FROM payonetic_portal.payonetic_payout_report where `Date Time` LIKE '2025-07-08%';

# SELECT count(*) FROM payonetic_portal.payonetic_payout_wallet where `Date Time` LIKE '2025-07-08%';

# SELECT count(*) FROM payonetic_portal.payonetic_add_fund_report where `Date Time` LIKE '2025-07-08%';

# SELECT count(*) FROM swiftsend_portal.swiftsend_payout_report where `Date Time` LIKE '2025-07-08%';

# SELECT count(*) FROM swiftsend_portal.swiftsend_add_fund_report where `Date` LIKE '2025-07-08%';

# SELECT count(*) FROM swiftsend_portal.swiftsend_payout_wallet where `Date` LIKE '2025-07-08%';



from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os
import sys
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import mysql.connector
from mysql.connector import Error
import re
from datetime import datetime, timedelta
import pandas as pd
import datetime
from datetime import datetime as dt
from typing import List, Dict, Optional, Any


class ConfigManager:
    """Centralized configuration management"""
    DATABASE_CONFIGS = {
        'fbpayripe': {
            'tables': ['payin', 'payout', 'wallet'],
            'default_host': 'localhost',
            'default_port': 3306
        },
        'anateck': {
            'tables': ['payout', 'wallet', 'payon_payout', 'payon_owallet', 'idfc', 'yes', 'kotak'],
            'default_host': 'localhost',
            'default_port': 3306
        },
        'attroidfc': {
            'tables': ['payout', 'wallet', 'payon_payout', 'payon_owallet', 'idfc', 'yes', 'kotak'],
            'default_host': 'localhost',
            'default_port': 3306
        },
        'vibepay_Portal': {
            'tables': ['vibepay_payin_report', 'vibepay_payin_wallet', 'vibepay_payout_report', 'vibepay_payout_wallet', 'vibepayin_add_fund_report', 'vibepay_add_fund_report'],
            'default_host': 'localhost',
            'default_port': 3306
        },
        'sabpaisa': {
            'tables': ['payin', 'settelment_report', 'settel_hai'],
            'default_host': 'localhost',
            'default_port': 3306
        },
        
        'rozarpayx': {
            'tables': ['payout_report', 'transaction_report'],
            'default_host': 'localhost',
            'default_port': 3306
        },
        'swiftsend_portal': {
            'tables': ['swiftsend_payout_report', 'swiftsend_payout_wallet', 'swiftsend_add_fund_report', 'swiftsend_finzen_payout'],
            'default_host': 'localhost',
            'default_port': 3306
        },
        'payonetic_portal': {
            'tables': ['payonetic_payout_report', 'payonetic_payout_wallet', 'payonetic_add_fund_report'],
            'default_host': 'localhost',
            'default_port': 3306
        }
    }

    # Define expected table formats for specific database-table combinations
    TABLE_FORMATS = {
        'anateck.idfc': [
            "Transaction Date", "Payment date", "Narrative", "Customer Reference No", 
            "Cheque No", "Debit", "Credit", "Running Balance"
        ],
        'attroidfc.idfc': [
            "Transaction Date", "Payment date", "Narrative", "Customer Reference No", 
            "Cheque No", "Debit", "Credit", "Running Balance"
        ],
        'anateck.yes': [
            "Transaction Date", "Value date", "Transaction Description", "Reference No", 
            "Debit Amount", "Credit Amount", "Running Balance"
        ],
        'attroidfc.yes': [
            "Transaction Date", "Value date", "Transaction Description", "Reference No", 
            "Debit Amount", "Credit Amount", "Running Balance"
        ],
        'anateck.kotak': [
            "#", "TRANSACTION DATE", "VALUE DATE", "TRANSACTION DETAILS", 
            "CHQ / REF NO.", "DEBIT/CREDIT(₹)", "BALANCE(₹)"
        ],
        'attroidfc.kotak': [
            "#", "TRANSACTION DATE", "VALUE DATE", "TRANSACTION DETAILS", 
            "CHQ / REF NO.", "DEBIT/CREDIT(₹)", "BALANCE(₹)"
        ]
    }

    @classmethod
    def get_database_options(cls):
        """Generate database-table combinations"""
        options = []
        for db, config in cls.DATABASE_CONFIGS.items():
            for table in config['tables']:
                options.append(f"{db}.{table}")
        return options
    
   

    @classmethod
    def get_table_instructions(cls, db_name: str, table_name: str) -> str:
        """Get specific instructions for a database table"""
        instructions = {
            'fbpayripe.payin': 'Remove First Col. Created Time and Then Upload',
            'fbpayripe.payout': 'Upload as it is.',
            'fbpayripe.wallet': 'Upload as it is.',
            'anateck.payout': 'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'anateck.wallet': 'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'anateck.payon_payout': 'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'anateck.payon_owallet': 'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'anateck.idfc': 'Extract specific table format data and process UTR/UID',
            'anateck.yes': 'Extract specific table format data and process dates/amounts',
            'anateck.kotak': 'Extract specific table format data and process dates/amounts',
            'attroidfc.payout': 'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'attroidfc.wallet': 'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'attroidfc.payon_payout': 'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'attroidfc.payon_owallet': 'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'attroidfc.idfc': 'Extract specific table format data and process UTR/UID',
            'attroidfc.yes': 'Extract specific table format data and process dates/amounts',
            'attroidfc.kotak': 'Extract specific table format data and process dates/amounts',
            'vibepay_Portal.vibepay_payin_report': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'vibepay_Portal.vibepay_payin_wallet': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'vibepay_Portal.vibepay_payout_report': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'vibepay_Portal.vibepay_payout_wallet': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'vibepay_Portal.vibepay_add_fund_report': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'vibepay_Portal.vibepayin_add_fund_report': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'sabpaisa.settel_hai': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'sabpaisa.payin': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'sabpaisa.settelment_report': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',

            'rozarpayx.payout_report': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'rozarpayx.transaction_report': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            
            'swiftsend_portal.swiftsend_payout_report': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'swiftsend_portal.swiftsend_payout_wallet': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'swiftsend_portal.swiftsend_finzen_payout': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'swiftsend_portal.swiftsend_add_fund_report': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'payonetic_portal.payonetic_payout_report': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'payonetic_portal.payonetic_payout_wallet': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.',
            'payonetic_portal.payonetic_add_fund_report': 'Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.'
        }
        return instructions.get(f"{db_name}.{table_name}", "Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format.")

    @classmethod
    def get_expected_columns(cls, db_name: str, table_name: str) -> Optional[List[str]]:
        """Get expected column format for specific table"""
        key = f"{db_name}.{table_name}"
        return cls.TABLE_FORMATS.get(key)

    @classmethod
    def requires_table_format_extraction(cls, db_name: str, table_name: str) -> bool:
        """Check if table requires specific format extraction"""
        return f"{db_name}.{table_name}" in cls.TABLE_FORMATS

class LoggerManager:
    """Advanced logging configuration"""
    @staticmethod
    def setup_logger(log_file='data_uploader.log'):
        """Configure comprehensive logging"""
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        full_log_path = os.path.join(log_dir, log_file)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(full_log_path, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)

class TableFormatExtractor:
    """Handles extraction of specific table formats from Excel files"""
    
    @staticmethod
    def extract_table_data(df: pd.DataFrame, expected_columns: List[str], logger) -> pd.DataFrame:
        """Extract table data based on expected column format"""
        logger.info(f"Looking for table with columns: {expected_columns}")
        
        # Find the header row that matches expected columns
        header_row_idx = TableFormatExtractor._find_header_row(df, expected_columns, logger)
        
        if header_row_idx is None:
            raise ValueError(f"Could not find table with expected columns: {expected_columns}")
        
        logger.info(f"Found table header at row {header_row_idx}")
        
        # Extract data starting from header row
        table_df = df.iloc[header_row_idx:].copy()
        
        # Set the header row as column names
        table_df.columns = table_df.iloc[0]
        table_df = table_df.iloc[1:]  # Remove header row from data
        
        # Filter to only expected columns (in case there are extra columns)
        available_columns = [col for col in expected_columns if col in table_df.columns]
        if len(available_columns) != len(expected_columns):
            missing_cols = set(expected_columns) - set(available_columns)
            logger.warning(f"Missing expected columns: {missing_cols}")
        
        # Select only the expected columns that are available
        table_df = table_df[available_columns].copy()
        
        # Remove empty rows
        table_df = table_df.dropna(how='all')
        
        # Reset index
        table_df.reset_index(drop=True, inplace=True)
        
        logger.info(f"Extracted table data: {len(table_df)} rows, {len(table_df.columns)} columns")
        logger.info(f"Extracted columns: {list(table_df.columns)}")
        
        return table_df
    
    @staticmethod
    def _find_header_row(df: pd.DataFrame, expected_columns: List[str], logger) -> Optional[int]:
        """Find the row that contains the expected column headers"""
        
        def normalize_text(text):
            """Normalize text for comparison"""
            if pd.isna(text):
                return ""
            return str(text).strip().lower()
        
        # Normalize expected columns for comparison
        normalized_expected = [normalize_text(col) for col in expected_columns]
        
        # Search through rows to find matching headers
        for idx in range(min(50, len(df))):  # Search first 50 rows
            row_values = df.iloc[idx].values
            normalized_row = [normalize_text(val) for val in row_values]
            
            # Check if this row contains the expected columns
            matches = 0
            for expected_col in normalized_expected:
                if expected_col in normalized_row:
                    matches += 1
            
            # If we found most of the expected columns (allow some flexibility)
            match_ratio = matches / len(normalized_expected)
            logger.debug(f"Row {idx}: Found {matches}/{len(normalized_expected)} columns (ratio: {match_ratio:.2f})")
            
            if match_ratio >= 0.7:  # At least 70% of columns should match
                # Verify this looks like a header row
                if TableFormatExtractor._looks_like_header_row(row_values, expected_columns):
                    return idx
        
        return None
    
    @staticmethod
    def _looks_like_header_row(row_values, expected_columns) -> bool:
        """Check if a row looks like it contains headers"""
        non_empty_count = sum(1 for val in row_values if pd.notna(val) and str(val).strip())
        
        # Header row should have multiple non-empty values
        if non_empty_count < len(expected_columns) * 0.5:
            return False
        
        # Check if values look like column names (contain alphabetic characters)
        text_like_count = 0
        for val in row_values:
            if pd.notna(val):
                str_val = str(val).strip()
                if str_val and any(c.isalpha() for c in str_val):
                    text_like_count += 1
        
        return text_like_count >= len(expected_columns) * 0.5

class DataProcessor:
    @staticmethod
    def process_idfc_reference_numbers(df: pd.DataFrame, logger) -> pd.DataFrame:
        """Process IDFC specific columns to extract reference numbers from Narrative.
        Extracts:
        - Customer Reference No (12-digit number after UPI/MOB/)
        - Cheque No (alphanumeric code after the reference number)
        Example: From "UPI/MOB/105300149400/GAK1747992827326100" extracts:
            Customer Reference No: 105300149400
            Cheque No: GAK1747992827326100
        """
        if 'Narrative' not in df.columns:
            logger.warning("Narrative column not found in IDFC data")
            return df
            
        processed_df = df.copy()
        
        # Initialize columns with None if they don't exist
        processed_df['Customer Reference No'] = processed_df.get('Customer Reference No', None)
        processed_df['Cheque No'] = processed_df.get('Cheque No', None)
        
        # Primary pattern to extract both values in one pass
        primary_pattern = re.compile(r'UPI/MOB/(\d{10,15})/([A-Za-z0-9]{8,30})')
        
        for idx, row in processed_df.iterrows():
            narrative = str(row['Narrative']).strip()
            
            # Try to extract both values with primary pattern first
            primary_match = primary_pattern.search(narrative)
            if primary_match:
                processed_df.at[idx, 'Customer Reference No'] = primary_match.group(1)
                processed_df.at[idx, 'Cheque No'] = primary_match.group(2)
                logger.debug(f"Row {idx}: Extracted both values - Ref: {primary_match.group(1)}, Cheque: {primary_match.group(2)}")
                continue
            
            # Fallback logic if primary pattern fails
            # Extract Customer Reference No
            if pd.isna(processed_df.at[idx, 'Customer Reference No']) or str(processed_df.at[idx, 'Customer Reference No']).strip() in ['--', 'nan', 'None', '', 'null']:
                ref_match = re.search(r'/(\d{10,15})(?:/|$)', narrative)
                if ref_match:
                    processed_df.at[idx, 'Customer Reference No'] = ref_match.group(1)
                    logger.debug(f"Row {idx}: Extracted Customer Reference No (fallback): {ref_match.group(1)}")
            
            # Extract Cheque No
            if pd.isna(processed_df.at[idx, 'Cheque No']) or str(processed_df.at[idx, 'Cheque No']).strip() in ['--', 'nan', 'None', '', 'null']:
                cheque_match = re.search(r'/([A-Za-z0-9]{8,30})(?:/|$)', narrative)
                if cheque_match and not cheque_match.group(1).isdigit():  # Ensure it's not just digits
                    processed_df.at[idx, 'Cheque No'] = cheque_match.group(1)
                    logger.debug(f"Row {idx}: Extracted Cheque No (fallback): {cheque_match.group(1)}")
        
        # Count how many values were extracted
        initial_ref_count = processed_df['Customer Reference No'].notna().sum()
        initial_cheque_count = processed_df['Cheque No'].notna().sum()
        
        # Additional processing for rows where we got one value but not the other
        for idx, row in processed_df.iterrows():
            narrative = str(row['Narrative']).strip()
            
            # If we have Reference No but not Cheque No
            if pd.notna(processed_df.at[idx, 'Customer Reference No']) and pd.isna(processed_df.at[idx, 'Cheque No']):
                ref_no = str(processed_df.at[idx, 'Customer Reference No'])
                cheque_match = re.search(rf'{re.escape(ref_no)}/([A-Za-z0-9]{{8,30}})', narrative)
                if cheque_match:
                    processed_df.at[idx, 'Cheque No'] = cheque_match.group(1)
                    logger.debug(f"Row {idx}: Extracted Cheque No using known RefNo: {cheque_match.group(1)}")
            
            # If we have Cheque No but not Reference No
            elif pd.notna(processed_df.at[idx, 'Cheque No']) and pd.isna(processed_df.at[idx, 'Customer Reference No']):
                cheque_no = str(processed_df.at[idx, 'Cheque No'])
                ref_match = re.search(r'/(\d{10,15})/' + re.escape(cheque_no), narrative)
                if ref_match:
                    processed_df.at[idx, 'Customer Reference No'] = ref_match.group(1)
                    logger.debug(f"Row {idx}: Extracted RefNo using known ChequeNo: {ref_match.group(1)}")
        
        # Final counts after additional processing
        final_ref_count = processed_df['Customer Reference No'].notna().sum()
        final_cheque_count = processed_df['Cheque No'].notna().sum()
        
        logger.info("IDFC Reference Number Processing Summary:")
        logger.info(f"- Initial Customer Reference Numbers: {initial_ref_count}")
        logger.info(f"- Initial Cheque Numbers: {initial_cheque_count}")
        logger.info(f"- Final Customer Reference Numbers: {final_ref_count}")
        logger.info(f"- Final Cheque Numbers: {final_cheque_count}")
        logger.info(f"- New Reference Numbers extracted: {final_ref_count - initial_ref_count}")
        logger.info(f"- New Cheque Numbers extracted: {final_cheque_count - initial_cheque_count}")
        
        return processed_df

    @staticmethod
    def process_data(df: pd.DataFrame, db_name: str, table_name: str, logger) -> pd.DataFrame:
        """Main data processing pipeline with enhanced handling for all database tables"""
        processed_df = df.copy()
        
        # Step 1: Protect scientific notation first (for all numeric columns)
        processed_df = DataProcessor.protect_scientific_notation(processed_df)
        
        # Step 2: Process dates (for all date-like columns)
        processed_df = DataProcessor.format_dates(processed_df, logger)
        
        # Step 3: Clean numeric columns (remove commas, handle negatives)
        processed_df = DataProcessor.clean_amount_columns(processed_df)
        
        # Step 4: Apply table-specific processing if needed
        if ConfigManager.requires_table_format_extraction(db_name, table_name):
            expected_cols = ConfigManager.get_expected_columns(db_name, table_name)
            if expected_cols:
                processed_df = TableFormatExtractor.extract_table_data(processed_df, expected_cols, logger)
        
        # Apply any additional table-specific processing
        processed_df = DataProcessor.apply_table_specific_processing(processed_df, db_name, table_name, logger)
        
        return processed_df

    @staticmethod
    def apply_table_specific_processing(df: pd.DataFrame, db_name: str, table_name: str, logger) -> pd.DataFrame:
        """Apply specific processing based on database and table"""
        # Common processing for all bank statement tables
        if table_name.lower() in ['idfc', 'yes', 'kotak']:
            df = DataProcessor.process_bank_statement_columns(df, table_name.lower(), logger)
        
        # Remove first column if instructed
        instructions = ConfigManager.get_table_instructions(db_name, table_name)
        if "Remove First Col. Created Time" in instructions:
            df = DataProcessor.remove_first_column(df)
        
        return df

    @staticmethod
    def process_bank_statement_columns(df: pd.DataFrame, bank_name: str, logger) -> pd.DataFrame:
        """Standard processing for all bank statement columns"""
        processed_df = df.copy()
        
        # Common date processing for all bank statements
        date_cols = [col for col in processed_df.columns if 'date' in col.lower()]
        for col in date_cols:
            processed_df[col] = DataProcessor._convert_date_column(processed_df[col], logger)
        
        # Common amount processing
        amount_cols = [col for col in processed_df.columns if any(
            kw in col.lower() for kw in ['amount', 'debit', 'credit', 'balance', 'value']
        )]
        
        for col in amount_cols:
            processed_df[col] = DataProcessor._clean_numeric_value(processed_df[col], logger)
        
        # Bank-specific processing
        if bank_name == 'idfc':
            processed_df = DataProcessor.process_idfc_reference_numbers(processed_df, logger)
        elif bank_name == 'yes':
            processed_df = DataProcessor._process_yes_bank_columns(processed_df, logger)
        elif bank_name == 'kotak':
            processed_df = DataProcessor._process_kotak_columns(processed_df, logger)
        
        return processed_df    

    @staticmethod
    def _clean_numeric_value(series: pd.Series, logger) -> pd.Series:
        """Robust numeric cleaning for all numeric columns"""
        # Convert to string first for consistent processing
        str_series = series.astype(str)
        
        # Remove commas, currency symbols, and whitespace
        cleaned = str_series.str.replace(r'[₹,$€£,\s]', '', regex=True)
        
        # Handle negative numbers in parentheses
        cleaned = cleaned.str.replace(r'\(([\d.]+)\)', r'-\1', regex=True)
        
        # Convert to numeric, coercing errors to NaN
        numeric = pd.to_numeric(cleaned, errors='coerce')
        
        # Log conversion issues
        if numeric.isna().any():
            problem_values = str_series[numeric.isna()].unique()
            logger.warning(
                f"Could not convert {len(problem_values)} values to numeric. "
                f"Sample problem values: {problem_values[:5]}"
            )
        
        return numeric.fillna(0)


    @staticmethod
    def protect_scientific_notation(df: pd.DataFrame) -> pd.DataFrame:
        """Protect scientific notation numbers from being converted"""
        processed_df = df.copy()
        
        for col in processed_df.columns:
            # Skip date columns
            if any(keyword in str(col).lower() for keyword in ['date', 'time']):
                continue
                
            # Process numeric columns
            if processed_df[col].dtype == 'object':
                # Check if column contains scientific notation
                sci_mask = processed_df[col].astype(str).str.contains(
                    r'^\d+\.?\d*[Ee][+-]?\d+$', 
                    na=False
                )
                
                if sci_mask.any():
                    # Convert scientific notation to full string representation
                    processed_df.loc[sci_mask, col] = (
                        processed_df.loc[sci_mask, col]
                        .apply(lambda x: format(float(x), '.{}f'.format(
                            len(str(x).split('.')[1]) if '.' in str(x) else 0
                        )))
                    )
        
        return processed_df
    @staticmethod
    def _process_kotak_bank_file(df: pd.DataFrame, logger) -> pd.DataFrame:
        """Process Kotak Excel file - Extract ALL data from specified columns"""
        logger.info("Processing Kotak Excel file - Extracting ALL data")
        
        # Expected column headers exactly as they appear in Excel
        EXPECTED_HEADERS = [
            "#", 
            "TRANSACTION DATE", 
            "VALUE DATE", 
            "TRANSACTION DETAILS", 
            "CHQ / REF NO.", 
            "DEBIT/CREDIT(₹)", 
            "BALANCE(₹)"
        ]
        
        # Convert DataFrame to string for processing
        str_df = df.astype(str).apply(lambda x: x.str.strip())
        
        # Find header row (might not be the first row)
        header_row_idx = None
        for idx in range(len(str_df)):
            row_values = str_df.iloc[idx].values
            # Check if this row contains the expected headers
            header_matches = 0
            for header in EXPECTED_HEADERS:
                if any(header.upper() in str(val).upper() for val in row_values):
                    header_matches += 1
            
            if header_matches >= 5:  # At least 5 headers should match
                header_row_idx = idx
                logger.info(f"Found header row at index {idx}")
                break
        
        if header_row_idx is None:
            # If no clear header row found, assume first row is header
            header_row_idx = 0
            logger.warning("Header row not clearly identified, using first row")
        
        # Extract data starting from after header row
        if header_row_idx < len(str_df) - 1:
            data_df = str_df.iloc[header_row_idx + 1:].copy()
            # Set column names from header row
            data_df.columns = str_df.iloc[header_row_idx].values
        else:
            raise ValueError("No data rows found after header")
        
        # Clean column names
        data_df.columns = [str(col).strip() for col in data_df.columns]
        
        # Map columns to expected names (in case of slight variations)
        column_mapping = {}
        for col in data_df.columns:
            col_upper = str(col).upper()
            if '#' in col_upper or 'SERIAL' in col_upper or 'S.NO' in col_upper or 'SR.NO' in col_upper:
                column_mapping['#'] = col
            elif 'TRANSACTION DATE' in col_upper or 'TXN DATE' in col_upper:
                column_mapping['TRANSACTION DATE'] = col
            elif 'VALUE DATE' in col_upper or 'VAL DATE' in col_upper:
                column_mapping['VALUE DATE'] = col
            elif 'TRANSACTION DETAILS' in col_upper or 'PARTICULARS' in col_upper or 'DESCRIPTION' in col_upper:
                column_mapping['TRANSACTION DETAILS'] = col
            elif ('CHQ' in col_upper and 'REF' in col_upper) or 'REF NO' in col_upper or 'REFERENCE' in col_upper:
                column_mapping['CHQ / REF NO.'] = col
            elif ('DEBIT' in col_upper and 'CREDIT' in col_upper) or 'AMOUNT' in col_upper or 'DR/CR' in col_upper:
                column_mapping['DEBIT/CREDIT(₹)'] = col
            elif 'BALANCE' in col_upper and '₹' in col_upper:
                column_mapping['BALANCE(₹)'] = col
            elif 'BALANCE' in col_upper and 'BAL' in col_upper:
                column_mapping['BALANCE(₹)'] = col
        
        logger.debug(f"Column mapping: {column_mapping}")
        
        # Create processed DataFrame with ALL data (no filtering)
        processed_data = {}
        
        # Serial Number - Extract ALL entries
        if '#' in column_mapping:
            serial_data = data_df[column_mapping['#']].fillna('').astype(str)
            # Keep original values, don't filter
            processed_data['#'] = serial_data
        else:
            # Generate serial numbers for all rows
            processed_data['#'] = [str(i+1) for i in range(len(data_df))]
        
        # Transaction Date - Extract ALL entries
        if 'TRANSACTION DATE' in column_mapping:
            date_col = data_df[column_mapping['TRANSACTION DATE']].astype(str)
            # Handle dates with newlines or extra text but keep ALL entries
            date_col = date_col.str.split('\n').str[0].str.strip()
            
            # Process dates but keep original if conversion fails
            processed_dates = []
            for date_val in date_col:
                if date_val in ['nan', 'None', '']:
                    processed_dates.append('')
                else:
                    try:
                        parsed_date = pd.to_datetime(date_val, dayfirst=True, errors='coerce')
                        if pd.isna(parsed_date):
                            processed_dates.append(date_val)  # Keep original if can't parse
                        else:
                            processed_dates.append(parsed_date.strftime('%Y-%m-%d %H:%M:%S'))
                    except:
                        processed_dates.append(date_val)  # Keep original if error
            
            processed_data['TRANSACTION DATE'] = processed_dates
        else:
            processed_data['TRANSACTION DATE'] = [''] * len(data_df)
        
        # Value Date - Extract ALL entries
        if 'VALUE DATE' in column_mapping:
            value_date_col = data_df[column_mapping['VALUE DATE']].astype(str)
            
            processed_value_dates = []
            for date_val in value_date_col:
                if date_val in ['nan', 'None', '']:
                    processed_value_dates.append('')
                else:
                    try:
                        parsed_date = pd.to_datetime(date_val, dayfirst=True, errors='coerce')
                        if pd.isna(parsed_date):
                            processed_value_dates.append(date_val)  # Keep original
                        else:
                            processed_value_dates.append(parsed_date.strftime('%Y-%m-%d %H:%M:%S'))
                    except:
                        processed_value_dates.append(date_val)  # Keep original
            
            processed_data['VALUE DATE'] = processed_value_dates
        else:
            processed_data['VALUE DATE'] = [''] * len(data_df)
        
        # Transaction Details - Extract ALL entries (including empty ones)
        if 'TRANSACTION DETAILS' in column_mapping:
            details_col = data_df[column_mapping['TRANSACTION DETAILS']].fillna('').astype(str)
            # Keep ALL entries, even empty ones
            processed_data['TRANSACTION DETAILS'] = details_col.tolist()
        else:
            processed_data['TRANSACTION DETAILS'] = [''] * len(data_df)
        
        # CHQ/REF NO - Extract ALL entries
        if 'CHQ / REF NO.' in column_mapping:
            ref_col = data_df[column_mapping['CHQ / REF NO.']].fillna('').astype(str)
            processed_data['CHQ / REF NO.'] = ref_col.tolist()
        else:
            processed_data['CHQ / REF NO.'] = [''] * len(data_df)
        
        # DEBIT/CREDIT Amount - Extract ALL entries
        if 'DEBIT/CREDIT(₹)' in column_mapping:
            amount_col = data_df[column_mapping['DEBIT/CREDIT(₹)']].astype(str)
            
            def clean_amount_keep_all(amount_str):
                if pd.isna(amount_str) or str(amount_str).strip() in ['', 'nan', 'None']:
                    return 0.0
                
                # Remove currency symbols, commas, and spaces
                cleaned = re.sub(r'[₹,\s]', '', str(amount_str))
                
                # Check if negative (debit)
                is_negative = '-' in cleaned or '(' in cleaned
                
                # Extract numeric value
                numeric_part = re.sub(r'[^\d.]', '', cleaned)
                
                try:
                    amount = float(numeric_part) if numeric_part else 0.0
                    return -amount if is_negative else amount
                except:
                    return 0.0
            
            processed_data['DEBIT/CREDIT(₹)'] = amount_col.apply(clean_amount_keep_all).tolist()
        else:
            processed_data['DEBIT/CREDIT(₹)'] = [0.0] * len(data_df)
        
        # Balance - Extract ALL entries
        if 'BALANCE(₹)' in column_mapping:
            balance_col = data_df[column_mapping['BALANCE(₹)']].astype(str)
            
            def clean_balance_keep_all(balance_str):
                if pd.isna(balance_str) or str(balance_str).strip() in ['', 'nan', 'None']:
                    return 0.0
                
                try:
                    # Remove currency symbols, commas, and spaces
                    cleaned = re.sub(r'[₹,\s]', '', str(balance_str))
                    # Handle negative balances
                    is_negative = '-' in cleaned or '(' in cleaned
                    numeric_part = re.sub(r'[^\d.]', '', cleaned)
                    
                    if numeric_part:
                        amount = float(numeric_part)
                        return -amount if is_negative else amount
                    else:
                        return 0.0
                except:
                    return 0.0
            
            processed_data['BALANCE(₹)'] = balance_col.apply(clean_balance_keep_all).tolist()
        else:
            processed_data['BALANCE(₹)'] = [0.0] * len(data_df)
        
        # Create final DataFrame with ALL data
        processed_df = pd.DataFrame(processed_data)
        
        # NO FILTERING - Keep all rows including empty ones
        # Reset index to ensure clean numbering
        processed_df = processed_df.reset_index(drop=True)
        
        # Ensure proper column order
        column_order = ['#', 'TRANSACTION DATE', 'VALUE DATE', 'TRANSACTION DETAILS', 'CHQ / REF NO.', 'DEBIT/CREDIT(₹)', 'BALANCE(₹)']
        available_columns = [col for col in column_order if col in processed_df.columns]
        processed_df = processed_df[available_columns]
        
        logger.info(f"Successfully extracted ALL {len(processed_df)} rows from Kotak file")
        logger.debug(f"Final columns: {list(processed_df.columns)}")
        
        if len(processed_df) > 0:
            logger.debug(f"Sample data (first 3 rows):\n{processed_df.head(3).to_string()}")
            logger.debug(f"Sample data (last 3 rows):\n{processed_df.tail(3).to_string()}")
        
        # Show summary of data extraction
        logger.info(f"Data extraction summary:")
        logger.info(f"- Total rows extracted: {len(processed_df)}")
        logger.info(f"- Rows with transaction details: {len(processed_df[processed_df['TRANSACTION DETAILS'].astype(str).str.strip() != ''])}")
        logger.info(f"- Rows with empty transaction details: {len(processed_df[processed_df['TRANSACTION DETAILS'].astype(str).str.strip() == ''])}")
        logger.info(f"- Rows with non-zero amounts: {len(processed_df[processed_df['DEBIT/CREDIT(₹)'] != 0])}")
        logger.info(f"- Rows with zero amounts: {len(processed_df[processed_df['DEBIT/CREDIT(₹)'] == 0])}")
        
        return processed_df

    @staticmethod
    def _process_yes_bank_file(df: pd.DataFrame, logger) -> pd.DataFrame:
        """Robust processing for YES Bank statement format"""
        logger.info("Processing YES Bank file with enhanced type handling")
        
        # Convert all data to strings first for consistent processing
        str_df = df.astype(str)
        
        # Expected YES Bank columns with flexible matching
        yes_columns = {
            'transaction date': 'Transaction Date',
            'value date': 'Value date',
            'transaction description': 'Transaction Description', 
            'reference no': 'Reference No',
            'debit amount': 'Debit Amount',
            'credit amount': 'Credit Amount',
            'running balance': 'Running Balance'
        }
        
        # Find header row by column patterns
        header_row_idx = None
        for idx in range(min(20, len(str_df))):  # Check first 20 rows
            row_text = '|'.join(str_df.iloc[idx].fillna('').astype(str)).lower()
            matches = sum(1 for col in yes_columns if col in row_text)
            
            if matches >= 4:  # Found majority of columns
                header_row_idx = idx
                logger.info(f"Found header at row {idx}")
                break
        
        if header_row_idx is None:
            # Fallback: Look for transaction pattern
            for idx in range(len(str_df)):
                if re.search(r'\d{2}[-\/]\d{2}[-\/]\d{4}', str(str_df.iloc[idx, 0])):
                    header_row_idx = idx - 1
                    break
        
        if header_row_idx is None:
            raise ValueError(
                "YES Bank format not recognized. Ensure file contains:\n" +
                "\n".join(f"- {col}" for col in yes_columns.values())
            )
        
        # Extract data table
        data_df = str_df.iloc[header_row_idx+1:].copy()
        data_df.columns = str_df.iloc[header_row_idx].values
        
        # Standardize column names
        data_df.columns = [
            yes_columns.get(col.strip().lower(), col.strip()) 
            for col in data_df.columns
        ]
        
        # Select available columns
        available_cols = [col for col in yes_columns.values() if col in data_df.columns]
        processed_df = data_df[available_cols].copy()
        
        # Convert all data to string and clean
        processed_df = processed_df.applymap(
            lambda x: str(x).strip() if pd.notna(x) else ''
        )
        
        # Remove empty rows
        processed_df = processed_df[
            ~processed_df.iloc[:, 0].isin(['', 'nan', 'None'])
        ]
        
        # Date processing (safe conversion)
        date_cols = [col for col in processed_df.columns if 'date' in col.lower()]
        for col in date_cols:
            processed_df[col] = pd.to_datetime(
                processed_df[col],
                errors='coerce',
                dayfirst=True
            ).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Amount processing (type-safe)
        amount_cols = ['Debit Amount', 'Credit Amount', 'Running Balance']
        for col in amount_cols:
            if col in processed_df.columns:
                # Ensure we're working with strings
                col_data = processed_df[col].astype(str)
                
                # Remove CR/DR markers and non-numeric characters
                clean_values = col_data.str.replace(
                    r'[^\d.]', '', regex=True
                ).replace('', '0')
                
                # Safe numeric conversion
                processed_df[col] = pd.to_numeric(
                    clean_values,
                    errors='coerce'
                ).fillna(0)
        
        logger.info(f"Successfully processed {len(processed_df)} transactions")
        return processed_df
    
    @staticmethod
    def _is_idfc_file(df: pd.DataFrame) -> bool:
        """Check if the file matches IDFC format patterns"""
        str_df = df.astype(str)
        
        # Check for IDFC-specific patterns in first 10 rows
        idfc_indicators = [
            'idfc',
            'bank',
            'transaction',
            'statement',
            'debit',
            'credit',
            'balance'
        ]
        
        text_content = ' '.join(str_df.head(10).values.flatten().tolist()).lower()
        return any(indicator in text_content for indicator in idfc_indicators)
    
      
    @staticmethod
    def _process_idfc_file(df: pd.DataFrame, logger) -> pd.DataFrame:
        """Processing specifically for IDFC bank statements"""
        logger.info("Processing IDFC file structure")
        
        # Convert all data to strings for consistent processing
        str_df = df.astype(str)
        
        # Expected IDFC columns (in various possible naming formats)
        idfc_columns = {
            'transaction date': 'Transaction Date',
            'payment date': 'Payment date',
            'narrative': 'Narrative',
            'customer reference no': 'Customer Reference No',
            'cheque no': 'Cheque No',
            'debit': 'Debit',
            'credit': 'Credit',
            'running balance': 'Running Balance'
        }
        
        # Find header row by looking for column patterns
        header_row_idx = None
        for idx in range(min(20, len(str_df))):  # Check first 20 rows
            row_text = '|'.join(str_df.iloc[idx].fillna('').astype(str)).lower()
            matches = sum(1 for col in idfc_columns if col in row_text)
            
            if matches >= 4:  # Found at least half of expected columns
                header_row_idx = idx
                logger.info(f"Found IDFC header at row {idx}")
                break
        
        if header_row_idx is None:
            raise ValueError(
                "IDFC file format not recognized. "
                "Please ensure the file contains standard IDFC columns: "
                f"{list(idfc_columns.values())}"
            )
        
        # Extract data table
        data_df = str_df.iloc[header_row_idx+1:].copy()
        data_df.columns = str_df.iloc[header_row_idx].values
        
        # Standardize column names
        data_df.columns = [
            idfc_columns.get(col.strip().lower(), col.strip())
            for col in data_df.columns
        ]
        
        # Select only IDFC columns that exist in the file
        available_columns = [col for col in idfc_columns.values() if col in data_df.columns]
        processed_df = data_df[available_columns].copy()
        
        # Remove empty rows
        processed_df = processed_df[
            ~processed_df.iloc[:, 0].str.strip().isin(['', 'nan', 'None'])
        ]
        
        # Date processing
        date_cols = [col for col in processed_df.columns if 'date' in col.lower()]
        for col in date_cols:
            processed_df[col] = pd.to_datetime(
                processed_df[col],
                errors='coerce',
                dayfirst=True
            ).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Amount processing
        amount_cols = ['Debit', 'Credit', 'Running Balance']
        for col in amount_cols:
            if col in processed_df.columns:
                processed_df[col] = (
                    processed_df[col]
                    .str.replace('[^\d.]', '', regex=True)
                    .replace('', '0')
                    .astype(float)
                )
        
        # NEW: Process reference numbers from Narrative
        processed_df = DataProcessor.process_idfc_reference_numbers(processed_df, logger)
        
        logger.info(f"Processed {len(processed_df)} IDFC transactions")
        return processed_df
    @staticmethod
    def format_dates(df: pd.DataFrame, logger) -> pd.DataFrame:
        """Robust date formatting with comprehensive error handling
        
        Args:
            df: Input DataFrame containing date columns to be formatted
            logger: Logger instance for recording processing information
            
        Returns:
            DataFrame with date columns formatted as MySQL-compatible strings (YYYY-MM-DD HH:MM:SS)
        """
        date_related_keywords = ['date', 'time', 'datetime', 'timestamp', 'created', 'modified','processed', 'reversed', 'scheduled']
        
        for col in df.columns:
            col_name_lower = str(col).lower()
            
            # Check if column name suggests it contains dates
            if any(keyword in col_name_lower for keyword in date_related_keywords):
                original_values = df[col].copy()
                str_dates = df[col].astype(str).str.strip()
                
                # Replace various NA representations with None
                str_dates = str_dates.replace(['', 'NA', 'N/A', 'nan', 'None', 'NULL', 'NaT'], None)
                
                # Initialize result with original values
                result = str_dates.copy()
                
                # First handle invalid dates with day=0
                invalid_mask = str_dates.str.contains(r'1900-01-00', na=False)
                if invalid_mask.any():
                    logger.warning(f"Found {invalid_mask.sum()} invalid dates with day=00 in column '{col}'")
                    result[invalid_mask] = '1900-01-01 00:00:00'
                
                # Define format handlers with priority
                format_handlers = [
                    {
                        'name': 'period-separated time',
                        'pattern': r'\d{2}-\d{2}-\d{4} \d{2}\.\d{2}\.\d{2}',
                        'preprocessor': lambda x: re.sub(
                            r'(\d{2})-(\d{2})-(\d{4}) (\d{2})\.(\d{2})\.(\d{2})',
                            r'\3-\2-\1 \4:\5:\6',  # Reformat to YYYY-MM-DD HH:MM:SS
                            x
                        ),
                        'format': '%Y-%m-%d %H:%M:%S'
                    },
                    {
                        'name': 'AM/PM format',
                        'pattern': r'\d{1,2}/\d{1,2}/\d{2,4} \d{1,2}:\d{2}:\d{2} [AP]M',
                        'format': '%d/%m/%Y %I:%M:%S %p'
                    },
                    {
                        'name': 'Excel serial date',
                        'pattern': r'^\d{5}(\.\d+)?$',  # Excel serial dates (e.g., 44256.0)
                        'converter': lambda x: (datetime(1899, 12, 30) + timedelta(days=float(x))).strftime('%Y-%m-%d %H:%M:%S')
                    },
                    {
                        'name': 'ISO8601',
                        'pattern': r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',  # ISO format
                        'format': 'ISO8601'
                    },
                    {
                        'name': 'YYYY-MM-DD',
                        'pattern': r'^\d{4}-\d{2}-\d{2}$',
                        'format': '%Y-%m-%d',
                        'postprocessor': lambda x: f"{x} 00:00:00"
                    },
                    {
                        'name': 'DD/MM/YYYY',
                        'pattern': r'^\d{2}/\d{2}/\d{4}$',
                        'format': '%d/%m/%Y',
                        'postprocessor': lambda x: f"{x} 00:00:00"
                    },
                    {
                        'name': 'MM/DD/YYYY',
                        'pattern': r'^\d{2}/\d{2}/\d{4}$',
                        'format': '%m/%d/%Y',
                        'postprocessor': lambda x: f"{x} 00:00:00"
                    },
                    {
                        'name': 'day-first',
                        'pattern': None,  # Try on all remaining
                        'dayfirst': True
                    },
                    {
                        'name': 'month-first',
                        'pattern': None  # Try on all remaining
                    }
                ]
                
                # Process each format in sequence
                remaining_mask = str_dates.notna()
                for handler in format_handlers:
                    if not remaining_mask.any():
                        break
                    
                    # Apply pattern if specified
                    if handler['pattern']:
                        mask = remaining_mask & str_dates.str.contains(handler['pattern'], regex=True, na=False)
                        if not mask.any():
                            continue
                    else:
                        mask = remaining_mask
                    
                    logger.info(f"Attempting {handler['name']} conversion on {mask.sum()} values in column '{col}'")
                    
                    try:
                        # Preprocess if specified
                        to_convert = str_dates[mask]
                        if 'preprocessor' in handler:
                            to_convert = to_convert.apply(handler['preprocessor'])
                        
                        # Convert using specified method
                        if 'converter' in handler:
                            converted = to_convert.apply(handler['converter'])
                        elif 'format' in handler:
                            if handler['format'] == 'ISO8601':
                                converted = pd.to_datetime(to_convert, format='ISO8601', errors='coerce')
                            else:
                                converted = pd.to_datetime(to_convert, format=handler['format'], errors='coerce')
                        elif 'dayfirst' in handler:
                            converted = pd.to_datetime(to_convert, dayfirst=handler['dayfirst'], errors='coerce')
                        else:
                            converted = pd.to_datetime(to_convert, errors='coerce')
                        
                        # Post-process if specified
                        if 'postprocessor' in handler and converted.notna().any():
                            converted = converted.apply(handler['postprocessor'])
                        
                        # Update successful conversions
                        success_mask = converted.notna()
                        if success_mask.any():
                            result[mask & success_mask] = converted[success_mask]
                            remaining_mask &= ~(mask & success_mask)
                            
                            logger.info(f"Successfully converted {success_mask.sum()} values using {handler['name']}")
                            for val in to_convert[success_mask].head(3):
                                logger.debug(f"  Converted: {val}")
                        
                        # Log failures
                        fail_mask = mask & ~success_mask
                        if fail_mask.any():
                            logger.debug(f"Failed to convert {fail_mask.sum()} values using {handler['name']}")
                            for val in to_convert[~success_mask].head(3):
                                logger.debug(f"  Failed: {val}")
                    
                    except Exception as e:
                        logger.error(f"Error during {handler['name']} conversion: {str(e)}")
                        continue
                
                # Final validation before assignment
                try:
                    # Ensure we're not trying to use .dt accessor on non-datetime values
                    datetime_mask = pd.to_datetime(result, errors='coerce').notna()
                    final_result = result.copy()
                    
                    # Format datetime values
                    final_result[datetime_mask] = pd.to_datetime(result[datetime_mask]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # For any remaining invalid dates, try to salvage them
                    invalid_dates = ~datetime_mask & remaining_mask
                    if invalid_dates.any():
                        logger.warning(f"Attempting to salvage {invalid_dates.sum()} invalid dates in column '{col}'")
                        
                        # Try to extract date parts from malformed dates
                        for idx in invalid_dates[invalid_dates].index:
                            date_str = str(str_dates[idx])
                            
                            # Try to extract YYYY-MM-DD pattern from any string
                            date_match = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})', date_str)
                            time_match = re.search(r'(\d{2})[:\.](\d{2})[:\.](\d{2})', date_str)
                            
                            if date_match:
                                try:
                                    year, month, day = date_match.groups()
                                    time_part = "00:00:00"
                                    
                                    if time_match:
                                        hours, minutes, seconds = time_match.groups()
                                        time_part = f"{hours}:{minutes}:{seconds}"
                                    
                                    reconstructed = f"{year}-{month}-{day} {time_part}"
                                    parsed = pd.to_datetime(reconstructed, errors='coerce')
                                    
                                    if not pd.isna(parsed):
                                        final_result[idx] = parsed.strftime('%Y-%m-%d %H:%M:%S')
                                        datetime_mask[idx] = True
                                        logger.debug(f"Salvaged date: {date_str} -> {final_result[idx]}")
                                except:
                                    pass
                    
                    df[col] = final_result
                    
                    # Log final results
                    success_count = datetime_mask.sum()
                    fail_count = len(df) - success_count
                    logger.info(f"Column '{col}' processing complete. Success: {success_count}, Failed: {fail_count}")
                    
                    if fail_count > 0:
                        sample_fails = str_dates[~datetime_mask].head(3).tolist()
                        logger.warning(f"Failed conversions in '{col}'. Sample: {sample_fails}")
                
                except Exception as e:
                    logger.error(f"Final processing error for column '{col}': {str(e)}")
                    df[col] = original_values  # Revert to original on error
        
        return df

    @staticmethod
    def _convert_date_column(series: pd.Series, logger) -> pd.Series:
        """Robust date conversion for a single column with handling for invalid dates"""
        # Convert to string first
        str_series = series.astype(str).str.strip()
        
        # Initialize result series
        converted = pd.Series(index=series.index, dtype='object')
        
        # First handle invalid dates with day=0
        invalid_mask = str_series.str.contains(r'1900-01-00', na=False)
        if invalid_mask.any():
            logger.warning(f"Found {invalid_mask.sum()} invalid dates with day=00")
            # Replace with first valid day of month
            converted[invalid_mask] = '1900-01-01 00:00:00'
        
        # Process remaining valid dates
        valid_mask = ~invalid_mask & str_series.notna() & (str_series != '')
        if valid_mask.any():
            # Pattern for period-separated time format: "DD-MM-YYYY HH.MM.SS"
            period_time_pattern = r'(\d{2})-(\d{2})-(\d{4}) (\d{2})\.(\d{2})\.(\d{2})'
            
            # Convert period-separated time format first
            period_mask = str_series[valid_mask].str.contains(period_time_pattern, regex=True, na=False)
            if period_mask.any():
                logger.info(f"Processing {period_mask.sum()} period-separated datetime values")
                try:
                    # Replace periods with colons and convert to standard format
                    normalized = str_series[valid_mask][period_mask].str.replace(
                        period_time_pattern,
                        r'\3-\2-\1 \4:\5:\6',  # Reformat to YYYY-MM-DD HH:MM:SS
                        regex=True
                    )
                    converted[valid_mask][period_mask] = pd.to_datetime(normalized, errors='coerce')
                except Exception as e:
                    logger.error(f"Period-time conversion error: {e}")
                    converted[valid_mask][period_mask] = pd.NaT
            
            # Pattern for AM/PM format: "26/06/2025 01:55:50 PM"
            ampm_pattern = r'\d{1,2}/\d{1,2}/\d{2,4} \d{1,2}:\d{2}:\d{2} [AP]M'
            
            # Convert AM/PM format
            ampm_mask = str_series[valid_mask].str.contains(ampm_pattern, regex=True, na=False) & ~period_mask
            if ampm_mask.any():
                logger.info(f"Processing {ampm_mask.sum()} AM/PM datetime values")
                try:
                    converted[valid_mask][ampm_mask] = pd.to_datetime(
                        str_series[valid_mask][ampm_mask],
                        format='%d/%m/%Y %I:%M:%S %p',
                        errors='coerce'
                    )
                except Exception as e:
                    logger.error(f"AM/PM conversion error: {e}")
                    converted[valid_mask][ampm_mask] = pd.NaT
            
            # Process remaining values with standard datetime parsing
            remaining_mask = valid_mask & ~period_mask & ~ampm_mask
            if remaining_mask.any():
                logger.info(f"Processing {remaining_mask.sum()} standard datetime values")
                try:
                    # Try day-first format (common in international formats)
                    converted[remaining_mask] = pd.to_datetime(
                        str_series[remaining_mask],
                        dayfirst=True,
                        errors='coerce'
                    )
                    
                    # For values that failed day-first, try month-first
                    failed_mask = remaining_mask & converted.isna()
                    if failed_mask.any():
                        converted[failed_mask] = pd.to_datetime(
                            str_series[failed_mask],
                            dayfirst=False,
                            errors='coerce'
                        )
                except Exception as e:
                    logger.error(f"Standard datetime conversion error: {e}")
        
        # Format as MySQL datetime string
        formatted = converted.dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # For any remaining unconverted values, keep original but log warning
        still_failed = formatted.isna() & str_series.notna() & (str_series != '')
        if still_failed.any():
            logger.warning(
                f"Failed to convert {still_failed.sum()} datetime values. "
                f"Sample problem values: {str_series[still_failed].head(3).tolist()}"
            )
            formatted[still_failed] = str_series[still_failed]
        
        return formatted
    
    def validate_data_before_upload(df, db_name, table_name):
        """Validate data before upload to catch issues early"""
        if ConfigManager.requires_table_format_extraction(db_name, table_name):
            expected_cols = ConfigManager.get_expected_columns(db_name, table_name)
            missing = set(expected_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Missing expected columns: {missing}")
        
        # Check for empty DataFrame
        if df.empty:
            raise ValueError("No data to upload after processing")

    
    @staticmethod
    def remove_first_column(df: pd.DataFrame) -> pd.DataFrame:
        """Remove the first column from DataFrame"""
        if len(df.columns) > 0:
            df = df.iloc[:, 1:]
        """Convert all date columns to YYYY-MM-DD HH:MM:SS format"""
        date_related_keywords = ['date', 'time', 'datetime', 'timestamp']
        
        for col in df.columns:
            col_name_lower = str(col).lower()
            
            # Check if column name suggests it contains dates
            if any(keyword in col_name_lower for keyword in date_related_keywords):
                try:
                    # Try to parse as datetime
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                    # Format non-null datetime values
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    # Replace NaT with empty string
                    df[col] = df[col].fillna('')
                except Exception as e:
                    continue
            return df
    
    @staticmethod
    def clean_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced numeric processing for all amount columns"""
        amount_keywords = ['amount', 'debit', 'credit', 'balance', '₹', 'value']
        
        for col in df.columns:
            col_name_lower = str(col).lower()
            
            if any(keyword in col_name_lower for keyword in amount_keywords):
                # Skip if already numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    continue
                    
                # Convert to string and clean
                col_data = df[col].astype(str).str.strip()
                
                # Remove common non-numeric characters
                col_data = col_data.str.replace(r'[₹,$€£,\s]', '', regex=True)
                
                # Handle negative numbers in parentheses
                paren_neg_mask = col_data.str.contains(r'\(.*\)', na=False)
                if paren_neg_mask.any():
                    df.loc[paren_neg_mask, col] = (
                        '-' + col_data[paren_neg_mask].str.replace(r'[\(\)]', '', regex=True)
                    )
                    col_data = df[col].astype(str).str.strip()
                
                # Convert to numeric
                try:
                    df[col] = pd.to_numeric(col_data, errors='raise')
                except:
                    # Fallback for mixed content
                    numeric_part = col_data.str.extract(r'([-+]?\d*\.?\d+)', expand=False)
                    df[col] = pd.to_numeric(numeric_part, errors='coerce')
        
        return df
    
    @staticmethod
    def extract_utr_uid(df: pd.DataFrame) -> pd.DataFrame:
        """Extract UTR and UID from narration column for IDFC tables"""
        narration_col = None
        
        # Find narration column (case insensitive)
        for col in df.columns:
            if 'narration' in str(col).lower() or 'narrative' in str(col).lower():
                narration_col = col
                break
        
        if narration_col is None:
            return df
        
        # Create new columns for UTR and UID
        df['UTR'] = df[narration_col].apply(lambda x: DataProcessor._extract_utr(x))
        df['UID'] = df[narration_col].apply(lambda x: DataProcessor._extract_uid(x))
        
        return df
    
    @staticmethod
    def _extract_utr(narration: str) -> str:
        """Extract UTR from narration"""
        if not isinstance(narration, str):
            return ''
        
        # Look for UTR pattern (typically alphanumeric, 12-22 characters)
        utr_pattern = r'(?:UTR|Ref No)[:/]?\s*([A-Za-z0-9]{12,22})'
        match = re.search(utr_pattern, narration, re.IGNORECASE)
        return match.group(1) if match else ''
    
    @staticmethod
    def _extract_uid(narration: str) -> str:
        """Extract UID from narration"""
        if not isinstance(narration, str):
            return ''
        
        # Look for UID pattern (typically alphanumeric, 8-16 characters)
        uid_pattern = r'(?:UID|Customer ID)[:/]?\s*([A-Za-z0-9]{8,16})'
        match = re.search(uid_pattern, narration, re.IGNORECASE)
        return match.group(1) if match else ''

import pandas as pd
import re
from typing import Dict, List, Optional

class DateTimeConverter:
    """Handles conversion of all datetime formats to yyyy-mm-dd hh:mm:ss"""
    
    @staticmethod
    def convert_all_dates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all datetime columns in DataFrame to yyyy-mm-dd hh:mm:ss format.
        Preserves original values when conversion fails.
        """
        date_cols = DateTimeConverter._identify_date_columns(df)
        
        for col in date_cols:
            df[col] = df[col].apply(DateTimeConverter._convert_single_date)
        
        return df
    
    @staticmethod
    def _identify_date_columns(df: pd.DataFrame) -> List[str]:
        """Identify columns that likely contain datetime values"""
        date_cols = []
        date_keywords = ['date', 'time', 'timestamp', 'created', 'modified', 'at', 'processed_at', 'modified','processed_at', 'reversed_at', 'scheduled_at']
        
        for col in df.columns:
            col_lower = str(col).lower()
            # Check if column name suggests it contains dates
            if any(keyword in col_lower for keyword in date_keywords):
                date_cols.append(col)
            else:
                # Check sample values for date-like patterns
                sample = df[col].head(20).dropna()
                if len(sample) > 0 and any(DateTimeConverter._looks_like_date(str(x)) for x in sample):
                    date_cols.append(col)
        
        return date_cols
    
    @staticmethod
    def _looks_like_date(value: str) -> bool:
        """Check if a string looks like a date/time"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
            r'\d{2}/\d{2}/\d{2}',  # DD/MM/YY
            r'\d{4}\.\d{2}\.\d{2}',  # YYYY.MM.DD
            r'\d{1,2}-\d{1,2}-\d{4}',  # M-D-YYYY
            r'\d{1,2}:\d{2}:\d{2}',  # HH:MM:SS
            r'\d{1,2}:\d{2} [AP]M',  # HH:MM AM/PM
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',  # ISO format
            r'\d{2}-\d{2}-\d{4} \d{2}.\d{2}.\d{2}'  # DD-MM-YYYY HH.MM.SS
        ]
        
        return any(re.search(pattern, value) for pattern in date_patterns)
    
    @staticmethod
    def _convert_single_date(value) -> str:
        """Convert a single value to yyyy-mm-dd hh:mm:ss format"""
        if pd.isna(value) or value in ['', 'None', 'nan', 'null']:
            return ''
        
        str_value = str(value).strip()
        
        # Try common format conversions
        try:
            # Handle period-separated times first (e.g., "2025-04-21 23.07.44")
            if re.search(r'\d{4}-\d{2}-\d{2} \d{2}\.\d{2}\.\d{2}', str_value):
                normalized = str_value.replace('.', ':')
                dt = pd.to_datetime(normalized, format='%Y-%m-%d %H:%M:%S')
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Handle AM/PM format (e.g., "26/06/2025 01:55:50 PM")
            if re.search(r'\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M', str_value, re.IGNORECASE):
                dt = pd.to_datetime(str_value, format='%d/%m/%Y %I:%M:%S %p')
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Handle ISO format (e.g., "2025-06-27T15:45:00")
            if 'T' in str_value:
                dt = pd.to_datetime(str_value, format='ISO8601')
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Try day-first format (e.g., "27/06/2025 10:30:00")
            if re.search(r'\d{2}/\d{2}/\d{4}', str_value):
                dt = pd.to_datetime(str_value, dayfirst=True)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Try month-first format (e.g., "06/28/2025 08:00:00")
            if re.search(r'\d{2}/\d{2}/\d{4}', str_value):
                dt = pd.to_datetime(str_value)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Try other common formats
            dt = pd.to_datetime(str_value, infer_datetime_format=True)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
            
        except (ValueError, TypeError):
            # If all conversions fail, return original value
            return str_value

# Example usage:
if __name__ == "__main__":
    # Sample data with various datetime formats
    data = {
        'transaction_date': ['2025-04-21 23.07.44', '26/06/2025 01:55:50 PM', '2025-06-27T15:45:00'],
        'created_at': ['04/21/2025 11.07 PM', '2025-06-26 13.55.50', 'invalid'],
        'processed_at': ['04/21/2025 11.07 PM', '2025-06-26 13.55.50', 'invalid'],
        'processed_at': ['04/21/2025 11.07 PM', '2025-06-26 13.55.50', 'invalid'],
        'reversed_at': ['04/21/2025 11.07 PM', '2025-06-26 13.55.50', 'invalid'],
        'scheduled_at': ['04/21/2025 11.07 PM', '2025-06-26 13.55.50', 'invalid'],
        'other_column': [1, 2, 3]  # Non-date column
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    
    # Convert all datetime columns
    converter = DateTimeConverter()
    converted_df = converter.convert_all_dates(df)
    
    print("\nConverted DataFrame:")
    print(converted_df)

class DatabaseManager:
    """Handles all database operations"""
    def __init__(self, host, username, password, database):
        self.connection_params = {
            'host': host,
            'user': username,
            'password': password,
            'database': database,
            'autocommit': False,
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci'
        }
        self.logger = LoggerManager.setup_logger()

    def create_connection(self):
        """Create a database connection with error handling"""
        try:
            connection = mysql.connector.connect(**self.connection_params)
            return connection
        except mysql.connector.Error as err:
            self.logger.error(f"Database connection error: {err}")
            raise

    def create_database_if_not_exists(self, database_name):
        """Ensure database exists with proper character set"""
        try:
            connection = mysql.connector.connect(
                host=self.connection_params['host'],
                user=self.connection_params['user'],
                password=self.connection_params['password']
            )
            cursor = connection.cursor()
            
            cursor.execute(f"""
                CREATE DATABASE IF NOT EXISTS `{database_name}` 
                CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """)
            
            connection.commit()
            cursor.close()
            connection.close()
            
        except mysql.connector.Error as err:
            self.logger.error(f"Database creation error: {err}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        try:
            connection = self.create_connection()
            cursor = connection.cursor()
            
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = '{self.connection_params['database']}' 
                AND table_name = '{table_name}'
            """)
            
            exists = cursor.fetchone()[0] == 1
            cursor.close()
            connection.close()
            
            return exists
            
        except Exception as e:
            self.logger.error(f"Error checking table existence: {e}")
            raise

    def get_table_columns(self, table_name: str) -> List[str]:
        """Get column names for an existing table"""
        try:
            connection = self.create_connection()
            cursor = connection.cursor()
            
            cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
            columns = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            connection.close()
            
            return columns
            
        except Exception as e:
            self.logger.error(f"Error getting table columns: {e}")
            raise

    def get_primary_key(self, table_name: str) -> Optional[str]:
        """Get primary key column for a table"""
        try:
            connection = self.create_connection()
            cursor = connection.cursor()
            
            cursor.execute(f"""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = '{self.connection_params['database']}' 
                AND TABLE_NAME = '{table_name}' 
                AND COLUMN_KEY = 'PRI'
            """)
            
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            
            return result[0] if result else None
            
        except Exception as e:
            self.logger.error(f"Error getting primary key: {e}")
            raise

    def optimize_for_bulk_load(self, cursor):
        """Optimize MySQL server settings for bulk loading"""
        try:
            cursor.execute("SET autocommit = 0")
            cursor.execute("SET unique_checks = 0")
            cursor.execute("SET foreign_key_checks = 0")
            cursor.execute("SET sql_log_bin = 0")  # If not replicating
            self.logger.info("Optimized MySQL for bulk loading")
        except Exception as e:
            self.logger.warning(f"Could not optimize for bulk load: {e}")

    def restore_after_bulk_load(self, cursor):
        """Restore normal MySQL settings"""
        try:
            cursor.execute("SET autocommit = 1")
            cursor.execute("SET unique_checks = 1")
            cursor.execute("SET foreign_key_checks = 1")
            cursor.execute("SET sql_log_bin = 1")
            self.logger.info("Restored normal MySQL settings")
        except Exception as e:
            self.logger.warning(f"Could not restore settings: {e}")    

    def export_idfc_data_to_excel(self, save_path: str):
        """Export attroidfc.idfc data to Excel file at specified location"""
        try:
            connection = self.create_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Query to get all data from attroidfc.idfc table
            query = "SELECT * FROM idfc"
            cursor.execute(query)
            
            # Fetch all rows
            rows = cursor.fetchall()
            
            if not rows:
                self.logger.warning("No data found in attroidfc.idfc table")
                return False
            
            # Create DataFrame from the data
            df = pd.DataFrame(rows)
            
            # Apply IDFC-specific processing
            df = DataProcessor.extract_utr_uid(df)
            
            # Save to Excel
            df.to_excel(save_path, index=False, engine='openpyxl')
            
            self.logger.info(f"Successfully exported attroidfc.idfc data to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting IDFC data: {e}")
            raise
        finally:
            cursor.close()
            connection.close()

class DataUploader:
    """Handles the complete data upload process"""
    def __init__(self, database_manager):
        self.db_manager = database_manager
        self.logger = LoggerManager.setup_logger()

    def validate_data_before_upload(self, df, db_name, table_name):
        """Validate data before upload to catch issues early"""
        if ConfigManager.requires_table_format_extraction(db_name, table_name):
            expected_cols = ConfigManager.get_expected_columns(db_name, table_name)
            missing = set(expected_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Missing expected columns: {missing}")
        
        # Check for empty DataFrame
        if df.empty:
            raise ValueError("No data to upload after processing")

    class DataUploader:
        """Handles the complete data upload process with performance optimizations"""
        def __init__(self, database_manager):
            self.db_manager = database_manager
            self.logger = LoggerManager.setup_logger()
            self.temp_dir = os.path.join(os.getcwd(), 'temp_uploads')
            os.makedirs(self.temp_dir, exist_ok=True)

        def upload_data_file(self, file_path, db_name, table_name):
            """
            Optimized main file upload method using fastest available approach
            """
            try:
                start_time = time.time()
                
                # Read and process file
                df = self._read_and_process_file(file_path, db_name, table_name)
                
                # Validate before upload
                self.validate_data_before_upload(df, db_name, table_name)
                
                # Choose fastest upload method based on data size
                if len(df) > 100000:  # For very large files
                    success = self._upload_with_load_data_infile(df, db_name, table_name)
                else:
                    success = self._upload_with_bulk_insert(df, db_name, table_name)
                
                if success:
                    elapsed = time.time() - start_time
                    self.logger.info(f"✅ Upload completed in {elapsed:.2f} seconds")
                    self.logger.info(f"   Rows: {len(df)} | Speed: {len(df)/max(elapsed, 0.1):.0f} rows/sec")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Upload failed: {e}", exc_info=True)
                raise

        def _read_and_process_file(self, file_path, db_name, table_name):
            """Memory-efficient file reading and processing"""
            # Use chunks for very large files
            if os.path.getsize(file_path) > 50 * 1024 * 1024:  # >50MB
                chunks = []
                for chunk in pd.read_excel(file_path, chunksize=100000, engine='openpyxl'):
                    processed = DataProcessor.process_data(chunk, db_name, table_name, self.logger)
                    chunks.append(processed)
                return pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
                return DataProcessor.process_data(df, db_name, table_name, self.logger)

        def _upload_with_load_data_infile(self, df, db_name, table_name):
            """
            Fastest upload method - uses MySQL's LOAD DATA INFILE
            Requires FILE privilege and writes temporary file
            """
            self.logger.info("Using LOAD DATA INFILE (fastest method)")
            
            # Create temp CSV file
            temp_file = os.path.join(self.temp_dir, f"upload_{int(time.time())}.csv")
            try:
                # Write CSV without headers and with proper escaping
                df.to_csv(temp_file, index=False, header=False, 
                        quoting=csv.QUOTE_MINIMAL, escapechar='\\')
                
                connection = self.db_manager.create_connection()
                cursor = connection.cursor()
                
                # Disable indexes for faster upload
                self._disable_indexes(cursor, table_name)
                
                # Build LOAD DATA INFILE command
                columns = ','.join([f'`{col}`' for col in df.columns])
                load_cmd = f"""
                LOAD DATA LOCAL INFILE '{temp_file}'
                INTO TABLE `{table_name}`
                FIELDS TERMINATED BY ',' 
                OPTIONALLY ENCLOSED BY '"'
                ESCAPED BY '\\\\'
                LINES TERMINATED BY '\\n'
                ({columns})
                """
                
                # Execute with timing
                start = time.time()
                cursor.execute(load_cmd)
                connection.commit()
                elapsed = time.time() - start
                
                self.logger.info(f"LOAD DATA INFILE completed in {elapsed:.2f} seconds")
                
                # Re-enable indexes
                self._enable_indexes(cursor, table_name)
                connection.commit()
                
                return True
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        def _upload_with_bulk_insert(self, df, db_name, table_name):
            """Optimized bulk insert method for medium-sized datasets"""
            self.logger.info("Using optimized bulk insert")
            
            connection = self.db_manager.create_connection()
            cursor = connection.cursor()
            
            try:
                # Prepare query
                columns = ','.join([f'`{col}`' for col in df.columns])
                placeholders = ','.join(['%s'] * len(df.columns))
                query = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"
                
                # Convert DataFrame to list of tuples
                data = [tuple(x) for x in df.to_numpy()]
                
                # Batch settings
                batch_size = min(5000, max(1000, len(df) // 10))  # Dynamic batch size
                total_rows = len(data)
                inserted = 0
                
                # Disable indexes for faster upload
                self._disable_indexes(cursor, table_name)
                
                # Insert in batches with timing
                start_time = time.time()
                for i in range(0, total_rows, batch_size):
                    batch = data[i:i + batch_size]
                    
                    try:
                        cursor.executemany(query, batch)
                        connection.commit()
                        inserted += len(batch)
                        
                        # Progress logging
                        if i % (batch_size * 10) == 0:  # Log every 10 batches
                            elapsed = time.time() - start_time
                            remaining = (total_rows - inserted) / (inserted / max(elapsed, 0.1))
                            self.logger.info(
                                f"Progress: {inserted}/{total_rows} "
                                f"({inserted/total_rows:.1%}) | "
                                f"Est. remaining: {remaining:.1f}s"
                            )
                    
                    except Exception as e:
                        self.logger.error(f"Batch error: {e}")
                        connection.rollback()
                        # Fallback to single-row inserts for failed batch
                        for row in batch:
                            try:
                                cursor.execute(query, row)
                                connection.commit()
                                inserted += 1
                            except:
                                connection.rollback()
                
                # Re-enable indexes
                self._enable_indexes(cursor, table_name)
                connection.commit()
                
                return True
                
            finally:
                connection.close()

        def _disable_indexes(self, cursor, table_name):
            """Disable non-unique indexes for faster upload"""
            try:
                cursor.execute(f"ALTER TABLE `{table_name}` DISABLE KEYS")
                self.logger.info("Disabled non-unique indexes for faster upload")
            except Exception as e:
                self.logger.warning(f"Could not disable indexes: {e}")

        def _enable_indexes(self, cursor, table_name):
            """Re-enable indexes after upload"""
            try:
                cursor.execute(f"ALTER TABLE `{table_name}` ENABLE KEYS")
                self.logger.info("Re-enabled indexes")
            except Exception as e:
                self.logger.warning(f"Could not enable indexes: {e}")

        def _prepare_for_fast_upload(self, cursor, table_name, df):
            """
            Prepare table for fast upload by:
            1. Temporarily increasing buffer sizes
            2. Disabling foreign key checks
            3. Setting autocommit off
            """
            try:
                # Increase buffer sizes
                cursor.execute("SET GLOBAL bulk_insert_buffer_size = 1024 * 1024 * 256")  # 256MB
                cursor.execute("SET GLOBAL max_allowed_packet = 1024 * 1024 * 64")  # 64MB
                
                # Disable foreign key checks
                cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                cursor.execute("SET UNIQUE_CHECKS = 0")
                
                # Set transaction isolation
                cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")
                
                self.logger.info("Optimized MySQL settings for fast upload")
                
            except Exception as e:
                self.logger.warning(f"Could not optimize MySQL settings: {e}")

        def _restore_after_upload(self, cursor):
            """Restore original MySQL settings"""
            try:
                cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
                cursor.execute("SET UNIQUE_CHECKS = 1")
                self.logger.info("Restored original MySQL settings")
            except Exception as e:
                self.logger.warning(f"Could not restore MySQL settings: {e}")

    def _create_table(self, cursor, table_name, df):
        """Create table with columns matching DataFrame structure"""
        column_definitions = []
        
        for column in df.columns:
            col_data = df[column]
            mysql_type = self._determine_mysql_type(col_data)
            column_definitions.append(f"`{column}` {mysql_type}")
        
        columns_sql = ",\n            ".join(column_definitions)
        
        create_table_query = f"""
        CREATE TABLE `{table_name}` (
            {columns_sql}
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        cursor.execute(create_table_query)
        self.logger.info(f"Table created: {table_name} with {len(df.columns)} columns")

    def _determine_mysql_type(self, series):
        """Determine appropriate MySQL data type for pandas Series"""
        # Convert empty strings to None for type detection
        series = series.replace('', None)
        
        # Try to detect numeric types first
        try:
            numeric_series = pd.to_numeric(series.dropna())
            if numeric_series.empty:
                return "TEXT"
            
            # Check if it's integer type
            if all(numeric_series == numeric_series.astype(int)):
                max_val = numeric_series.max()
                min_val = numeric_series.min()
                
                if min_val >= -128 and max_val <= 127:
                    return "TINYINT"
                elif min_val >= -32768 and max_val <= 32767:
                    return "SMALLINT"
                elif min_val >= -8388608 and max_val <= 8388607:
                    return "MEDIUMINT"
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    return "INT"
                else:
                    return "BIGINT"  # Default to BIGINT for large integers
            else:
                return "DOUBLE"
        except:
            pass
        
        # Try to detect datetime
        try:
            datetime_series = pd.to_datetime(series.dropna(), errors='raise')
            if not datetime_series.empty:
                return "DATETIME"
        except:
            pass
        
        # For text data, analyze length
        max_length = series.astype(str).str.len().max()
        if pd.isna(max_length) or max_length <= 255:
            return "VARCHAR(255)"
        elif max_length <= 65535:
            return "TEXT"
        elif max_length <= 16777215:
            return "MEDIUMTEXT"
        else:
            return "LONGTEXT"

    def _insert_data(self, connection, cursor, table_name, df):
        """Insert new data into MySQL table"""
        placeholders = ", ".join(["%s"] * len(df.columns))
        column_names = ", ".join([f"`{col}`" for col in df.columns])
        insert_query = f"INSERT INTO `{table_name}` ({column_names}) VALUES ({placeholders})"
        
        batch_size = 500
        batch = []
        rows_processed = 0
        total_rows = len(df)
        
        self.logger.info(f"Starting insert of {total_rows} rows")
        
        for _, row in df.iterrows():
            row_values = []
            for val in row:
                if pd.isna(val) or val == '':
                    row_values.append(None)
                else:
                    row_values.append(str(val))
            
            batch.append(tuple(row_values))
            rows_processed += 1
            
            if len(batch) >= batch_size:
                try:
                    cursor.executemany(insert_query, batch)
                    connection.commit()
                    self.logger.info(f"Inserted batch: {rows_processed}/{total_rows} rows")
                    batch = []
                except Exception as e:
                    self.logger.error(f"Batch insert error: {e}")
                    connection.rollback()
                    # Fallback to individual inserts
                    for single_row in batch:
                        try:
                            cursor.execute(insert_query, single_row)
                            connection.commit()
                        except Exception as e:
                            self.logger.warning(f"Skipped problematic row: {e}")
                            connection.rollback()
                    batch = []
        
        # Insert remaining batch
        if batch:
            try:
                cursor.executemany(insert_query, batch)
                connection.commit()
                self.logger.info(f"Final batch inserted: {rows_processed}/{total_rows} rows")
            except Exception as e:
                self.logger.error(f"Final batch error: {e}")
                connection.rollback()
                # Fallback to individual inserts
                for single_row in batch:
                    try:
                        cursor.execute(insert_query, single_row)
                        connection.commit()
                    except Exception as e:
                        self.logger.warning(f"Skipped problematic final row: {e}")
                        connection.rollback()
        
       
        self.logger.info(f"✅ COMPLETE DATA INSERT: {rows_processed}")



    def _upsert_data(self, connection, cursor, table_name, df, primary_key=None):
        """Optimized upsert implementation with duplicate checking"""
        if not primary_key:
            return self._insert_data(connection, cursor, table_name, df)
        
        # First check for existing records
        self.log_status("Checking for duplicate records...")
        
        # Get all existing primary key values
        cursor.execute(f"SELECT {primary_key} FROM `{table_name}`")
        existing_ids = {row[0] for row in cursor.fetchall()}
        
        # Filter out rows that already exist
        new_rows = df[~df[primary_key].isin(existing_ids)]
        
        if len(new_rows) == 0:
            self.log_status("All records already exist - no new data to insert")
            return
        
        self.log_status(f"Found {len(new_rows)} new records (out of {len(df)})")
        
        # Prepare data for insert
        columns = [f"`{col}`" for col in new_rows.columns]
        placeholders = ["%s"] * len(new_rows.columns)
        
        query = f"""
        INSERT INTO `{table_name}` ({",".join(columns)})
        VALUES ({",".join(placeholders)})
        """
        
        # Convert to list of tuples
        data = [tuple(x) for x in new_rows.to_numpy()]
        
        # Batch settings
        batch_size = min(2000, max(500, len(new_rows) // 20))
        total_rows = len(data)
        processed = 0
        
        # Optimize for bulk load
        self.db_manager.optimize_for_bulk_load(cursor)
        self._disable_indexes(cursor, table_name)
        
        start_time = time.time()
        
        try:
            for i in range(0, total_rows, batch_size):
                batch = data[i:i + batch_size]
                cursor.executemany(query, batch)
                connection.commit()
                processed += len(batch)
                
                # Progress reporting
                if i % (batch_size * 10) == 0:
                    elapsed = time.time() - start_time
                    rate = processed / max(elapsed, 0.1)
                    remaining = (total_rows - processed) / rate
                    self.log_status(
                        f"Insert progress: {processed}/{total_rows} "
                        f"({processed/total_rows:.1%}) | "
                        f"Rate: {rate:.0f} rows/sec | "
                        f"Remaining: {remaining:.1f}s"
                    )
                    
        finally:
            self._enable_indexes(cursor, table_name)
            self.db_manager.restore_after_bulk_load(cursor)
            connection.commit()

    def _offer_idfc_export(self):
        """Offer to export attroidfc.idfc data after upload"""
        response = messagebox.askyesno(
            "Export IDFC Data",
            "Would you like to export the attroidfc.idfc data to an Excel file now?"
        )
        
        if response:
            # Get save location from user
            save_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                title="Save IDFC Data As"
            )
            
            if save_path:
                try:
                    success = self.db_manager.export_idfc_data_to_excel(save_path)
                    if success:
                        messagebox.showinfo(
                            "Export Successful",
                            f"attroidfc.idfc data successfully exported to:\n{save_path}"
                        )
                except Exception as e:
                    messagebox.showerror(
                        "Export Failed",
                        f"Failed to export IDFC data: {str(e)}"
                    )

class MySQLExcelUploaderGUI:


    """Main GUI application for MySQL Excel Uploader"""
    def __init__(self, root):
        self.root = root
        self.root.title("MySQL Excel Data Uploader")
        self.root.geometry("900x700")
        
        # Hide the main window IMMEDIATELY - before any other operations
        # self.root.withdraw()
        
        # Initialize logger first
        self.logger = LoggerManager.setup_logger()
        self.logger.info("Application starting...")
        
        # Setup UI (but don't pack/show it yet)
        self.setup_ui()
        
        # Set close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Start with password check after a short delay to ensure window is hidden
        self.root.after(100, self.check_password)

    def setup_ui(self):
        """Create comprehensive UI - only shown after successful authentication"""
        # Main window is already hidden in __init__, no need to hide again
        
        style = ttk.Style()
        style.theme_use('default')

        # Modern style configurations
        style.configure("TFrame", background="#f5f7fa")
        style.configure("TLabelFrame", background="#ffffff", borderwidth=2, relief="solid", padding=10)
        style.configure("TLabelFrame.Label", background="#ffffff", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#ffffff", font=("Segoe UI", 10))
        style.configure("TEntry", font=("Segoe UI", 10), padding=8, fieldbackground="#f8f9fa")
        style.configure("TCombobox", padding=5)
        
        # Button styles
        style.configure("Primary.TButton", 
                    background="#4CAF50", 
                    foreground="white",
                    font=("Segoe UI", 10, "bold"),
                    padding=8,
                    borderwidth=0)
        style.map("Primary.TButton",
                background=[("active", "#45a049"), ("pressed", "#388e3c")])
        
        style.configure("Secondary.TButton",
                    background="#3498db",
                    foreground="white",
                    font=("Segoe UI", 10, "bold"),
                    padding=8,
                    borderwidth=0)
        style.map("Secondary.TButton",
                background=[("active", "#2980b9"), ("pressed", "#2472a4")])
        
        style.configure("TButton", font=("Segoe UI", 9))

        # Main frame with modern background
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill='both', expand=True)

        # Connection Frame with card-like appearance
        connection_frame = ttk.LabelFrame(main_frame, text="Database Connection", padding="15")
        connection_frame.pack(fill='x', pady=(0, 15), ipadx=5, ipady=5)

        # Connection Fields with improved layout
        fields = [
            ("Host:", "192.168.1.171", "-", 0),
            ("Username:", "SRA", "%", 1),
            ("Password:", "123", "*", 2),
            ("Database.Table:", ConfigManager.get_database_options()[0], None, 3)
        ]

        for label_text, default_value, show_char, row in fields:
            ttk.Label(connection_frame, text=label_text).grid(row=row, column=0, sticky='w', padx=5, pady=5)
            
            if label_text == "Database.Table:":
                self.db_table_var = tk.StringVar(value=default_value)
                db_table_dropdown = ttk.Combobox(
                    connection_frame, 
                    textvariable=self.db_table_var, 
                    values=ConfigManager.get_database_options(),
                    state="readonly"
                )
                db_table_dropdown.grid(row=row, column=1, sticky='ew', padx=5, pady=5)
            else:
                entry = ttk.Entry(connection_frame, show=show_char)
                entry.insert(0, default_value)
                entry.grid(row=row, column=1, sticky='ew', padx=5, pady=5)
                
                # Store references to the entry widgets
                if label_text == "Host:":
                    self.host_entry = entry
                elif label_text == "Username:":
                    self.username_entry = entry
                elif label_text == "Password:":
                    self.password_entry = entry

        # Instructions display with better styling
        self.instructions_var = tk.StringVar(value="Select a database.table to see instructions")
        ttk.Label(connection_frame, text="Instructions:").grid(row=4, column=0, sticky='nw', padx=5, pady=5)
        instructions_label = ttk.Label(
            connection_frame, 
            textvariable=self.instructions_var,
            wraplength=400,
            foreground="#2c3e50",
            font=("Segoe UI", 9)
        )
        instructions_label.grid(row=4, column=1, sticky='w', padx=5, pady=5)

        # Configure grid weights
        connection_frame.columnconfigure(1, weight=1)

        # Update instructions when selection changes
        self.db_table_var.trace_add('write', self.update_instructions)

        # File Upload Frame with card-like appearance
        upload_frame = ttk.LabelFrame(main_frame, text="File Upload", padding="15")
        upload_frame.pack(fill='x', pady=(0, 15), ipadx=5, ipady=5)

        # File selection with improved layout
        ttk.Label(upload_frame, text="Select File:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(upload_frame, textvariable=self.file_path_var, state='readonly')
        self.file_path_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        browse_btn = ttk.Button(upload_frame, text="Browse", command=self.browse_file)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)

        # Action buttons with modern styling
        button_frame = ttk.Frame(upload_frame)
        button_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(5, 0))
        
        upload_btn = ttk.Button(
            button_frame,
            text="Upload Excel File",
            command=self.initiate_upload,
            style="Primary.TButton"
        )
        upload_btn.pack(side='left', expand=True, fill='x', padx=2)

        generate_pdf_btn = ttk.Button(
            button_frame,
            text="Generate PDF Report",
            command=self.generate_pdf_button_handler,
            style="Secondary.TButton"
        )
        generate_pdf_btn.pack(side='left', expand=True, fill='x', padx=2)

        # Configure grid weights
        upload_frame.columnconfigure(1, weight=1)

        # Status Frame with card-like appearance
        status_frame = ttk.LabelFrame(main_frame, text="Upload Status", padding="15")
        status_frame.pack(fill='both', expand=True, ipadx=5, ipady=5)

        # Status Text with Scrollbar and modern styling
        text_frame = ttk.Frame(status_frame)
        text_frame.pack(fill='both', expand=True)

        self.status_text = tk.Text(
            text_frame, 
            wrap='word',
            font=('Consolas', 10),
            padx=10,
            pady=10,
            bg="#f8f9fa",
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="#dfe6e9"
        )
        
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)

        self.status_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Clear button with modern styling
        clear_btn = ttk.Button(
            status_frame, 
            text="Clear Log", 
            command=self.clear_log,
            style="TButton"
        )
        clear_btn.pack(pady=(10, 5))

    
    
    
    def on_close(self):
        """Handle application close event"""
        self.logger.info("Application closing")
        self.root.destroy()    

    def check_password(self):
        """Show password dialog with enhanced UI and security"""
        import hashlib
        from tkinter import ttk, messagebox
        import tkinter as tk

        PASSWORD_HASH = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"
        MAX_ATTEMPTS = 3
        self.attempts = 0

        # Create password dialog
        password_dialog = tk.Toplevel(self.root)
        password_dialog.title("🔒 Secure Login")
        password_dialog.geometry("400x300")
        password_dialog.configure(bg="#f5f7fa")
        password_dialog.resizable(False, False)

        # Make it modal and ensure it's on top
        password_dialog.transient(self.root)
        password_dialog.grab_set()
        password_dialog.focus_force()
        password_dialog.lift()
        password_dialog.attributes('-topmost', True)

        # Center the dialog
        password_dialog.update_idletasks()
        x = (password_dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (password_dialog.winfo_screenheight() // 2) - (300 // 2)
        password_dialog.geometry(f"+{x}+{y}")

        # Handle dialog close - exit app if password dialog is closed
        def on_dialog_close():
            self.logger.info("Password dialog closed - exiting application")
            self.root.quit()
        
        password_dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)

        # Custom styles
        style = ttk.Style()
        style.theme_use("clam")
        
        # Card frame for content
        style.configure("AuthCard.TFrame", background="white", borderwidth=0, 
                    relief="solid", bordercolor="#e0e0e0")
        main_frame = ttk.Frame(password_dialog, style="AuthCard.TFrame", padding=25)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Style configurations
        style.configure("AuthTitle.TLabel", background="white", 
                    font=("Segoe UI", 14, "bold"), foreground="#2c3e50")
        style.configure("AuthText.TLabel", background="white", 
                    font=("Segoe UI", 10), foreground="#7f8c8d")
        style.configure("AuthEntry.TEntry", font=("Segoe UI", 10), padding=10, 
                    fieldbackground="#f8f9fa", bordercolor="#dfe6e9", 
                    relief="solid", borderwidth=1)
        style.configure("AuthButton.TButton", font=("Segoe UI", 10, "bold"), 
                    padding=10, background="#3498db", foreground="white", 
                    borderwidth=0)
        style.map("AuthButton.TButton",
                background=[("active", "#2980b9"), ("pressed", "#2472a4")])
        style.configure("AuthError.TEntry", bordercolor="#e74c3c", 
                    fieldbackground="#fdedec")

        # Widgets
        ttk.Label(main_frame, text="🔐 Admin Authentication", style="AuthTitle.TLabel").pack()
        
        ttk.Label(main_frame, text="Please enter your admin password to continue", 
                style="AuthText.TLabel").pack(pady=(0, 20))
        
        password_var = tk.StringVar()
        password_entry = ttk.Entry(main_frame, textvariable=password_var, 
                                show="•", style="AuthEntry.TEntry")
        password_entry.pack(fill="x", pady=(0, 10))
        password_entry.focus_set()
        
        status_label = ttk.Label(main_frame, text="", foreground="#e74c3c", 
                                font=("Segoe UI", 9), anchor="center")
        status_label.pack(fill="x", pady=5)

        # Password visibility toggle
        def toggle_password():
            if password_entry['show'] == '':
                password_entry.config(show='•')
                eye_button.config(text='👁')
            else:
                password_entry.config(show='')
                eye_button.config(text='🔒')
        
        eye_button = tk.Button(main_frame, text="👁", command=toggle_password, 
                            bg="white", bd=0, activebackground="white", 
                            cursor="hand2", font=("Segoe UI", 10))
        eye_button.place(in_=password_entry, relx=1.0, x=-35, y=2, height=32)

        def verify_password():
            self.attempts += 1
            input_hash = hashlib.sha256(password_var.get().encode()).hexdigest()

            if input_hash == PASSWORD_HASH:
                password_dialog.destroy()
                self.root.deiconify()  # Show the main window
                self.root.lift()       # Bring to front
                self.root.focus_force() # Give focus
                self.logger.info("✅ Successful password authentication")
            else:
                remaining = MAX_ATTEMPTS - self.attempts
                if remaining > 0:
                    status_label.config(text=f"❌ Incorrect password, {remaining} attempt(s) remaining")
                    password_var.set("")
                    password_entry.config(style="AuthError.TEntry")
                    password_entry.focus_set()  # Return focus to entry
                    self.logger.warning(f"⚠️ Failed password attempt {self.attempts}/{MAX_ATTEMPTS}")
                else:
                    self.logger.error("❌ Max password attempts reached. Exiting.")
                    messagebox.showerror("Access Denied", 
                                        "Maximum attempts reached. Application will now close.", 
                                        parent=password_dialog)
                    self.root.quit()

        submit_btn = ttk.Button(main_frame, text="Unlock System", 
                            command=verify_password, style="AuthButton.TButton")
        submit_btn.pack(fill="x", pady=(15, 0))
        
        # Footer note
        ttk.Label(main_frame, text="For authorized personnel only", 
                font=("Segoe UI", 8), foreground="#bdc3c7").pack(side="bottom", pady=(10, 0))

        # Bind Enter key to verify password
        password_dialog.bind('<Return>', lambda e: verify_password())
        
        # Ensure dialog stays on top
        password_dialog.after(100, lambda: password_dialog.attributes('-topmost', True))

    def update_instructions(self, *args):
        """Update instructions when database.table selection changes"""
        db_table = self.db_table_var.get()
        if '.' in db_table:
            db_name, table_name = db_table.split('.', 1)
            instructions = ConfigManager.get_table_instructions(db_name, table_name)
            self.instructions_var.set(instructions)

    def generate_pdf_button_handler(self):
        """Handle Generate PDF button click"""
        try:
            # Get current database and table selection
            db_table = self.db_table_var.get()
            if '.' not in db_table:
                messagebox.showerror("Error", "Invalid database.table selection")
                return
            database, table = db_table.split('.', 1)

            # Get current file paths
            file_paths = self.file_path_var.get()
            file_list = file_paths.split("; ") if file_paths else []

            # Get current status log
            log_lines = self.status_text.get("1.0", tk.END).strip().splitlines()

            # Estimate total uploaded rows from last log line (optional logic)
            total_rows = 0
            for line in reversed(log_lines):
                if "Total rows uploaded:" in line:
                    try:
                        total_rows = int(line.split(":")[1].strip())
                    except:
                        pass
                    break

            # Generate PDF
            self.generate_pdf_report(database, table, file_list, total_rows, log_lines)

        except Exception as e:
            self.log_status(f"⚠️ Error generating PDF from button: {e}")
            messagebox.showerror("PDF Generation Failed", str(e))


    @classmethod
    def get_table_file_location(cls, db_name: str, table_name: str) -> str:
        """Get predefined file location for a database table"""
        if not db_name or not table_name:
            return os.getcwd()
        
        locations = {
            # 'fbpayripe.payin': r"\\server\F DRIVE DESKTOP\Parties Record\Fbpayripe\FY 2025-26\01 Statements and Records\01 Payin Report",
            # 'fbpayripe.payout': r"\\server\F DRIVE DESKTOP\Parties Record\Fbpayripe\FY 2025-26\01 Statements and Records\02 Payout Report",
            # 'fbpayripe.wallet': r"\\server\F DRIVE DESKTOP\Parties Record\Fbpayripe\FY 2025-26\01 Statements and Records\03 Wallet Report",
            
            # 'anateck.payout': r"\\server\F DRIVE DESKTOP\Parties Record\Anateck\FY 2025-26\01 Statements and Records\01 Payout Report",
            # 'anateck.wallet': r"\\server\F DRIVE DESKTOP\Parties Record\Anateck\FY 2025-26\01 Statements and Records\02 Wallet Report",
            # 'anateck.payon_payout': r"\\server\F DRIVE DESKTOP\Parties Record\Anateck\FY 2025-26\01 Statements and Records\03 Payon Payout Report",
            # 'anateck.payon_owallet': r"\\server\F DRIVE DESKTOP\Parties Record\Anateck\FY 2025-26\01 Statements and Records\04 Payon Wallet Report",
            # 'anateck.idfc': r"\\server\F DRIVE DESKTOP\Parties Record\Anateck\FY 2025-26\01 Statements and Records\05 IDFC Bank Statements",
            # 'anateck.yes': r"\\server\F DRIVE DESKTOP\Parties Record\Anateck\FY 2025-26\01 Statements and Records\06 Yes Bank Statements",
            # 'anateck.kotak': r"\\server\F DRIVE DESKTOP\Parties Record\Anateck\FY 2025-26\01 Statements and Records\07 Kotak Bank Statements",
            
            # 'attroidfc.payout': r"\\server\F DRIVE DESKTOP\Parties Record\Attroidfc\FY 2025-26\01 Statements and Records\01 Payout Report",
            # 'attroidfc.wallet': r"\\server\F DRIVE DESKTOP\Parties Record\Attroidfc\FY 2025-26\01 Statements and Records\02 Wallet Report",
            # 'attroidfc.payon_payout': r"\\server\F DRIVE DESKTOP\Parties Record\Attroidfc\FY 2025-26\01 Statements and Records\03 Payon Payout Report",
            # 'attroidfc.payon_owallet': r"\\server\F DRIVE DESKTOP\Parties Record\Attroidfc\FY 2025-26\01 Statements and Records\04 Payon Wallet Report",
            # 'attroidfc.idfc': r"\\server\F DRIVE DESKTOP\Parties Record\Attroidfc\FY 2025-26\01 Statements and Records\05 IDFC Bank Statements",
            # 'attroidfc.yes': r"\\server\F DRIVE DESKTOP\Parties Record\Attroidfc\FY 2025-26\01 Statements and Records\06 Yes Bank Statements",
            # 'attroidfc.kotak': r"\\server\F DRIVE DESKTOP\Parties Record\Attroidfc\FY 2025-26\01 Statements and Records\07 Kotak Bank Statements",
            
            'vibepay_Portal.vibepay_payin_report': r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\01 Vibepay Payin Data\Payin Report",
            'vibepay_Portal.vibepay_payin_wallet': r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\01 Vibepay Payin Data\Payin Wallet",
            'vibepay_Portal.vibepay_payout_report': r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\02 Vibepay Payout Data\Payout Report (Daily)",
            'vibepay_Portal.vibepay_payout_wallet': r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\02 Vibepay Payout Data\Payout Wallet",
            'vibepay_Portal.vibepay_add_fund_report': r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\05 Vibepay Add Fund Report",
            'vibepay_Portal.vibepayin_add_fund_report': r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\01 Vibepay Payin Data\Add Fund Report",
            
            'sabpaisa.settel_hai': r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\03 Subpaisa Payin Report\Transaction Report",
            'sabpaisa.payin': r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\03 Subpaisa Payin Report\Transaction Report",
            'sabpaisa.settelment_report': r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\03 Subpaisa Payin Report\Settlement Report",

            'rozarpayx.payout_report':  r"\\Server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\08 RazorPayX\Reports",
            'rozarpayx.transaction_report':  r"\\Server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\08 RazorPayX\Reports",
            
            'swiftsend_portal.swiftsend_payout_report': r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\Payout Report (Daily)",
            'swiftsend_portal.swiftsend_payout_wallet': r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\Payout Wallet",
            'swiftsend_portal.swiftsend_finzen_payout': r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\07 Finzen\Payout Statement",
            'swiftsend_portal.swiftsend_add_fund_report': r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\Add fund report",            
            'payonetic_portal.payonetic_payout_report': r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\05 Payonetic\Payout Report (Daily)",
            'payonetic_portal.payonetic_payout_wallet': r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\05 Payonetic\Wallet",
            'payonetic_portal.payonetic_add_fund_report': r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\05 Payonetic\Add fund"
        }
        
        return locations.get(f"{db_name}.{table_name}", os.getcwd())        


    def browse_file(self):
        """File selection dialog for multiple files with predefined locations"""
        try:
            # Get the current selected database and table
            db_table = self.db_table_var.get()
            if '.' not in db_table:
                messagebox.showerror("Error", "Please select a valid database.table first")
                return
                
            db_name, table_name = db_table.split('.', 1)
            
            # Get the predefined location for this database.table combination
            initial_dir = self.get_table_file_location(db_name, table_name)
            
            # Check if the directory exists, if not use default
            if not os.path.exists(initial_dir):
                initial_dir = os.getcwd()  # Use current working directory as fallback
                self.log_status(f"⚠️ Predefined location not found, using current directory")
                self.log_status(f"Expected location was: {initial_dir}")
            
            self.log_status(f"Opening file browser at: {initial_dir}")
            
            file_paths = filedialog.askopenfilenames(
                title=f"Select Files for {db_name}.{table_name}",
                initialdir=initial_dir,  # Set the initial directory
                filetypes=[
                    ("Excel Files", "*.xlsx *.xls"),
                    ("CSV Files", "*.csv"),
                    ("All Files", "*.*")
                ]
            )
            
            if file_paths:
                self.file_path_var.set("; ".join(file_paths))
                self.log_status(f"Selected {len(file_paths)} files")
                
        except Exception as e:
            self.log_status(f"Error in file browser: {e}")
            messagebox.showerror("File Browser Error", str(e))



    def initiate_upload(self):
        """Initiate file upload process (Excel or CSV)"""
        try:
            # Validate inputs
            file_paths = self.file_path_var.get()
            if not file_paths:
                messagebox.showerror("Error", "Please select files to upload")
                return

            # Split multiple file paths
            file_list = file_paths.split("; ") if "; " in file_paths else [file_paths]
            
            # Validate files exist
            for file_path in file_list:
                if not os.path.exists(file_path):
                    messagebox.showerror("Error", f"File does not exist: {file_path}")
                    return

                # Validate file types
                if not (file_path.lower().endswith(('.xlsx', '.xls', '.csv'))):
                    messagebox.showerror("Error", f"Invalid file type: {file_path}\nPlease select Excel (.xlsx, .xls) or CSV (.csv) files")
                    return

            # Get connection details
            host = self.host_entry.get().strip()
            username = self.username_entry.get().strip()
            password = self.password_entry.get()
            
            if not all([host, username]):
                messagebox.showerror("Error", "Please fill in Host and Username")
                return

            db_table = self.db_table_var.get()
            if '.' not in db_table:
                messagebox.showerror("Error", "Invalid database.table selection")
                return
                
            database, table = db_table.split('.', 1)

            # Start upload in separate thread
            self.log_status("=" * 50)
            self.log_status(f"Starting upload process for {len(file_list)} files...")
            self.log_status(f"Files: {file_paths}")
            self.log_status(f"Target: {database}.{table}")
            self.log_status(f"Instructions: {ConfigManager.get_table_instructions(database, table)}")
            self.log_status("=" * 50)

            threading.Thread(
                target=self.upload_file_thread, 
                args=(host, username, password, database, table, file_paths), 
                daemon=True
            ).start()

        except Exception as e:
            self.log_status(f"Upload initialization error: {e}")
            messagebox.showerror("Error", f"Upload initialization failed: {e}")

    def generate_pdf_report(self, database, table, file_list, total_rows, log_lines):
        """Save a well-formatted PDF report with auto page breaks, and skip if no relevant content."""
        try:
            today = datetime.datetime.today().strftime('%d-%m-%Y')
            file_name = f"Upload_Report_{today}.pdf"
            save_dir = r"C:\Users\Lenovo\OneDrive\Desktop\Rajan\Database Data"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, file_name)

            # Filter and clean log lines (remove timestamps)
            filtered_lines = []
            for line in log_lines:
                cleaned_line = line.strip()
                if cleaned_line.startswith('['):
                    bracket_end = cleaned_line.find(']')
                    if bracket_end != -1:
                        cleaned_line = cleaned_line[bracket_end + 1:].strip()
                if (
                    "FILES COMBINED AND UPLOADED SUCCESSFULLY" in cleaned_line
                    or "Target Database:" in cleaned_line
                    or "Target Table:" in cleaned_line
                    or "Total rows uploaded:" in cleaned_line
                    or "UPLOAD FAILED:" in cleaned_line
                    or "No valid data found for" in cleaned_line
                ):
                    filtered_lines.append(cleaned_line)

            # Skip if no relevant log lines found
            if not filtered_lines:
                self.log_status("⚠️ No relevant log lines found. PDF report not generated.")
                messagebox.showinfo("No Data", "No relevant information to generate a PDF report.")
                return

            # PDF setup
            c = canvas.Canvas(save_path, pagesize=A4)
            width, height = A4
            left_margin = 60
            line_height = 25
            section_spacing = 35
            bottom_margin = 80
            top_start = height - 80
            y = top_start

            def check_page_break(current_y, space_needed):
                nonlocal c, y
                if current_y - space_needed < bottom_margin:
                    c.showPage()
                    y = top_start

            # Header
            c.setFont("Helvetica-Bold", 18)
            c.drawString(left_margin, y, "UPLOAD SUMMARY REPORT")
            y -= 15
            c.line(left_margin, y, width - 60, y)
            y -= section_spacing

            # Date
            c.setFont("Helvetica", 12)
            c.drawString(left_margin, y, f"Report Generated: {today}")
            y -= section_spacing

            # Process filtered log lines
            for line in filtered_lines:
                if "FILES COMBINED AND UPLOADED SUCCESSFULLY" in line:
                    check_page_break(y, section_spacing)
                    c.setFont("Helvetica-Bold", 14)
                    c.setFillColorRGB(0, 0.7, 0)
                    c.drawString(left_margin, y, "✓ FILES COMBINED AND UPLOADED SUCCESSFULLY!")
                    c.setFillColorRGB(0, 0, 0)
                    y -= section_spacing

                elif "Target Database:" in line:
                    check_page_break(y, 2 * line_height)
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(left_margin, y, "Database Details:")
                    y -= line_height
                    c.setFont("Helvetica", 11)
                    c.drawString(left_margin + 20, y, f"• {line}")
                    y -= line_height

                elif "Target Table:" in line or "Total rows uploaded:" in line:
                    check_page_break(y, line_height)
                    c.setFont("Helvetica", 11)
                    c.drawString(left_margin + 20, y, f"• {line}")
                    y -= line_height

                elif "UPLOAD FAILED:" in line or "No valid data found for" in line:
                    # Wrap long error text
                    max_width = width - left_margin - 80
                    c.setFont("Helvetica", 11)
                    words = (f"Error: {line.split(':', 1)[1].strip() if ':' in line else line}").split()
                    lines, current_line = [], ""

                    for word in words:
                        test_line = current_line + " " + word if current_line else word
                        if c.stringWidth(test_line, "Helvetica", 11) > max_width:
                            if current_line:
                                lines.append(current_line)
                                current_line = word
                            else:
                                lines.append(word)
                                current_line = ""
                        else:
                            current_line = test_line
                    if current_line:
                        lines.append(current_line)

                    check_page_break(y, line_height * (len(lines) + 2))
                    c.setFont("Helvetica-Bold", 12)
                    c.setFillColorRGB(0.8, 0, 0)
                    c.drawString(left_margin, y, "✗ UPLOAD FAILED")
                    y -= line_height

                    c.setFont("Helvetica", 11)
                    for error_line in lines:
                        c.drawString(left_margin + 20, y, error_line)
                        y -= line_height

                    c.setFillColorRGB(0, 0, 0)
                    y -= 10

            # Footer
            check_page_break(y, 40)
            y = 60
            c.setFont("Helvetica", 10)
            c.setFillColorRGB(0.5, 0.5, 0.5)
            c.drawString(left_margin, y, f"Generated on {datetime.datetime.now().strftime('%d-%m-%Y at %H:%M:%S')}")
            c.drawString(width - 150, y, "Auto-generated Report")

            c.save()
            self.log_status(f"📄 PDF report saved at: {save_path}")
            messagebox.showinfo("PDF Generated", f"PDF report saved:\n{save_path}")

        except Exception as e:
            self.log_status(f"⚠️ Failed to generate PDF report: {e}")






    def upload_file_thread(self, host, username, password, database, table, file_paths):
        """Threaded upload method for multiple files"""
        try:
            # Create database manager
            self.log_status("Connecting to database...")
            db_manager = DatabaseManager(host, username, password, database)
            
            # Ensure database exists
            self.log_status(f"Ensuring database '{database}' exists...")
            db_manager.create_database_if_not_exists(database)
            
            # Create data uploader
            uploader = DataUploader(db_manager)
            
            # Process each file
            combined_df = None
            file_list = file_paths.split("; ") if "; " in file_paths else [file_paths]
            
            for i, file_path in enumerate(file_list, 1):
                self.log_status(f"\nProcessing file {i}/{len(file_list)}: {os.path.basename(file_path)}")
                
                # Read and process the file
                if file_path.lower().endswith('.csv'):
                    df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
                else:
                    df = pd.read_excel(file_path, dtype=str, keep_default_na=False, engine='openpyxl')
                
                processed_df = DataProcessor.process_data(df, database, table, self.logger)
                
                # Combine with previous files
                if combined_df is None:
                    combined_df = processed_df
                else:
                    combined_df = pd.concat([combined_df, processed_df], ignore_index=True)
                
                self.log_status(f"Added {len(processed_df)} rows from this file")
            
            # Upload combined data
            if combined_df is not None:
                self.log_status(f"\nUploading combined data ({len(combined_df)} total rows)...")
                uploader.validate_data_before_upload(combined_df, database, table)

                # Create database connection
                connection = db_manager.create_connection()
                cursor = connection.cursor()

                # Check if table exists
                table_exists = db_manager.table_exists(table)

                if table_exists:
                    self.log_status(f"Table {table} exists, will upsert data")
                    existing_columns = db_manager.get_table_columns(table)
                    primary_key = db_manager.get_primary_key(table)

                    # Ensure all columns exist
                    for col in combined_df.columns:
                        if col not in existing_columns:
                            col_type = uploader._determine_mysql_type(combined_df[col])
                            self.log_status(f"Adding missing column {col} as {col_type}")
                            cursor.execute(f"ALTER TABLE `{table}` ADD COLUMN `{col}` {col_type}")
                            connection.commit()

                    # Upload with upsert
                    uploader._upsert_data(connection, cursor, table, combined_df, primary_key)
                else:
                    self.log_status(f"Table {table} doesn't exist, creating new table")
                    uploader._create_table(cursor, table, combined_df)
                    connection.commit()
                    uploader._insert_data(connection, cursor, table, combined_df)

                connection.close()

                # ✅ Success message
                self.log_status("=" * 50)
                self.log_status(f"✅ {len(file_list)} FILES COMBINED AND UPLOADED SUCCESSFULLY!")
                self.log_status(f"Target Database: {database}")
                self.log_status(f"Target Table: {table}")
                self.log_status(f"Total rows uploaded: {len(combined_df)}")
                self.log_status("=" * 50)
           
                # Special handling for attroidfc.idfc - offer to export
                if database.lower() == 'attroidfc' and table.lower() == 'idfc':
                    self.root.after(0, uploader._offer_idfc_export)
                
            else:
                self.log_status("No valid data found in any files")
                
        except Exception as e:
            error_msg = f"Upload failed: {str(e)}"
            self.log_status("=" * 50)
            self.log_status(f"❌ UPLOAD FAILED: {error_msg}")
            self.log_status("=" * 50)
            
            # Show error dialog
            self.root.after(0, lambda: messagebox.showerror(
                "Upload Failed", 
                error_msg,
                detail=str(e)
            ))

    def log_status(self, message):
        """Thread-safe status logging"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        def update_ui():
            self.status_text.insert(tk.END, formatted_message)
            self.status_text.see(tk.END)
            self.root.update_idletasks()
        
        if threading.current_thread() == threading.main_thread():
            update_ui()
        else:
            self.root.after(0, update_ui)

    def clear_log(self):
        """Clear the status log"""
        self.status_text.delete(1.0, tk.END)


def main():
    """Main application entry point"""
    root = tk.Tk()
    
    # Set window icon if available
    try:
        root.iconbitmap(default='icon.ico')
    except:
        pass
    
    app = MySQLExcelUploaderGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()





# SELECT count(*) FROM vibepay_Portal.vibepay_payin_report WHERE DATE(`Date Time`) BETWEEN '2025-07-05' AND '2025-07-06';
# SELECT count(*) FROM vibepay_Portal.vibepay_payin_wallet WHERE DATE(`Date`) BETWEEN '2025-07-05' AND '2025-07-06';
# SELECT count(*) FROM vibepay_Portal.vibepay_payout_report WHERE DATE(`Date Time`) BETWEEN '2025-07-05' AND '2025-07-06';
# SELECT count(*) FROM vibepay_Portal.vibepay_payout_wallet WHERE DATE(`Date`) BETWEEN '2025-07-05' AND '2025-07-06';
# SELECT count(*) FROM vibepay_Portal.vibepay_add_fund_report WHERE DATE(`Date Time`) BETWEEN '2025-07-05' AND '2025-07-06';
# SELECT count(*) FROM vibepay_Portal.vibepayin_add_fund_report WHERE DATE(`Date`) BETWEEN '2025-07-05' AND '2025-07-06';


# SELECT count(*) FROM sabpaisa.payin WHERE DATE(`Transaction Date`) BETWEEN '2025-07-05' AND '2025-07-06';
# SELECT count(*) FROM sabpaisa.settelment_report WHERE DATE(`Transaction Date`) BETWEEN '2025-07-05' AND '2025-07-06';


# SELECT count(*) FROM swiftsend_portal.swiftsend_payout_report WHERE DATE(`Date Time`) BETWEEN '2025-07-05' AND '2025-07-06';
# SELECT count(*) FROM swiftsend_portal.swiftsend_payout_wallet WHERE DATE(`Date`) BETWEEN '2025-07-05' AND '2025-07-06';
# SELECT count(*) FROM swiftsend_portal.swiftsend_add_fund_report WHERE DATE(`Date`) BETWEEN '2025-07-05' AND '2025-07-06';




# SELECT count(*) FROM payonetic_portal.payonetic_payout_report WHERE DATE(`Date Time`) BETWEEN '2025-07-05' AND '2025-07-06';
# SELECT count(*) FROM payonetic_portal.payonetic_payout_wallet WHERE DATE(`Date Time`) BETWEEN '2025-07-05' AND '2025-07-06';
# SELECT count(*) FROM payonetic_portal.payonetic_add_fund_report WHERE DATE(`Date Time`) BETWEEN '2025-07-05' AND '2025-07-06';