import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os, sys, logging, threading, re
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import mysql.connector
import datetime
from datetime import datetime as dt, timedelta
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# ConfigManager  (no more hardcoded DATABASE_CONFIGS)
# ─────────────────────────────────────────────────────────────────────────────
class ConfigManager:
    TABLE_FORMATS = {
        'anateck.idfc':    ["Transaction Date","Payment date","Narrative","Customer Reference No","Cheque No","Debit","Credit","Running Balance"],
        'attroidfc.idfc':  ["Transaction Date","Payment date","Narrative","Customer Reference No","Cheque No","Debit","Credit","Running Balance"],
        'anateck.yes':     ["Transaction Date","Value date","Transaction Description","Reference No","Debit Amount","Credit Amount","Running Balance"],
        'attroidfc.yes':   ["Transaction Date","Value date","Transaction Description","Reference No","Debit Amount","Credit Amount","Running Balance"],
        'anateck.kotak':   ["#","TRANSACTION DATE","VALUE DATE","TRANSACTION DETAILS","CHQ / REF NO.","DEBIT/CREDIT(₹)","BALANCE(₹)"],
        'attroidfc.kotak': ["#","TRANSACTION DATE","VALUE DATE","TRANSACTION DETAILS","CHQ / REF NO.","DEBIT/CREDIT(₹)","BALANCE(₹)"],
    }

    @classmethod
    def get_table_instructions(cls, db_name: str, table_name: str) -> str:
        instructions = {
            'fbpayripe.payin':        'Remove First Col. Created Time and Then Upload',
            'fbpayripe.payout':       'Upload as it is.',
            'fbpayripe.wallet':       'Upload as it is.',
            'anateck.payout':         'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'anateck.wallet':         'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'anateck.payon_payout':   'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'anateck.payon_owallet':  'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'anateck.idfc':           'Extract specific table format data and process UTR/UID',
            'anateck.yes':            'Extract specific table format data and process dates/amounts',
            'anateck.kotak':          'Extract specific table format data and process dates/amounts',
            'attroidfc.payout':       'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'attroidfc.wallet':       'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'attroidfc.payon_payout': 'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'attroidfc.payon_owallet':'Change DATE FORMATS TO YYYY-MM-DD HH:MM:SS and Remove commas in amount col.',
            'attroidfc.idfc':         'Extract specific table format data and process UTR/UID',
            'attroidfc.yes':          'Extract specific table format data and process dates/amounts',
            'attroidfc.kotak':        'Extract specific table format data and process dates/amounts',
        }
        return instructions.get(
            f"{db_name}.{table_name}",
            "Upload as it is. Empty columns will be set to NULL. Dates should be in YYYY-MM-DD HH:MM:SS format."
        )

    @classmethod
    def get_expected_columns(cls, db_name: str, table_name: str) -> Optional[List[str]]:
        return cls.TABLE_FORMATS.get(f"{db_name}.{table_name}")

    @classmethod
    def requires_table_format_extraction(cls, db_name: str, table_name: str) -> bool:
        return f"{db_name}.{table_name}" in cls.TABLE_FORMATS


# ─────────────────────────────────────────────────────────────────────────────
# LoggerManager
# ─────────────────────────────────────────────────────────────────────────────
class LoggerManager:
    @staticmethod
    def setup_logger(log_file='data_uploader.log'):
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, log_file), encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TableFormatExtractor
# ─────────────────────────────────────────────────────────────────────────────
class TableFormatExtractor:
    @staticmethod
    def extract_table_data(df, expected_columns, logger):
        header_row_idx = TableFormatExtractor._find_header_row(df, expected_columns, logger)
        if header_row_idx is None:
            raise ValueError(f"Could not find table with expected columns: {expected_columns}")
        table_df = df.iloc[header_row_idx:].copy()
        table_df.columns = table_df.iloc[0]
        table_df = table_df.iloc[1:]
        available = [c for c in expected_columns if c in table_df.columns]
        table_df = table_df[available].dropna(how='all').reset_index(drop=True)
        return table_df

    @staticmethod
    def _find_header_row(df, expected_columns, logger):
        norm = lambda t: "" if pd.isna(t) else str(t).strip().lower()
        ne = [norm(c) for c in expected_columns]
        for idx in range(min(50, len(df))):
            row = df.iloc[idx].values
            nr = [norm(v) for v in row]
            ratio = sum(1 for ec in ne if ec in nr) / len(ne)
            if ratio >= 0.7:
                non_empty = sum(1 for v in row if pd.notna(v) and str(v).strip())
                if non_empty >= len(expected_columns) * 0.5:
                    return idx
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DataProcessor
# ─────────────────────────────────────────────────────────────────────────────
class DataProcessor:
    @staticmethod
    def process_data(df, db_name, table_name, logger):
        processed_df = df.copy()
        processed_df = DataProcessor.protect_scientific_notation(processed_df)
        processed_df = DataProcessor.format_dates(processed_df, logger)
        processed_df = DataProcessor.clean_amount_columns(processed_df)
        if ConfigManager.requires_table_format_extraction(db_name, table_name):
            ec = ConfigManager.get_expected_columns(db_name, table_name)
            if ec:
                processed_df = TableFormatExtractor.extract_table_data(processed_df, ec, logger)
        processed_df = DataProcessor.apply_table_specific_processing(processed_df, db_name, table_name, logger)
        return processed_df

    @staticmethod
    def apply_table_specific_processing(df, db_name, table_name, logger):
        if table_name.lower() in ['idfc', 'yes', 'kotak']:
            df = DataProcessor.process_bank_statement_columns(df, table_name.lower(), logger)
        if "Remove First Col. Created Time" in ConfigManager.get_table_instructions(db_name, table_name):
            df = df.iloc[:, 1:] if len(df.columns) > 0 else df
        return df

    @staticmethod
    def process_bank_statement_columns(df, bank_name, logger):
        for col in [c for c in df.columns if 'date' in c.lower()]:
            df[col] = DataProcessor._convert_date_column(df[col], logger)
        for col in [c for c in df.columns if any(k in c.lower() for k in ['amount','debit','credit','balance','value'])]:
            df[col] = DataProcessor._clean_numeric_value(df[col], logger)
        if bank_name == 'idfc':
            df = DataProcessor.process_idfc_reference_numbers(df, logger)
        return df

    @staticmethod
    def process_idfc_reference_numbers(df, logger):
        if 'Narrative' not in df.columns:
            return df
        p = re.compile(r'UPI/MOB/(\d{10,15})/([A-Za-z0-9]{8,30})')
        for idx, row in df.iterrows():
            narrative = str(row['Narrative']).strip()
            m = p.search(narrative)
            if m:
                df.at[idx, 'Customer Reference No'] = m.group(1)
                df.at[idx, 'Cheque No'] = m.group(2)
        return df

    @staticmethod
    def _clean_numeric_value(series, logger):
        cleaned = series.astype(str).str.replace(r'[₹,$€£,\s]', '', regex=True)
        cleaned = cleaned.str.replace(r'\(([\d.]+)\)', r'-\1', regex=True)
        return pd.to_numeric(cleaned, errors='coerce').fillna(0)

    @staticmethod
    def protect_scientific_notation(df):
        for col in df.columns:
            if any(k in str(col).lower() for k in ['date','time']):
                continue
            if df[col].dtype == 'object':
                mask = df[col].astype(str).str.contains(r'^\d+\.?\d*[Ee][+-]?\d+$', na=False)
                if mask.any():
                    df.loc[mask, col] = df.loc[mask, col].apply(
                        lambda x: format(float(x), '.{}f'.format(len(str(x).split('.')[1]) if '.' in str(x) else 0))
                    )
        return df

    @staticmethod
    def format_dates(df, logger):
        date_kw = ['date','time','datetime','timestamp','created','modified','processed','reversed','scheduled']
        for col in df.columns:
            if not any(k in str(col).lower() for k in date_kw):
                continue
            original = df[col].copy()
            s = df[col].astype(str).str.strip().replace(['','NA','N/A','nan','None','NULL','NaT'], None)
            result = s.copy()
            # fix day=00
            bad = s.str.contains(r'1900-01-00', na=False)
            result[bad] = '1900-01-01 00:00:00'
            handlers = [
                {'name':'period-sep', 'pattern':r'\d{2}-\d{2}-\d{4} \d{2}\.\d{2}\.\d{2}',
                 'preprocessor': lambda x: re.sub(r'(\d{2})-(\d{2})-(\d{4}) (\d{2})\.(\d{2})\.(\d{2})',r'\3-\2-\1 \4:\5:\6',x),
                 'format':'%Y-%m-%d %H:%M:%S'},
                {'name':'AMPM',    'pattern':r'\d{1,2}/\d{1,2}/\d{2,4} \d{1,2}:\d{2}:\d{2} [AP]M', 'format':'%d/%m/%Y %I:%M:%S %p'},
                {'name':'ISO8601', 'pattern':r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', 'format':'ISO8601'},
                {'name':'YYYY-MM-DD','pattern':r'^\d{4}-\d{2}-\d{2}$','format':'%Y-%m-%d','post':lambda x: f"{x} 00:00:00"},
                {'name':'DD/MM/YYYY','pattern':r'^\d{2}/\d{2}/\d{4}$','format':'%d/%m/%Y','post':lambda x: f"{x} 00:00:00"},
                {'name':'dayfirst','pattern':None,'dayfirst':True},
                {'name':'default', 'pattern':None},
            ]
            remaining = s.notna()
            for h in handlers:
                if not remaining.any(): break
                mask = remaining & s.str.contains(h['pattern'], regex=True, na=False) if h['pattern'] else remaining
                if not mask.any(): continue
                try:
                    to_conv = s[mask]
                    if 'preprocessor' in h: to_conv = to_conv.apply(h['preprocessor'])
                    if h.get('format') == 'ISO8601':
                        conv = pd.to_datetime(to_conv, format='ISO8601', errors='coerce')
                    elif 'format' in h:
                        conv = pd.to_datetime(to_conv, format=h['format'], errors='coerce')
                    elif 'dayfirst' in h:
                        conv = pd.to_datetime(to_conv, dayfirst=h['dayfirst'], errors='coerce')
                    else:
                        conv = pd.to_datetime(to_conv, errors='coerce')
                    if 'post' in h and conv.notna().any():
                        conv = conv.apply(h['post'])
                    ok = conv.notna()
                    if ok.any():
                        result[mask & ok] = conv[ok]
                        remaining &= ~(mask & ok)
                except Exception as e:
                    logger.error(f"Date handler '{h['name']}' error: {e}")
            try:
                dm = pd.to_datetime(result, errors='coerce').notna()
                final = result.copy()
                final[dm] = pd.to_datetime(result[dm]).dt.strftime('%Y-%m-%d %H:%M:%S')
                df[col] = final
            except Exception as e:
                logger.error(f"Final date processing error for '{col}': {e}")
                df[col] = original
        return df

    @staticmethod
    def _convert_date_column(series, logger):
        s = series.astype(str).str.strip()
        converted = pd.Series(index=series.index, dtype='object')
        bad = s.str.contains(r'1900-01-00', na=False)
        converted[bad] = '1900-01-01 00:00:00'
        valid = ~bad & s.notna() & (s != '')
        if valid.any():
            try:
                converted[valid] = pd.to_datetime(s[valid], dayfirst=True, errors='coerce')
            except Exception as e:
                logger.error(f"Date conversion error: {e}")
        fmt = converted.dt.strftime('%Y-%m-%d %H:%M:%S')
        failed = fmt.isna() & s.notna() & (s != '')
        fmt[failed] = s[failed]
        return fmt

    @staticmethod
    def clean_amount_columns(df):
        for col in df.columns:
            if not any(k in str(col).lower() for k in ['amount','debit','credit','balance','₹','value']):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            cd = df[col].astype(str).str.strip().str.replace(r'[₹,$€£,\s]', '', regex=True)
            pm = cd.str.contains(r'\(.*\)', na=False)
            if pm.any():
                df.loc[pm, col] = '-' + cd[pm].str.replace(r'[\(\)]','',regex=True)
                cd = df[col].astype(str).str.strip()
            try:
                df[col] = pd.to_numeric(cd, errors='raise')
            except:
                df[col] = pd.to_numeric(cd.str.extract(r'([-+]?\d*\.?\d+)', expand=False), errors='coerce')
        return df

    @staticmethod
    def extract_utr_uid(df):
        nc = next((c for c in df.columns if 'narration' in str(c).lower() or 'narrative' in str(c).lower()), None)
        if nc is None: return df
        def _utr(n):
            if not isinstance(n, str): return ''
            m = re.search(r'(?:UTR|Ref No)[:/]?\s*([A-Za-z0-9]{12,22})', n, re.IGNORECASE)
            return m.group(1) if m else ''
        def _uid(n):
            if not isinstance(n, str): return ''
            m = re.search(r'(?:UID|Customer ID)[:/]?\s*([A-Za-z0-9]{8,16})', n, re.IGNORECASE)
            return m.group(1) if m else ''
        df['UTR'] = df[nc].apply(_utr)
        df['UID'] = df[nc].apply(_uid)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# DatabaseManager
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_DBS = {'information_schema', 'mysql', 'performance_schema', 'sys'}

class DatabaseManager:
    def __init__(self, host, username, password, database='information_schema'):
        self.host = host
        self.username = username
        self.password = password
        self.database = database
        self.logger = LoggerManager.setup_logger()

    def _conn(self, db=None):
        return mysql.connector.connect(
            host=self.host, user=self.username, password=self.password,
            database=db or self.database,
            charset='utf8mb4', collation='utf8mb4_unicode_ci', autocommit=False
        )

    def fetch_databases(self) -> List[str]:
        """Return all user databases from the server."""
        try:
            c = self._conn('information_schema')
            cur = c.cursor()
            cur.execute("SHOW DATABASES")
            dbs = [r[0] for r in cur.fetchall() if r[0].lower() not in SYSTEM_DBS]
            cur.close(); c.close()
            return sorted(dbs)
        except Exception as e:
            self.logger.error(f"Error fetching databases: {e}")
            return []

    def fetch_tables(self, db_name: str) -> List[str]:
        """Return all tables for a given database."""
        try:
            c = self._conn(db_name)
            cur = c.cursor()
            cur.execute("SHOW TABLES")
            tables = [r[0] for r in cur.fetchall()]
            cur.close(); c.close()
            return sorted(tables)
        except Exception as e:
            self.logger.error(f"Error fetching tables for {db_name}: {e}")
            return []

    def create_database_if_not_exists(self, db_name: str):
        c = self._conn('information_schema')
        cur = c.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        c.commit(); cur.close(); c.close()

    def create_table_skeleton(self, db_name: str, table_name: str):
        c = self._conn(db_name)
        cur = c.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                `id` INT NOT NULL AUTO_INCREMENT, PRIMARY KEY (`id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        c.commit(); cur.close(); c.close()

    def table_exists(self, db_name: str, table_name: str) -> bool:
        c = self._conn(db_name)
        cur = c.cursor()
        cur.execute(f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema='{db_name}' AND table_name='{table_name}'
        """)
        exists = cur.fetchone()[0] == 1
        cur.close(); c.close()
        return exists

    def get_table_columns(self, db_name: str, table_name: str) -> List[str]:
        c = self._conn(db_name)
        cur = c.cursor()
        cur.execute(f"SHOW COLUMNS FROM `{db_name}`.`{table_name}`")
        cols = [r[0] for r in cur.fetchall()]
        cur.close(); c.close()
        return cols

    def get_primary_key(self, db_name: str, table_name: str) -> Optional[str]:
        c = self._conn(db_name)
        cur = c.cursor()
        cur.execute(f"""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA='{db_name}' AND TABLE_NAME='{table_name}' AND COLUMN_KEY='PRI'
        """)
        r = cur.fetchone()
        cur.close(); c.close()
        return r[0] if r else None

    def export_idfc_to_excel(self, db_name: str, save_path: str) -> bool:
        c = self._conn(db_name)
        cur = c.cursor(dictionary=True)
        cur.execute("SELECT * FROM idfc")
        rows = cur.fetchall()
        cur.close(); c.close()
        if not rows: return False
        df = DataProcessor.extract_utr_uid(pd.DataFrame(rows))
        df.to_excel(save_path, index=False, engine='openpyxl')
        return True


# ─────────────────────────────────────────────────────────────────────────────
# DataUploader
# ─────────────────────────────────────────────────────────────────────────────
class DataUploader:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = LoggerManager.setup_logger()

    def validate(self, df, db_name, table_name):
        if ConfigManager.requires_table_format_extraction(db_name, table_name):
            ec = ConfigManager.get_expected_columns(db_name, table_name)
            missing = set(ec) - set(df.columns)
            if missing: raise ValueError(f"Missing expected columns: {missing}")
        if df.empty: raise ValueError("No data to upload after processing")

    def _mysql_type(self, series):
        series = series.replace('', None)
        try:
            num = pd.to_numeric(series.dropna())
            if num.empty: return "TEXT"
            if all(num == num.astype(int)):
                mx, mn = num.max(), num.min()
                for lo, hi, typ in [(-128,127,"TINYINT"),(-32768,32767,"SMALLINT"),
                                    (-8388608,8388607,"MEDIUMINT"),(-2147483648,2147483647,"INT")]:
                    if mn >= lo and mx <= hi: return typ
                return "BIGINT"
            return "DOUBLE"
        except: pass
        try:
            parsed = pd.to_datetime(series.dropna(), errors='coerce', format='mixed')
            if parsed.notna().all(): return "DATETIME"
            raise ValueError
        except: pass
        ml = series.astype(str).str.len().max()
        if pd.isna(ml) or ml <= 255: return "VARCHAR(255)"
        if ml <= 65535: return "TEXT"
        if ml <= 16777215: return "MEDIUMTEXT"
        return "LONGTEXT"

    def _create_table(self, cursor, db_name, table_name, df):
        defs = [f"`{c}` {self._mysql_type(df[c])}" for c in df.columns]
        cursor.execute(f"""
            CREATE TABLE `{db_name}`.`{table_name}` ({', '.join(defs)})
            ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)

    def _insert(self, conn, cursor, db_name, table_name, df):
        ph = ", ".join(["%s"] * len(df.columns))
        cols = ", ".join([f"`{c}`" for c in df.columns])
        q = f"INSERT INTO `{db_name}`.`{table_name}` ({cols}) VALUES ({ph})"
        batch, batch_size = [], 500
        for _, row in df.iterrows():
            batch.append(tuple(None if (pd.isna(v) or v == '') else str(v) for v in row))
            if len(batch) >= batch_size:
                try:
                    cursor.executemany(q, batch); conn.commit()
                except Exception as e:
                    self.logger.error(f"Batch error: {e}"); conn.rollback()
                    for r in batch:
                        try: cursor.execute(q, r); conn.commit()
                        except: conn.rollback()
                batch = []
        if batch:
            try: cursor.executemany(q, batch); conn.commit()
            except Exception as e:
                self.logger.error(f"Final batch error: {e}"); conn.rollback()

    def _upsert(self, conn, cursor, db_name, table_name, df, pk=None):
        if not pk:
            return self._insert(conn, cursor, db_name, table_name, df)
        cursor.execute(f"SELECT `{pk}` FROM `{db_name}`.`{table_name}`")
        existing = {r[0] for r in cursor.fetchall()}
        new_rows = df[~df[pk].isin(existing)]
        if new_rows.empty:
            self.logger.info("All records already exist – nothing new to insert")
            return
        self._insert(conn, cursor, db_name, table_name, new_rows)

    def upload(self, db_name, table_name, combined_df, log_fn):
        self.validate(combined_df, db_name, table_name)
        conn = self.db._conn(db_name)
        cursor = conn.cursor()
        if self.db.table_exists(db_name, table_name):
            log_fn(f"Table '{table_name}' exists – upserting")
            existing_cols = self.db.get_table_columns(db_name, table_name)
            pk = self.db.get_primary_key(db_name, table_name)
            for col in combined_df.columns:
                if col not in existing_cols:
                    t = self._mysql_type(combined_df[col])
                    log_fn(f"Adding column '{col}' ({t})")
                    cursor.execute(f"ALTER TABLE `{db_name}`.`{table_name}` ADD COLUMN `{col}` {t}")
                    conn.commit()
            self._upsert(conn, cursor, db_name, table_name, combined_df, pk)
        else:
            log_fn(f"Table '{table_name}' not found – creating")
            self._create_table(cursor, db_name, table_name, combined_df)
            conn.commit()
            self._insert(conn, cursor, db_name, table_name, combined_df)
        cursor.close(); conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# CreateDBTableDialog
# ─────────────────────────────────────────────────────────────────────────────
class CreateDBTableDialog(tk.Toplevel):
    def __init__(self, parent_gui):
        super().__init__(parent_gui.root)
        self.pg = parent_gui
        self.title("➕ Create New Database / Table")
        self.geometry("480x400")
        self.resizable(False, False)
        self.configure(bg="#f5f7fa")
        self.grab_set(); self.focus_force()
        self._center(); self._build()

    def _center(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth()-480)//2
        y = (self.winfo_screenheight()-400)//2
        self.geometry(f"+{x}+{y}")

    def _build(self):
        tk.Label(self, text="Create New Database / Table",
                 font=("Segoe UI",13,"bold"), bg="#f5f7fa", fg="#2c3e50").pack(pady=(16,2))
        tk.Label(self, text="Leave 'New Database' blank to use an existing one.",
                 font=("Segoe UI",9), bg="#f5f7fa", fg="#7f8c8d").pack()
        ttk.Separator(self).pack(fill="x", padx=18, pady=10)

        f = tk.Frame(self, bg="#f5f7fa"); f.pack(fill="both", padx=18)
        p = {"padx":10,"pady":6}

        tk.Label(f, text="New Database Name:", bg="#f5f7fa", font=("Segoe UI",10)).grid(row=0,column=0,sticky="w",**p)
        self.new_db = tk.StringVar()
        tk.Entry(f, textvariable=self.new_db, font=("Segoe UI",10), width=28).grid(row=0,column=1,sticky="ew",**p)

        tk.Label(f, text="— OR pick existing —", bg="#f5f7fa", fg="#95a5a6",
                 font=("Segoe UI",9)).grid(row=1,column=0,columnspan=3,pady=(0,2))

        tk.Label(f, text="Existing Database:", bg="#f5f7fa", font=("Segoe UI",10)).grid(row=2,column=0,sticky="w",**p)
        self.exist_db = tk.StringVar()
        self.exist_cb = ttk.Combobox(f, textvariable=self.exist_db, font=("Segoe UI",10), width=25, state="readonly")
        self.exist_cb.grid(row=2,column=1,sticky="ew",**p)
        tk.Button(f, text="🔄 Fetch", font=("Segoe UI",9), command=self._fetch,
                  relief="flat", bg="#3498db", fg="white", padx=6).grid(row=2,column=2,padx=(0,6))

        ttk.Separator(f,orient="horizontal").grid(row=3,column=0,columnspan=3,sticky="ew",pady=8)

        tk.Label(f, text="New Table Name:*", bg="#f5f7fa", font=("Segoe UI",10)).grid(row=4,column=0,sticky="w",**p)
        self.new_tbl = tk.StringVar()
        tk.Entry(f, textvariable=self.new_tbl, font=("Segoe UI",10), width=28).grid(row=4,column=1,sticky="ew",**p)
        f.columnconfigure(1, weight=1)

        self.status = tk.Label(self, text="", bg="#f5f7fa", font=("Segoe UI",9), fg="#e74c3c", wraplength=440)
        self.status.pack(pady=4)

        bf = tk.Frame(self, bg="#f5f7fa"); bf.pack(pady=8)
        tk.Button(bf, text="✅ Create", font=("Segoe UI",10,"bold"),
                  bg="#27ae60", fg="white", padx=14, pady=6, relief="flat",
                  command=self._create).pack(side="left", padx=6)
        tk.Button(bf, text="Cancel", font=("Segoe UI",10),
                  bg="#e74c3c", fg="white", padx=14, pady=6, relief="flat",
                  command=self.destroy).pack(side="left", padx=6)

    def _fetch(self):
        try:
            mgr = DatabaseManager(self.pg.host_entry.get().strip(),
                                  self.pg.username_entry.get().strip(),
                                  self.pg.password_entry.get())
            dbs = mgr.fetch_databases()
            self.exist_cb['values'] = dbs
            self.status.config(text=f"✔ {len(dbs)} databases loaded.", fg="#27ae60")
        except Exception as e:
            self.status.config(text=f"Connection error: {e}", fg="#e74c3c")

    def _create(self):
        new_db  = self.new_db.get().strip()
        exist   = self.exist_db.get().strip()
        new_tbl = self.new_tbl.get().strip()
        target  = new_db or exist
        if not target:
            self.status.config(text="Provide a new or existing database name."); return
        if not new_tbl:
            self.status.config(text="Table name is required."); return
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', target):
            self.status.config(text="Database name: letters/digits/_ only."); return
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', new_tbl):
            self.status.config(text="Table name: letters/digits/_ only."); return
        if not messagebox.askyesno("Confirm", f"Create database '{target}' (if new) and table '{new_tbl}'?", parent=self):
            return
        try:
            mgr = DatabaseManager(self.pg.host_entry.get().strip(),
                                  self.pg.username_entry.get().strip(),
                                  self.pg.password_entry.get())
            mgr.create_database_if_not_exists(target)
            mgr.create_table_skeleton(target, new_tbl)
            # refresh dropdowns
            self.pg._refresh_db_dropdown()
            self.pg.db_var.set(target)
            self.pg._load_tables(target)
            self.pg.table_var.set(new_tbl)
            self.pg.update_instructions()
            self.pg.log_status(f"✅ Created database '{target}' and table '{new_tbl}'")
            messagebox.showinfo("Success", f"Created:\n  Database: {target}\n  Table: {new_tbl}", parent=self)
            self.destroy()
        except Exception as e:
            self.status.config(text=f"Error: {e}", fg="#e74c3c")


# ─────────────────────────────────────────────────────────────────────────────
# MySQLExcelUploaderGUI  – TWO separate dropdowns: Database | Table
# ─────────────────────────────────────────────────────────────────────────────
class MySQLExcelUploaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MySQL Excel Data Uploader")
        self.root.geometry("980x740")
        self.logger = LoggerManager.setup_logger()
        self.logger.info("Application starting...")
        self.db_manager: Optional[DatabaseManager] = None
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.update()
        self.check_password()

    # ── UI ────────────────────────────────────────────────────────────────────
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TFrame",      background="#f5f7fa")
        style.configure("TLabelFrame", background="#ffffff", borderwidth=2, relief="solid", padding=10)
        style.configure("TLabelFrame.Label", background="#ffffff", font=("Segoe UI",10,"bold"))
        style.configure("TLabel",  background="#ffffff", font=("Segoe UI",10))
        style.configure("TEntry",  font=("Segoe UI",10), padding=8, fieldbackground="#f8f9fa")
        for name, bg, abg in [("Primary","#4CAF50","#45a049"),("Secondary","#3498db","#2980b9"),
                               ("Create","#8e44ad","#7d3c98"),("Fetch","#e67e22","#d35400")]:
            style.configure(f"{name}.TButton", background=bg, foreground="white",
                            font=("Segoe UI",10,"bold"), padding=8, borderwidth=0)
            style.map(f"{name}.TButton", background=[("active",abg)])

        main = ttk.Frame(self.root, padding="20")
        main.pack(fill='both', expand=True)

        # ── Connection ────────────────────────────────────────────────────────
        cf = ttk.LabelFrame(main, text="Database Connection", padding="15")
        cf.pack(fill='x', pady=(0,10))

        # Host / User / Password
        for row, (lbl, default, show) in enumerate([
            ("Host:", "192.168.1.11", "-"),
            ("Username:", "SRA", "%"),
            ("Password:", "123", "*"),
        ]):
            ttk.Label(cf, text=lbl).grid(row=row, column=0, sticky='w', padx=5, pady=5)
            e = ttk.Entry(cf, show=show); e.insert(0, default)
            e.grid(row=row, column=1, columnspan=3, sticky='ew', padx=5, pady=5)
            setattr(self, ['host_entry','username_entry','password_entry'][row], e)

        # ── Database row ──────────────────────────────────────────────────────
        ttk.Label(cf, text="Database:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.db_var = tk.StringVar()
        self.db_cb  = ttk.Combobox(cf, textvariable=self.db_var, state="readonly", width=30)
        self.db_cb.grid(row=3, column=1, sticky='ew', padx=5, pady=5)

        ttk.Button(cf, text="🔄 Load DBs", style="Fetch.TButton",
                   command=self._refresh_db_dropdown
                   ).grid(row=3, column=2, padx=(4,2), pady=5)

        ttk.Button(cf, text="➕ Create New DB / Table", style="Create.TButton",
                   command=lambda: CreateDBTableDialog(self)
                   ).grid(row=3, column=3, padx=(2,5), pady=5)

        # ── Table row ─────────────────────────────────────────────────────────
        ttk.Label(cf, text="Table:").grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.table_var = tk.StringVar()
        self.table_cb  = ttk.Combobox(cf, textvariable=self.table_var, state="readonly", width=30)
        self.table_cb.grid(row=4, column=1, sticky='ew', padx=5, pady=5)

        # bind DB selection → load tables
        self.db_var.trace_add('write', lambda *_: self._on_db_selected())
        self.table_var.trace_add('write', lambda *_: self.update_instructions())

        # Instructions
        self.instr_var = tk.StringVar(value="Select a database, then a table.")
        ttk.Label(cf, text="Instructions:").grid(row=5, column=0, sticky='nw', padx=5, pady=5)
        ttk.Label(cf, textvariable=self.instr_var, wraplength=560, foreground="#2c3e50",
                  font=("Segoe UI",9)).grid(row=5, column=1, columnspan=3, sticky='w', padx=5, pady=5)

        cf.columnconfigure(1, weight=1)

        # ── File Upload ───────────────────────────────────────────────────────
        uf = ttk.LabelFrame(main, text="File Upload", padding="15")
        uf.pack(fill='x', pady=(0,10))

        ttk.Label(uf, text="Select File:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.file_var = tk.StringVar()
        ttk.Entry(uf, textvariable=self.file_var, state='readonly'
                  ).grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        ttk.Button(uf, text="Browse", command=self.browse_file
                   ).grid(row=0, column=2, padx=5, pady=5)

        bf = ttk.Frame(uf); bf.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(5,0))
        ttk.Button(bf, text="⬆ Upload Excel File",
                   command=self.initiate_upload, style="Primary.TButton"
                   ).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(bf, text="📄 Generate PDF Report",
                   command=self.generate_pdf_button_handler, style="Secondary.TButton"
                   ).pack(side='left', expand=True, fill='x', padx=2)
        uf.columnconfigure(1, weight=1)

        # ── Status ────────────────────────────────────────────────────────────
        sf = ttk.LabelFrame(main, text="Upload Status", padding="15")
        sf.pack(fill='both', expand=True)
        tf = ttk.Frame(sf); tf.pack(fill='both', expand=True)
        self.status_text = tk.Text(tf, wrap='word', font=('Consolas',10),
                                   padx=10, pady=10, bg="#f8f9fa", relief="flat",
                                   borderwidth=1, highlightthickness=1,
                                   highlightbackground="#dfe6e9")
        sb = ttk.Scrollbar(tf, orient='vertical', command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=sb.set)
        self.status_text.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        ttk.Button(sf, text="Clear Log", command=self.clear_log).pack(pady=(10,5))

    # ── DB / Table dynamic loading ────────────────────────────────────────────
    def _refresh_db_dropdown(self):
        """Connect and load all databases into the DB combobox."""
        host = self.host_entry.get().strip()
        user = self.username_entry.get().strip()
        pwd  = self.password_entry.get()
        if not host or not user:
            messagebox.showerror("Error", "Please enter Host and Username first.")
            return
        try:
            self.db_manager = DatabaseManager(host, user, pwd)
            dbs = self.db_manager.fetch_databases()
            if dbs:
                self.db_cb['values'] = dbs
                self.log_status(f"✔ Loaded {len(dbs)} databases.")
            else:
                self.log_status("⚠ No user databases found.")
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            self.log_status(f"❌ Connection failed: {e}")

    def _on_db_selected(self):
        db = self.db_var.get().strip()
        if not db:
            return
        self._load_tables(db)

    def _load_tables(self, db_name: str):
        """Load tables for the selected database."""
        if not self.db_manager:
            host = self.host_entry.get().strip()
            user = self.username_entry.get().strip()
            pwd  = self.password_entry.get()
            self.db_manager = DatabaseManager(host, user, pwd)
        try:
            tables = self.db_manager.fetch_tables(db_name)
            self.table_cb['values'] = tables
            self.table_var.set('')          # clear previous selection
            if tables:
                self.log_status(f"✔ Loaded {len(tables)} tables from '{db_name}'.")
            else:
                self.log_status(f"⚠ No tables found in '{db_name}'.")
        except Exception as e:
            self.log_status(f"❌ Failed to load tables: {e}")

    def update_instructions(self):
        db    = self.db_var.get().strip()
        table = self.table_var.get().strip()
        if db and table:
            self.instr_var.set(ConfigManager.get_table_instructions(db, table))
        else:
            self.instr_var.set("Select a database, then a table.")

    # ── helpers ───────────────────────────────────────────────────────────────
    def on_close(self):
        self.root.destroy()

    def _get_db_table(self):
        """Return (db, table) or raise if not selected."""
        db    = self.db_var.get().strip()
        table = self.table_var.get().strip()
        if not db or not table:
            raise ValueError("Please select both a Database and a Table.")
        return db, table

    def browse_file(self):
        try:
            db, table = self._get_db_table()
        except ValueError as e:
            messagebox.showerror("Error", str(e)); return
        initial_dir = self._file_location(db, table)
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()
        paths = filedialog.askopenfilenames(
            title=f"Select Files for {db}.{table}",
            initialdir=initial_dir,
            filetypes=[("Excel/CSV","*.xlsx *.xls *.csv"),("All Files","*.*")]
        )
        if paths:
            self.file_var.set("; ".join(paths))
            self.log_status(f"Selected {len(paths)} file(s)")

    def initiate_upload(self):
        try:
            db, table = self._get_db_table()
        except ValueError as e:
            messagebox.showerror("Error", str(e)); return

        file_paths = self.file_var.get()
        if not file_paths:
            messagebox.showerror("Error", "Please select files to upload."); return

        file_list = file_paths.split("; ") if "; " in file_paths else [file_paths]
        for fp in file_list:
            if not os.path.exists(fp):
                messagebox.showerror("Error", f"File not found:\n{fp}"); return
            if not fp.lower().endswith(('.xlsx','.xls','.csv')):
                messagebox.showerror("Error", f"Invalid file type:\n{fp}"); return

        host = self.host_entry.get().strip()
        user = self.username_entry.get().strip()
        pwd  = self.password_entry.get()
        if not host or not user:
            messagebox.showerror("Error", "Please fill in Host and Username."); return

        self.log_status("=" * 50)
        self.log_status(f"Starting upload for {len(file_list)} file(s)…")
        self.log_status(f"Target: {db}.{table}")
        self.log_status("=" * 50)

        threading.Thread(
            target=self._upload_thread,
            args=(host, user, pwd, db, table, file_list),
            daemon=True
        ).start()

    def _upload_thread(self, host, user, pwd, db, table, file_list):
        try:
            mgr = DatabaseManager(host, user, pwd)
            mgr.create_database_if_not_exists(db)
            uploader = DataUploader(mgr)
            combined = None

            for i, fp in enumerate(file_list, 1):
                self.log_status(f"Processing file {i}/{len(file_list)}: {os.path.basename(fp)}")
                df = (pd.read_csv(fp, dtype=str, keep_default_na=False)
                      if fp.lower().endswith('.csv')
                      else pd.read_excel(fp, dtype=str, keep_default_na=False, engine='openpyxl'))
                processed = DataProcessor.process_data(df, db, table, self.logger)
                combined = processed if combined is None else pd.concat([combined, processed], ignore_index=True)
                self.log_status(f"  → {len(processed)} rows from this file")

            if combined is not None:
                self.log_status(f"Uploading {len(combined)} total rows…")
                uploader.upload(db, table, combined, self.log_status)
                self.log_status("=" * 50)
                self.log_status(f"✅ {len(file_list)} FILE(S) UPLOADED SUCCESSFULLY!")
                self.log_status(f"Target Database: {db}")
                self.log_status(f"Target Table: {table}")
                self.log_status(f"Total rows uploaded: {len(combined)}")
                self.log_status("=" * 50)
                if db.lower() == 'attroidfc' and table.lower() == 'idfc':
                    self.root.after(0, self._offer_idfc_export, db)
            else:
                self.log_status("No valid data found in any files.")
        except Exception as e:
            self.log_status("=" * 50)
            self.log_status(f"❌ UPLOAD FAILED: {e}")
            self.log_status("=" * 50)
            self.root.after(0, lambda: messagebox.showerror("Upload Failed", str(e)))

    def _offer_idfc_export(self, db):
        if messagebox.askyesno("Export IDFC", "Export IDFC data to Excel?"):
            path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                filetypes=[("Excel","*.xlsx")])
            if path:
                try:
                    mgr = DatabaseManager(self.host_entry.get().strip(),
                                          self.username_entry.get().strip(),
                                          self.password_entry.get())
                    if mgr.export_idfc_to_excel(db, path):
                        messagebox.showinfo("Exported", f"Saved to:\n{path}")
                except Exception as e:
                    messagebox.showerror("Export Failed", str(e))

    # ── PDF ──────────────────────────────────────────────────────────────────
    def generate_pdf_button_handler(self):
        try:
            db, table = self._get_db_table()
        except ValueError as e:
            messagebox.showerror("Error", str(e)); return
        log_lines = self.status_text.get("1.0", tk.END).strip().splitlines()
        self.generate_pdf_report(db, table, log_lines)

    def generate_pdf_report(self, database, table, log_lines):
        try:
            today = datetime.datetime.today().strftime('%d-%m-%Y')
            save_dir = r"C:\Users\Lenovo\OneDrive\Desktop\Rajan\Database Data"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"Upload_Report_{today}.pdf")
            keywords = ["FILES UPLOADED SUCCESSFULLY","FILE(S) UPLOADED SUCCESSFULLY",
                        "Target Database:","Target Table:","Total rows uploaded:",
                        "UPLOAD FAILED:","No valid data found"]
            filtered = []
            for line in log_lines:
                cl = line.strip()
                if cl.startswith('['):
                    be = cl.find(']')
                    if be != -1: cl = cl[be+1:].strip()
                if any(k in cl for k in keywords):
                    filtered.append(cl)
            if not filtered:
                messagebox.showinfo("No Data","No relevant information to generate a PDF report.")
                return
            c = canvas.Canvas(save_path, pagesize=A4)
            w, h = A4; lm, lh, ss, bm = 60, 25, 35, 80; y = h - 80
            def pb(needed):
                nonlocal y
                if y - needed < bm: c.showPage(); y = h - 80
            c.setFont("Helvetica-Bold",18); c.drawString(lm,y,"UPLOAD SUMMARY REPORT"); y -= 15
            c.line(lm,y,w-60,y); y -= ss
            c.setFont("Helvetica",12); c.drawString(lm,y,f"Report Generated: {today}"); y -= ss
            for line in filtered:
                if "UPLOADED SUCCESSFULLY" in line:
                    pb(ss); c.setFont("Helvetica-Bold",14); c.setFillColorRGB(0,0.7,0)
                    c.drawString(lm,y,"✓ "+line); c.setFillColorRGB(0,0,0); y -= ss
                elif any(k in line for k in ["Target Database:","Target Table:","Total rows uploaded:"]):
                    pb(lh); c.setFont("Helvetica",11); c.drawString(lm+20,y,f"• {line}"); y -= lh
                elif any(k in line for k in ["UPLOAD FAILED:","No valid data found"]):
                    pb(lh*2); c.setFont("Helvetica-Bold",12); c.setFillColorRGB(0.8,0,0)
                    c.drawString(lm,y,"✗ UPLOAD FAILED"); y -= lh
                    c.setFont("Helvetica",11); c.drawString(lm+20,y,line)
                    c.setFillColorRGB(0,0,0); y -= lh
            c.setFont("Helvetica",10); c.setFillColorRGB(0.5,0.5,0.5)
            c.drawString(lm, 60, f"Generated on {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
            c.save()
            self.log_status(f"📄 PDF saved: {save_path}")
            messagebox.showinfo("PDF Generated", f"Saved:\n{save_path}")
        except Exception as e:
            self.log_status(f"⚠️ PDF failed: {e}")

    # ── file location map ─────────────────────────────────────────────────────
    @staticmethod
    def _file_location(db, table):
        locations = {
            'vibepay_Portal.vibepay_payin_report':      r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\01 Vibepay Payin Data\Payin Report",
            'vibepay_Portal.vibepay_payin_wallet':      r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\01 Vibepay Payin Data\Payin Wallet",
            'vibepay_Portal.vibepay_payout_report':     r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\02 Vibepay Payout Data\Payout Report (Daily)",
            'vibepay_Portal.vibepay_payout_wallet':     r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\02 Vibepay Payout Data\Payout Wallet",
            'vibepay_Portal.vibepay_add_fund_report':   r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\05 Vibepay Add Fund Report",
            'vibepay_Portal.vibepayin_add_fund_report': r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\01 Vibepay Payin Data\Add Fund Report",
            'sabpaisa.payin':             r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\03 Subpaisa Payin Report\Transaction Report",
            'sabpaisa.settel_hai':        r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\03 Subpaisa Payin Report\Transaction Report",
            'sabpaisa.settelment_report': r"\\server\F DRIVE DESKTOP\Parties Record\Vibepay\FY 2025-26\01 Statements and Records\03 Subpaisa Payin Report\Settlement Report",
            'rozarpayx.payout_report':    r"\\Server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\08 RazorPayX\Reports",
            'rozarpayx.transaction_report': r"\\Server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\08 RazorPayX\Reports",
            'swiftsend_portal.swiftsend_payout_report':   r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\Payout Report (Daily)",
            'swiftsend_portal.swiftsend_payout_wallet':   r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\Payout Wallet",
            'swiftsend_portal.swiftsend_finzen_payout':   r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\07 Finzen\Payout Statement",
            'swiftsend_portal.swiftsend_add_fund_report': r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\Add fund report",
            'payonetic_portal.payonetic_payout_report':   r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\05 Payonetic\Payout Report (Daily)",
            'payonetic_portal.payonetic_payout_wallet':   r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\05 Payonetic\Wallet",
            'payonetic_portal.payonetic_add_fund_report': r"\\server\F DRIVE DESKTOP\Parties Record\Swiftsend\2.F.Y. 2025-26\05 Payonetic\Add fund",
        }
        return locations.get(f"{db}.{table}", os.getcwd())

    # ── status log ────────────────────────────────────────────────────────────
    def log_status(self, message):
        ts  = datetime.datetime.now().strftime("%H:%M:%S")
        msg = f"[{ts}] {message}\n"
        def upd():
            self.status_text.insert(tk.END, msg)
            self.status_text.see(tk.END)
            self.root.update_idletasks()
        if threading.current_thread() == threading.main_thread(): upd()
        else: self.root.after(0, upd)

    def clear_log(self):
        self.status_text.delete(1.0, tk.END)

    # ── Password ──────────────────────────────────────────────────────────────
    def check_password(self):
        import hashlib
        PASSWORD_HASH = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"
        MAX_ATTEMPTS  = 3
        self.attempts = 0

        dlg = tk.Toplevel(self.root)
        dlg.title("🔒 Secure Login")
        dlg.geometry("400x300")
        dlg.configure(bg="#f5f7fa")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set(); dlg.focus_force(); dlg.lift()
        dlg.attributes('-topmost', True)
        dlg.update_idletasks()
        x = (dlg.winfo_screenwidth()-400)//2
        y = (dlg.winfo_screenheight()-300)//2
        dlg.geometry(f"+{x}+{y}")
        dlg.protocol("WM_DELETE_WINDOW", lambda: self.root.quit())

        mf = tk.Frame(dlg, bg="white", padx=25, pady=20)
        mf.pack(pady=20, padx=20, fill="both", expand=True)
        tk.Label(mf, text="🔐 Admin Authentication", font=("Segoe UI",14,"bold"),
                 bg="white", fg="#2c3e50").pack()
        tk.Label(mf, text="Enter your admin password to continue.",
                 font=("Segoe UI",10), bg="white", fg="#7f8c8d").pack(pady=(0,16))

        pv = tk.StringVar()
        pe = tk.Entry(mf, textvariable=pv, show="•", font=("Segoe UI",11))
        pe.pack(fill="x", pady=(0,8)); pe.focus_set()

        sl = tk.Label(mf, text="", fg="#e74c3c", font=("Segoe UI",9), bg="white")
        sl.pack()

        def verify():
            self.attempts += 1
            if hashlib.sha256(pv.get().encode()).hexdigest() == PASSWORD_HASH:
                dlg.destroy()
                self.root.lift(); self.root.focus_force()
            else:
                rem = MAX_ATTEMPTS - self.attempts
                if rem > 0:
                    sl.config(text=f"❌ Incorrect. {rem} attempt(s) left")
                    pv.set(""); pe.focus_set()
                else:
                    messagebox.showerror("Access Denied","Max attempts reached.", parent=dlg)
                    self.root.quit()

        tk.Button(mf, text="Unlock System", font=("Segoe UI",10,"bold"),
                  bg="#3498db", fg="white", padx=10, pady=6, relief="flat",
                  command=verify).pack(fill="x", pady=(12,0))
        tk.Label(mf, text="For authorized personnel only",
                 font=("Segoe UI",8), fg="#bdc3c7", bg="white").pack(side="bottom", pady=(8,0))
        dlg.bind('<Return>', lambda e: verify())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    try: root.iconbitmap(default='icon.ico')
    except: pass
    app = MySQLExcelUploaderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()