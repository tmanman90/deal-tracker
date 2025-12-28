import streamlit as st
import pandas as pd
import altair as alt
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
from datetime import datetime, date
import re

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DEAL TRACKER // TERMINAL",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS: VINTAGE TERMINAL THEME
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0');

    /* MAIN TERMINAL BACKGROUND */
    .stApp {
        background-color: #050a0e;
        color: #33ff00;
        /* DO NOT set font-family here (it breaks icon ligatures) */
    }
    
    /* CRT SCANLINE EFFECT OVERLAY */
    .stApp::before {
        content: " ";
        display: block;
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
        z-index: 9999;
        background-size: 100% 2px, 3px 100%;
        pointer-events: none;
    }

    /* TERMINAL TYPOGRAPHY (TEXT ONLY) */
    h1, h2, h3, h4, h5, h6,
    p, label, li, a,
    .stMarkdown, .stMarkdown *,
    .stTextInput input, .stTextArea textarea,
    .stButton button,
    div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"],
    .stDataFrame {
        font-family: 'Courier New', Courier, monospace !important;
        text-shadow: 0 0 2px #33ff00aa;
    }
    
    /* Make sure expander icons keep their icon font */
    div[data-testid="stExpander"] summary [aria-hidden="true"],
    div[data-testid="stExpander"] summary span[class*="material"],
    div[data-testid="stExpander"] summary i,
    div[data-testid="stExpander"] summary > svg {
        font-family: "Material Symbols Rounded", "Material Icons" !important;
        font-variation-settings: 'opsz' 24, 'wght' 400, 'FILL' 0, 'GRAD' 0;
        text-shadow: none !important;
        opacity: 1 !important;
    }

    h1 { color: #33ff00; letter-spacing: 2px; text-transform: uppercase; border-bottom: 2px solid #33ff00; padding-bottom: 10px; }
    h2, h3 { color: #ffbf00; } /* Amber secondary */
    
    /* DATAFRAME STYLING */
    .stDataFrame {
        border: 1px solid #33ff00;
        box-shadow: 0 0 10px #33ff00aa;
    }

    /* METRIC CARDS */
    div[data-testid="stMetric"] {
        background-color: #0d1117;
        border: 1px solid #33ff00;
        padding: 10px;
        box-shadow: 2px 2px 0px #33ff00;
    }
    div[data-testid="stMetricLabel"] { color: #ffbf00 !important; }
    div[data-testid="stMetricValue"] { color: #33ff00 !important; font-weight: bold; }

    /* WIDGETS */
    .stButton > button {
        background-color: #000;
        color: #33ff00;
        border: 1px solid #33ff00;
        text-transform: uppercase;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #33ff00;
        color: #000;
        box-shadow: 0 0 10px #33ff00;
    }
    
    /* INPUTS */
    div[data-baseweb="input"] {
        background-color: #0d1117;
        border: 1px solid #33ff00;
        color: #33ff00;
    }
    
    /* PROGRESS BAR */
    .stProgress > div > div > div > div {
        background-color: #b026ff; /* Neon Purple */
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #000;
        border-right: 1px solid #33ff00;
    }
    
    /* DIAGNOSTIC BOX STYLE */
    .diagnostic-box {
        background-color: #0d1117;
        border: 1px solid #33ff00;
        padding: 15px;
        border-radius: 5px;
        color: #e6ffff; /* Very bright cyan/white */
        font-size: 1.1rem; /* Larger font */
        line-height: 1.6;
        box-shadow: 0 0 5px #33ff00aa;
    }
    .diagnostic-label {
        color: #ffbf00; /* Amber for labels */
        font-weight: bold;
    }
    .diagnostic-value {
        color: #33ff00; /* Neon green for values */
        font-weight: bold;
    }

    /* TICKER TAPE */
    .ticker-wrap {
        width: 100%;
        overflow: hidden;
        background-color: #000;
        border-bottom: 1px solid #33ff00;
        padding: 5px 0;
        margin-bottom: 20px;
        white-space: nowrap; 
    }
    .ticker {
        display: inline-block;
        white-space: nowrap;
        /* Seamless loop: move from 0 to -50% (assuming content is duplicated once) */
        animation: ticker 120s linear infinite; 
        will-change: transform;
        transform: translate3d(0, 0, 0);
        backface-visibility: hidden;
        perspective: 1000px;
    }
    .ticker-item {
        display: inline-block;
        padding: 0 2rem;
        font-size: 1.2rem;
        color: #33ff00;
    }
    @keyframes ticker {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }

    /* CUSTOM DEAL ROW STYLE */
    .deal-row {
        background-color: #0d1117;
        border: 1px solid #33ff00;
        margin-bottom: 10px;
        padding: 10px;
        transition: transform 0.1s;
    }
    .deal-row:hover {
        transform: scale(1.01);
        box-shadow: 0 0 10px #33ff00;
        border: 1px solid #e6ffff;
    }
    .deal-stat {
        font-size: 0.9rem;
        color: #888;
    }
    .deal-val-green { color: #33ff00; font-weight: bold; font-size: 1.1rem; }
    .deal-val-red { color: #ff3333; font-weight: bold; font-size: 1.1rem; }
    .deal-val-amber { color: #ffbf00; font-weight: bold; font-size: 1.1rem; }
    .deal-title { font-size: 1.3rem; font-weight: bold; color: #e6ffff; text-transform: uppercase; }
    
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# AUTH & DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_data():
    """
    Connects to Google Sheets via Secrets and returns cleaned DataFrames.
    Uses 'safe read' to ignore empty columns causing duplicate header errors.
    """
    try:
        # Load Secrets
        scope = ['https://www.googleapis.com/auth/spreadsheets']
        credentials_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(credentials_dict, scopes=scope)
        client = gspread.authorize(creds)
        sheet_id = st.secrets["deal_tracker_sheet_id"]
        
        # Open Workbook
        sh = client.open_by_key(sheet_id)

        # HELPER: Safe read that drops empty columns
        def safe_read_sheet(ws):
            data = ws.get_all_values()
            if not data:
                return pd.DataFrame()
            
            # Clean headers: strip whitespace to avoid key errors (e.g. "Artist " -> "Artist")
            headers = [str(h).strip() for h in data[0]]
            
            # Create DF from raw data
            df = pd.DataFrame(data[1:], columns=headers)
            
            # Remove any columns that have an empty string as a header
            # This fixes "duplicate header" errors from trailing empty columns
            if '' in df.columns:
                df = df.drop(columns=[''], axis=1)
                
            return df
        
        # 1. READ DASHBOARD
        ws_dash = sh.worksheet("DASHBOARD")
        df_dash = safe_read_sheet(ws_dash)
        
        # 2. READ ACTUALS
        ws_act = sh.worksheet("ACTUALS")
        df_act = safe_read_sheet(ws_act)
        
        # 3. READ DEALS (For Analyzer Data)
        try:
            ws_deals = sh.worksheet("DEALS")
            df_deals = safe_read_sheet(ws_deals)
        except:
            df_deals = pd.DataFrame() # Fallback if sheet missing
        
        return df_dash, df_act, df_deals
        
    except Exception as e:
        st.error(f"SYSTEM FAILURE: Connection Refused. {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# -----------------------------------------------------------------------------
# CLEANING & LOGIC UTILS
# -----------------------------------------------------------------------------
def clean_currency(val):
    """Converts currency strings to floats."""
    if pd.isna(val) or str(val).strip() == "":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    # Remove common currency symbols
    clean_str = str(val).replace('$', '').replace(',', '').replace(' ', '')
    try:
        return float(clean_str)
    except:
        return 0.0

def clean_percent(val):
    """Converts '50%', 50, or 0.5 to a 0-1 float."""
    if pd.isna(val) or str(val).strip() == "":
        return 0.0
    if isinstance(val, (int, float)):
        return val if val <= 1.5 else val / 100.0 # Heuristic: if > 1.5, assume it's whole number percent
    s = str(val).strip().replace('%', '')
    try:
        f = float(s)
        # If input was "45", returns 0.45. If "0.45", returns 0.45
        return f / 100.0 if f > 1.5 else f
    except:
        return 0.0

def parse_flexible_date(date_str):
    """
    Tries multiple date formats to handle mixed '2025' and '25' years.
    Returns NaT if all fail.
    """
    if pd.isna(date_str) or str(date_str).strip() == "":
        return pd.NaT
    
    date_str = str(date_str).strip()
    
    # List of formats to try
    formats = [
        '%m/%d/%Y',  # 09/30/2025
        '%m/%d/%y',  # 09/30/25
        '%Y-%m-%d',  # 2025-09-30
        '%d-%b-%y',  # 30-Sep-25
    ]
    
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
            
    # Last resort: let pandas guess
    return pd.to_datetime(date_str, errors='coerce')

def calculate_pace_metrics(row, count, current_date_override=None, recent_velocity=0.0):
    """
    Calculates Grade and Pace based on Benchmark.
    Includes 'Half-Month Mulligan' for the initial period.
    Handles 'Legacy' deals (missing Analyzer data) by defaulting to Executed Advance + 12 Months.
    Uses current_date_override (latest data date) as 'today' for month calculation.
    
    Implements 'Velocity Override': If recent velocity suggests strong performance, it overrides cumulative pace.
    
    Returns (Grade, Pace Ratio, Eligible Boolean, Elapsed Months, Expected Progress, Is_Legacy_Flag).
    """
    # 1. Eligibility Check (Requires 3 data points)
    if count < 3:
        return "N/A", 0.0, False, 0, 0.0, False
    
    # 2. Determine Target Amount & Timeline (Legacy Fallback Logic)
    
    target_amount = 0.0
    target_months = 12.0 # Default
    is_legacy = False
    
    # Strategy & Selected Advance from DEALS sheet (merged)
    strat = str(row.get('Selected Strategy', '')).strip()
    sel_adv_raw = row.get('Selected Advance', '')
    sel_adv = clean_currency(sel_adv_raw)
    
    # Explicit Logic:
    # If we have a Strategy AND a valid Selected Advance > 0, it is an ANALYZER deal.
    # Otherwise, it is a LEGACY deal.
    
    if strat and sel_adv > 0:
        # ANALYZER MODE
        is_legacy = False
        target_amount = sel_adv
        
        # Use Label Breakeven Months if available, else 12
        lbm = row.get('Label Breakeven Months', 12)
        try:
            val = float(str(lbm).replace(',','').strip())
            if val > 0:
                target_months = val
            else:
                target_months = 12.0
        except:
            target_months = 12.0
    else:
        # LEGACY MODE
        is_legacy = True
        # Use Executed Advance as the target
        target_amount = clean_currency(row.get('Executed Advance', 0))
        target_months = 12.0

    # 3. Parse Dates
    try:
        if 'Forecast Start Date' not in row:
             return "N/A", 0.0, False, 0, 0.0, False
             
        forecast_start = parse_flexible_date(row['Forecast Start Date'])
        if pd.isna(forecast_start):
            return "N/A", 0.0, False, 0, 0.0, False
    except:
        return "N/A", 0.0, False, 0, 0.0, False
        
    # 4. Calculate Elapsed Months vs Benchmark (Whole Months Logic)
    # Use override date (latest actuals date) if provided, else today
    today = current_date_override if current_date_override else pd.Timestamp.now()
    
    if forecast_start > today:
        elapsed_months = 0.0
    else:
        # Whole Month Calculation: (YearDiff * 12) + MonthDiff + 1 (inclusive)
        # e.g. Dec 2025 - Oct 2025 = (0 * 12) + (12 - 10) + 1 = 3 Months
        month_diff = (today.year - forecast_start.year) * 12 + (today.month - forecast_start.month)
        elapsed_months = float(month_diff + 1)
    
    elapsed_months = max(1.0, elapsed_months)
    
    # --- THE HALF-MONTH MULLIGAN ---
    # Subtract 0.5 months from the timeline to forgive the initial 'trickle'.
    effective_months = max(0.0, elapsed_months - 0.5)
    
    # Calculate expected progress based on this effective time
    if target_months > 0:
        expected_progress = effective_months / target_months
    else:
        expected_progress = 0
        
    # Cap at 1.0 (100%)
    expected_progress = min(1.0, expected_progress)
    
    # 5. Actual progress (Cumulative)
    # UPDATED: Always use Executed Advance as target amount for actual progress
    target_amount_for_grading = clean_currency(row.get('Executed Advance', 0))
    cum_receipts = clean_currency(row.get('Cum Receipts', 0))
    
    if target_amount_for_grading > 0:
        actual_progress = cum_receipts / target_amount_for_grading
    else:
        actual_progress = 0.0
    
    # Cumulative Pace Ratio
    if expected_progress == 0:
        cumulative_ratio = 0
    else:
        cumulative_ratio = actual_progress / expected_progress

    # --- VELOCITY OVERRIDE LOGIC ---
    # Calculate Velocity Pace
    velocity_ratio = 0.0
    if count >= 3 and target_amount_for_grading > 0:
        # Extrapolate last 3 mo avg to 12 months
        projected_annual = recent_velocity * 12.0
        # Compare to Target (Executed Advance)
        velocity_ratio = projected_annual / target_amount_for_grading

    # Final Pace Ratio: Best of Cumulative vs Velocity
    pace_ratio = max(cumulative_ratio, velocity_ratio)
        
    # --- GRADING BANDS (THE SAFE BET SCALE) ---
    # A++: >= 2.0 (Recoup in 6 months or less)
    if pace_ratio >= 2.0: grade = "A++"
    elif pace_ratio >= 1.15: grade = "A+"
    elif pace_ratio >= 1.00: grade = "A"
    elif pace_ratio >= 0.90: grade = "B+"
    elif pace_ratio >= 0.75: grade = "B"
    elif pace_ratio >= 0.60: grade = "C"
    elif pace_ratio >= 0.40: grade = "D"
    else: grade = "F"
        
    return grade, pace_ratio, True, elapsed_months, expected_progress, is_legacy

def sanitize_terminal_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # remove zero-width + BOM + NBSP
    s = s.replace("\u00a0", " ").replace("\ufeff", "")
    s = re.sub(r"[\u200b-\u200d]", "", s)  # zero width
    # kill newlines/tabs that can cause weird wraps
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return s.strip()

# -----------------------------------------------------------------------------
# DATA PROCESSING WRAPPER
# -----------------------------------------------------------------------------
def process_data(df_dash, df_act, df_deals):
    # Ensure required columns exist
    if df_dash.empty:
        return df_dash, df_act, None

    # MERGE DEALS DATA INTO DASHBOARD IF AVAILABLE
    if not df_deals.empty:
        # Normalize IDs for merge
        if 'Deal ID' in df_dash.columns:
            df_dash['Deal ID Str'] = df_dash['Deal ID'].astype(str).str.strip()
        
        if 'Deal ID' in df_deals.columns:
            df_deals['Deal ID Str'] = df_deals['Deal ID'].astype(str).str.strip()
            
            # Columns to bring over, including 'Tags', 'Artist Share', 'Is JV'
            cols_to_merge = ['Selected Strategy', 'Selected Advance', 'Label Breakeven Months', 'Tags', 'Artist Share', 'Is JV']
            cols_existing = [c for c in cols_to_merge if c in df_deals.columns]
            
            if cols_existing and 'Deal ID Str' in df_dash.columns:
                # Merge
                df_merged = df_dash.merge(df_deals[['Deal ID Str'] + cols_existing], on='Deal ID Str', how='left', suffixes=('', '_y'))
                
                # Cleanup
                for c in cols_existing:
                    if f'{c}_y' in df_merged.columns:
                        df_merged[c] = df_merged[f'{c}_y'].fillna(df_merged[c])
                        df_merged.drop(columns=[f'{c}_y'], inplace=True)
                
                # Fill NaN tags with empty string
                if 'Tags' in df_merged.columns:
                    df_merged['Tags'] = df_merged['Tags'].fillna('')
                
                df_dash = df_merged

    # --- TAG PARSING (NEW) ---
    if 'Tags' not in df_dash.columns:
        df_dash['Tags'] = ''
        
    # Helper to parse tags
    def parse_tags_list(val):
        if pd.isna(val) or str(val).strip() == "":
            return []
        parts = [p.strip() for p in str(val).split(',')]
        return [p for p in parts if p]

    df_dash['Tags List'] = df_dash['Tags'].apply(parse_tags_list)

    # --- CALCULATE LABEL SHARE PCT & IS JV LOGIC ---
    # 1. Ensure columns exist if merge missed them
    if 'Artist Share' not in df_dash.columns:
        df_dash['Artist Share'] = 0.0
    if 'Is JV' not in df_dash.columns:
        df_dash['Is JV'] = False
        
    # 2. Normalize Artist Share
    df_dash['Artist Share Pct'] = df_dash['Artist Share'].apply(clean_percent)
    
    # 3. Normalize Is JV (Boolean)
    def clean_jv_bool(val):
        s = str(val).strip().upper()
        # Common boolean string representations in Sheets
        return s in ['TRUE', 'YES', '1', 'ON', 'CHECKED']

    df_dash['Is JV Clean'] = df_dash['Is JV'].apply(clean_jv_bool)
    
    # 4. Compute Label Share Pct
    def compute_label_share(row):
        asp = row.get('Artist Share Pct', 0.0)
        
        # Missing-data rule: If Artist Share is blank/0, return NaN
        # (clean_percent returns 0.0 for blanks/zeros)
        if asp <= 0.0001: 
            return np.nan
            
        # Label Share Pct = 1 - Artist Share Pct
        label_share = 1.0 - asp
        
        # If Is JV is true -> Label Share Pct = Label Share Pct * 0.5
        if row.get('Is JV Clean', False):
            label_share = label_share * 0.5
            
        return label_share

    df_dash['Label Share Pct'] = df_dash.apply(compute_label_share, axis=1)

    # 1. GLOBAL STRIP: Clean Deal IDs
    if 'Deal ID' in df_act.columns:
        df_act['Deal ID'] = df_act['Deal ID'].astype(str).str.strip()
    if 'Deal ID' in df_dash.columns:
        df_dash['Deal ID'] = df_dash['Deal ID'].astype(str).str.strip()

    # ADD NORMALIZED IDs (FIX for Market Pulse & consistent joins)
    if 'Deal ID' in df_act.columns:
        df_act['did_norm'] = df_act['Deal ID'].astype(str).str.replace('\u00a0', ' ').str.strip()
    if 'Deal ID' in df_dash.columns:
        df_dash['did_norm'] = df_dash['Deal ID'].astype(str).str.replace('\u00a0', ' ').str.strip()

    # Process Actuals Dates & Values
    if 'Net Receipts' in df_act.columns:
        df_act['Net Receipts'] = df_act['Net Receipts'].apply(clean_currency)
    
    # ROBUST DATE CLEANING FOR ACTUALS
    if 'Period End Date' in df_act.columns:
        df_act['Period End Date'] = df_act['Period End Date'].apply(parse_flexible_date)
        
    # --- SMART START DATE LOGIC START ---
    # Map Deal ID -> Earliest Actual Date (Snapped to 1st of Month)
    actual_starts = {}
    if not df_act.empty and 'Period End Date' in df_act.columns:
        # Drop NaTs just in case
        valid_dates = df_act.dropna(subset=['Period End Date'])
        if not valid_dates.empty:
            # Find min date per deal
            min_dates = valid_dates.groupby('Deal ID')['Period End Date'].min()
            # Snap to 1st of month: 10/31/2025 -> 10/01/2025
            # This ensures duration math counts the full first month
            actual_starts = min_dates.apply(lambda x: x.replace(day=1)).to_dict()
            
    # Override Forecast Start Date in Dashboard
    def smart_start_date(row):
        did = str(row.get('Deal ID', '')).strip()
        manual_date = row.get('Forecast Start Date')
        
        # Use Actuals if available, otherwise fallback to Manual
        if did in actual_starts:
            return actual_starts[did]
        return manual_date
        
    df_dash['Forecast Start Date'] = df_dash.apply(smart_start_date, axis=1)
    # --- SMART START DATE LOGIC END ---
    
    # Determine "Latest Data Date" from Actuals
    # This will be used as "Today" for calculating elapsed months
    current_date_override = None
    if not df_act.empty and 'Period End Date' in df_act.columns:
         valid_dates_all = df_act.dropna(subset=['Period End Date'])
         if not valid_dates_all.empty:
             current_date_override = valid_dates_all['Period End Date'].max()
             
    # Calculate Recent Velocity Map (Last 3 Months Avg per Deal)
    # AND COMPUTE PRINTER METRICS
    velocity_map = {}
    lifetime_map = {} # New map for Lifetime Average
    
    # Compute Printer metrics
    printer_metrics = {} # Key: did, Value: dict of metrics
    
    def detect_trickle_first_month(receipts_list):
        """
        Approach A (Trickle detection):
        - Floor rule: Month1 <= 50
        - Explosive jump: Month2 >= 5x Month1 (but only if Month2 is meaningful)
        - Tiny vs early baseline: Month1 < 10% of median(M2..M4), if baseline exists & meaningful
        Returns: (is_trickle: bool, reason: str)
        """
        if not receipts_list or len(receipts_list) < 2:
            return False, ""

        m1 = float(receipts_list[0] or 0.0)
        m2 = float(receipts_list[1] or 0.0)

        TRICKLE_FLOOR = 50.0
        JUMP_MULT = 5.0

        # Guardrails to reduce false flags from tiny numbers
        MIN_MEANINGFUL_M2_FOR_JUMP = 500.0      # prevents "80 -> 420" being flagged
        MIN_BASELINE_FOR_TINY_TEST = 300.0      # prevents weird tiny baselines

        # Rule 1: Hard floor
        if m1 <= TRICKLE_FLOOR:
            return True, f"TRICKLE_FLOOR (M1=${m1:,.0f} <= ${TRICKLE_FLOOR:,.0f})"

        # Rule 2: Explosive jump (only if M2 is meaningful and M1 > 0)
        if m1 > 0 and m2 >= MIN_MEANINGFUL_M2_FOR_JUMP and (m2 / m1) >= JUMP_MULT:
            return True, f"EXPLOSIVE_JUMP (M2=${m2:,.0f} is {(m2/m1):.1f}x M1=${m1:,.0f})"

        # Rule 3: Tiny vs early baseline (median of M2..M4)
        baseline_window = receipts_list[1:4]  # months 2-4 (up to 3 values)
        baseline_vals = [float(x) for x in baseline_window if x is not None]
        if baseline_vals:
            baseline_med = float(np.median(baseline_vals))
            if baseline_med >= MIN_BASELINE_FOR_TINY_TEST and m1 < 0.10 * baseline_med:
                return True, f"TINY_VS_BASELINE (M1=${m1:,.0f} < 10% of median(M2..M4)=${baseline_med:,.0f})"

        return False, ""

    if not df_act.empty and 'Period End Date' in df_act.columns:
         valid_dates_act = df_act.dropna(subset=['Period End Date']).sort_values('Period End Date')
         if not valid_dates_act.empty:
             # Calculate metrics per deal
             for did, group in valid_dates_act.groupby('Deal ID'):
                 group = group.sort_values('Period End Date')
                 months_count = len(group)
                 
                 last_month_val = 0.0
                 prev_month_val = 0.0
                 sma3 = 0.0
                 printer_score = 0.0
                 mom_pct = 0.0
                 
                 if months_count > 0:
                     last_month_val = group.iloc[-1]['Net Receipts']
                 
                 if months_count >= 2:
                     prev_month_val = group.iloc[-2]['Net Receipts']
                     if prev_month_val > 0:
                         mom_pct = (last_month_val - prev_month_val) / prev_month_val
                     else:
                         mom_pct = 0.0
                 
                 # --- TRICKLE DETECTION (Month 1 only) ---
                 receipts_list = group["Net Receipts"].astype(float).tolist()
                 is_trickle, trickle_reason = detect_trickle_first_month(receipts_list)
                 
                 # --- CALCULATE LIFETIME AVERAGE ---
                 if receipts_list:
                     lifetime_avg = float(np.mean(receipts_list))
                 else:
                     lifetime_avg = 0.0
                 lifetime_map[did] = lifetime_avg

                 # --- SMA3 RAW vs ADJ ---
                 # RAW: last 3 months as-is
                 if months_count >= 3:
                     sma3_raw = float(np.mean(receipts_list[-3:]))
                 elif months_count > 0:
                     sma3_raw = float(np.mean(receipts_list))
                 else:
                     sma3_raw = 0.0

                 # ADJ: exclude Month 1 if trickle detected
                 # (use up to last 3 months of the remaining series; if only 1 month remains, it's that value)
                 if is_trickle and months_count >= 2:
                     adj_series = receipts_list[1:]  # drop Month 1
                 else:
                     adj_series = receipts_list

                 if len(adj_series) >= 3:
                     sma3_adj = float(np.mean(adj_series[-3:]))
                 elif len(adj_series) > 0:
                     sma3_adj = float(np.mean(adj_series))
                 else:
                     sma3_adj = 0.0

                 # Effective SMA used by the app (keeps existing logic intact)
                 sma3_effective = sma3_adj if is_trickle else sma3_raw

                 # Printer score uses effective SMA
                 printer_score = (last_month_val / sma3_effective) if sma3_effective > 0 else 0.0
                 
                 # Velocity map used by Pace Metrics uses effective SMA
                 velocity_map[did] = sma3_effective
                 
                 # Printer Eligibility
                 # (MonthsCount >= 4) AND (SMA3 >= 500)
                 is_printer = (months_count >= 4) and (sma3_effective >= 500)
                 
                 printer_metrics[did] = {
                     'MonthsCount': months_count,
                     'LastMonth': last_month_val,
                     'PrevMonth': prev_month_val,
                     'SMA3': sma3_effective,          # <— keep your existing downstream references working
                     'SMA3_RAW': sma3_raw,            # <— add
                     'SMA3_ADJ': sma3_adj,            # <— add
                     'PrinterScore': printer_score,
                     'MoM_pct': mom_pct,
                     'IsPrinterEligible': is_printer,
                     'TrickleDetected': is_trickle,
                     'TrickleReason': trickle_reason
                 }
    
    # --- NEW RECOUPMENT MAP LOGIC ---
    # Find the date when cumulative receipts >= executed advance
    recoupment_map = {}
    if not df_act.empty and 'Period End Date' in df_act.columns:
        valid_dates_act = df_act.dropna(subset=['Period End Date']).sort_values('Period End Date')
        if not valid_dates_act.empty:
             # We need executed advance for comparison. 
             # It's cleaner to iterate deals from dash and filter act
             for _, row in df_dash.iterrows():
                 did = str(row.get('Deal ID', ''))
                 exec_adv = clean_currency(row.get('Executed Advance', 0))
                 
                 if exec_adv > 0:
                     deal_txns = valid_dates_act[valid_dates_act['Deal ID'] == did].copy()
                     if not deal_txns.empty:
                         deal_txns['Cum'] = deal_txns['Net Receipts'].cumsum()
                         # Find first row where Cum >= Exec Adv
                         recouped_rows = deal_txns[deal_txns['Cum'] >= exec_adv]
                         if not recouped_rows.empty:
                             recoupment_map[did] = recouped_rows.iloc[0]['Period End Date']

    # Clean Dashboard Numerics
    numeric_cols = ['Executed Advance', 'Cum Receipts', 'Remaining to BE']
    for col in numeric_cols:
        if col in df_dash.columns:
            df_dash[col] = df_dash[col].apply(clean_currency)
        else:
            df_dash[col] = 0.0
            
    # --- COMPUTE LBL CUM ($) ---
    # LBL Cum = Cum Receipts * Label Share Pct
    # Do this AFTER 'Cum Receipts' is guaranteed numeric (clean_currency above)
    if 'Label Share Pct' in df_dash.columns and 'Cum Receipts' in df_dash.columns:
        df_dash['LBL Cum'] = np.where(
            pd.notna(df_dash['Label Share Pct']),
            df_dash['Cum Receipts'] * df_dash['Label Share Pct'],
            np.nan
        )
    else:
        df_dash['LBL Cum'] = np.nan
    
    # Calculate Data Eligibility
    eligibility_map = {}
    if not df_act.empty:
        if 'Deal ID' in df_act.columns:
            # Count ALL rows matching the ID.
            counts = df_act.groupby('Deal ID').size()
            eligibility_map = counts.to_dict()
    
    # Enrich Dashboard with Grades
    grades = []
    ratios = []
    is_eligible = []
    elapsed_list = []
    data_points_list = []
    expected_list = []
    legacy_list = []
    recent_velocity_list = []
    lifetime_avg_list = [] # New list for Lifetime Avg
    recouped_date_list = [] # Store recoupment date
    
    # Printer Metric Lists
    pm_months_count = []
    pm_last_month = []
    pm_prev_month = []
    pm_sma3 = []
    pm_sma3_raw = []
    pm_sma3_adj = []
    pm_score = []
    pm_mom = []
    pm_eligible = []
    pm_trickle = []
    pm_trickle_reason = []
    
    for _, row in df_dash.iterrows():
        did = str(row.get('Deal ID', ''))
        
        # Get count (default 0)
        count = eligibility_map.get(did, 0)
        
        # Get velocity (default 0)
        recent_vel = velocity_map.get(did, 0.0)
        
        # Get lifetime avg (default 0)
        lifetime_val = lifetime_map.get(did, 0.0)
        
        # Get printer metrics
        pm = printer_metrics.get(did, {
            'MonthsCount': 0, 'LastMonth': 0.0, 'PrevMonth': 0.0, 
            'SMA3': 0.0, 'SMA3_RAW': 0.0, 'SMA3_ADJ': 0.0, 'PrinterScore': 0.0, 'MoM_pct': 0.0, 'IsPrinterEligible': False,
            'TrickleDetected': False, 'TrickleReason': ''
        })
        
        pm_months_count.append(pm['MonthsCount'])
        pm_last_month.append(pm['LastMonth'])
        pm_prev_month.append(pm['PrevMonth'])
        pm_sma3.append(pm['SMA3'])
        pm_sma3_raw.append(pm.get('SMA3_RAW', 0.0))
        pm_sma3_adj.append(pm.get('SMA3_ADJ', 0.0))
        pm_score.append(pm['PrinterScore'])
        pm_mom.append(pm['MoM_pct'])
        pm_eligible.append(pm['IsPrinterEligible'])
        pm_trickle.append(pm['TrickleDetected'])
        pm_trickle_reason.append(pm['TrickleReason'])
        
        # Determine specific "Today" for this deal
        # If recouped, freeze time at recoupment date
        recoup_date = recoupment_map.get(did)
        
        if recoup_date:
            deal_calc_date = recoup_date
        else:
            deal_calc_date = current_date_override
        
        # Now returns 6 values including is_legacy
        # Pass deal_calc_date to fix "fractional month" / freeze logic
        g, r, e, el_m, exp_prog, is_leg = calculate_pace_metrics(row, count, deal_calc_date, recent_velocity=recent_vel)
        
        grades.append(g)
        ratios.append(r)
        is_eligible.append(e)
        elapsed_list.append(el_m)
        data_points_list.append(count)
        expected_list.append(exp_prog)
        legacy_list.append(is_leg)
        recent_velocity_list.append(recent_vel)
        lifetime_avg_list.append(lifetime_val)
        recouped_date_list.append(recoup_date) # Can be NaT/None
        
    df_dash['Grade'] = grades
    df_dash['Pace Ratio'] = ratios
    df_dash['Is Eligible'] = is_eligible
    df_dash['Elapsed Months'] = elapsed_list
    df_dash['Data Points Found'] = data_points_list 
    df_dash['Expected Recoupment'] = expected_list
    df_dash['Is Legacy'] = legacy_list
    df_dash['Recent Velocity'] = recent_velocity_list
    df_dash['Lifetime Avg'] = lifetime_avg_list
    df_dash['Recoupment Date'] = recouped_date_list
    
    # Add Printer Columns
    df_dash['MonthsCount'] = pm_months_count
    df_dash['LastMonth'] = pm_last_month
    df_dash['PrevMonth'] = pm_prev_month
    df_dash['SMA3'] = pm_sma3
    df_dash['SMA3_RAW'] = pm_sma3_raw
    df_dash['SMA3_ADJ'] = pm_sma3_adj
    df_dash['PrinterScore'] = pm_score
    df_dash['MoM_pct'] = pm_mom
    df_dash['IsPrinterEligible'] = pm_eligible
    df_dash['TrickleDetected'] = pm_trickle
    df_dash['TrickleReason'] = pm_trickle_reason
    
    # UPDATED: Always set Target Amount to Executed Advance for dashboard display
    df_dash['Target Amount'] = df_dash['Executed Advance']
    df_dash['% to BE Clean'] = df_dash.apply(lambda r: (r['Cum Receipts']/r['Target Amount']) if r['Target Amount'] > 0 else 0, axis=1)
    
    return df_dash, df_act, current_date_override

# -----------------------------------------------------------------------------
# UI: PORTFOLIO PAGE
# -----------------------------------------------------------------------------
def show_portfolio(df_dash, df_act, current_date_override):
    st.title(">>> GLOBAL DEAL TRACKER_")
    
    # --- DEBUG DISPLAY: REPORTING DATE ---
    if current_date_override:
        current_date_str = current_date_override.strftime('%Y-%m-%d')
        # st.caption(f"REPORTING DATE: {current_date_str}")
    
    # --- TICKER TAPE ---
    ticker_items = []
    if not df_dash.empty:
        if 'Pace Ratio' in df_dash.columns:
            sorted_deals = df_dash.sort_values('Pace Ratio', ascending=False)
        else:
            sorted_deals = df_dash
            
        for _, row in sorted_deals.iterrows():
            artist_name = row.get('Artist / Project', row.get('Artist', row.get('Project', 'Unknown')))
            
            symbol = "▲" if row.get('Pace Ratio', 0) >= 1.0 else "▼"
            pct = row.get('% to BE Clean', 0)
            if pd.isna(pct): pct = 0.0
            
            item = f"{artist_name} ({row.get('Grade', 'N/A')}) {symbol} {pct*100:.1f}%"
            ticker_items.append(item)
    
    # Duplicate items for smoother infinite scroll illusion
    full_ticker_list = ticker_items + ticker_items 
    
    ticker_html = f"""
    <div class="ticker-wrap">
        <div class="ticker">
            <span class="ticker-item">{' &nbsp;&nbsp;&nbsp; /// &nbsp;&nbsp;&nbsp; '.join(full_ticker_list)}</span>
        </div>
    </div>
    """
    st.markdown(ticker_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- FILTERS ---
    col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 1])
    
    with col1:
        search = st.text_input("SEARCH ARTIST OR DEAL ID", "").lower()
        
    with col2:
        all_status = df_dash['Status'].unique().tolist() if 'Status' in df_dash.columns else []
        status_filter = st.multiselect("STATUS", all_status, default=all_status)
        
    with col3:
        eligible_only = st.checkbox("ELIGIBLE FOR GRADE", value=False)
        
    with col4:
        sort_opt = st.selectbox("SORT BY", ["Grade", "Remaining to BE", "% to BE", "Cum Receipts", "Delta Months"], index=0)

    # TAG FILTER LOGIC
    all_tags = set()
    if 'Tags List' in df_dash.columns:
        for tags in df_dash['Tags List']:
            all_tags.update(tags)
    all_tags = sorted(list(all_tags))
        
    with col5:
        tag_filter = st.multiselect("TAGS", all_tags, default=[])
        
    with col6:
        if "label_mode" not in st.session_state:
            st.session_state.label_mode = False
        label_mode_val = st.checkbox("LABEL MODE", value=st.session_state.label_mode)
        st.session_state.label_mode = label_mode_val
    
    # Filter Logic
    filtered = df_dash.copy()
    
    # Normalize Deal ID for debugging/robustness in filtering
    if 'Deal ID' in filtered.columns:
        filtered['did_norm'] = filtered['Deal ID'].astype(str).str.replace('\u00a0', ' ').str.strip()
    else:
        filtered['did_norm'] = ""
    
    # Requirement #2: Drop bad/empty Deal IDs immediately
    filtered = filtered[
        filtered['did_norm'].notna() & 
        (filtered['did_norm'] != "") & 
        (filtered['did_norm'].str.lower() != "nan")
    ]

    if search:
        mask = pd.Series([False] * len(filtered))
        # Handle Artist column name robustly
        if 'Artist / Project' in filtered.columns:
            mask = mask | filtered['Artist / Project'].astype(str).str.lower().str.contains(search)
        elif 'Artist' in filtered.columns:
            mask = mask | filtered['Artist'].astype(str).str.lower().str.contains(search)
            
        if 'Deal ID' in filtered.columns:
            mask = mask | filtered['Deal ID'].astype(str).str.lower().str.contains(search)
        filtered = filtered[mask]
        
    if status_filter and 'Status' in filtered.columns:
        filtered = filtered[filtered['Status'].isin(status_filter)]
        
    if eligible_only:
        filtered = filtered[filtered['Is Eligible'] == True]

    # Apply Tag Filter
    if tag_filter and 'Tags List' in filtered.columns:
        # keep rows where ANY selected tag is in Tags List
        filtered = filtered[filtered['Tags List'].apply(lambda x: any(t in x for t in tag_filter))]
        
    # Sort Logic
    ascending = False
    sort_col = None
    
    if sort_opt == "Remaining to BE":
        sort_col = "Remaining to BE"
    elif sort_opt == "% to BE":
        sort_col = "% to BE Clean"
    elif sort_opt == "Grade":
        # Custom Grade Sort - Updated A++
        grade_order = {"A++": 0, "A+": 1, "A": 2, "B+": 3, "B": 4, "C": 5, "D": 6, "F": 7, "WAITING": 8, "PENDING": 9, "N/A": 10}
        filtered['Grade_Rank'] = filtered['Grade'].map(grade_order).fillna(99)
        sort_col = "Grade_Rank"
        ascending = True
    elif sort_opt == "Cum Receipts":
        sort_col = "Cum Receipts"
    elif sort_opt == "Delta Months":
         sort_col = "Delta Months"

    if sort_col and sort_col in filtered.columns:
        if sort_col == "Delta Months":
             filtered['Delta Months'] = pd.to_numeric(filtered['Delta Months'], errors='coerce').fillna(0)
        filtered = filtered.sort_values(by=sort_col, ascending=ascending)

    # --- KPI CARDS ---
    
    active_deals = len(filtered)
    total_adv = filtered['Executed Advance'].sum() if 'Executed Advance' in filtered.columns else 0
    total_rec = filtered['Cum Receipts'].sum() if 'Cum Receipts' in filtered.columns else 0
    total_lbl_cum = filtered['LBL Cum'].fillna(0).sum() if 'LBL Cum' in filtered.columns else 0
    
    # Weighted Metrics Logic
    w_pct = 0.0
    if total_adv > 0:
        w_pct = (total_rec / total_adv) * 100
        
    w_grade = "N/A"
    if total_adv > 0 and 'Pace Ratio' in filtered.columns and 'Is Eligible' in filtered.columns:
        eligible_deals = filtered[filtered['Is Eligible'] == True]
        
        if not eligible_deals.empty:
            eligible_deals['Weighted Score'] = eligible_deals['Pace Ratio'] * eligible_deals['Executed Advance']
            
            total_eligible_adv = eligible_deals['Executed Advance'].sum()
            total_score = eligible_deals['Weighted Score'].sum()
            
            if total_eligible_adv > 0:
                overall_ratio = total_score / total_eligible_adv
                
                if overall_ratio >= 2.00: w_grade = "A++"
                elif overall_ratio >= 1.15: w_grade = "A+"
                elif overall_ratio >= 1.00: w_grade = "A"
                elif overall_ratio >= 0.90: w_grade = "B+"
                elif overall_ratio >= 0.80: w_grade = "B"
                elif overall_ratio >= 0.70: w_grade = "C+"
                elif overall_ratio >= 0.60: w_grade = "C"
                elif overall_ratio >= 0.50: w_grade = "D"
                else: w_grade = "F"
    
    if st.session_state.label_mode:
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        
        kpi1.metric("ACTIVE DEALS", active_deals)
        kpi2.metric("TOTAL ADVANCES", f"${total_adv:,.0f}")
        kpi3.metric("TOTAL CUM RECEIPTS", f"${total_rec:,.0f}")
        kpi4.metric("TOTAL LBL CUM", f"${total_lbl_cum:,.0f}")
        kpi5.metric("WEIGHTED RECOUPMENT", f"{w_pct:.1f}%")
        kpi6.metric("WEIGHTED GRADE", w_grade)
    else:
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        
        kpi1.metric("ACTIVE DEALS", active_deals)
        kpi2.metric("TOTAL ADVANCES", f"${total_adv:,.0f}")
        kpi3.metric("TOTAL CUM RECEIPTS", f"${total_rec:,.0f}")
        kpi4.metric("WEIGHTED RECOUPMENT", f"{w_pct:.1f}%")
        kpi5.metric("WEIGHTED GRADE", w_grade)
    
    st.markdown("---")
    
    # --- ROSTER TABLE (CUSTOM UI) ---
    st.markdown("### > SELECT DEAL TO INITIALIZE ANALYSIS")
    
    # Header Row
    st.markdown("""
    <div style="display: flex; border-bottom: 2px solid #33ff00; padding-bottom: 5px; margin-bottom: 10px; font-weight: bold; color: #ffbf00;">
        <div style="flex: 3;">ARTIST / PROJECT</div>
        <div style="flex: 1;">ID</div>
        <div style="flex: 1;">STATUS</div>
        <div style="flex: 1;">GRADE</div>
        <div style="flex: 1.2; text-align: right;">LBL CUM</div>
        <div style="flex: 1; text-align: right;">RECOUPED</div>
        <div style="flex: 1.5; text-align: right;">REMAINING</div>
        <div style="flex: 1;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    for i, row in enumerate(filtered.to_dict('records')):
        # Clean Data for Display
        artist = row.get('Artist / Project', row.get('Artist', 'Unknown'))
        
        # Check for Tags and append multiple badges
        tags_list = row.get('Tags List', [])
        tags_html = ""
        for t in tags_list:
            tags_html += f' <span style="font-size: 0.7em; border: 1px solid #33ff00; padding: 2px 6px; margin-left: 4px; border-radius: 4px; color: #33ff00;">{t}</span>'
            
        if tags_html:
            artist += tags_html
            
        did = row.get('did_norm', 'N/A')
        did_disp = row.get('Deal ID', 'N/A')
        status = row.get('Status', '-')
        
        grade = row.get('Grade', 'WAITING') if row.get('Is Eligible', False) else "PENDING"
        
        # Updated grade color logic
        if grade in ["A++", "A+", "A", "B+"]:
            grade_color = "#33ff00" # Green
        elif grade == "B":
            grade_color = "#ffbf00" # Amber (Warning)
        elif grade in ["C", "D", "F"]:
            grade_color = "#ff3333" # Red
        else:
            grade_color = "#888" # Grey/Default
        
        pct_val = row.get('% to BE Clean', 0)
        pct_str = f"{pct_val*100:.1f}%"
        
        rem_val = row.get('Remaining to BE', 0)
        rem_str = f"${rem_val:,.0f}" if isinstance(rem_val, (int, float)) else str(rem_val)
        
        # New LBL CUM logic
        lbl_val = row.get('LBL Cum', np.nan)
        if pd.isna(lbl_val):
            lbl_str = "—"
            lbl_color = "#888"
        else:
            lbl_str = f"${float(lbl_val):,.0f}"
            lbl_color = "#b026ff"
        
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([3, 1, 1, 1, 1.2, 1, 1.5, 1])
        
        with c1:
            st.markdown(f"<div style='padding-top: 5px; font-weight: bold; color: #e6ffff;'>{artist}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='padding-top: 5px; color: #888;'>{did_disp}</div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div style='padding-top: 5px;'>{status}</div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div style='padding-top: 5px; color: {grade_color}; font-weight: bold;'>{grade}</div>", unsafe_allow_html=True)
        with c5:
            st.markdown(f"<div style='padding-top: 5px; text-align: right; color: {lbl_color}; font-weight: bold;'>{lbl_str}</div>", unsafe_allow_html=True)
        with c6:
            st.markdown(f"<div style='padding-top: 5px; text-align: right; color: #33ff00;'>{pct_str}</div>", unsafe_allow_html=True)
        with c7:
            st.markdown(f"<div style='padding-top: 5px; text-align: right; color: #ffbf00;'>{rem_str}</div>", unsafe_allow_html=True)
        with c8:
            if st.button("OPEN", key=f"btn_open_{did}_{i}"):
                st.session_state['selected_deal_id'] = did
                st.rerun()
                
        st.markdown("<div style='border-bottom: 1px solid #333; margin-bottom: 5px;'></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------------------------
    # MARKET PULSE // BIG CHART (ELIGIBLE DEALS ONLY)
    # ---------------------------------------------------------------------
    st.markdown("### > MARKET PULSE // BIG CHART")

    # Guardrails
    if df_dash.empty:
        st.info("No DASHBOARD data loaded.")
    elif df_act.empty:
        st.info("No ACTUALS data loaded.")
    elif "Is Eligible" not in df_dash.columns:
        st.info("Missing 'Is Eligible' in DASHBOARD (grading not computed).")
    elif "did_norm" not in df_dash.columns or "did_norm" not in df_act.columns:
        st.info("Missing did_norm normalization. (process_data should create this.)")
    else:
        # 1) Eligible deals only
        eligible = df_dash[df_dash["Is Eligible"] == True].copy()

        if eligible.empty:
            st.info("No deals are eligible for grading yet (need 3+ actuals).")
        else:
            try:
                # 2) Default sort by Cum Receipts (DESC)
                eligible["Cum Receipts"] = pd.to_numeric(eligible["Cum Receipts"], errors="coerce").fillna(0)
                eligible = eligible.sort_values("Cum Receipts", ascending=False).copy()

                # 3) Build dropdown label
                eligible["ArtistName"] = eligible.apply(
                    lambda r: r.get("Artist / Project", r.get("Artist", r.get("Project", "UNKNOWN"))),
                    axis=1
                )

                eligible["Label"] = (
                    eligible["ArtistName"].astype(str)
                    + " | CUM $" + eligible["Cum Receipts"].map(lambda x: f"{x:,.0f}")
                    + " | [" + eligible["did_norm"].astype(str) + "]"
                )

                # CRITICAL FIX: Deduplicate based on Label to prevent st.selectbox crash
                eligible = eligible.drop_duplicates(subset=["Label"])

                label_to_id = dict(zip(eligible["Label"], eligible["did_norm"]))

                # Default selection = top Cum Receipts
                default_label = eligible["Label"].iloc[0]

                selected_label = st.selectbox(
                    "SELECT DEAL (ELIGIBLE ONLY) // DEFAULT = TOP CUM RECEIPTS",
                    eligible["Label"].tolist(),
                    index=0
                )
                selected_id = label_to_id[selected_label]

                # 4) Pull actuals for selected deal
                act = df_act[df_act["did_norm"] == selected_id].copy()

                if act.empty:
                    st.info("No ACTUALS rows found for this deal.")
                elif "Period End Date" not in act.columns:
                    st.info("ACTUALS missing 'Period End Date'.")
                else:
                    act = act.dropna(subset=["Period End Date"]).sort_values("Period End Date")

                    if act.empty:
                        st.info("No valid dated ACTUALS found for this deal.")
                    else:
                        # Keep last 24 periods for a clean terminal look
                        act = act.tail(24).copy()
                        
                        # --- LABEL MODE LOGIC (Step 2C.1) ---
                        is_lbl_mode = st.session_state.get('label_mode', False)
                        valid_chart_data = True
                        metric_prefix = ""
                        
                        # Calculate NetUsed based on mode
                        if is_lbl_mode:
                            metric_prefix = "LBL "
                            # Get Label Share Pct
                            lbl_pct = np.nan
                            sel_row = eligible[eligible['did_norm'] == selected_id]
                            if not sel_row.empty:
                                lbl_pct = sel_row.iloc[0].get('Label Share Pct', np.nan)
                            
                            if pd.isna(lbl_pct) or lbl_pct <= 0:
                                valid_chart_data = False
                                m1, m2, m3 = st.columns(3)
                                m1.metric(f"{metric_prefix}LAST MONTH", "N/A")
                                m2.metric(f"{metric_prefix}MoM %", "N/A")
                                m3.metric(f"{metric_prefix}RUN-RATE (SMA3)", "N/A")
                                st.info("LABEL MODE: Missing Artist Share → cannot compute Label receipts for this deal.")
                            else:
                                act['NetUsed'] = act['Net Receipts'] * lbl_pct
                        else:
                             # Standard Mode
                             act['NetUsed'] = act['Net Receipts']

                        if valid_chart_data:
                            # "Candles" derived from month-to-month receipts
                            # Close = this month receipts
                            # Open  = prior month receipts
                            # USE NetUsed INSTEAD OF Net Receipts
                            act["Close"] = pd.to_numeric(act["NetUsed"], errors="coerce").fillna(0)
                            act["Open"] = act["Close"].shift(1)
    
                            # --- TRICKLE VISUAL FIX (Approach A) ---
                            # If Month 1 was flagged as trickle, we "ignore" it for comparisons by flattening Month2's candle:
                            # Set Month2 Open = Month2 Close (prevents a giant "up candle" from a tiny/partial Month1)
                            trickle_flag = False
                            trickle_reason = ""
                            # Pull trickle flag from DASHBOARD for the selected deal
                            dash_row = eligible[eligible["did_norm"] == selected_id]
                            if not dash_row.empty:
                                trickle_flag = bool(dash_row.iloc[0].get("TrickleDetected", False))
                                trickle_reason = str(dash_row.iloc[0].get("TrickleReason", "")).strip()
                            
                            if trickle_flag and len(act) >= 2:
                                # act is sorted by date, so second row is month 2
                                idx_m2 = act.index[1]
                                act.loc[idx_m2, "Open"] = act.loc[idx_m2, "Close"]
    
                            # Candle body bounds
                            act["BodyTop"] = act[["Open", "Close"]].max(axis=1)
                            act["BodyBot"] = act[["Open", "Close"]].min(axis=1)
    
                            # Wicks (we don’t have intra-month high/low, so wick = body range)
                            act["High"] = act["BodyTop"]
                            act["Low"] = act["BodyBot"]
    
                            # Moving averages
                            act["SMA3_RAW"] = act["Close"].rolling(3).mean()
                            if trickle_flag and len(act) >= 1:
                                close_adj = act["Close"].copy()
                                close_adj.iloc[0] = np.nan  # exclude Month 1
                                # rolling mean ignores NaNs; min_periods=2 prevents Month2 from just echoing M2
                                act["SMA3"] = close_adj.rolling(3, min_periods=2).mean()
                            else:
                                act["SMA3"] = act["SMA3_RAW"]
                            
                            act["SMA6"] = act["Close"].rolling(6).mean()
    
                            # Headline stats
                            last_close = float(act["Close"].iloc[-1])
                            prev_close = float(act["Close"].iloc[-2]) if len(act) >= 2 else last_close
                            
                            # --- MoM TRICKLE GUARDRAIL (restored) ---
                            TRICKLE_FLOOR = 50.0
                            if is_lbl_mode and pd.notna(lbl_pct):
                                TRICKLE_FLOOR = TRICKLE_FLOOR * lbl_pct # Adjust floor for label share
                                
                            # Treat MoM as TRICKLE when the prior month denominator is tiny
                            # (fixes cases like China where prev month is $0.04)
                            mom_is_trickle = (len(act) >= 2 and prev_close <= TRICKLE_FLOOR)
                            
                            if mom_is_trickle:
                                mom_pct = None
                            else:
                                mom_pct = ((last_close - prev_close) / prev_close) if prev_close else None
                                
                            sma3 = float(act["SMA3"].iloc[-1]) if pd.notna(act["SMA3"].iloc[-1]) else 0.0
    
                            m1, m2, m3 = st.columns(3)
                            m1.metric(f"{metric_prefix}LAST MONTH", f"${last_close:,.0f}")
                            
                            if mom_pct is None:
                                m2.metric(f"{metric_prefix}MoM %", "TRICKLE" if mom_is_trickle else "N/A")
                            else:
                                m2.metric(f"{metric_prefix}MoM %", f"{mom_pct*100:+.1f}%")
                                
                            m3.metric(f"{metric_prefix}RUN-RATE (SMA3)", f"${sma3:,.0f}/mo")
                            
                            if trickle_flag:
                                safe_reason = sanitize_terminal_text(trickle_reason)
                                st.markdown(
                                    f"""
                                    <div style="
                                        margin-top: 6px;
                                        color: #33ff00;
                                        font-family: 'Courier New', monospace;
                                        font-size: 12px;
                                        white-space: nowrap;
                                        word-break: normal;
                                        overflow-x: auto;
                                    ">
                                        TRICKLE DETECTED: {safe_reason if safe_reason else "TRUE"}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
    
                            # CRITICAL FIX 2: Create a clean subset DF for Altair to prevent serialization errors
                            chart_data = act[["Period End Date", "Open", "Close", "Low", "High", "BodyBot", "BodyTop", "SMA3", "SMA3_RAW", "SMA6"]].copy()
                            # Ensure strictly datetime and reset index
                            chart_data['Period End Date'] = pd.to_datetime(chart_data['Period End Date'])
                            chart_data = chart_data.reset_index(drop=True)
    
                            # Make a compact month axis (Bloomberg feel)
                            chart_data["DateStr"] = chart_data["Period End Date"].dt.strftime("%b %y")
                            chart_data["MonthIndex"] = range(1, len(chart_data) + 1)
                            
                            # Remove the first month (which has no prior month to compare against)
                            # This prevents the "giant green candle from zero" artifact
                            chart_data = chart_data.dropna(subset=["Open"])
    
                            # Altair chart: terminal-style candles
                            base = alt.Chart(chart_data).encode(
                                x=alt.X(
                                    "DateStr:O",
                                    title=None,
                                    sort=None,
                                    axis=alt.Axis(labelColor="#33ff00", tickColor="#33ff00", labelAngle=0),
                                    scale=alt.Scale(paddingOuter=0.02, paddingInner=0.2)
                                )
                            )
    
                            # Wick
                            wick = base.mark_rule(opacity=0.9, color="#33ff00").encode(
                                y=alt.Y("Low:Q", title=None,
                                        axis=alt.Axis(labelColor="#33ff00", grid=False, domain=False, ticks=False)),
                                y2="High:Q"
                            )
    
                            # Body (green up, red down)
                            body = base.mark_bar(size=10).encode(
                                y="BodyBot:Q",
                                y2="BodyTop:Q",
                                color=alt.condition(
                                    "datum.Close >= datum.Open",
                                    alt.value("#33ff00"),
                                    alt.value("#ff3333")
                                ),
                                tooltip=[
                                    alt.Tooltip("Period End Date:T", title="Period"),
                                    alt.Tooltip("Open:Q", format=",.0f"),
                                    alt.Tooltip("Close:Q", format=",.0f"),
                                    alt.Tooltip("SMA3:Q", format=",.0f"),
                                    alt.Tooltip("SMA6:Q", format=",.0f"),
                                ]
                            )
    
                            sma3_line = base.mark_line(strokeWidth=1, color="#ffbf00").encode(y="SMA3:Q")
                            sma6_line = base.mark_line(strokeWidth=1, strokeDash=[3, 3], color="#b026ff").encode(y="SMA6:Q")
    
                            chart = (
                                alt.layer(wick, body, sma3_line, sma6_line)
                                .properties(height=360)
                                .configure(background="#050a0e")
                                .configure_view(fill="#050a0e", stroke=None)
                                .configure_axis(grid=False)
                            )
    
                            st.altair_chart(chart, use_container_width=True, theme=None)
            
            except Exception as e:
                st.error(f"Error rendering Market Pulse chart: {str(e)}")

    # ---------------------------------------------------------------------
    # RANGE SCANNER MODULE (NEW)
    # ---------------------------------------------------------------------
    st.markdown("### > LABEL SHARE // RANGE SCANNER")
    
    if df_act.empty or 'Period End Date' not in df_act.columns:
        st.warning("Range Scanner unavailable: No Actuals data.")
    else:
        # 1. Build Date Options
        valid_dates = df_act.dropna(subset=['Period End Date'])['Period End Date'].unique()
        valid_dates = sorted(valid_dates)
        
        if not valid_dates:
            st.warning("Range Scanner unavailable: No valid dates found.")
        else:
            # Create readable labels "Oct 2025" -> Date object map
            date_map = {pd.to_datetime(d).strftime('%b %Y'): pd.to_datetime(d) for d in valid_dates}
            month_labels = list(date_map.keys())
            
            # Default: Last 3 months
            default_end_idx = len(month_labels) - 1
            default_start_idx = max(0, default_end_idx - 2)
            
            rs_c1, rs_c2 = st.columns(2)
            with rs_c1:
                # ADDED KEY TO PREVENT CONFLICT
                start_month_str = st.selectbox("START MONTH", month_labels, index=default_start_idx, key="rs_start")
            with rs_c2:
                # ADDED KEY TO PREVENT CONFLICT
                end_month_str = st.selectbox("END MONTH", month_labels, index=default_end_idx, key="rs_end")
                
            start_date = date_map[start_month_str]
            end_date = date_map[end_month_str]
            
            if start_date > end_date:
                st.error("INVALID RANGE: Start Month must be before or equal to End Month.")
            else:
                # 2. Filters Row
                f_c1, f_c2, f_c3, f_c4 = st.columns(4)
                
                # A) Artists
                # Use Artist / Project if available, else Artist
                if 'Artist / Project' in df_dash.columns:
                    art_col = 'Artist / Project'
                else:
                    art_col = 'Artist'
                
                all_artists = sorted(df_dash[art_col].dropna().unique().tolist())
                with f_c1:
                    # ADDED KEY TO PREVENT CONFLICT
                    sel_artists = st.multiselect("ARTISTS", all_artists, default=[], key="rs_artists")
                    
                # B) Tags (from Tags List)
                all_tags_scan = set()
                if 'Tags List' in df_dash.columns:
                    for t_list in df_dash['Tags List']:
                        all_tags_scan.update(t_list)
                all_tags_scan = sorted(list(all_tags_scan))
                with f_c2:
                    # ADDED KEY TO PREVENT CONFLICT
                    sel_tags = st.multiselect("TAGS", all_tags_scan, default=[], key="rs_tags")
                    
                # C) JV Filter (Is JV Clean)
                with f_c3:
                    # ADDED KEY TO PREVENT CONFLICT
                    jv_mode = st.selectbox("JV FILTER", ["ALL", "JV ONLY", "NON-JV ONLY"], index=0, key="rs_jv")
                    
                # D) View Mode
                with f_c4:
                    # ADDED KEY TO PREVENT CONFLICT
                    view_mode = st.selectbox("VIEW MODE", ["SUMMARY", "DRIVERS", "FULL BREAKDOWN"], index=0, key="rs_view")
                    
                # 3. Define Deal Universe (df_dash filtered)
                # Start with all valid deals
                scanner_universe = df_dash[
                    df_dash['did_norm'].notna() & 
                    (df_dash['did_norm'] != "") & 
                    (df_dash['did_norm'].str.lower() != "nan")
                ].copy()
                
                # Apply Artist Filter
                if sel_artists:
                    scanner_universe = scanner_universe[scanner_universe[art_col].isin(sel_artists)]
                    
                # Apply Tag Filter (ANY match)
                if sel_tags and 'Tags List' in scanner_universe.columns:
                    scanner_universe = scanner_universe[scanner_universe['Tags List'].apply(lambda x: any(t in x for t in sel_tags))]
                    
                # Apply JV Filter
                if jv_mode == "JV ONLY":
                    scanner_universe = scanner_universe[scanner_universe['Is JV Clean'] == True]
                elif jv_mode == "NON-JV ONLY":
                    scanner_universe = scanner_universe[scanner_universe['Is JV Clean'] == False]
                    
                universe_ids = scanner_universe['did_norm'].unique()
                
                if len(universe_ids) == 0:
                    st.warning("No deals match current filters.")
                else:
                    # 4. Compute Range Receipts
                    # Filter df_act for Universe IDs AND Date Range
                    range_act = df_act[
                        (df_act['did_norm'].isin(universe_ids)) &
                        (df_act['Period End Date'] >= start_date) &
                        (df_act['Period End Date'] <= end_date)
                    ].copy()
                    
                    # Group by Deal ID to get Range Receipts
                    if not range_act.empty:
                        range_sums = range_act.groupby('did_norm')['Net Receipts'].sum().reset_index()
                        range_sums.rename(columns={'Net Receipts': 'Range Receipts'}, inplace=True)
                    else:
                        range_sums = pd.DataFrame(columns=['did_norm', 'Range Receipts'])
                        
                    # Merge Range Data into Universe
                    # (Left merge to keep deals with 0 receipts in range if they are in universe)
                    scanner_data = scanner_universe.merge(range_sums, on='did_norm', how='left')
                    scanner_data['Range Receipts'] = scanner_data['Range Receipts'].fillna(0.0)
                    
                    # Compute Range Label Share
                    # Range Label Share = Range Receipts * Label Share Pct
                    # If Label Share Pct is NaN -> Range Label Share is NaN
                    scanner_data['Range Label Share'] = np.where(
                        pd.notna(scanner_data['Label Share Pct']),
                        scanner_data['Range Receipts'] * scanner_data['Label Share Pct'],
                        np.nan
                    )
                    
                    # Ensure All-Time columns exist
                    if 'LBL Cum' not in scanner_data.columns:
                        scanner_data['LBL Cum'] = np.nan
                    if 'Cum Receipts' not in scanner_data.columns:
                        scanner_data['Cum Receipts'] = 0.0
                        
                    # 5. Render Outputs
                    
                    # --- MODE: SUMMARY ---
                    if view_mode == "SUMMARY":
                        k1, k2, k3, k4, k5, k6 = st.columns(6)
                        
                        # Aggregates
                        total_range_receipts = scanner_data['Range Receipts'].sum()
                        total_range_lbl = scanner_data['Range Label Share'].sum() # Sums non-NaNs automatically
                        
                        active_in_range = len(scanner_data[scanner_data['Range Receipts'] > 0])
                        deals_w_share = len(scanner_data[scanner_data['Label Share Pct'].notna()])
                        
                        cum_all_time = scanner_data['Cum Receipts'].sum()
                        lbl_cum_all_time = scanner_data['LBL Cum'].sum()
                        
                        k1.metric("RANGE RECEIPTS", f"${total_range_receipts:,.0f}")
                        k2.metric("RANGE LBL SHARE", f"${total_range_lbl:,.0f}")
                        k3.metric("ACTIVE (RANGE)", active_in_range)
                        k4.metric("DEALS W/ SHARE", deals_w_share)
                        k5.metric("CUM (ALL TIME)", f"${cum_all_time:,.0f}")
                        k6.metric("LBL CUM (ALL TIME)", f"${lbl_cum_all_time:,.0f}")
                        
                    # --- MODE: DRIVERS ---
                    elif view_mode == "DRIVERS":
                        st.markdown("##### > TOP 10 DRIVERS (SORTED BY RANGE LABEL SHARE)")
                        
                        # Sort by Range Label Share Descending (NaNs at end)
                        drivers = scanner_data.sort_values('Range Label Share', ascending=False, na_position='last').head(10)
                        
                        # Custom Table Header
                        st.markdown("""
                        <div style="display: flex; border-bottom: 2px solid #33ff00; padding-bottom: 5px; margin-bottom: 10px; font-weight: bold; color: #ffbf00; font-size: 0.9em;">
                            <div style="flex: 3;">ARTIST / PROJECT</div>
                            <div style="flex: 1.5; text-align: right;">RANGE RECEIPTS</div>
                            <div style="flex: 1.5; text-align: right;">RANGE LBL SHARE</div>
                            <div style="flex: 1; text-align: right;">LBL %</div>
                            <div style="flex: 0.8; text-align: center;">IS JV</div>
                            <div style="flex: 2; padding-left: 15px;">TAGS</div>
                            <div style="flex: 0.8;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for i, row in enumerate(drivers.to_dict('records')):
                            name = row.get(art_col, 'Unknown')
                            rr = row.get('Range Receipts', 0)
                            rls = row.get('Range Label Share', np.nan)
                            lpct = row.get('Label Share Pct', np.nan)
                            is_jv = row.get('Is JV Clean', False)
                            tags_l = row.get('Tags List', [])
                            
                            # Formatting
                            rr_str = f"${rr:,.0f}"
                            if pd.isna(rls):
                                rls_str = "N/A"
                                rls_color = "#888"
                            else:
                                rls_str = f"${rls:,.0f}"
                                rls_color = "#b026ff" # Purple
                                
                            if pd.isna(lpct):
                                lpct_str = "-"
                            else:
                                lpct_str = f"{lpct*100:.1f}%"
                                
                            jv_str = "YES" if is_jv else "NO"
                            
                            # Tags Badge String
                            tags_html = ""
                            for t in tags_l[:3]: # Max 3
                                tags_html += f'<span style="border:1px solid #33ff00; border-radius:3px; padding:1px 4px; font-size:0.7em; margin-right:4px; color:#33ff00;">{t}</span>'
                            if len(tags_l) > 3:
                                tags_html += f'<span style="font-size:0.7em; color:#888;">+{len(tags_l)-3}</span>'
                                
                            # ID for open
                            did_val = row.get('did_norm')
                            
                            dc1, dc2, dc3, dc4, dc5, dc6, dc7 = st.columns([3, 1.5, 1.5, 1, 0.8, 2, 0.8])
                            
                            with dc1: st.markdown(f"<div style='font-weight:bold; color:#e6ffff;'>{name}</div>", unsafe_allow_html=True)
                            with dc2: st.markdown(f"<div style='text-align:right; color:#33ff00;'>{rr_str}</div>", unsafe_allow_html=True)
                            with dc3: st.markdown(f"<div style='text-align:right; color:{rls_color}; font-weight:bold;'>{rls_str}</div>", unsafe_allow_html=True)
                            with dc4: st.markdown(f"<div style='text-align:right;'>{lpct_str}</div>", unsafe_allow_html=True)
                            with dc5: st.markdown(f"<div style='text-align:center;'>{jv_str}</div>", unsafe_allow_html=True)
                            with dc6: st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
                            with dc7:
                                if st.button("OPEN", key=f"drv_open_{i}"):
                                    st.session_state['selected_deal_id'] = did_val
                                    st.rerun()
                            st.markdown("<div style='border-bottom: 1px solid #222; margin-bottom: 3px;'></div>", unsafe_allow_html=True)

                    # --- MODE: FULL BREAKDOWN ---
                    elif view_mode == "FULL BREAKDOWN":
                        st.markdown(f"##### > FULL BREAKDOWN ({len(scanner_data)} DEALS)")
                        
                        # Table Header
                        st.markdown("""
                        <div style="display: flex; border-bottom: 2px solid #33ff00; padding-bottom: 5px; margin-bottom: 10px; font-weight: bold; color: #ffbf00; font-size: 0.85em;">
                            <div style="flex: 2.5;">ARTIST / PROJECT</div>
                            <div style="flex: 1;">DEAL ID</div>
                            <div style="flex: 1.5;">TAGS</div>
                            <div style="flex: 0.5; text-align: center;">JV</div>
                            <div style="flex: 0.8; text-align: right;">ART%</div>
                            <div style="flex: 0.8; text-align: right;">LBL%</div>
                            <div style="flex: 1.2; text-align: right;">RNG RECPT</div>
                            <div style="flex: 1.2; text-align: right;">RNG LBL</div>
                            <div style="flex: 1.2; text-align: right;">CUM (ALL)</div>
                            <div style="flex: 1.2; text-align: right;">LBL (ALL)</div>
                            <div style="flex: 0.7;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Sort by Range Label Share desc by default
                        full_table = scanner_data.sort_values('Range Label Share', ascending=False, na_position='last')
                        
                        for i, row in enumerate(full_table.to_dict('records')):
                            name = row.get(art_col, 'Unknown')
                            did_disp = row.get('Deal ID', 'N/A')
                            
                            tags_l = row.get('Tags List', [])
                            is_jv = row.get('Is JV Clean', False)
                            
                            asp = row.get('Artist Share Pct', 0)
                            lsp = row.get('Label Share Pct', np.nan)
                            
                            rr = row.get('Range Receipts', 0)
                            rls = row.get('Range Label Share', np.nan)
                            
                            cum = row.get('Cum Receipts', 0)
                            lbl_cum = row.get('LBL Cum', np.nan)
                            
                            # Format
                            tags_str = ", ".join(tags_l[:2]) + ("..." if len(tags_l)>2 else "")
                            jv_char = "Y" if is_jv else "N"
                            
                            asp_str = f"{asp*100:.0f}%" if asp > 0 else "-"
                            if pd.isna(lsp): lsp_str = "-"
                            else: lsp_str = f"{lsp*100:.0f}%"
                            
                            rr_str = f"${rr:,.0f}"
                            
                            if pd.isna(rls): rls_str = "-"
                            else: rls_str = f"${rls:,.0f}"
                            
                            cum_str = f"${cum:,.0f}"
                            
                            if pd.isna(lbl_cum): lbl_cum_str = "-"
                            else: lbl_cum_str = f"${lbl_cum:,.0f}"
                            
                            did_val = row.get('did_norm')
                            
                            col_spec = [2.5, 1, 1.5, 0.5, 0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 0.7]
                            cols = st.columns(col_spec)
                            
                            with cols[0]: st.markdown(f"<div style='white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#e6ffff;'>{name}</div>", unsafe_allow_html=True)
                            with cols[1]: st.markdown(f"<div style='font-size:0.8em; color:#888;'>{did_disp}</div>", unsafe_allow_html=True)
                            with cols[2]: st.markdown(f"<div style='font-size:0.8em; color:#888;'>{tags_str}</div>", unsafe_allow_html=True)
                            with cols[3]: st.markdown(f"<div style='text-align:center;'>{jv_char}</div>", unsafe_allow_html=True)
                            with cols[4]: st.markdown(f"<div style='text-align:right; color:#888;'>{asp_str}</div>", unsafe_allow_html=True)
                            with cols[5]: st.markdown(f"<div style='text-align:right;'>{lsp_str}</div>", unsafe_allow_html=True)
                            with cols[6]: st.markdown(f"<div style='text-align:right; color:#33ff00;'>{rr_str}</div>", unsafe_allow_html=True)
                            with cols[7]: st.markdown(f"<div style='text-align:right; color:#b026ff; font-weight:bold;'>{rls_str}</div>", unsafe_allow_html=True)
                            with cols[8]: st.markdown(f"<div style='text-align:right; color:#888;'>{cum_str}</div>", unsafe_allow_html=True)
                            with cols[9]: st.markdown(f"<div style='text-align:right; color:#888;'>{lbl_cum_str}</div>", unsafe_allow_html=True)
                            with cols[10]:
                                if st.button("GO", key=f"full_open_{i}"):
                                    st.session_state['selected_deal_id'] = did_val
                                    st.rerun()
                            st.markdown("<div style='border-bottom: 1px solid #111; margin-bottom: 1px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN APP LOOP
# -----------------------------------------------------------------------------
def main():
    df_dash_raw, df_act_raw, df_deals_raw = load_data()
    if df_dash_raw.empty:
        st.error("DATABASE OFFLINE: CHECK CONNECTIONS OR SHEET HEADERS.")
        st.stop()
    
    # Process returns 3 values
    df_dash, df_act, current_date_override = process_data(df_dash_raw, df_act_raw, df_deals_raw)
    
    if 'selected_deal_id' in st.session_state:
        show_detail(df_dash, df_act, st.session_state['selected_deal_id'])
    else:
        # Pass override date to show_portfolio for debug caption
        show_portfolio(df_dash, df_act, current_date_override)

if __name__ == "__main__":
    main()
