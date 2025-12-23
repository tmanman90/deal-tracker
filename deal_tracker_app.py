import streamlit as st
import pandas as pd
import altair as alt
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
from datetime import datetime, date

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
    /* MAIN TERMINAL BACKGROUND */
    .stApp {
        background-color: #050a0e;
        color: #33ff00;
        font-family: 'Courier New', Courier, monospace;
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

    /* TYPOGRAPHY */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        font-family: 'Courier New', Courier, monospace !important;
        text-shadow: 0 0 2px #33ff00aa;
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
    }
    .ticker {
        display: inline-block;
        white-space: nowrap;
        animation: ticker 30s linear infinite;
    }
    .ticker-item {
        display: inline-block;
        padding: 0 2rem;
        font-size: 1.2rem;
        color: #33ff00;
    }
    @keyframes ticker {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
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

def calculate_pace_metrics(row, count, deal_meta=None):
    """
    Calculates Grade and Pace based on Benchmark.
    Includes 'Ramp-up Curve' for first 4 months.
    Handles 'Legacy' deals (missing Analyzer data) by defaulting to 12 Months.
    
    Returns (Grade, Pace Ratio, Eligible Boolean, Elapsed Months, Expected Progress, Is_Legacy_Flag).
    """
    # 1. Eligibility Check (Requires 3 data points)
    if count < 3:
        return "N/A", 0.0, False, 0, 0.0, False
    
    # 2. Determine Timeline (Target Months)
    # Check if we have analyzer data. 
    target_months = 12.0 # Default
    is_legacy = False
    
    strat = str(row.get('Selected Strategy', '')).strip()
    sel_adv_raw = row.get('Selected Advance', '')
    sel_adv = clean_currency(sel_adv_raw)
    
    # Check if deal is analyzed (has strategy and selected advance)
    if strat and sel_adv > 0:
        # ANALYZER MODE
        is_legacy = False
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
        
    # 4. Calculate Elapsed Months vs Benchmark
    today = pd.Timestamp.now()
    if forecast_start > today:
        elapsed_months = 0 
    else:
        elapsed_months = (today - forecast_start).days / 30.4375
    
    elapsed_months = max(0.1, elapsed_months)
    
    # --- RAMP-UP CURVE LOGIC (TIGHTENED v2) ---
    linear_progress = elapsed_months / target_months
    
    if elapsed_months <= 1.5:
        curve_factor = 0.6 # M1: 1/20 vs 1/12
    elif elapsed_months <= 2.5:
        curve_factor = 0.66 # M2: 2/18 vs 2/12
    elif elapsed_months <= 3.5:
        curve_factor = 0.85 # M3: 3/14 vs 3/12
    else:
        curve_factor = 1.0 # M4+: Full speed
        
    expected_progress = linear_progress * curve_factor
    expected_progress = min(1.0, expected_progress)
    
    # 5. Actual progress
    # UPDATED: Always use Executed Advance as target amount for actual progress
    target_amount = clean_currency(row.get('Executed Advance', 0))
    cum_receipts = clean_currency(row.get('Cum Receipts', 0))
    
    if target_amount > 0:
        actual_progress = cum_receipts / target_amount
    else:
        actual_progress = 0.0
    
    # Pace Ratio
    if expected_progress == 0:
        pace_ratio = 0
    else:
        pace_ratio = actual_progress / expected_progress
        
    # --- GRADING BANDS ---
    if pace_ratio >= 1.10: grade = "A+"
    elif pace_ratio >= 1.00: grade = "A"
    elif pace_ratio >= 0.90: grade = "B+"
    elif pace_ratio >= 0.80: grade = "B"
    elif pace_ratio >= 0.70: grade = "C+"
    elif pace_ratio >= 0.60: grade = "C"
    elif pace_ratio >= 0.50: grade = "D"
    else: grade = "F"
        
    return grade, pace_ratio, True, elapsed_months, expected_progress, is_legacy

# -----------------------------------------------------------------------------
# DATA PROCESSING WRAPPER
# -----------------------------------------------------------------------------
def process_data(df_dash, df_act, df_deals):
    # Ensure required columns exist
    if df_dash.empty:
        return df_dash, df_act

    # MERGE DEALS DATA INTO DASHBOARD IF AVAILABLE
    if not df_deals.empty:
        # Normalize IDs for merge
        if 'Deal ID' in df_dash.columns:
            df_dash['Deal ID Str'] = df_dash['Deal ID'].astype(str).str.strip()
        
        if 'Deal ID' in df_deals.columns:
            df_deals['Deal ID Str'] = df_deals['Deal ID'].astype(str).str.strip()
            
            # Columns to bring over, including 'Tags'
            cols_to_merge = ['Selected Strategy', 'Selected Advance', 'Label Breakeven Months', 'Tags']
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

    # 1. GLOBAL STRIP: Clean Deal IDs
    if 'Deal ID' in df_act.columns:
        df_act['Deal ID'] = df_act['Deal ID'].astype(str).str.strip()
    if 'Deal ID' in df_dash.columns:
        df_dash['Deal ID'] = df_dash['Deal ID'].astype(str).str.strip()

    # Process Actuals Dates & Values
    if 'Net Receipts' in df_act.columns:
        df_act['Net Receipts'] = df_act['Net Receipts'].apply(clean_currency)
    
    # ROBUST DATE CLEANING FOR ACTUALS
    if 'Period End Date' in df_act.columns:
        df_act['Period End Date'] = df_act['Period End Date'].apply(parse_flexible_date)
    
    # Clean Dashboard Numerics
    numeric_cols = ['Executed Advance', 'Cum Receipts', 'Remaining to BE']
    for col in numeric_cols:
        if col in df_dash.columns:
            df_dash[col] = df_dash[col].apply(clean_currency)
        else:
            df_dash[col] = 0.0
    
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
    
    for _, row in df_dash.iterrows():
        did = str(row.get('Deal ID', ''))
        
        # Get count (default 0)
        count = eligibility_map.get(did, 0)
        
        # Now returns 6 values including is_legacy
        g, r, e, el_m, exp_prog, is_leg = calculate_pace_metrics(row, count)
        
        grades.append(g)
        ratios.append(r)
        is_eligible.append(e)
        elapsed_list.append(el_m)
        data_points_list.append(count)
        expected_list.append(exp_prog)
        legacy_list.append(is_leg)
        
    df_dash['Grade'] = grades
    df_dash['Pace Ratio'] = ratios
    df_dash['Is Eligible'] = is_eligible
    df_dash['Elapsed Months'] = elapsed_list
    df_dash['Data Points Found'] = data_points_list 
    df_dash['Expected Recoupment'] = expected_list
    df_dash['Is Legacy'] = legacy_list
    
    # UPDATED: Always set Target Amount to Executed Advance for dashboard display
    df_dash['Target Amount'] = df_dash['Executed Advance']
    df_dash['% to BE Clean'] = df_dash.apply(lambda r: (r['Cum Receipts']/r['Target Amount']) if r['Target Amount'] > 0 else 0, axis=1)
    
    return df_dash, df_act

# -----------------------------------------------------------------------------
# UI: PORTFOLIO PAGE
# -----------------------------------------------------------------------------
def show_portfolio(df_dash, df_act):
    st.title(">>> GLOBAL DEAL TRACKER_")
    
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
    
    ticker_html = f"""
    <div class="ticker-wrap">
        <div class="ticker">
            <span class="ticker-item">{' &nbsp;&nbsp;&nbsp; /// &nbsp;&nbsp;&nbsp; '.join(ticker_items)}</span>
        </div>
    </div>
    """
    st.markdown(ticker_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- FILTERS ---
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        search = st.text_input("SEARCH ARTIST OR DEAL ID", "").lower()
        
    with col2:
        all_status = df_dash['Status'].unique().tolist() if 'Status' in df_dash.columns else []
        status_filter = st.multiselect("STATUS", all_status, default=all_status)
        
    with col3:
        eligible_only = st.checkbox("ELIGIBLE FOR GRADE", value=False)
        
    with col4:
        sort_opt = st.selectbox("SORT BY", ["Grade", "Remaining to BE", "% to BE", "Cum Receipts", "Delta Months"], index=0)
    
    # Filter Logic
    filtered = df_dash.copy()
    
    if search:
        mask = pd.Series([False] * len(filtered))
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
        
    # Sort Logic
    ascending = False
    sort_col = None
    
    if sort_opt == "Remaining to BE":
        sort_col = "Remaining to BE"
    elif sort_opt == "% to BE":
        sort_col = "% to BE Clean"
    elif sort_opt == "Grade":
        sort_col = "Grade"
        ascending = True # A is 'smaller' than B in ascii
    elif sort_opt == "Cum Receipts":
        sort_col = "Cum Receipts"
    elif sort_opt == "Delta Months":
         sort_col = "Delta Months"

    if sort_col and sort_col in filtered.columns:
        if sort_col == "Delta Months":
             filtered['Delta Months'] = pd.to_numeric(filtered['Delta Months'], errors='coerce').fillna(0)
        filtered = filtered.sort_values(by=sort_col, ascending=ascending)

    # --- KPI CARDS ---
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    kpi1.metric("ACTIVE DEALS", len(filtered))
    
    total_adv = filtered['Executed Advance'].sum() if 'Executed Advance' in filtered.columns else 0
    kpi2.metric("TOTAL ADVANCES", f"${total_adv:,.0f}")
    
    total_rec = filtered['Cum Receipts'].sum() if 'Cum Receipts' in filtered.columns else 0
    kpi3.metric("TOTAL CUM RECEIPTS", f"${total_rec:,.0f}")
    
    # Weighted % to BE
    if total_adv > 0:
        w_pct = (total_rec / total_adv) * 100
        kpi4.metric("WEIGHTED RECOUPMENT", f"{w_pct:.1f}%")
    else:
        kpi4.metric("WEIGHTED RECOUPMENT", "0.0%")
        
    # Weighted Grade Calculation
    if total_adv > 0 and 'Pace Ratio' in filtered.columns and 'Is Eligible' in filtered.columns:
        eligible_deals = filtered[filtered['Is Eligible'] == True]
        
        if not eligible_deals.empty:
            eligible_deals['Weighted Score'] = eligible_deals['Pace Ratio'] * eligible_deals['Executed Advance']
            
            total_eligible_adv = eligible_deals['Executed Advance'].sum()
            total_score = eligible_deals['Weighted Score'].sum()
            
            if total_eligible_adv > 0:
                overall_ratio = total_score / total_eligible_adv
                
                if overall_ratio >= 1.10: w_grade = "A+"
                elif overall_ratio >= 1.00: w_grade = "A"
                elif overall_ratio >= 0.90: w_grade = "B+"
                elif overall_ratio >= 0.80: w_grade = "B"
                elif overall_ratio >= 0.70: w_grade = "C+"
                elif overall_ratio >= 0.60: w_grade = "C"
                elif overall_ratio >= 0.50: w_grade = "D"
                else: w_grade = "F"
                
                kpi5.metric("WEIGHTED GRADE", w_grade)
            else:
                kpi5.metric("WEIGHTED GRADE", "N/A")
        else:
            kpi5.metric("WEIGHTED GRADE", "N/A")
    else:
        kpi5.metric("WEIGHTED GRADE", "N/A")
    
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
        <div style="flex: 1; text-align: right;">RECOUPED</div>
        <div style="flex: 1.5; text-align: right;">REMAINING</div>
        <div style="flex: 1;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    for idx, row in filtered.iterrows():
        # Clean Data for Display
        artist = row.get('Artist / Project', row.get('Artist', 'Unknown'))
        
        # Check for Tags and append badge if 'AI'
        tags = str(row.get('Tags', '')).upper()
        if 'AI' in tags:
            artist += ' <span style="font-size: 0.7em; border: 1px solid #33ff00; padding: 2px 6px; margin-left: 8px; border-radius: 4px; color: #33ff00;">AI</span>'
            
        did = row.get('Deal ID', 'N/A')
        status = row.get('Status', '-')
        
        grade = row.get('Grade', 'WAITING') if row.get('Is Eligible', False) else "PENDING"
        grade_color = "#33ff00" if "A" in grade or "B+" in grade else "#ffbf00" if "B" in grade or "C+" in grade else "#ff3333" if "C" in grade or "D" in grade or "F" in grade else "#888"
        
        pct_val = row.get('% to BE Clean', 0)
        pct_str = f"{pct_val*100:.1f}%"
        
        rem_val = row.get('Remaining to BE', 0)
        rem_str = f"${rem_val:,.0f}" if isinstance(rem_val, (int, float)) else str(rem_val)
        
        c1, c2, c3, c4, c5, c6, c7 = st.columns([3, 1, 1, 1, 1, 1.5, 1])
        
        with c1:
            st.markdown(f"<div style='padding-top: 5px; font-weight: bold; color: #e6ffff;'>{artist}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='padding-top: 5px; color: #888;'>{did}</div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div style='padding-top: 5px;'>{status}</div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div style='padding-top: 5px; color: {grade_color}; font-weight: bold;'>{grade}</div>", unsafe_allow_html=True)
        with c5:
            st.markdown(f"<div style='padding-top: 5px; text-align: right; color: #33ff00;'>{pct_str}</div>", unsafe_allow_html=True)
        with c6:
            st.markdown(f"<div style='padding-top: 5px; text-align: right; color: #ffbf00;'>{rem_str}</div>", unsafe_allow_html=True)
        with c7:
            if st.button("OPEN", key=f"btn_{did}"):
                st.session_state['selected_deal_id'] = did
                st.rerun()
                
        st.markdown("<div style='border-bottom: 1px solid #333; margin-bottom: 5px;'></div>", unsafe_allow_html=True)

    st.markdown("---")

    # --- MARKET PULSE (Moved to Bottom) ---
    st.markdown("### > MARKET PULSE")
    
    if not df_act.empty and not df_dash.empty:
        pulse_data = []
        active_ids = df_dash['Deal ID'].unique()
        
        for did in active_ids:
            deal_subset = df_act[df_act['Deal ID'] == did].copy()
            if not deal_subset.empty:
                if 'Period End Date' in deal_subset.columns:
                    deal_subset = deal_subset.dropna(subset=['Period End Date']).sort_values('Period End Date')
                    
                    # Need Target Amount to calculate % Recouped
                    adv_row = df_dash[df_dash['Deal ID'] == did]
                    if not adv_row.empty:
                        target_amt = adv_row.iloc[0].get('Target Amount', 0)
                        if target_amt > 0:
                            deal_subset['CumNet'] = deal_subset['Net Receipts'].cumsum()
                            deal_subset['PctRecouped'] = deal_subset['CumNet'] / target_amt
                            
                            for i, (idx, row) in enumerate(deal_subset.iterrows()):
                                artist_label = adv_row.iloc[0].get('Artist / Project', 
                                              adv_row.iloc[0].get('Artist', 
                                              adv_row.iloc[0].get('Project', did)))
                                
                                pulse_data.append({
                                    'Deal ID': did,
                                    'MonthIndex': i + 1, 
                                    'PctRecouped': row['PctRecouped'],
                                    'Artist': artist_label
                                })

        if pulse_data:
            pulse_df = pd.DataFrame(pulse_data)
            
            neon_range = ['#39FF14', '#00FFFF', '#FF00FF', '#FFFFFF', '#FFFF00']
            
            lines = alt.Chart(pulse_df).mark_line(
                interpolate='linear', 
                strokeWidth=2
            ).encode(
                x=alt.X('MonthIndex', title='Months Since Launch', axis=alt.Axis(
                    domain=False, tickSize=0, grid=True, gridColor='#333333', gridDash=[4, 4],
                    labelColor='#33ff00', titleColor='#ffbf00'
                )),
                y=alt.Y('PctRecouped', title='Recoupment %', axis=alt.Axis(
                    format='%', domain=False, tickSize=0, grid=True, gridColor='#333333', gridDash=[4, 4],
                    labelColor='#33ff00', titleColor='#ffbf00'
                )),
                color=alt.Color('Artist', scale=alt.Scale(range=neon_range), legend=None),
                tooltip=['Artist', 'MonthIndex', alt.Tooltip('PctRecouped', format='.1%')]
            )
            
            area = alt.Chart(pulse_df).mark_area(
                interpolate='linear',
                opacity=0.1,
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='#33ff00', offset=0),
                           alt.GradientStop(color='rgba(0, 0, 0, 0)', offset=1)],
                    x1=1, x2=1, y1=1, y2=0
                )
            ).encode(
                x='MonthIndex',
                y='PctRecouped',
                color=alt.Color('Artist', scale=alt.Scale(range=neon_range), legend=None)
            )
            
            pulse_chart = (area + lines).properties(
                height=300,
                width='container',
                background='transparent'
            ).configure_view(
                strokeWidth=0,
                fill=None
            )
            
            st.altair_chart(pulse_chart, use_container_width=True, theme=None)
        else:
            st.info("No transaction data available for Pulse Chart.")

# -----------------------------------------------------------------------------
# UI: DEAL DETAIL PAGE
# -----------------------------------------------------------------------------
def show_detail(df_dash, df_act, deal_id):
    if 'Deal ID' not in df_dash.columns:
        st.error("CONFIGURATION ERROR: 'Deal ID' column missing from sheet.")
        return
        
    deal_id = str(deal_id)
    deal_subset = df_dash[df_dash['Deal ID'] == deal_id]
    
    if deal_subset.empty:
        st.error(f"ERROR: Deal ID {deal_id} not found in DASHBOARD.")
        if st.button("RESET"):
            del st.session_state['selected_deal_id']
            st.rerun()
        return

    deal_row = deal_subset.iloc[0]
    deal_act = pd.DataFrame()
    if not df_act.empty and 'Deal ID' in df_act.columns:
        deal_act = df_act[df_act['Deal ID'] == deal_id].copy()
    
    if st.button("<< RETURN TO DASHBOARD"):
        del st.session_state['selected_deal_id']
        st.rerun()
        
    artist_name = deal_row.get('Artist / Project', 
                  deal_row.get('Artist', 
                  deal_row.get('Project', 'UNKNOWN ARTIST')))
    
    st.title(f"// ANALYZING: {artist_name} [{deal_id}]")
    
    # --- HEADER STATS ---
    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
    grade_display = deal_row['Grade'] if deal_row['Is Eligible'] else "PENDING"
    status_val = deal_row.get('Status', '-')
    
    # Ensure values are clean
    adv_val = clean_currency(deal_row.get('Executed Advance', 0))
    cum_val = clean_currency(deal_row.get('Cum Receipts', 0))
    rem_val = clean_currency(deal_row.get('Remaining to BE', 0))
    
    # NOTE: Header stats use '% to BE Clean' which now reflects Legacy logic automatically
    pct_val = deal_row.get('% to BE Clean', 0) * 100
    
    start_date = parse_flexible_date(deal_row.get('Forecast Start Date'))
    start_date_str = start_date.strftime('%b %d, %Y').upper() if pd.notna(start_date) else '-'
    be_date = parse_flexible_date(deal_row.get('Predicted BE Date'))
    be_date_str = be_date.strftime('%b %Y').upper() if pd.notna(be_date) else '-'

    row1_c1.metric("STATUS", status_val)
    row1_c2.metric("PERFORMANCE GRADE", grade_display, delta_color="normal")
    row1_c3.metric("EXECUTED ADVANCE", f"${adv_val:,.0f}")
    row1_c4.metric("% RECOUPED", f"{pct_val:.1f}%")

    row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
    row2_c1.metric("CUM RECEIPTS", f"${cum_val:,.0f}")
    row2_c2.metric("REMAINING", f"${rem_val:,.0f}")
    row2_c3.metric("FORECAST START", start_date_str)
    row2_c4.metric("EST BREAKEVEN", be_date_str)

    st.markdown("---")

    # --- PACE BLOCK ---
    st.markdown("### > PACE ANALYSIS")
    
    col_gauge, col_stats = st.columns([1, 2])
    
    with col_gauge:
        actual_recoup = deal_row.get('% to BE Clean', 0)
        expected_recoup = deal_row.get('Expected Recoupment', 0)
        pace_ratio = deal_row.get('Pace Ratio', 0)
        
        chart_val = min(1.0, max(0.0, actual_recoup))
        
        if pace_ratio < 0.70:
            bar_color = 'red' 
        elif pace_ratio < 0.90:
            bar_color = 'orange' 
        else:
            bar_color = '#33ff00' 
            
        gauge_df = pd.DataFrame({
            'val': [chart_val], 
            'label': ['Recoupment'], 
            'color': [bar_color],
            'Recoupment': [f"{actual_recoup*100:.1f}%"],
            'Forecast': [f"{expected_recoup*100:.1f}%"]
        })
        
        gauge = alt.Chart(gauge_df).mark_bar(size=40).encode(
            x=alt.X('val', scale=alt.Scale(domain=[0, 1.0]), title="Recoupment Progress (0% - 100%)", axis=alt.Axis(format='%')),
            color=alt.Color('color', scale=None, legend=None),
            tooltip=['label', 'Recoupment', 'Forecast']
        ).properties(height=80, title="RECOUPMENT METER")
        
        rule = alt.Chart(pd.DataFrame({'x': [expected_recoup]})).mark_rule(color='white', strokeDash=[4, 4], size=3).encode(x='x')
        st.altair_chart(gauge + rule, use_container_width=True)
        
    with col_stats:
        if deal_row.get('Is Eligible', False):
            elapsed = deal_row.get('Elapsed Months', 0)
            recoup_pct = deal_row.get('% to BE Clean', 0) * 100
            expected_recoup_pct = deal_row.get('Expected Recoupment', 0) * 100 
            is_legacy = deal_row.get('Is Legacy', False)
            tag_val = str(deal_row.get('Tags', '')).upper()
            
            if elapsed <= 4.5:
                 note = "(Curved for ramp-up)"
            else:
                 note = "(Linear)"
            
            legacy_flag = "<br><span style='color: #888; font-size: 0.9rem;'>*Non-Deal Analyzer Forecasting*</span>" if is_legacy else ""
            
            # Artist Type Line
            artist_type_line = ""
            if tag_val:
                artist_type_line = f"<br><span class='diagnostic-label'>ARTIST TYPE:</span> <span class='diagnostic-value' style='color: #33ff00;'>{tag_val}</span>"
            
            st.markdown(f"""
            <div class="diagnostic-box">
                <span class="diagnostic-label">DEAL AGE:</span> <span class="diagnostic-value">{elapsed:.1f} MONTHS</span><br>
                <span class="diagnostic-label">FORECASTED RECOUPMENT:</span> <span class="diagnostic-value">{expected_recoup_pct:.1f}%</span><br>
                <span class="diagnostic-label">ACTUAL RECOUPMENT:</span> <span class="diagnostic-value">{recoup_pct:.1f}%</span><br>
                <span class="diagnostic-label">PACE RATIO:</span> <span class="diagnostic-value">{pace_ratio:.2f}x</span>
                {artist_type_line}
                {legacy_flag}
            </div>
            """, unsafe_allow_html=True)
        else:
            count_found = deal_row.get('Data Points Found', 0)
            st.warning(f"INSUFFICIENT DATA: FOUND {count_found} ACTUALS (NEED 3).")

    # --- CHARTS ---
    st.markdown("### > PERFORMANCE VISUALIZATION")
    
    if not deal_act.empty:
        if 'Period End Date' in deal_act.columns:
            deal_act = deal_act.dropna(subset=['Period End Date']).copy()
            deal_act = deal_act.sort_values('Period End Date')
        
        if not deal_act.empty:
            deal_act['MonthIndex'] = range(1, len(deal_act) + 1)
            deal_act['MonthLabel'] = deal_act['MonthIndex'].apply(lambda x: f"M{x}")
            deal_act['DateStr'] = deal_act['Period End Date'].dt.strftime('%b %Y')
            
            if 'Net Receipts' in deal_act.columns:
                deal_act['CumNet'] = deal_act['Net Receipts'].cumsum()
                deal_act['Rolling3'] = deal_act['Net Receipts'].rolling(window=3).mean()
            else:
                deal_act['CumNet'] = 0
                deal_act['Rolling3'] = 0
                deal_act['Net Receipts'] = 0
            
            # Forecast Data (Linear)
            # Use 'Target Amount' from deal_row to draw forecast line correctly
            target_amt = deal_row.get('Target Amount', adv_val) # adv_val is executed advance
            
            max_month = deal_act['MonthIndex'].max()
            forecast_data = []
            monthly_forecast = target_amt / 12.0 if target_amt > 0 else 0
            for i in range(1, max_month + 1):
                forecast_data.append({
                    'MonthIndex': i,
                    'ForecastCum': monthly_forecast * i,
                    'Type': 'Forecast'
                })
            
            df_forecast = pd.DataFrame(forecast_data)
            
            c1, c2 = st.columns(2)
            
            with c1:
                bar = alt.Chart(deal_act).mark_bar(color='#b026ff').encode(
                    x=alt.X('MonthLabel', sort=None, title='Period'),
                    y=alt.Y('Net Receipts', title='Net Receipts'),
                    tooltip=['MonthLabel', 'DateStr', 'Net Receipts']
                ).properties(title="MONTHLY ACTUALS")
                st.altair_chart(bar, use_container_width=True)
                
            with c2:
                # Actual Line
                line_actual = alt.Chart(deal_act).mark_line(color='#33ff00', point=True).encode(
                    x=alt.X('MonthLabel', sort=None, title='Period'),
                    y=alt.Y('CumNet', title='Cumulative Net'),
                    tooltip=['MonthLabel', 'DateStr', 'CumNet']
                )
                
                # Forecast Line
                if not df_forecast.empty:
                    df_forecast['MonthLabel'] = df_forecast['MonthIndex'].apply(lambda x: f"M{x}")
                    line_forecast = alt.Chart(df_forecast).mark_line(
                        color='#ffbf00', 
                        strokeDash=[5, 5]
                    ).encode(
                        x=alt.X('MonthLabel', sort=None),
                        y=alt.Y('ForecastCum'),
                        tooltip=[alt.Tooltip('ForecastCum', title='Forecast Cumulative')]
                    )
                    final_chart = (line_actual + line_forecast).resolve_scale(y='shared')
                else:
                    final_chart = line_actual

                # Add Advance Line
                rule_adv = alt.Chart(pd.DataFrame({'y': [adv_val]})).mark_rule(color='white', strokeDash=[2, 2]).encode(y='y')
                
                st.altair_chart(final_chart + rule_adv, use_container_width=True)
                
                st.markdown("""
                <div style="text-align: center; font-size: 0.8rem;">
                    <span style="color: #33ff00;">● Actual</span> &nbsp;&nbsp; 
                    <span style="color: #ffbf00;">--- Forecast (12mo Pace)</span>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("### > TERMINAL FORECAST")
            last_rolling = deal_act['Rolling3'].iloc[-1] if len(deal_act) > 0 else 0
            remaining = rem_val
            
            if remaining <= 0:
                st.success("STATUS: RECOUPED. TARGET ACHIEVED.")
            elif last_rolling > 0:
                months_to_go = remaining / last_rolling
                st.markdown(f"""
                BASED ON LAST 3 MONTHS AVG (${last_rolling:,.0f}/mo):
                ESTIMATED TIME TO RECOUP: **{months_to_go:.1f} MONTHS**
                """)
            else:
                st.error("VELOCITY ERROR: RECIEPTS TOO LOW TO PROJECT RECOUPMENT.")
        else:
             st.warning("DATA ERROR: ACTUALS FOUND BUT DATES ARE INVALID/MISSING.")
    else:
        st.warning("NO ACTUALS DATA FOUND ON SERVER.")

# -----------------------------------------------------------------------------
# MAIN APP LOOP
# -----------------------------------------------------------------------------
def main():
    df_dash_raw, df_act_raw, df_deals_raw = load_data()
    if df_dash_raw.empty:
        st.error("DATABASE OFFLINE: CHECK CONNECTIONS OR SHEET HEADERS.")
        st.stop()
    df_dash, df_act = process_data(df_dash_raw, df_act_raw, df_deals_raw)
    
    if 'selected_deal_id' in st.session_state:
        show_detail(df_dash, df_act, st.session_state['selected_deal_id'])
    else:
        show_portfolio(df_dash, df_act)

if __name__ == "__main__":
    main()
