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
        
        return df_dash, df_act
        
    except Exception as e:
        st.error(f"SYSTEM FAILURE: Connection Refused. {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

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

def calculate_pace_metrics(row, count):
    """
    Calculates Grade and Pace based on Benchmark.
    Includes 'Ramp-up Curve' for first 4 months.
    Returns (Grade, Pace Ratio, Eligible Boolean, Elapsed Months).
    """
    # 1. Eligibility Check (Requires 3 data points)
    if count < 3:
        return "N/A", 0.0, False, 0
    
    # 2. Parse Dates
    try:
        if 'Forecast Start Date' not in row:
             return "N/A", 0.0, False, 0
             
        forecast_start = parse_flexible_date(row['Forecast Start Date'])
        if pd.isna(forecast_start):
            return "N/A", 0.0, False, 0
    except:
        return "N/A", 0.0, False, 0
        
    # 3. Calculate Elapsed Months vs Benchmark
    today = pd.Timestamp.now()
    if forecast_start > today:
        elapsed_months = 0 
    else:
        elapsed_months = (today - forecast_start).days / 30.4375
    
    elapsed_months = max(0.1, elapsed_months)
    
    # --- RAMP-UP CURVE LOGIC ---
    # Accounts for slow "trickle-in" payments from distributors in early months.
    # Instead of linear (x/12), we use a softer denominator.
    
    if elapsed_months <= 1.5:
        # Month 1: Expect 1/24th pace
        expected_progress = elapsed_months / 24.0
    elif elapsed_months <= 2.5:
        # Month 2: Expect 2/20th pace
        expected_progress = elapsed_months / 20.0
    elif elapsed_months <= 3.5:
        # Month 3: Expect 3/18th pace
        expected_progress = elapsed_months / 18.0
    elif elapsed_months <= 4.5:
        # Month 4: Expect 4/16th pace
        expected_progress = elapsed_months / 16.0
    else:
        # Month 5+: Standard linear expectation (x/12)
        expected_progress = elapsed_months / 12.0

    # Cap expectation at 1.0 (100%)
    expected_progress = min(1.0, expected_progress)
    
    # Actual progress
    if '% to BE' in row:
        actual_progress = clean_percent(row['% to BE'])
    else:
        actual_progress = 0.0
    
    # Pace Ratio
    if expected_progress == 0:
        pace_ratio = 0
    else:
        pace_ratio = actual_progress / expected_progress
        
    # --- UPDATED GRADING BANDS (WITH PLUSES) ---
    # Target (Pace Ratio = 1.0) is the goal.
    # A+: >= 1.10 (Beat target by 10%+)
    # A:  >= 1.00 (At target)
    # B+: >= 0.90 (Within 10% of target)
    # B:  >= 0.80
    # C+: >= 0.70
    # C:  >= 0.60
    # D:  >= 0.50
    # F:  < 0.50
    
    if pace_ratio >= 1.10:
        grade = "A+"
    elif pace_ratio >= 1.00:
        grade = "A"
    elif pace_ratio >= 0.90:
        grade = "B+"
    elif pace_ratio >= 0.80:
        grade = "B"
    elif pace_ratio >= 0.70:
        grade = "C+"
    elif pace_ratio >= 0.60:
        grade = "C"
    elif pace_ratio >= 0.50:
        grade = "D"
    else:
        grade = "F"
        
    return grade, pace_ratio, True, elapsed_months

# -----------------------------------------------------------------------------
# DATA PROCESSING WRAPPER
# -----------------------------------------------------------------------------
def process_data(df_dash, df_act):
    # Ensure required columns exist
    if df_dash.empty or df_act.empty:
        return df_dash, df_act

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
    
    for _, row in df_dash.iterrows():
        did = str(row.get('Deal ID', ''))
        
        # Get count (default 0)
        count = eligibility_map.get(did, 0)
        
        g, r, e, el_m = calculate_pace_metrics(row, count)
        grades.append(g)
        ratios.append(r)
        is_eligible.append(e)
        elapsed_list.append(el_m)
        data_points_list.append(count)
        
    df_dash['Grade'] = grades
    df_dash['Pace Ratio'] = ratios
    df_dash['Is Eligible'] = is_eligible
    df_dash['Elapsed Months'] = elapsed_list
    df_dash['Data Points Found'] = data_points_list 
    
    # Clean % to BE for display/sorting
    if '% to BE' in df_dash.columns:
        df_dash['% to BE Clean'] = df_dash['% to BE'].apply(clean_percent)
    else:
        df_dash['% to BE Clean'] = 0.0
    
    return df_dash, df_act

# -----------------------------------------------------------------------------
# UI: PORTFOLIO PAGE
# -----------------------------------------------------------------------------
def show_portfolio(df_dash):
    st.title(">>> GLOBAL DEAL TRACKER_")
    st.markdown("---")
    
    # --- FILTERS ---
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        search = st.text_input("SEARCH ARTIST OR DEAL ID", "").lower()
        
    with col2:
        # Get unique statuses
        all_status = df_dash['Status'].unique().tolist() if 'Status' in df_dash.columns else []
        status_filter = st.multiselect("STATUS", all_status, default=all_status)
        
    with col3:
        eligible_only = st.checkbox("ELIGIBLE FOR GRADE", value=False)
        
    with col4:
        sort_opt = st.selectbox("SORT BY", ["Remaining to BE", "% to BE", "Grade", "Cum Receipts", "Delta Months"])
    
    # Filter Logic
    filtered = df_dash.copy()
    
    if search:
        mask = pd.Series([False] * len(filtered))
        if 'Artist / Project' in filtered.columns:
            mask = mask | filtered['Artist / Project'].astype(str).str.lower().str.contains(search)
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
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
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
    
    st.markdown("---")
    
    # --- ROSTER TABLE ---
    # Prepare display dataframe
    display_cols = [
        'Artist / Project', 'Deal ID', 'Status', 'Grade', '% to BE', 
        'Remaining to BE', 'Executed Advance', 'Predicted BE Date'
    ]
    existing_cols = [c for c in display_cols if c in filtered.columns]
    
    display_df = filtered[existing_cols].copy()
    
    # --- DISPLAY FORMATTING ---
    if '% to BE' in display_df.columns and '% to BE Clean' in filtered.columns:
        display_df['% to BE'] = filtered['% to BE Clean'].apply(lambda x: f"{x*100:.1f}%")
        
    for col in ['Remaining to BE', 'Executed Advance']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")

    if 'Predicted BE Date' in display_df.columns:
        display_df['Predicted BE Date'] = display_df['Predicted BE Date'].apply(parse_flexible_date)
        display_df['Predicted BE Date'] = display_df['Predicted BE Date'].dt.strftime('%b %Y').fillna('-').str.upper()

    if 'Grade' in display_df.columns:
        mask_ineligible = filtered['Is Eligible'] == False
        display_df.loc[mask_ineligible, 'Grade'] = "WAITING"
        
    st.markdown("### > SELECT DEAL TO INITIALIZE ANALYSIS")
    
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        height=500
    )
    
    if len(event.selection.rows) > 0:
        row_idx = event.selection.rows[0]
        if 'Deal ID' in filtered.columns:
            selected_deal_id = filtered.iloc[row_idx]['Deal ID']
            st.session_state['selected_deal_id'] = selected_deal_id
            st.rerun()

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
    adv_val = deal_row.get('Executed Advance', 0)
    pct_val = deal_row.get('% to BE Clean', 0) * 100
    cum_val = deal_row.get('Cum Receipts', 0)
    rem_val = deal_row.get('Remaining to BE', 0)
    
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
        pace_ratio = deal_row.get('Pace Ratio', 0)
        chart_val = min(2.0, max(0.0, pace_ratio))
        
        # Color logic based on new bands
        if chart_val < 0.70:
            bar_color = 'red' # F, D, C
        elif chart_val < 0.90:
            bar_color = 'orange' # C+, B
        else:
            bar_color = '#33ff00' # Neon Green (B+, A, A+)
            
        gauge_df = pd.DataFrame({'val': [chart_val], 'label': ['Pace'], 'color': [bar_color]})
        gauge = alt.Chart(gauge_df).mark_bar(size=40).encode(
            x=alt.X('val', scale=alt.Scale(domain=[0, 2]), title="Pace Ratio (1.0 = On Track)"),
            color=alt.Color('color', scale=None, legend=None)
        ).properties(height=80, title="PACE METER")
        
        rule = alt.Chart(pd.DataFrame({'x': [1.0]})).mark_rule(color='white', strokeDash=[5, 5]).encode(x='x')
        st.altair_chart(gauge + rule, use_container_width=True)
        
    with col_stats:
        if deal_row.get('Is Eligible', False):
            elapsed = deal_row.get('Elapsed Months', 0)
            recoup_pct = deal_row.get('% to BE Clean', 0) * 100
            
            if elapsed <= 4.5:
                 note = "(Curved for ramp-up)"
            else:
                 note = "(Linear)"
                 
            st.info(f"""
            **DIAGNOSTIC:**
            Deal is {elapsed:.1f} months into cycle {note}.
            Actual Recoupment: {recoup_pct:.1f}%
            Pace Ratio: {pace_ratio:.2f}x
            """)
        else:
            count_found = deal_row.get('Data Points Found', 0)
            st.warning(f"INSUFFICIENT DATA: FOUND {count_found} ACTUALS (NEED 3).")

    # --- CHARTS ---
    st.markdown("### > PERFORMANCE VISUALIZATION (NORMALIZED M1, M2...)")
    
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
            
            c1, c2 = st.columns(2)
            
            with c1:
                bar = alt.Chart(deal_act).mark_bar(color='#b026ff').encode(
                    x=alt.X('MonthLabel', sort=None, title='Period'),
                    y=alt.Y('Net Receipts', title='Net Receipts'),
                    tooltip=['MonthLabel', 'DateStr', 'Net Receipts']
                ).properties(title="MONTHLY ACTUALS")
                st.altair_chart(bar, use_container_width=True)
                
            with c2:
                line = alt.Chart(deal_act).mark_line(color='#33ff00', point=True).encode(
                    x=alt.X('MonthLabel', sort=None, title='Period'),
                    y=alt.Y('CumNet', title='Cumulative Net'),
                    tooltip=['MonthLabel', 'DateStr', 'CumNet']
                )
                rule_adv = alt.Chart(pd.DataFrame({'y': [adv_val]})).mark_rule(color='#ffbf00', strokeDash=[4, 4]).encode(y='y')
                st.altair_chart(line + rule_adv, use_container_width=True)
                
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
    df_dash_raw, df_act_raw = load_data()
    if df_dash_raw.empty:
        st.error("DATABASE OFFLINE: CHECK CONNECTIONS OR SHEET HEADERS.")
        st.stop()
    df_dash, df_act = process_data(df_dash_raw, df_act_raw)
    
    if 'selected_deal_id' in st.session_state:
        show_detail(df_dash, df_act, st.session_state['selected_deal_id'])
    else:
        show_portfolio(df_dash)

if __name__ == "__main__":
    main()
