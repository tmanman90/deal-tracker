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
        
        # 1. READ DASHBOARD
        ws_dash = sh.worksheet("DASHBOARD")
        df_dash = pd.DataFrame(ws_dash.get_all_records())
        
        # 2. READ ACTUALS
        ws_act = sh.worksheet("ACTUALS")
        df_act = pd.DataFrame(ws_act.get_all_records())
        
        return df_dash, df_act
        
    except Exception as e:
        st.error(f"SYSTEM FAILURE: Connection Refused. {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# -----------------------------------------------------------------------------
# CLEANING & LOGIC UTILS
# -----------------------------------------------------------------------------
def clean_currency(val):
    """Converts currency strings to floats."""
    if isinstance(val, (int, float)):
        return float(val)
    if not val or val == "":
        return 0.0
    # Remove common currency symbols
    clean_str = str(val).replace('$', '').replace(',', '').replace(' ', '')
    try:
        return float(clean_str)
    except:
        return 0.0

def clean_percent(val):
    """Converts '50%', 50, or 0.5 to a 0-1 float."""
    if pd.isna(val) or val == "":
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

def calculate_pace_metrics(row, data_months_count):
    """
    Calculates Grade and Pace based on Benchmark (12 months).
    Returns (Grade, Pace Ratio, Eligible Boolean, Elapsed Months).
    """
    # 1. Eligibility Check
    if data_months_count < 5:
        return "N/A", 0.0, False, 0
    
    # 2. Parse Dates
    try:
        forecast_start = pd.to_datetime(row['Forecast Start Date'])
        if pd.isna(forecast_start):
            return "N/A", 0.0, False, 0
    except:
        return "N/A", 0.0, False, 0
        
    # 3. Calculate Elapsed Months vs Benchmark
    today = pd.Timestamp.now()
    if forecast_start > today:
        elapsed_months = 0 # Not started
    else:
        elapsed_months = (today - forecast_start).days / 30.4375
    
    # Avoid div by zero or negative
    elapsed_months = max(0.1, elapsed_months)
    
    # Benchmark: 100% Recoupment in 12 Months
    # Expected progress (capped at 1.0 i.e. 100%)
    expected_progress = min(1.0, elapsed_months / 12.0)
    
    # Actual progress
    actual_progress = clean_percent(row['% to BE'])
    
    # Pace Ratio
    if expected_progress == 0:
        pace_ratio = 0
    else:
        pace_ratio = actual_progress / expected_progress
        
    # 4. Assign Grade
    if pace_ratio >= 1.25:
        grade = "A"
    elif pace_ratio >= 1.05:
        grade = "B"
    elif pace_ratio >= 0.85:
        grade = "C"
    elif pace_ratio >= 0.65:
        grade = "D"
    else:
        grade = "F"
        
    return grade, pace_ratio, True, elapsed_months

# -----------------------------------------------------------------------------
# DATA PROCESSING WRAPPER
# -----------------------------------------------------------------------------
def process_data(df_dash, df_act):
    # Process Actuals Dates & Values
    df_act['Net Receipts'] = df_act['Net Receipts'].apply(clean_currency)
    df_act['Period End Date'] = pd.to_datetime(df_act['Period End Date'], errors='coerce')
    
    # Clean Dashboard Numerics
    df_dash['Executed Advance'] = df_dash['Executed Advance'].apply(clean_currency)
    df_dash['Cum Receipts'] = df_dash['Cum Receipts'].apply(clean_currency)
    df_dash['Remaining to BE'] = df_dash['Remaining to BE'].apply(clean_currency)
    
    # Calculate Data Eligibility (Distinct Months per Deal)
    # Group by Deal ID, convert date to period 'M' to count unique months
    eligibility_map = {}
    if not df_act.empty:
        temp = df_act.dropna(subset=['Period End Date']).copy()
        temp['MonthPeriod'] = temp['Period End Date'].dt.to_period('M')
        counts = temp.groupby('Deal ID')['MonthPeriod'].nunique()
        eligibility_map = counts.to_dict()
    
    # Enrich Dashboard with Grades
    grades = []
    ratios = []
    is_eligible = []
    elapsed_list = []
    
    for _, row in df_dash.iterrows():
        did = str(row['Deal ID'])
        months_count = eligibility_map.get(did, 0) # default 0 if no actuals
        g, r, e, el_m = calculate_pace_metrics(row, months_count)
        grades.append(g)
        ratios.append(r)
        is_eligible.append(e)
        elapsed_list.append(el_m)
        
    df_dash['Grade'] = grades
    df_dash['Pace Ratio'] = ratios
    df_dash['Is Eligible'] = is_eligible
    df_dash['Elapsed Months'] = elapsed_list
    
    # Clean % to BE for display/sorting
    df_dash['% to BE Clean'] = df_dash['% to BE'].apply(clean_percent)
    
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
        filtered = filtered[
            filtered['Artist / Project'].astype(str).str.lower().str.contains(search) | 
            filtered['Deal ID'].astype(str).str.lower().str.contains(search)
        ]
        
    if status_filter:
        filtered = filtered[filtered['Status'].isin(status_filter)]
        
    if eligible_only:
        filtered = filtered[filtered['Is Eligible'] == True]
        
    # Sort Logic
    ascending = False
    if sort_opt == "Remaining to BE":
        sort_col = "Remaining to BE"
    elif sort_opt == "% to BE":
        sort_col = "% to BE Clean"
    elif sort_opt == "Grade":
        # Sort Grade alphabetically (A is 'smaller' than B, so ascending=True brings A to top)
        sort_col = "Grade"
        ascending = True
    elif sort_opt == "Cum Receipts":
        sort_col = "Cum Receipts"
    else:
        sort_col = "Delta Months" # Assuming this exists or handled
        # Clean Delta Months just in case
        if "Delta Months" in filtered.columns:
            filtered['Delta Months'] = pd.to_numeric(filtered['Delta Months'], errors='coerce').fillna(0)

    if sort_col in filtered.columns:
        filtered = filtered.sort_values(by=sort_col, ascending=ascending)

    # --- KPI CARDS ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric("ACTIVE DEALS", len(filtered))
    
    total_adv = filtered['Executed Advance'].sum()
    kpi2.metric("TOTAL ADVANCES", f"${total_adv:,.0f}")
    
    total_rec = filtered['Cum Receipts'].sum()
    kpi3.metric("TOTAL CUM RECEIPTS", f"${total_rec:,.0f}")
    
    # Weighted % to BE (Total Rec / Total Adv)
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
    # Ensure cols exist
    existing_cols = [c for c in display_cols if c in filtered.columns]
    
    display_df = filtered[existing_cols].copy()
    
    # Hide Grade if not eligible
    if 'Grade' in display_df.columns:
        # We need to map back to original indices to check eligibility, 
        # but display_df is filtered. We can join on index or re-apply logic.
        # Simplest: use 'Is Eligible' from filtered df
        mask_ineligible = filtered['Is Eligible'] == False
        display_df.loc[mask_ineligible, 'Grade'] = "WAITING"
        
    # Use Streamlit's selection API
    st.markdown("### > SELECT DEAL TO INITIALIZE ANALYSIS")
    
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        height=500
    )
    
    # Check selection
    if len(event.selection.rows) > 0:
        row_idx = event.selection.rows[0]
        # Get the actual Deal ID from the filtered dataframe
        selected_deal_id = filtered.iloc[row_idx]['Deal ID']
        st.session_state['selected_deal_id'] = selected_deal_id
        st.rerun()

# -----------------------------------------------------------------------------
# UI: DEAL DETAIL PAGE
# -----------------------------------------------------------------------------
def show_detail(df_dash, df_act, deal_id):
    # Get Deal Data
    deal_row = df_dash[df_dash['Deal ID'] == deal_id].iloc[0]
    
    # Deal Specific Actuals
    deal_act = df_act[df_act['Deal ID'] == deal_id].copy()
    
    # --- NAVIGATION ---
    if st.button("<< RETURN TO DASHBOARD"):
        del st.session_state['selected_deal_id']
        st.rerun()
        
    st.title(f"// ANALYZING: {deal_row['Artist / Project']} [{deal_id}]")
    
    # --- HEADER STATS ---
    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
    
    # Formatting helper
    grade_display = deal_row['Grade'] if deal_row['Is Eligible'] else "PENDING"
    
    row1_c1.metric("STATUS", deal_row['Status'])
    row1_c2.metric("PERFORMANCE GRADE", grade_display, delta_color="normal")
    row1_c3.metric("EXECUTED ADVANCE", f"${deal_row['Executed Advance']:,.0f}")
    row1_c4.metric("% RECOUPED", f"{deal_row['% to BE Clean']*100:.1f}%")

    row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
    row2_c1.metric("CUM RECEIPTS", f"${deal_row['Cum Receipts']:,.0f}")
    row2_c2.metric("REMAINING", f"${deal_row['Remaining to BE']:,.0f}")
    row2_c3.metric("FORECAST START", str(deal_row['Forecast Start Date']))
    row2_c4.metric("EST BREAKEVEN", str(deal_row['Predicted BE Date']))

    st.markdown("---")

    # --- PACE BLOCK ---
    st.markdown("### > PACE ANALYSIS")
    
    col_gauge, col_stats = st.columns([1, 2])
    
    with col_gauge:
        # Create a simple gauge chart using Altair
        pace_ratio = deal_row['Pace Ratio']
        
        # Clamp for visualization
        chart_val = min(2.0, max(0.0, pace_ratio))
        
        # Simple bar gauge
        gauge_df = pd.DataFrame({'val': [chart_val], 'label': ['Pace']})
        
        gauge = alt.Chart(gauge_df).mark_bar(size=40).encode(
            x=alt.X('val', scale=alt.Scale(domain=[0, 2]), title="Pace Ratio (1.0 = On Track)"),
            color=alt.condition(
                alt.datum.val < 0.8,
                alt.value('red'),
                alt.condition(
                    alt.datum.val < 1.0,
                    alt.value('orange'),
                    alt.value('#33ff00') # Neon Green
                )
            )
        ).properties(height=80, title="PACE METER")
        
        # Add benchmark line at 1.0
        rule = alt.Chart(pd.DataFrame({'x': [1.0]})).mark_rule(color='white', strokeDash=[5, 5]).encode(x='x')
        
        st.altair_chart(gauge + rule, use_container_width=True)
        
    with col_stats:
        if deal_row['Is Eligible']:
            st.info(f"""
            **DIAGNOSTIC:**
            Deal is {deal_row['Elapsed Months']:.1f} months into cycle (Benchmark: 12.0).
            Expected Recoupment: {min(100, (deal_row['Elapsed Months']/12)*100):.1f}%
            Actual Recoupment: {deal_row['% to BE Clean']*100:.1f}%
            Pace Ratio: {pace_ratio:.2f}x
            """)
        else:
            st.warning("INSUFFICIENT DATA FOR PACE GRADING (<5 MONTHS ACTUALS)")

    # --- CHARTS ---
    st.markdown("### > PERFORMANCE VISUALIZATION (NORMALIZED M1, M2...)")
    
    if not deal_act.empty:
        # Prepare Data for Charts
        # Sort by date
        deal_act = deal_act.sort_values('Period End Date')
        
        # Normalize Date to "M#"
        deal_act['MonthIndex'] = range(1, len(deal_act) + 1)
        deal_act['MonthLabel'] = deal_act['MonthIndex'].apply(lambda x: f"M{x}")
        
        # Calculate Cumulative
        deal_act['CumNet'] = deal_act['Net Receipts'].cumsum()
        
        # Rolling Average (3 months)
        deal_act['Rolling3'] = deal_act['Net Receipts'].rolling(window=3).mean()
        
        c1, c2 = st.columns(2)
        
        with c1:
            # BAR CHART: MONTHLY RECEIPTS
            bar = alt.Chart(deal_act).mark_bar(color='#b026ff').encode(
                x=alt.X('MonthLabel', sort=None, title='Period'),
                y=alt.Y('Net Receipts', title='Net Receipts'),
                tooltip=['MonthLabel', 'Net Receipts', 'Period End Date']
            ).properties(title="MONTHLY ACTUALS")
            st.altair_chart(bar, use_container_width=True)
            
        with c2:
            # LINE CHART: CUMULATIVE vs ADVANCE
            line = alt.Chart(deal_act).mark_line(color='#33ff00', point=True).encode(
                x=alt.X('MonthLabel', sort=None, title='Period'),
                y=alt.Y('CumNet', title='Cumulative Net'),
                tooltip=['MonthLabel', 'CumNet']
            )
            
            # Advance Line
            adv_val = deal_row['Executed Advance']
            rule_adv = alt.Chart(pd.DataFrame({'y': [adv_val]})).mark_rule(color='#ffbf00', strokeDash=[4, 4]).encode(y='y')
            
            st.altair_chart(line + rule_adv, use_container_width=True)
            
        # --- PROJECTION LOGIC ---
        st.markdown("### > TERMINAL FORECAST")
        
        last_rolling = deal_act['Rolling3'].iloc[-1] if len(deal_act) > 0 else 0
        remaining = deal_row['Remaining to BE']
        
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
        st.warning("NO ACTUALS DATA FOUND ON SERVER.")


# -----------------------------------------------------------------------------
# MAIN APP LOOP
# -----------------------------------------------------------------------------
def main():
    # Load Data
    df_dash_raw, df_act_raw = load_data()
    
    if df_dash_raw.empty:
        st.stop()
        
    # Process
    df_dash, df_act = process_data(df_dash_raw, df_act_raw)
    
    # Router
    if 'selected_deal_id' in st.session_state:
        show_detail(df_dash, df_act, st.session_state['selected_deal_id'])
    else:
        show_portfolio(df_dash)

if __name__ == "__main__":
    main()