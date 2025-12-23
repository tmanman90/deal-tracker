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
