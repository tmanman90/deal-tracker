# Sort Logic
    ascending = False
    sort_col = None
    
    # ENSURE TIE-BREAKER COLUMN EXISTS AND IS NUMERIC
    if "Cum Receipts" not in filtered.columns:
        filtered["Cum Receipts"] = 0.0
    filtered["Cum Receipts"] = pd.to_numeric(filtered["Cum Receipts"], errors="coerce").fillna(0)
    
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
        # Special numeric handling for Delta Months (keep existing behavior)
        if sort_col == "Delta Months":
             filtered['Delta Months'] = pd.to_numeric(filtered['Delta Months'], errors='coerce').fillna(0)
        
        # IMPLEMENT NEW MULTI-COLUMN SORT LOGIC
        if sort_opt == "Grade":
             # Grade Rank Ascending (Best First), Tie-Breaker: High Receipts First
             filtered = filtered.sort_values(by=["Grade_Rank", "Cum Receipts"], ascending=[True, False])
        elif sort_col == "Cum Receipts":
             # Explicit Sort by Receipts Descending (no tie breaker needed)
             filtered = filtered.sort_values(by="Cum Receipts", ascending=False)
        else:
             # Primary Sort Key, Tie-Breaker: High Receipts First
             filtered = filtered.sort_values(by=[sort_col, "Cum Receipts"], ascending=[ascending, False])
