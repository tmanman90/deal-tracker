I need to implement a significant logic update to the grading system called the "Half-Month Mulligan." This is designed to forgive the initial "streaming lag" (slow first month) without permanently punishing the artist’s cumulative grade.

Please update the script with the following changes:

1. Replace the entire Ramp-Up Curve logic in calculate_pace_metrics: Remove the existing linear_progress and curve_factor blocks (v1, v2, or v3). Replace them with this "Effective Time" logic:

Python

    # --- THE HALF-MONTH MULLIGAN ---
    # Subtract 0.5 months from the timeline to forgive the initial 'trickle'.
    # This raises the Pace Ratio by lowering the denominator, not the numerator.
    effective_months = max(0, elapsed_months - 0.5)
    
    # Calculate expected progress based on this effective time
    if target_months > 0:
        expected_progress = effective_months / target_months
    else:
        expected_progress = 0
        
    # Cap at 1.0 (100%)
    expected_progress = min(1.0, expected_progress)
2. Update the Grading Thresholds (The 'Safe Bet' Scale): Use these specific thresholds in the calculate_pace_metrics function to define "Success" as being on track for ~13.5 months:

A+ (Unicorn): pace_ratio >= 1.15

A (On Track): pace_ratio >= 1.00

B+ (Safe/Green): pace_ratio >= 0.90

B (Caution/Amber): pace_ratio >= 0.75

C (At Risk/Red): pace_ratio >= 0.60

D (Failure): pace_ratio >= 0.40

F: < 0.40

3. Update show_portfolio Color Logic: Ensure the grade_color logic inside the loop matches these groupings:

Green (#33ff00): Grades A+, A, and B+

Amber (#ffbf00): Grade B

Red (#ff3333): Grades C, D, and F

Note: Keep the "Smart Start Date" logic and the "Whole Month" calculation as they are. This update only changes how expected_progress is derived and how grades are assigned to the resulting ratio.

Please generate the full updated script.
