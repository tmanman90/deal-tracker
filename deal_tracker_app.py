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
            + "  |  CUM $" + eligible["Cum Receipts"].map(lambda x: f"{x:,.0f}")
            + "  |  [" + eligible["did_norm"].astype(str) + "]"
        )

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

                # "Candles" derived from month-to-month receipts
                # Close = this month receipts
                # Open  = prior month receipts
                act["Close"] = pd.to_numeric(act["Net Receipts"], errors="coerce").fillna(0)
                act["Open"] = act["Close"].shift(1).fillna(act["Close"])

                # Candle body bounds
                act["BodyTop"] = act[["Open", "Close"]].max(axis=1)
                act["BodyBot"] = act[["Open", "Close"]].min(axis=1)

                # Wicks (we don’t have intra-month high/low, so wick = body range)
                act["High"] = act["BodyTop"]
                act["Low"] = act["BodyBot"]

                # Moving averages
                act["SMA3"] = act["Close"].rolling(3).mean()
                act["SMA6"] = act["Close"].rolling(6).mean()

                # Headline stats
                last_close = float(act["Close"].iloc[-1])
                prev_close = float(act["Close"].iloc[-2]) if len(act) >= 2 else last_close
                mom_pct = ((last_close - prev_close) / prev_close) if prev_close else 0.0
                sma3 = float(act["SMA3"].iloc[-1]) if pd.notna(act["SMA3"].iloc[-1]) else 0.0

                m1, m2, m3 = st.columns(3)
                m1.metric("LAST MONTH", f"${last_close:,.0f}")
                m2.metric("MoM %", f"{mom_pct*100:+.1f}%")
                m3.metric("RUN-RATE (SMA3)", f"${sma3:,.0f}/mo")

                # Altair chart: terminal-style candles
                base = alt.Chart(act).encode(
                    x=alt.X(
                        "Period End Date:T",
                        title=None,
                        axis=alt.Axis(labelColor="#33ff00", tickColor="#33ff00")
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
