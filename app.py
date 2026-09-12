import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & DARK THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Valuation & DCF - Multi-Factor OLS",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Theme Custom Styling */
    .stApp {
        background-color: #0b0d17;
        color: #e2e8f0;
    }
    .metric-card {
        background-color: #161b26;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 11px;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        margin-top: 4px;
    }
    .formula-box {
        background-color: #0d1322;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 15px;
        font-family: monospace;
        color: #60a5fa;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HEADER & NAVIGATION
# -----------------------------------------------------------------------------
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.title("UBER ANALYTICS")
    st.caption("02. Multi-Factor OLS Econometric Regression Engine — Matrix OLS Model")
with head_col2:
    st.metric(label="NYSE: UBER", value="$72.50", delta="+1.2%")

st.divider()

# -----------------------------------------------------------------------------
# 3. SIDEBAR & INITIAL DATA SET
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")
target_ticker = st.sidebar.text_input("Target Stock Ticker", value="UBER").upper()
target_price = st.sidebar.number_input("Target Current Share Price ($)", value=72.50, step=0.50)
ntm_ebitda_per_share = st.sidebar.number_input("NTM EBITDA / Share ($)", value=5.00, step=0.25)
net_debt_per_share = st.sidebar.number_input("Net Debt / Share ($)", value=0.00, step=0.25)

# Initial dataset matching peer dispersion
default_data = {
    "Ticker": ["LYFT", "DASH", "GRUB", "ABNB", "EXPE", "BKNG", "BUYN", target_ticker],
    "Rev_Growth": [11.2, 22.5, 7.8, 19.4, 12.0, 16.2, 24.1, 16.5],
    "FCF_Margin": [8.5, 18.2, 4.1, 28.5, 11.2, 31.0, 20.4, 15.0],
    "Actual_EV_EBITDA": [9.8, 21.8, 8.2, 23.5, 14.1, 18.4, 25.1, 14.2]
}

st.sidebar.subheader("Peer Data Editor")
df_input = st.sidebar.data_editor(
    pd.DataFrame(default_data),
    num_rows="dynamic",
    key="peer_editor"
)

# -----------------------------------------------------------------------------
# 4. ECONOMETRIC MULTI-FACTOR OLS CALCULATION
# -----------------------------------------------------------------------------
X = df_input[["Rev_Growth", "FCF_Margin"]]
Y = df_input["Actual_EV_EBITDA"]
X_with_const = sm.add_constant(X)

# Fit OLS Model
ols_model = sm.OLS(Y, X_with_const).fit()

# Extract Parameters
beta_0, beta_1, beta_2 = ols_model.params
r_squared = ols_model.rsquared
adj_r2 = ols_model.rsquared_adj
f_stat = ols_model.fvalue
multiple_r = np.sqrt(r_squared) if r_squared >= 0 else 0

# Predict values
df_input["Predicted_EV_EBITDA"] = ols_model.predict(X_with_const)

# Isolate Target Row
target_mask = df_input["Ticker"] == target_ticker
if target_mask.any():
    target_row = df_input[target_mask].iloc[0]
    actual_multiple = target_row["Actual_EV_EBITDA"]
    predicted_multiple = target_row["Predicted_EV_EBITDA"]
else:
    actual_multiple = 14.2
    predicted_multiple = 17.1

valuation_spread = ((predicted_multiple - actual_multiple) / actual_multiple) * 100
implied_fair_value = (predicted_multiple * ntm_ebitda_per_share) - net_debt_per_share
upside_pct = ((implied_fair_value - target_price) / target_price) * 100

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD LAYOUT
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1.6, 1.0])

# --- LEFT COLUMN: SCATTER PLOT & PARITY LINE ---
with left_col:
    st.subheader("MODEL CALIBRATION: ACTUAL VS. ECONOMETRIC PREDICTED MULTIPLE")

    # Filter out target vs peers for custom plotting colors
    peers_df = df_input[df_input["Ticker"] != target_ticker]
    target_df = df_input[df_input["Ticker"] == target_ticker]

    fig = go.Figure()

    # 45-Degree Parity Line
    min_axis = min(df_input["Predicted_EV_EBITDA"].min(), df_input["Actual_EV_EBITDA"].min()) - 2
    max_axis = max(df_input["Predicted_EV_EBITDA"].max(), df_input["Actual_EV_EBITDA"].max()) + 2

    fig.add_trace(go.Scatter(
        x=[min_axis, max_axis],
        y=[min_axis, max_axis],
        mode="lines",
        name="45° Multiple Parity Line",
        line=dict(color="#7c3aed", width=2, dash="dash")
    ))

    # Peer Scatter Points
    fig.add_trace(go.Scatter(
        x=peers_df["Predicted_EV_EBITDA"],
        y=peers_df["Actual_EV_EBITDA"],
        mode="markers+text",
        text=peers_df["Ticker"],
        textposition="top center",
        name="Peer Universe",
        marker=dict(size=12, color="#c084fc", line=dict(color="#ffffff", width=1))
    ))

    # Target Company Point
    fig.add_trace(go.Scatter(
        x=target_df["Predicted_EV_EBITDA"],
        y=target_df["Actual_EV_EBITDA"],
        mode="markers+text",
        text=target_df["Ticker"],
        textposition="bottom right",
        name=f"{target_ticker} (Discount Gap)",
        marker=dict(size=18, color="#10b981", line=dict(color="#ffffff", width=2))
    ))

    fig.update_layout(
        xaxis_title="Model Predicted EV / EBITDA (x)",
        yaxis_title="Actual Market EV / EBITDA (x)",
        template="plotly_dark",
        paper_bgcolor="#0b0d17",
        plot_bgcolor="#0b0d17",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("📌 **Interpretation:** Points BELOW 45° line = Undervalued vs. Fundamentals | Points ABOVE 45° line = Overvalued / Premium")

# --- RIGHT COLUMN: MODEL METRICS & IMPLIED FAIR VALUE ---
with right_col:
    st.subheader("MULTIVARIATE MATRIX FORMULA")
    
    st.markdown(f"""
    <div class="formula-box">
        <b>Predicted EV/EBITDA =</b><br>
        ({beta_0:.2f}) + ({beta_1:.2f} × Rev Growth %) + ({beta_2:.2f} × FCF Margin %)
    </div>
    """, unsafe_allow_html=True)

    # OLS Model Performance Stats
    col_r, col_adj, col_f = st.columns(3)
    with col_r:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Multiple R</div><div class="metric-value">{multiple_r:.3f}</div></div>', unsafe_allow_html=True)
    with col_adj:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Adjusted R²</div><div class="metric-value">{adj_r2:.3f}</div></div>', unsafe_allow_html=True)
    with col_f:
        st.markdown(f'<div class="metric-card"><div class="metric-title">F-Statistic</div><div class="metric-value">{f_stat:.1f}</div></div>', unsafe_allow_html=True)

    st.divider()

    # Target Valuation Diagnostics
    st.markdown(f"**{target_ticker} Actual Multiple:** `{actual_multiple:.1f}x EV/EBITDA`")
    st.markdown(f"**Model Predicted Multiple:** `{predicted_multiple:.1f}x EV/EBITDA`")
    
    status_text = "Underpriced vs Model" if valuation_spread > 0 else "Overpriced vs Model"
    st.markdown(f"**Multi-Factor Valuation Spread:** <span style='color:#10b981; font-weight:bold;'>{valuation_spread:.1f}% ({status_text})</span>", unsafe_allow_html=True)

    st.divider()

    # Final Output Valuation
    st.subheader("MULTI-FACTOR IMPLIED FAIR VALUE")
    val_c1, val_c2 = st.columns(2)
    with val_c1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Implied Fair Price</div><div class="metric-value" style="color:#ffffff;">${implied_fair_value:.2f}</div></div>', unsafe_allow_html=True)
    with val_c2:
        upside_color = "#10b981" if upside_pct > 0 else "#ef4444"
        st.markdown(f'<div class="metric-card"><div class="metric-title">Upside to Parity</div><div class="metric-value" style="color:{upside_color};">+{upside_pct:.1f}%</div></div>', unsafe_allow_html=True)
