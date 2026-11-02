import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="AutoValuate | Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom Glassmorphism & Modern UI Styling ----
st.markdown("""
<style>
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        color: #f8fafc;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-card h1 { margin: 0; font-size: 2.1rem; font-weight: 800; }
    .hero-card p { margin: 0.4rem 0 0 0; color: #94a3b8; font-size: 0.98rem; }
    .verdict-pill {
        display: inline-block;
        padding: 0.45rem 1.1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    .verdict-great { background: rgba(16, 185, 129, 0.18); color: #10b981; border: 1px solid #10b981; }
    .verdict-fair  { background: rgba(59, 130, 246, 0.18); color: #3b82f6; border: 1px solid #3b82f6; }
    .verdict-high  { background: rgba(239, 68, 68, 0.18);  color: #ef4444; border: 1px solid #ef4444; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load():
    return joblib.load("model.pkl")

data = load()
model = data["pipeline"]
DATA_YEAR = int(data["data_year"])
MAE_USD = float(data["results"]["Random Forest"]["MAE"])

if "garage" not in st.session_state:
    st.session_state.garage = []

# Parse fuel & cylinder combinations from engine_type ("fuel | cylinders")
raw_engines = data["engines"]
fuels = sorted(list({e.split(" | ")[0] for e in raw_engines if " | " in e}))
brands_raw = data["brands"]

def get_tree_predictions(row_df):
    """Query individual decision trees inside the Random Forest for IQR range."""
    try:
        pre = model.named_steps["pre"]
        rf = model.named_steps["model"]
        X_trans = pre.transform(row_df)
        return np.array([tree.predict(X_trans)[0] for tree in rf.estimators_])
    except Exception:
        base = model.predict(row_df)[0]
        return np.linspace(max(500, base - MAE_USD), base + MAE_USD, 20)

# ---- Quick-Fill Vehicle Presets ----
PRESETS = {
    "Custom Configuration": None,
    "🟢 Daily Commuter (2018 Toyota, 45k mi)": {"brand": "toyota", "fuel": "gas", "cyl": "4 cylinders", "year": 2018, "mileage": 45000},
    "🔵 Workhorse Truck (2016 Ford, 95k mi)": {"brand": "ford", "fuel": "gas", "cyl": "8 cylinders", "year": 2016, "mileage": 95000},
    "🟣 Luxury Cruiser (2019 BMW, 30k mi)": {"brand": "bmw", "fuel": "gas", "cyl": "6 cylinders", "year": 2019, "mileage": 30000},
    "🟠 Budget First Car (2010 Honda, 135k mi)": {"brand": "honda", "fuel": "gas", "cyl": "4 cylinders", "year": 2010, "mileage": 135000},
}

# ---- Sidebar Controls ----
with st.sidebar:
    st.header("⚙️ Vehicle & Display Settings")

    currency = st.radio("Display Currency", ["USD ($)", "INR (₹)"], horizontal=True)
    rate = 84.0 if currency == "INR (₹)" else 1.0
    sym = "₹" if currency == "INR (₹)" else "$"

    preset_choice = st.selectbox("⚡ Quick-Load Preset", list(PRESETS.keys()))
    preset = PRESETS[preset_choice]

    st.divider()

    # Determine defaults based on preset
    def_brand = preset["brand"] if (preset and preset["brand"] in brands_raw) else ("toyota" if "toyota" in brands_raw else brands_raw[0])
    brand = st.selectbox(
        "Brand",
        options=brands_raw,
        index=brands_raw.index(def_brand),
        format_func=lambda x: x.replace("-", " ").title()
    )

    def_fuel = preset["fuel"] if (preset and preset["fuel"] in fuels) else fuels[0]
    fuel = st.selectbox(
        "Fuel Type",
        options=fuels,
        index=fuels.index(def_fuel),
        format_func=lambda x: x.title()
    )

    valid_cyls = [e.split(" | ")[1] for e in raw_engines if e.startswith(f"{fuel} | ")]
    def_cyl = preset["cyl"] if (preset and preset["cyl"] in valid_cyls) else valid_cyls[0]
    cylinders = st.selectbox(
        "Cylinders",
        options=valid_cyls,
        index=valid_cyls.index(def_cyl),
        format_func=lambda x: x.title()
    )
    engine = f"{fuel} | {cylinders}"

    year = st.slider(
        "Year of Manufacture",
        min_value=1990,
        max_value=DATA_YEAR,
        value=preset["year"] if preset else 2015
    )

    mileage = st.slider(
        "Mileage (miles driven)",
        min_value=1000,
        max_value=400000,
        value=preset["mileage"] if preset else 80000,
        step=2000,
        format="%d mi"
    )

    st.divider()
    asking_usd = st.number_input(
        "Seller's Asking Price (in USD $)",
        min_value=0,
        max_value=120000,
        value=0,
        step=500,
        help="Optional: Enter a listing price to grade the deal."
    )

    pin_car = st.button("📌 Pin to Comparison Garage", use_container_width=True, type="primary")

# ---- Compute Valuation ----
age = DATA_YEAR - year
row = pd.DataFrame([{
    "brand": brand,
    "mileage": mileage,
    "engine_type": engine,
    "age": age,
}])

pred_usd = float(model.predict(row)[0])
tree_preds_usd = get_tree_predictions(row)
p25_usd, p75_usd = np.percentile(tree_preds_usd, [25, 75])

# Convert to active currency
pred_val = pred_usd * rate
mae_val = MAE_USD * rate
p25_val = p25_usd * rate
p75_val = p75_usd * rate
asking_val = asking_usd * rate

if pin_car:
    st.session_state.garage.append({
        "Car": f"{year} {brand.title()}",
        "Engine": f"{fuel.title()} | {cylinders.title()}",
        "Mileage (mi)": mileage,
        "Age (yrs)": age,
        f"Estimated Price ({sym})": round(pred_val),
        f"IQR Range ({sym})": f"{sym}{p25_val:,.0f} – {sym}{p75_val:,.0f}"
    })
    st.toast(f"Pinned {year} {brand.title()} to your garage!", icon="🚗")

# ---- Hero Banner ----
st.markdown("""
<div class="hero-card">
    <h1>🚗 Used Car Resale Price Intelligence</h1>
    <p>Interactive Random Forest valuation, deal analyzer, and depreciation simulator</p>
</div>
""", unsafe_allow_html=True)

# ---- Top Metrics Row ----
m1, m2, m3, m4 = st.columns(4)
m1.metric("Estimated Resale Price", f"{sym}{pred_val:,.0f}", f"± {sym}{mae_val:,.0f} Typical Error", delta_color="off")
m2.metric("Fair Market Spread (IQR)", f"{sym}{p25_val:,.0f} – {sym}{p75_val:,.0f}", "60 Decision Trees", delta_color="off")
m3.metric("Vehicle Age", f"{age} Years", f"Manufactured {year}", delta_color="off")
miles_per_yr = mileage / max(1, age)
m4.metric("Usage Intensity", f"{miles_per_yr:,.0f} mi/yr", "Low usage" if miles_per_yr <= 12000 else "High usage",
          delta_color="normal" if miles_per_yr <= 12000 else "inverse")

# ---- Main Interactive Tabs ----
tab_val, tab_sim, tab_gar, tab_perf = st.tabs([
    "🎯 Live Valuation & Deal Grader",
    "📉 Interactive Depreciation Simulator",
    f"🏎️ Comparison Garage ({len(st.session_state.garage)})",
    "🧠 Model Benchmarks"
])

with tab_val:
    left, right = st.columns([1.1, 1])
    with left:
        max_axis = max(pred_val * 1.6, asking_val * 1.25, 25000 * rate)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta" if asking_usd > 0 else "gauge+number",
            value=asking_val if asking_usd > 0 else pred_val,
            number={"prefix": sym, "valueformat": ",.0f"},
            delta={"reference": pred_val, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}} if asking_usd > 0 else None,
            title={"text": "Seller Asking Price vs. Model Estimate" if asking_usd > 0 else f"Fair Resale Estimate ({currency})"},
            gauge={
                "axis": {"range": [0, max_axis], "tickprefix": sym},
                "bar": {"color": "#38bdf8"},
                "steps": [
                    {"range": [0, p25_val], "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [p25_val, p75_val], "color": "rgba(59, 130, 246, 0.25)"},
                    {"range": [p75_val, max_axis], "color": "rgba(239, 68, 68, 0.25)"},
                ],
                "threshold": {"line": {"color": "#f59e0b", "width": 4}, "thickness": 0.8, "value": pred_val}
            }
        ))
        fig_g.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=15))
        st.plotly_chart(fig_g, use_container_width=True)

        if asking_usd > 0:
            diff_val = asking_val - pred_val
            if asking_val <= p25_val:
                st.markdown(f'<span class="verdict-pill verdict-great">🔥 Great Deal: {sym}{abs(diff_val):,.0f} below fair value!</span>', unsafe_allow_html=True)
            elif asking_val <= p75_val:
                st.markdown('<span class="verdict-pill verdict-fair">✅ Fair Listing: Within normal market range</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="verdict-pill verdict-high">⚠️ Overpriced: {sym}{diff_val:,.0f} above predicted value</span>', unsafe_allow_html=True)

    with right:
        fig_hist = px.histogram(
            x=tree_preds_usd * rate,
            nbins=18,
            labels={"x": f"Tree Estimate ({sym})", "y": "Tree Count"},
            title="Vote Distribution Across All 60 Random Forest Trees",
            color_discrete_sequence=["#6366f1"]
        )
        fig_hist.add_vline(x=pred_val, line_dash="dash", line_color="#f59e0b", annotation_text="Ensemble Mean")
        fig_hist.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=15), showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

with tab_sim:
    st.caption(f"Simulating how age and mileage affect a **{brand.title()} ({fuel.title()} | {cylinders})**.")
    sc1, sc2 = st.columns(2)

    with sc1:
        yrs = np.arange(1995, DATA_YEAR + 1)
        df_yr = pd.DataFrame({"brand": brand, "mileage": mileage, "engine_type": engine, "age": DATA_YEAR - yrs})
        preds_yr = model.predict(df_yr) * rate

        fig_yr = go.Figure()
        fig_yr.add_trace(go.Scatter(x=yrs, y=preds_yr, mode="lines+markers", name="Resale Curve", line=dict(color="#10b981", width=3)))
        fig_yr.add_trace(go.Scatter(x=[year], y=[pred_val], mode="markers", name="Selected Car",
                                    marker=dict(color="#f59e0b", size=14, line=dict(color="white", width=2))))
        fig_yr.update_layout(title="Price by Manufacture Year", xaxis_title="Year", yaxis_title=f"Price ({sym})", height=350)
        st.plotly_chart(fig_yr, use_container_width=True)

    with sc2:
        miles_grid = np.linspace(5000, 300000, 25)
        df_mi = pd.DataFrame({"brand": brand, "mileage": miles_grid, "engine_type": engine, "age": age})
        preds_mi = model.predict(df_mi) * rate

        fig_mi = go.Figure()
        fig_mi.add_trace(go.Scatter(x=miles_grid, y=preds_mi, mode="lines+markers", name="Mileage Curve", line=dict(color="#38bdf8", width=3)))
        fig_mi.add_trace(go.Scatter(x=[mileage], y=[pred_val], mode="markers", name="Selected Car",
                                    marker=dict(color="#f59e0b", size=14, line=dict(color="white", width=2))))
        fig_mi.update_layout(title=f"Price vs. Mileage (Fixed at Year {year})", xaxis_title="Mileage (mi)", yaxis_title=f"Price ({sym})", height=350)
        st.plotly_chart(fig_mi, use_container_width=True)

with tab_gar:
    if not st.session_state.garage:
        st.info("No vehicles pinned yet! Click **📌 Pin to Comparison Garage** in the sidebar to compare cars.")
    else:
        gdf = pd.DataFrame(st.session_state.garage)
        price_col = [c for c in gdf.columns if c.startswith("Estimated Price")][0]
        fig_bar = px.bar(gdf, x="Car", y=price_col, color="Engine", text_auto=",.0f", title="Side-by-Side Garage Comparison")
        fig_bar.update_layout(height=340)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(gdf, use_container_width=True, hide_index=True)

        c_dl, c_clr = st.columns([1, 4])
        csv_bytes = gdf.to_csv(index=False).encode("utf-8")
        c_dl.download_button("⬇️ Export Garage CSV", data=csv_bytes, file_name="car_valuation_comparison.csv", mime="text/csv")
        if c_clr.button("🗑️ Clear Garage"):
            st.session_state.garage = []
            st.rerun()

with tab_perf:
    res_df = pd.DataFrame(data["results"]).T.reset_index().rename(columns={"index": "Algorithm"})
    res_df["MAE"] = res_df["MAE"] * rate
    res_df["RMSE"] = res_df["RMSE"] * rate

    pc1, pc2 = st.columns([1.2, 1])
    with pc1:
        melted = res_df.melt(id_vars="Algorithm", var_name="Metric", value_name="Error")
        fig_p = px.bar(melted, x="Algorithm", y="Error", color="Metric", barmode="group", text_auto=",.0f",
                       title=f"Test Set Error Comparison ({currency})")
        fig_p.update_layout(height=330)
        st.plotly_chart(fig_p, use_container_width=True)
    with pc2:
        st.dataframe(res_df.style.format({"MAE": f"{sym}{{:,.0f}}", "RMSE": f"{sym}{{:,.0f}}"}), use_container_width=True, hide_index=True)
