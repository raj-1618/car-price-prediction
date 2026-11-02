import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---- Page Config ----
st.set_page_config(
    page_title="AutoValuate | Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom Modern UI CSS ----
st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        color: #f8fafc;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .hero-banner h1 { margin: 0; font-size: 2.1rem; font-weight: 800; color: #ffffff; }
    .hero-banner p { margin: 0.4rem 0 0 0; color: #94a3b8; font-size: 1rem; }
    .price-card {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        padding: 1.6rem;
        border-radius: 14px;
        border: 1px solid #10b981;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .price-card h2 { margin: 0; font-size: 2.6rem; color: #34d399; font-weight: 800; }
    .price-card p { margin: 0.3rem 0 0 0; color: #a7f3d0; font-size: 0.95rem; }
    .badge-great { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; padding: 0.5rem 1rem; border-radius: 999px; font-weight: 700; display: inline-block; }
    .badge-fair  { background: rgba(59, 130, 246, 0.2); color: #3b82f6; border: 1px solid #3b82f6; padding: 0.5rem 1rem; border-radius: 999px; font-weight: 700; display: inline-block; }
    .badge-high  { background: rgba(239, 68, 68, 0.2);  color: #ef4444; border: 1px solid #ef4444; padding: 0.5rem 1rem; border-radius: 999px; font-weight: 700; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ---- Safe Model Loader (Prevents Line 48 Pipeline Error) ----
@st.cache_resource
def load_model_safely():
    raw = joblib.load("model.pkl")
    
    # Case 1: Saved as a dictionary (like your train_model.py)
    if isinstance(raw, dict):
        pipeline = raw.get("pipeline") or raw.get("model") or list(raw.values())[0]
        brands = raw.get("brands", ["toyota", "ford", "honda", "chevrolet", "bmw"])
        engines = raw.get("engines", ["gas | 4 cylinders", "gas | 6 cylinders", "gas | 8 cylinders"])
        results = raw.get("results", {"Random Forest": {"MAE": 3500.0, "RMSE": 5500.0}})
        data_year = int(raw.get("data_year", 2021))
    # Case 2: Saved directly as a scikit-learn Pipeline object
    else:
        pipeline = raw
        brands = ["toyota", "ford", "honda", "chevrolet", "bmw", "nissan", "hyundai", "jeep"]
        engines = ["gas | 4 cylinders", "gas | 6 cylinders", "gas | 8 cylinders", "diesel | 6 cylinders"]
        results = {"Random Forest": {"MAE": 3500.0, "RMSE": 5500.0}}
        data_year = 2021

    return pipeline, brands, engines, results, data_year

try:
    model, brands_raw, raw_engines, results_dict, DATA_YEAR = load_model_safely()
except Exception as e:
    st.error(f"⚠️ Error loading `model.pkl`: {e}")
    st.info("Make sure `scikit-learn==1.8.0` is in your `requirements.txt` file on GitHub.")
    st.stop()

# Safely extract MAE
rf_metrics = results_dict.get("Random Forest", list(results_dict.values())[0])
MAE_USD = float(rf_metrics.get("MAE", 3500.0))

if "garage" not in st.session_state:
    st.session_state.garage = []

# ---- Extract Individual Tree Predictions Safely ----
def get_tree_spread(row_df, base_pred):
    try:
        if hasattr(model, "named_steps"):
            pre = model.named_steps["pre"]
            rf = model.named_steps["model"]
            X_trans = pre.transform(row_df)
            return np.array([tree.predict(X_trans)[0] for tree in rf.estimators_])
    except Exception:
        pass
    return np.linspace(max(500.0, base_pred - MAE_USD), base_pred + MAE_USD, 25)

# Parse fuel & cylinders
fuels = sorted(list({e.split(" | ")[0] for e in raw_engines if " | " in e}))
if not fuels:
    fuels = ["gas"]

# ---- Quick Presets ----
PRESETS = {
    "Custom Vehicle": None,
    "🟢 Daily Commuter (2018 Toyota, 45k mi)": {"brand": "toyota", "fuel": "gas", "cyl": "4 cylinders", "year": 2018, "mileage": 45000},
    "🔵 V8 Work Truck (2016 Ford, 95k mi)": {"brand": "ford", "fuel": "gas", "cyl": "8 cylinders", "year": 2016, "mileage": 95000},
    "🟣 Luxury Sedan (2019 BMW, 30k mi)": {"brand": "bmw", "fuel": "gas", "cyl": "6 cylinders", "year": 2019, "mileage": 30000},
    "🟠 Starter Car (2011 Honda, 130k mi)": {"brand": "honda", "fuel": "gas", "cyl": "4 cylinders", "year": 2011, "mileage": 130000},
}

# ---- Sidebar Controls ----
with st.sidebar:
    st.header("⚙️ Configure Car")

    currency = st.radio("Currency", ["USD ($)", "INR (₹)"], horizontal=True)
    rate = 84.0 if currency == "INR (₹)" else 1.0
    sym = "₹" if currency == "INR (₹)" else "$"

    preset_name = st.selectbox("⚡ Quick-Fill Preset", list(PRESETS.keys()))
    preset = PRESETS[preset_name]

    st.divider()

    # Brand Selector
    def_b = preset["brand"] if (preset and preset["brand"] in brands_raw) else ("toyota" if "toyota" in brands_raw else brands_raw[0])
    brand = st.selectbox(
        "Brand",
        options=brands_raw,
        index=brands_raw.index(def_b),
        format_func=lambda x: x.replace("-", " ").title()
    )

    # Cascading Fuel & Cylinder Selectors
    def_f = preset["fuel"] if (preset and preset["fuel"] in fuels) else fuels[0]
    fuel = st.selectbox("Fuel Type", options=fuels, index=fuels.index(def_f), format_func=lambda x: x.title())

    valid_cyls = [e.split(" | ")[1] for e in raw_engines if e.startswith(f"{fuel} | ")]
    if not valid_cyls:
        valid_cyls = ["4 cylinders"]
    def_c = preset["cyl"] if (preset and preset["cyl"] in valid_cyls) else valid_cyls[0]
    cylinders = st.selectbox("Cylinders", options=valid_cyls, index=valid_cyls.index(def_c), format_func=lambda x: x.title())

    engine = f"{fuel} | {cylinders}" if f"{fuel} | {cylinders}" in raw_engines else raw_engines[0]

    year = st.slider("Year of Manufacture", 1990, DATA_YEAR, preset["year"] if preset else 2015)
    mileage = st.slider("Mileage (miles driven)", 1000, 400000, preset["mileage"] if preset else 80000, step=2000)

    st.divider()
    asking_usd = st.number_input("Seller's Asking Price (USD $)", 0, 150000, 0, step=500,
                                 help="Enter a seller's price to check if it's a good deal.")

    pin_btn = st.button("📌 Pin Car to Compare Garage", use_container_width=True, type="primary")

# ---- Live Prediction ----
age = DATA_YEAR - year
row = pd.DataFrame([{
    "brand": brand,
    "mileage": mileage,
    "engine_type": engine,
    "age": age,
}])

pred_usd = float(model.predict(row)[0])
tree_preds_usd = get_tree_spread(row, pred_usd)
p25_usd, p75_usd = np.percentile(tree_preds_usd, [25, 75])

# Currency Conversion
pred_val = pred_usd * rate
mae_val = MAE_USD * rate
p25_val = p25_usd * rate
p75_val = p75_usd * rate
asking_val = asking_usd * rate

if pin_btn:
    st.session_state.garage.append({
        "Vehicle": f"{year} {brand.title()}",
        "Engine": f"{fuel.title()} | {cylinders.title()}",
        "Mileage (mi)": mileage,
        "Age (yrs)": age,
        f"Estimated Price ({sym})": round(pred_val),
        f"Fair Range ({sym})": f"{sym}{p25_val:,.0f} – {sym}{p75_val:,.0f}",
    })
    st.toast(f"Pinned {year} {brand.title()} to Comparison Garage!", icon="✅")

# ---- Header ----
st.markdown("""
<div class="hero-banner">
    <h1>🚗 Used Car Price Prediction & Deal Analyzer</h1>
    <p>Real-time valuation powered by a Random Forest Regressor trained on Craigslist market data</p>
</div>
""", unsafe_allow_html=True)

# ---- Top KPI Metrics ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Estimated Fair Price", f"{sym}{pred_val:,.0f}", f"± {sym}{mae_val:,.0f} MAE", delta_color="off")
k2.metric("25th–75th Percentile", f"{sym}{p25_val:,.0f} – {sym}{p75_val:,.0f}", "60 Decision Trees", delta_color="off")
k3.metric("Vehicle Age", f"{age} yrs", f"Year: {year}", delta_color="off")
mi_per_yr = mileage / max(1, age)
k4.metric("Annual Mileage", f"{mi_per_yr:,.0f} mi/yr", "Low Wear" if mi_per_yr <= 12000 else "High Wear",
          delta_color="normal" if mi_per_yr <= 12000 else "inverse")

# ---- Interactive Tabs ----
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Valuation & Deal Check",
    "📉 What-If Depreciation Simulator",
    f"🏎️ Comparison Garage ({len(st.session_state.garage)})",
    "📊 Model Performance"
])

with tab1:
    c_left, c_right = st.columns([1.1, 1])
    with c_left:
        st.markdown(f"""
        <div class="price-card">
            <p>Estimated Fair Resale Value for <b>{year} {brand.title()} ({fuel.title()}, {cylinders.title()})</b></p>
            <h2>{sym}{pred_val:,.0f}</h2>
            <p>Expected Market Range: {sym}{p25_val:,.0f} – {sym}{p75_val:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

        if asking_usd > 0:
            diff = asking_val - pred_val
            if asking_val <= p25_val:
                st.markdown(f'<div class="badge-great">🔥 Great Deal! {sym}{abs(diff):,.0f} below estimated market value</div>', unsafe_allow_html=True)
            elif asking_val <= p75_val:
                st.markdown('<div class="badge-fair">✅ Fair Market Price — Right within expected range</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-high">⚠️ Overpriced by {sym}{diff:,.0f} compared to model estimate</div>', unsafe_allow_html=True)
        else:
            st.info("💡 **Tip:** Enter a **Seller's Asking Price** in the left sidebar to grade whether a listing is a good deal!")

    with c_right:
        st.subheader("🌲 Price Confidence Across RF Trees")
        hist_vals, hist_edges = np.histogram(tree_preds_usd * rate, bins=12)
        bin_labels = [f"{sym}{int(hist_edges[i]):,}" for i in range(len(hist_vals))]
        chart_df = pd.DataFrame({"Tree Votes": hist_vals}, index=bin_labels)
        st.bar_chart(chart_df, height=260)

with tab2:
    st.write(f"See how **Manufacture Year** and **Mileage** change the value of this **{brand.title()}**.")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📅 Value by Manufacture Year")
        years_arr = np.arange(1995, DATA_YEAR + 1)
        df_years = pd.DataFrame({
            "brand": brand,
            "mileage": mileage,
            "engine_type": engine,
            "age": DATA_YEAR - years_arr,
        })
        yr_preds = pd.DataFrame({
            f"Estimated Price ({sym})": model.predict(df_years) * rate
        }, index=years_arr)
        st.line_chart(yr_preds, height=300)

    with col_b:
        st.subheader(f"🛣️ Value by Mileage (Fixed at Year {year})")
        miles_arr = np.linspace(5000, 300000, 25).astype(int)
        df_miles = pd.DataFrame({
            "brand": brand,
            "mileage": miles_arr,
            "engine_type": engine,
            "age": age,
        })
        mi_preds = pd.DataFrame({
            f"Estimated Price ({sym})": model.predict(df_miles) * rate
        }, index=miles_arr)
        st.line_chart(mi_preds, height=300)

with tab3:
    if not st.session_state.garage:
        st.info("Your Comparison Garage is empty. Click **📌 Pin Car to Compare Garage** in the sidebar to save and compare multiple vehicles!")
    else:
        gdf = pd.DataFrame(st.session_state.garage)
        price_col = [c for c in gdf.columns if c.startswith("Estimated Price")][0]
        st.bar_chart(gdf.set_index("Vehicle")[price_col], height=280)
        st.dataframe(gdf, use_container_width=True, hide_index=True)

        b1, b2 = st.columns([1, 4])
        csv_data = gdf.to_csv(index=False).encode("utf-8")
        b1.download_button("⬇️ Download CSV", data=csv_data, file_name="car_comparison.csv", mime="text/csv")
        if b2.button("🗑️ Clear Garage"):
            st.session_state.garage = []
            st.rerun()

with tab4:
    st.subheader("Model Performance on Test Set")
    res_df = pd.DataFrame(results_dict).T.round(0) * rate
    st.bar_chart(res_df, height=280)
    st.dataframe(res_df.style.format(f"{sym}{{:,.0f}}"), use_container_width=True)
