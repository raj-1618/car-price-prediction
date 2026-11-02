import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# ---------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_sample():
    return pd.read_csv("sample.csv")

D = load_model()
S = load_sample()
YEAR = D["data_year"]
BRANDS, ENGINES = D["brands"], D["engines"]

def row(brand, engine, year, mileage):
    return pd.DataFrame([{"brand": brand, "mileage": mileage,
                          "engine_type": engine, "age": YEAR - year}])

def predict(model_name, rows):
    return D["models"][model_name].predict(rows)

def rf_range(rows):
    """Estimate + 10th/90th percentile across the individual trees of the forest."""
    pipe = D["models"]["Random Forest"]
    X = pipe.named_steps["pre"].transform(rows)
    per_tree = np.array([t.predict(X) for t in pipe.named_steps["model"].estimators_])
    return (np.percentile(per_tree, 10, axis=0),
            per_tree.mean(axis=0),
            np.percentile(per_tree, 90, axis=0))

def money(x):
    return f"${x:,.0f}"

# ---------------------------------------------------------------- sidebar
st.sidebar.header("🚗 Your car")
brand = st.sidebar.selectbox("Brand", BRANDS, index=BRANDS.index("toyota") if "toyota" in BRANDS else 0)
default_engine = ENGINES.index("gas | 4 cylinders") if "gas | 4 cylinders" in ENGINES else 0
engine = st.sidebar.selectbox("Engine type (fuel | cylinders)", ENGINES, index=default_engine)
year = st.sidebar.slider("Year of manufacture", 1990, YEAR, 2015)
mileage = st.sidebar.slider("Mileage (miles)", 1000, 300000, 80000, step=1000)
model_name = st.sidebar.radio("Model", ["Random Forest", "Linear Regression"],
                              help="Random Forest is more accurate. Linear Regression is the baseline.")
st.sidebar.caption("The result updates instantly when you change anything.")

# ---------------------------------------------------------------- header
st.title("🚗 Used Car Price Prediction")
st.write("Estimate the fair resale price of a used car. Trained on "
         f"**{D['n_rows']:,}** cleaned listings from the Kaggle Used Cars (Craigslist) dataset. Prices in US$.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🎯 Predict", "⚖️ Compare cars", "📉 What-if curves", "🔎 Explore data", "🧠 Model insights"])

# ---------------------------------------------------------------- tab 1
with tab1:
    r = row(brand, engine, year, mileage)
    low, mid, high = rf_range(r)
    est = float(predict(model_name, r)[0])
    est = max(est, 0)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Estimated price ({model_name})", money(est))
    c2.metric("Likely range (Random Forest)", f"{money(low[0])} – {money(high[0])}")
    brand_median = S.loc[S["brand"] == brand, "price"].median()
    if not np.isnan(brand_median):
        c3.metric(f"Median {brand} in data", money(brand_median),
                  delta=f"{est - brand_median:+,.0f} vs median", delta_color="off")

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=est,
        number={"prefix": "$", "valueformat": ",.0f"},
        gauge={"axis": {"range": [0, max(60000, high[0] * 1.3)]},
               "bar": {"color": "#2563eb"},
               "steps": [{"range": [low[0], high[0]], "color": "#bfdbfe"}]},
        title={"text": "Estimate (shaded = likely range)"}))
    fig.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30))
    st.plotly_chart(fig, width="stretch")

    st.subheader("💬 Is this listing a good deal?")
    ask = st.number_input("Asking price you saw ($)", min_value=0, value=int(round(est, -2)), step=500)
    if ask > 0:
        diff = (ask - mid[0]) / mid[0] * 100
        if ask < low[0]:
            st.success(f"🟢 Great deal. {abs(diff):.0f}% below the typical price. Check the car's condition carefully.")
        elif ask <= mid[0]:
            st.success(f"🟢 Good price. {abs(diff):.0f}% below the typical price.")
        elif ask <= high[0]:
            st.warning(f"🟡 Fair, slightly high. {diff:.0f}% above the typical price. You can negotiate.")
        else:
            st.error(f"🔴 Overpriced. {diff:.0f}% above the typical price.")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.write("Car A comes from the sidebar. Set up Car B and compare.")
    cb1, cb2 = st.columns(2)
    with cb1:
        brand_b = st.selectbox("Car B brand", BRANDS, index=BRANDS.index("honda") if "honda" in BRANDS else 0, key="bb")
        engine_b = st.selectbox("Car B engine", ENGINES, index=default_engine, key="eb")
    with cb2:
        year_b = st.slider("Car B year", 1990, YEAR, 2018, key="yb")
        mileage_b = st.slider("Car B mileage", 1000, 300000, 50000, step=1000, key="mb")

    pa = float(predict(model_name, row(brand, engine, year, mileage))[0])
    pb = float(predict(model_name, row(brand_b, engine_b, year_b, mileage_b))[0])
    name_a = f"A: {brand.title()} {year}"
    name_b = f"B: {brand_b.title()} {year_b}"
    cmp_df = pd.DataFrame({"Car": [name_a, name_b], "Price": [pa, pb]})
    figc = px.bar(cmp_df, x="Car", y="Price", color="Car", text=cmp_df["Price"].map(money))
    figc.update_layout(showlegend=False, height=380, yaxis_title="Estimated price ($)")
    st.plotly_chart(figc, width="stretch")
    cheaper = name_a if pa < pb else name_b
    st.info(f"{cheaper} is estimated to cost {money(abs(pa - pb))} less.")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.write("See how the price of your car changes if you change one thing.")
    ages = np.arange(0, YEAR - 1990 + 1)
    years = YEAR - ages
    dep = pd.concat([row(brand, engine, int(y), mileage) for y in years])
    mil = np.arange(1000, 300001, 5000)
    mdf = pd.concat([row(brand, engine, year, int(m)) for m in mil])

    left, right = st.columns(2)
    with left:
        fa = go.Figure()
        for mn in ["Random Forest", "Linear Regression"]:
            fa.add_trace(go.Scatter(x=years, y=predict(mn, dep), name=mn, mode="lines"))
        fa.add_vline(x=year, line_dash="dot", annotation_text="your car")
        fa.update_layout(title="Price vs year of manufacture", xaxis_title="Year",
                         yaxis_title="Price ($)", height=400, legend=dict(orientation="h"))
        st.plotly_chart(fa, width="stretch")
    with right:
        fm = go.Figure()
        for mn in ["Random Forest", "Linear Regression"]:
            fm.add_trace(go.Scatter(x=mil, y=predict(mn, mdf), name=mn, mode="lines"))
        fm.add_vline(x=mileage, line_dash="dot", annotation_text="your car")
        fm.update_layout(title="Price vs mileage", xaxis_title="Mileage (miles)",
                         yaxis_title="Price ($)", height=400, legend=dict(orientation="h"))
        st.plotly_chart(fm, width="stretch")

    st.subheader("Same car, different engines")
    eng_df = pd.concat([row(brand, e, year, mileage) for e in ENGINES])
    eng_res = pd.DataFrame({"Engine": ENGINES, "Price": predict(model_name, eng_df)})
    eng_res = eng_res.sort_values("Price", ascending=False).head(10)
    fe = px.bar(eng_res, x="Price", y="Engine", orientation="h")
    fe.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fe, width="stretch")

# ---------------------------------------------------------------- tab 4
with tab4:
    st.write("Explore the real data the model learned from (random sample of 8,000 cars).")
    top = S["brand"].value_counts().head(8).index.tolist()
    chosen = st.multiselect("Brands to show", sorted(S["brand"].unique()), default=top[:4])
    yr_min, yr_max = st.slider("Year range", int(S["year"].min()), YEAR, (2005, YEAR))
    f = S[S["brand"].isin(chosen) & S["year"].between(yr_min, yr_max)]
    st.caption(f"{len(f):,} cars match")
    if len(f) == 0:
        st.warning("No cars match these filters.")
    else:
        e1, e2 = st.columns(2)
        with e1:
            st.plotly_chart(px.scatter(f, x="mileage", y="price", color="brand", opacity=0.5,
                                       title="Price vs mileage"), width="stretch")
        with e2:
            st.plotly_chart(px.box(f, x="brand", y="price", color="brand",
                                   title="Price distribution by brand"), width="stretch")
        avg_year = f.groupby("year")["price"].mean().reset_index()
        st.plotly_chart(px.line(avg_year, x="year", y="price", markers=True,
                                title="Average price by year of manufacture"), width="stretch")

# ---------------------------------------------------------------- tab 5
with tab5:
    st.subheader("Model performance on unseen test data")
    res = pd.DataFrame(D["results"]).T.reset_index().rename(columns={"index": "Model"})
    m1, m2 = st.columns([1, 2])
    with m1:
        st.dataframe(res.style.format({"MAE": "${:,.0f}", "RMSE": "${:,.0f}"}), hide_index=True)
        st.caption("MAE = average error. RMSE = like MAE but punishes big mistakes more. Lower is better.")
    with m2:
        long = res.melt(id_vars="Model", var_name="Metric", value_name="Error ($)")
        st.plotly_chart(px.bar(long, x="Metric", y="Error ($)", color="Model", barmode="group"),
                        width="stretch")

    st.subheader("Actual vs predicted price")
    pick = st.radio("Show model", ["Random Forest", "Linear Regression"], horizontal=True, key="avp")
    tp = D["test_points"]
    fav = px.scatter(tp, x="Actual", y=pick, opacity=0.4)
    top_v = float(tp["Actual"].max())
    fav.add_trace(go.Scatter(x=[0, top_v], y=[0, top_v], mode="lines",
                             line=dict(dash="dash", color="red"), name="Perfect prediction"))
    fav.update_layout(height=450, xaxis_title="Actual price ($)", yaxis_title="Predicted price ($)")
    st.plotly_chart(fav, width="stretch")

    st.subheader("What matters most for the price?")
    imp = pd.DataFrame({"Feature": list(D["importance"].keys()),
                        "Importance": list(D["importance"].values())}).sort_values("Importance")
    st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h"), width="stretch")
    st.caption("Source: Random Forest feature importance, with brand and engine categories added together.")
