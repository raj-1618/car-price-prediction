import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

import json

# ---------------------------------------------------------------- motion
# Motion thesis: the price is the product, so it is the one authored moment
# (the estimate counts and the range bar glides to the new value). Everything
# else is quiet feedback: tab panels fade in, verdicts slide in, charts morph.
st.markdown("""
<style>
:root { --ease-out: cubic-bezier(0.16, 1, 0.3, 1); --accent: #2A8FB0; }
@keyframes panel-in   { from { opacity: 0; transform: translateY(6px); }  to { opacity: 1; transform: none; } }
@keyframes verdict-in { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: none; } }
@keyframes fade-only  { from { opacity: 0; } to { opacity: 1; } }
[data-baseweb="tab-panel"]  { animation: panel-in 240ms var(--ease-out); }
[data-testid="stAlert"]     { animation: verdict-in 220ms var(--ease-out); }
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input { transition: border-color 150ms ease-out, box-shadow 150ms ease-out; }
::selection { background: rgba(42, 143, 176, 0.30); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  [data-baseweb="tab-panel"], [data-testid="stAlert"] { animation: fade-only 120ms linear; }
}
</style>
""", unsafe_allow_html=True)

HERO_HTML = """
<style>
  :root { --fg:#1B1F2A; --muted:#5B6B7F; --track:#E1E6EC; --band:rgba(31,111,139,.30); --accent:#1F6F8B;
          --ease:cubic-bezier(0.16,1,0.3,1); }
  :root.dark { --fg:#F1F4F8; --muted:#A9B4C2; --track:#2A3140; --band:rgba(111,195,222,.32); --accent:#6FC3DE; }
  html, body { margin:0; background:transparent; color:var(--fg); font-family:"Source Sans Pro", system-ui, sans-serif; }
  .label { font-size:14px; color:var(--muted); }
  .price { font-size:56px; font-weight:700; line-height:1.1; letter-spacing:-0.02em; font-variant-numeric:tabular-nums; }
  .range { font-size:16px; font-variant-numeric:tabular-nums; margin-top:2px; }
  .range span { color:var(--muted); }
  .bar { position:relative; height:10px; border-radius:5px; background:var(--track); margin:26px 9px 8px; }
  .band { position:absolute; inset:0; background:var(--band); border-radius:5px; }
  .dot { position:absolute; left:0; top:50%; width:18px; height:18px; margin:-9px 0 0 -9px; border-radius:50%;
         background:var(--accent); box-shadow:0 2px 6px rgba(0,0,0,.28); }
  .anim .band { transition:clip-path .6s var(--ease); }
  .anim .dot  { transition:transform .6s var(--ease); }
  .ticks { display:flex; justify-content:space-between; margin:0 9px; font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }
  .note { font-size:14px; color:var(--muted); margin-top:10px; }
</style>
<div id="wrap" role="img">
  <div class="label" id="label"></div>
  <div class="price" id="price"></div>
  <div class="range"><span>Likely range</span> <b id="low"></b> <span>to</span> <b id="high"></b></div>
  <div class="bar" id="bar"><div class="band" id="band"></div><div class="dot" id="dot"></div></div>
  <div class="ticks"><span>$0</span><span>$15k</span><span>$30k</span><span>$45k</span><span>$60k+</span></div>
  <div class="note" id="note"></div>
</div>
<script>
const D = __DATA__;
const DOMAIN = 60000;
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (id) => document.getElementById(id);
const fmt = (v) => "$" + Math.round(v).toLocaleString("en-US");
const frac = (v) => Math.min(Math.max(v / DOMAIN, 0), 1);

// follow the Streamlit theme (light or dark) and font
try {
  const app = parent.document.querySelector(".stApp");
  const cs = parent.getComputedStyle(app);
  const m = cs.backgroundColor.match(/\\d+/g).map(Number);
  const lum = (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255;
  document.documentElement.classList.toggle("dark", lum < 0.45);
  document.body.style.fontFamily = cs.fontFamily;
  parent.document.fonts.forEach((f) => { try { document.fonts.add(f); } catch (e) {} });
} catch (e) {
  document.documentElement.classList.toggle("dark", matchMedia("(prefers-color-scheme: dark)").matches);
}

$("label").textContent = D.label;
$("note").textContent = D.note;
$("wrap").setAttribute("aria-label",
  D.label + " " + fmt(D.est) + ", likely range " + fmt(D.low) + " to " + fmt(D.high) + ". " + D.note);

function place(v) {
  const W = $("bar").clientWidth;
  $("dot").style.transform = "translateX(" + frac(v.est) * W + "px)";
  $("band").style.clipPath = "inset(0 " + (1 - frac(v.high)) * 100 + "% 0 " + frac(v.low) * 100 + "% round 5px)";
}
function show(v) { $("price").textContent = fmt(v.est); $("low").textContent = fmt(v.low); $("high").textContent = fmt(v.high); }

show(D.prev); place(D.prev);                 // start where the last estimate was
const moved = ["est", "low", "high"].some((k) => Math.abs(D.cur[k] - D.prev[k]) > 0.5);
if (!moved || reduce) {
  show(D.cur); place(D.cur);
  document.body.classList.add("anim");
} else {
  void document.body.offsetWidth;            // commit the starting position, then glide
  document.body.classList.add("anim");
  place(D.cur);
  const t0 = performance.now(), DUR = 600;
  const ease = (t) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));
  (function tick(now) {
    const t = ease(Math.min((now - t0) / DUR, 1));
    show({ est: D.prev.est + (D.cur.est - D.prev.est) * t,
           low: D.prev.low + (D.cur.low - D.prev.low) * t,
           high: D.prev.high + (D.cur.high - D.prev.high) * t });
    if (now - t0 < DUR) requestAnimationFrame(tick); else show(D.cur);
  })(t0);
}
</script>
"""

def price_hero(label, note, cur, prev):
    payload = {"label": label, "note": note, "est": cur["est"], "low": cur["low"], "high": cur["high"],
               "cur": cur, "prev": prev}
    html = HERO_HTML.replace("__DATA__", json.dumps(payload)).strip()
    if hasattr(st, "iframe"):          # current Streamlit
        st.iframe(html, height=235)
    else:                              # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=235)

def show(fig, **kw):
    """Plotly chart that morphs between states instead of jumping."""
    fig.update_layout(transition={"duration": 400, "easing": "cubic-in-out"})
    st.plotly_chart(fig, **kw)

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

    cur = {"est": est, "low": float(low[0]), "high": float(high[0])}
    last = st.session_state.get("hero_last")
    if last is None:
        st.session_state["hero_last"] = cur
        st.session_state["hero_prev"] = cur
    elif last != cur:
        st.session_state["hero_prev"] = last
        st.session_state["hero_last"] = cur
    brand_median = S.loc[S["brand"] == brand, "price"].median()
    if np.isnan(brand_median):
        note = ""
    else:
        gap = est - brand_median
        note = f"{money(abs(gap))} {'above' if gap >= 0 else 'below'} the median {brand} in the data ({money(brand_median)})"
    if model_name != "Random Forest":
        note = ("Range comes from Random Forest. " + note).strip()
    price_hero(f"Estimated price ({model_name})", note, cur, st.session_state["hero_prev"])

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
    show(figc, width="stretch")
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
        show(fa, width="stretch")
    with right:
        fm = go.Figure()
        for mn in ["Random Forest", "Linear Regression"]:
            fm.add_trace(go.Scatter(x=mil, y=predict(mn, mdf), name=mn, mode="lines"))
        fm.add_vline(x=mileage, line_dash="dot", annotation_text="your car")
        fm.update_layout(title="Price vs mileage", xaxis_title="Mileage (miles)",
                         yaxis_title="Price ($)", height=400, legend=dict(orientation="h"))
        show(fm, width="stretch")

    st.subheader("Same car, different engines")
    eng_df = pd.concat([row(brand, e, year, mileage) for e in ENGINES])
    eng_res = pd.DataFrame({"Engine": ENGINES, "Price": predict(model_name, eng_df)})
    eng_res = eng_res.sort_values("Price", ascending=False).head(10)
    fe = px.bar(eng_res, x="Price", y="Engine", orientation="h")
    fe.update_layout(height=380, yaxis={"categoryorder": "total ascending"})
    show(fe, width="stretch")

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
            show(px.scatter(f, x="mileage", y="price", color="brand", opacity=0.5,
                                       title="Price vs mileage"), width="stretch")
        with e2:
            show(px.box(f, x="brand", y="price", color="brand",
                                   title="Price distribution by brand"), width="stretch")
        avg_year = f.groupby("year")["price"].mean().reset_index()
        show(px.line(avg_year, x="year", y="price", markers=True,
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
        show(px.bar(long, x="Metric", y="Error ($)", color="Model", barmode="group"),
                        width="stretch")

    st.subheader("Actual vs predicted price")
    pick = st.radio("Show model", ["Random Forest", "Linear Regression"], horizontal=True, key="avp")
    tp = D["test_points"]
    fav = px.scatter(tp, x="Actual", y=pick, opacity=0.4)
    top_v = float(tp["Actual"].max())
    fav.add_trace(go.Scatter(x=[0, top_v], y=[0, top_v], mode="lines",
                             line=dict(dash="dash", color="red"), name="Perfect prediction"))
    fav.update_layout(height=450, xaxis_title="Actual price ($)", yaxis_title="Predicted price ($)")
    show(fav, width="stretch")

    st.subheader("What matters most for the price?")
    imp = pd.DataFrame({"Feature": list(D["importance"].keys()),
                        "Importance": list(D["importance"].values())}).sort_values("Importance")
    show(px.bar(imp, x="Importance", y="Feature", orientation="h"), width="stretch")
    st.caption("Source: Random Forest feature importance, with brand and engine categories added together.")
