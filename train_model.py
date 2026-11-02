import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_YEAR = 2021  # this Craigslist dataset was scraped in 2021

cols = ["price", "year", "manufacturer", "cylinders", "fuel", "odometer"]
df = pd.read_csv("vehicles.csv", usecols=cols)

# ---- Cleaning ----
df = df.dropna(subset=["price", "year", "manufacturer", "odometer", "fuel"])
df["cylinders"] = df["cylinders"].fillna("unknown")
df = df[(df["price"] >= 500) & (df["price"] <= 100000)]
df = df[(df["odometer"] > 0) & (df["odometer"] <= 400000)]
df = df[(df["year"] >= 1990) & (df["year"] <= DATA_YEAR)]
df["year"] = df["year"].astype(int)
df["age"] = DATA_YEAR - df["year"]
df["engine_type"] = df["fuel"] + " | " + df["cylinders"]   # engine type = fuel + cylinders
df = df.rename(columns={"manufacturer": "brand", "odometer": "mileage"})
print("Rows after cleaning:", len(df))

X = df[["brand", "mileage", "engine_type", "age"]]
y = df["price"]

def make_pre():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["brand", "engine_type"]),
        ("num", StandardScaler(), ["mileage", "age"]),
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=18, min_samples_leaf=5,
                                           random_state=42, n_jobs=-1),
}

results, fitted, preds = {}, {}, {}
for name, m in models.items():
    pipe = Pipeline([("pre", make_pre()), ("model", m)])
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, p)
    rmse = float(np.sqrt(mean_squared_error(y_test, p)))
    results[name] = {"MAE": float(mae), "RMSE": rmse}
    fitted[name] = pipe
    preds[name] = p
    print(f"{name}: MAE = ${mae:,.0f} | RMSE = ${rmse:,.0f}")

# ---- Feature importance (grouped back to the 4 original inputs) ----
rf = fitted["Random Forest"]
ohe = rf.named_steps["pre"].named_transformers_["cat"]
names = list(ohe.get_feature_names_out(["brand", "engine_type"])) + ["mileage", "age"]
imp = rf.named_steps["model"].feature_importances_
grouped = {"Brand": 0.0, "Engine type": 0.0, "Mileage": 0.0, "Age": 0.0}
for n, v in zip(names, imp):
    if n.startswith("brand_"):
        grouped["Brand"] += float(v)
    elif n.startswith("engine_type_"):
        grouped["Engine type"] += float(v)
    elif n == "mileage":
        grouped["Mileage"] += float(v)
    else:
        grouped["Age"] += float(v)

# ---- Test-set points for the actual-vs-predicted chart ----
rng = np.random.RandomState(1)
idx = rng.choice(len(y_test), 1500, replace=False)
test_points = pd.DataFrame({
    "Actual": y_test.values[idx],
    "Random Forest": preds["Random Forest"][idx],
    "Linear Regression": preds["Linear Regression"][idx],
})

# ---- Small sample of the cleaned data for the Explore tab ----
df.sample(8000, random_state=7)[["brand", "year", "mileage", "engine_type", "age", "price"]] \
  .to_csv("sample.csv", index=False)

joblib.dump({
    "models": fitted,
    "brands": sorted(df["brand"].unique().tolist()),
    "engines": sorted(df["engine_type"].unique().tolist()),
    "results": results,
    "importance": grouped,
    "test_points": test_points,
    "data_year": DATA_YEAR,
    "n_rows": int(len(df)),
}, "model.pkl", compress=3)
print("Saved model.pkl and sample.csv")
