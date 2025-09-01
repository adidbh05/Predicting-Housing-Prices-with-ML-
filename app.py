"""
House Price Prediction CLI for London data (synthetic or real).

- Generate synthetic dataset with London-style features.
- Train/evaluate a model with preprocessing.
- Predict on new rows and save outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _flatten_text(x):
    # Accepts 2D array-like with a single column; returns 1D array of strings
    return np.ravel(x)


# ------------------------------ Config types ------------------------------
@dataclass
class TrainConfig:
    data_path: str
    target: str
    model_path: str = "artifacts/model.joblib"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = None
    cv: Optional[int] = None


# ------------------------ Synthetic Data Generation ------------------------
def generate_synthetic(n_rows: int, output_path: str, new_output_path: str, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    boroughs = [
        "Camden","Greenwich","Hackney","Hammersmith and Fulham","Islington","Kensington and Chelsea",
        "Lambeth","Lewisham","Southwark","Tower Hamlets","Wandsworth","Westminster","City of London",
        "Barnet","Bexley","Brent","Bromley","Croydon","Ealing","Enfield","Haringey","Harrow",
        "Havering","Hillingdon","Hounslow","Kingston upon Thames","Merton","Newham","Redbridge",
        "Richmond upon Thames","Sutton","Waltham Forest","Barking and Dagenham",
    ]
    borough_multiplier = {
        "Westminster": 1.6,
        "Kensington and Chelsea": 1.8,
        "Camden": 1.4,
        "City of London": 1.7,
        "Islington": 1.3,
        "Hammersmith and Fulham": 1.35,
        "Wandsworth": 1.25,
        "Richmond upon Thames": 1.3,
        "Southwark": 1.15,
        "Tower Hamlets": 1.2,
        "Lambeth": 1.15,
        "Hackney": 1.15,
        "Greenwich": 1.05,
        "Merton": 1.1,
        "Kingston upon Thames": 1.1,
        "Barnet": 1.05,
        "Hounslow": 0.95,
        "Hillingdon": 0.9,
        "Ealing": 1.0,
        "Haringey": 1.05,
        "Harrow": 1.0,
        "Enfield": 0.95,
        "Redbridge": 0.95,
        "Bromley": 0.95,
        "Croydon": 0.9,
        "Waltham Forest": 0.95,
        "Sutton": 0.9,
        "Havering": 0.85,
        "Bexley": 0.85,
        "Newham": 0.95,
        "Barking and Dagenham": 0.8,
        "Lewisham": 0.95,
        "Brent": 1.0,
    }
    property_types = ["Flat", "Terraced", "Semi-Detached", "Detached", "Maisonette", "Studio"]
    amenity_tokens = [
        "balcony", "garden", "parking", "garage", "gym", "concierge", "lift", "new_build",
        "near_tube", "river_view", "park_view", "study", "ensuite", "loft", "cellar",
    ]

    def sample_amenities(size):
        counts = rng.integers(1, 7, size=size)
        res = []
        for c in counts:
            idx = rng.choice(len(amenity_tokens), size=c, replace=False)
            res.append(", ".join(sorted([amenity_tokens[i] for i in idx])))
        return res

    n = n_rows
    df = pd.DataFrame(
        {
            "location": rng.choice(boroughs, size=n),
            "square_footage": rng.normal(850, 300, size=n).clip(250, 3500).round(0),
            "bedrooms": rng.integers(1, 6, size=n),
            "bathrooms": rng.integers(1, 4, size=n),
            "property_type": rng.choice(property_types, size=n, p=[0.45, 0.2, 0.15, 0.08, 0.07, 0.05]),
            "year_built": rng.integers(1850, 2024, size=n),
            "distance_to_tube_km": rng.gamma(shape=2.0, scale=0.6, size=n).clip(0.05, 6.0),
            "amenities": sample_amenities(n),
        }
    )

    base = 600_000
    loc_mult = df["location"].map(lambda b: borough_multiplier.get(b, 1.0)).values
    sqft = df["square_footage"].values
    beds = df["bedrooms"].values
    baths = df["bathrooms"].values
    year = df["year_built"].values
    dist = df["distance_to_tube_km"].values
    prop = (
        df["property_type"]
        .map({"Flat": -0.05, "Studio": -0.12, "Maisonette": -0.03, "Terraced": 0.0, "Semi-Detached": 0.08, "Detached": 0.2})
        .fillna(0.0)
        .values
    )
    amen_count = df["amenities"].str.count(",").fillna(0).values + 1
    age_term = 2024 - year

    price = (
        base * loc_mult
        + 3000 * (sqft - 800)
        + 40_000 * (beds - 2)
        + 30_000 * (baths - 1)
        + 50_000 * prop
        + 12_000 * (amen_count - 3)
        - 35_000 * np.log1p(dist)
        - 1500 * np.maximum(age_term - 50, 0)
    )
    noise = rng.normal(0, 80_000, size=n)
    df["price"] = (price + noise).clip(120_000, 5_000_000).round(0)
    df.to_csv(output_path, index=False)
    print(f"Wrote training data to {output_path} ({len(df)} rows)")

    m = max(20, n // 200)
    new_df = df.drop(columns=["price"]).sample(m, random_state=random_state).reset_index(drop=True)
    new_df.to_csv(new_output_path, index=False)
    print(f"Wrote new listings to {new_output_path} ({len(new_df)} rows)")


# ------------------------------ Preprocessing ------------------------------
def _infer_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str], List[str]]:
    cols = [c for c in df.columns if c != target]
    text_cols = [c for c in cols if c.lower() == "amenities"]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    # Categorical: strings or pandas Categorical dtype
    categorical_cols = [
        c
        for c in cols
        if (pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)) and c not in text_cols
    ]
    numeric_cols = [c for c in numeric_cols if c not in categorical_cols and c not in text_cols]
    return numeric_cols, categorical_cols, text_cols


def _build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], text_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]),
                categorical_cols,
            )
        )
    if text_cols:
        text_col = text_cols[0]
        transformers.append(
            (
                "txt",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                    ("flatten", FunctionTransformer(_flatten_text, accept_sparse=False)),
                    ("tfidf", TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", min_df=1)),
                ]),
                [text_col],
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


# --------------------------------- Train ----------------------------------
def train(cfg: TrainConfig) -> Optional[dict]:
    df = pd.read_csv(cfg.data_path)
    if cfg.target not in df.columns:
        raise ValueError(f"Target column '{cfg.target}' not found in data.")

    numeric_cols, categorical_cols, text_cols = _infer_feature_types(df, cfg.target)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols, text_cols)
    X = df.drop(columns=[cfg.target])
    y = df[cfg.target].astype(float)

    if cfg.cv:
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, random_state=cfg.random_state, n_jobs=-1)),
        ])
        kf = KFold(n_splits=cfg.cv, shuffle=True, random_state=cfg.random_state)
        neg_mae = cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=kf, n_jobs=-1)
        r2 = cross_val_score(model, X, y, scoring="r2", cv=kf, n_jobs=-1)
        print(f"CV ({cfg.cv}-fold) MAE: {-neg_mae.mean():.3f} ± {neg_mae.std():.3f}")
        print(f"CV ({cfg.cv}-fold) R2 : {r2.mean():.3f} ± {r2.std():.3f}")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, random_state=cfg.random_state, n_jobs=-1)),
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, preds)
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2  : {r2:.3f}")

    os.makedirs(os.path.dirname(cfg.model_path) or ".", exist_ok=True)
    joblib.dump(model, cfg.model_path)
    meta = {
        "target": cfg.target,
        "features": list(X.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "text_cols": text_cols,
        "metrics": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
    }
    meta_path = os.path.join(os.path.dirname(cfg.model_path) or ".", "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model to {cfg.model_path}")
    print(f"Saved metadata to {meta_path}")
    # Save per-row test predictions for comparison
    test_out = X_test.copy()
    test_out[cfg.target] = y_test.values
    test_out["predicted_price"] = preds
    test_pred_path = os.path.join(os.path.dirname(cfg.model_path) or ".", "test_predictions.csv")
    test_out.to_csv(test_pred_path, index=False)
    print(f"Saved test predictions to {test_pred_path}")
    return meta["metrics"]


# -------------------------------- Predict ---------------------------------
def _load_input(path: str, json_orient: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".json":
        return pd.read_json(path, orient=json_orient)
    raise ValueError("Unsupported input format. Use CSV or JSON.")


def predict(model_path: str, input_path: str, json_orient: str, output_path: Optional[str]) -> pd.DataFrame:
    model: Pipeline = joblib.load(model_path)
    X_new = _load_input(input_path, json_orient)
    preds = model.predict(X_new)
    out_df = X_new.copy()
    out_df["predicted_price"] = preds
    if output_path:
        out_df.to_csv(output_path, index=False)
        print(f"Wrote predictions to {output_path}")
    else:
        print(out_df.head(10))
    return out_df


def evaluate(model_path: str, data_path: str, target: str, json_orient: str, output_path: Optional[str]) -> dict:
    model: Pipeline = joblib.load(model_path)
    df = _load_input(data_path, json_orient)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {data_path}")
    X = df.drop(columns=[target])
    y_true = df[target].astype(float)
    preds = model.predict(X)
    mae = mean_absolute_error(y_true, preds)
    mse = mean_squared_error(y_true, preds)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, preds)
    print(f"Evaluate MAE : {mae:.3f}")
    print(f"Evaluate RMSE: {rmse:.3f}")
    print(f"Evaluate R2  : {r2:.3f}")
    out_df = df.copy()
    out_df["predicted_price"] = preds
    if output_path:
        out_df.to_csv(output_path, index=False)
        print(f"Wrote evaluation predictions to {output_path}")
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def build_report(pred_path: str, target: str, group_by: Optional[str], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(pred_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {pred_path}")
    if "predicted_price" not in df.columns:
        raise ValueError("'predicted_price' column not found in predictions file")

    y_true = df[target].astype(float)
    y_pred = df["predicted_price"].astype(float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    sns.set_theme(style="whitegrid")

    # 1) Actual vs Predicted scatter
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=12, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title(f"Actual vs Predicted (RMSE={rmse:,.0f}, MAE={mae:,.0f}, R2={r2:.3f})")
    avp_path = os.path.join(outdir, "scatter_actual_vs_pred.png")
    plt.tight_layout()
    plt.savefig(avp_path, dpi=150)
    plt.close()

    # 2) Residuals histogram
    residuals = y_pred - y_true
    plt.figure(figsize=(7, 4))
    sns.histplot(residuals, bins=40, kde=True)
    plt.axvline(0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("Residual (predicted - actual)")
    plt.title("Residuals distribution")
    resid_hist_path = os.path.join(outdir, "residual_hist.png")
    plt.tight_layout()
    plt.savefig(resid_hist_path, dpi=150)
    plt.close()

    # 3) Residuals vs Predicted
    plt.figure(figsize=(7, 4))
    plt.scatter(y_pred, residuals, s=10, alpha=0.5)
    plt.axhline(0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("Predicted price")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    resid_scatter_path = os.path.join(outdir, "residual_vs_pred.png")
    plt.tight_layout()
    plt.savefig(resid_scatter_path, dpi=150)
    plt.close()

    # 4) Error by group (if available)
    group_png = None
    if group_by and group_by in df.columns:
        g = df.groupby(group_by).apply(lambda x: np.mean(np.abs(x["predicted_price"] - x[target]))).sort_values(ascending=False)
        top = g.head(15)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=top.values, y=top.index, orient="h")
        plt.xlabel("Mean Absolute Error")
        plt.ylabel(group_by)
        plt.title(f"MAE by {group_by} (Top 15)")
        group_png = os.path.join(outdir, "mae_by_group.png")
        plt.tight_layout()
        plt.savefig(group_png, dpi=150)
        plt.close()

    # Save summary JSON
    summary = {"mae": mae, "rmse": rmse, "r2": r2, "rows": int(len(df))}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Minimal HTML report
    html = [
        "<html><head><meta charset='utf-8'><title>Model Report</title></head><body>",
        "<h1>Model Performance Report</h1>",
        f"<p><b>Rows:</b> {len(df)} | <b>MAE:</b> {mae:,.0f} | <b>RMSE:</b> {rmse:,.0f} | <b>R2:</b> {r2:.3f}</p>",
        f"<h2>Actual vs Predicted</h2><img src='scatter_actual_vs_pred.png' width='600'>",
        f"<h2>Residuals Distribution</h2><img src='residual_hist.png' width='600'>",
        f"<h2>Residuals vs Predicted</h2><img src='residual_vs_pred.png' width='600'>",
    ]
    if group_png:
        html.append(f"<h2>MAE by {group_by}</h2><img src='mae_by_group.png' width='700'>")
    html.append("</body></html>")
    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write("\n".join(html))
    print(f"Report written to {outdir}/index.html")


# ---------------------------------- CLI -----------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="London House Price Prediction CLI")
    sub = p.add_subparsers(dest="command", required=True)

    pg = sub.add_parser("generate", help="Generate synthetic London house price data")
    pg.add_argument("--rows", type=int, default=5000, help="Number of rows to generate")
    pg.add_argument("--output", dest="output_path", default="london_house_prices.csv", help="Output CSV for training data")
    pg.add_argument("--new-output", dest="new_output_path", default="new_listings.csv", help="Output CSV for sample new listings")

    pt = sub.add_parser("train", help="Train a model from CSV")
    pt.add_argument("--data", dest="data_path", required=True, help="Path to training CSV")
    pt.add_argument("--target", dest="target", default="price", help="Target column name")
    pt.add_argument("--model", dest="model_path", default="artifacts/model.joblib", help="Output path for model")
    pt.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    pt.add_argument("--random-state", type=int, default=42, help="Random seed")
    pt.add_argument("--n-estimators", type=int, default=300, help="RandomForest n_estimators")
    pt.add_argument("--max-depth", type=int, default=None, help="RandomForest max_depth")
    pt.add_argument("--cv", type=int, default=None, help="If set, run K-fold CV only")

    pp = sub.add_parser("predict", help="Predict using a saved model")
    pp.add_argument("--model", dest="model_path", required=True, help="Path to saved model.joblib")
    pp.add_argument("--input", dest="input_path", required=True, help="CSV or JSON of new rows")
    pp.add_argument("--json-orient", dest="json_orient", default="records", help="Pandas JSON orient for reading")
    pp.add_argument("--output", dest="output_path", default=None, help="Optional CSV to write predictions")

    ev = sub.add_parser("evaluate", help="Predict on labeled data and compute metrics")
    ev.add_argument("--model", dest="model_path", required=True, help="Path to saved model.joblib")
    ev.add_argument("--data", dest="data_path", required=True, help="CSV or JSON with feature columns and target present")
    ev.add_argument("--target", dest="target", default="price", help="Target column name")
    ev.add_argument("--json-orient", dest="json_orient", default="records", help="Pandas JSON orient for reading")
    ev.add_argument("--output", dest="output_path", default="evaluation_predictions.csv", help="CSV to write rows with predictions")

    rp = sub.add_parser("report", help="Create visual report from predictions with ground truth")
    rp.add_argument("--predictions", dest="pred_path", default="artifacts/test_predictions.csv", help="CSV with actual target and predicted_price columns")
    rp.add_argument("--target", dest="target", default="price", help="Actual target column present in predictions CSV")
    rp.add_argument("--group-by", dest="group_by", default="location", help="Optional column to analyze error distribution")
    rp.add_argument("--outdir", dest="outdir", default="reports", help="Directory to save charts and HTML report")

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.command == "generate":
        generate_synthetic(args.rows, args.output_path, args.new_output_path, random_state=42)
    elif args.command == "train":
        cfg = TrainConfig(
            data_path=args.data_path,
            target=args.target,
            model_path=args.model_path,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            cv=args.cv,
        )
        train(cfg)
    elif args.command == "predict":
        predict(args.model_path, args.input_path, args.json_orient, args.output_path)
    elif args.command == "evaluate":
        evaluate(args.model_path, args.data_path, args.target, args.json_orient, args.output_path)
    elif args.command == "report":
        build_report(args.pred_path, args.target, args.group_by, args.outdir)


if __name__ == "__main__":
    main()
