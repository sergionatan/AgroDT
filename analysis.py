from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_gen import generate_mock_data, save_dataset


DATA_PATH = Path("data/raw/olive_digital_twin.csv")
EDA_DIR = Path("outputs/eda")
MODEL_DIR = Path("models")

FEATURES = [
    "temperature_c",
    "humidity_pct",
    "rainfall_mm",
    "soil_ph",
    "nitrogen_mgkg",
    "phosphorus_mgkg",
    "potassium_mgkg",
    "soil_moisture_pct",
    "lai",
    "fruit_maturation_rate",
]
TARGET = "productivity_ton_ha"


def ensure_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    df = generate_mock_data()
    save_dataset(df, path)
    return df


def identify_outliers_iqr(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include="number")
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = ((numeric < lower) | (numeric > upper)).sum().sort_values(ascending=False)
    return mask


def run_eda(df: pd.DataFrame) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    summary = df.describe(include="all")
    summary.to_csv(EDA_DIR / "summary_stats.csv")

    corr = df.select_dtypes(include="number").corr(numeric_only=True)
    corr.to_csv(EDA_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="RdYlGn", center=0, annot=False)
    plt.title("Mapa de Correlação - Digital Twin Oliveira")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    climate_cols = ["temperature_c", "humidity_pct", "rainfall_mm", "soil_moisture_pct"]
    plt.figure(figsize=(12, 7))
    for col in climate_cols:
        sns.kdeplot(df[col], fill=True, label=col)
    plt.title("Distribuições Climáticas e Umidade do Solo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_DIR / "climate_distributions.png", dpi=160)
    plt.close()

    outlier_counts = identify_outliers_iqr(df)
    outlier_counts.to_csv(EDA_DIR / "outlier_counts_iqr.csv", header=["outlier_count"])

    # Insights automáticos priorizando relações críticas para saúde da plantação.
    top_corr_pairs = (
        corr.unstack()
        .reset_index(name="corr")
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    top_corr_pairs = top_corr_pairs[top_corr_pairs["feature_a"] < top_corr_pairs["feature_b"]]
    top_corr_pairs["abs_corr"] = top_corr_pairs["corr"].abs()
    top_corr_pairs = top_corr_pairs.sort_values("abs_corr", ascending=False).head(8)
    top_corr_pairs.to_csv(EDA_DIR / "top_correlations.csv", index=False)

    print("=== Insights EDA ===")
    print("Top correlações absolutas:")
    print(top_corr_pairs[["feature_a", "feature_b", "corr"]].to_string(index=False))
    print("\nOutliers por variável (IQR):")
    print(outlier_counts.to_string())
    print(f"\nArtefatos EDA salvos em: {EDA_DIR.resolve()}")


def train_model(df: pd.DataFrame) -> Pipeline:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), FEATURES)],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=350,
                    max_depth=12,
                    min_samples_split=4,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "r2": round(float(r2_score(y_test, preds)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "olive_productivity_model.joblib")
    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)

    print("\n=== Resultados de Treinamento ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value}")
    print(f"Modelo salvo em: {(MODEL_DIR / 'olive_productivity_model.joblib').resolve()}")
    return model


def main() -> None:
    df = ensure_dataset()
    run_eda(df)
    train_model(df)


if __name__ == "__main__":
    main()
