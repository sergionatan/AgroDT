from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GenerationConfig:
    n_days: int = 730
    seed: int = 42
    start_date: str = "2024-01-01"


def _seasonal_wave(day_index: np.ndarray, period: int = 365) -> np.ndarray:
    return np.sin(2 * np.pi * day_index / period)


def generate_mock_data(config: GenerationConfig = GenerationConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    date_range = pd.date_range(config.start_date, periods=config.n_days, freq="D")
    day_idx = np.arange(config.n_days)
    seasonal = _seasonal_wave(day_idx)

    temperature_c = 22 + 8 * seasonal + rng.normal(0, 1.6, size=config.n_days)
    # Injeta ondas de calor para refletir extremos climáticos no contexto mediterrâneo.
    heatwave_days = rng.random(config.n_days) < 0.06
    temperature_c += heatwave_days * rng.uniform(4.0, 9.5, size=config.n_days)
    humidity_pct = 60 - 12 * seasonal + rng.normal(0, 4.5, size=config.n_days)
    rainfall_mm = np.clip(3.5 - 1.8 * seasonal + rng.gamma(shape=2, scale=1.1, size=config.n_days), 0, None)

    soil_moisture_pct = np.clip(
        26 + 0.9 * rainfall_mm + 0.12 * humidity_pct - 0.42 * temperature_c + rng.normal(0, 2.2, size=config.n_days),
        8,
        58,
    )
    soil_ph = np.clip(6.7 + rng.normal(0, 0.22, size=config.n_days), 5.8, 8.1)

    nitrogen_mgkg = np.clip(34 + 7 * seasonal + rng.normal(0, 5.2, size=config.n_days), 12, 62)
    phosphorus_mgkg = np.clip(21 + 4.2 * seasonal + rng.normal(0, 2.9, size=config.n_days), 7, 42)
    potassium_mgkg = np.clip(195 + 23 * seasonal + rng.normal(0, 12, size=config.n_days), 95, 330)

    nutrient_balance = (
        0.022 * nitrogen_mgkg
        + 0.018 * phosphorus_mgkg
        + 0.009 * potassium_mgkg
        - 0.95 * np.abs(soil_ph - 6.8)
    )
    heat_stress = 0.12 * np.maximum(temperature_c - 31, 0) + 0.18 * np.maximum(temperature_c - 36, 0)
    drought_stress = 0.06 * np.maximum(24 - soil_moisture_pct, 0)
    climate_stress = heat_stress + drought_stress

    lai = np.clip(2.0 + 0.14 * nutrient_balance + 0.05 * soil_moisture_pct - climate_stress + rng.normal(0, 0.25, size=config.n_days), 0.9, 6.8)
    fruit_maturation_rate = np.clip(
        38 + 1.5 * temperature_c + 2.4 * lai - 0.8 * rainfall_mm + rng.normal(0, 3.1, size=config.n_days),
        15,
        100,
    )

    productivity_ton_ha = np.clip(
        0.55
        + 0.015 * nitrogen_mgkg
        + 0.012 * phosphorus_mgkg
        + 0.004 * potassium_mgkg
        + 0.022 * soil_moisture_pct
        + 0.16 * lai
        - 0.28 * np.abs(soil_ph - 6.8)
        - 0.09 * np.maximum(temperature_c - 32, 0)
        - 0.20 * np.maximum(temperature_c - 37, 0)
        - 0.18 * np.maximum(18 - soil_moisture_pct, 0)
        + rng.normal(0, 0.36, size=config.n_days),
        0.0,
        8.8,
    )

    planted_area_ha = 55
    climate_shock = (temperature_c > 40) & (soil_moisture_pct < 16)
    disease_shock = rng.random(config.n_days) < 0.03
    shock_factor = np.where(climate_shock | disease_shock, rng.uniform(0.0, 0.45, size=config.n_days), 1.0)
    harvest_volume_ton = np.clip(
        productivity_ton_ha * planted_area_ha * shock_factor * rng.normal(1.0, 0.06, size=config.n_days),
        0.0,
        None,
    )

    oil_price_eur_l = np.clip(
        6.6 + 0.15 * np.cos(2 * np.pi * day_idx / 180) + 0.09 * (2.4 - productivity_ton_ha) + rng.normal(0, 0.2, size=config.n_days),
        4.0,
        10.0,
    )
    liters_per_ton = 210
    operational_cost_eur = 9000 + 120 * planted_area_ha + np.maximum(0, temperature_c - 35) * 150 * planted_area_ha / 55
    revenue_eur = harvest_volume_ton * liters_per_ton * oil_price_eur_l - operational_cost_eur

    return pd.DataFrame(
        {
            "date": date_range,
            "temperature_c": temperature_c.round(2),
            "humidity_pct": humidity_pct.round(2),
            "rainfall_mm": rainfall_mm.round(2),
            "soil_ph": soil_ph.round(2),
            "nitrogen_mgkg": nitrogen_mgkg.round(2),
            "phosphorus_mgkg": phosphorus_mgkg.round(2),
            "potassium_mgkg": potassium_mgkg.round(2),
            "soil_moisture_pct": soil_moisture_pct.round(2),
            "lai": lai.round(2),
            "fruit_maturation_rate": fruit_maturation_rate.round(2),
            "productivity_ton_ha": productivity_ton_ha.round(3),
            "harvest_volume_ton": harvest_volume_ton.round(2),
            "oil_price_eur_l": oil_price_eur_l.round(3),
            "revenue_eur": revenue_eur.round(2),
        }
    )


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    output_path = Path("data/raw/olive_digital_twin.csv")
    df = generate_mock_data()
    save_dataset(df, output_path)
    print(f"Dataset sintético gerado em: {output_path.resolve()}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
