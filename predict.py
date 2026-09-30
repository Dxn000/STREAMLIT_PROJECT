import os
import argparse
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_solar_model.pkl")
BASELINE_PATH = os.path.join(BASE_DIR, "models", "linear_baseline_model.pkl")


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python train.py` first to generate it."
        )
    return joblib.load(path)


def classify_generation_status(ac_power: float, irradiation: float) -> dict:
    """Classifies plant operational status and calculates clean energy equivalents."""
    if irradiation <= 0.001 or ac_power < 1.0:
        status = "Nighttime / Standby Mode"
        badge = "[IDLE]"
        desc = "Zero or near-zero solar irradiance detected. Inverter in standby idle state."
    elif irradiation < 0.25:
        status = "Low Solar Yield (Dawn/Dusk/Cloudy)"
        badge = "[LOW YIELD]"
        desc = "Low solar irradiance. Sub-optimal power output under diffused sunlight."
    elif irradiation < 0.65:
        status = "Moderate Generation (Standard Sun)"
        badge = "[OPERATIONAL]"
        desc = "Healthy solar harvest under moderate sunlight conditions."
    else:
        status = "Peak Generation (Optimal Sunlight)"
        badge = "[PEAK EFFICIENCY]"
        desc = "High irradiance operating regime. Photovoltaic cells delivering maximum grid-feed power."

    # Environmental impact calculations
    # Average household uses approx. 1.2 kW average continuous power
    homes_powered = round(ac_power / 1.2, 1) if ac_power > 0 else 0
    # Average grid emission factor: ~0.82 kg CO2 per kWh avoided
    co2_saved_kg_per_hour = round(ac_power * 0.82, 1) if ac_power > 0 else 0

    return {
        "status": status,
        "badge": badge,
        "desc": desc,
        "homes_powered": homes_powered,
        "co2_saved_kg": co2_saved_kg_per_hour,
    }


def predict_single(ambient_temp: float, module_temp: float, irradiation: float, daily_yield: float, hour: int):
    best_model = load_model(MODEL_PATH)
    baseline_model = load_model(BASELINE_PATH)

    temp_diff = round(module_temp - ambient_temp, 2)

    input_df = pd.DataFrame([{
        "ambient_temperature": float(ambient_temp),
        "module_temperature": float(module_temp),
        "temp_diff": float(temp_diff),
        "irradiation": float(irradiation),
        "daily_yield": float(daily_yield),
        "hour": int(hour),
    }])

    pred_ac = max(0.0, float(best_model.predict(input_df)[0]))
    base_ac = max(0.0, float(baseline_model.predict(input_df)[0]))

    metrics = classify_generation_status(pred_ac, irradiation)

    print("\n" + "=" * 65)
    print("     SOLAR PHOTOVOLTAIC POWER GENERATION FORECAST REPORT       ")
    print("=" * 65)
    print("Environmental & Sensor Telemetry:")
    print(f"  Solar Irradiation     : {irradiation:.4f} kW/m²")
    print(f"  Ambient Temperature   : {ambient_temp:.1f} °C")
    print(f"  PV Module Temperature : {module_temp:.1f} °C (Thermal Gradient: {temp_diff:+.1f} °C)")
    print(f"  Time of Day           : Hour {hour:02d}:00")
    print(f"  Accumulated Day Yield : {daily_yield:,.1f} kWh")
    print("-" * 65)
    print(f"Operational State: {metrics['badge']} {metrics['status']}")
    print(f"Diagnosis: {metrics['desc']}")
    print("-" * 65)
    print(f"Predicted AC Grid Feed Power (Champion GBDT): {pred_ac:,.2f} kW")
    print(f"Baseline Linear Model Estimate             : {base_ac:,.2f} kW")
    print(f"Algorithm Prediction Delta                 : {pred_ac - base_ac:+,.2f} kW")
    print("-" * 65)
    print("Clean Energy Equivalents:")
    print(f"  * Homes Powered Simultaneously  : ~{metrics['homes_powered']:,.1f} households")
    print(f"  * CO2 Emissions Avoided / Hour  : ~{metrics['co2_saved_kg']:,.1f} kg CO2/hr")
    print("=" * 65 + "\n")
    return pred_ac


def predict_batch(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    model = load_model(MODEL_PATH)
    df = pd.read_csv(file_path)

    required = ["ambient_temperature", "module_temperature", "irradiation", "daily_yield", "hour"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {file_path}")

    if "temp_diff" not in df.columns:
        df["temp_diff"] = df["module_temperature"] - df["ambient_temperature"]

    features = ["ambient_temperature", "module_temperature", "temp_diff", "irradiation", "daily_yield", "hour"]
    preds = np.clip(model.predict(df[features]), 0, None)
    df["predicted_ac_power_kw"] = np.round(preds, 2)

    output_path = os.path.splitext(file_path)[0] + "_forecasts.csv"
    df.to_csv(output_path, index=False)
    print(f"Batch predictions saved to: {output_path} ({len(df)} records)")


def interactive_prompt():
    print("\n--- Enter Solar Farm Telemetry Readings ---")
    irradiation = float(input("Solar Irradiation (kW/m², e.g. 0.75 for strong sun, 0 for night) [0.75]: ") or 0.75)
    ambient_temp = float(input("Ambient Air Temperature (°C) [30.0]: ") or 30.0)
    module_temp = float(input("PV Panel Module Temperature (°C) [48.0]: ") or 48.0)
    hour = int(input("Hour of Day (0-23) [13]: ") or 13)
    daily_yield = float(input("Accumulated Daily Yield (kWh) [3500.0]: ") or 3500.0)

    predict_single(ambient_temp, module_temp, irradiation, daily_yield, hour)


def main():
    parser = argparse.ArgumentParser(description="Forecast Solar Power Generation Output")
    parser.add_argument("--ambient_temp", type=float, help="Ambient temperature (°C)")
    parser.add_argument("--module_temp", type=float, help="Module surface temperature (°C)")
    parser.add_argument("--irradiation", type=float, help="Solar irradiation (kW/m^2)")
    parser.add_argument("--daily_yield", type=float, help="Daily yield accumulated so far (kWh)")
    parser.add_argument("--hour", type=int, help="Hour of the day (0-23)")
    parser.add_argument("--file", type=str, help="CSV file for batch telemetry prediction")

    args = parser.parse_args()

    if args.file:
        predict_batch(args.file)
    elif all(v is not None for v in [args.ambient_temp, args.module_temp, args.irradiation, args.daily_yield, args.hour]):
        predict_single(args.ambient_temp, args.module_temp, args.irradiation, args.daily_yield, args.hour)
    else:
        interactive_prompt()


if __name__ == "__main__":
    main()
