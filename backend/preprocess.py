"""
TrafficGuard Bengaluru — Data Preprocessing
=============================================
Reads the RAW Bengaluru Traffic Police illegal-parking violation export
(jan_to_may_police_violation_anonymized.csv — 298,450 real enforcement
records, Nov 2023–Apr 2024) and turns it into:

  1. data/violations.parquet   -> cleaned, typed, ready-to-query dataset
  2. data/static_cache.json    -> pre-computed aggregates for instant first paint

Run once after placing the raw CSV at backend/data/raw_violations.csv:
    python preprocess.py
"""

import ast
import json
import os
import re

import numpy as np
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(__file__), "data", "raw_violations.csv")
PARQUET_PATH = os.path.join(os.path.dirname(__file__), "data", "violations.parquet")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "static_cache.json")

# Bengaluru bounding box — anything outside this is bad GPS and is dropped
LAT_MIN, LAT_MAX = 12.70, 13.20
LON_MIN, LON_MAX = 77.30, 77.85

# Relative road-space footprint per vehicle class (used for the Congestion
# Impact Score). Bigger footprint => one parked vehicle blocks proportionally
# more of the carriageway. Calibrated from typical Indian Road Congress (IRC)
# passenger-car-equivalent (PCE) guidance, rounded for explainability.
VEHICLE_WEIGHT = {
    "SCOOTER": 1.0, "MOTOR CYCLE": 1.0, "MOPED": 1.0,
    "CAR": 2.0, "JEEP": 2.0, "VAN": 2.2, "PASSENGER AUTO": 1.6,
    "GOODS AUTO": 2.4, "TEMPO": 2.8, "MAXI-CAB": 2.6,
    "LGV": 3.2, "MINI LORRY": 3.6, "TRACTOR": 3.6,
    "HGV": 4.5, "LORRY/GOODS VEHICLE": 4.5, "TANKER": 4.8,
    "BUS (BMTC/KSRTC)": 5.0, "PRIVATE BUS": 5.0, "TOURIST BUS": 5.0,
    "SCHOOL VEHICLE": 4.0, "FACTORY BUS": 5.0, "OTHERS": 2.0,
}
DEFAULT_WEIGHT = 2.0


def parse_violation_list(raw):
    if pd.isna(raw):
        return []
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, list):
            return [str(v).strip() for v in val]
        return [str(val)]
    except Exception:
        # fall back: strip brackets/quotes and split on comma
        cleaned = re.sub(r'[\[\]"]', "", str(raw))
        return [v.strip() for v in cleaned.split(",") if v.strip()]


def load_and_clean():
    print(f"Reading raw CSV: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"  raw rows: {len(df):,}")

    # ---- datetime ----
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_datetime"])
    df["created_datetime"] = df["created_datetime"].dt.tz_convert("Asia/Kolkata")
    df["date"] = df["created_datetime"].dt.date.astype(str)
    df["hour"] = df["created_datetime"].dt.hour
    df["dow"] = df["created_datetime"].dt.dayofweek  # 0=Mon
    df["month_label"] = df["created_datetime"].dt.strftime("%b %Y")
    df["month_sort"] = df["created_datetime"].dt.strftime("%Y-%m")

    # ---- geography ----
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[
        (df["latitude"].between(LAT_MIN, LAT_MAX))
        & (df["longitude"].between(LON_MIN, LON_MAX))
    ]
    # cluster nearby points into a ~110m hotspot cell
    df["cell_lat"] = (df["latitude"] * 1000).round().astype(int) / 1000.0
    df["cell_lon"] = (df["longitude"] * 1000).round().astype(int) / 1000.0
    df["cell_id"] = df["cell_lat"].astype(str) + "_" + df["cell_lon"].astype(str)

    # ---- vehicle type ----
    df["vehicle_type"] = df["vehicle_type"].fillna("OTHERS").str.upper().str.strip()
    df["impact_weight"] = df["vehicle_type"].map(VEHICLE_WEIGHT).fillna(DEFAULT_WEIGHT)

    # ---- violation types (multi-label) ----
    df["violation_list"] = df["violation_type"].apply(parse_violation_list)

    # ---- police station / junction cleanup ----
    df["police_station"] = df["police_station"].fillna("Unknown").str.strip()
    df["junction_name"] = df["junction_name"].fillna("No Junction").str.strip()

    keep_cols = [
        "id", "latitude", "longitude", "cell_id", "cell_lat", "cell_lon",
        "location", "vehicle_number", "vehicle_type", "impact_weight",
        "violation_list", "created_datetime", "date", "hour", "dow",
        "month_label", "month_sort", "police_station", "junction_name",
        "validation_status",
    ]
    df = df[keep_cols].reset_index(drop=True)
    print(f"  clean rows: {len(df):,}")
    return df


def build_cell_lookup(df):
    """Most frequent human-readable location string + dominant junction per cell."""
    lookup = {}
    for cell_id, g in df.groupby("cell_id"):
        loc = g["location"].mode()
        junc = g["junction_name"][g["junction_name"] != "No Junction"].mode()
        lookup[cell_id] = {
            "lat": float(g["cell_lat"].iloc[0]),
            "lon": float(g["cell_lon"].iloc[0]),
            "location": loc.iloc[0] if len(loc) else g["location"].iloc[0],
            "junction": junc.iloc[0] if len(junc) else "No Junction",
            "police_station": g["police_station"].mode().iloc[0],
        }
    return lookup


def compute_hotspots(df, lookup, top_n=60):
    grp = df.groupby("cell_id").agg(
        violations=("id", "count"),
        impact=("impact_weight", "sum"),
        repeat_vehicles=("vehicle_number", lambda s: int(s.duplicated().sum())),
    ).reset_index()

    max_impact = grp["impact"].max() or 1
    grp["congestion_score"] = (grp["impact"] / max_impact * 100).round(1)

    grp = grp.sort_values("congestion_score", ascending=False).head(top_n)

    hotspots = []
    for _, row in grp.iterrows():
        info = lookup.get(row["cell_id"], {})
        score = row["congestion_score"]
        if score >= 70:
            severity, action = "critical", "Dispatch enforcement + tow van immediately"
        elif score >= 40:
            severity, action = "high", "Increase patrol frequency; deploy signage"
        elif score >= 18:
            severity, action = "moderate", "Schedule periodic patrol checks"
        else:
            severity, action = "low", "Monitor via CCTV; issue warning on detection"

        sub = df[df["cell_id"] == row["cell_id"]]
        top_violations = (
            sub["violation_list"].explode().value_counts().head(3).index.tolist()
        )
        top_vehicle = sub["vehicle_type"].value_counts().idxmax()
        peak_hour = int(sub["hour"].value_counts().idxmax())

        hotspots.append({
            "cell_id": row["cell_id"],
            "lat": info.get("lat"),
            "lon": info.get("lon"),
            "location": info.get("location", "Unknown"),
            "junction": info.get("junction", "No Junction"),
            "police_station": info.get("police_station", "Unknown"),
            "violations": int(row["violations"]),
            "repeat_vehicles": int(row["repeat_vehicles"]),
            "congestion_score": float(score),
            "severity": severity,
            "recommended_action": action,
            "top_violations": top_violations,
            "top_vehicle_type": top_vehicle,
            "peak_hour": peak_hour,
        })
    return hotspots


def compute_overview(df):
    total = len(df)
    return {
        "total_violations": int(total),
        "date_start": df["created_datetime"].min().strftime("%d %b %Y"),
        "date_end": df["created_datetime"].max().strftime("%d %b %Y"),
        "unique_locations": int(df["cell_id"].nunique()),
        "unique_police_stations": int(df["police_station"].nunique()),
        "unique_vehicles_flagged": int(df["vehicle_number"].nunique()),
        "repeat_offender_vehicles": int(
            df["vehicle_number"].value_counts().gt(1).sum()
        ),
        "top_vehicle_type": df["vehicle_type"].value_counts().idxmax(),
        "top_violation_type": df["violation_list"].explode().value_counts().idxmax(),
        "busiest_police_station": df["police_station"].value_counts().idxmax(),
    }


def compute_trends(df):
    monthly = (
        df.groupby(["month_sort", "month_label"]).size()
        .reset_index(name="count")
        .sort_values("month_sort")
    )
    vehicle = df["vehicle_type"].value_counts().head(10).reset_index()
    vehicle.columns = ["vehicle_type", "count"]
    violation = df["violation_list"].explode().value_counts().head(10).reset_index()
    violation.columns = ["violation_type", "count"]
    station = df["police_station"].value_counts().head(10).reset_index()
    station.columns = ["police_station", "count"]

    hour_dow = (
        df.groupby(["dow", "hour"]).size().reset_index(name="count")
    )
    hour_dow_matrix = [[0] * 24 for _ in range(7)]
    for _, r in hour_dow.iterrows():
        hour_dow_matrix[int(r["dow"])][int(r["hour"])] = int(r["count"])

    return {
        "monthly": monthly[["month_label", "count"]].to_dict("records"),
        "by_vehicle_type": vehicle.to_dict("records"),
        "by_violation_type": violation.to_dict("records"),
        "by_police_station": station.to_dict("records"),
        "hour_dow_matrix": hour_dow_matrix,  # rows = Mon..Sun, cols = 0..23
    }


def compute_heatpoints(df, max_points=15000):
    sample = df if len(df) <= max_points else df.sample(max_points, random_state=42)
    pts = sample.groupby(["cell_lat", "cell_lon"]).size().reset_index(name="w")
    return [[float(r.cell_lat), float(r.cell_lon), int(r.w)] for r in pts.itertuples()]


def main():
    df = load_and_clean()
    lookup = build_cell_lookup(df)

    print("Writing parquet …")
    df.drop(columns=["violation_list"]).assign(
        violation_list=df["violation_list"].apply(json.dumps)
    ).to_parquet(PARQUET_PATH, index=False)

    print("Computing aggregates …")
    cache = {
        "overview": compute_overview(df),
        "hotspots": compute_hotspots(df, lookup),
        "trends": compute_trends(df),
        "heatpoints": compute_heatpoints(df),
        "cell_lookup": lookup,
    }
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

    print(f"Done. Parquet: {os.path.getsize(PARQUET_PATH)/1e6:.1f} MB, "
          f"Cache: {os.path.getsize(CACHE_PATH)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
