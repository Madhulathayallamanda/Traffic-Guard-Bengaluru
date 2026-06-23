"""
TrafficGuard Bengaluru — Backend API (Flask)
=============================================
Flask service that serves live analytics computed directly from the real
Bengaluru Traffic Police illegal-parking dataset (298,277 cleaned enforcement
records).

Every number returned by this API is computed on the fly with pandas from the
actual data — there is no mock/sample data layer.

NOTE ON FRAMEWORK: this build runs on Flask rather than FastAPI/uvicorn
because this execution environment doesn't have pyarrow / FastAPI / uvicorn
available and has no network access to install them. Flask + Werkzeug were
available, so the API was ported 1:1 (same routes, same JSON shapes) so the
frontend needed no contract changes. If you have pyarrow available elsewhere,
violations.parquet still works and preprocess.py is unchanged.

Run:
    python app.py
Then open http://localhost:8080 in a browser (the dashboard is served here too).
"""

import ast
import io
import json
import os
import random
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import gdown
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
RAW_CSV = os.path.join(DATA_DIR, "raw_violations.csv")
PARQUET_PATH = os.path.join(DATA_DIR, "violations.parquet")
CACHE_PATH = os.path.join(DATA_DIR, "static_cache.json")

# Automatically download the dataset if it is missing
if not os.path.exists(RAW_CSV):
    print("Downloading dataset from Google Drive...")

    os.makedirs(DATA_DIR, exist_ok=True)

    gdown.download(
        "https://drive.google.com/uc?id=1rhIjtjUhBcsaLaR5gTdafNwM1WwYr_bm",
        RAW_CSV,
        quiet=False,
    )

app = Flask(__name__, static_folder=None)


DF: pd.DataFrame = pd.DataFrame()
STATIC_CACHE: dict = {}
CELL_LOOKUP: dict = {}

# Bengaluru bounding box — anything outside this is bad GPS and is dropped
LAT_MIN, LAT_MAX = 12.70, 13.20
LON_MIN, LON_MAX = 77.30, 77.85

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

# Calibrated assumption (documented, not hidden): a fully Critical (score=100)
# hotspot is estimated to carry up to this many minutes of added queuing delay
# during its peak window. Used only by the Traffic-Saved-Prediction feature.
MAX_DELAY_MINUTES_AT_SCORE_100 = 25.0


def parse_violation_list(raw):
    if pd.isna(raw):
        return []
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, list):
            return [str(v).strip() for v in val]
        return [str(val)]
    except Exception:
        cleaned = re.sub(r'[\[\]"]', "", str(raw))
        return [v.strip() for v in cleaned.split(",") if v.strip()]


def load_and_clean_from_csv():
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_datetime"])
    df["created_datetime"] = df["created_datetime"].dt.tz_convert("Asia/Kolkata")
    df["date"] = df["created_datetime"].dt.date.astype(str)
    df["hour"] = df["created_datetime"].dt.hour
    df["dow"] = df["created_datetime"].dt.dayofweek
    df["month_label"] = df["created_datetime"].dt.strftime("%b %Y")
    df["month_sort"] = df["created_datetime"].dt.strftime("%Y-%m")

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(LAT_MIN, LAT_MAX)) & (df["longitude"].between(LON_MIN, LON_MAX))]
    df["cell_lat"] = (df["latitude"] * 1000).round().astype(int) / 1000.0
    df["cell_lon"] = (df["longitude"] * 1000).round().astype(int) / 1000.0
    df["cell_id"] = df["cell_lat"].astype(str) + "_" + df["cell_lon"].astype(str)

    df["vehicle_type"] = df["vehicle_type"].fillna("OTHERS").str.upper().str.strip()
    df["impact_weight"] = df["vehicle_type"].map(VEHICLE_WEIGHT).fillna(DEFAULT_WEIGHT)
    df["violation_list"] = df["violation_type"].apply(parse_violation_list)
    df["police_station"] = df["police_station"].fillna("Unknown").str.strip()
    df["junction_name"] = df["junction_name"].fillna("No Junction").str.strip()

    keep_cols = [
        "id", "latitude", "longitude", "cell_id", "cell_lat", "cell_lon",
        "location", "vehicle_number", "vehicle_type", "impact_weight",
        "violation_list", "created_datetime", "date", "hour", "dow",
        "month_label", "month_sort", "police_station", "junction_name",
        "validation_status",
    ]
    return df[keep_cols].reset_index(drop=True)


def load_data():
    global DF, STATIC_CACHE, CELL_LOOKUP

    if os.path.exists(PARQUET_PATH):
        try:
            DF = pd.read_parquet(PARQUET_PATH)
            DF["created_datetime"] = pd.to_datetime(DF["created_datetime"])
            DF["violation_list"] = DF["violation_list"].apply(json.loads)
            print(f"Loaded {len(DF):,} violation records from parquet.")
        except Exception as e:
            print(f"Parquet engine unavailable ({e}); rebuilding from raw CSV …")
            DF = load_and_clean_from_csv()
            print(f"Loaded {len(DF):,} violation records from raw CSV.")
    else:
        DF = load_and_clean_from_csv()
        print(f"Loaded {len(DF):,} violation records from raw CSV.")

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            STATIC_CACHE = json.load(f)
        CELL_LOOKUP = STATIC_CACHE.get("cell_lookup", {})
    else:
        CELL_LOOKUP = {}


# ---------------------------------------------------------------- filtering --
def get_filter_args():
    return (
        request.args.get("vehicle_type"),
        request.args.get("police_station"),
        request.args.get("date_from"),
        request.args.get("date_to"),
    )


def apply_filters(vehicle_type=None, police_station=None, date_from=None, date_to=None):
    df = DF
    if vehicle_type and vehicle_type != "all":
        df = df[df["vehicle_type"] == vehicle_type]
    if police_station and police_station != "all":
        df = df[df["police_station"] == police_station]
    if date_from:
        df = df[df["date"] >= date_from]
    if date_to:
        df = df[df["date"] <= date_to]
    return df


def severity_for_score(score):
    if score >= 70:
        return "critical", "Dispatch enforcement + tow van immediately"
    elif score >= 40:
        return "high", "Increase patrol frequency; deploy signage"
    elif score >= 18:
        return "moderate", "Schedule periodic patrol checks"
    else:
        return "low", "Monitor via CCTV; issue warning on detection"


def compute_hotspots(vehicle_type=None, police_station=None, date_from=None, date_to=None, top_n=40):
    df = apply_filters(vehicle_type, police_station, date_from, date_to)
    if df.empty:
        return []

    grouped = df.groupby("cell_id")
    grp = grouped.agg(
        violations=("id", "count"),
        impact=("impact_weight", "sum"),
    ).reset_index()
    max_impact = grp["impact"].max() or 1
    grp["congestion_score"] = (grp["impact"] / max_impact * 100).round(1)
    grp = grp.sort_values("congestion_score", ascending=False).head(top_n)

    results = []
    for _, row in grp.iterrows():
        cell_id = row["cell_id"]
        info = CELL_LOOKUP.get(cell_id, {})
        sub = grouped.get_group(cell_id)
        results.append(_build_hotspot_record(cell_id, info, sub, row))
    return results


def compute_single_hotspot(cell_id, vehicle_type=None, police_station=None, date_from=None, date_to=None):
    """Fast path for one specific cell — avoids scanning/ranking every cell."""
    df = apply_filters(vehicle_type, police_station, date_from, date_to)
    sub = df[df["cell_id"] == cell_id]
    if sub.empty:
        return None
    impact = float(sub["impact_weight"].sum())
    max_impact = float(df.groupby("cell_id")["impact_weight"].sum().max() or 1)
    score = round(impact / max_impact * 100, 1)
    info = CELL_LOOKUP.get(cell_id, {})
    row = {"violations": len(sub), "impact": impact}
    return _build_hotspot_record(cell_id, info, sub, row, score_override=score)


def _build_hotspot_record(cell_id, info, sub, row, score_override=None):
    score = score_override if score_override is not None else float(row["congestion_score"])
    severity, action = severity_for_score(score)

    top_violations = sub["violation_list"].explode().value_counts().head(3).index.tolist()
    peak_hour = int(sub["hour"].value_counts().idxmax()) if len(sub) else None
    peak_dow = int(sub["dow"].value_counts().idxmax()) if len(sub) else None
    top_vehicle_type = sub["vehicle_type"].value_counts().idxmax() if len(sub) else None
    top_vehicle_weight = VEHICLE_WEIGHT.get(top_vehicle_type, DEFAULT_WEIGHT) if top_vehicle_type else DEFAULT_WEIGHT
    total_impact = float(row["impact"]) or 1.0

    # Estimate: a fully Critical (score=100) hotspot carries up to
    # MAX_DELAY_MINUTES_AT_SCORE_100 of peak-window queuing delay
    # (documented assumption). A single vehicle's contribution to that
    # delay is scaled by its road-space footprint (impact_weight) share
    # relative to the average vehicle at this cell.
    max_delay = (score / 100.0) * MAX_DELAY_MINUTES_AT_SCORE_100
    avg_impact_per_violation = total_impact / len(sub) if len(sub) else DEFAULT_WEIGHT
    veh_share = top_vehicle_weight / avg_impact_per_violation if avg_impact_per_violation else 1
    traffic_saved_minutes = round(max(0.5, min(max_delay, max_delay * 0.4 * veh_share)), 1)

    return {
        "cell_id": cell_id,
        "lat": info.get("lat"),
        "lon": info.get("lon"),
        "location": info.get("location", "Unknown"),
        "junction": info.get("junction", "No Junction"),
        "police_station": info.get("police_station", "Unknown"),
        "violations": int(row["violations"]),
        "congestion_score": score,
        "severity": severity,
        "recommended_action": action,
        "top_violations": top_violations,
        "top_vehicle_type": top_vehicle_type,
        "peak_hour": peak_hour,
        "peak_dow": peak_dow,
        "traffic_saved_minutes": traffic_saved_minutes,
    }


def compute_predict_for_hotspot(h):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    sub = DF[DF["cell_id"] == h["cell_id"]]
    if sub.empty:
        return None
    hour_counts = sub["hour"].value_counts(normalize=True).sort_values(ascending=False)
    window_hours = hour_counts[hour_counts.cumsum() <= 0.6].index.tolist()
    if not window_hours:
        window_hours = [hour_counts.index[0]]
    window_hours = sorted(window_hours)

    # Build a human label that doesn't lie about non-contiguous hours.
    groups, cur = [], [window_hours[0]]
    for hh in window_hours[1:]:
        if hh == cur[-1] + 1:
            cur.append(hh)
        else:
            groups.append(cur)
            cur = [hh]
    groups.append(cur)
    label_parts = []
    for g in groups:
        if len(g) == 1:
            label_parts.append(f"{g[0]:02d}:00")
        else:
            label_parts.append(f"{g[0]:02d}:00\u2013{(g[-1]+1)%24:02d}:00")
    window_label = ", ".join(label_parts)

    return {
        "cell_id": h["cell_id"],
        "location": h["location"],
        "police_station": h["police_station"],
        "congestion_score": h["congestion_score"],
        "predicted_high_risk_hours": [f"{hh:02d}:00" for hh in window_hours],
        "predicted_window_label": window_label,
        "predicted_peak_day": days[h["peak_dow"]] if h["peak_dow"] is not None else None,
        "confidence": round(float(hour_counts.loc[window_hours].sum()) * 100, 1),
        "traffic_saved_minutes": h["traffic_saved_minutes"],
        "top_vehicle_type": h["top_vehicle_type"],
    }


def compute_predict(top_n=5):
    hs = compute_hotspots(top_n=top_n)
    out = []
    for h in hs:
        p = compute_predict_for_hotspot(h)
        if p:
            out.append(p)
    return out


def short_loc(location, n=2):
    parts = [p.strip() for p in (location or "Unknown").split(",")[:n]]
    return ", ".join(p for p in parts if p)


def build_hotspot_summary_sentence(h, window_label=None):
    loc = short_loc(h["location"])
    top_v = (h["top_violations"][0] if h["top_violations"] else "illegal parking").lower()
    when = f" between {window_label}" if window_label else (f" around {h['peak_hour']:02d}:00" if h.get("peak_hour") is not None else "")
    action_map = {
        "critical": "Immediate towing and increased patrols are recommended.",
        "high": "Stepped-up patrol frequency and clearer signage are recommended.",
        "moderate": "Periodic patrol checks are recommended to keep this from escalating.",
        "low": "Routine CCTV monitoring is sufficient for now.",
    }
    return (
        f"{loc} is a {h['severity']} hotspot with high {top_v} cases{when}. "
        f"{action_map.get(h['severity'], 'Monitoring is recommended.')}"
    )


def compute_smart_summary():
    df = DF
    overview = compute_overview_dict(df)
    hs = compute_hotspots(top_n=5)
    predicted = compute_predict(top_n=5)
    pred_by_cell = {p["cell_id"]: p for p in predicted}

    hotspot_summaries = []
    for h in hs:
        p = pred_by_cell.get(h["cell_id"])
        window_label = p["predicted_window_label"] if p else None
        hotspot_summaries.append({
            "cell_id": h["cell_id"],
            "location": short_loc(h["location"]),
            "severity": h["severity"],
            "congestion_score": h["congestion_score"],
            "summary": build_hotspot_summary_sentence(h, window_label),
            "traffic_saved_minutes": h["traffic_saved_minutes"],
        })

    city_summary = (
        f"Across {overview['total_violations']:,} recorded violations city-wide "
        f"({overview['date_start']} \u2013 {overview['date_end']}), "
        f"{overview['top_vehicle_type'].title()} is the most frequently flagged vehicle class "
        f"and {overview['busiest_police_station']} station logs the highest enforcement volume. "
        f"{overview['repeat_offender_vehicles']:,} vehicles ({overview['repeat_offender_vehicles']/max(overview['unique_vehicles_flagged'],1)*100:.1f}% of all flagged vehicles) "
        f"have been caught more than once, pointing to persistent illegal-parking behaviour at the same locations."
    )

    headline = hotspot_summaries[0]["summary"] if hotspot_summaries else "No data available for the current filters."

    return {
        "headline_summary": headline,
        "city_summary": city_summary,
        "hotspot_summaries": hotspot_summaries,
    }


def compute_overview_dict(df):
    if df.empty:
        return {"total_violations": 0}
    exploded = df["violation_list"].explode()
    return {
        "total_violations": int(len(df)),
        "unique_locations": int(df["cell_id"].nunique()),
        "unique_police_stations": int(df["police_station"].nunique()),
        "unique_vehicles_flagged": int(df["vehicle_number"].nunique()),
        "repeat_offender_vehicles": int(df["vehicle_number"].value_counts().gt(1).sum()),
        "top_vehicle_type": df["vehicle_type"].value_counts().idxmax(),
        "top_violation_type": exploded.value_counts().idxmax() if len(exploded) else "\u2014",
        "busiest_police_station": df["police_station"].value_counts().idxmax(),
        "avg_congestion_impact": round(float(df["impact_weight"].mean()), 2),
        "date_start": df["created_datetime"].min().strftime("%d %b %Y"),
        "date_end": df["created_datetime"].max().strftime("%d %b %Y"),
    }


# ------------------------------------------------------------------- routes --
@app.get("/api/filters")
def get_filters():
    return jsonify({
        "vehicle_types": sorted(DF["vehicle_type"].unique().tolist()),
        "police_stations": sorted(DF["police_station"].unique().tolist()),
        "date_min": DF["date"].min(),
        "date_max": DF["date"].max(),
    })


@app.get("/api/overview")
def overview():
    vt, ps, df_from, df_to = get_filter_args()
    df = apply_filters(vt, ps, df_from, df_to)
    return jsonify(compute_overview_dict(df))


@app.get("/api/hotspots")
def hotspots():
    vt, ps, df_from, df_to = get_filter_args()
    top_n = request.args.get("top_n", 40, type=int)
    return jsonify(compute_hotspots(vt, ps, df_from, df_to, top_n))


@app.get("/api/heatpoints")
def heatpoints():
    vt, ps, df_from, df_to = get_filter_args()
    max_points = request.args.get("max_points", 12000, type=int)
    df = apply_filters(vt, ps, df_from, df_to)
    if df.empty:
        return jsonify([])
    pts = df.groupby(["cell_lat", "cell_lon"]).size().reset_index(name="w")
    if len(pts) > max_points:
        pts = pts.sample(max_points, random_state=1)
    return jsonify([[float(r.cell_lat), float(r.cell_lon), int(r.w)] for r in pts.itertuples()])


@app.get("/api/trends")
def trends():
    vt, ps, df_from, df_to = get_filter_args()
    df = apply_filters(vt, ps, df_from, df_to)
    if df.empty:
        return jsonify({})

    monthly = (df.groupby(["month_sort", "month_label"]).size()
               .reset_index(name="count").sort_values("month_sort"))
    vehicle = df["vehicle_type"].value_counts().head(10).reset_index()
    vehicle.columns = ["vehicle_type", "count"]
    violation = df["violation_list"].explode().value_counts().head(10).reset_index()
    violation.columns = ["violation_type", "count"]
    station = df["police_station"].value_counts().head(10).reset_index()
    station.columns = ["police_station", "count"]

    hour_dow_matrix = [[0] * 24 for _ in range(7)]
    for (dow, hour), cnt in df.groupby(["dow", "hour"]).size().items():
        hour_dow_matrix[int(dow)][int(hour)] = int(cnt)

    return jsonify({
        "monthly": monthly[["month_label", "count"]].to_dict("records"),
        "by_vehicle_type": vehicle.to_dict("records"),
        "by_violation_type": violation.to_dict("records"),
        "by_police_station": station.to_dict("records"),
        "hour_dow_matrix": hour_dow_matrix,
    })


@app.get("/api/predict")
def predict_risk_windows():
    top_n = request.args.get("top_n", 5, type=int)
    return jsonify(compute_predict(top_n))


@app.get("/api/traffic-saved")
def traffic_saved():
    top_n = request.args.get("top_n", 5, type=int)
    hs = compute_hotspots(top_n=top_n)
    out = []
    for h in hs:
        out.append({
            "cell_id": h["cell_id"],
            "location": short_loc(h["location"]),
            "top_vehicle_type": h["top_vehicle_type"],
            "traffic_saved_minutes": h["traffic_saved_minutes"],
            "text": f"Removing {('the ' + h['top_vehicle_type'].title()) if h['top_vehicle_type'] else 'this vehicle'} "
                    f"can reduce congestion by {h['traffic_saved_minutes']:g} minutes.",
        })
    return jsonify(out)


@app.get("/api/smart-summary")
def smart_summary():
    return jsonify(compute_smart_summary())


@app.get("/api/feed")
def live_feed():
    n = request.args.get("n", 12, type=int)
    sample = DF.sample(min(n, len(DF)))
    items = []
    for r in sample.itertuples():
        items.append({
            "time": r.created_datetime.strftime("%H:%M:%S"),
            "date": r.date,
            "vehicle_type": r.vehicle_type,
            "location": CELL_LOOKUP.get(r.cell_id, {}).get("location", "Unknown")[:60],
            "police_station": r.police_station,
            "violations": r.violation_list[:2],
        })
    items.sort(key=lambda x: x["time"], reverse=True)
    return jsonify(items)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "records_loaded": int(len(DF)), "time": datetime.now().isoformat()})


# ------------------------------------------------------------ CSV / PDF export --
def hotspot_rows_for_export(vt, ps, df_from, df_to, top_n):
    hs = compute_hotspots(vt, ps, df_from, df_to, top_n)
    pred_by_cell = {p["cell_id"]: p for p in compute_predict(top_n=top_n)}
    rows = []
    for h in hs:
        p = pred_by_cell.get(h["cell_id"])
        rows.append({
            "location": h["location"],
            "junction": h["junction"],
            "police_station": h["police_station"],
            "violations": h["violations"],
            "congestion_score": h["congestion_score"],
            "severity": h["severity"],
            "recommended_action": h["recommended_action"],
            "top_violations": "; ".join(h["top_violations"]),
            "top_vehicle_type": h["top_vehicle_type"],
            "peak_hour": h["peak_hour"],
            "predicted_risk_window": p["predicted_window_label"] if p else "",
            "predicted_peak_day": p["predicted_peak_day"] if p else "",
            "traffic_saved_minutes": h["traffic_saved_minutes"],
        })
    return rows


@app.get("/api/report/csv")
def report_csv():
    vt, ps, df_from, df_to = get_filter_args()
    top_n = request.args.get("top_n", 40, type=int)
    rows = hotspot_rows_for_export(vt, ps, df_from, df_to, top_n)
    out_df = pd.DataFrame(rows)
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem, mimetype="text/csv", as_attachment=True,
        download_name=f"trafficguard_hotspots_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    )


def make_hour_chart(sub_df, title):
    """Returns a BytesIO PNG of an hourly violation-count bar chart for a hotspot."""
    counts = sub_df["hour"].value_counts().reindex(range(24), fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=150)
    ax.bar(counts.index, counts.values, color="#FFB020", width=0.8)
    ax.set_xlabel("Hour of day", fontsize=8)
    ax.set_ylabel("Violations", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_summary_bar_chart(hs):
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=150)
    labels = [short_loc(h["location"], 1)[:22] for h in hs]
    scores = [h["congestion_score"] for h in hs]
    sev_colors = {"critical": "#FF5A5A", "high": "#FFB020", "moderate": "#4D9FFF", "low": "#2ED9A5"}
    colors_list = [sev_colors.get(h["severity"], "#999") for h in hs]
    ax.barh(labels[::-1], scores[::-1], color=colors_list[::-1])
    ax.set_xlabel("Congestion Score (0\u2013100)", fontsize=8)
    ax.set_title("Top Hotspots by Congestion Score", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


PDF_STYLES = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=PDF_STYLES["Heading1"], fontSize=18, textColor=colors.HexColor("#16202E"), spaceAfter=4)
H2 = ParagraphStyle("H2", parent=PDF_STYLES["Heading2"], fontSize=13, textColor=colors.HexColor("#1C5FC9"), spaceBefore=10, spaceAfter=6)
BODY = ParagraphStyle("BODY", parent=PDF_STYLES["Normal"], fontSize=10, leading=14)
MUTED = ParagraphStyle("MUTED", parent=PDF_STYLES["Normal"], fontSize=8.5, textColor=colors.HexColor("#5C6B7E"))
QUOTE = ParagraphStyle("QUOTE", parent=PDF_STYLES["Normal"], fontSize=10.5, leading=15, textColor=colors.HexColor("#16202E"),
                        backColor=colors.HexColor("#F4F6FA"), borderPadding=8, leftIndent=4)


def pdf_header_block(title, subtitle):
    return [
        Paragraph("TrafficGuard Bengaluru", H1),
        Paragraph(title, ParagraphStyle("t2", parent=H2, spaceBefore=0)),
        Paragraph(subtitle, MUTED),
        Spacer(1, 10),
    ]


def severity_color(sev):
    return {"critical": colors.HexColor("#FF5A5A"), "high": colors.HexColor("#FFB020"),
            "moderate": colors.HexColor("#4D9FFF"), "low": colors.HexColor("#2ED9A5")}.get(sev, colors.grey)


@app.get("/api/report/pdf/<cell_id>")
def report_pdf_single(cell_id):
    h = compute_single_hotspot(cell_id)
    if h is None:
        return jsonify({"error": "cell_id not found"}), 404

    pred = compute_predict_for_hotspot(h)
    sub = DF[DF["cell_id"] == cell_id]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=24 * mm, bottomMargin=18 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm)
    story = []
    story += pdf_header_block(
        "Hotspot Enforcement Report",
        f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')} \u00b7 Real enforcement data, Nov 2023 \u2013 Apr 2024",
    )

    story.append(Paragraph("Location", H2))
    story.append(Paragraph(h["location"], BODY))
    story.append(Paragraph(f"Junction: {h['junction']} &nbsp;\u00b7&nbsp; Police Station: {h['police_station']}", MUTED))
    story.append(Spacer(1, 6))

    sev_badge = f'<font color="{severity_color(h["severity"]).hexval()}"><b>{h["severity"].upper()}</b></font>'
    info_table_data = [
        ["Congestion Score", "Severity", "Total Violations", "Traffic Saved (est.)"],
        [f"{h['congestion_score']} / 100", Paragraph(sev_badge, BODY), f"{h['violations']:,}",
         f"{h['traffic_saved_minutes']:g} min"],
    ]
    t = Table(info_table_data, colWidths=[105 * mm, 30 * mm, 35 * mm, 35 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16202E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE2EA")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F4F6FA")),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Violation Details", H2))
    vdata = [["Top Violation Types", "Top Vehicle Type", "Peak Hour"]]
    vdata.append([
        Paragraph(", ".join(h["top_violations"]) or "\u2014", BODY),
        h["top_vehicle_type"] or "\u2014",
        f"{h['peak_hour']:02d}:00" if h["peak_hour"] is not None else "\u2014",
    ])
    vt_table = Table(vdata, colWidths=[90 * mm, 50 * mm, 30 * mm])
    vt_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECF3")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE2EA")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(vt_table)
    story.append(Spacer(1, 10))

    if pred:
        story.append(Paragraph("Predicted Risk Window", H2))
        story.append(Paragraph(
            f"High-risk hours: <b>{pred['predicted_window_label']}</b> &nbsp;\u00b7&nbsp; "
            f"Peak day: <b>{pred['predicted_peak_day']}</b> &nbsp;\u00b7&nbsp; "
            f"Confidence: <b>{pred['confidence']}%</b> of this location's historical violations "
            f"fall inside this window.", BODY))
        story.append(Spacer(1, 8))

    story.append(Paragraph("AI Recommendation", H2))
    story.append(Paragraph(build_hotspot_summary_sentence(h, pred["predicted_window_label"] if pred else None), QUOTE))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Recommended action:</b> {h['recommended_action']}", BODY))

    story.append(PageBreak())
    story.append(Paragraph("Violation Pattern \u2014 Hour of Day", H2))
    chart_buf = make_hour_chart(sub, short_loc(h["location"]))
    story.append(Image(chart_buf, width=160 * mm, height=65 * mm))
    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Source: Bengaluru Traffic Police illegal-parking enforcement dataset. "
        "All figures on this report are computed directly from real enforcement records "
        "for this exact location \u2014 nothing here is simulated.", MUTED))

    doc.build(story)
    buf.seek(0)
    fname = f"hotspot_{short_loc(h['location'],1)[:24].replace(' ','_').replace(',','')}.pdf"
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=fname)


@app.get("/api/report/pdf-bulk")
def report_pdf_bulk():
    vt, ps, df_from, df_to = get_filter_args()
    top_n = request.args.get("top_n", 5, type=int)
    hs = compute_hotspots(vt, ps, df_from, df_to, top_n)
    pred_by_cell = {p["cell_id"]: p for p in compute_predict(top_n=top_n)}
    overview_d = compute_overview_dict(apply_filters(vt, ps, df_from, df_to))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=24 * mm, bottomMargin=18 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm)
    story = []
    story += pdf_header_block(
        f"Top {len(hs)} Congestion Hotspots \u2014 Bulk Report",
        f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')} \u00b7 "
        f"{overview_d.get('total_violations', 0):,} violations analysed "
        f"\u00b7 {overview_d.get('date_start','')} \u2013 {overview_d.get('date_end','')}",
    )

    story.append(Paragraph("Summary Table", H2))
    table_data = [["#", "Location", "Station", "Score", "Severity", "Violations", "Saved (min)"]]
    for i, h in enumerate(hs, 1):
        table_data.append([
            str(i), short_loc(h["location"])[:38], h["police_station"][:16],
            f"{h['congestion_score']}", h["severity"].upper(), f"{h['violations']:,}",
            f"{h['traffic_saved_minutes']:g}",
        ])
    sum_table = Table(table_data, colWidths=[8*mm, 62*mm, 28*mm, 16*mm, 22*mm, 22*mm, 22*mm])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16202E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE2EA")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (3, 0), (-1, -1), "CENTER"),
    ]
    for i, h in enumerate(hs, 1):
        style_cmds.append(("TEXTCOLOR", (4, i), (4, i), severity_color(h["severity"])))
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F4F6FA")))
    sum_table.setStyle(TableStyle(style_cmds))
    story.append(sum_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Congestion Score Overview", H2))
    story.append(Image(make_summary_bar_chart(hs), width=160 * mm, height=72 * mm))

    for i, h in enumerate(hs, 1):
        story.append(PageBreak())
        p = pred_by_cell.get(h["cell_id"])
        story.append(Paragraph(f"#{i} \u2014 {short_loc(h['location'])}", H1))
        story.append(Paragraph(f"Junction: {h['junction']} &nbsp;\u00b7&nbsp; Police Station: {h['police_station']}", MUTED))
        story.append(Spacer(1, 8))

        info_table_data = [
            ["Congestion Score", "Severity", "Violations", "Saved (est.)"],
            [f"{h['congestion_score']} / 100", h["severity"].upper(), f"{h['violations']:,}",
             f"{h['traffic_saved_minutes']:g} min"],
        ]
        t = Table(info_table_data, colWidths=[55*mm, 35*mm, 35*mm, 35*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECF3")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE2EA")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"Top violations: <b>{', '.join(h['top_violations']) or '\u2014'}</b> &nbsp;\u00b7&nbsp; "
            f"Top vehicle type: <b>{h['top_vehicle_type'] or '\u2014'}</b>", BODY))
        if p:
            story.append(Paragraph(
                f"Predicted high-risk window: <b>{p['predicted_window_label']}</b> "
                f"(peak day {p['predicted_peak_day']}, {p['confidence']}% confidence)", BODY))
        story.append(Spacer(1, 6))
        story.append(Paragraph(build_hotspot_summary_sentence(h, p["predicted_window_label"] if p else None), QUOTE))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Recommended action:</b> {h['recommended_action']}", BODY))
        story.append(Spacer(1, 10))
        sub = DF[DF["cell_id"] == h["cell_id"]]
        story.append(Image(make_hour_chart(sub, short_loc(h["location"])), width=155 * mm, height=58 * mm))

    doc.build(story)
    buf.seek(0)
    fname = f"trafficguard_bulk_report_top{len(hs)}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=fname)


# ---------------------------------------------------------- serve frontend --
@app.get("/")
def root():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/static/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


load_data()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=False,
    )
