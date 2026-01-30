import pandas as pd
import numpy as np

EV_PATH = "results/step5q/step5q_event_rows.csv"
DATA_PATH = "data/data.csv"
OUT_PATH = "results/step5q/step5q_gating_month_lat.csv"

KEYS = ["grid_id", "year", "month"]

# ----------------------------
# helpers
# ----------------------------
def ensure_coord_cols(df: pd.DataFrame):
    """
    Make sure df has 'lat_c' and 'lon_c' columns.
    If merge created lat_c_x/lat_c_y etc., coalesce and rename.
    """
    # LAT
    if "lat_c" not in df.columns:
        cand = [c for c in ["lat_c_x", "lat_c_y"] if c in df.columns]
        if len(cand) == 0:
            raise KeyError("No lat_c found (neither lat_c nor lat_c_x/lat_c_y).")
        # coalesce: prefer first non-null across candidates
        lat = None
        for c in cand:
            lat = df[c] if lat is None else lat.combine_first(df[c])
        df["lat_c"] = lat

    # LON
    if "lon_c" not in df.columns:
        cand = [c for c in ["lon_c_x", "lon_c_y"] if c in df.columns]
        if len(cand) == 0:
            raise KeyError("No lon_c found (neither lon_c nor lon_c_x/lon_c_y).")
        lon = None
        for c in cand:
            lon = df[c] if lon is None else lon.combine_first(df[c])
        df["lon_c"] = lon

    return df

def cliffs_delta(x, y):
    x = pd.to_numeric(x, errors="coerce").dropna().values
    y = pd.to_numeric(y, errors="coerce").dropna().values
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # O(n^2) but groups here are not huge; OK for diagnostics
    gt = sum(i > j for i in x for j in y)
    lt = sum(i < j for i in x for j in y)
    return (gt - lt) / (len(x) * len(y))

# ----------------------------
# load
# ----------------------------
ev = pd.read_csv(EV_PATH)
df = pd.read_csv(DATA_PATH)

# ----------------------------
# merge background variables from data.csv
# ----------------------------
vars_bg = [
    "NDVI_avg",
    "srad_mj_m2_day",
    "vpd_pa",
    "tmmn_celsius",
    "lat_c",
    "lon_c",
]
df_bg = df[KEYS + vars_bg].copy()

# IMPORTANT: use suffixes to avoid silent overwrites
ev = ev.merge(df_bg, on=KEYS, how="left", suffixes=("", "_bg"))

# handle lat/lon naming collisions robustly
# if ev already had lat_c, merge created lat_c_bg, etc.
if "lat_c_bg" in ev.columns and "lat_c" in ev.columns:
    ev["lat_c"] = ev["lat_c"].combine_first(ev["lat_c_bg"])
if "lon_c_bg" in ev.columns and "lon_c" in ev.columns:
    ev["lon_c"] = ev["lon_c"].combine_first(ev["lon_c_bg"])

# If instead merge produced lat_c_x/lat_c_y (older versions / different merge),
# coalesce them
ev = ensure_coord_cols(ev)

# ----------------------------
# classify by ΔGPP
# ----------------------------
ev["group"] = np.where(
    ev["delta_gpp"] < 0, "facilitation",
    np.where(ev["delta_gpp"] > 0, "suppression", "neutral")
)
ev = ev[(ev["group"] != "neutral") & (ev["isMHW"] == 1)].copy()

# ----------------------------
# latitude bands (you can adjust bins later)
# ----------------------------
lat_bins = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
ev["lat_band"] = pd.cut(ev["lat_c"], bins=lat_bins, include_lowest=True)

# ----------------------------
# gating summary by month × lat_band
# ----------------------------
MIN_N = 50
rows = []

for (m, lb), sub in ev.groupby(["month", "lat_band"], dropna=True):
    n_all = len(sub)
    if n_all < MIN_N:
        continue

    fac = sub[sub["group"] == "facilitation"]
    sup = sub[sub["group"] == "suppression"]
    if len(fac) < 10 or len(sup) < 10:
        continue

    row = {
        "month": int(m),
        "lat_band": str(lb),
        "n_all": int(n_all),
        "n_fac": int(len(fac)),
        "n_sup": int(len(sup)),
        "fac_ratio": float(len(fac) / n_all),
    }

    for v in ["NDVI_avg", "srad_mj_m2_day", "vpd_pa", "tmmn_celsius"]:
        row[f"{v}_med_fac"] = float(fac[v].median())
        row[f"{v}_med_sup"] = float(sup[v].median())
        row[f"{v}_delta_med"] = row[f"{v}_med_fac"] - row[f"{v}_med_sup"]
        row[f"{v}_cliffs"] = float(cliffs_delta(fac[v], sup[v]))

    rows.append(row)

out = pd.DataFrame(rows).sort_values("fac_ratio", ascending=False)
out.to_csv(OUT_PATH, index=False)

print("=" * 80)
print("Gating diagnostic (month × lat_band) generated")
print(f"Saved to: {OUT_PATH}")
print("=" * 80)
print(out.head(12).to_string(index=False))
