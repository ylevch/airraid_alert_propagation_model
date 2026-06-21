import os
import time
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ==========================================
# --- CONFIGURATION & TESTING MODES ---
# ==========================================
# 1. Training Sample Mode:
#    "last_365_days"      -> Use the most recent 365 days of data in the dataset (dynamic rolling year)
#    "2025", "2024", etc. -> Use a specific calendar year
TRAINING_SAMPLE_MODE = "last_365_days"

# 2. RUN_MODE options:
#    "api"               -> Fetch live data using the API token below (falls back to demo snapshot if empty)
#    "mig_takeoff"       -> Simulate a nationwide MiG alarm (All 24 oblasts turned Red instantly)
#    "border_escalation" -> Simulate active alerts localized to Kharkiv, Sumy, Chernihiv, and Kyivska oblasts
RUN_MODE = "api"

ALERTS_API_TOKEN = ""  # Put your secret token here if you have one
# ==========================================

HISTORICAL_URL = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv"
ALERTS_URL = "https://api.alerts.in.ua/v3/alerts/active.json"
KYIV_TZ = ZoneInfo("Europe/Kyiv")
TIMESTAMP_BASE = 1640000000

LUID_TO_OBLAST = {
    "31": "Kyiv",
    "1293": "Kharkivska oblast",
    "564": "Zaporizka oblast",
    "4": "Vinnytska oblast",
    "36": "Vinnytska oblast",
    "37": "Vinnytska oblast",
    "35": "Vinnytska oblast",
    "33": "Vinnytska oblast",
    "32": "Vinnytska oblast",
    "34": "Vinnytska oblast",
    "8": "Volynska oblast",
    "38": "Volynska oblast",
    "41": "Volynska oblast",
    "40": "Volynska oblast",
    "39": "Volynska oblast",
    "9": "Dnipropetrovska oblast",
    "44": "Dnipropetrovska oblast",
    "42": "Dnipropetrovska oblast",
    "46": "Dnipropetrovska oblast",
    "47": "Dnipropetrovska oblast",
    "45": "Dnipropetrovska oblast",
    "43": "Dnipropetrovska oblast",
    "48": "Dnipropetrovska oblast",
    "28": "Donetska oblast",
    "54": "Donetska oblast",
    "55": "Donetska oblast",
    "51": "Donetska oblast",
    "53": "Donetska oblast",
    "49": "Donetska oblast",
    "50": "Donetska oblast",
    "52": "Donetska oblast",
    "56": "Donetska oblast",
    "10": "Zhytomyrska oblast",
    "57": "Zhytomyrska oblast",
    "59": "Zhytomyrska oblast",
    "60": "Zhytomyrska oblast",
    "58": "Zhytomyrska oblast",
    "11": "Zakarpatska oblast",
    "61": "Zakarpatska oblast",
    "65": "Zakarpatska oblast",
    "63": "Zakarpatska oblast",
    "64": "Zakarpatska oblast",
    "66": "Zakarpatska oblast",
    "62": "Zakarpatska oblast",
    "12": "Zaporizka oblast",
    "147": "Zaporizka oblast",
    "146": "Zaporizka oblast",
    "149": "Zaporizka oblast",
    "148": "Zaporizka oblast",
    "145": "Zaporizka oblast",
    "5564": "Zaporizka oblast",
    "13": "Ivano-Frankivska oblast",
    "68": "Ivano-Frankivska oblast",
    "67": "Ivano-Frankivska oblast",
    "71": "Ivano-Frankivska oblast",
    "70": "Ivano-Frankivska oblast",
    "69": "Ivano-Frankivska oblast",
    "72": "Ivano-Frankivska oblast",
    "14": "Kyivska oblast",
    "78": "Kyivska oblast",
    "79": "Kyivska oblast",
    "75": "Kyivska oblast",
    "73": "Kyivska oblast",
    "74": "Kyivska oblast",
    "76": "Kyivska oblast",
    "77": "Kyivska oblast",
    "15": "Kirovohradska oblast",
    "82": "Kirovohradska oblast",
    "81": "Kirovohradska oblast",
    "83": "Kirovohradska oblast",
    "80": "Kirovohradska oblast",
    "27": "Lvivska oblast",
    "91": "Lvivska oblast",
    "94": "Lvivska oblast",
    "90": "Lvivska oblast",
    "88": "Lvivska oblast",
    "89": "Lvivska oblast",
    "92": "Lvivska oblast",
    "93": "Lvivska oblast",
    "17": "Mykolaivska oblast",
    "96": "Mykolaivska oblast",
    "95": "Mykolaivska oblast",
    "98": "Mykolaivska oblast",
    "97": "Mykolaivska oblast",
    "18": "Odeska oblast",
    "101": "Odeska oblast",
    "100": "Odeska oblast",
    "105": "Odeska oblast",
    "102": "Odeska oblast",
    "104": "Odeska oblast",
    "99": "Odeska oblast",
    "103": "Odeska oblast",
    "19": "Poltavska oblast",
    "107": "Poltavska oblast",
    "106": "Poltavska oblast",
    "108": "Poltavska oblast",
    "109": "Poltavska oblast",
    "5": "Rivnenska oblast",
    "110": "Rivnenska oblast",
    "111": "Rivnenska oblast",
    "112": "Rivnenska oblast",
    "113": "Rivnenska oblast",
    "20": "Sumska oblast",
    "117": "Sumska oblast",
    "118": "Sumska oblast",
    "116": "Sumska oblast",
    "114": "Sumska oblast",
    "115": "Sumska oblast",
    "21": "Ternopilska oblast",
    "120": "Ternopilska oblast",
    "119": "Ternopilska oblast",
    "121": "Ternopilska oblast",
    "22": "Kharkivska oblast",
    "125": "Kharkivska oblast",
    "127": "Kharkivska oblast",
    "126": "Kharkivska oblast",
    "123": "Kharkivska oblast",
    "128": "Kharkivska oblast",
    "124": "Kharkivska oblast",
    "122": "Kharkivska oblast",
    "23": "Khersonska oblast",
    "129": "Khersonska oblast",
    "133": "Khersonska oblast",
    "131": "Khersonska oblast",
    "130": "Khersonska oblast",
    "132": "Khersonska oblast",
    "3": "Khmelnytska oblast",
    "135": "Khmelnytska oblast",
    "134": "Khmelnytska oblast",
    "136": "Khmelnytska oblast",
    "24": "Cherkaska oblast",
    "150": "Cherkaska oblast",
    "153": "Cherkaska oblast",
    "151": "Cherkaska oblast",
    "152": "Cherkaska oblast",
    "26": "Chernivetska oblast",
    "138": "Chernivetska oblast",
    "139": "Chernivetska oblast",
    "137": "Chernivetska oblast",
    "25": "Chernihivska oblast",
    "144": "Chernihivska oblast",
    "141": "Chernihivska oblast",
    "142": "Chernihivska oblast",
    "143": "Chernihivska oblast",
    "140": "Chernihivska oblast",
}


def map_luid_to_english_oblast(luid):
    luid_str = str(luid)
    if luid_str in LUID_TO_OBLAST:
        return LUID_TO_OBLAST[luid_str]
    elif (
        luid_str.startswith("5")
        and len(luid_str) > 1
        and luid_str[1:] in LUID_TO_OBLAST
    ):
        return LUID_TO_OBLAST[luid_str[1:]]
    return None


def get_live_level_and_weight(luid, name):
    luid_str = str(luid)
    oblast_ids = {
        "3",
        "4",
        "5",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "31",
    }
    if luid_str in oblast_ids:
        return "oblast", 1.0
    if "громада" in str(name).lower() or luid_str.startswith("5"):
        return "hromada", 0.1
    return "raion", 0.3


# --- Part 1: Fit Historical VARX Model ---
start_time = time.time()
print("Fetching and loading historical training dataset...")
df_hist = pd.read_csv(HISTORICAL_URL)

df_hist["started_at"] = pd.to_datetime(df_hist["started_at"]).dt.tz_localize(None)
df_hist["finished_at"] = pd.to_datetime(df_hist["finished_at"]).dt.tz_localize(None)

# Remove permanent sirens
exclude_regions = ["Luhanska oblast", "Avtonomna Respublika Krym", "Crimea"]
df_hist = df_hist[~df_hist["oblast"].isin(exclude_regions)]

start_date = "2022-03-15 16:00:00"
end_date = df_hist["finished_at"].max()
time_grid = pd.date_range(start=start_date, end=end_date, freq="10min")

df_hist["weight"] = df_hist["level"].apply(
    lambda lvl: 1.0 if lvl == "oblast" else (0.3 if lvl == "raion" else 0.1)
)
target_cols = sorted(df_hist["oblast"].unique())
num_oblasts = len(target_cols)

# Kyiv name resolution
kyiv_city_col = None
for col in target_cols:
    if "kyiv" in col.lower() and "oblast" not in col.lower():
        kyiv_city_col = col
        break
if kyiv_city_col is None:
    for col in target_cols:
        if "kyiv" in col.lower():
            kyiv_city_col = col
            break
if kyiv_city_col is None:
    kyiv_city_col = "Kyiv"

for k, v in list(LUID_TO_OBLAST.items()):
    if v == "Kyiv":
        LUID_TO_OBLAST[k] = kyiv_city_col

print(f"Dynamically mapped Kyiv City LUID ('31') to column name: '{kyiv_city_col}'")


print("Constructing historical training matrix...")
ts_array = np.zeros((len(time_grid), num_oblasts), dtype=np.float32)
grid_values = time_grid.values

for col_idx, obl in enumerate(target_cols):
    obl_df = df_hist[df_hist["oblast"] == obl]
    if obl_df.empty:
        continue
    starts = np.searchsorted(grid_values, obl_df["started_at"].values)
    ends = np.searchsorted(grid_values, obl_df["finished_at"].values)
    weights = obl_df["weight"].values

    starts = np.clip(starts, 0, len(time_grid) - 1)
    ends = np.clip(ends, 0, len(time_grid) - 1)

    obl_array = np.zeros(len(time_grid), dtype=np.float32)
    for s, e, w in zip(starts, ends, weights):
        if s <= e:
            obl_array[s : e + 1] = np.maximum(obl_array[s : e + 1], w)
    ts_array[:, col_idx] = obl_array

ts_matrix = pd.DataFrame(ts_array, index=time_grid, columns=target_cols)

# Build Exogenous controls
threshold_oblasts = int(0.75 * num_oblasts)
active_counts = (ts_matrix > 0.5).sum(axis=1)
ts_matrix["nationwide_shock"] = (active_counts >= threshold_oblasts).astype(float)
ts_matrix["is_night"] = (
    (ts_matrix.index.hour >= 22) | (ts_matrix.index.hour < 6)
).astype(float)
ts_matrix["is_winter"] = ts_matrix.index.month.isin([10, 11, 12, 1, 2, 3]).astype(float)
mean_active_proportion = ts_matrix[target_cols].mean(axis=1)
ts_matrix["intensity_7d"] = mean_active_proportion.rolling(
    window=1008, min_periods=1
).mean()

# --- Part 1B: Slicing the Training Sample ---
max_grid_date = ts_matrix.index.max()
if TRAINING_SAMPLE_MODE == "last_365_days":
    start_slice = max_grid_date - pd.Timedelta(days=365)
    sub_df = ts_matrix.loc[start_slice:max_grid_date]
    print(
        f"Training on the LAST 365 DAYS ({start_slice.strftime('%Y-%m-%d')} to {max_grid_date.strftime('%Y-%m-%d')})..."
    )
else:
    sub_df = ts_matrix.loc[TRAINING_SAMPLE_MODE]
    print(f"Training on year specific qualities: {TRAINING_SAMPLE_MODE}...")

# --- Part 1C: Calculate Expected Shock Duration ---
# Identify contiguous blocks of 1.0 in "nationwide_shock"
shocks_series = sub_df["nationwide_shock"]
blocks = (shocks_series != shocks_series.shift()).cumsum()
block_lengths = shocks_series[shocks_series == 1.0].groupby(blocks).size()

if len(block_lengths) > 0:
    avg_intervals = block_lengths.mean()
    expected_duration_minutes = int(round(avg_intervals * 10))
    MIG_PERSISTENCE_STEPS = int(round(avg_intervals))
else:
    expected_duration_minutes = 60  # Default fallback
    MIG_PERSISTENCE_STEPS = 6

print(
    f"Empirically calculated expected MiG/Nationwide shock duration: {expected_duration_minutes} minutes ({MIG_PERSISTENCE_STEPS} steps)."
)

# Fit the VARX Model on the selected subsample
lags = 6
Y = sub_df[target_cols]
exog_cols = ["nationwide_shock", "is_night", "is_winter", "intensity_7d"]
X_exog = sub_df[exog_cols]

lagged_dfs = []
for i in range(1, lags + 1):
    lagged_dfs.append(Y.shift(i).add_suffix(f"_lag{i}"))

X_features = pd.concat(lagged_dfs + [X_exog], axis=1).dropna()
Y_aligned = Y.loc[X_features.index]

print(f"Fitting Ridge VARX model...")
model = Ridge(alpha=10.0)
model.fit(X_features.values, Y_aligned.values)
print("Historical model training complete.")


# --- Part 2: State Generator (API vs. Simulation) ---
now = datetime.now(KYIV_TZ)
now_ts = int(time.time()) - TIMESTAMP_BASE

if RUN_MODE == "api":
    print(
        f"\nFetching live alerts state as of {now.strftime('%Y-%m-%d %H:%M:%S')} (Kyiv Time)..."
    )
    headers = {}
    if ALERTS_API_TOKEN:
        headers["Authorization"] = f"Bearer {ALERTS_API_TOKEN}"
    try:
        response = requests.get(ALERTS_URL, headers=headers, timeout=10)
        live_alerts = response.json().get("alerts", [])
        print(f"Successfully connected. Found {len(live_alerts)} live alert signals.")
    except Exception as e:
        print(f"API Connection failed ({e}). Falling back to demo snapshot...")
        live_alerts = [
            {"luid": "31", "n": "м. Київ", "s": now_ts - 1200},  # Kyiv City
            {"luid": "20", "n": "Сумська область", "s": now_ts - 2400},  # Sumska
        ]

elif RUN_MODE == "mig_takeoff":
    print(f"\n[SIMULATOR] Simulating a Nationwide MiG-31K Takeoff...")
    live_alerts = []
    # Turn every single oblast red (weight = 1.0, started 10 minutes ago)
    oblast_ids = [
        "3",
        "4",
        "5",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "31",
    ]
    for luid in oblast_ids:
        live_alerts.append(
            {
                "luid": luid,
                "n": "Oblast Level Siren",
                "s": now_ts - 600,  # 10 minutes ago
            }
        )

elif RUN_MODE == "border_escalation":
    print(f"\n[SIMULATOR] Simulating local border escalation...")
    live_alerts = [
        {"luid": "25", "n": "Чернігівська область", "s": now_ts - 1800},
        {"luid": "20", "n": "Сумська область", "s": now_ts - 2400},
        {"luid": "22", "n": "Харківська область", "s": now_ts - 3000},
        {"luid": "14", "n": "Київська область", "s": now_ts - 600},
    ]

# --- Part 3: Map Live Alerts onto Historical Grid Lags ---
H_steps = [now - timedelta(minutes=10 * (lags - i)) for i in range(lags)]
live_state_history = {col: np.zeros(lags) for col in target_cols}

print("Harmonizing and mapping active alerts onto our historical feature grid...")
for alert in live_alerts:
    luid = str(alert["luid"])
    api_name = alert.get("n", "")
    oblast_col = map_luid_to_english_oblast(luid)

    if oblast_col and oblast_col in target_cols:
        _, weight = get_live_level_and_weight(luid, api_name)
        start_timestamp = alert.get("s")
        if start_timestamp:
            start_dt = datetime.fromtimestamp(
                start_timestamp + TIMESTAMP_BASE, tz=KYIV_TZ
            )
        else:
            start_dt = now

        for step_idx, step_time in enumerate(H_steps):
            if start_dt <= step_time:
                live_state_history[oblast_col][step_idx] = max(
                    live_state_history[oblast_col][step_idx], weight
                )

# Detect if a nationwide shock is currently active at t=0
current_active_oblasts = sum(
    1 for col in target_cols if live_state_history[col][-1] > 0.5
)
live_nationwide_shock_active = current_active_oblasts >= threshold_oblasts

if live_nationwide_shock_active:
    print(
        f"** ALERT: Nationwide MiG/Shock detected! ({current_active_oblasts}/{num_oblasts} oblasts active) **"
    )
else:
    print(
        f"Localized alert patterns detected. ({current_active_oblasts}/{num_oblasts} oblasts active)"
    )

# --- Part 4: Run Probability Trajectory Simulation with Dynamic Schedulers ---
steps_to_simulate = 36  # Forecast 360 minutes (6 hours)
history = np.zeros((lags + steps_to_simulate, len(target_cols)))

for col_idx, col in enumerate(target_cols):
    history[0:lags, col_idx] = live_state_history[col]

print(f"Simulating alert trajectories...")
for t in range(lags, lags + steps_to_simulate):
    lagged_vector = []
    for l in range(1, lags + 1):
        lagged_vector.extend(history[t - l, :])

    future_time = now + timedelta(minutes=10 * (t - lags + 1))

    is_night = 1.0 if (future_time.hour >= 22 or future_time.hour < 6) else 0.0
    is_winter = 1.0 if future_time.month in [10, 11, 12, 1, 2, 3] else 0.0
    intensity_7d = sub_df["intensity_7d"].mean()

    # DYNAMIC SHOCK PERSISTENCE: Shock persists for the calculated empirical length, then decays
    if live_nationwide_shock_active and (t - lags < MIG_PERSISTENCE_STEPS):
        nationwide_shock = 1.0
    else:
        nationwide_shock = 0.0

    exog_values = [nationwide_shock, is_night, is_winter, intensity_7d]
    feature_vector = np.array(lagged_vector + exog_values).reshape(1, -1)

    pred = model.predict(feature_vector)[0]
    pred = np.clip(pred, 0.0, 1.0)
    history[t, :] = pred

# Generate forecast dataframe
forecast_steps = [i * 10 for i in range(steps_to_simulate + 1)]
forecast_df = pd.DataFrame(
    history[lags - 1 :], columns=target_cols, index=forecast_steps
)

# --- Part 5: Print Report and Generate Plots ---
print(f"\n=== Current Active Alert Intensity (Last 1 Hour) ===")
cols_to_print = [kyiv_city_col, "Kyivska oblast", "Sumska oblast", "Kharkivska oblast"]
for col in cols_to_print:
    if col in live_state_history:
        current_val = live_state_history[col][-1]
        print(
            f"  {col:<18} : {current_val:.1f} (Level: {'Active' if current_val > 0.0 else 'Quiet'})"
        )

print(f"\n=== Predicted Probability Trajectory for {kyiv_city_col} (Next 6 Hours) ===")
print(f"Minutes Ahead | Expected Probability/Intensity")
print("-" * 45)
for mins in range(0, (steps_to_simulate * 10) + 1, 30):
    val = forecast_df.loc[mins, kyiv_city_col]
    print(f"  +{mins:<10} | {val:.4f} ({val * 100:.1f}%)")

# Visualize Kyiv and neighboring trajectories
visualize_oblasts = [
    kyiv_city_col,
    "Kyivska oblast",
    "Sumska oblast",
    "Kharkivska oblast",
    "Chernihivska oblast",
]
visualize_oblasts = [o for o in visualize_oblasts if o in target_cols]

plt.figure(figsize=(11, 6.5))
for obl in visualize_oblasts:
    plt.plot(forecast_df.index, forecast_df[obl], marker="o", label=obl)

plt.title(
    f"Real-Time 6-Hour Alert Forecast Trajectory (Base Time: {now.strftime('%H:%M Kyiv Time')})\n"
    f"Training Mode: {TRAINING_SAMPLE_MODE.upper()} | Empirical Takeoff Duration: {expected_duration_minutes} min"
)
plt.xlabel("Time ahead (minutes)")
plt.ylabel("Predicted Alert Probability/Intensity")
plt.xticks(np.arange(0, (steps_to_simulate * 10) + 1, 30))
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
