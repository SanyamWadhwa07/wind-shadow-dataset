"""
Wind Shadow Dataset — Medium Difficulty, Feature Engineering Focus
Train: 750 snapshots × 80 turbines = 60,000 rows
Test:  125 snapshots × 80 turbines = 10,000 rows
Target: power_kw (single column, 0–6000 kW)

Key design:
  - downstream_dist and wake_deficit NOT given — must be derived
  - Raw features R² ~ 0.72, engineered R² ~ 0.93
  - Gap comes from deriving spatial features from wind_direction + grid position
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED         = 42
N_SNAP_TR    = 750
N_SNAP_TE    = 125
N_ROWS       = 10
N_COLS       = 8
RATED_POWER  = 6000.0
CUTOUT_SPEED = 25.0
NOISE_FRAC   = 0.012


def power_curve(ws):
    p  = np.zeros_like(ws, dtype=float)
    m1 = (ws >= 3.0)  & (ws < 12.0)
    m2 = (ws >= 12.0) & (ws < CUTOUT_SPEED)
    p[m1] = RATED_POWER * ((ws[m1] - 3.0) / 9.0) ** 3
    p[m2] = RATED_POWER
    return p


def jensen_deficit(downstream_dist, turbulence_intensity, crosswind_dist):
    # Highly nonlinear wake model with crosswind effects
    k       = 0.03 + 0.12 * turbulence_intensity
    dd      = np.maximum(downstream_dist, 0.0)
    
    # Strong downstream deficit
    deficit = 1.0 / (1.0 + k * dd * 15.0 + 1e-4) ** 3.0
    
    # Crosswind gaussian spread (requires crosswind_dist to model)
    crosswind_effect = np.exp(-8.0 * crosswind_dist**2 / (1.0 + dd + 0.1))
    deficit = deficit * (0.3 + 0.7 * crosswind_effect)
    
    return np.clip(deficit, 0.25, 1.0)


def generate_weather(n, ood=False):
    rng = np.random.default_rng(99 if ood else SEED)

    if ood:
        n1, n2, n3 = n//3, n//3, n - 2*(n//3)
        ws = np.concatenate([
            rng.uniform(25.5, 33.0, n1),
            np.clip(rng.weibull(2.2, n2) * 9.5, 3, 22),
            np.clip(rng.weibull(2.2, n3) * 9.5, 3, 22),
        ])
        wd = np.concatenate([
            rng.uniform(0, 360, n1),
            rng.uniform(0, 360, n2),   # OOD: all directions
            rng.uniform(0, 360, n3),
        ])
        temp = np.concatenate([
            rng.uniform(0, 15, n1),
            rng.uniform(-20, -14, n2),
            rng.uniform(5, 20, n3),
        ])
        shear = np.concatenate([
            rng.uniform(0.05, 0.25, n1),
            rng.uniform(0.30, 0.45, n2),
            rng.uniform(0.05, 0.25, n3),
        ])
        ti = np.concatenate([
            rng.uniform(0.04, 0.18, n1),
            rng.uniform(0.04, 0.12, n2),
            rng.uniform(0.06, 0.22, n3),
        ])
    else:
        ws    = np.clip(rng.weibull(2.2, n) * 9.5, 3, 24.5)  # floor at 3 (cut-in)
        wd    = rng.uniform(0, 360, n)   # KEY FIX: full 360° so row_norm != downstream
        temp  = rng.normal(9, 6, n)
        shear = rng.beta(2, 5, n) * 0.35 + 0.05
        ti    = rng.uniform(0.04, 0.18, n)

    density = np.clip(
        1.225 * (1 - 0.0034 * (temp - 15) / 10)
        + rng.normal(0, 0.003, n), 1.10, 1.35
    )
    sst   = np.clip(temp + rng.uniform(2, 5, n), 2, 28)
    wave  = np.clip(rng.exponential(1.2, n) + 0.3, 0.1, 8.0)
    press = np.clip(rng.normal(1013, 10, n), 975, 1040)
    solar = np.clip(np.maximum(0, rng.normal(280, 220, n)), 0, 950)
    hour  = rng.integers(0, 24, n)
    month = rng.integers(1, 13, n)

    return pd.DataFrame({
        "wind_speed_ms":        np.round(ws, 3),
        "wind_direction_deg":   np.round(wd, 2),
        "turbulence_intensity": np.round(ti, 4),
        "air_density_kgm3":     np.round(density, 4),
        "wind_shear_exp":       np.round(shear, 4),
        "ambient_temp_c":       np.round(temp, 2),
        "sea_surface_temp_c":   np.round(sst, 2),
        "wave_height_m":        np.round(wave, 3),
        "pressure_hpa":         np.round(press, 1),
        "solar_irradiance_wm2": np.round(solar, 1),
        "hour_of_day":          hour.astype(int),
        "month":                month.astype(int),
    })


def expand_to_long(weather_df, id_offset=0, ood=False):
    rng = np.random.default_rng(88 if ood else 55)
    n   = len(weather_df)

    ws      = weather_df["wind_speed_ms"].values
    wd      = weather_df["wind_direction_deg"].values
    temp    = weather_df["ambient_temp_c"].values
    density = weather_df["air_density_kgm3"].values
    shear   = weather_df["wind_shear_exp"].values
    ti      = weather_df["turbulence_intensity"].values

    wd_rad   = np.deg2rad(wd)
    wind_sin = np.sin(wd_rad)
    wind_cos = np.cos(wd_rad)
    base_p   = power_curve(ws)

    chunks = []
    for row in range(1, N_ROWS + 1):
        for col in range(1, N_COLS + 1):

            row_norm = (row - 1) / (N_ROWS - 1)
            col_norm = (col - 1) / (N_COLS - 1)

            # Internal only — not given as features
            downstream = row_norm * wind_cos + col_norm * wind_sin
            crosswind  = row_norm * wind_sin - col_norm * wind_cos
            wake_def   = jensen_deficit(downstream, ti, crosswind)

            # Power target with highly nonlinear effects
            power = base_p.copy() * wake_def * (density / 1.225)
            
            # Deep wake has quadratic shear sensitivity
            shear_factor = 1.0
            if row >= 7:
                shear_factor = np.clip(1 - 0.25 * (shear - 0.12)**2 * 20, 0.55, 1.0)
            elif row >= 4:
                shear_factor = np.clip(1 - 0.12 * (shear - 0.15), 0.80, 1.0)
            power *= shear_factor
            
            # Wake-TI interaction (stronger in deep wake zones)
            wake_intensity = np.maximum(0, downstream)
            ti_effect = 1.0 - 0.20 * ti * wake_intensity**0.7
            power *= np.clip(ti_effect, 0.70, 1.0)
            
            # Crosswind clustering effect (turbines aligned crosswind interfere)
            crosswind_penalty = 1.0 - 0.08 * np.exp(-3.0 * crosswind**2)
            power *= crosswind_penalty

            icing = np.where(
                temp < -10,
                0.70 + 0.03 * np.clip(temp + 10, -10, 0),
                1.0
            )
            power *= icing

            # Add noise THEN floor — fixes the spurious zeros
            noise = rng.normal(0, NOISE_FRAC * RATED_POWER, n)
            power = power + noise

            # Floor at 10 kW for non-cutout, hard zero for cutout
            is_cutout = (ws >= CUTOUT_SPEED)
            power     = np.where(is_cutout, 0.0,
                            np.clip(power, 10.0, RATED_POWER))

            chunk = weather_df.copy().reset_index(drop=True)
            # Only give raw integer positions - competitors must normalize
            chunk["turbine_row"]  = row
            chunk["turbine_col"]  = col
            chunk["is_edge_col"]  = int(col == 1 or col == N_COLS)
            chunk["power_kw"]     = np.round(power, 2)
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df.insert(0, "id", np.arange(id_offset, id_offset + len(df)))
    return df


# ── Generate ──────────────────────────────────────────────────
print(f"Generating {N_SNAP_TR} training snapshots → "
      f"{N_SNAP_TR * N_ROWS * N_COLS:,} rows...")
train_w    = generate_weather(N_SNAP_TR, ood=False)
train_long = expand_to_long(train_w, id_offset=0, ood=False)

print(f"Generating {N_SNAP_TE} test snapshots (OOD) → "
      f"{N_SNAP_TE * N_ROWS * N_COLS:,} rows...")
test_w    = generate_weather(N_SNAP_TE, ood=True)
test_long = expand_to_long(test_w, id_offset=len(train_long), ood=True)

sample_sub = pd.DataFrame({
    "id":       test_long["id"].values,
    "power_kw": 0.0
})

Path("public").mkdir(exist_ok=True)
train_long.to_csv("public/train.csv", index=False)
test_long.drop(columns=["power_kw"]).to_csv("public/test.csv", index=False)
sample_sub.to_csv("public/sample_submission.csv", index=False)

# ── Verify R² gap ─────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# Raw: no wind-aligned features — competitor hasn't engineered yet
raw_feats = [
    "wind_speed_ms", "wind_direction_deg", "turbulence_intensity",
    "air_density_kgm3", "wind_shear_exp", "ambient_temp_c",
    "turbine_row", "turbine_col", "is_edge_col"
]

# Engineered: competitor derives these from wind_direction + grid position
def add_engineered(df):
    wd_rad = np.deg2rad(df["wind_direction_deg"])
    df = df.copy()
    # First normalize the grid positions
    df["row_norm"] = (df["turbine_row"] - 1) / (N_ROWS - 1)
    df["col_norm"] = (df["turbine_col"] - 1) / (N_COLS - 1)
    # Then compute wind-aligned features
    df["wind_sin"]          = np.sin(wd_rad)
    df["wind_cos"]          = np.cos(wd_rad)
    df["downstream_dist"]   = df["row_norm"] * df["wind_cos"] + \
                               df["col_norm"] * df["wind_sin"]
    df["crosswind_dist"]    = df["row_norm"] * df["wind_sin"] - \
                               df["col_norm"] * df["wind_cos"]
    dd = np.maximum(df["downstream_dist"], 0)
    k  = 0.03 + 0.12 * df["turbulence_intensity"]
    
    # Main wake deficit with crosswind
    base_deficit = 1.0 / (1.0 + k * dd * 15.0 + 1e-4) ** 3.0
    crosswind_effect = np.exp(-8.0 * df["crosswind_dist"]**2 / (1.0 + dd + 0.1))
    df["wake_deficit"]      = base_deficit * (0.3 + 0.7 * crosswind_effect)
    
    # Higher-order features
    df["ws_cubed"]          = df["wind_speed_ms"] ** 3
    df["density_ratio"]     = df["air_density_kgm3"] / 1.225
    df["wake_x_ws"]         = df["wake_deficit"] * df["wind_speed_ms"]
    df["down_x_ws"]         = df["downstream_dist"] * df["wind_speed_ms"]
    df["ti_x_wake"]         = df["turbulence_intensity"] * dd**0.7
    df["cross_sq"]          = df["crosswind_dist"]**2
    df["shear_sq"]          = df["wind_shear_exp"]**2
    return df

eng_feats = raw_feats + [
    "row_norm", "col_norm", "wind_sin", "wind_cos", "downstream_dist", "crosswind_dist",
    "wake_deficit", "ws_cubed", "density_ratio", "wake_x_ws", "down_x_ws", 
    "ti_x_wake", "cross_sq", "shear_sq"
]

sample    = train_long.sample(10000, random_state=42)
sample_e  = add_engineered(sample)
y         = sample["power_kw"].values

r2_raw = r2_score(y, LinearRegression()
                  .fit(sample[raw_feats], y)
                  .predict(sample[raw_feats]))
r2_eng = r2_score(y, LinearRegression()
                  .fit(sample_e[eng_feats], y)
                  .predict(sample_e[eng_feats]))

print(f"\n── Signal Verification ──────────────────────────────")
print(f"Linear Regression (raw features):        R² = {r2_raw:.3f}")
print(f"Linear Regression (engineered features): R² = {r2_eng:.3f}")
print(f"Gap: {r2_eng - r2_raw:.3f}  (target > 0.15)")
print(f"\ntrain.csv:             {train_long.shape}")
print(f"test.csv:              {test_long.drop(columns=['power_kw']).shape}")
print(f"sample_submission.csv: {sample_sub.shape}")
print(f"\nPower stats (train):")
print(f"  min  = {train_long['power_kw'].min():.1f} kW")
print(f"  max  = {train_long['power_kw'].max():.1f} kW")
print(f"  mean = {train_long['power_kw'].mean():.1f} kW")
print(f"  std  = {train_long['power_kw'].std():.1f} kW")
print(f"  zeros = {(train_long['power_kw'] == 0.0).sum()} rows")
print(f"\nExpected leaderboard:")
print(f"  Naive mean baseline:   ~{train_long['power_kw'].std():.0f} kW RMSE")
print(f"  XGBoost raw features:  ~280 kW")
print(f"  XGBoost engineered:    ~90  kW")
print(f"  Near-optimal:          ~40  kW")