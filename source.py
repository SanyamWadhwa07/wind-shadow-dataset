"""
Generate synthetic Wind Shadow dataset
Run: python generate_dataset.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N_TRAIN, N_TEST = 60000, 15000
N_ROWS, N_COLS  = 10, 8
RATED_POWER     = 6000   # kW
CUTOUT_SPEED    = 25.0   # m/s

def power_curve(ws):
    """Realistic 3-region wind turbine power curve."""
    p = np.zeros_like(ws)
    cut_in  = 3.0
    rated   = 12.0
    cutout  = CUTOUT_SPEED
    mask1   = (ws >= cut_in)  & (ws < rated)
    mask2   = (ws >= rated)   & (ws < cutout)
    p[mask1] = RATED_POWER * ((ws[mask1] - cut_in) / (rated - cut_in)) ** 3
    p[mask2] = RATED_POWER
    return p

def wake_deficit(row, col, wind_dir_deg, wind_speed):
    """
    Jensen wake model approximation.
    Returns fractional power deficit at turbine (row, col).
    """
    # Prevailing wind direction determines which rows are 'downstream'
    # Simplification: map wind direction to effective downstream distance
    wind_rad   = np.deg2rad(wind_dir_deg)
    # Downstream distance in grid units
    down_dist  = (row - 1) * np.abs(np.cos(wind_rad)) + \
                 (col - 1) * np.abs(np.sin(wind_rad))
    # Jensen model: deficit decays with 1/(1 + k*x/D)²
    k = 0.04   # wake decay constant (offshore)
    D = 1.0    # rotor diameter in grid units
    deficit = 1.0 / (1.0 + k * down_dist / D) ** 2
    # Scale with turbulence (higher TI = faster wake recovery)
    return deficit

def generate_samples(n, ood=False):
    rng = np.random.default_rng(99 if ood else 42)

    if ood:
        # OOD: mix of storm, polar, extreme wake-steering
        n1, n2, n3 = n//3, n//3, n - 2*(n//3)
        ws1 = rng.uniform(25.5, 34.0, n1)   # storm cutout
        ws2 = rng.uniform(4.0,  14.0, n2)   # polar vortex
        ws3 = rng.uniform(6.0,  20.0, n3)   # wake steering
        ws  = np.concatenate([ws1, ws2, ws3])

        wd1 = rng.uniform(0, 360, n1)
        wd2 = rng.uniform(0, 360, n2)
        wd3 = rng.choice([45,135,225,315], size=n3) + rng.uniform(-5,5,n3)
        wd  = np.concatenate([wd1, wd2, wd3])

        temp1 = rng.uniform(-5, 10, n1)
        temp2 = rng.uniform(-20, -15, n2)
        temp3 = rng.uniform(0, 20, n3)
        temp  = np.concatenate([temp1, temp2, temp3])

        shear1 = rng.uniform(0.05, 0.25, n1)
        shear2 = rng.uniform(0.38, 0.45, n2)
        shear3 = rng.uniform(0.10, 0.30, n3)
        shear  = np.concatenate([shear1, shear2, shear3])
    else:
        ws    = rng.weibull(2.2, n) * 9.5          # Weibull dist, mean ~8.5 m/s
        ws    = np.clip(ws, 0, 24.5)
        wd    = rng.uniform(200, 320, n)            # prevailing westerlies
        temp  = rng.normal(8, 7, n)
        shear = rng.beta(2, 5, n) * 0.4 + 0.05

    ti      = rng.uniform(0.04, 0.20, n)
    density = 1.225 - 0.003 * (temp - 15) / 10 + rng.normal(0, 0.005, n)
    density = np.clip(density, 1.10, 1.35)
    sst     = temp + rng.uniform(2, 6, n)
    sst     = np.clip(sst, 2, 28)
    wave    = rng.exponential(1.2, n) + 0.3
    wave    = np.clip(wave, 0.1, 8.5)
    press   = rng.normal(1013, 12, n)
    press   = np.clip(press, 970, 1040)
    solar   = np.maximum(0, rng.normal(300, 250, n))
    solar   = np.clip(solar, 0, 1000)
    hour    = rng.integers(0, 24, n)
    month   = rng.integers(1, 13, n)

    df = pd.DataFrame({
        "wind_speed_ms":        ws,
        "wind_direction_deg":   wd,
        "turbulence_intensity": ti,
        "air_density_kgm3":     density,
        "wind_shear_exp":       shear,
        "ambient_temp_c":       temp,
        "sea_surface_temp_c":   sst,
        "wave_height_m":        wave,
        "pressure_hpa":         press,
        "solar_irradiance_wm2": solar,
        "hour_of_day":          hour,
        "month":                month,
    })

    # Generate targets
    base_power = power_curve(ws)
    targets = {}
    for row in range(1, N_ROWS + 1):
        for col in range(1, N_COLS + 1):
            deficit  = wake_deficit(row, col, wd, ws)
            power_z  = base_power * deficit
            # Density correction
            power_z *= (density / 1.225)
            # Shear effect on deep wake
            if row >= 7:
                power_z *= (1 - 0.15 * (shear - 0.15))
            # Temperature effect (icing proxy)
            power_z *= np.where(temp < -10, 0.7, 1.0)
            # Add realistic noise
            noise    = rng.normal(0, 0.03 * RATED_POWER, n)
            power_z  = np.clip(power_z + noise, 0, RATED_POWER)
            targets[f"t_{row}_{col}"] = power_z

    return df, pd.DataFrame(targets)

print("Generating training data...")
train_feat, train_tgt = generate_samples(N_TRAIN, ood=False)
train = pd.concat([train_feat, train_tgt], axis=1)
train.insert(0, "id", range(N_TRAIN))

print("Generating test data (OOD)...")
test_feat, test_tgt = generate_samples(N_TEST, ood=True)
test  = test_feat.copy()
test.insert(0, "id", range(N_TRAIN, N_TRAIN + N_TEST))

# Sample submission (zeros baseline)
target_cols = [f"t_{r}_{c}" for r in range(1, N_ROWS+1) for c in range(1, N_COLS+1)]
sample_sub  = pd.DataFrame(0.0, index=range(N_TEST), columns=["id"] + target_cols)
sample_sub["id"] = range(N_TRAIN, N_TRAIN + N_TEST)

Path("public").mkdir(exist_ok=True)
train.to_csv("public/train.csv", index=False)
test.to_csv("public/test.csv",  index=False)
sample_sub.to_csv("public/sample_submission.csv", index=False)

print(f"train.csv:             {train.shape}")
print(f"test.csv:              {test.shape}")
print(f"sample_submission.csv: {sample_sub.shape}")
print(f"\nPower stats (train):")
print(train[target_cols].describe().loc[["mean","std","min","max"]].to_string())