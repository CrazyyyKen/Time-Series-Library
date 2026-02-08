import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1) Generate base dataset (A, B aligned, C scaled)
# ============================================================
def generate_aligned_peaks_dataset(
    minutes=5,
    sample_rate_hz=1000,
    start_time="2026-02-04 00:00:00",
    freq_a_hz=5.0,
    gain_b=0.8,
    gain_c=0.6,
):
    total_seconds = minutes * 60
    n = total_seconds * sample_rate_hz
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    date_index = pd.Timestamp(start_time) + pd.to_timedelta(t, unit="s")

    # Sensor A
    A = np.sin(2 * np.pi * freq_a_hz * t)

    # Sensor B (half frequency, aligned so first peak matches A's second peak)
    freq_b_hz = freq_a_hz / 2.0
    tA2 = 5.0 / (4.0 * freq_a_hz)
    phiB = (np.pi / 2.0) - (2.0 * np.pi * freq_b_hz * tA2)
    B = gain_b * np.sin(2 * np.pi * freq_b_hz * t + phiB)

    # Sensor C (same frequency as A)
    C = gain_c * A

    df = pd.DataFrame({"date": date_index, "1": A, "2": B, "3": C})
    return df


# ============================================================
# 2) Inject faults: make ~20% of timesteps faulty per sensor
#    Faults are obvious single-point anomalies.
# ============================================================
def inject_faults_by_percentage(
    df: pd.DataFrame,
    sensor_cols=("1", "2", "3"),
    fault_fraction: float = 0.20,
    fault_mode: str = "spike",  # "spike" or "flip"
    spike_value: float = 3.0,  # used for "spike"
    spike_jitter: float = 0.5,  # add +/- jitter to spike
    seed: int = 42,
):
    """
    For each sensor independently:
      - select fault_fraction of ALL indices
      - replace those values with an obvious fault

    fault_mode:
      - "spike": set value to ±(spike_value + jitter)
      - "flip" : multiply by -spike_value (sign flip + scale)
    """
    if not (0.0 < fault_fraction < 1.0):
        raise ValueError("fault_fraction must be between 0 and 1 (exclusive).")

    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    print("Total data points:", n)

    for col in sensor_cols:
        m = int(round(n * fault_fraction))
        fault_idx = rng.choice(n, size=m, replace=False)

        if fault_mode == "spike":
            # Randomly choose + or - spikes
            signs = rng.choice([-1.0, 1.0], size=m)
            jitter = rng.uniform(-spike_jitter, spike_jitter, size=m)
            out.loc[out.index[fault_idx], col] = signs * (spike_value + jitter)

        elif fault_mode == "flip":
            out.loc[out.index[fault_idx], col] *= -spike_value

        else:
            raise ValueError("fault_mode must be 'spike' or 'flip'.")

        print(f"Sensor {col}: faulty_points={m} " f"({100.0*m/n:.2f}% of timesteps)")

    return out


# ============================================================
# 3) Plot (non-GUI friendly)
# ============================================================
def save_sensor_plots(df, sample_rate_hz, seconds_to_plot=2.0, out_dir="plots_faulty"):
    os.makedirs(out_dir, exist_ok=True)
    n_plot = min(int(seconds_to_plot * sample_rate_hz), len(df))
    t = np.arange(n_plot) / sample_rate_hz

    for col in ["1", "2", "3"]:
        plt.figure()
        plt.plot(t, df[col].iloc[:n_plot].to_numpy())
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title(f"Sensor {col} (first {seconds_to_plot} s)")
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(out_dir, f"sensor_{col}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print("Saved plot:", path)


# ============================================================
# 4) Main
# ============================================================
if __name__ == "__main__":
    SAMPLE_RATE = 1000

    df = generate_aligned_peaks_dataset(
        minutes=5,
        sample_rate_hz=SAMPLE_RATE,
        freq_a_hz=5.0,
        gain_b=0.8,
        gain_c=0.6,
    )

    # Make 20% of timesteps faulty per sensor
    df_faulty = inject_faults_by_percentage(
        df,
        sensor_cols=("1", "2", "3"),
        fault_fraction=0.20,
        fault_mode="spike",  # "spike" makes it VERY obvious
        spike_value=3.0,
        spike_jitter=0.5,
        seed=42,
    )

    df_faulty.to_csv("iot_faulty_20_percent.csv", index=False)
    print("Saved: iot_faulty_20_percent.csv")

    save_sensor_plots(
        df_faulty,
        sample_rate_hz=SAMPLE_RATE,
        seconds_to_plot=2.0,
        out_dir="plots_faulty_20pct",
    )
