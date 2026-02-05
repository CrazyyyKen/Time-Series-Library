import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_aligned_peaks_dataset(
    minutes: int = 5,
    sample_rate_hz: int = 1000,
    start_time: str = "2026-02-04 00:00:00",
    freq_a_hz: float = 5.0,  # Sensor A frequency
    gain_b: float = 0.8,  # Sensor B amplitude scale
    gain_c: float = 0.6,  # Sensor C amplitude scale (same freq as A)
) -> pd.DataFrame:
    """
    Noise-free dataset:
      A(t) = sin(2π fA t)
      B(t) = gain_b * sin(2π (fA/2) t + phiB), with phiB chosen so:
             B's 1st peak aligns with A's 2nd peak
      C(t) = gain_c * A(t)
    """
    if minutes < 5:
        raise ValueError("minutes must be at least 5.")

    total_seconds = minutes * 60
    n = total_seconds * sample_rate_hz
    t = np.arange(n, dtype=np.float64) / sample_rate_hz

    date_index = pd.Timestamp(start_time) + pd.to_timedelta(t, unit="s")

    # Sensor A
    A = np.sin(2 * np.pi * freq_a_hz * t)

    # Sensor B frequency (half of A)
    freq_b_hz = freq_a_hz / 2.0

    # Align B's first peak with A's second peak:
    # For sine: peak when (2π f t + phi) = π/2 + 2πk
    # A second peak time: tA2 = (1/4 + 1)/fA = 5/(4 fA)
    tA2 = 5.0 / (4.0 * freq_a_hz)

    # Choose phiB so that B has a peak at tA2 (k=0 for "first peak"):
    # 2π fB tA2 + phiB = π/2
    phiB = (np.pi / 2.0) - (2.0 * np.pi * freq_b_hz * tA2)  # simplifies to -3π/4

    B = gain_b * np.sin(2 * np.pi * freq_b_hz * t + phiB)

    # Sensor C: same frequency as A, scaled
    C = gain_c * A

    return pd.DataFrame({"date": date_index, "1": A, "2": B, "3": C})


def save_sensor_plots(
    df: pd.DataFrame,
    sample_rate_hz: int,
    seconds_to_plot: float = 2.0,
    out_dir: str = "plots",
):
    os.makedirs(out_dir, exist_ok=True)
    n_plot = min(int(seconds_to_plot * sample_rate_hz), len(df))
    t = np.arange(n_plot, dtype=np.float64) / sample_rate_hz

    for col in ["1", "2", "3"]:
        plt.figure()
        plt.plot(t, df[col].iloc[:n_plot].to_numpy())
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title(f"Sensor {col} (first {seconds_to_plot} s)")
        plt.grid(True)
        plt.tight_layout()
        plt.axvline(0.25, color="red", linestyle="--")
        path = os.path.join(out_dir, f"sensor_{col}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    SAMPLE_RATE = 1000

    df = generate_aligned_peaks_dataset(
        minutes=5, sample_rate_hz=SAMPLE_RATE, freq_a_hz=5.0, gain_b=0.8, gain_c=0.6
    )

    print(df.head())
    print("Rows:", len(df))

    df.to_csv("iot_aligned_peaks.csv", index=False)
    print("Saved: iot_aligned_peaks.csv")

    save_sensor_plots(
        df, sample_rate_hz=SAMPLE_RATE, seconds_to_plot=2.0, out_dir="plots"
    )
