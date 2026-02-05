import os
import argparse
import numpy as np

RESULTS_DIR = "results"


def compute_metrics(pred_path: str, true_path: str):
    pred = np.load(pred_path)
    true = np.load(true_path)
    diff = pred - true
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    mae_per_sensor = np.mean(np.abs(diff), axis=(0, 1))
    return mse, mae, mae_per_sensor


def find_result_folders(results_dir: str, prefix: str | None):
    folders = []
    for name in os.listdir(results_dir):
        if prefix and not name.startswith(prefix):
            continue
        path = os.path.join(results_dir, name)
        if not os.path.isdir(path):
            continue
        pred_path = os.path.join(path, "pred.npy")
        true_path = os.path.join(path, "true.npy")
        if os.path.isfile(pred_path) and os.path.isfile(true_path):
            folders.append((name, pred_path, true_path))
    return sorted(folders, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(description="Summarize MAE/MSE from results folders.")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Only include result folders whose names start with this prefix.",
    )
    args = parser.parse_args()

    if not os.path.isdir(RESULTS_DIR):
        raise FileNotFoundError(f"Missing results directory: {RESULTS_DIR}")

    folders = find_result_folders(RESULTS_DIR, args.prefix)
    if not folders:
        msg = f"No results with pred.npy/true.npy in {RESULTS_DIR}"
        if args.prefix:
            msg += f" matching prefix '{args.prefix}'"
        raise FileNotFoundError(msg)

    print("folder,mse,mae,mae_s1,mae_s2,mae_s3")
    for name, pred_path, true_path in folders:
        mse, mae, mae_per_sensor = compute_metrics(pred_path, true_path)
        mae_s = ",".join(f"{v:.6f}" for v in mae_per_sensor)
        print(f"{name},{mse:.6f},{mae:.6f},{mae_s}")


if __name__ == "__main__":
    main()
