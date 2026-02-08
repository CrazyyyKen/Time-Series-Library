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


def load_pred_true(result_dir: str):
    pred_path = os.path.join(result_dir, "pred.npy")
    true_path = os.path.join(result_dir, "true.npy")
    if not os.path.isfile(pred_path) or not os.path.isfile(true_path):
        raise FileNotFoundError(f"Missing pred.npy/true.npy in {result_dir}")
    return np.load(pred_path), np.load(true_path)


def anomaly_eval(normal_dir: str, anomaly_dir: str, z: float, label_delta: float):
    pred_n, true_n = load_pred_true(normal_dir)
    pred_a, true_a = load_pred_true(anomaly_dir)

    if pred_n.shape != pred_a.shape or true_n.shape != true_a.shape:
        raise ValueError("Normal and anomaly results must have matching shapes.")

    err_n = np.abs(pred_n - true_n)
    err_a = np.abs(pred_a - true_a)

    # per-sensor threshold from normal errors
    mu = np.mean(err_n, axis=(0, 1))
    sigma = np.std(err_n, axis=(0, 1))
    threshold = mu + z * sigma

    pred_anom = err_a > threshold
    true_anom = np.abs(true_a - true_n) > label_delta

    # accuracy per sensor and overall
    acc_per_sensor = np.mean(pred_anom == true_anom, axis=(0, 1))
    acc_overall = np.mean(pred_anom == true_anom)

    # counts
    tp = np.sum(pred_anom & true_anom, axis=(0, 1))
    tn = np.sum(~pred_anom & ~true_anom, axis=(0, 1))
    fp = np.sum(pred_anom & ~true_anom, axis=(0, 1))
    fn = np.sum(~pred_anom & true_anom, axis=(0, 1))
    total_true_anom = np.sum(true_anom, axis=(0, 1))
    total_pred_anom = np.sum(pred_anom, axis=(0, 1))

    print("anomaly_eval")
    print(f"normal_dir: {normal_dir}")
    print(f"anomaly_dir: {anomaly_dir}")
    print(f"z: {z}")
    print(f"label_delta: {label_delta}")
    print(f"threshold_per_sensor: {threshold}")
    print(f"accuracy_per_sensor: {acc_per_sensor}")
    print(f"accuracy_overall: {acc_overall}")
    print(f"true_anomalies_per_sensor: {total_true_anom}")
    print(f"pred_anomalies_per_sensor: {total_pred_anom}")
    print(f"tp_per_sensor: {tp}")
    print(f"fp_per_sensor: {fp}")
    print(f"fn_per_sensor: {fn}")
    print(f"tn_per_sensor: {tn}")


def main():
    parser = argparse.ArgumentParser(description="Summarize MAE/MSE or run anomaly evaluation.")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Only include result folders whose names start with this prefix.",
    )
    parser.add_argument(
        "--anomaly_eval",
        action="store_true",
        default=False,
        help="Run anomaly evaluation using normal/anomaly results folders.",
    )
    parser.add_argument("--normal_results", default=None, help="Results folder for normal data.")
    parser.add_argument("--anomaly_results", default=None, help="Results folder for anomaly data.")
    parser.add_argument("--z", type=float, default=3.0, help="Z-score threshold for anomalies.")
    parser.add_argument(
        "--label_delta",
        type=float,
        default=1e-6,
        help="Delta for labeling anomalies from true_anom vs true_norm.",
    )
    args = parser.parse_args()

    if not os.path.isdir(RESULTS_DIR):
        raise FileNotFoundError(f"Missing results directory: {RESULTS_DIR}")

    if args.anomaly_eval:
        if not args.normal_results or not args.anomaly_results:
            raise ValueError("--anomaly_eval requires --normal_results and --anomaly_results")
        anomaly_eval(args.normal_results, args.anomaly_results, args.z, args.label_delta)
        return

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
