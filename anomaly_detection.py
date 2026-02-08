import argparse
import os
import numpy as np
import pandas as pd


def load_pred_true(result_dir: str):
    pred_path = os.path.join(result_dir, "pred.npy")
    true_path = os.path.join(result_dir, "true.npy")
    if not os.path.isfile(pred_path) or not os.path.isfile(true_path):
        raise FileNotFoundError(f"Missing pred.npy/true.npy in {result_dir}")
    return np.load(pred_path), np.load(true_path)


def load_csv_data(path: str):
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    return df.values


def get_split_bounds(n: int, seq_len: int):
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_val = n - num_train - num_test
    border1s = [0, num_train - seq_len, n - num_test - seq_len]
    border2s = [num_train, num_train + num_val, n]
    return border1s, border2s


def build_test_pred_windows(data: np.ndarray, seq_len: int, pred_len: int):
    n = data.shape[0]
    border1s, border2s = get_split_bounds(n, seq_len)
    start = border1s[2]
    end = border2s[2]
    data_test = data[start:end]
    num_windows = len(data_test) - seq_len - pred_len + 1
    if num_windows <= 0:
        raise ValueError("Not enough data to build test windows.")
    out = np.empty((num_windows, pred_len, data.shape[1]), dtype=data.dtype)
    for i in range(num_windows):
        out[i] = data_test[i + seq_len : i + seq_len + pred_len]
    return out


def get_test_segment(data: np.ndarray, seq_len: int):
    n = data.shape[0]
    border1s, border2s = get_split_bounds(n, seq_len)
    start = border1s[2]
    end = border2s[2]
    return data[start:end]


def window_to_unique_mask(window_mask: np.ndarray, seq_len: int, pred_len: int, test_len: int):
    num_windows = window_mask.shape[0]
    num_features = window_mask.shape[2]
    unique = np.zeros((test_len, num_features), dtype=bool)
    for i in range(num_windows):
        s = i + seq_len
        e = s + pred_len
        unique[s:e] |= window_mask[i]
    return unique


def main():
    parser = argparse.ArgumentParser(description="Anomaly detection from forecast errors.")
    parser.add_argument("--normal_results", required=True, help="Results folder from normal dataset.")
    parser.add_argument("--normal_csv", required=True, help="Normal CSV path.")
    parser.add_argument("--anomaly_csv", required=True, help="Anomaly CSV path.")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length.")
    parser.add_argument("--pred_len", type=int, required=True, help="Prediction length.")
    parser.add_argument("--z", type=float, default=3.0, help="Z-score threshold from normal errors.")
    parser.add_argument("--label_delta", type=float, default=1e-6, help="Delta for labeling anomalies.")
    args = parser.parse_args()

    pred_n, true_n = load_pred_true(args.normal_results)
    err_n = np.abs(pred_n - true_n)
    mu = np.mean(err_n, axis=(0, 1))
    sigma = np.std(err_n, axis=(0, 1))
    threshold = mu + args.z * sigma

    normal_raw = load_csv_data(args.normal_csv)
    anomaly_raw = load_csv_data(args.anomaly_csv)
    if normal_raw.shape != anomaly_raw.shape:
        raise ValueError("Normal and anomaly CSVs must have matching shapes.")

    true_a_raw = build_test_pred_windows(anomaly_raw, args.seq_len, args.pred_len)
    if true_a_raw.shape != pred_n.shape:
        raise ValueError(
            f"CSV-derived windows {true_a_raw.shape} do not match results {pred_n.shape}."
        )

    err_a = np.abs(pred_n - true_a_raw)
    pred_anom = err_a > threshold

    true_n_raw = build_test_pred_windows(normal_raw, args.seq_len, args.pred_len)
    true_anom = np.abs(true_a_raw - true_n_raw) > args.label_delta

    acc_per_sensor = np.mean(pred_anom == true_anom, axis=(0, 1))
    acc_overall = np.mean(pred_anom == true_anom)
    tp = np.sum(pred_anom & true_anom, axis=(0, 1))
    tn = np.sum(~pred_anom & ~true_anom, axis=(0, 1))
    fp = np.sum(pred_anom & ~true_anom, axis=(0, 1))
    fn = np.sum(~pred_anom & true_anom, axis=(0, 1))
    total_true = np.sum(true_anom, axis=(0, 1))
    total_pred = np.sum(pred_anom, axis=(0, 1))

    print("window_eval")
    print(f"threshold_per_sensor: {threshold}")
    print(f"accuracy_per_sensor: {acc_per_sensor}")
    print(f"accuracy_overall: {acc_overall}")
    print(f"true_anomalies_per_sensor: {total_true}")
    print(f"pred_anomalies_per_sensor: {total_pred}")
    print(f"tp_per_sensor: {tp}")
    print(f"fp_per_sensor: {fp}")
    print(f"fn_per_sensor: {fn}")
    print(f"tn_per_sensor: {tn}")

    test_n = get_test_segment(normal_raw, args.seq_len)
    test_a = get_test_segment(anomaly_raw, args.seq_len)
    true_unique = np.abs(test_a - test_n) > args.label_delta
    test_len = test_n.shape[0]
    pred_unique = window_to_unique_mask(pred_anom, args.seq_len, args.pred_len, test_len)

    acc_u = np.mean(pred_unique == true_unique, axis=0)
    acc_u_overall = np.mean(pred_unique == true_unique)
    tp_u = np.sum(pred_unique & true_unique, axis=0)
    tn_u = np.sum(~pred_unique & ~true_unique, axis=0)
    fp_u = np.sum(pred_unique & ~true_unique, axis=0)
    fn_u = np.sum(~pred_unique & true_unique, axis=0)
    total_true_u = np.sum(true_unique, axis=0)
    total_pred_u = np.sum(pred_unique, axis=0)

    print("unique_timestep_eval")
    print(f"unique_accuracy_per_sensor: {acc_u}")
    print(f"unique_accuracy_overall: {acc_u_overall}")
    print(f"unique_true_anomalies_per_sensor: {total_true_u}")
    print(f"unique_pred_anomalies_per_sensor: {total_pred_u}")
    print(f"unique_tp_per_sensor: {tp_u}")
    print(f"unique_fp_per_sensor: {fp_u}")
    print(f"unique_fn_per_sensor: {fn_u}")
    print(f"unique_tn_per_sensor: {tn_u}")

    print("scenario_table")
    print("Scenario,Actual Anomaly Points,Detected Anomaly Points,Detection Accuracy")
    labels = ["Sensor A", "Sensor B", "Sensor C"]
    for i, label in enumerate(labels):
        print(f"Fault in {label},{total_true_u[i]},{total_pred_u[i]},{acc_u[i]:.6f}")


if __name__ == "__main__":
    main()
