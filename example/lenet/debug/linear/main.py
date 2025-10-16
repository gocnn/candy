import os
import numpy as np


def save_npy(path, arr):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, np.asarray(arr, dtype="<f4"), allow_pickle=False)


def main():
    here = os.path.dirname(__file__)
    base = os.path.normpath(os.path.join(here, ".."))
    py_out = os.path.join(base, "artifacts", "py_out")
    weights_path = os.path.join(base, "artifacts", "weights.npz")

    if not os.path.exists(py_out) or not os.path.exists(weights_path):
        raise SystemExit("Run lenet.py first to produce artifacts (py_out and weights.npz)")

    z = np.load(weights_path)
    # fc1: x in [1, 9216], w [128, 9216], b [128], y [1, 128]
    x1 = np.load(os.path.join(py_out, "40_flatten_out.npy")).astype("<f4")
    w1 = z["fc1.weight"].astype("<f4")
    b1 = z["fc1.bias"].astype("<f4")
    y1 = x1 @ w1.T + b1

    save_npy(os.path.join(here, "real_input_fc1.npy"), x1)
    save_npy(os.path.join(here, "real_weight_fc1.npy"), w1)
    save_npy(os.path.join(here, "real_bias_fc1.npy"), b1)
    save_npy(os.path.join(here, "real_output_fc1.npy"), y1)

    print("=== FC1 ===")
    print("x1:", x1.shape, "w1:", w1.shape, "b1:", b1.shape, "y1:", y1.shape)
    print("fc1 expected first_5:", y1.reshape(-1)[:5])

    # fc2: x in [1, 128] (post ReLU), w [10, 128], b [10], y [1, 10]
    x2 = np.load(os.path.join(py_out, "51_relu3_out.npy")).astype("<f4")
    w2 = z["fc2.weight"].astype("<f4")
    b2 = z["fc2.bias"].astype("<f4")
    y2 = x2 @ w2.T + b2

    save_npy(os.path.join(here, "real_input_fc2.npy"), x2)
    save_npy(os.path.join(here, "real_weight_fc2.npy"), w2)
    save_npy(os.path.join(here, "real_bias_fc2.npy"), b2)
    save_npy(os.path.join(here, "real_output_fc2.npy"), y2)

    print("=== FC2 ===")
    print("x2:", x2.shape, "w2:", w2.shape, "b2:", b2.shape, "y2:", y2.shape)
    print("fc2 expected first_5:", y2.reshape(-1)[:5])
    print("\nTest files saved in:", here)


if __name__ == "__main__":
    main()