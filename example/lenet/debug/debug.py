import argparse
import glob
import os
import sys
import numpy as np


def list_npy(d):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.npy")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default="artifacts/py_out")
    ap.add_argument("--go", default="artifacts/go_out")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--stop-first", action="store_true")
    args = ap.parse_args()

    py_names = set(list_npy(args.py))
    go_names = set(list_npy(args.go))

    if not py_names:
        print(f"No .npy files found in {args.py}")
    if not go_names:
        print(f"No .npy files found in {args.go}")

    missing_in_go = sorted(py_names - go_names)
    missing_in_py = sorted(go_names - py_names)
    if missing_in_go:
        print("Missing in go_out:", missing_in_go)
    if missing_in_py:
        print("Missing in py_out:", missing_in_py)

    names = sorted(py_names & go_names)
    fails = 0
    worst = (None, -1.0)

    for n in names:
        a = np.load(os.path.join(args.py, n))
        b = np.load(os.path.join(args.go, n))
        if a.shape != b.shape:
            print(f"{n}: shape mismatch {a.shape} vs {b.shape}")
            fails += 1
            if args.stop_first:
                break
            continue
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        ok = np.allclose(a, b, rtol=args.rtol, atol=args.atol)
        print(f"{n}: max_abs_diff={diff:.6g} {'OK' if ok else 'FAIL'}")
        if diff > worst[1]:
            worst = (n, diff)
        if not ok:
            fails += 1
            if args.stop_first:
                break

    if fails == 0:
        print("All compared layers matched within tolerance.")
        if worst[0] is not None:
            print(f"Worst layer {worst[0]} diff={worst[1]:.6g}")
        sys.exit(0)
    else:
        print(f"FAILED layers: {fails}")
        if worst[0] is not None:
            print(f"Worst layer {worst[0]} diff={worst[1]:.6g}")
        sys.exit(1)


if __name__ == "__main__":
    main()