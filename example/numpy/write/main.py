from __future__ import annotations
from pathlib import Path
import numpy as np


def main() -> None:
    # Load and print f32.npy
    a = np.load("f32.npy")
    print(f"f32.npy: shape={a.shape}, dtype={a.dtype}")
    print(a)
    print()

    # Load and print f64.npy
    b = np.load("f64.npy")
    print(f"f64.npy: shape={b.shape}, dtype={b.dtype}")
    print(b)
    print()

    # Load and print u8.npy
    c = np.load("u8.npy")
    print(f"u8.npy: shape={c.shape}, dtype={c.dtype}")
    print(c)
    print()

    # Load and print u32.npy
    d = np.load("u32.npy")
    print(f"u32.npy: shape={d.shape}, dtype={d.dtype}")
    print(d)
    print()

    # Load and print i64.npy
    e = np.load("i64.npy")
    print(f"i64.npy: shape={e.shape}, dtype={e.dtype}")
    print(e)
    print()

    # Load and print pack.npz
    z = np.load("pack.npz")
    print(f"pack.npz contains keys: {z.files}")
    for k in z.files:
        arr = z[k]
        print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
        print(arr)
        print()


if __name__ == "__main__":
    main()
