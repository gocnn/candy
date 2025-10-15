from __future__ import annotations
import numpy as np


def main() -> None:
    # Write f32.npy
    A = np.array([[-1.5, 0.0, 3.14], [2.7, -0.8, 1.2]], dtype=np.float32)
    np.save("f32.npy", A)

    # Write f64.npy
    B = np.array([100.5, -50.25, 0.333], dtype=np.float64)
    np.save("f64.npy", B)

    # Write u8.npy
    C = np.array([255, 128, 0], dtype=np.uint8)
    np.save("u8.npy", C)

    # Write u32.npy
    D = np.array([1000000, 500, 42], dtype=np.uint32)
    np.save("u32.npy", D)

    # Write i64.npy
    E = np.array([-9876543210, 0, 1234567890], dtype=np.int64)
    np.save("i64.npy", E)

    # Write pack.npz
    F = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    G = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    np.savez("pack.npz", matrix_A=A, matrix_F=F, matrix_G=G)


if __name__ == "__main__":
    main()
