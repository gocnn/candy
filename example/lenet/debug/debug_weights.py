import numpy as np
import sys
import os

def analyze_array(name, arr):
    print(f"{name}:")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  min/max: {arr.min():.6f} / {arr.max():.6f}")
    print(f"  mean/std: {arr.mean():.6f} / {arr.std():.6f}")
    print(f"  first few: {arr.flat[:5]}")
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_weights.py <file.npz|file.npy>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if path.endswith('.npz'):
        print("=== NPZ Analysis ===")
        data = np.load(path)
        for key in sorted(data.keys()):
            analyze_array(key, data[key])
    elif path.endswith('.npy'):
        print("=== NPY Analysis ===")
        arr = np.load(path)
        name = os.path.basename(path)
        analyze_array(name, arr)
    else:
        print("Error: File must be .npy or .npz")
        sys.exit(1)

if __name__ == "__main__":
    main()