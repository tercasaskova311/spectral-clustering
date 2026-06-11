#!/usr/bin/env python3
"""
generate_hpc_data.py
====================
Generates dummy square matrices with raw floating-point numbers for HPC scaling benchmarks.
Removes all complex RBF kernel computations to act as a fast data feeder.

Required Matrices:
- n = 500   (Weak scaling p=1)
- n = 1000  (Weak scaling p=2)
- n = 2000  (Strong scaling p=1..32, Weak scaling p=4)
- n = 4000  (Strong scaling benchmark 2, Weak scaling p=8)

Output Format: Space-separated square matrix, one row per line.
"""

import os
import argparse
import numpy as np

# ── Project Specific Sizes ────────────────────────────────────────────────────
TARGET_SIZES = [500, 700,1000, 1500, 1700, 2000, 4000]
SEED         = 42

def generate_and_save_matrix(n, path, rng):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w") as f:
        for _ in range(n):
            # Simulates realistic numbers (uniform values between 0.0 and 1.0)
            row = rng.random(size=n)
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
            
    file_mb = (n * n * 8) / (1024 * 1024)
    print(f"  --> Saved to {path} ({n}x{n}) [~{file_mb:.1f} MB]")

# ── Execution ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate arbitrary random matrices for HPC Benchmarks")
    parser.add_argument("--output-dir", default="data/input")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=== Generating Random Input Matrices ===")
    print(f"Destination: {args.output_dir}/\n")

    for n in TARGET_SIZES:
        print(f"Processing Matrix Size n = {n}...")
        filename = f"matrix_n{n}.txt"
        path = os.path.join(args.output_dir, filename)
        generate_and_save_matrix(n, path, rng)

    print("\nGeneration Complete!")
    print(f"  - Strong Data: Use 'matrix_n2000.txt' and 'matrix_n4000.txt'")
    print(f"  - Weak Data:   Map n=500(p1), n=1000(p2), n=2000(p4), n=4000(p8)")

if __name__ == "__main__":
    main()