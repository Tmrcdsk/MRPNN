import argparse
from pathlib import Path

import numpy as np


FEATURE_COLUMNS = 579
TOTAL_COLUMNS = 580


def read_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != TOTAL_COLUMNS:
        raise RuntimeError(f"{path} has shape {data.shape}, expected (*, {TOTAL_COLUMNS}).")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MRPNN binary .npy data.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=2e-6)
    args = parser.parse_args()

    features = np.load(args.features, mmap_mode="r")
    targets = np.load(args.targets, mmap_mode="r")

    if features.ndim != 2 or features.shape[1] != FEATURE_COLUMNS:
        raise RuntimeError(f"{args.features} has shape {features.shape}, expected (*, {FEATURE_COLUMNS}).")
    if targets.ndim != 1:
        raise RuntimeError(f"{args.targets} has shape {targets.shape}, expected 1D targets.")
    if features.shape[0] != targets.shape[0]:
        raise RuntimeError(f"Row mismatch: features={features.shape[0]}, targets={targets.shape[0]}.")
    if features.dtype != np.float32 or targets.dtype != np.float32:
        raise RuntimeError(f"Expected float32 arrays, got features={features.dtype}, targets={targets.dtype}.")

    print(f"features: shape={features.shape}, dtype={features.dtype}")
    print(f"targets : shape={targets.shape}, dtype={targets.dtype}")

    if args.csv is not None:
        csv_data = read_csv(args.csv)
        if csv_data.shape[0] != features.shape[0]:
            raise RuntimeError(f"CSV/binary row mismatch: csv={csv_data.shape[0]}, binary={features.shape[0]}.")

        feature_diff = np.abs(np.asarray(features) - csv_data[:, :FEATURE_COLUMNS])
        target_diff = np.abs(np.asarray(targets) - csv_data[:, FEATURE_COLUMNS])
        max_feature_diff = float(feature_diff.max(initial=0.0))
        max_target_diff = float(target_diff.max(initial=0.0))
        print(f"max feature diff: {max_feature_diff:.9g}")
        print(f"max target diff : {max_target_diff:.9g}")
        if max_feature_diff > args.atol or max_target_diff > args.atol:
            raise RuntimeError(
                f"Binary data differs from CSV beyond atol={args.atol}: "
                f"features={max_feature_diff}, targets={max_target_diff}."
            )

    print("result: PASS")


if __name__ == "__main__":
    main()
