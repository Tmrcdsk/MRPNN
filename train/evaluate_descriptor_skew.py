import argparse
import os
import pathlib
from pathlib import Path

import numpy as np
import torch

from train_mrpnn import FEATURE_COLUMNS, MRPNN, load_checkpoint


if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath


def read_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != FEATURE_COLUMNS + 1:
        raise RuntimeError(f"{path} has {data.shape[1]} columns; expected {FEATURE_COLUMNS + 1}.")
    return data


def quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values, q).item())


def rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((a - b) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure MRPNN sensitivity to host/GPU descriptor skew.")
    parser.add_argument("--host", type=Path, default=Path("Data/skew_host.csv"))
    parser.add_argument("--gpu", type=Path, default=Path("Data/skew_gpu.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    host = read_csv(args.host)
    gpu = read_csv(args.gpu)
    if host.shape != gpu.shape:
        raise RuntimeError(f"Host/GPU CSV shapes differ: {host.shape} vs {gpu.shape}.")

    target_host = host[:, FEATURE_COLUMNS]
    target_gpu = gpu[:, FEATURE_COLUMNS]
    max_target_delta = float(np.max(np.abs(target_host - target_gpu)))
    if max_target_delta > 1e-6:
        raise RuntimeError(f"Paired CSV targets differ; max target delta is {max_target_delta}.")

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MRPNN().to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    host_x = torch.from_numpy(host[:, :FEATURE_COLUMNS]).to(device)
    gpu_x = torch.from_numpy(gpu[:, :FEATURE_COLUMNS]).to(device)
    target = torch.from_numpy(target_host).to(device)

    host_preds = []
    gpu_preds = []
    with torch.no_grad():
        for start in range(0, host_x.shape[0], args.batch_size):
            end = min(start + args.batch_size, host_x.shape[0])
            host_log = model(host_x[start:end])
            gpu_log = model(gpu_x[start:end])
            host_preds.append(model.decode_raw_radiance(host_x[start:end], host_log).detach().cpu())
            gpu_preds.append(model.decode_raw_radiance(gpu_x[start:end], gpu_log).detach().cpu())

    host_raw = torch.cat(host_preds).float()
    gpu_raw = torch.cat(gpu_preds).float()
    target_cpu = target.detach().cpu().float()

    abs_diff = torch.abs(host_raw - gpu_raw)
    rel_diff = abs_diff / torch.clamp(torch.abs(host_raw), min=1e-6)

    host_gpu_rmse = rmse(host_raw, gpu_raw)
    host_target_rmse = rmse(host_raw, target_cpu)
    gpu_target_rmse = rmse(gpu_raw, target_cpu)

    print(f"rows: {host.shape[0]}")
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print("")
    print("host vs gpu raw radiance:")
    print(f"  mean abs: {float(abs_diff.mean().item()):.8g}")
    print(f"  p95 abs : {quantile(abs_diff, 0.95):.8g}")
    print(f"  p99 abs : {quantile(abs_diff, 0.99):.8g}")
    print(f"  max abs : {float(abs_diff.max().item()):.8g}")
    print(f"  mean rel: {float(rel_diff.mean().item()):.8g}")
    print(f"  p95 rel : {quantile(rel_diff, 0.95):.8g}")
    print("")
    print("RMSE:")
    print(f"  host vs gpu   : {float(host_gpu_rmse.item()):.8g}")
    print(f"  host vs target: {float(host_target_rmse.item()):.8g}")
    print(f"  gpu  vs target: {float(gpu_target_rmse.item()):.8g}")
    print("")
    ratio = float(host_gpu_rmse.item() / max(host_target_rmse.item(), 1e-8))
    print(f"host_gpu_rmse / host_target_rmse: {ratio:.8g}")
    if ratio < 0.1 and abs(float(gpu_target_rmse.item() - host_target_rmse.item())) < 0.1 * max(float(host_target_rmse.item()), 1e-8):
        print("verdict: LOW sensitivity to host/GPU descriptor skew")
    else:
        print("verdict: NEEDS ATTENTION; descriptor skew measurably changes model output")


if __name__ == "__main__":
    main()
