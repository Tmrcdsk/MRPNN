import argparse
import pathlib
import re
from pathlib import Path

import torch


LOW_BLOCK_TAGS = ["01", "11", "21", "31", "41", "51", "61", "71"]
HIGH_BLOCK_TAGS = ["01", "11", "21", "31"]
LOW_SE_NAMES = [("LSE01W", "LSE02W"), ("LSE11W", "LSE12W"), ("LSE21W", "LSE22W"), ("LSE31W", "LSE32W"), ("LSE41W", "LSE42W"), ("LSE51W", "LSE52W"), ("LSE61W", "LSE62W"), ("LSE71W", "LSE72W")]
HIGH_SE_NAMES = [("LDSE01W", "LDSE02W"), ("LDSE11W", "LDSE12W"), ("LDSE21W", "LDSE22W"), ("LDSE31W", "LDSE32W")]

DECL_RE = re.compile(r'^(?P<prefix>__device__(?: __constant__| const) float )(?P<name>\w+)\[(?P<size>\d+)\] = \{$')


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().contiguous().view(-1).tolist()


def load_checkpoint_portable(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu")
    except NotImplementedError as exc:
        message = str(exc)
        replacement: tuple[str, type[pathlib.Path], type[pathlib.Path]] | None = None
        if "PosixPath" in message:
            replacement = ("PosixPath", pathlib.PosixPath, pathlib.WindowsPath)
        elif "WindowsPath" in message:
            replacement = ("WindowsPath", pathlib.WindowsPath, pathlib.PosixPath)
        if replacement is None:
            raise
        _, original_cls, compatible_cls = replacement
        setattr(pathlib, replacement[0], compatible_cls)
        try:
            return torch.load(path, map_location="cpu")
        finally:
            setattr(pathlib, replacement[0], original_cls)


def build_symbol_map(state: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}

    for idx, tag in enumerate(HIGH_BLOCK_TAGS):
        out[f"LD{tag}W"] = tensor_to_list(state[f"high_density.{idx}.weight"])
        out[f"LD{tag}B"] = tensor_to_list(state[f"high_density.{idx}.bias"])
        out[f"LD_Tr{tag}W"] = tensor_to_list(state[f"high_transmittance.{idx}.weight"])
        out[f"LD_Tr{tag}B"] = tensor_to_list(state[f"high_transmittance.{idx}.bias"])
        out[f"LD_Hg{tag}W"] = tensor_to_list(state[f"high_phase.{idx}.weight"])
        out[f"LD_Hg{tag}B"] = tensor_to_list(state[f"high_phase.{idx}.bias"])
        se_w1, se_w2 = HIGH_SE_NAMES[idx]
        out[se_w1] = tensor_to_list(state[f"high_se.{idx}.fc1.weight"])
        out[se_w2] = tensor_to_list(state[f"high_se.{idx}.fc2.weight"])

    out["LD41W"] = tensor_to_list(state["high_head_density.weight"])
    out["LD41B"] = tensor_to_list(state["high_head_density.bias"])
    out["LD_Tr41W"] = tensor_to_list(state["high_head_transmittance.weight"])
    out["LD_Tr41B"] = tensor_to_list(state["high_head_transmittance.bias"])
    out["LD_Hg41W"] = tensor_to_list(state["high_head_phase.weight"])
    out["LD_Hg41B"] = tensor_to_list(state["high_head_phase.bias"])

    for idx, tag in enumerate(LOW_BLOCK_TAGS):
        out[f"L{tag}W"] = tensor_to_list(state[f"low_density.{idx}.weight"])
        out[f"L{tag}B"] = tensor_to_list(state[f"low_density.{idx}.bias"])
        out[f"L_Tr{tag}W"] = tensor_to_list(state[f"low_transmittance.{idx}.weight"])
        out[f"L_Tr{tag}B"] = tensor_to_list(state[f"low_transmittance.{idx}.bias"])
        out[f"L_Hg{tag}W"] = tensor_to_list(state[f"low_phase.{idx}.weight"])
        out[f"L_Hg{tag}B"] = tensor_to_list(state[f"low_phase.{idx}.bias"])
        se_w1, se_w2 = LOW_SE_NAMES[idx]
        out[se_w1] = tensor_to_list(state[f"low_se.{idx}.fc1.weight"])
        out[se_w2] = tensor_to_list(state[f"low_se.{idx}.fc2.weight"])

    out["L81W"] = tensor_to_list(state["low_head_density.weight"])
    out["L81B"] = tensor_to_list(state["low_head_density.bias"])
    out["L_Tr81W"] = tensor_to_list(state["low_head_transmittance.weight"])
    out["L_Tr81B"] = tensor_to_list(state["low_head_transmittance.bias"])
    out["L_Hg81W"] = tensor_to_list(state["low_head_phase.weight"])
    out["L_Hg81B"] = tensor_to_list(state["low_head_phase.bias"])

    out["LSEFin1W"] = tensor_to_list(state["global_se.fc1.weight"])
    out["LSEFin2W"] = tensor_to_list(state["global_se.fc2.weight"])
    out["LGGSW"] = tensor_to_list(state["global_embed.weight"])
    out["LGGSB"] = tensor_to_list(state["global_embed.bias"])

    out["LC0W"] = tensor_to_list(state["lc0.weight"])
    out["LC0B"] = tensor_to_list(state["lc0.bias"])
    out["LC1W"] = tensor_to_list(state["lc1.weight"])
    out["LC1B"] = tensor_to_list(state["lc1.bias"])
    out["LC2W"] = tensor_to_list(state["lc2.weight"])
    out["LC2B"] = tensor_to_list(state["lc2.bias"])
    out["LXW"] = tensor_to_list(state["lx.weight"])
    out["LXB"] = tensor_to_list(state["lx.bias"])

    for idx in range(6):
        out[f"LX{2 * idx}W"] = tensor_to_list(state[f"residual_blocks.{idx}.0.weight"])
        out[f"LX{2 * idx}B"] = tensor_to_list(state[f"residual_blocks.{idx}.0.bias"])
        out[f"LX{2 * idx + 1}W"] = tensor_to_list(state[f"residual_blocks.{idx}.1.weight"])
        out[f"LX{2 * idx + 1}B"] = tensor_to_list(state[f"residual_blocks.{idx}.1.bias"])

    out["LX12W"] = tensor_to_list(state["output_layer.weight"])
    out["LX12B"] = tensor_to_list(state["output_layer.bias"])
    return out


def format_values(values: list[float], per_line: int = 8) -> str:
    chunks = []
    for start in range(0, len(values), per_line):
        part = values[start:start + per_line]
        line = ",".join(format(float(v), ".9g") for v in part)
        if start + per_line < len(values):
            line += ","
        chunks.append(line)
    return "\n".join(chunks)


def render_header(template_path: Path, symbol_map: dict[str, list[float]], checkpoint_path: Path) -> str:
    template_lines = template_path.read_text(encoding="utf-8").splitlines()
    output_lines = ["#pragma once", f"// Generated from {checkpoint_path.as_posix()}"]

    for line in template_lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        match = DECL_RE.match(stripped)
        if not match:
            continue
        prefix = match.group("prefix")
        name = match.group("name")
        expected_size = int(match.group("size"))
        if name not in symbol_map:
            raise KeyError(f"No exported tensor for symbol {name}")
        values = symbol_map[name]
        if len(values) != expected_size:
            raise ValueError(f"{name}: expected {expected_size} values, got {len(values)}")
        output_lines.append(f"{prefix}{name}[{expected_size}] = {{")
        output_lines.append(format_values(values))
        output_lines.append("};")

    return "\n".join(output_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained MRPNN checkpoint to core/NNWeight.cuh format.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path("core/NNWeight.cuh"))
    parser.add_argument("--output", type=Path, default=Path("core/NNWeight.cuh"))
    args = parser.parse_args()

    checkpoint = load_checkpoint_portable(args.checkpoint)
    state = checkpoint["model_state"]
    symbol_map = build_symbol_map(state)
    rendered = render_header(args.template, symbol_map, args.checkpoint)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output} from {args.checkpoint}")


if __name__ == "__main__":
    main()
