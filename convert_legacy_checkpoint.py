#!/usr/bin/env python3
"""
Convert legacy KL-Flow checkpoints to current model.safetensors format.

Legacy checkpoint format (expected):
- torch.load(path) -> dict with key "model" (state dict), or directly a state dict
- keys like:
    _orig_mod.transformer.wte.weight
    _orig_mod.transformer.h.0.attn.c_q.weight
    ...
    _orig_mod.lm_head.weight
    _orig_mod.t_embedder.mlp.0.weight

Current format (target):
- safetensors file with keys matching current FlowMatchingTransformer:
    token_emb.weight
    blocks.{i}.attn.q_proj.weight
    ...
    lm_head.weight
    time_embedder.mlp.0.weight
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List

import torch
from safetensors.torch import save_file


def _strip_prefixes(key: str) -> str:
    """Normalize old keys by removing common wrappers."""
    for prefix in ("_orig_mod.", "module."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _map_key_and_tensor(old_key: str, tensor: torch.Tensor) -> Tuple[str, torch.Tensor]:
    """
    Map one legacy key/tensor to new format.
    Returns (new_key, new_tensor). Raises KeyError when unmapped.
    """
    k = _strip_prefixes(old_key)

    # Token embedding: old embedding matrix -> new Linear weight (transpose required).
    if k == "transformer.wte.weight":
        return "token_emb.weight", tensor.transpose(0, 1).contiguous()

    # Output head is already [vocab, hidden] in both variants.
    if k == "lm_head.weight":
        return "lm_head.weight", tensor

    # Time embedder rename.
    if k.startswith("t_embedder.mlp."):
        return "time_embedder.mlp." + k[len("t_embedder.mlp."):], tensor

    # Transformer block attention / MLP.
    m = re.match(r"^transformer\.h\.(\d+)\.(attn|mlp)\.(.+)$", k)
    if not m:
        raise KeyError(f"Unmapped key: {old_key}")

    layer_idx = m.group(1)
    branch = m.group(2)
    tail = m.group(3)

    if branch == "attn":
        attn_map = {
            "c_q.weight": f"blocks.{layer_idx}.attn.q_proj.weight",
            "c_k.weight": f"blocks.{layer_idx}.attn.k_proj.weight",
            "c_v.weight": f"blocks.{layer_idx}.attn.v_proj.weight",
            "c_proj.weight": f"blocks.{layer_idx}.attn.out_proj.weight",
        }
        if tail in attn_map:
            return attn_map[tail], tensor
        raise KeyError(f"Unmapped attn key: {old_key}")

    # mlp
    mlp_map = {
        "c_fc.weight": f"blocks.{layer_idx}.ff.fc1.weight",
        "c_proj.weight": f"blocks.{layer_idx}.ff.fc2.weight",
    }
    if tail in mlp_map:
        return mlp_map[tail], tensor
    raise KeyError(f"Unmapped mlp key: {old_key}")


def convert_checkpoint(input_path: Path, output_path: Path, strict: bool = False) -> None:
    ckpt = torch.load(input_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        old_state = ckpt["model"]
    elif isinstance(ckpt, dict):
        old_state = ckpt
    else:
        raise ValueError("Unsupported checkpoint payload. Expected dict or dict with key 'model'.")

    new_state: Dict[str, torch.Tensor] = {}
    unmapped: List[str] = []

    for old_key, tensor in old_state.items():
        if not isinstance(tensor, torch.Tensor):
            # Skip non-parameter entries silently (rare but harmless).
            continue
        try:
            new_key, new_tensor = _map_key_and_tensor(old_key, tensor)
            new_state[new_key] = new_tensor
        except KeyError:
            unmapped.append(old_key)

    if not new_state:
        raise RuntimeError("No parameters were mapped. Check that this is the expected legacy checkpoint.")

    if strict and unmapped:
        preview = "\n".join(unmapped[:20])
        raise RuntimeError(
            f"Strict mode failed: {len(unmapped)} unmapped keys.\n"
            f"First unmapped keys:\n{preview}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(new_state, str(output_path))

    print(f"Converted checkpoint: {input_path}")
    print(f"Saved safetensors:   {output_path}")
    print(f"Mapped tensors:      {len(new_state)}")
    print(f"Unmapped tensors:    {len(unmapped)}")
    if unmapped:
        print("First unmapped keys:")
        for k in unmapped[:20]:
            print(f"  - {k}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert legacy KL-Flow checkpoint to current model.safetensors format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to legacy checkpoint (.pt/.pth) loaded via torch.load",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model.safetensors"),
        help="Output path for converted safetensors (default: ./model.safetensors)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any tensor key cannot be mapped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(args.input, args.output, strict=args.strict)


if __name__ == "__main__":
    main()
