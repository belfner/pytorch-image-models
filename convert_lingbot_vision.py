#!/usr/bin/env python
"""Convert LingBot-Vision checkpoints to timm-native weights with a full parity gate.

Per-variant pipeline: download the pinned source checkpoint and verify its SHA-256,
unwrap the state dict via a strict fail-closed allowlist, map it through timm's existing
``eva.checkpoint_filter_fn``, require exact key/shape equality and a strict load, run
structural asserts (per-block Q/V bias against the source slices, zero-K verification on
both source and target, giant bias absence), prove fp32 numerical parity against the
reference implementation (CLS / register / patch partitions plus the token-pooled public
forward and a default-average-pool smoke, at 512x512 and 384x512 and on a real image
through both preprocessing paths), serialize with ``save_for_hf(safe_serialization='both')``
into a fresh directory, strict-reload the serialized safetensors, hash the exact payload,
and write a JSON manifest with source, converter, reference, and environment provenance.

Uploading is a separate step: run ``upload_lingbot_vision.py`` after all four variants
convert, and ``verify_lingbot_vision.py`` for clean-cache checks of the uploaded repos.

Run from the repo root with the ``lingbot-vision`` submodule initialized
(``git submodule update --init``): ``python convert_lingbot_vision.py --variants small``.
Requires this timm checkout's environment plus ``omegaconf``, ``Pillow``, ``safetensors``,
and ``huggingface_hub`` for the reference implementation and serialization. A different
timm checkout can be selected via ``LINGBOT_TIMM_ROOT``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

TIMM_ROOT = Path(os.environ.get("LINGBOT_TIMM_ROOT", str(Path(__file__).resolve().parent)))
sys.path.insert(0, str(TIMM_ROOT))
sys.path.insert(0, str(TIMM_ROOT / "lingbot-vision"))

import timm
from timm.data import create_transform, resolve_data_config
from timm.models._hub import save_for_hf
from timm.models.eva import checkpoint_filter_fn

from lingbot_vision.loader import load_pretrained_backbone
from lingbot_vision.preprocess import load_image

from lingbot_common import (
    ATOL,
    PAYLOAD_FILES,
    PRETRAINED_TAG,
    RTOL,
    SPECS,
    WRAPPER_ALLOWLIST,
    save_dir_for,
    sha256_file,
    validate_payload,
)

REF_ROOT = TIMM_ROOT / "lingbot-vision"
REAL_IMAGE = REF_ROOT / "assets" / "lingbot_depth2_mirror_glass.jpg"


def strip_uniform_prefix(sd: dict, prefix: str) -> dict:
    """Strip a leading prefix that must cover all keys or none, preserving uniqueness.

    Args:
      sd (dict): State dict to normalize.
      prefix (str): Leading prefix to remove.

    Returns:
      dict: State dict with the prefix removed from every key, or the input unchanged
        when the prefix appears on zero keys.

    Raises:
      ValueError: The prefix covers only part of the keys, still appears mid-key after
        stripping, or stripping produces duplicate keys.
    """
    prefixed = [k for k in sd if k.startswith(prefix)]
    if len(prefixed) == 0:
        leftovers = [k for k in sd if prefix in k]
        if len(leftovers) > 0:
            raise ValueError(f"'{prefix}' appears mid-key in {leftovers[:3]}")
        return sd
    if len(prefixed) != len(sd):
        raise ValueError(f"mixed layout: {len(prefixed)} of {len(sd)} keys carry the '{prefix}' prefix")
    new_keys = [k[len(prefix):] for k in sd]
    if len(set(new_keys)) != len(new_keys):
        raise ValueError(f"stripping '{prefix}' produces duplicate keys")
    leftovers = [k for k in new_keys if prefix in k]
    if len(leftovers) > 0:
        raise ValueError(f"'{prefix}' still present after stripping: {leftovers[:3]}")
    return dict(zip(new_keys, sd.values()))


def unwrap_state_dict(ckpt: dict) -> tuple[dict, str]:
    """Unwrap a source checkpoint into a flat backbone state dict.

    Requires exactly one allowlisted wrapper key at the top level, then strips a uniform
    ``_orig_mod.`` compile prefix and a uniform ``backbone.`` prefix, rejecting mixed or
    colliding layouts.

    Args:
      ckpt (dict): Raw object returned by ``torch.load``.

    Returns:
      tuple[dict, str]: The flat state dict and the wrapper key that held it.

    Raises:
      ValueError: The top level held zero or multiple allowlisted wrapper keys, or a
        prefix normalization failed its uniformity/uniqueness checks.
    """
    present = [k for k in WRAPPER_ALLOWLIST if k in ckpt and isinstance(ckpt[k], dict)]
    if len(present) != 1:
        raise ValueError(f"expected exactly one wrapper key from {WRAPPER_ALLOWLIST}, found {present}")
    wrapper = present[0]
    sd = strip_uniform_prefix(dict(ckpt[wrapper]), "_orig_mod.")
    sd = strip_uniform_prefix(sd, "backbone.")
    return sd, wrapper


def assert_qkv_bias_structure(variant: str, model: torch.nn.Module, sd: dict) -> None:
    """Assert per-block QKV bias structure against the source state dict.

    For S/B/L: every block carries Q and V bias parameters whose values equal the
    corresponding source ``qkv.bias`` slices, every source K slice is exactly zero, and
    every target K buffer exists and is exactly zero. For giant: the source has no
    ``qkv.bias`` keys and no block carries Q, K, or V bias attributes.

    Args:
      variant (str): One of small, base, large, giant.
      model (torch.nn.Module): Converted timm model.
      sd (dict): Unwrapped source state dict.

    Raises:
      AssertionError: Any block or source tensor violates the expected structure.
    """
    spec = SPECS[variant]
    if spec["qkv_bias"]:
        for i, blk in enumerate(model.blocks):
            src = sd[f"blocks.{i}.attn.qkv.bias"]
            q, k, v = src.chunk(3)
            assert blk.attn.q_bias is not None and blk.attn.v_bias is not None, f"block {i} lacks q/v bias"
            assert torch.count_nonzero(k) == 0, f"block {i} source K bias is nonzero"
            assert torch.equal(blk.attn.q_bias.data, q), f"block {i} q_bias differs from source slice"
            assert torch.equal(blk.attn.v_bias.data, v), f"block {i} v_bias differs from source slice"
            k_bias = getattr(blk.attn, "k_bias", None)
            assert k_bias is not None, f"block {i} lacks the zero K bias buffer"
            assert torch.count_nonzero(k_bias) == 0, f"block {i} target K bias buffer is nonzero"
    else:
        src_bias_keys = [k for k in sd if k.endswith("attn.qkv.bias")]
        assert len(src_bias_keys) == 0, f"unexpected source qkv bias keys: {src_bias_keys[:3]}"
        for i, blk in enumerate(model.blocks):
            for attr in ("q_bias", "k_bias", "v_bias"):
                assert getattr(blk.attn, attr, None) is None, f"block {i} unexpectedly has {attr}"


def convert_variant(variant: str) -> tuple[torch.nn.Module, dict]:
    """Download, verify, unwrap, filter, and strictly load one variant into timm.

    Args:
      variant (str): One of small, base, large, giant.

    Returns:
      tuple[torch.nn.Module, dict]: The loaded eval-mode timm model (constructed with
        its factory defaults, average pooling included) and a manifest fragment for the
        conversion stage.

    Raises:
      ValueError: Source hash mismatch or filtered/target key/shape set mismatch.
      AssertionError: A structural assert failed.
    """
    spec = SPECS[variant]
    ckpt_path = Path(hf_hub_download(spec["repo_id"], "model.pt", revision=spec["revision"]))
    src_hash = sha256_file(ckpt_path)
    if src_hash != spec["sha256"]:
        raise ValueError(f"{variant}: source hash {src_hash} does not match pinned {spec['sha256']}")
    print(f"[{variant}] source verified: {ckpt_path} sha256={src_hash[:16]}...")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd, wrapper = unwrap_state_dict(ckpt)
    print(f"[{variant}] unwrapped via key '{wrapper}': {len(sd)} tensors")

    model = timm.create_model(spec["arch"], pretrained=False)
    filtered = checkpoint_filter_fn(sd, model)
    target = model.state_dict()
    f_keys, t_keys = set(filtered), set(target)
    if f_keys != t_keys:
        raise ValueError(
            f"{variant}: key set mismatch; missing={sorted(t_keys - f_keys)[:5]} unexpected={sorted(f_keys - t_keys)[:5]}"
        )
    shape_mismatch = [k for k in t_keys if tuple(filtered[k].shape) != tuple(target[k].shape)]
    if len(shape_mismatch) > 0:
        raise ValueError(f"{variant}: shape mismatch on {shape_mismatch[:5]}")
    model.load_state_dict(filtered, strict=True)
    model.eval()

    blk = model.blocks[0]
    assert len(model.blocks) == spec["depth"], f"depth {len(model.blocks)} != {spec['depth']}"
    assert model.embed_dim == spec["embed_dim"], f"embed_dim {model.embed_dim} != {spec['embed_dim']}"
    assert model.reg_token is not None and model.reg_token.shape == (1, 4, spec["embed_dim"])
    assert model.num_prefix_tokens == 5, f"num_prefix_tokens {model.num_prefix_tokens} != 5"
    assert model.global_pool == "avg", f"expected factory default avg pooling, got {model.global_pool}"
    assert_qkv_bias_structure(variant, model, sd)
    if spec["ffn_hidden"] is not None:
        assert blk.mlp.fc1_g.out_features == spec["ffn_hidden"], (
            f"SwiGLU hidden {blk.mlp.fc1_g.out_features} != {spec['ffn_hidden']}"
        )

    dtypes = sorted({str(v.dtype) for v in filtered.values()})
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{variant}] strict load ok: {n_params / 1e6:.1f}M params, dtypes={dtypes}")
    return model, {
        "source_repo": spec["repo_id"], "source_revision": spec["revision"], "source_sha256": src_hash,
        "wrapper_key": wrapper, "source_tensors": len(sd), "converted_tensors": len(filtered),
        "param_count": n_params, "dtypes": dtypes,
    }


def compare_partitions(ref_out: dict, timm_seq: torch.Tensor, label: str) -> dict:
    """Compare CLS / register / patch partitions between reference and timm outputs.

    Args:
      ref_out (dict): Reference forward output with ``x_norm_clstoken``,
        ``x_storage_tokens``, ``x_norm_patchtokens``.
      timm_seq (torch.Tensor): timm ``forward_features`` token sequence [B, N, C].
      label (str): Input label for reporting.

    Returns:
      dict: Per-partition max abs diff, mean abs diff, and cosine similarity.
    """
    parts = {
        "cls": (ref_out["x_norm_clstoken"].unsqueeze(1), timm_seq[:, :1]),
        "registers": (ref_out["x_storage_tokens"], timm_seq[:, 1:5]),
        "patches": (ref_out["x_norm_patchtokens"], timm_seq[:, 5:]),
    }
    metrics = {}
    for name, (ref_t, timm_t) in parts.items():
        torch.testing.assert_close(timm_t, ref_t, rtol=RTOL, atol=ATOL)
        diff = (timm_t - ref_t).abs()
        cos = torch.nn.functional.cosine_similarity(
            timm_t.reshape(-1, timm_t.shape[-1]), ref_t.reshape(-1, ref_t.shape[-1]), dim=-1
        )
        metrics[name] = {
            "max_abs": diff.max().item(), "mean_abs": diff.mean().item(), "cos_min": cos.min().item(),
        }
        print(f"    {label}/{name}: max_abs={metrics[name]['max_abs']:.3e} cos_min={metrics[name]['cos_min']:.6f}")
    return metrics


def parity_gate(variant: str, model: torch.nn.Module) -> dict:
    """Run the fp32 parity gate for one variant against the reference implementation.

    The token-pooled public forward temporarily switches ``model.global_pool`` (Eva
    pooling dispatches on it at call time) and restores the average default afterward;
    a default-average forward smoke runs on the restored model.

    Args:
      variant (str): One of small, base, large, giant.
      model (torch.nn.Module): Converted timm model with factory-default pooling.

    Returns:
      dict: Parity metrics per input and partition.

    Raises:
      AssertionError: Any partition or the public forward exceeded tolerances, or
        the resolved timm transform diverged from reference preprocessing.
    """
    spec = SPECS[variant]
    ref, _ = load_pretrained_backbone(
        variant=variant, device="cpu", dtype="fp32", revision=spec["revision"], verbose=False,
    )
    metrics = {}
    g = torch.Generator().manual_seed(42)
    with torch.inference_mode():
        for label, hw in (("512x512", (512, 512)), ("384x512", (384, 512))):
            x = torch.randn(1, 3, *hw, generator=g, dtype=torch.float32)
            ref_out = ref(x, is_training=True)
            timm_seq = model.forward_features(x)
            metrics[label] = compare_partitions(ref_out, timm_seq, label)

        x = torch.randn(1, 3, 512, 512, generator=g, dtype=torch.float32)
        ref_cls = ref(x)
        assert model.global_pool == "avg"
        model.global_pool = "token"
        try:
            timm_pooled = model(x)
        finally:
            model.global_pool = "avg"
        torch.testing.assert_close(timm_pooled, ref_cls, rtol=RTOL, atol=ATOL)
        pub_diff = (timm_pooled - ref_cls).abs().max().item()
        metrics["public_forward_token_pool"] = {"max_abs": pub_diff}
        print(f"    public forward (token pool): max_abs={pub_diff:.3e}")

        avg_out = model(x)
        assert avg_out.shape == (1, spec["embed_dim"]) and torch.isfinite(avg_out).all()
        metrics["public_forward_avg_pool_smoke"] = "pass"
        print("    public forward (default avg pool): smoke pass")

        img_norm, _, _ = load_image(str(REAL_IMAGE), size=512, mode="square")
        data_cfg = resolve_data_config(model=model)
        transform = create_transform(**data_cfg, is_training=False)
        from PIL import Image
        timm_in = transform(Image.open(REAL_IMAGE).convert("RGB")).unsqueeze(0)
        torch.testing.assert_close(timm_in, img_norm, rtol=0.0, atol=1e-6)
        metrics["transform_max_abs"] = (timm_in - img_norm).abs().max().item()
        print(f"    transform parity: max_abs={metrics['transform_max_abs']:.3e}")
        ref_out = ref(img_norm, is_training=True)
        timm_seq = model.forward_features(timm_in)
        metrics["real_image"] = compare_partitions(ref_out, timm_seq, "real_image")
    del ref
    return metrics


def serialize_and_reload(variant: str, model: torch.nn.Module, out_dir: Path, overwrite: bool) -> Path:
    """Serialize the converted model for HF into a fresh directory and strict-reload it.

    Args:
      variant (str): One of small, base, large, giant.
      model (torch.nn.Module): Converted, parity-verified timm model.
      out_dir (Path): Root output directory; the save dir is ``<arch>.<tag>`` beneath it.
      overwrite (bool): Remove a pre-existing save directory before writing.

    Returns:
      Path: The save directory.

    Raises:
      ValueError: The save directory already exists and ``overwrite`` is False.
      AssertionError: A serialized tensor differs from the in-memory state.
    """
    import safetensors.torch

    spec = SPECS[variant]
    save_dir = save_dir_for(variant, out_dir)
    if save_dir.exists():
        if not overwrite:
            raise ValueError(f"{save_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(save_dir)
    sd = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(sd, strict=True)
    save_for_hf(model, str(save_dir), safe_serialization="both")

    reload_sd = safetensors.torch.load_file(save_dir / "model.safetensors")
    fresh = timm.create_model(spec["arch"], pretrained=False)
    fresh.load_state_dict(reload_sd, strict=True)
    for k, v in fresh.state_dict().items():
        assert torch.equal(v, sd[k]), f"serialized tensor mismatch: {k}"
    with open(save_dir / "config.json") as f:
        saved_cfg = json.load(f)
    assert saved_cfg.get("global_pool", "avg") == "avg", f"config.json global_pool: {saved_cfg.get('global_pool')}"
    del fresh, reload_sd
    print(f"[{variant}] serialized to {save_dir} and strict-reloaded safetensors")
    return save_dir


def write_model_card(variant: str, save_dir: Path, parity: dict) -> None:
    """Write the staging repo model card following the standard timm card layout.

    Args:
      variant (str): One of small, base, large, giant.
      save_dir (Path): Directory holding the serialized artifacts.
      parity (dict): Parity metrics from :func:`parity_gate`.
    """
    spec = SPECS[variant]
    arch = spec["arch"]
    name = f"{arch}.{PRETRAINED_TAG}"
    model_id = name
    p512 = parity["512x512"]
    max_abs = max(p512[part]["max_abs"] for part in ("cls", "registers", "patches"))
    if variant == "giant":
        trained = "Pretrained with masked boundary modeling by the paper authors"
    else:
        trained = "Distilled from the masked-boundary-pretrained ViT-Giant/16 teacher by the paper authors"
    head = f"""---
tags:
- image-feature-extraction
- timm
- transformers
pipeline_tag: image-feature-extraction
library_name: timm
license: apache-2.0
---
# Model card for {name}

A LingBot-Vision ViT-{variant.capitalize()}/16 image feature encoder. {trained}
and converted to timm's Eva/DINOv3 implementation.

## Model Notes
* Token layout: CLS at index 0, four register tokens at indices 1-4, patch tokens thereafter.
  The pretrained cfg uses `global_pool='avg'` over patch tokens; pass `global_pool='token'` at
  creation to reproduce the upstream CLS representation.
* fp32 forward outputs match the reference implementation with max abs diff {max_abs:.1e} on CLS,
  register, and patch tokens at 512x512 and 384x512. Conversion provenance, the pinned source
  revision, and per-partition parity metrics are recorded in `manifest.json`.
* Converted from https://huggingface.co/{spec['repo_id']} at revision `{spec['revision'][:12]}`.
  The `{model_id}` architecture is pending in timm (PR); its pretrained cfg resolves the weights
  from this repo, so the usage below works on a timm checkout that includes the LingBot entrypoints.

## Model Details
- **Model Type:** Image Feature Encoder
- **Model Stats:**
  - Params (M): {spec['params_m']}
  - GMACs: {spec['gmacs']}
  - Activations (M): {spec['macts']}
  - Image size: 512 x 512
- **Original:** https://github.com/robbyant/lingbot-vision
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Pretrain Dataset:** 161M-image curated web corpus (see paper)
- **Papers:**
  - Vision Pretraining for Dense Spatial Perception: https://arxiv.org/abs/2607.05247
  - PyTorch Image Models: https://github.com/huggingface/pytorch-image-models

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm
import torch

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('{model_id}', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Feature Map Extraction
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    '{model_id}',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

for o in output:
    # print shape of each feature map in output
    print(o.shape)
```

### Image Embeddings
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    '{model_id}',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)
output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 1029, {spec['embed_dim']}) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).

## Citation
"""
    citation = """```bibtex
@article{lingbot-vision2026,
  title={Vision Pretraining for Dense Spatial Perception},
  author={Fu, Zelin and Tan, Bin and Sun, Changjiang and Liu, Shaohui and Zheng, Kecheng and Xu, Yinghao and Zhu, Xing and Shen, Yujun and Xue, Nan},
  journal={arXiv preprint arXiv:2607.05247},
  year={2026}
}
```
```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\\url{https://github.com/huggingface/pytorch-image-models}}
}
```
"""
    (save_dir / "README.md").write_text(head + citation)


def collect_provenance() -> dict:
    """Collect converter, reference checkout, and environment provenance for the manifest.

    Returns:
      dict: Converter hash, timm/reference commits and dirty state, dependency versions.
    """
    import huggingface_hub
    import safetensors

    def git(root: Path, *argv: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *argv], capture_output=True, text=True, check=True,
        ).stdout.strip()

    return {
        "converter_sha256": sha256_file(Path(__file__)),
        "timm_root": str(TIMM_ROOT),
        "timm_commit": git(TIMM_ROOT, "rev-parse", "HEAD"),
        "timm_dirty": len(git(TIMM_ROOT, "status", "--porcelain", "--untracked-files=no")) > 0,
        "reference_root": str(REF_ROOT),
        "reference_commit": git(REF_ROOT, "rev-parse", "HEAD"),
        "reference_dirty": len(git(REF_ROOT, "status", "--porcelain", "--untracked-files=no")) > 0,
        "torch": torch.__version__,
        "timm_version": timm.__version__,
        "safetensors": safetensors.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "python": platform.python_version(),
    }


def main() -> None:
    """Run conversion, parity gate, and serialization per variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(SPECS), default=["small"])
    parser.add_argument("--output-dir", type=Path, default=TIMM_ROOT / "converted")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing per-variant output directories.")
    parser.add_argument(
        "--refresh-cards", action="store_true",
        help="Rewrite README.md from each existing manifest's parity metrics, verify weight/config hashes, "
             "and update the manifest payload hashes. Skips conversion; generation provenance is preserved.",
    )
    args = parser.parse_args()

    if args.refresh_cards:
        for variant in args.variants:
            save_dir = save_dir_for(variant, args.output_dir)
            with open(save_dir / "manifest.json") as f:
                manifest = json.load(f)
            for name in ("config.json", "model.safetensors", "pytorch_model.bin"):
                actual = sha256_file(save_dir / name)
                if actual != manifest["artifacts"][name]:
                    raise ValueError(f"{save_dir / name}: hash {actual} != manifest {manifest['artifacts'][name]}")
            write_model_card(variant, save_dir, manifest["parity"])
            manifest["artifacts"] = {name: sha256_file(save_dir / name) for name in PAYLOAD_FILES}
            manifest["card_refresh_provenance"] = collect_provenance()
            with open(save_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            validate_payload(variant, save_dir)
            print(f"[{variant}] card refreshed and payload revalidated: {save_dir}")
        return

    provenance = collect_provenance()
    for variant in args.variants:
        print(f"=== {variant} ===")
        model, convert_info = convert_variant(variant)
        parity = parity_gate(variant, model)
        save_dir = serialize_and_reload(variant, model, args.output_dir, args.overwrite)
        write_model_card(variant, save_dir, parity)
        manifest = {
            "variant": variant, "arch": SPECS[variant]["arch"], "tag": PRETRAINED_TAG,
            **convert_info, "save_dir": str(save_dir),
            "artifacts": {name: sha256_file(save_dir / name) for name in PAYLOAD_FILES},
            "parity": parity, "provenance": provenance,
        }
        with open(save_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        del model
        print(f"[{variant}] complete\n")


if __name__ == "__main__":
    main()
