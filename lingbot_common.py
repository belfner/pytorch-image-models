"""Shared constants and payload validation for the LingBot-Vision conversion tooling.

Single source of truth for the variant specs (pinned source revisions and hashes, target
timm architectures), staging repo identity, payload layout, parity tolerances, and the
persisted-state payload validator used by both the conversion and upload scripts.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

WRAPPER_ALLOWLIST = ("teacher", "model_state", "state_dict", "model", "backbone")

SPECS = {
    "small": {
        "repo_id": "robbyant/lingbot-vision-vit-small",
        "revision": "127cbcec380de0bcd55bdc1b1fad3819850a6514",
        "sha256": "dca36562cb6b0b34504df6edc18fa282c5ef06fb375c3e91d5487247a1096f9d",
        "arch": "vit_small_patch16_lingbot",
        "params_m": 21.6, "gmacs": 22.2, "macts": 43.07,
        "embed_dim": 384, "depth": 12, "qkv_bias": True, "ffn_hidden": None,
    },
    "base": {
        "repo_id": "robbyant/lingbot-vision-vit-base",
        "revision": "f606f8c6c4002234ea68038f4d7c7cf57da96dfa",
        "sha256": "783dfb59014c34e9f0013db60bf6f5cc3c6b0604eb9528fb5df65e80769c5825",
        "arch": "vit_base_patch16_lingbot",
        "params_m": 85.7, "gmacs": 88.1, "macts": 86.14,
        "embed_dim": 768, "depth": 12, "qkv_bias": True, "ffn_hidden": None,
    },
    "large": {
        "repo_id": "robbyant/lingbot-vision-vit-large",
        "revision": "5e0370623d4fa5db945d00bc47a8545eed407d6b",
        "sha256": "5b5eb67ebbf990b747658ecf90f1cf2b93f5b0e8dfdfbceb060ae1fc364deb8f",
        "arch": "vit_large_patch16_lingbot",
        "params_m": 303.1, "gmacs": 311.81, "macts": 228.65,
        "embed_dim": 1024, "depth": 24, "qkv_bias": True, "ffn_hidden": None,
    },
    "giant": {
        "repo_id": "robbyant/lingbot-vision-vit-giant",
        "revision": "f87d0865a0ae06e640a09be153cb0ba2bda5156d",
        "sha256": "ba9d0b3058b12166c491079b09a67aae6c9cd4cc46299717524c58ba6a4fe8bf",
        "arch": "vit_giant_patch16_lingbot",
        "params_m": 1134.5, "gmacs": 1167.15, "macts": 654.86,
        "embed_dim": 1536, "depth": 40, "qkv_bias": False, "ffn_hidden": 4096,
    },
}

STAGING_OWNER = "belfner"
PRETRAINED_TAG = "robbyant"
PAYLOAD_FILES = ("config.json", "model.safetensors", "pytorch_model.bin", "README.md")
RTOL, ATOL = 1e-4, 1e-5


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
      path (Path): File to hash.

    Returns:
      str: Hex digest of the file contents.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def save_dir_for(variant: str, output_dir: Path) -> Path:
    """Return the per-variant save directory beneath an output root.

    Args:
      variant (str): One of small, base, large, giant.
      output_dir (Path): Root output directory.

    Returns:
      Path: ``<output_dir>/<arch>.<tag>``.
    """
    return output_dir / f"{SPECS[variant]['arch']}.{PRETRAINED_TAG}"


def validate_payload(variant: str, save_dir: Path) -> dict:
    """Validate a save directory's payload bytes and manifest/config semantics.

    Checks the exact five-file set, every payload hash, manifest identity and source
    pinning against ``SPECS``, config.json architecture/pooling/license/preprocessing,
    presence and acceptance of every required parity metric, and generation provenance.
    This is the persisted-state gate the upload script relies on.

    Args:
      variant (str): One of small, base, large, giant.
      save_dir (Path): Directory to validate.

    Returns:
      dict: The parsed manifest.

    Raises:
      ValueError: Any structural, identity, semantic, or parity check failed.
    """
    spec = SPECS[variant]
    entries = sorted(p.relative_to(save_dir).as_posix() for p in save_dir.rglob("*"))
    expected = sorted((*PAYLOAD_FILES, "manifest.json"))
    if entries != expected:
        raise ValueError(f"{save_dir}: payload set {entries} != expected {expected}")
    with open(save_dir / "manifest.json") as f:
        manifest = json.load(f)
    if sorted(manifest["artifacts"]) != sorted(PAYLOAD_FILES):
        raise ValueError(f"{save_dir}: manifest artifact keys {sorted(manifest['artifacts'])} != {sorted(PAYLOAD_FILES)}")
    for name in PAYLOAD_FILES:
        actual = sha256_file(save_dir / name)
        if actual != manifest["artifacts"][name]:
            raise ValueError(f"{save_dir / name}: hash {actual} != manifest {manifest['artifacts'][name]}")

    identity = {
        "variant": variant, "arch": spec["arch"], "tag": PRETRAINED_TAG,
        "source_repo": spec["repo_id"], "source_revision": spec["revision"], "source_sha256": spec["sha256"],
    }
    for key, want in identity.items():
        if manifest.get(key) != want:
            raise ValueError(f"{save_dir}: manifest {key}={manifest.get(key)!r} != {want!r}")

    with open(save_dir / "config.json") as f:
        cfg = json.load(f)
    pcfg = cfg.get("pretrained_cfg", {})
    cfg_expect = {
        ("architecture",): spec["arch"], ("global_pool",): "avg",
        ("pretrained_cfg", "license"): "apache-2.0", ("pretrained_cfg", "crop_mode"): "squash",
        ("pretrained_cfg", "interpolation"): "bilinear", ("pretrained_cfg", "crop_pct"): 1.0,
        ("pretrained_cfg", "input_size"): [3, 512, 512], ("pretrained_cfg", "fixed_input_size"): True,
    }
    for path_keys, want in cfg_expect.items():
        node = cfg if len(path_keys) == 1 else pcfg
        got = node.get(path_keys[-1])
        if got != want:
            raise ValueError(f"{save_dir}: config {'.'.join(path_keys)}={got!r} != {want!r}")

    def accept(value: float, bound: float, what: str) -> None:
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value > bound:
            raise ValueError(f"{save_dir}: {what} = {value!r} fails the <= {bound} acceptance bound")

    parity = manifest["parity"]
    for label in ("512x512", "384x512", "real_image"):
        for part in ("cls", "registers", "patches"):
            metric = parity.get(label, {}).get(part)
            if metric is None:
                raise ValueError(f"{save_dir}: parity metric {label}/{part} missing")
            accept(metric["max_abs"], ATOL, f"parity {label}/{part} max_abs")
    accept(parity.get("public_forward_token_pool", {}).get("max_abs", float("nan")), ATOL, "token-pool public forward")
    if parity.get("public_forward_avg_pool_smoke") != "pass":
        raise ValueError(f"{save_dir}: default-average forward smoke missing")
    accept(parity.get("transform_max_abs", float("nan")), 1e-6, "transform parity max_abs")

    prov = manifest.get("provenance", {})
    for key in ("converter_sha256", "timm_commit", "reference_commit", "torch", "timm_version"):
        if key not in prov:
            raise ValueError(f"{save_dir}: provenance missing {key}")
    return manifest
