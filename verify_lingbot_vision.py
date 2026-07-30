#!/usr/bin/env python
"""Clean-cache verification of the LingBot-Vision staging repos.

Runs each requested variant sequentially, each in its own subprocess, so every variant
gets a fresh empty ``HF_HOME`` (all artifacts download from the Hub) and a clean memory
baseline. Within its subprocess a variant is loaded three ways (untagged entrypoint
resolving the default tag, explicit tagged name, and the ``hf-hub:`` model id), runs a
real 512x512 inference through ``forward_features`` checking token count, feature width,
and finiteness, and asserts the model stats (Params (M), GMACs, Activations (M), via
``fvcore`` with the same recipe as ``benchmark.py``) against the pinned values in
``lingbot_common.SPECS``. One command covers the family: ``python verify_lingbot_vision.py``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from lingbot_common import PRETRAINED_TAG, SPECS, STAGING_OWNER


def verify(variant: str) -> None:
    """Load one staging repo via all three public entry forms and run real inference.

    Args:
      variant (str): One of small, base, large, giant.

    Raises:
      AssertionError: A load produced the wrong shape, non-finite features, or model
        stats differing from the pinned Params/GMACs/Activations values.
    """
    import torch
    from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis

    timm_root = Path(os.environ.get("LINGBOT_TIMM_ROOT", str(Path(__file__).resolve().parent)))
    sys.path.insert(0, str(timm_root))
    import timm

    spec = SPECS[variant]
    arch, dim = spec["arch"], spec["embed_dim"]
    x = torch.randn(1, 3, 512, 512, generator=torch.Generator().manual_seed(0))
    names = (arch, f"{arch}.{PRETRAINED_TAG}", f"hf-hub:{STAGING_OWNER}/{arch}.{PRETRAINED_TAG}")
    for i, name in enumerate(names):
        model = timm.create_model(name, pretrained=True)
        model.eval()
        with torch.inference_mode():
            seq = model.forward_features(x)
        assert seq.shape == (1, 1029, dim), f"{name}: {seq.shape}"
        assert torch.isfinite(seq).all(), f"{name}: non-finite features"
        print(f"[{variant}] {name}: forward_features {tuple(seq.shape)} ok, patch-mean norm "
              f"{seq[:, 5:].mean(1).norm().item():.3f}", flush=True)
        if i == 0:
            params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
            example = torch.ones((1, 3, 512, 512), dtype=next(model.parameters()).dtype)
            gmacs = round(FlopCountAnalysis(model, example).total() / 1e9, 2)
            macts = round(ActivationCountAnalysis(model, example).total() / 1e6, 2)
            expected = (spec["params_m"], spec["gmacs"], spec["macts"])
            assert (params_m, gmacs, macts) == expected, (
                f"{name}: stats (params_m, gmacs, macts) = {(params_m, gmacs, macts)} != pinned {expected}"
            )
            print(f"[{variant}] stats ok: params {params_m}M, {gmacs} GMACs, {macts}M activations", flush=True)
        del model


def main() -> None:
    """Spawn one clean-cache subprocess per requested variant and report the outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(SPECS), default=list(SPECS))
    parser.add_argument("--worker", metavar="VARIANT", choices=list(SPECS), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker is not None:
        print(f"[{args.worker}] HF_HOME={os.environ['HF_HOME']}", flush=True)
        verify(args.worker)
        return

    for variant in args.models:
        hf_home = tempfile.mkdtemp(prefix=f"lingbot_verify_{variant}_hf_")
        try:
            env = dict(os.environ, HF_HOME=hf_home)
            subprocess.run([sys.executable, __file__, "--worker", variant], env=env, check=True)
        finally:
            shutil.rmtree(hf_home, ignore_errors=True)
    print(f"all requested staging loads verified: {' '.join(args.models)}")


if __name__ == "__main__":
    main()
