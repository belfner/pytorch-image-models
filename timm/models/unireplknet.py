""" UniRepLKNet

Paper: `UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud,
Time-Series and Image Recognition` - https://arxiv.org/abs/2410.08049

Original code and weights from:
* https://github.com/AILab-CVC/UniRepLKNet, original copyright below

UniRepLKNet pairs a small number of large depth-wise kernels for receptive field with many small-kernel
blocks for depth. Large-kernel blocks (LarK) use the `DilatedReparamBlock`: a non-dilated K x K DW conv
plus several parallel dilated small-kernel DW convs whose sparse equivalent kernels are merged into the
large kernel at inference. The block body (BN -> SE -> 1x1 Linear -> GELU -> GRN -> 1x1 Linear -> BN) is
ConvNeXt-V2 derived with BN used so the BNs can be folded into adjacent linear/conv layers.

This implementation is a timm-idiomatic refactor that shares `DropPath`, `LayerNorm2d`,
`GlobalResponseNorm`, `SEModule`, and `NormMlpClassifierHead` with the rest of timm.
`UniRepLKNetBlock.reparameterize` and `DilatedReparamBlock.reparameterize` collapse the
training-time structure into the inference-time form and are driven by
`timm.utils.reparameterize_model`.

Hacked together for timm
"""
# UniRepLKNet
# Copyright (c) 2024 AILab-CVC. Licensed under Apache 2.0.

from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    DropPath,
    LayerNorm2d,
    NormMlpClassifierHead,
    SEModule,
    calculate_drop_path_rates,
    to_2tuple,
    trunc_normal_,
)
from timm.layers.grn import GlobalResponseNorm

from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import named_apply
from ._registry import generate_default_cfgs, register_model

__all__ = ['UniRepLKNet']


_DEFAULT_DILATED_REPARAM_SETTINGS: Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {
    17: ((5, 9, 3, 3, 3), (1, 2, 4, 5, 7)),
    15: ((5, 7, 3, 3, 3), (1, 2, 3, 5, 7)),
    13: ((5, 7, 3, 3, 3), (1, 2, 3, 4, 5)),
    11: ((5, 5, 3, 3, 3), (1, 2, 3, 4, 5)),
    9: ((5, 5, 3, 3), (1, 2, 3, 4)),
    7: ((5, 3, 3), (1, 2, 3)),
    5: ((3, 3), (1, 2)),
}


def _convert_dilated_to_nondilated(kernel: torch.Tensor, dilate_rate: int) -> torch.Tensor:
    """Expand a dilated DW kernel into the equivalent non-dilated sparse kernel."""
    identity_kernel = torch.ones((1, 1, 1, 1), device=kernel.device, dtype=kernel.dtype)
    if kernel.size(1) == 1:
        return F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
    slices = []
    for i in range(kernel.size(1)):
        slices.append(F.conv_transpose2d(kernel[:, i:i + 1], identity_kernel, stride=dilate_rate))
    return torch.cat(slices, dim=1)


def _merge_dilated_into_large_kernel(
        large_kernel: torch.Tensor,
        dilated_kernel: torch.Tensor,
        dilated_r: int,
) -> torch.Tensor:
    """Add a dilated small kernel into a non-dilated large kernel after sparsifying."""
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = _convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    return large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)


def _fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold a BatchNorm2d into the preceding Conv2d, returning the fused (weight, bias)."""
    conv_bias = torch.zeros_like(bn.running_mean) if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    fused_weight = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
    fused_bias = bn.bias + (conv_bias - bn.running_mean) * bn.weight / std
    return fused_weight, fused_bias


class DilatedReparamBlock(nn.Module):
    """Dilated reparameterization block: a non-dilated large-kernel DW conv augmented by parallel
    dilated small-kernel DW convs whose sparse equivalent kernels are merged at inference.

    The dilated branches live in `ModuleList`s so the block is TorchScript-friendly.
    """

    def __init__(
            self,
            channels: int,
            kernel_size: int,
            device=None,
            dtype=None,
    ) -> None:
        """Initialize Dilated Reparam Block.

        Args:
            channels: Number of channels (depth-wise, so equal in/out).
            kernel_size: Size of the large kernel (5/7/9/11/13/15/17).
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        if kernel_size not in _DEFAULT_DILATED_REPARAM_SETTINGS:
            raise ValueError(f'DilatedReparamBlock requires kernel_size in {list(_DEFAULT_DILATED_REPARAM_SETTINGS)}')
        self.kernel_size = kernel_size
        self.kernel_sizes, self.dilates = _DEFAULT_DILATED_REPARAM_SETTINGS[kernel_size]

        self.lk_origin = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=channels,
            bias=False,
            **dd,
        )
        self.origin_bn: Optional[nn.BatchNorm2d] = nn.BatchNorm2d(channels, **dd)
        self.dil_convs = nn.ModuleList([
            nn.Conv2d(
                channels,
                channels,
                kernel_size=k,
                stride=1,
                padding=(r * (k - 1) + 1) // 2,
                dilation=r,
                groups=channels,
                bias=False,
                **dd,
            )
            for k, r in zip(self.kernel_sizes, self.dilates)
        ])
        self.dil_bns = nn.ModuleList([nn.BatchNorm2d(channels, **dd) for _ in self.kernel_sizes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the BN-normalized outputs of the large-kernel and all dilated branches."""
        if self.origin_bn is None:
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for conv, bn in zip(self.dil_convs, self.dil_bns):
            out = out + bn(conv(x))
        return out

    def reparameterize(self) -> None:
        """Fold all dilated branches and BNs into a single non-dilated large-kernel conv (in-place).

        After this call the block contains only ``self.lk_origin`` (a single ``Conv2d`` with bias);
        ``origin_bn`` is set to ``None`` and the dilated branch ModuleLists are emptied. The exposed
        ``reparameterize`` name allows ``timm.utils.reparameterize_model`` to drive the fusion.
        """
        if self.origin_bn is None:
            return
        origin_k, origin_b = _fuse_bn(self.lk_origin, self.origin_bn)
        for r, conv, bn in zip(self.dilates, self.dil_convs, self.dil_bns):
            branch_k, branch_b = _fuse_bn(conv, bn)
            origin_k = _merge_dilated_into_large_kernel(origin_k, branch_k, r)
            origin_b = origin_b + branch_b
        merged = nn.Conv2d(
            origin_k.size(0),
            origin_k.size(0),
            kernel_size=origin_k.size(2),
            stride=1,
            padding=origin_k.size(2) // 2,
            dilation=1,
            groups=origin_k.size(0),
            bias=True,
        )
        merged.weight.data = origin_k
        merged.bias.data = origin_b
        self.lk_origin = merged
        self.origin_bn = None
        self.dil_convs = nn.ModuleList()
        self.dil_bns = nn.ModuleList()


class UniRepLKNetBlock(nn.Module):
    """UniRepLKNet residual block.

    Pipeline (residual branch):
        dwconv -> norm(BN) -> se -> permute_to_NHWC -> pwconv1 -> GELU -> grn ->
        pwconv2 -> permute_to_NCHW -> bn -> scale by gamma -> drop_path
    plus the identity shortcut.

    Small-kernel ("SmaK") blocks set kernel_size in {3, 5} and use a plain depth-wise Conv2d
    for `dwconv`. Large-kernel ("LarK") blocks use `DilatedReparamBlock` for kernel_size >= 7.
    """

    def __init__(
            self,
            dim: int,
            kernel_size: int,
            drop_path: float = 0.,
            layer_scale_init_value: float = 1e-6,
            ffn_factor: int = 4,
            device=None,
            dtype=None,
    ) -> None:
        """Initialize UniRepLKNet block.

        Args:
            dim: Channel dimension.
            kernel_size: 0 for identity dwconv, 3 or 5 for SmaK Conv2d, >=7 for LarK DilatedReparamBlock.
            drop_path: Drop path rate.
            layer_scale_init_value: Initial value for per-channel LayerScale (gamma). Disabled if None or <=0.
            ffn_factor: FFN expansion ratio.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        if kernel_size == 0:
            self.dwconv = nn.Identity()
            self.norm = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, **dd)
            self.norm = nn.BatchNorm2d(dim, **dd)
        else:
            if kernel_size not in (3, 5):
                raise ValueError(f'SmaK kernel_size must be 3 or 5, got {kernel_size}')
            self.dwconv = nn.Conv2d(
                dim, dim,
                kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                dilation=1, groups=dim, bias=False,
                **dd,
            )
            self.norm = nn.BatchNorm2d(dim, **dd)

        self.se = SEModule(dim, rd_channels=dim // 4, **dd)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Linear(dim, ffn_dim, **dd)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(ffn_dim, channels_last=True, **dd)
        self.pwconv2 = nn.Linear(ffn_dim, dim, bias=False, **dd)
        self.bn = nn.BatchNorm2d(dim, **dd)

        if layer_scale_init_value is not None and layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim, **dd))
        else:
            self.register_parameter('gamma', None)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        y = self.dwconv(x)
        y = self.norm(y)
        y = self.se(y)
        y = y.permute(0, 2, 3, 1)
        y = self.pwconv1(y)
        y = self.act(y)
        y = self.grn(y)
        y = self.pwconv2(y)
        y = y.permute(0, 3, 1, 2)
        y = self.bn(y)
        if self.gamma is not None:
            y = y.mul(self.gamma.view(1, -1, 1, 1))
        return x + self.drop_path(y)

    @torch.no_grad()
    def reparameterize(self) -> None:
        """Fuse the BN, GRN bias, LayerScale, and pwconv2 BN into the inference-time form.

        After this call the block computes the same function with fewer ops:

        * ``self.dwconv`` becomes a single ``Conv2d`` with bias (BN folded in) for both
          LarK (``DilatedReparamBlock``) and SmaK (plain depth-wise conv) branches.
        * ``self.norm`` becomes ``Identity``.
        * ``self.grn.bias`` is absorbed into a refreshed bias on ``self.pwconv2``.
          ``self.grn.bias`` is zeroed so its forward contribution disappears.
        * ``self.pwconv2`` becomes a ``Linear`` with bias that has the trailing BN and the
          LayerScale ``gamma`` multiplicatively folded in.
        * ``self.bn`` becomes ``Identity`` and ``self.gamma`` is dropped.

        Safe to call repeatedly; subsequent calls are no-ops because the gating attributes
        (``origin_bn`` on the dwconv, ``BatchNorm2d`` instances, ``self.gamma``) have all
        been removed or replaced after the first call.
        """
        # Fold the pre-pwconv1 stage: dwconv + (optional) BN -> single conv with bias.
        if isinstance(self.dwconv, DilatedReparamBlock):
            self.dwconv.reparameterize()
            inner_conv = self.dwconv.lk_origin
            if isinstance(self.norm, nn.BatchNorm2d):
                fused_w, fused_b = _fuse_bn(inner_conv, self.norm)
                inner_conv.weight.data = fused_w
                if inner_conv.bias is None:
                    inner_conv.bias = nn.Parameter(fused_b)
                else:
                    inner_conv.bias.data = fused_b
                self.norm = nn.Identity()
        elif isinstance(self.dwconv, nn.Conv2d) and isinstance(self.norm, nn.BatchNorm2d):
            fused_w, fused_b = _fuse_bn(self.dwconv, self.norm)
            new_conv = nn.Conv2d(
                self.dwconv.in_channels,
                self.dwconv.out_channels,
                kernel_size=self.dwconv.kernel_size,
                stride=self.dwconv.stride,
                padding=self.dwconv.padding,
                dilation=self.dwconv.dilation,
                groups=self.dwconv.groups,
                bias=True,
            )
            new_conv.weight.data = fused_w
            new_conv.bias.data = fused_b
            self.dwconv = new_conv
            self.norm = nn.Identity()

        # Collapse the trailing GRN-bias / pwconv2 / BN / LayerScale chain into one Linear.
        if isinstance(self.bn, nn.BatchNorm2d):
            bn = self.bn
            std = (bn.running_var + bn.eps).sqrt()
            bn_scale = bn.weight / std
            bn_shift = bn.bias - bn.running_mean * bn_scale
            gamma = self.gamma.data if self.gamma is not None else torch.ones_like(bn.weight)

            linear = self.pwconv2
            grn_bias = self.grn.bias.data
            # GRN bias contributes (linear.weight @ grn_bias) as an extra constant after pwconv2.
            grn_projected = linear.weight.data @ grn_bias.view(-1, 1)
            extra_bias = grn_projected.view(-1)
            if linear.bias is not None:
                extra_bias = extra_bias + linear.bias.data

            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = linear.weight.data * (bn_scale * gamma).view(-1, 1)
            new_linear.bias.data = (bn_shift + extra_bias * bn_scale) * gamma
            self.pwconv2 = new_linear

            self.grn.bias.data.zero_()
            self.bn = nn.Identity()
            self.gamma = None


class UniRepLKNetDownsample(nn.Module):
    """Inter-stage 2x downsample: stride-2 3x3 conv followed by channels-first LayerNorm."""

    def __init__(self, in_chs: int, out_chs: int, device=None, dtype=None) -> None:
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=2, padding=1, **dd)
        self.norm = LayerNorm2d(out_chs, eps=1e-6, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class UniRepLKNetStem(nn.Module):
    """Patch-overlap stem: Conv3x3 s2 -> LN -> GELU -> Conv3x3 s2 -> LN (total stride 4)."""

    def __init__(self, in_chans: int, dim: int, device=None, dtype=None) -> None:
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        mid = dim // 2
        self.conv1 = nn.Conv2d(in_chans, mid, kernel_size=3, stride=2, padding=1, **dd)
        self.norm1 = LayerNorm2d(mid, eps=1e-6, **dd)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid, dim, kernel_size=3, stride=2, padding=1, **dd)
        self.norm2 = LayerNorm2d(dim, eps=1e-6, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(self.conv1(x))
        x = self.act(x)
        return self.norm2(self.conv2(x))


class UniRepLKNetStage(nn.Module):
    """A UniRepLKNet stage: optional downsample, followed by N residual blocks."""

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            depth: int,
            kernel_sizes: Tuple[int, ...],
            drop_path_rates: List[float],
            layer_scale_init_value: float = 1e-6,
            ffn_factor: int = 4,
            downsample: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        """Initialize stage.

        Args:
            in_chs: Input channel count.
            out_chs: Output channel count (also block dim).
            depth: Number of residual blocks.
            kernel_sizes: Per-block kernel sizes (length == depth).
            drop_path_rates: Per-block drop path rates (length == depth).
            layer_scale_init_value: LayerScale init for each block.
            ffn_factor: FFN expansion ratio.
            downsample: If True, prepend a UniRepLKNetDownsample (in_chs -> out_chs, stride 2).
                If False, in_chs must equal out_chs.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        if len(kernel_sizes) != depth:
            raise ValueError(f'kernel_sizes length {len(kernel_sizes)} does not match depth {depth}')
        if len(drop_path_rates) != depth:
            raise ValueError(f'drop_path_rates length {len(drop_path_rates)} does not match depth {depth}')
        if downsample:
            self.downsample = UniRepLKNetDownsample(in_chs, out_chs, **dd)
        else:
            if in_chs != out_chs:
                raise ValueError('downsample=False requires in_chs == out_chs')
            self.downsample = nn.Identity()
        self.blocks = nn.Sequential(*[
            UniRepLKNetBlock(
                dim=out_chs,
                kernel_size=kernel_sizes[i],
                drop_path=drop_path_rates[i],
                layer_scale_init_value=layer_scale_init_value,
                ffn_factor=ffn_factor,
                **dd,
            )
            for i in range(depth)
        ])
        self.grad_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for blk in self.blocks:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        else:
            x = self.blocks(x)
        return x


# Depth presets and the kernel-size templates the upstream code pairs with them.
_UNIREPLKNET_A_F_P_DEPTHS: Tuple[int, ...] = (2, 2, 6, 2)
_UNIREPLKNET_N_DEPTHS: Tuple[int, ...] = (2, 2, 8, 2)
_UNIREPLKNET_T_DEPTHS: Tuple[int, ...] = (3, 3, 18, 3)
_UNIREPLKNET_S_B_L_XL_DEPTHS: Tuple[int, ...] = (3, 3, 27, 3)


def _default_kernel_sizes(depths: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    """Match upstream's default per-stage kernel size templates by depth tuple."""
    if depths == _UNIREPLKNET_A_F_P_DEPTHS:
        return ((3, 3), (13, 13), (13, 13, 13, 13, 13, 13), (13, 13))
    if depths == _UNIREPLKNET_N_DEPTHS:
        return ((3, 3), (13, 13), (13, 13, 13, 13, 13, 13, 13, 13), (13, 13))
    if depths == _UNIREPLKNET_T_DEPTHS:
        return (
            (3, 3, 3),
            (13, 13, 13),
            (13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3),
            (13, 13, 13),
        )
    if depths == _UNIREPLKNET_S_B_L_XL_DEPTHS:
        return (
            (3, 3, 3),
            (13, 13, 13),
            (13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3,
             13, 3, 3, 13, 3, 3, 13, 3, 3),
            (13, 13, 13),
        )
    raise ValueError(f'No default kernel-size template for depths={depths}; pass kernel_sizes explicitly')


class UniRepLKNet(nn.Module):
    """UniRepLKNet model.

    A four-stage ConvNeXt-style large-kernel ConvNet with `DilatedReparamBlock` depth-wise convs in
    large-kernel blocks. Stage 0 always uses small-kernel (3x3) blocks; stages 1-3 use a mix of
    large-kernel (13x13 by default) and small-kernel blocks per the kernel-size template for the
    given depth preset.

    The model is built in training form; call `timm.utils.reparameterize_model(model)` (or invoke
    `reparameterize()` on individual `DilatedReparamBlock` / `UniRepLKNetBlock` submodules) to
    collapse the dilated branches and fold BatchNorms into the surrounding linear/conv layers for
    inference. Reparameterization is shape-preserving; outputs match the training-time forward to
    numerical precision.
    """

    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            depths: Tuple[int, ...] = (3, 3, 27, 3),
            dims: Tuple[int, ...] = (96, 192, 384, 768),
            kernel_sizes: Optional[Tuple[Tuple[int, ...], ...]] = None,
            ffn_factor: int = 4,
            layer_scale_init_value: float = 1e-6,
            head_init_scale: float = 1.,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            device=None,
            dtype=None,
    ) -> None:
        """Initialize UniRepLKNet.

        Args:
            in_chans: Number of input channels.
            num_classes: Number of classification classes. 0 disables the classifier.
            global_pool: Global pooling type (passed to NormMlpClassifierHead).
            depths: Number of blocks in each of the four stages.
            dims: Channel count for each of the four stages.
            kernel_sizes: Per-stage tuple of per-block kernel sizes. If None, uses the default
                template paired with `depths`.
            ffn_factor: FFN expansion ratio inside each block.
            layer_scale_init_value: Init for the per-channel LayerScale (gamma) in each block.
            head_init_scale: Init scaling applied to the classifier weights and biases.
            drop_rate: Pre-classifier dropout rate.
            drop_path_rate: Maximum stochastic depth rate; linearly scaled across blocks.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        depths = tuple(depths)
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError('UniRepLKNet expects 4 stages: len(depths) and len(dims) must be 4')
        if kernel_sizes is None:
            kernel_sizes = _default_kernel_sizes(depths)
        if len(kernel_sizes) != 4:
            raise ValueError('kernel_sizes must have one tuple per stage (4 total)')
        for i in range(4):
            if len(kernel_sizes[i]) != depths[i]:
                raise ValueError(f'kernel_sizes[{i}] length {len(kernel_sizes[i])} != depths[{i}]={depths[i]}')

        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        self.num_features = self.head_hidden_size = dims[-1]
        self.feature_info: List[Dict[str, Union[int, str]]] = []

        self.stem = UniRepLKNetStem(in_chans, dims[0], **dd)

        # Build stages.
        dp_rates = calculate_drop_path_rates(drop_path_rate, depths, stagewise=True)
        stages = []
        prev_chs = dims[0]
        stride = 4
        for i in range(4):
            downsample = i > 0
            if downsample:
                stride *= 2
            stages.append(UniRepLKNetStage(
                in_chs=prev_chs,
                out_chs=dims[i],
                depth=depths[i],
                kernel_sizes=tuple(kernel_sizes[i]),
                drop_path_rates=list(dp_rates[i]),
                layer_scale_init_value=layer_scale_init_value,
                ffn_factor=ffn_factor,
                downsample=downsample,
                **dd,
            ))
            prev_chs = dims[i]
            self.feature_info.append(dict(num_chs=prev_chs, reduction=stride, module=f'stages.{i}'))
        self.stages = nn.Sequential(*stages)

        self.head = NormMlpClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            norm_layer='layernorm2d',
            **dd,
        )

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Union[str, List]]:
        """Return regex group matchers for layer-wise LR decay.

        Args:
            coarse: If True, group all blocks within a stage together.
        """
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
            ],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing per stage."""
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Return the final classifier module."""
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classifier head to a new num_classes / pool type."""
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward and collect per-stage feature maps.

        Args:
            x: Input image tensor.
            indices: Stage indices to collect (int = last N, list = explicit, None = all).
            norm: Apply head norm to the final-stage intermediate if it is included.
            stop_early: Stop iteration after the last requested stage.
            output_fmt: Only `'NCHW'` is supported.
            intermediates_only: Return only the intermediates list.

        Returns:
            Either a list of intermediates or `(final_features, intermediates)`.
        """
        if output_fmt != 'NCHW':
            raise ValueError("output_fmt must be 'NCHW'")
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        intermediates: List[torch.Tensor] = []

        x = self.stem(x)
        last_idx = len(self.stages) - 1
        if torch.jit.is_scripting() or not stop_early:
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]
        feat_idx = 0
        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                if norm and feat_idx == last_idx:
                    intermediates.append(self.head.norm(x))
                else:
                    intermediates.append(x)

        if intermediates_only:
            return intermediates
        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ) -> List[int]:
        """Drop stages after the last requested intermediate; optionally prune head."""
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]
        if prune_head:
            self.reset_classifier(0, '')
        elif prune_norm:
            self.head.norm = nn.Identity()
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module: nn.Module, name: Optional[str] = None, head_init_scale: float = 1.0) -> None:
    """Initialize weights: truncated normal for conv/linear weights, zeros for biases.

    Args:
        module: Module being initialized.
        name: Fully-qualified module name (used to scale the classifier).
        head_init_scale: Multiplier applied to `head.fc` weight/bias after init.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        trunc_normal_(module.weight, std=.02)
        if getattr(module, 'bias', None) is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear) and name and name.endswith('head.fc'):
            module.weight.data.mul_(head_init_scale)
            if module.bias is not None:
                module.bias.data.mul_(head_init_scale)


def _cfg(url: str = '', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': (7, 7),
        'crop_pct': 0.875,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv1',
        'classifier': 'head.fc',
        'license': 'apache-2.0',
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'unireplknet_a.in1k': _cfg(
        hf_hub_id='belfner/unireplknet_a.in1k',
    ),
    'unireplknet_f.in1k': _cfg(
        hf_hub_id='belfner/unireplknet_f.in1k',
    ),
    'unireplknet_p.in1k': _cfg(
        hf_hub_id='belfner/unireplknet_p.in1k',
    ),
    'unireplknet_n.in1k': _cfg(
        hf_hub_id='belfner/unireplknet_n.in1k',
    ),
    'unireplknet_t.in1k': _cfg(
        hf_hub_id='belfner/unireplknet_t.in1k',
    ),
    'unireplknet_s.in1k': _cfg(
        hf_hub_id='belfner/unireplknet_s.in1k',
    ),
    'unireplknet_s.in22k': _cfg(
        hf_hub_id='belfner/unireplknet_s.in22k',
        num_classes=21841,
    ),
    'unireplknet_s.in22k_ft_in1k_384': _cfg(
        hf_hub_id='belfner/unireplknet_s.in22k_ft_in1k_384',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
    'unireplknet_b.in22k': _cfg(
        hf_hub_id='belfner/unireplknet_b.in22k',
        num_classes=21841,
    ),
    'unireplknet_b.in22k_ft_in1k_384': _cfg(
        hf_hub_id='belfner/unireplknet_b.in22k_ft_in1k_384',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
    'unireplknet_l.in22k': _cfg(
        hf_hub_id='belfner/unireplknet_l.in22k',
        num_classes=21841,
        input_size=(3, 192, 192), pool_size=(6, 6), crop_pct=0.875,
    ),
    'unireplknet_l.in22k_ft_in1k_384': _cfg(
        hf_hub_id='belfner/unireplknet_l.in22k_ft_in1k_384',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
    'unireplknet_xl.in22k': _cfg(
        hf_hub_id='belfner/unireplknet_xl.in22k',
        num_classes=21841,
        input_size=(3, 192, 192), pool_size=(6, 6), crop_pct=0.875,
    ),
    'unireplknet_xl.in22k_ft_in1k_384': _cfg(
        hf_hub_id='belfner/unireplknet_xl.in22k_ft_in1k_384',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
})


def _create_unireplknet(variant: str, pretrained: bool = False, **kwargs) -> UniRepLKNet:
    return build_model_with_cfg(
        UniRepLKNet,
        variant,
        pretrained,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )


@register_model
def unireplknet_a(pretrained: bool = False, **kwargs) -> UniRepLKNet:
    model_args = dict(depths=_UNIREPLKNET_A_F_P_DEPTHS, dims=(40, 80, 160, 320))
    return _create_unireplknet('unireplknet_a', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def unireplknet_f(pretrained: bool = False, **kwargs) -> UniRepLKNet:
    model_args = dict(depths=_UNIREPLKNET_A_F_P_DEPTHS, dims=(48, 96, 192, 384))
    return _create_unireplknet('unireplknet_f', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def unireplknet_p(pretrained: bool = False, **kwargs) -> UniRepLKNet:
    model_args = dict(depths=_UNIREPLKNET_A_F_P_DEPTHS, dims=(64, 128, 256, 512))
    return _create_unireplknet('unireplknet_p', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def unireplknet_n(pretrained: bool = False, **kwargs) -> UniRepLKNet:
    model_args = dict(depths=_UNIREPLKNET_N_DEPTHS, dims=(80, 160, 320, 640))
    return _create_unireplknet('unireplknet_n', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def unireplknet_t(pretrained: bool = False, **kwargs) -> UniRepLKNet:
    model_args = dict(depths=_UNIREPLKNET_T_DEPTHS, dims=(80, 160, 320, 640))
    return _create_unireplknet('unireplknet_t', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def unireplknet_s(pretrained: bool = False, **kwargs) -> UniRepLKNet:
    model_args = dict(depths=_UNIREPLKNET_S_B_L_XL_DEPTHS, dims=(96, 192, 384, 768))
    return _create_unireplknet('unireplknet_s', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def unireplknet_b(pretrained: bool = False, **kwargs) -> UniRepLKNet:
    model_args = dict(depths=_UNIREPLKNET_S_B_L_XL_DEPTHS, dims=(128, 256, 512, 1024))
    return _create_unireplknet('unireplknet_b', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def unireplknet_l(pretrained: bool = False, **kwargs) -> UniRepLKNet:
    model_args = dict(depths=_UNIREPLKNET_S_B_L_XL_DEPTHS, dims=(192, 384, 768, 1536))
    return _create_unireplknet('unireplknet_l', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def unireplknet_xl(pretrained: bool = False, **kwargs) -> UniRepLKNet:
    model_args = dict(depths=_UNIREPLKNET_S_B_L_XL_DEPTHS, dims=(256, 512, 1024, 2048))
    return _create_unireplknet('unireplknet_xl', pretrained=pretrained, **dict(model_args, **kwargs))
