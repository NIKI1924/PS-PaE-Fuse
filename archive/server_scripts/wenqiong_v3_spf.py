#!/usr/bin/env python
"""wenqiong_v3_spf.py — Wenqiong v3 (Spherical Probabilistic Fusion)

设计哲学(三支柱):
  P1 Respect the Sphere:      SphereHyperNet(球谐 L=4 + 纬度 RBF) 替换 LocationHyperNet
  P2 Respect Probability:     SigmaHead + Gaussian CRPS 替代 NormMSE
  P3 Respect Extremes:        MultiScaleContextBranch(patch 内 4× pool);
                              Focal+Tversky 组合(在 losses_v3.py 里实现)

v2 保留(zero 改动,直接 import):
  - ConvBlock, I2MoEInteraction, AdaptiveRouterV2
  - DualPathRefinement, ClassificationRefinement

v3 新增/替换:
  - MultiScaleContextBranch        P3 解决 patch 感受野
  - SphereHyperNet                 P1 球面几何编码
  - RegressionHead                 替代 LocationHyperNet 的最后一层
  - ClassificationHead             替代 ClassificationHyperNet 的最后一层
  - SigmaHead                      P2 σ 预测头
  - WenqiongV3SPF                  主模型,forward 返回 (mu, sigma, cls_logits)
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- v2 保留模块直接复用 ---
from wenqiong_dualtask import (
    ConvBlock, I2MoEInteraction, AdaptiveRouterV2,
    DualPathRefinement, ClassificationRefinement,
    N_VARS, N_MODELS, VAR_SHORT,
    IDX_T2M, IDX_U10, IDX_V10, IDX_MSL, IDX_Z500, IDX_T850,
)

# ============================================================
# Constants
# ============================================================
GRID_H, GRID_W = 721, 1440          # 0.25° 全球网格
PATCH_H, PATCH_W = 96, 96
VAR_ERROR_SCALE_V3 = [1.5, 5.0, 2.3, 300.0, 450.0, 1.6]  # Z500 从 1850→450


# ============================================================
# MultiScaleContextBranch (P3)
# ============================================================

class MultiScaleContextBranch(nn.Module):
    """Patch 内的"粗尺度 synoptic 摘要": 4× avg-pool → 3-layer conv → bilinear upsample。

    提供大尺度系统(锋面、气旋)在 patch 内的 context,弥补 96×96 的感受野不足。
    参考 MoWE (arXiv:2509.09052) 的极简实现。

    Input:  (B, 24, H, W)  — 4 models × 6 vars
    Output: (B, d_ctx, H, W)
    """
    def __init__(self, in_ch=N_VARS * N_MODELS, mid_ch=32, d_ctx=16, pool_factor=4):
        super().__init__()
        self.pool_factor = pool_factor
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.GroupNorm(8, mid_ch), nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.GroupNorm(8, mid_ch), nn.GELU(),
            nn.Conv2d(mid_ch, d_ctx, 1),
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        c = F.avg_pool2d(x, self.pool_factor)
        c = self.net(c)
        return F.interpolate(c, size=(H, W), mode='bilinear', align_corners=False)


# ============================================================
# Real Spherical Harmonics (precomputed buffer)
# ============================================================

def real_sph_harm_basis(L_max, lat_deg, lon_deg):
    """计算实球谐基函数 Y_{l,m}, l=0..L_max, m=-l..l。

    使用 scipy.special.sph_harm:
      Y_{l,m}^scipy(phi, theta) is complex (Condon-Shortley convention).
    实球谐:
      Y_{l,0}^real   = Re(Y_{l,0})
      Y_{l,m>0}^real = sqrt(2) · Re(Y_{l,m})
      Y_{l,m<0}^real = sqrt(2) · Im(Y_{l,|m|})
    theta = colatitude (from north pole): cos(theta) = sin(lat_in_deg) — wait, check:
      lat=+90 (N pole) → theta=0;  lat=-90 (S pole) → theta=π;  lat=0 → theta=π/2.
      So colatitude = (π/2 - lat) in radians = (90 - lat) in degrees.

    Args:
        L_max:   max degree
        lat_deg: lat 数组 (任意 shape),单位度,[-90, 90]
        lon_deg: lon 数组 (相同 shape),单位度,[0, 360)

    Returns:
        (N_basis, *lat.shape) numpy array, N_basis = (L_max+1)^2
    """
    # scipy.special.sph_harm 在 scipy 1.15+ 被弃用,重命名为 sph_harm_y。
    # 两种 API 都支持;优先新 API。
    import warnings
    try:
        from scipy.special import sph_harm_y
        def _complex_Ylm(m, l, phi, theta):
            # sph_harm_y(n=l, m=m, theta=colat, phi=lon) returns complex Y_l^m
            return sph_harm_y(l, m, theta, phi)
    except ImportError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from scipy.special import sph_harm
        def _complex_Ylm(m, l, phi, theta):
            # sph_harm(m, n=l, phi, theta) returns complex Y_l^m
            return sph_harm(m, l, phi, theta)

    theta = np.radians(90.0 - lat_deg)   # colatitude
    phi = np.radians(lon_deg)
    basis = []
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            if m == 0:
                y = _complex_Ylm(0, l, phi, theta).real
            elif m > 0:
                y = np.sqrt(2.0) * _complex_Ylm(m, l, phi, theta).real
            else:  # m < 0
                y = np.sqrt(2.0) * _complex_Ylm(-m, l, phi, theta).imag
            basis.append(y.astype(np.float32))
    return np.stack(basis, axis=0)  # ((L_max+1)^2, *lat.shape)


# ============================================================
# SphereHyperNet (P1 核心)
# ============================================================

class SphereHyperNet(nn.Module):
    """球面位置编码 + 纬度 RBF + lead embedding → per-pixel feature map。

    为两个分支共享产生 (B, d_out, pH, pW) 的球面几何条件特征。
    替代 v2 的 LocationHyperNet 和 ClassificationHyperNet 的 Fourier K=8。

    关键组件:
      1. Real Spherical Harmonics L_max=4 (25 basis) — 预计算全球 buffer
         Y_{1,0} = sqrt(3/(4π))·cos(θ) 直接编码 Z500 的纬度主导梯度
      2. Latitude Gaussian RBF (n_lat_rbf=8 centers, σ=12° learnable)
         捕捉急流带 / 副热带高压等纬度局部特征
      3. Lead-time embedding (7 leads → d_lead=32)

    Args:
        L_max:         球谐最大阶(default 4, 25 基)
        n_lat_rbf:     纬度 RBF 中心数(default 8)
        d_lead:        lead-time embedding 维度
        d_out:         输出通道数(两分支共享的球面特征)
        patch_h/w:     patch 大小
        grid_h/w:     全球网格大小(用于预计算 SH)
    """
    def __init__(self, L_max=4, n_lat_rbf=8, d_lead=32, d_out=32,
                 patch_h=PATCH_H, patch_w=PATCH_W,
                 grid_h=GRID_H, grid_w=GRID_W, n_lead_hours=7):
        super().__init__()
        self.L_max = L_max
        self.n_sh = (L_max + 1) ** 2
        self.n_lat_rbf = n_lat_rbf
        self.grid_h, self.grid_w = grid_h, grid_w
        self.patch_h, self.patch_w = patch_h, patch_w
        self.d_out = d_out

        # --- 预计算全球 SH buffer ---
        # lat[i] = 90 - i*0.25 for i in 0..grid_h-1 = 720
        # lon[j] = j*0.25 for j in 0..grid_w-1 = 1439
        lat_1d = 90.0 - np.arange(grid_h, dtype=np.float64) * 0.25
        lon_1d = np.arange(grid_w, dtype=np.float64) * 0.25
        lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d, indexing='ij')
        sh = real_sph_harm_basis(L_max, lat_2d, lon_2d)  # (n_sh, 721, 1440)
        # ~25 × 721 × 1440 × 4 bytes ≈ 104 MB — 大但可接受
        self.register_buffer('sh_basis_global',
                             torch.from_numpy(sh).contiguous())

        # --- 纬度 RBF ---
        self.register_buffer('lat_centers',
                             torch.linspace(-90.0, 90.0, n_lat_rbf))
        self.lat_sigma = nn.Parameter(torch.tensor(12.0))  # learnable in degrees

        # --- Lead embedding ---
        self.lead_embed = nn.Embedding(n_lead_hours, d_lead)

        # --- Fusion MLP (1x1 conv) ---
        d_in = self.n_sh + self.n_lat_rbf + d_lead
        self.mlp = nn.Sequential(
            nn.Conv2d(d_in, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, d_out, 1),
        )
        # 小随机初始化,避免 v2 的 zero cold-start
        for m in self.mlp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def _gather_sh_patch(self, lat0, lon0, flip_flag=None):
        """按 (lat0, lon0) 从全球 SH buffer 切 patch。

        Returns: (B, n_sh, pH, pW)
        """
        B = lat0.shape[0]
        device = lat0.device
        pH, pW = self.patch_h, self.patch_w

        h_off = torch.arange(pH, device=device, dtype=torch.long)  # (pH,)
        w_off = torch.arange(pW, device=device, dtype=torch.long)  # (pW,)

        if flip_flag is not None and flip_flag.any():
            flip_l = flip_flag.long()
            w_off_flip = (pW - 1 - w_off).unsqueeze(0).expand(B, -1)  # (B, pW)
            w_off_norm = w_off.unsqueeze(0).expand(B, -1)             # (B, pW)
            w_off_b = torch.where(flip_l.unsqueeze(1) > 0,
                                  w_off_flip, w_off_norm)              # (B, pW)
        else:
            w_off_b = w_off.unsqueeze(0).expand(B, -1)

        row_idx = lat0.unsqueeze(1) + h_off.unsqueeze(0)              # (B, pH)
        row_idx = row_idx.clamp(min=0, max=self.grid_h - 1)
        col_idx = (lon0.unsqueeze(1) + w_off_b) % self.grid_w         # (B, pW)

        # 用 advanced indexing 抽取
        # Broadcast to (B, pH, pW)
        row_exp = row_idx.unsqueeze(-1).expand(B, pH, pW)             # (B, pH, pW)
        col_exp = col_idx.unsqueeze(1).expand(B, pH, pW)              # (B, pH, pW)
        # sh_basis_global[:, row_exp, col_exp] → (n_sh, B, pH, pW)
        sh = self.sh_basis_global[:, row_exp, col_exp]
        return sh.permute(1, 0, 2, 3).contiguous()                     # (B, n_sh, pH, pW)

    def _compute_lat_grid(self, lat0):
        """Returns (B, pH) latitude in degrees."""
        device = lat0.device
        h_off = torch.arange(self.patch_h, device=device, dtype=torch.float32)
        return 90.0 - (lat0.float().unsqueeze(1) + h_off.unsqueeze(0)) * 0.25

    def forward(self, lat0, lon0, lead_hour_idx, flip_flag=None):
        B = lat0.shape[0]
        pH, pW = self.patch_h, self.patch_w

        # SH patch
        sh = self._gather_sh_patch(lat0, lon0, flip_flag)            # (B, n_sh, pH, pW)

        # Latitude RBF
        lat_deg = self._compute_lat_grid(lat0)                        # (B, pH)
        diff = lat_deg.unsqueeze(1).unsqueeze(-1) - \
               self.lat_centers.view(1, -1, 1, 1)                    # (B, n_lat_rbf, pH, 1)
        rbf = torch.exp(-(diff ** 2) /
                        (2.0 * self.lat_sigma.clamp(min=1.0) ** 2 + 1e-6))
        rbf = rbf.expand(-1, -1, -1, pW)                              # (B, n_lat_rbf, pH, pW)

        # Lead embedding
        le = self.lead_embed(lead_hour_idx)                           # (B, d_lead)
        le = le.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, pH, pW)    # (B, d_lead, pH, pW)

        z = torch.cat([sh, rbf, le], dim=1)
        return self.mlp(z)                                            # (B, d_out, pH, pW)


# ============================================================
# Small heads (替代 v2 的两个 HyperNet 的最后一层)
# ============================================================

class RegressionHead(nn.Module):
    """Conv1x1(sph_feat) → scale(6) + bias(6), 应用到 consensus_raw + scale·refined + bias。
    """
    def __init__(self, d_in=32, n_vars=N_VARS):
        super().__init__()
        self.n_vars = n_vars
        self.conv = nn.Conv2d(d_in, 2 * n_vars, 1)
        nn.init.normal_(self.conv.weight, std=0.02)
        nn.init.zeros_(self.conv.bias)

    def forward(self, sph_feat, refined, consensus_raw):
        params = self.conv(sph_feat)                   # (B, 2V, H, W)
        scale_raw = params[:, :self.n_vars]
        bias = params[:, self.n_vars:]
        scale = 0.5 + torch.sigmoid(scale_raw)         # ∈ [0.5, 1.5]
        return consensus_raw + scale * refined + bias


class ClassificationHead(nn.Module):
    """Conv1x1(sph_feat) → logit_bias(n_out), 加到 base_logits。"""
    def __init__(self, d_in=32, n_out=N_VARS * 3):
        super().__init__()
        self.conv = nn.Conv2d(d_in, n_out, 1)
        nn.init.normal_(self.conv.weight, std=0.02)
        nn.init.zeros_(self.conv.bias)

    def forward(self, sph_feat, base_logits):
        return base_logits + self.conv(sph_feat)


# ============================================================
# SigmaHead (P2: 预测 σ 用于 CRPS)
# ============================================================

def _softplus_inv(y):
    """softplus^{-1}(y) = log(e^y - 1);大 y 时直接返回 y(稳定)。"""
    if y > 20:
        return y
    # y = log(1 + e^x) → e^y = 1 + e^x → x = log(e^y - 1)
    return float(math.log(math.expm1(y)))


class SigmaHead(nn.Module):
    """预测 log σ 的小 conv,每变量初始化 σ ≈ VAR_ERROR_SCALE_V3[v]。

    Input:  concat(spread, sph_feat), (B, V+d_sph, H, W)
    Output: σ, (B, V, H, W), softplus 保正,per-var clamp。
    """
    def __init__(self, d_spread=N_VARS, d_sph=32, d_mid=32, n_vars=N_VARS,
                 init_sigma_per_var=None, sigma_cap_mult=5.0):
        super().__init__()
        self.n_vars = n_vars
        self.net = nn.Sequential(
            nn.Conv2d(d_spread + d_sph, d_mid, 1),
            nn.GELU(),
            nn.Conv2d(d_mid, n_vars, 1),
        )
        # 小随机初始化 weights;通过 bias 设定初始 σ
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

        # Per-variable initial sigma (in raw units)
        if init_sigma_per_var is None:
            init_sigma_per_var = VAR_ERROR_SCALE_V3
        init_bias = [_softplus_inv(s - 1e-3) for s in init_sigma_per_var]
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor(init_bias, dtype=torch.float32))

        # Per-variable upper cap
        cap = [s * sigma_cap_mult for s in init_sigma_per_var]
        self.register_buffer('sigma_cap',
                             torch.tensor(cap, dtype=torch.float32))

    def forward(self, spread, sph_feat):
        z = torch.cat([spread, sph_feat], dim=1)
        log_sigma = self.net(z)
        sigma = F.softplus(log_sigma) + 1e-3
        sigma = torch.minimum(sigma, self.sigma_cap.view(1, -1, 1, 1))
        return sigma


# ============================================================
# Main model: WenqiongV3SPF
# ============================================================

class WenqiongV3SPF(nn.Module):
    """Wenqiong v3 — Spherical Probabilistic Fusion

    Forward returns (mu, sigma, cls_logits).
    """
    def __init__(self, d_interact=48, d_route=32, base_ch=24, dropout=0.1,
                 n_thresholds=3, sph_L_max=4, sph_n_lat_rbf=8,
                 sph_d_lead=32, sph_d_out=32,
                 ctx_mid=32, ctx_d=16, use_context=True,
                 patch_h=PATCH_H, patch_w=PATCH_W):
        super().__init__()
        self.n_thresholds = n_thresholds
        self.use_context = use_context
        self.patch_h, self.patch_w = patch_h, patch_w
        self.sph_d_out = sph_d_out

        # --- Shared Encoder ---
        self.encoder_interact = I2MoEInteraction(d_interact)

        self.ctx_d = ctx_d if use_context else 0
        if use_context:
            self.context = MultiScaleContextBranch(
                in_ch=N_VARS * N_MODELS, mid_ch=ctx_mid, d_ctx=ctx_d,
                pool_factor=4)
        else:
            self.context = None

        # AdaptiveRouter input = interact + (optional) context
        self.encoder_router = AdaptiveRouterV2(
            d_interact + self.ctx_d, d_route)
        # v3 覆盖 v2 的 router 最后一层 zero-init:避免 context/sphere 在 step 0
        # 没有梯度流通(v2 的 zero-init 会让 correction=0,context 无梯度)。
        # 小随机 std=0.01 仍然给出近似均匀路由(avg_log(bias) 主导),但允许梯度流。
        with torch.no_grad():
            last_conv = self.encoder_router.router[-1]
            nn.init.normal_(last_conv.weight, std=0.01)
            # bias 保留 v2 的 1/N_MODELS=0.25(uniform routing prior)

        # --- Regression Branch (refinement from v2) ---
        self.reg_refinement = DualPathRefinement(
            interact_ch=d_interact, base_ch=base_ch, dropout=dropout)

        # --- Classification Branch (refinement from v2) ---
        self.cls_refinement = ClassificationRefinement(
            interact_ch=d_interact, base_ch=base_ch,
            n_thresholds=n_thresholds, dropout=dropout)

        # --- Shared SphereHyperNet ---
        self.sph = SphereHyperNet(
            L_max=sph_L_max, n_lat_rbf=sph_n_lat_rbf,
            d_lead=sph_d_lead, d_out=sph_d_out,
            patch_h=patch_h, patch_w=patch_w)

        # --- Small heads ---
        self.reg_head = RegressionHead(d_in=sph_d_out, n_vars=N_VARS)
        self.cls_head = ClassificationHead(
            d_in=sph_d_out, n_out=N_VARS * n_thresholds)
        self.sigma_head = SigmaHead(d_spread=N_VARS, d_sph=sph_d_out,
                                    n_vars=N_VARS,
                                    init_sigma_per_var=VAR_ERROR_SCALE_V3)

        # --- State ---
        self.last_routing_weights = None
        self.last_cls_logits = None

    def forward(self, x_norm, x_raw=None, lead_hour_idx=None,
                lat0=None, lon0=None, flip_flag=None):
        assert lat0 is not None and lon0 is not None, \
            "v3 requires lat0, lon0 for SphereHyperNet"
        B, _, pH, pW = x_norm.shape

        # --- Feature Engineering ---
        mn = x_norm.reshape(B, N_MODELS, N_VARS, pH, pW)
        consensus_n = mn.mean(dim=1)
        spread = mn.std(dim=1)
        dev_n = mn - consensus_n.unsqueeze(1)
        dev_flat = dev_n.reshape(B, -1, pH, pW)

        if x_raw is not None:
            consensus_raw = x_raw.reshape(B, N_MODELS, N_VARS, pH, pW).mean(dim=1)
        else:
            consensus_raw = consensus_n

        # --- Shared Encoder ---
        interact = self.encoder_interact(dev_flat, spread)      # (B, 48, H, W)

        if self.use_context:
            ctx = self.context(x_norm)                          # (B, ctx_d, H, W)
            enc_feat = torch.cat([interact, ctx], dim=1)        # (B, 48+ctx_d, H, W)
        else:
            enc_feat = interact

        correction, rw = self.encoder_router(enc_feat, dev_flat, lead_hour_idx)
        self.last_routing_weights = rw

        # --- Shared Sphere features ---
        sph_feat = self.sph(lat0, lon0, lead_hour_idx, flip_flag)  # (B, d_sph, H, W)

        # --- Regression ---
        refined = self.reg_refinement(correction, interact, mn, spread)  # (B, 6, H, W)
        mu = self.reg_head(sph_feat, refined, consensus_raw)             # (B, 6, H, W)
        sigma = self.sigma_head(spread, sph_feat)                         # (B, 6, H, W)

        # --- Classification ---
        cls_base = self.cls_refinement(spread, interact, dev_n)           # (B, 18, H, W)
        cls_logits = self.cls_head(sph_feat, cls_base)                    # (B, 18, H, W)
        self.last_cls_logits = cls_logits

        return mu, sigma, cls_logits

    def aux_losses(self):
        losses = {}
        if self.last_routing_weights is not None:
            losses['sparsity'] = self.encoder_router.sparsity_loss(
                self.last_routing_weights)
        return losses

    def param_summary(self):
        def _sum(m):
            if m is None:
                return 0
            return sum(p.numel() for p in m.parameters())

        enc = _sum(self.encoder_interact) + _sum(self.encoder_router)
        ctx = _sum(self.context)
        sph = _sum(self.sph)
        reg = _sum(self.reg_refinement) + _sum(self.reg_head) + _sum(self.sigma_head)
        cls = _sum(self.cls_refinement) + _sum(self.cls_head)
        total = sum(p.numel() for p in self.parameters())
        return {'encoder': enc, 'context': ctx, 'sph': sph,
                'regression': reg, 'classification': cls, 'total': total}


# ============================================================
# Quick smoke test
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    print("# Building WenqiongV3SPF (this will precompute SH buffer, may take ~5s)...")
    model = WenqiongV3SPF().eval()
    ps = model.param_summary()
    for k, v in ps.items():
        print(f"  {k:14s}: {v:,}")

    B = 2
    x_norm = torch.randn(B, 24, PATCH_H, PATCH_W)
    x_raw = torch.randn(B, 24, PATCH_H, PATCH_W) * 10
    lead_idx = torch.tensor([0, 3], dtype=torch.long)
    lat0 = torch.tensor([100, 500], dtype=torch.long)
    lon0 = torch.tensor([300, 800], dtype=torch.long)

    with torch.no_grad():
        mu, sigma, cls_logits = model(x_norm, x_raw, lead_idx,
                                       lat0=lat0, lon0=lon0)
    print("\n# Forward pass:")
    print(f"  mu:         {tuple(mu.shape)}")
    print(f"  sigma:      {tuple(sigma.shape)}    range [{sigma.min().item():.3f}, {sigma.max().item():.3f}]")
    print(f"  cls_logits: {tuple(cls_logits.shape)}")
    print("OK: wenqiong_v3_spf smoke test passed.")
