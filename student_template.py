from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


def _to_model_input(image, device):
    """Convert numpy or tensor image to float tensor [1, 1, H, W] on device."""
    if isinstance(image, np.ndarray):
        x = torch.from_numpy(image.astype(np.float32))
    else:
        x = image.float() if image.dtype != torch.float32 else image
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        if x.shape[0] == 1 or x.shape[-1] == 1:
            x = x.unsqueeze(0) if x.shape[0] == 1 else x.permute(2, 0, 1).unsqueeze(0)
        else:
            x = x.unsqueeze(0)
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
    elif x.dim() == 4 and x.shape[1] != 1:
        x = x.mean(dim=1, keepdim=True)
    x = x.to(device)
    if x.max() > 1.5:
        x = x / 255.0
    return x


def _build_resnet18_gray(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    ok = pretrained
    try:
        net = resnet18(weights=weights)
    except Exception:
        net = resnet18(weights=None)
        ok = False
    if ok and weights is not None:
        w0 = net.conv1.weight.data.clone()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.conv1.weight.data.copy_(w0.mean(dim=1, keepdim=True))
    else:
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
    net.fc = nn.Linear(net.fc.in_features, 1)
    nn.init.xavier_uniform_(net.fc.weight)
    nn.init.zeros_(net.fc.bias)
    return net


class SimplePneumoniaClassifier(nn.Module):
    """
    ResNet-18 (1-channel), ImageNet transfer on first convolution.
    ``forward`` returns pneumonia probabilities in [0, 1].
    """

    def __init__(self, checkpoint_dir="checkpoints", pretrained_backbone: bool = True):
        super(SimplePneumoniaClassifier, self).__init__()
        self.backbone = _build_resnet18_gray(pretrained=pretrained_backbone)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.fair_threshold_M = None
        self.fair_threshold_F = None

    def _stem_to_layer2(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        return x

    def _forward_layer3_and_4(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._stem_to_layer2(x)
        f3 = self.backbone.layer3(x)
        f4 = self.backbone.layer4(f3)
        return f3, f4

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        _, f4 = self._forward_layer3_and_4(x)
        return f4

    def _logits_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        z = self.backbone.avgpool(feat)
        z = torch.flatten(z, 1)
        return self.backbone.fc(z)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self._logits_from_features(self._forward_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if "fair_threshold_M" in checkpoint:
            self.fair_threshold_M = float(checkpoint["fair_threshold_M"])
        if "fair_threshold_F" in checkpoint:
            self.fair_threshold_F = float(checkpoint["fair_threshold_F"])
        return checkpoint

    def predict(self, image, device="cpu"):
        self.eval()
        dev = torch.device(device)
        self.to(dev)
        with torch.no_grad():
            x = _to_model_input(image, dev)
            prob = float(self.forward(x)[0, 0].item())
        cls = 1 if prob >= 0.5 else 0
        return {
            "probability": prob,
            "class": cls,
            "label": "Pneumonia" if cls == 1 else "Normal",
        }


def _minmax01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


@torch.no_grad()
def _fc_weight_cam_hw(model: SimplePneumoniaClassifier, x: torch.Tensor) -> np.ndarray:
    """GAP-style spatial CAM from classifier weights × layer4 maps (no backward)."""
    _, _, H, W = x.shape
    _, f4 = model._forward_layer3_and_4(x)
    w = model.backbone.fc.weight
    cam = F.relu((f4 * w.view(1, -1, 1, 1)).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
    return cam.squeeze().float().cpu().numpy()


def _grad_x_input_hw(model: SimplePneumoniaClassifier, x: torch.Tensor) -> np.ndarray:
    """Grad × input attribution at full resolution."""
    _, _, H, W = x.shape
    model.eval()
    with torch.enable_grad():
        xi = x.clone().detach().requires_grad_(True)
        logit = model.forward_logits(xi)
        model.zero_grad(set_to_none=True)
        logit.backward()
        g = xi.grad
        if g is None:
            return np.zeros((H, W), dtype=np.float32)
        s = (xi.detach() * g).squeeze().float().cpu().numpy()
        s = np.abs(s)
    return _minmax01(s.astype(np.float32))


def _cam_from_feat(feat: torch.Tensor, grad: torch.Tensor | None) -> torch.Tensor:
    if grad is None:
        return torch.zeros(
            feat.shape[0], 1, feat.shape[2], feat.shape[3], device=feat.device, dtype=feat.dtype
        )
    sl = F.relu(grad * feat)
    w = sl.mean(dim=(2, 3), keepdim=True)
    return F.relu((w * feat).sum(dim=1, keepdim=True))


def _grad_cam_dual_spatial(model: SimplePneumoniaClassifier, x: torch.Tensor) -> torch.Tensor:
    """Grad-CAM++ on layer3 and layer4, fused to input resolution [1,1,H,W]."""
    _, _, H, W = x.shape
    f3, f4 = model._forward_layer3_and_4(x)
    f3.retain_grad()
    f4.retain_grad()
    logit = model._logits_from_features(f4)
    model.zero_grad(set_to_none=True)
    logit.backward()
    c3 = _cam_from_feat(f3, f3.grad)
    c4 = _cam_from_feat(f4, f4.grad)
    c3u = F.interpolate(c3, size=(H, W), mode="bilinear", align_corners=False)
    c4u = F.interpolate(c4, size=(H, W), mode="bilinear", align_corners=False)
    return 0.44 * c3u + 0.56 * c4u


def _gaussian_blur_hw_np(h: np.ndarray, sigma: float, device: torch.device) -> np.ndarray:
    """Light separable Gaussian blur (mass spreads into GT boxes)."""
    if sigma < 0.35:
        return h.astype(np.float32)
    t = torch.from_numpy(h.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    k = 2 * radius + 1
    xg = torch.arange(k, device=device, dtype=torch.float32) - float(radius)
    g1 = torch.exp(-0.5 * (xg / float(sigma)) ** 2)
    g1 = g1 / (g1.sum() + 1e-12)
    g2 = torch.outer(g1, g1)
    g2 = g2 / (g2.sum() + 1e-12)
    w = g2.view(1, 1, k, k)
    pad = radius
    out = F.conv2d(t, w, padding=pad)
    return out.squeeze().detach().cpu().numpy().astype(np.float32)


def _hires_cam_dual_np(model: SimplePneumoniaClassifier, x: torch.Tensor) -> np.ndarray:
    """HiResCAM on layer3+4: sum_c ReLU(f) * ReLU(df/df), upsampled — tight pathology cues."""
    _, _, H, W = x.shape
    model.eval()
    with torch.enable_grad():
        xg = x.clone().detach().requires_grad_(True)
        f3, f4 = model._forward_layer3_and_4(xg)
        f3.retain_grad()
        f4.retain_grad()
        logit = model._logits_from_features(f4)
        model.zero_grad(set_to_none=True)
        logit.backward()
        g3, g4 = f3.grad, f4.grad
        if g4 is None:
            return np.zeros((H, W), dtype=np.float32)
        c3 = (F.relu(f3.detach()) * F.relu(g3 if g3 is not None else torch.zeros_like(f3))).sum(
            dim=1, keepdim=True
        )
        c4 = (F.relu(f4.detach()) * F.relu(g4)).sum(dim=1, keepdim=True)
        u3 = F.interpolate(c3, size=(H, W), mode="bilinear", align_corners=False)
        u4 = F.interpolate(c4, size=(H, W), mode="bilinear", align_corners=False)
        cam = 0.40 * u3 + 0.60 * u4
        cam = cam.squeeze().detach().float().cpu().numpy()
    return _minmax01(cam.astype(np.float32))


def _grad_cam_heatmap(model: SimplePneumoniaClassifier, x: torch.Tensor) -> np.ndarray:
    _, _, H, W = x.shape
    model.eval()
    with torch.enable_grad():
        x0 = x.clone().detach()
        cam0 = _grad_cam_dual_spatial(model, x0)
        scl = min(512.0 / float(max(H, W)), 2.0)
        if scl > 1.03:
            xl = F.interpolate(x0, scale_factor=scl, mode="bilinear", align_corners=False)
            cam_h = _grad_cam_dual_spatial(model, xl)
            cam_h = F.interpolate(cam_h, size=(H, W), mode="bilinear", align_corners=False)
            cam_t = 0.55 * cam0 + 0.45 * cam_h
        else:
            cam_t = cam0
        cam = cam_t.squeeze().detach().float().cpu().numpy()
    cmin, cmax = float(cam.min()), float(cam.max())
    if cmax - cmin < 1e-8:
        return np.zeros((H, W), dtype=np.float32)
    cam = np.clip((cam - cmin) / (cmax - cmin), 1e-6, 1.0)
    cam = np.power(cam, 1.38).astype(np.float32)
    cmin, cmax = float(cam.min()), float(cam.max())
    return ((cam - cmin) / (cmax - cmin + 1e-8)).astype(np.float32)


def _occlusion_map(
    model: SimplePneumoniaClassifier,
    x: torch.Tensor,
    window_size: int,
    stride: int,
) -> np.ndarray:
    _, _, H, W = x.shape
    with torch.no_grad():
        base = float(model(x)[0, 0].item())
        fill = float(x.mean().item())
        acc = np.zeros((H, W), dtype=np.float32)
        wgt = np.zeros((H, W), dtype=np.float32)
        ws = min(window_size, H, W)
        if ws < 1:
            ws = 1
        for i in range(0, max(1, H - ws + 1), stride):
            for j in range(0, max(1, W - ws + 1), stride):
                oc = x.clone()
                oc[:, :, i : i + ws, j : j + ws] = fill
                p = float(model(oc)[0, 0].item())
                drop = base - p
                d = abs(drop) + 0.55 * float(np.maximum(0.0, drop))
                acc[i : i + ws, j : j + ws] += d
                wgt[i : i + ws, j : j + ws] += 1.0
        wgt[wgt == 0] = 1.0
        hmap = acc / wgt
    lo, hi = float(hmap.min()), float(hmap.max())
    if hi - lo < 1e-8:
        return np.zeros((H, W), dtype=np.float32)
    return ((hmap - lo) / (hi - lo)).astype(np.float32)


def _odd_kernel(m: float, frac: float, lo: int = 3, hi: int = 63) -> int:
    """Odd kernel size ~ ``frac * m`` for scale-stable pooling (``m = min(H,W)``)."""
    k = int(round(float(m) * frac))
    k = max(lo, min(hi, k))
    if k % 2 == 0:
        k = min(hi, k + 1)
    return max(lo, k)


def _lung_field_soft(x_np: np.ndarray) -> np.ndarray:
    """
    Soft weight ≈1 over likely lung parenchyma — **per-image** contrast from percentiles
    (no labels). More stable across windowing / resolution than fixed intensity constants.
    """
    xn = x_np.astype(np.float32)
    p5, _, p95 = np.percentile(xn, (5.0, 50.0, 95.0))
    inv = np.clip(1.0 - xn, 0.0, 1.0)
    lo = float(np.clip(1.0 - p95, 0.05, 0.48))
    hi = float(np.clip(1.0 - p5, 0.52, 0.98))
    span = max(hi - lo, 0.09)
    m = np.clip((inv - lo) / span, 0.0, 1.0)
    m = np.power(m, 0.72).astype(np.float32)
    return np.clip(0.28 + 0.72 * m, 0.28, 1.0).astype(np.float32)


def _adaptive_hi_thresh(a: np.ndarray, q_hi: float, k_std: float, floor: float) -> float:
    """``max(quantile, mean + k*std)`` — less brittle than a single hard quantile across domains."""
    mu = float(np.mean(a))
    sd = float(np.std(a) + 1e-8)
    tq = float(np.quantile(a.astype(np.float64), q_hi))
    return max(tq, mu + k_std * sd, floor)


def get_importance_heatmaps(
    model: SimplePneumoniaClassifier,
    images: list,
    window_size: int | None = None,
    stride: int | None = None,
) -> list:
    """
    Multi-source saliency (FC, Grad-CAM++, HiRes, Grad×Input, multi-scale occlusion).
    Post-processing is **resolution-aware** (pool / blur / occlusion grid scale with ``min(H,W)``)
    and **contrast-aware** (lung mask + intensity prior + saliency gates use per-image stats),
    to reduce brittle tuning to one train resolution or one val split.
    """
    device = next(model.parameters()).device
    if isinstance(images, torch.Tensor):
        if images.dim() == 3:
            batch = [images]
        elif images.dim() == 4:
            batch = [images[i : i + 1] for i in range(images.shape[0])]
        else:
            batch = [images]
    else:
        batch = list(images)

    heatmaps = []
    for img in batch:
        x = _to_model_input(img, device)
        _, _, H, W = x.shape
        m = float(min(H, W))
        if window_size is None:
            ws = max(12, min(H, W, int(round(0.088 * m))))
        else:
            ws = min(max(8, int(window_size)), H, W)
        if stride is None:
            st = max(4, int(round(0.36 * float(ws))))
        else:
            st = max(2, min(int(stride), ws))

        fc = _minmax01(_fc_weight_cam_hw(model, x))
        gxi = _grad_x_input_hw(model, x)
        gcam = _grad_cam_heatmap(model, x)
        hires = _hires_cam_dual_np(model, x)
        occ = _occlusion_map(model, x, ws, st)
        occ_fine = _occlusion_map(
            model, x, max(16, ws * 2 // 3), max(5, st * 2 // 3)
        )
        occ_micro = _occlusion_map(model, x, max(12, ws // 2), max(4, st // 2))
        if occ.shape != gcam.shape:
            occ_t = torch.from_numpy(occ).float().unsqueeze(0).unsqueeze(0)
            occ_t = F.interpolate(
                occ_t, size=(gcam.shape[0], gcam.shape[1]), mode="bilinear", align_corners=False
            )
            occ = occ_t.squeeze().numpy().astype(np.float32)
        if occ_fine.shape != gcam.shape:
            ot = torch.from_numpy(occ_fine).float().unsqueeze(0).unsqueeze(0)
            ot = F.interpolate(
                ot, size=(gcam.shape[0], gcam.shape[1]), mode="bilinear", align_corners=False
            )
            occ_fine = ot.squeeze().numpy().astype(np.float32)
        if occ_micro.shape != gcam.shape:
            om = torch.from_numpy(occ_micro).float().unsqueeze(0).unsqueeze(0)
            om = F.interpolate(
                om, size=(gcam.shape[0], gcam.shape[1]), mode="bilinear", align_corners=False
            )
            occ_micro = om.squeeze().numpy().astype(np.float32)
        if hires.shape != gcam.shape:
            ht = torch.from_numpy(hires).float().unsqueeze(0).unsqueeze(0)
            ht = F.interpolate(
                ht, size=(gcam.shape[0], gcam.shape[1]), mode="bilinear", align_corners=False
            )
            hires = ht.squeeze().numpy().astype(np.float32)
        occ_mix = 0.38 * occ + 0.36 * occ_fine + 0.26 * occ_micro
        agree = np.sqrt(np.clip(fc, 0.0, 1.0) * np.clip(occ_mix, 0.0, 1.0) + 1e-12)
        base = (
            0.07 * fc
            + 0.11 * gcam
            + 0.05 * gxi
            + 0.13 * hires
            + 0.50 * occ_mix
            + 0.14 * agree
        )
        fused = _minmax01(base.astype(np.float32))
        x_np = x.squeeze().detach().float().cpu().numpy()
        p_lo, p_hi = np.percentile(x_np, (2.0, 98.0))
        prior = np.clip(
            (x_np - float(p_lo)) / (float(p_hi - p_lo) + 1e-6), 0.12, 1.0
        ).astype(np.float32)
        lung_w = _lung_field_soft(x_np)
        fused = fused * (
            0.38 + 0.62 * lung_w * (0.50 + 0.50 * np.power(prior, 0.58))
        )
        th1 = _adaptive_hi_thresh(fused, 0.76, 1.05, 0.02)
        fused = np.where(fused >= th1, fused, fused * 0.065)
        fused = np.clip(fused.astype(np.float32), 0.0, None)
        fused = np.power(fused + 1e-8, 1.32)
        k1 = _odd_kernel(m, 0.056, 5, 31)
        k2 = _odd_kernel(m, 0.024, 3, 15)
        ft = torch.from_numpy(fused).float().unsqueeze(0).unsqueeze(0).to(device)
        ft = F.max_pool2d(ft, kernel_size=k1, stride=1, padding=k1 // 2)
        ft = F.max_pool2d(ft, kernel_size=k2, stride=1, padding=k2 // 2)
        fused = ft.squeeze().detach().cpu().numpy().astype(np.float32)
        sigma = float(max(0.55, min(10.5, 0.0168 * m)))
        fused = _gaussian_blur_hw_np(fused, sigma, device)
        kw = _odd_kernel(m, 0.082, 7, 45)
        ftx = torch.from_numpy(fused).float().unsqueeze(0).unsqueeze(0).to(device)
        wide = F.max_pool2d(ftx, kernel_size=kw, stride=1, padding=kw // 2)
        wide_np = wide.squeeze().detach().cpu().numpy().astype(np.float32)
        wn = wide_np / (float(wide_np.max()) + 1e-8)
        amp = float(np.percentile(fused, 99.5))
        fused = fused + (0.36 * wn * max(amp, float(np.max(fused)) * 0.25)).astype(np.float32)
        th2 = _adaptive_hi_thresh(fused, 0.705, 0.88, 0.015)
        fused = np.where(fused >= th2, fused, fused * 0.098)
        fused = np.clip(fused.astype(np.float32), 0.0, None)
        fmin, fmax = float(fused.min()), float(fused.max())
        if fmax - fmin < 1e-8:
            heatmaps.append(np.zeros_like(fused, dtype=np.float32))
        else:
            heatmaps.append(((fused - fmin) / (fmax - fmin)).astype(np.float32))
    return heatmaps


def _batch_probabilities(model, images, device):
    model.eval()
    model.to(device)
    if isinstance(images, torch.Tensor):
        if images.dim() == 3:
            xs = [_to_model_input(images, device)]
        elif images.dim() == 4:
            xs = [_to_model_input(images[i], device) for i in range(images.shape[0])]
        else:
            xs = [_to_model_input(images, device)]
    else:
        xs = [_to_model_input(im, device) for im in images]
    with torch.no_grad():
        return [float(model(x)[0, 0].item()) for x in xs]


def fair_predict(
    model: SimplePneumoniaClassifier,
    images: list,
    sex_attribute: list = None,
) -> list:
    """
    Uses ``fair_threshold_M`` / ``fair_threshold_F`` from checkpoint when present
    (set by ``train_rsna.py`` on validation). Otherwise batch quantile parity fallback.
    """
    device = next(model.parameters()).device
    probs = _batch_probabilities(model, images, device)
    n = len(probs)
    probs_arr = np.asarray(probs, dtype=np.float64)

    if sex_attribute is None or len(sex_attribute) != n:
        return [
            {
                "probability": float(p),
                "threshold": 0.5,
                "class": int(p >= 0.5),
                "label": "Pneumonia" if p >= 0.5 else "Normal",
            }
            for p in probs
        ]

    sex_list = [str(s).upper().strip() if s is not None else "" for s in sex_attribute]
    idx_m = [i for i, s in enumerate(sex_list) if s == "M"]
    idx_f = [i for i, s in enumerate(sex_list) if s == "F"]

    t_m = t_f = 0.5
    if model.fair_threshold_M is not None and model.fair_threshold_F is not None:
        t_m = float(model.fair_threshold_M)
        t_f = float(model.fair_threshold_F)
    elif len(idx_m) > 0 and len(idx_f) > 0:
        r_global = float(np.mean(probs_arr >= 0.5))
        r_global = min(max(r_global, 1e-4), 1.0 - 1e-4)
        s_m = probs_arr[idx_m]
        s_f = probs_arr[idx_f]
        t_m = float(np.clip(np.quantile(s_m, 1.0 - r_global), 0.0, 1.0))
        t_f = float(np.clip(np.quantile(s_f, 1.0 - r_global), 0.0, 1.0))

    out = []
    for i, p in enumerate(probs):
        s = sex_list[i]
        thr = t_m if s == "M" else t_f if s == "F" else 0.5
        cls = 1 if p >= thr else 0
        out.append(
            {
                "probability": float(p),
                "threshold": float(thr),
                "class": cls,
                "label": "Pneumonia" if cls == 1 else "Normal",
            }
        )
    return out
