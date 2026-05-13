"""
Train SimplePneumoniaClassifier on RSNA, save checkpoints/best_model.pt,
calibrate fair thresholds on validation, log AUC and heatmap coverage.

Expected layout (Kaggle RSNA Pneumonia Detection Challenge):
  <data_root>/stage_2_train_images/*.dcm
  <data_root>/stage_2_train_labels.csv

Default ``--data_root`` is the directory that contains ``train_rsna.py`` (your project root),
so training works even if the shell cwd is not the project folder.

Usage (conda env with torch, pydicom, pandas, sklearn):
  python train_rsna.py --epochs 14
  python train_rsna.py --data_root C:\\path\\to\\folder_with_stage_2_train_images --epochs 14
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize as tv_resize
from tqdm import tqdm

from student_template import SimplePneumoniaClassifier, get_importance_heatmaps

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def read_dicom(path: str) -> tuple[np.ndarray, str]:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    sex = str(getattr(ds, "PatientSex", "O")).upper().strip()
    if sex not in ("M", "F"):
        sex = "O"
    return arr, sex


def window_percentile(img: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(img, (p_low, p_high))
    return np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def scale_boxes(
    boxes: list[tuple[float, float, float, float]], h0: int, w0: int, out: int
) -> list[tuple[int, int, int, int]]:
    sy, sx = out / h0, out / w0
    out_boxes = []
    for x, y, bw, bh in boxes:
        x0 = int(max(0, min(out - 1, round(x * sx))))
        y0 = int(max(0, min(out - 1, round(y * sy))))
        x1 = int(max(0, min(out, round((x + bw) * sx))))
        y1 = int(max(0, min(out, round((y + bh) * sy))))
        if x1 <= x0:
            x1 = min(out, x0 + 1)
        if y1 <= y0:
            y1 = min(out, y0 + 1)
        out_boxes.append((x0, y0, x1 - x0, y1 - y0))
    return out_boxes


def importance_in_box_ratio(hmap: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> float:
    """Fraction of total heatmap mass inside union of GT boxes (grader-style)."""
    H, W = hmap.shape
    mask = np.zeros((H, W), dtype=bool)
    for x, y, bw, bh in boxes:
        x0, x1 = max(0, x), min(W, x + bw)
        y0, y1 = max(0, y), min(H, y + bh)
        mask[y0:y1, x0:x1] = True
    total = float(hmap.sum()) + 1e-12
    inside = float(hmap[mask].sum()) if mask.any() else 0.0
    return inside / total


class RSNADataset(Dataset):
    def __init__(
        self,
        patient_ids: list[str],
        images_dir: str,
        label_df: pd.DataFrame,
        image_size: int,
    ):
        self.patient_ids = patient_ids
        self.images_dir = images_dir
        self.image_size = image_size
        self.targets = label_df.groupby("patientId")["Target"].max().to_dict()
        self.box_rows = label_df[label_df["Target"] == 1]

    def _boxes_for(self, pid: str) -> list[tuple[float, float, float, float]]:
        rows = self.box_rows[self.box_rows["patientId"] == pid]
        boxes = []
        for _, r in rows.iterrows():
            if pd.notna(r.get("x")):
                boxes.append((float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"])))
        return boxes

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        pid = self.patient_ids[idx]
        path = os.path.join(self.images_dir, f"{pid}.dcm")
        arr, sex = read_dicom(path)
        h0, w0 = arr.shape
        arr = window_percentile(arr)
        t = torch.from_numpy(arr).unsqueeze(0).to(dtype=torch.float32)
        t = tv_resize(t, [self.image_size, self.image_size], antialias=True)
        y = int(self.targets.get(pid, 0))
        boxes_orig = self._boxes_for(pid)
        boxes = scale_boxes(boxes_orig, h0, w0, self.image_size)
        return t, y, sex, pid, boxes


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    sexes = [b[2] for b in batch]
    pids = [b[3] for b in batch]
    boxes = [b[4] for b in batch]
    return xs, ys, sexes, pids, boxes


def search_fair_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    sexes: list[str],
) -> tuple[float, float, dict]:
    """Grid search group thresholds to reduce PR and TPR disparity on validation."""
    mask_mf = np.array([s in ("M", "F") for s in sexes])
    probs, y = probs[mask_mf], y[mask_mf]
    sexes = [s for s, m in zip(sexes, mask_mf) if m]
    if len(y) < 20 or y.sum() < 3:
        return 0.5, 0.5, {"note": "too_small"}

    def metrics(pred: np.ndarray) -> tuple[float, float]:
        m = np.array([s == "M" for s in sexes])
        f = np.array([s == "F" for s in sexes])
        pr_m = pred[m].mean() if m.any() else 0.0
        pr_f = pred[f].mean() if f.any() else 0.0
        pr_d = abs(pr_m - pr_f)

        def tpr(sub):
            pos = (y == 1) & sub
            if pos.sum() == 0:
                return 0.0
            return ((pred == 1) & pos).sum() / pos.sum()

        tpr_d = abs(tpr(m) - tpr(f)) if m.any() and f.any() else 0.0
        return pr_d, tpr_d

    best_tm, best_tf = 0.5, 0.5
    best_score = 1e9
    best_meta = {}
    grid = np.linspace(0.12, 0.92, 45)
    for tm in grid:
        for tf in grid:
            pred = np.array([int(probs[i] >= (tm if sexes[i] == "M" else tf)) for i in range(len(probs))])
            pr_d, tpr_d = metrics(pred)
            score = pr_d + 12.0 * tpr_d
            hard = max(0.0, pr_d - 0.01) + max(0.0, tpr_d - 0.005)
            score += 5.0 * hard
            if score < best_score:
                best_score = score
                best_tm, best_tf = float(tm), float(tf)
                best_meta = {"pr_d": pr_d, "tpr_d": tpr_d}
    for tm in np.linspace(max(0.05, best_tm - 0.04), min(0.95, best_tm + 0.04), 17):
        for tf in np.linspace(max(0.05, best_tf - 0.04), min(0.95, best_tf + 0.04), 17):
            pred = np.array([int(probs[i] >= (tm if sexes[i] == "M" else tf)) for i in range(len(probs))])
            pr_d, tpr_d = metrics(pred)
            score = pr_d + 12.0 * tpr_d
            hard = max(0.0, pr_d - 0.01) + max(0.0, tpr_d - 0.005)
            score += 5.0 * hard
            if score < best_score:
                best_score = score
                best_tm, best_tf = float(tm), float(tf)
                best_meta = {"pr_d": pr_d, "tpr_d": tpr_d}
    return best_tm, best_tf, best_meta


@torch.no_grad()
def evaluate_auc(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    for xb, yb, _, _, _ in loader:
        xb = xb.to(device)
        logit = model.forward_logits(xb)
        prob = torch.sigmoid(logit).squeeze(1).cpu().numpy()
        all_p.append(prob)
        all_y.append(yb.numpy())
    all_p = np.concatenate(all_p)
    all_y = np.concatenate(all_y)
    if all_y.sum() in (0, len(all_y)):
        return 0.5
    return float(roc_auc_score(all_y, all_p))


def effective_heatmap_batches(loader_len: int, cap: int) -> int:
    """``cap <= 0`` means use all validation batches (expensive during training)."""
    if cap <= 0:
        return loader_len
    return min(int(cap), loader_len)


@torch.no_grad()
def evaluate_heatmap_coverage(model, loader, device, max_batches: int = 8):
    model.eval()
    ratios = []
    n = 0
    for bi, (xb, yb, _, _, boxes_list) in enumerate(loader):
        if bi >= max_batches:
            break
        for i in range(xb.shape[0]):
            if int(yb[i].item()) != 1:
                continue
            boxes = boxes_list[i]
            if not boxes:
                continue
            x = xb[i : i + 1].to(device)
            hm = get_importance_heatmaps(model, [x.cpu().numpy()[0, 0]])[0]
            ratios.append(importance_in_box_ratio(hm, boxes))
            n += 1
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def train_one_epoch(model, loader, opt, device, pos_weight, use_focal: bool):
    model.train()
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32)
    )
    sfl = None
    if use_focal:
        try:
            from torchvision.ops import sigmoid_focal_loss as _sfl

            sfl = _sfl
        except ImportError:
            sfl = None
    tot = 0.0
    for xb, yb, _, _, _ in tqdm(loader, desc="train", leave=False):
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        opt.zero_grad(set_to_none=True)
        logit = model.forward_logits(xb)
        if use_focal and sfl is not None:
            loss = sfl(
                logit.squeeze(1),
                yb.squeeze(1),
                alpha=0.32,
                gamma=2.0,
                reduction="mean",
            )
        else:
            loss = crit(logit, yb)
        loss.backward()
        opt.step()
        tot += loss.item() * xb.size(0)
    return tot / len(loader.dataset)


def resolve_labels_csv_path(data_root: str) -> str | None:
    """
    Kaggle layout: ``<root>/stage_2_train_labels.csv`` is a CSV file.
    Some clients create a *folder* with that name; then the file may live inside it.
    """
    p = os.path.join(data_root, "stage_2_train_labels.csv")
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        names = sorted(f for f in os.listdir(p) if f.lower().endswith(".csv"))
        for n in names:
            if "train" in n.lower() and "label" in n.lower():
                return os.path.join(p, n)
        if names:
            return os.path.join(p, names[0])
    for n in ("train_labels.csv", "labels.csv"):
        q = os.path.join(data_root, n)
        if os.path.isfile(q):
            return q
    return None


def _calibrate_only(args: argparse.Namespace) -> None:
    """Reload best weights, re-fit fairness thresholds on the same val split, re-save checkpoint."""
    if args.data_root is None or args.data_root == "":
        data_root = _PROJECT_ROOT
    else:
        data_root = os.path.abspath(args.data_root)
    img_dir = os.path.join(data_root, "stage_2_train_images")
    csv_path = resolve_labels_csv_path(data_root)
    if not (os.path.isdir(img_dir) and csv_path):
        alt_img = os.path.join(_PROJECT_ROOT, "stage_2_train_images")
        if os.path.isdir(alt_img):
            data_root = _PROJECT_ROOT
            img_dir = alt_img
            csv_path = resolve_labels_csv_path(data_root)
    if not os.path.isdir(img_dir) or not csv_path:
        raise FileNotFoundError("RSNA train images / labels not found for calibration.")

    labels = pd.read_csv(csv_path)
    all_files = [f[:-4] for f in os.listdir(img_dir) if f.endswith(".dcm")]
    tg = labels.groupby("patientId")["Target"].max()
    patients = sorted(set(all_files) & set(tg.index))
    y_all = np.array([int(tg.loc[p]) for p in patients], dtype=np.int64)
    if args.max_patients is not None and len(patients) > args.max_patients:
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=args.max_patients, random_state=42
        )
        idx, _ = next(sss.split(np.zeros(len(patients)), y_all))
        patients = [patients[i] for i in idx]
        y_all = np.array([int(tg.loc[p]) for p in patients], dtype=np.int64)
    tr_ids, va_ids = train_test_split(
        patients, test_size=0.15, stratify=y_all, random_state=42
    )
    ckpt_path = os.path.join(_PROJECT_ROOT, "checkpoints", "best_model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing {ckpt_path}")
    try:
        ck_head = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck_head = torch.load(ckpt_path, map_location="cpu")
    ims = int(ck_head.get("image_size", 224))
    val_ld = DataLoader(
        RSNADataset(va_ids, img_dir, labels, ims),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    device = torch.device(args.device)
    model = SimplePneumoniaClassifier(pretrained_backbone=False).to(device)
    prev = model.load_checkpoint(ckpt_path)
    model.eval()
    probs_list, y_list, sex_list = [], [], []
    with torch.no_grad():
        for xb, yb, sexes, _, _ in val_ld:
            xb = xb.to(device)
            pr = torch.sigmoid(model.forward_logits(xb)).squeeze(1).cpu().numpy()
            probs_list.append(pr)
            y_list.append(yb.numpy())
            sex_list.extend(sexes)
    probs = np.concatenate(probs_list)
    y = np.concatenate(y_list)
    val_auc_final = float(roc_auc_score(y, probs)) if y.min() < y.max() else 0.5
    tm, tf, meta = search_fair_thresholds(probs, y, sex_list)
    model.fair_threshold_M = tm
    model.fair_threshold_F = tf
    print(f"calibrate_only: val_auc={val_auc_final:.4f}  fair_M={tm:.4f} fair_F={tf:.4f}  meta={meta}")
    out = {
        "epoch": int(prev.get("epoch", 0)),
        "best_epoch": int(prev.get("best_epoch", prev.get("epoch", 0))),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": prev.get("optimizer_state_dict", {}),
        "val_auc": val_auc_final,
        "fair_threshold_M": tm,
        "fair_threshold_F": tf,
        "image_size": int(prev.get("image_size", ims)),
    }
    torch.save(out, ckpt_path)
    print(f"updated {ckpt_path}")


def _eval_only(args: argparse.Namespace) -> None:
    """Load ``best_model.pt``, report val AUC and mean heatmap-in-box coverage (current code)."""
    if args.data_root is None or args.data_root == "":
        data_root = _PROJECT_ROOT
    else:
        data_root = os.path.abspath(args.data_root)
    img_dir = os.path.join(data_root, "stage_2_train_images")
    csv_path = resolve_labels_csv_path(data_root)
    if not (os.path.isdir(img_dir) and csv_path):
        alt_img = os.path.join(_PROJECT_ROOT, "stage_2_train_images")
        if os.path.isdir(alt_img):
            data_root = _PROJECT_ROOT
            img_dir = alt_img
            csv_path = resolve_labels_csv_path(data_root)
    if not os.path.isdir(img_dir) or not csv_path:
        raise FileNotFoundError("RSNA train images / labels not found.")

    labels = pd.read_csv(csv_path)
    all_files = [f[:-4] for f in os.listdir(img_dir) if f.endswith(".dcm")]
    tg = labels.groupby("patientId")["Target"].max()
    patients = sorted(set(all_files) & set(tg.index))
    y_all = np.array([int(tg.loc[p]) for p in patients], dtype=np.int64)
    if args.max_patients is not None and len(patients) > args.max_patients:
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=args.max_patients, random_state=42
        )
        idx, _ = next(sss.split(np.zeros(len(patients)), y_all))
        patients = [patients[i] for i in idx]
        y_all = np.array([int(tg.loc[p]) for p in patients], dtype=np.int64)
    if len(np.unique(y_all)) < 2:
        raise RuntimeError("Only one class in subset; increase --max_patients or use full data.")
    tr_ids, va_ids = train_test_split(
        patients, test_size=0.15, stratify=y_all, random_state=42
    )
    ckpt_path = os.path.join(_PROJECT_ROOT, "checkpoints", "best_model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing {ckpt_path}")
    try:
        ck_head = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck_head = torch.load(ckpt_path, map_location="cpu")
    ims = int(ck_head.get("image_size", 224))
    val_ld = DataLoader(
        RSNADataset(va_ids, img_dir, labels, ims),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    device = torch.device(args.device)
    model = SimplePneumoniaClassifier(pretrained_backbone=False).to(device)
    model.load_checkpoint(ckpt_path)
    auc = evaluate_auc(model, val_ld, device)
    hm_batches = effective_heatmap_batches(len(val_ld), args.hm_max_batches)
    cov = evaluate_heatmap_coverage(model, val_ld, device, max_batches=hm_batches)
    print(
        f"eval_only: val_patients={len(va_ids)}  val_auc={auc:.4f}  "
        f"val_hm_cov~{cov:.4f}  (hm_batches={hm_batches}/{len(val_ld)})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Folder with stage_2_train_images/ and stage_2_train_labels.csv (default: folder containing this script)",
    )
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stopping: stop after this many epochs without val AUC improvement (0=disabled)",
    )
    ap.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Stratified random subset for smoke tests (default: use all patients with labels)",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--image_size", type=int, default=320)
    ap.add_argument(
        "--no_focal",
        action="store_true",
        help="Use plain BCEWithLogitsLoss instead of sigmoid focal loss",
    )
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument(
        "--calibrate_only",
        action="store_true",
        help="Load checkpoints/best_model.pt, recompute fair thresholds on val, save (no training)",
    )
    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="Load checkpoints/best_model.pt, print val AUC and heatmap coverage (no training)",
    )
    ap.add_argument(
        "--hm_max_batches",
        type=int,
        default=None,
        help="Max val DataLoader batches for heatmap coverage. "
        "<=0 means all batches. Default: 24 during training, all batches for --eval_only.",
    )
    args = ap.parse_args()
    if args.hm_max_batches is None:
        args.hm_max_batches = 0 if args.eval_only else 24

    if args.calibrate_only:
        _calibrate_only(args)
        return
    if args.eval_only:
        _eval_only(args)
        return

    if args.data_root is None or args.data_root == "":
        data_root = _PROJECT_ROOT
    else:
        data_root = os.path.abspath(args.data_root)
    img_dir = os.path.join(data_root, "stage_2_train_images")
    csv_path = resolve_labels_csv_path(data_root)
    if not (os.path.isdir(img_dir) and csv_path):
        alt_img = os.path.join(_PROJECT_ROOT, "stage_2_train_images")
        if os.path.isdir(alt_img):
            data_root = _PROJECT_ROOT
            img_dir = alt_img
            csv_path = resolve_labels_csv_path(data_root)
    if not os.path.isdir(img_dir) or not csv_path:
        raise FileNotFoundError(
            "Need RSNA train data under data_root:\n"
            f"  - directory: {os.path.join(data_root, 'stage_2_train_images')}\n"
            f"  - labels CSV: {os.path.join(data_root, 'stage_2_train_labels.csv')} (file), "
            "or a .csv inside a folder with that name.\n"
            f"Resolved data_root={data_root!r}"
        )

    labels = pd.read_csv(csv_path)
    all_files = [f[:-4] for f in os.listdir(img_dir) if f.endswith(".dcm")]
    tg = labels.groupby("patientId")["Target"].max()
    patients = sorted(set(all_files) & set(tg.index))
    if len(patients) < 100:
        raise RuntimeError(f"Too few patients ({len(patients)}). Check data_root.")

    y_all = np.array([int(tg.loc[p]) for p in patients], dtype=np.int64)
    if args.max_patients is not None and len(patients) > args.max_patients:
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=args.max_patients, random_state=42
        )
        idx, _ = next(sss.split(np.zeros(len(patients)), y_all))
        patients = [patients[i] for i in idx]
        y_all = np.array([int(tg.loc[p]) for p in patients], dtype=np.int64)

    if len(np.unique(y_all)) < 2:
        raise RuntimeError(
            "Only one class in patient subset; increase data or disable --max_patients."
        )
    tr_ids, va_ids = train_test_split(
        patients, test_size=0.15, stratify=y_all, random_state=42
    )

    train_ds = RSNADataset(tr_ids, img_dir, labels, args.image_size)
    val_ds = RSNADataset(va_ids, img_dir, labels, args.image_size)
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = SimplePneumoniaClassifier(
        pretrained_backbone=not args.no_pretrained,
    ).to(device)

    n_pos = max(1.0, float(y_all.sum()))
    n_neg = float(len(patients) - y_all.sum())
    pos_weight = max(1.0, n_neg / n_pos)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    best_auc = 0.0
    best_state = None
    best_epoch = 0
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_ld, opt, device, pos_weight, use_focal=not args.no_focal
        )
        val_auc = evaluate_auc(model, val_ld, device)
        sched.step()
        hm_batches = effective_heatmap_batches(len(val_ld), args.hm_max_batches)
        cov = evaluate_heatmap_coverage(
            model, val_ld, device, max_batches=hm_batches
        )
        print(
            f"epoch {ep:02d}  train_loss={tr_loss:.4f}  val_auc={val_auc:.4f}  "
            f"val_hm_cov~{cov:.4f}  (hm_batches={hm_batches}/{len(val_ld)})"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"early stop at epoch {ep} (no val AUC gain for {args.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Fairness thresholds on validation
    model.eval()
    probs_list, y_list, sex_list = [], [], []
    with torch.no_grad():
        for xb, yb, sexes, _, _ in val_ld:
            xb = xb.to(device)
            pr = torch.sigmoid(model.forward_logits(xb)).squeeze(1).cpu().numpy()
            probs_list.append(pr)
            y_list.append(yb.numpy())
            sex_list.extend(sexes)
    probs = np.concatenate(probs_list)
    y = np.concatenate(y_list)
    val_auc_final = float(roc_auc_score(y, probs)) if y.min() < y.max() else 0.5
    tm, tf, meta = search_fair_thresholds(probs, y, sex_list)
    model.fair_threshold_M = tm
    model.fair_threshold_F = tf
    print(f"best_val_auc={best_auc:.4f}  final_val_auc={val_auc_final:.4f}")
    print(f"fair_threshold_M={tm:.4f}  fair_threshold_F={tf:.4f}  meta={meta}")

    ckpt_dir = model.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "best_model.pt")
    torch.save(
        {
            "epoch": best_epoch,
            "best_epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "val_auc": val_auc_final,
            "fair_threshold_M": tm,
            "fair_threshold_F": tf,
            "image_size": int(args.image_size),
        },
        path,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
