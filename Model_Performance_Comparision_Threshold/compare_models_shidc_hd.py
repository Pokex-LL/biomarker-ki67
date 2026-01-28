import os
import sys
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from skimage.measure import regionprops

from cellpose import models as cp_models, io as cp_io


IDX_TO_CLASS_DEFAULT = {0: "background", 1: "ki67_neg", 2: "ki67_pos"}
GT_MAP_DEFAULT = {1: "ki67_pos", 2: "ki67_neg"} 


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def build_model(arch: str, num_classes: int, device: torch.device):
    arch = arch.lower().strip()
    if arch == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        in_features = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_features, num_classes)
    elif arch == "resnext50":
        m = models.resnext50_32x4d(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"arch no soportada: {arch}")
    return m.to(device)


def load_checkpoint_state_dict(ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and any(k.startswith("classifier") or k.startswith("fc") for k in ckpt.keys()):
        return ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    raise RuntimeError(f"No pude inferir state_dict desde: {ckpt_path}")


class PatchDataset(Dataset):
    def __init__(self, patches_bgr: List[np.ndarray]):
        self.patches = patches_bgr
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        bgr = self.patches[idx]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.tf(rgb)


def get_smart_patch(img_bgr: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    half = size // 2
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = x1 + size, y1 + size

    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - w), max(0, y2 - h)

    crop_x1, crop_y1 = max(0, x1), max(0, y1)
    crop_x2, crop_y2 = min(w, x2), min(h, y2)

    patch = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    if pad_l or pad_t or pad_r or pad_b:
        patch = cv2.copyMakeBorder(
            patch, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REFLECT
        )
    return patch


def load_gt_shidc_points_json(img_stem: str, gt_root: Path, img_path: Optional[Path] = None) -> List[dict]:
    """
    SHIDC: JSON al lado de la imagen, mismo nombre base.
    Ej: p1_0299_6.jpg  <->  p1_0299_6.json

    Retorna lista:
      [{'x': float, 'y': float, 'label_id': int}, ...]
    label_id: 1=pos, 2=neg, 3=TIL (se ignora después si tu GT_MAP no lo mapea)
    """
    json_path = gt_root / f"{img_stem}.json"
    if not json_path.exists() and img_path is not None:
        json_path = img_path.with_suffix(".json")

    if not json_path.exists():
        return []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            pts = data
        elif isinstance(data, dict) and "points" in data and isinstance(data["points"], list):
            pts = data["points"]
        elif isinstance(data, dict):
            pts = None
            for v in data.values():
                if isinstance(v, list):
                    pts = v
                    break
            if pts is None:
                return []
        else:
            return []

        unified = []
        for pt in pts:
            if not isinstance(pt, dict):
                continue
            if "x" not in pt or "y" not in pt:
                continue
            lid = int(pt.get("label_id", 0))
            unified.append({"x": float(pt["x"]), "y": float(pt["y"]), "label_id": lid})

        return unified

    except Exception:
        return []



@dataclass
class ImageArtifacts:
    img_name: str
    masks: np.ndarray          
    props: List             
    cand_xy_maskid: List[Tuple[float, float, int]]
    patches_bgr: List[np.ndarray]
    gt_points: List[dict]


def segment_and_extract(
    img_path: Path,
    gt_root: Path,
    cellpose_model,
    patch_size: int,
    cellpose_eval_params: dict
) -> ImageArtifacts:

    img_rgb = cp_io.imread(str(img_path))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    gt_points = load_gt_shidc_points_json(img_path.stem, gt_root, img_path=img_path)


    masks, _, _ = cellpose_model.eval(img_rgb, **cellpose_eval_params)
    props = regionprops(masks)

    patches = []
    cand = []
    for p in props:
        cy, cx = p.centroid  
        patches.append(get_smart_patch(img_bgr, cx, cy, patch_size))
        cand.append((float(cx), float(cy), int(p.label)))

    return ImageArtifacts(
        img_name=img_path.name,
        masks=masks,
        props=props,
        cand_xy_maskid=cand,
        patches_bgr=patches,
        gt_points=gt_points
    )


def infer_probs_for_patches(
    model: nn.Module,
    patches_bgr: List[np.ndarray],
    device: torch.device,
    batch_size: int,
    num_workers: int = 0
) -> np.ndarray:
    """
    Retorna probs Nx3 (softmax).
    """
    if len(patches_bgr) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    ds = PatchDataset(patches_bgr)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    probs = []
    use_cuda = (device.type == "cuda")
    amp_device = "cuda" if use_cuda else "cpu"

    model.eval()
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=amp_device, enabled=use_cuda):
                logits = model(batch)
                p = F.softmax(logits, dim=1).float().detach().cpu().numpy()
            probs.append(p)

    return np.concatenate(probs, axis=0)


def apply_threshold_rule(probs: np.ndarray, thr: float) -> np.ndarray:
    """
      si p_bg > thr => BG
      else => argmax entre pos/neg (comparando p_pos vs p_neg)
    Retorna pred_idx N (0,1,2)
    """
    if probs.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    p_bg = probs[:, 0]
    p_neg = probs[:, 1]
    p_pos = probs[:, 2]

    pred = np.where(
        p_bg > thr,
        0,
        np.where(p_pos > p_neg, 2, 1)
    ).astype(np.int64)
    return pred


def match_preds_to_gt_inside_mask(
    masks: np.ndarray,
    cand_xy_maskid: List[Tuple[float, float, int]],
    pred_idx: np.ndarray,
    gt_points: List[dict],
    idx_to_class: Dict[int, str],
    gt_map: Dict[int, str],
) -> Tuple[dict, dict]:
    """
    Evalúa por imagen.
    Devuelve:
      - counts dict: TP/FP/FN/Wrong_Class por clase + totales
      - ki67 dict: gt_pos, gt_neg, pred_pos, pred_neg, gt_index, pred_index
    """
    classes = ["ki67_pos", "ki67_neg"]
    counts = {
        "tp_pos": 0, "fp_pos": 0, "fn_pos": 0,
        "tp_neg": 0, "fp_neg": 0, "fn_neg": 0,
        "wrong_class": 0,
        "n_gt_pos": 0, "n_gt_neg": 0,
        "n_pred_pos": 0, "n_pred_neg": 0,
        "n_pred_bg": 0
    }

    # Conteo GT real (por puntos)
    for pt in gt_points:
        cls = gt_map.get(int(pt["label_id"]), "unknown")
        if cls == "ki67_pos":
            counts["n_gt_pos"] += 1
        elif cls == "ki67_neg":
            counts["n_gt_neg"] += 1

    gt_found = [False] * len(gt_points)

    # Procesar predicciones NO-BG
    for i, pidx in enumerate(pred_idx):
        pred_cls = idx_to_class[int(pidx)]
        cx, cy, mask_id = cand_xy_maskid[i]

        if pred_cls == "background":
            counts["n_pred_bg"] += 1
            continue
        if pred_cls == "ki67_pos":
            counts["n_pred_pos"] += 1
        elif pred_cls == "ki67_neg":
            counts["n_pred_neg"] += 1

        # Buscar un GT dentro de la misma máscara
        match_gt_cls = None
        matched = False

        for g_i, pt in enumerate(gt_points):
            if gt_found[g_i]:
                continue
            gx, gy = float(pt["x"]), float(pt["y"])
            iy, ix = int(round(gy)), int(round(gx))
            if 0 <= iy < masks.shape[0] and 0 <= ix < masks.shape[1]:
                if int(masks[iy, ix]) == int(mask_id):
                    matched = True
                    gt_found[g_i] = True
                    match_gt_cls = gt_map.get(int(pt["label_id"]), "unknown")
                    break

        if not matched:
            # FP (predijo pos o neg, pero no corresponde a ningún GT)
            if pred_cls == "ki67_pos":
                counts["fp_pos"] += 1
            elif pred_cls == "ki67_neg":
                counts["fp_neg"] += 1
        else:
            # matched: TP si clase coincide, si no Wrong_Class (y cuenta como error para ambos)
            if pred_cls == match_gt_cls:
                if pred_cls == "ki67_pos":
                    counts["tp_pos"] += 1
                elif pred_cls == "ki67_neg":
                    counts["tp_neg"] += 1
            else:
                counts["wrong_class"] += 1
                # tu interpretación práctica: equivale a un FP para la clase predicha y FN para la GT
                if pred_cls == "ki67_pos":
                    counts["fp_pos"] += 1
                elif pred_cls == "ki67_neg":
                    counts["fp_neg"] += 1

                if match_gt_cls == "ki67_pos":
                    counts["fn_pos"] += 1
                elif match_gt_cls == "ki67_neg":
                    counts["fn_neg"] += 1

    # Los GT no encontrados => FN
    for g_i, found in enumerate(gt_found):
        if not found:
            gt_cls = gt_map.get(int(gt_points[g_i]["label_id"]), "unknown")
            if gt_cls == "ki67_pos":
                counts["fn_pos"] += 1
            elif gt_cls == "ki67_neg":
                counts["fn_neg"] += 1

    # Ki67 index por imagen (solo pos/neg)
    gt_pos = counts["n_gt_pos"]
    gt_neg = counts["n_gt_neg"]
    pred_pos = counts["tp_pos"] + counts["fp_pos"]  # pos predichos (incluye los wrong_class ya sumados como fp)
    pred_neg = counts["tp_neg"] + counts["fp_neg"]

    gt_den = gt_pos + gt_neg
    pred_den = pred_pos + pred_neg

    gt_index = (gt_pos / gt_den) if gt_den > 0 else np.nan
    pred_index = (pred_pos / pred_den) if pred_den > 0 else np.nan

    ki67 = {
        "gt_pos": gt_pos, "gt_neg": gt_neg,
        "pred_pos": pred_pos, "pred_neg": pred_neg,
        "gt_index": gt_index, "pred_index": pred_index
    }
    return counts, ki67


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def corr_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if len(a) < 2:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_img_dir", type=str, required=True)
    ap.add_argument("--test_gt_dir", type=str, required=True)

    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--thresholds", type=str, default="0.50,0.60,0.70,0.75,0.80,0.82,0.84,0.85,0.86,0.88,0.90,0.92,0.95,0.98,0.99")
    ap.add_argument("--patch_size", type=int, default=72)

    ap.add_argument("--cellpose_model_type", type=str, default="cpsam")
    ap.add_argument("--cellpose_diameter", type=float, default=72)
    ap.add_argument("--cellpose_invert", action="store_true")
    ap.add_argument("--cellpose_cellprob_threshold", type=float, default=-1.0)
    ap.add_argument("--cellpose_flow_threshold", type=float, default=0.8)
    ap.add_argument("--cellpose_min_size", type=int, default=25)

    ap.add_argument("--batch_infer", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--seed", type=int, default=42)

    # Models: pasa como JSON dict {arch: ckpt_path}
    # ejemplo:
    # --models_json '{"resnext50":"/.../best_f1posneg.pth","convnext_tiny":"/.../best_f1posneg.pth","efficientnet_v2_s":"/.../best_f1posneg.pth"}'
    ap.add_argument("--models_json", type=str, required=True)

    args = ap.parse_args()
    set_seed(args.seed)

    test_img_dir = Path(args.test_img_dir)
    test_gt_dir = Path(args.test_gt_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip() != ""]
    with open(args.models_json, "r") as f:
        models_map = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_files = sorted(list(test_img_dir.glob("*.png")))
    if not img_files:
        img_files = sorted(list(test_img_dir.glob("*.jpg")))
    if not img_files:
        raise FileNotFoundError(f"No hay imágenes en {test_img_dir}")

    print("Cargando Cellpose...")
    cp = cp_models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=args.cellpose_model_type)
    cp_eval_params = dict(
        diameter=args.cellpose_diameter,
        invert=bool(args.cellpose_invert),
        cellprob_threshold=args.cellpose_cellprob_threshold,
        flow_threshold=args.cellpose_flow_threshold,
        min_size=args.cellpose_min_size
    )

    print(f"Precomputando segmentación+parches: {len(img_files)} imágenes")
    artifacts: List[ImageArtifacts] = []
    for ip in tqdm(img_files, desc="Cellpose+patches"):
        art = segment_and_extract(
            img_path=ip,
            gt_root=test_gt_dir,
            cellpose_model=cp,
            patch_size=args.patch_size,
            cellpose_eval_params=cp_eval_params
        )
        artifacts.append(art)

    summary_rows = []
    ki67_rows = []

    best_rows = []

    for arch, info in models_map.items():
        ckpt_path = Path(info["checkpoint"])
        print("\n" + "=" * 80)
        print(f"Modelo: {arch} | ckpt: {ckpt_path}")
        print("=" * 80)

        model = build_model(arch, num_classes=3, device=device)
        sd = load_checkpoint_state_dict(ckpt_path, device=device)
        model.load_state_dict(sd, strict=True)
        model.eval()

        probs_cache: List[np.ndarray] = []
        for art in tqdm(artifacts, desc=f"Inferencia {arch}"):
            probs = infer_probs_for_patches(
                model=model,
                patches_bgr=art.patches_bgr,
                device=device,
                batch_size=args.batch_infer,
                num_workers=args.num_workers
            )
            probs_cache.append(probs)

        best_thr = None
        best_f1_posneg = -1.0

        for thr in thresholds:
            # aggregate counts over dataset
            agg = {
                "tp_pos": 0, "fp_pos": 0, "fn_pos": 0,
                "tp_neg": 0, "fp_neg": 0, "fn_neg": 0,
                "wrong_class": 0,
                "n_gt_pos": 0, "n_gt_neg": 0,
                "n_pred_pos": 0, "n_pred_neg": 0,
                "n_pred_bg": 0,
            }

            gt_idx_list = []
            pred_idx_list = []

            for art, probs in zip(artifacts, probs_cache):
                pred_idx = apply_threshold_rule(probs, thr)
                counts, ki67 = match_preds_to_gt_inside_mask(
                    masks=art.masks,
                    cand_xy_maskid=art.cand_xy_maskid,
                    pred_idx=pred_idx,
                    gt_points=art.gt_points,
                    idx_to_class=IDX_TO_CLASS_DEFAULT,
                    gt_map=GT_MAP_DEFAULT
                )

                # sum counts
                for k in agg.keys():
                    agg[k] += int(counts.get(k, 0))

                # store ki67 per image (para análisis por thr)
                ki67_rows.append({
                    "arch": arch,
                    "threshold": thr,
                    "image": art.img_name,
                    **ki67
                })

                # for correlation across images
                gt_idx_list.append(ki67["gt_index"])
                pred_idx_list.append(ki67["pred_index"])

            # PRF por clase
            p_pos, r_pos, f1_pos = prf(agg["tp_pos"], agg["fp_pos"], agg["fn_pos"])
            p_neg, r_neg, f1_neg = prf(agg["tp_neg"], agg["fp_neg"], agg["fn_neg"])

            # Macro pos/neg
            f1_posneg = (f1_pos + f1_neg) / 2.0
            prec_posneg = (p_pos + p_neg) / 2.0
            rec_posneg = (r_pos + r_neg) / 2.0

            # Micro pos/neg 
            tp_micro = agg["tp_pos"] + agg["tp_neg"]
            fp_micro = agg["fp_pos"] + agg["fp_neg"]
            fn_micro = agg["fn_pos"] + agg["fn_neg"]
            p_micro, r_micro, f1_micro = prf(tp_micro, fp_micro, fn_micro)

            # Ki67 index metrics (por imagen)
            gt_idx_arr = np.array(gt_idx_list, dtype=np.float64)
            pred_idx_arr = np.array(pred_idx_list, dtype=np.float64)

            valid = np.isfinite(gt_idx_arr) & np.isfinite(pred_idx_arr)
            if valid.sum() > 0:
                mae = float(np.mean(np.abs(gt_idx_arr[valid] - pred_idx_arr[valid])))
                mse = float(np.mean((gt_idx_arr[valid] - pred_idx_arr[valid]) ** 2))
                pear = corr_pearson(gt_idx_arr[valid], pred_idx_arr[valid])
            else:
                mae, mse, pear = float("nan"), float("nan"), float("nan")

            row = {
                "arch": arch,
                "threshold": thr,

                # nucleus-level (pos/neg) - macro 
                "prec_posneg_macro": prec_posneg,
                "rec_posneg_macro": rec_posneg,
                "f1_posneg_macro": f1_posneg,

                # nucleus-level micro (extra)
                "prec_posneg_micro": p_micro,
                "rec_posneg_micro": r_micro,
                "f1_posneg_micro": f1_micro,

                # por clase
                "prec_pos": p_pos, "rec_pos": r_pos, "f1_pos": f1_pos,
                "prec_neg": p_neg, "rec_neg": r_neg, "f1_neg": f1_neg,

                # conteos
                **agg,

                # ki67 index (image-level)
                "ki67_mae": mae,
                "ki67_mse": mse,
                "ki67_pearson": pear,
                "n_images_valid_index": int(valid.sum()),
            }
            summary_rows.append(row)

            if f1_posneg > best_f1_posneg:
                best_f1_posneg = f1_posneg
                best_thr = thr

        best_rows.append({
            "arch": arch,
            "best_threshold_by_f1posneg": best_thr,
            "best_f1posneg_macro": best_f1_posneg
        })


    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary_by_threshold.csv"
    summary_df.to_csv(summary_path, index=False)

    ki67_df = pd.DataFrame(ki67_rows)
    ki67_path = out_dir / "ki67_index_by_image.csv"
    ki67_df.to_csv(ki67_path, index=False)

    best_df = pd.DataFrame(best_rows)
    best_path = out_dir / "best_thresholds.csv"
    best_df.to_csv(best_path, index=False)

    print("\nListo.")
    print(f"Summary: {summary_path}")
    print(f"Ki67 idx: {ki67_path}")
    print(f"Best thr: {best_path}")

    print("\nMejores umbrales por F1(Pos/Neg) (macro):")
    print(best_df.sort_values("best_f1posneg_macro", ascending=False).to_string(index=False))

    print("\nMejor fila (por F1 pos/neg) por modelo:")
    for arch in best_df["arch"].tolist():
        thr = float(best_df[best_df["arch"] == arch]["best_threshold_by_f1posneg"].iloc[0])
        best_row = summary_df[(summary_df["arch"] == arch) & (summary_df["threshold"] == thr)].iloc[0]
        keep = [
            "arch", "threshold",
            "f1_posneg_macro", "prec_posneg_macro", "rec_posneg_macro",
            "ki67_mae", "ki67_pearson",
            "tp_pos", "fp_pos", "fn_pos", "tp_neg", "fp_neg", "fn_neg",
            "wrong_class"
        ]
        print(pd.DataFrame([best_row[keep]]).to_string(index=False))


if __name__ == "__main__":
    main()