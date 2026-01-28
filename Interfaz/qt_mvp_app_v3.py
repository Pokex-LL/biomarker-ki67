import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import models, transforms

from skimage.measure import regionprops

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget,
    QComboBox, QLineEdit, QMessageBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem
)

from cellpose import models as cp_models


@dataclass
class Config:
    patch_size: int = 50
    cell_diameter: int = 47
    thr_bg: float = 0.98
    batch_size: int = 256


@dataclass
class Result:
    masks: np.ndarray               
    cents: List[Tuple[float, float]]  
    labels: np.ndarray               
    n_cells: int
    n_pos: int
    pct: float


def bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def rgba_to_qpixmap(img_rgba: np.ndarray) -> QPixmap:
    h, w, ch = img_rgba.shape
    assert ch == 4
    bytes_per_line = ch * w
    qimg = QImage(img_rgba.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())



def load_checkpoint_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]

    if isinstance(ckpt, dict) and any(
        isinstance(k, str) and (k.startswith("features.") or k.startswith("classifier.") or k.startswith("fc."))
        for k in ckpt.keys()
    ):
        return ckpt

    raise RuntimeError(f"No pude inferir state_dict desde: {ckpt_path}")


def strip_module_prefix(sd: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix):]: v for k, v in sd.items()}


def build_convnext_tiny_3c(device: torch.device) -> nn.Module:
    m = models.convnext_tiny(weights=None)
    in_features = m.classifier[2].in_features
    m.classifier[2] = nn.Linear(in_features, 3) 
    return m.to(device)


VAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, patches_bgr: List[np.ndarray]):
        self.patches = patches_bgr

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        bgr = self.patches[idx]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return VAL_TF(rgb)


@torch.inference_mode()
def predict_probs_3c(model: nn.Module, patches_bgr: List[np.ndarray], device: torch.device, batch_size: int) -> np.ndarray:
    if len(patches_bgr) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    ds = PatchDataset(patches_bgr)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []

    use_amp = (device.type == "cuda")
    for xb in dl:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(xb)
            pb = torch.softmax(logits, dim=1).detach().float().cpu().numpy()
        probs.append(pb)

    return np.concatenate(probs, axis=0)


def apply_bg_threshold_rule(probs_3c: np.ndarray, thr_bg: float) -> np.ndarray:
    """
    Output labels:
      -1 -> BG (descartado por alta prob. de BG)
       0 -> neg
       1 -> pos
    Regla:
      si P(BG) >= thr_bg => -1
      si P(BG) <  thr_bg => argmax entre (neg,pos)
    """
    if probs_3c.ndim != 2 or probs_3c.shape[1] != 3:
        raise ValueError("probs_3c debe ser Nx3")

    p_bg = probs_3c[:, 0]
    labels = np.full((len(probs_3c),), -1, dtype=np.int32)

    is_cell = p_bg < thr_bg
    if np.any(is_cell):
        sub = probs_3c[is_cell, 1:3]  
        cls = np.argmax(sub, axis=1)  
        labels[is_cell] = cls

    return labels

def instance_masks_and_centroids(masks: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    props = regionprops(masks)
    cents: List[Tuple[float, float]] = []
    for p in props:
        cy, cx = p.centroid 
        cents.append((float(cx), float(cy)))
    return masks, cents


def get_smart_patch_bgr(img_bgr: np.ndarray, cx: float, cy: float, patch_size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    half = patch_size // 2

    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    if pad_l or pad_t or pad_r or pad_b:
        img_pad = cv2.copyMakeBorder(img_bgr, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REFLECT_101)
        x1 += pad_l; x2 += pad_l
        y1 += pad_t; y2 += pad_t
        patch = img_pad[y1:y2, x1:x2]
    else:
        patch = img_bgr[y1:y2, x1:x2]

    return patch


def mask_overlay_rgba(masks: np.ndarray, alpha: int = 80) -> np.ndarray:
    h, w = masks.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    on = masks > 0
    rgba[on, 1] = 255
    rgba[on, 2] = 255
    rgba[on, 3] = alpha
    return rgba

class ImageCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.base_item: Optional[QGraphicsPixmapItem] = None
        self.mask_item: Optional[QGraphicsPixmapItem] = None
        self.point_items: List[QGraphicsEllipseItem] = []

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def clear_points(self):
        for it in self.point_items:
            self.scene().removeItem(it)
        self.point_items = []

    def set_base_image(self, pix: QPixmap):
        self.scene().clear()
        self.base_item = self.scene().addPixmap(pix)
        self.base_item.setZValue(0)
        self.mask_item = None
        self.point_items = []
        self.setSceneRect(QRectF(pix.rect()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def set_mask_overlay(self, pix: QPixmap):
        if self.mask_item is not None:
            self.scene().removeItem(self.mask_item)
        self.mask_item = self.scene().addPixmap(pix)
        self.mask_item.setZValue(1)

    def add_point(self, x: float, y: float, kind: str):
        r = 3.5
        item = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        if kind == "pos":
            item.setBrush(QColor(255, 0, 0, 220))
        else:
            item.setBrush(QColor(0, 255, 0, 220))
        item.setPen(QColor(0, 0, 0, 120))
        item.setZValue(2)
        self.scene().addItem(item)
        self.point_items.append(item)

    def wheelEvent(self, event):
        zoom_in = 1.25
        zoom_out = 1 / zoom_in
        if event.angleDelta().y() > 0:
            self.scale(zoom_in, zoom_in)
        else:
            self.scale(zoom_out, zoom_out)



PATIENT_RE = re.compile(r"^([a-z0-9]+)_\d+_.*\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Biomarker MVP")
        self.cfg = Config()
        self.results_by_path: Dict[Path, Result] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.cpsam: Optional[cp_models.CellposeModel] = None

        self.images_root: Optional[Path] = None
        self.by_patient: Dict[str, List[Path]] = {}

        # ---- UI layout first ----
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.left = QVBoxLayout()
        self.right = QVBoxLayout()
        layout.addLayout(self.left, 0)
        layout.addLayout(self.right, 1)

        # ---- Right-side toggles ----
        self.cb_show_mask = QCheckBox("Ver máscara")
        self.cb_show_points = QCheckBox("Ver positivos")
        self.cb_show_points_neg = QCheckBox("Ver negativos")
        self.cb_show_mask.setChecked(True)
        self.cb_show_points.setChecked(True)
        self.cb_show_points_neg.setChecked(True)
        self.cb_show_mask.stateChanged.connect(lambda _: self._refresh_overlays())
        self.cb_show_points.stateChanged.connect(lambda _: self._refresh_overlays())
        self.cb_show_points_neg.stateChanged.connect(lambda _: self._refresh_overlays())
        self.cb_show_mask.setEnabled(False)
        self.cb_show_points.setEnabled(False)
        self.cb_show_points_neg.setEnabled(False)

        self.right.addWidget(self.cb_show_mask)
        self.right.addWidget(self.cb_show_points)
        self.right.addWidget(self.cb_show_points_neg)

        # ---- Controls: folder ----
        self.ed_root = QLineEdit()
        self.ed_root.setPlaceholderText("Carpeta de imágenes (Pacientex_y_*.png)")
        btn_pick = QPushButton("Elegir carpeta…")
        btn_pick.clicked.connect(self.pick_folder)
        row0 = QHBoxLayout()
        row0.addWidget(self.ed_root)
        row0.addWidget(btn_pick)
        self.left.addLayout(row0)

        # ---- Patient + images ----
        self.cmb_patient = QComboBox()
        self.cmb_patient.currentIndexChanged.connect(self.on_patient_changed)
        self.left.addWidget(QLabel("Paciente:"))
        self.left.addWidget(self.cmb_patient)

        self.lst_images = QListWidget()
        self.lst_images.currentRowChanged.connect(self.on_image_changed)
        self.left.addWidget(QLabel("Imágenes:"))
        self.left.addWidget(self.lst_images, 1)

        # ---- Model path ----
        self.ed_pth = QLineEdit()
        self.ed_pth.setPlaceholderText("Ruta .pth (ConvNeXt)")
        btn_pth = QPushButton("Elegir .pth…")
        btn_pth.clicked.connect(self.pick_pth)
        row1 = QHBoxLayout()
        row1.addWidget(self.ed_pth)
        row1.addWidget(btn_pth)
        self.left.addLayout(row1)

        # ---- Params ----
        self.sp_patch = QSpinBox()
        self.sp_patch.setRange(16, 256)
        self.sp_patch.setValue(self.cfg.patch_size)
        self.sp_patch.valueChanged.connect(lambda v: setattr(self.cfg, "patch_size", int(v)))

        self.sp_diam = QSpinBox()
        self.sp_diam.setRange(5, 200)
        self.sp_diam.setValue(self.cfg.cell_diameter)
        self.sp_diam.valueChanged.connect(lambda v: setattr(self.cfg, "cell_diameter", int(v)))

        self.sp_thr = QDoubleSpinBox()
        self.sp_thr.setRange(0.0, 1.0)
        self.sp_thr.setSingleStep(0.01)
        self.sp_thr.setValue(self.cfg.thr_bg)
        self.sp_thr.valueChanged.connect(lambda v: setattr(self.cfg, "thr_bg", float(v)))

        self.left.addWidget(QLabel("Tamaño Parche"))
        self.left.addWidget(self.sp_patch)
        self.left.addWidget(QLabel("Diametro Célula"))
        self.left.addWidget(self.sp_diam)
        self.left.addWidget(QLabel("Umbral Fondo"))
        self.left.addWidget(self.sp_thr)

        # ---- Buttons ----
        self.btn_run = QPushButton("Segmentar y Clasificar (imagen)")
        self.btn_run.clicked.connect(self.run_pipeline_on_current)

        self.btn_run_all = QPushButton("Segmentar y Clasificar todo el paciente")
        self.btn_run_all.clicked.connect(self.run_pipeline_on_patient)

        self.btn_export = QPushButton("Exportar resultados del paciente")
        self.btn_export.clicked.connect(self.export_patient)

        self.left.addWidget(self.btn_run)
        self.left.addWidget(self.btn_run_all)
        self.left.addWidget(self.btn_export)

        # ---- Status ----
        self.lbl_status = QLabel(f"Device: {self.device.type.upper()} | torch.cuda.is_available={torch.cuda.is_available()}")
        self.left.addWidget(self.lbl_status)

        self.lbl_metrics = QLabel("n_cells: - | n_pos: - | %Ki67: -")
        self.left.addWidget(self.lbl_metrics)

        # ---- Canvas ----
        self.canvas = ImageCanvas()
        self.right.addWidget(self.canvas, 1)

    # ---------- UI actions ----------
    def pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if not folder:
            return
        self.images_root = Path(folder)
        self.ed_root.setText(str(self.images_root))
        self._index_patients()

    def pick_pth(self):
        f, _ = QFileDialog.getOpenFileName(self, "Seleccionar .pth", filter="PyTorch (*.pth *.pt)")
        if not f:
            return
        self.ed_pth.setText(f)
        self.model = None  # fuerza recarga si cambias .pth

    def _index_patients(self):
        assert self.images_root is not None
        self.by_patient = {}
        for p in sorted(self.images_root.glob("*")):
            if not p.is_file():
                continue
            m = PATIENT_RE.match(p.name)
            if not m:
                continue
            pid = m.group(1)
            self.by_patient.setdefault(pid, []).append(p)

        self.cmb_patient.blockSignals(True)
        self.cmb_patient.clear()
        for pid in sorted(self.by_patient.keys()):
            self.cmb_patient.addItem(pid)
        self.cmb_patient.blockSignals(False)

        self.lst_images.clear()
        self.canvas.scene().clear()
        self.lbl_metrics.setText("n_cells: - | n_pos: - | %Ki67: -")
        self.cb_show_mask.setEnabled(False)
        self.cb_show_points.setEnabled(False)
        self.cb_show_points_neg.setEnabled(False)

        if self.cmb_patient.count() > 0:
            self.cmb_patient.setCurrentIndex(0)
            self.on_patient_changed(0)
        else:
            QMessageBox.warning(self, "Sin pacientes", "No encontré archivos tipo Pacientex_y_*.png en esa carpeta.")

    def _current_image_path(self) -> Optional[Path]:
        pid = self.cmb_patient.currentText().strip()
        row = self.lst_images.currentRow()
        paths = self.by_patient.get(pid, [])
        if row < 0 or row >= len(paths):
            return None
        return paths[row]

    def on_patient_changed(self, idx: int):
        self.results_by_path.clear()   # borra todas las máscaras y coord en memoria, para evitar que se sature.

        self.canvas.clear_points()
        if self.canvas.mask_item is not None:
            self.canvas.scene().removeItem(self.canvas.mask_item)
            self.canvas.mask_item = None

        self.lbl_metrics.setText("n_cells: - | n_pos: - | %Ki67: -")
        self.cb_show_mask.setEnabled(False)
        self.cb_show_points.setEnabled(False)

        pid = self.cmb_patient.currentText().strip()
        self.lst_images.clear()
        for p in self.by_patient.get(pid, []):
            self.lst_images.addItem(p.name)
        if self.lst_images.count() > 0:
            self.lst_images.setCurrentRow(0)

    def on_image_changed(self, row: int):
        p = self._current_image_path()
        if p is None:
            return

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Error", f"No pude leer: {p}")
            return

        self.canvas.set_base_image(bgr_to_qpixmap(img))

        has_res = (p in self.results_by_path)
        self.cb_show_mask.setEnabled(has_res)
        self.cb_show_points.setEnabled(has_res)
        self.cb_show_points_neg.setEnabled(has_res)
        if not has_res:
            self.lbl_metrics.setText("n_cells: - | n_pos: - | %Ki67: -")

        self._refresh_overlays()

    def _refresh_overlays(self):
        p = self._current_image_path()
        if p is None:
            return

        self.canvas.clear_points()
        if self.canvas.mask_item is not None:
            self.canvas.scene().removeItem(self.canvas.mask_item)
            self.canvas.mask_item = None

        if p not in self.results_by_path:
            return

        res = self.results_by_path[p]

        if self.cb_show_mask.isChecked():
            overlay = mask_overlay_rgba(res.masks, alpha=70)
            self.canvas.set_mask_overlay(rgba_to_qpixmap(overlay))

        if self.cb_show_points.isChecked():
            for (cx, cy), lab in zip(res.cents, res.labels):
                if lab == 1:
                    self.canvas.add_point(cx, cy, "pos")

        if self.cb_show_points_neg.isChecked():
            for (cx, cy), lab in zip(res.cents, res.labels):
                if lab == 0:
                    self.canvas.add_point(cx, cy, "neg")

        self.lbl_metrics.setText(f"n_cells: {res.n_cells} | n_pos: {res.n_pos} | %Ki67: {res.pct:.2f}")

    # ---------- Pipeline ----------
    def _ensure_models_loaded(self):
        pth = self.ed_pth.text().strip()
        if not pth:
            raise RuntimeError("Selecciona el .pth del modelo (convnext).")

        if self.model is None:
            self.lbl_status.setText(f"Device: {self.device.type.upper()} | cargando modelo...")
            self.model = build_convnext_tiny_3c(self.device)
            sd = load_checkpoint_state_dict(Path(pth))
            sd = strip_module_prefix(sd)
            self.model.load_state_dict(sd, strict=True)
            self.model.eval()

        if self.cpsam is None:
            self.lbl_status.setText(f"Device: {self.device.type.upper()} | cargando cpsam...")
            self.cpsam = cp_models.CellposeModel(
                gpu=torch.cuda.is_available(),
                pretrained_model="cpsam",
            )

        self.lbl_status.setText(f"Device: {self.device.type.upper()} | listo")

    def run_pipeline_on_current(self):
        try:
            self._ensure_models_loaded()

            img_path = self._current_image_path()
            if img_path is None:
                raise RuntimeError("Selecciona una imagen.")

            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"No pude leer: {img_path}")

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            masks, *_ = self.cpsam.eval(img_rgb, diameter=self.cfg.cell_diameter)
            masks, cents = instance_masks_and_centroids(masks)

            patches = [get_smart_patch_bgr(img_bgr, cx, cy, self.cfg.patch_size) for (cx, cy) in cents]
            probs_3c = predict_probs_3c(self.model, patches, self.device, self.cfg.batch_size)
            labels = apply_bg_threshold_rule(probs_3c, thr_bg=self.cfg.thr_bg)

            is_cell = labels != -1
            n_cells = int(np.sum(is_cell))
            n_pos = int(np.sum(labels == 1))
            pct = 100.0 * n_pos / max(n_cells, 1)

            self.results_by_path[img_path] = Result(
                masks=masks,
                cents=cents,
                labels=labels,
                n_cells=n_cells,
                n_pos=n_pos,
                pct=pct,
            )

            self.canvas.set_base_image(bgr_to_qpixmap(img_bgr))
            self.cb_show_mask.setEnabled(True)
            self.cb_show_points.setEnabled(True)
            self.cb_show_points_neg.setEnabled(True)
            self._refresh_overlays()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_pipeline_on_patient(self):
        try:
            self._ensure_models_loaded()
            pid = self.cmb_patient.currentText().strip()
            paths = self.by_patient.get(pid, [])
            if not paths:
                raise RuntimeError("No hay imágenes para este paciente.")

            self.btn_run_all.setEnabled(False)
            self.lbl_status.setText(f"Procesando paciente {pid}...")

            for i, p in enumerate(paths, start=1):
                if p in self.results_by_path:
                    continue

                img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                masks, *_ = self.cpsam.eval(img_rgb, diameter=self.cfg.cell_diameter)
                masks, cents = instance_masks_and_centroids(masks)

                patches = [get_smart_patch_bgr(img_bgr, cx, cy, self.cfg.patch_size) for (cx, cy) in cents]
                probs_3c = predict_probs_3c(self.model, patches, self.device, self.cfg.batch_size)
                labels = apply_bg_threshold_rule(probs_3c, thr_bg=self.cfg.thr_bg)

                is_cell = labels != -1
                n_cells = int(np.sum(is_cell))
                n_pos = int(np.sum(labels == 1))
                pct = 100.0 * n_pos / max(n_cells, 1)

                self.results_by_path[p] = Result(masks=masks, cents=cents, labels=labels,
                                                 n_cells=n_cells, n_pos=n_pos, pct=pct)

                self.lbl_status.setText(f"Procesando {pid}: {i}/{len(paths)}")
                QApplication.processEvents()

            pcur = self._current_image_path()
            if pcur is not None:
                has_res = (pcur in self.results_by_path)
                self.cb_show_mask.setEnabled(has_res)
                self.cb_show_points.setEnabled(has_res)
                self.cb_show_points_neg.setEnabled(has_res)
                self._refresh_overlays()

            self.lbl_status.setText(f"Listo: {pid} procesado.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.btn_run_all.setEnabled(True)

    def export_patient(self):
        try:
            pid = self.cmb_patient.currentText().strip()
            paths = self.by_patient.get(pid, [])
            if not paths:
                raise RuntimeError("No hay imágenes para este paciente.")

            out_dir = QFileDialog.getExistingDirectory(self, "Selecciona carpeta de export")
            if not out_dir:
                return
            out_dir = Path(out_dir) / pid
            out_dir.mkdir(parents=True, exist_ok=True)

            csv_path = out_dir / f"{pid}_coords.csv"
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("patient,image,cx,cy,label\n")
                for p in paths:
                    if p not in self.results_by_path:
                        continue
                    res = self.results_by_path[p]
                    for (cx, cy), lab in zip(res.cents, res.labels):
                        if lab == -1:
                            continue
                        label_str = "pos" if lab == 1 else "neg"
                        f.write(f"{pid},{p.name},{cx:.2f},{cy:.2f},{label_str}\n")

            for p in paths:
                if p not in self.results_by_path:
                    continue
                res = self.results_by_path[p]

                img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue

                stem = p.stem

                mask_png = out_dir / f"{stem}_mask_bin.png"
                bin_mask = (res.masks > 0).astype(np.uint8) * 255
                cv2.imwrite(str(mask_png), bin_mask)

                overlay_bgr = img_bgr.copy()
                on = res.masks > 0
                ov = overlay_bgr.astype(np.float32)
                ov[on] = 0.7 * ov[on] + 0.3 * np.array([0, 0, 255], dtype=np.float32)
                overlay_bgr = np.clip(ov, 0, 255).astype(np.uint8)

                for (cx, cy), lab in zip(res.cents, res.labels):
                    if lab == -1:
                        continue
                    color = (0, 255, 0) if lab == 1 else (0, 255, 255)
                    cv2.circle(overlay_bgr, (int(round(cx)), int(round(cy))), 3, color, -1, lineType=cv2.LINE_AA)

                overlay_path = out_dir / f"{stem}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay_bgr)

            QMessageBox.information(self, "Export listo", f"Exportado en:\n{out_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 850)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
