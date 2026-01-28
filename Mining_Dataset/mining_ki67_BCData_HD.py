import sys
import os
import cv2
import numpy as np
import warnings
import logging
import contextlib
import h5py 
from pathlib import Path
from tqdm.auto import tqdm
from skimage.measure import regionprops

from cellpose import models, core, io


BASE_ROOT = Path("/media/HDD2/BCData/BCData")

OUTPUT_ROOT = Path("/media/HDD2/PatchBCData")

PATCH_SIZE = 50       
DIAMETRO_EVAL = 47    

MODEL_TYPE = 'cpsam'
GPU = True

INFERENCE_CONFIG = {
    'diameter': DIAMETRO_EVAL,
    'channels': [0, 0],       
    'invert': True,           
    'cellprob_threshold': -1.0,
    'flow_threshold': 0.8,     
    'min_size': 20             
}

PHASES = ['train', 'validation']


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def get_smart_patch(img, cx, cy, size):
    """Extrae parche con padding espejo si es necesario."""
    h, w = img.shape[:2]
    half = size // 2
    
    x1, y1 = int(cx - half), int(cy - half)
    x2, y2 = x1 + size, y1 + size

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    crop_x1, crop_y1 = max(0, x1), max(0, y1)
    crop_x2, crop_y2 = min(w, x2), min(h, y2)

    patch = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        patch = cv2.copyMakeBorder(patch, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        
    return patch

def load_gt_bcdata(img_stem, phase_root):
    """
    Carga coordenadas Positivas y Negativas desde los H5 de BCData.
    phase_root: Path base de la fase (ej: .../BCData/annotations/train)
    """
    h5_name = f"{img_stem}.h5"
    
    path_pos = phase_root / "positive" / h5_name
    path_neg = phase_root / "negative" / h5_name
    
    def extract_coords(path):
        if not path.exists(): return []
        try:
            with h5py.File(path, 'r') as f:
                if 'coordinates' in f.keys():
                    return f['coordinates'][:]
                elif 'coordinate' in f.keys():
                    return f['coordinate'][:]
        except: return []
        return []

    pts_pos = extract_coords(path_pos) 
    pts_neg = extract_coords(path_neg) 
    
    return pts_pos, pts_neg



def generar_dataset_bcdata():
    io.logger_setup()
    logging.getLogger("cellpose").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    
    print(f"Iniciando Generación de Parches BCData (50x50)")
    print(f"Config: {INFERENCE_CONFIG}")
    print(f"Salida: {OUTPUT_ROOT}")

    try:
        print("Cargando modelo...")
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            model = models.CellposeModel(gpu=GPU, pretrained_model=MODEL_TYPE)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    for phase in PHASES:
        print(f"\nExample Processing Phase: {phase.upper()}...")
        
        img_dir = BASE_ROOT / "images" / phase
        annot_dir = BASE_ROOT / "annotations" / phase
        
        out_phase_dir = OUTPUT_ROOT / phase.capitalize() 
        
        for cat in ['pos', 'neg', 'BG']:
            ensure_dir(out_phase_dir / cat)

        img_files = sorted(list(img_dir.glob("*.png")))
        if not img_files: img_files = sorted(list(img_dir.glob("*.jpg")))
        
        if not img_files:
            print(f"No hay imágenes en {img_dir}. Saltando fase.")
            continue

        contadores = {'pos': 0, 'neg': 0, 'BG': 0}
        
        pbar = tqdm(img_files, desc=f"Procesando {phase}")
        for img_path in pbar:
            base_name = img_path.stem
            
            img_rgb = io.imread(str(img_path))
            if img_rgb is None: continue
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            pts_pos, pts_neg = load_gt_bcdata(base_name, annot_dir)

            try:
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    masks, _, _ = model.eval(img_rgb, **INFERENCE_CONFIG)
            except: continue
            
            masks_ocupadas = set()

            if len(pts_pos) > 0:
                for i, pt in enumerate(pts_pos):
                    cx, cy = pt[0], pt[1]
                    
                    patch = get_smart_patch(img_bgr, cx, cy, PATCH_SIZE)
                    fname = f"{base_name}_ID{i}_pos.jpg"
                    cv2.imwrite(str(out_phase_dir / 'pos' / fname), patch)
                    contadores['pos'] += 1

                    try:
                        if 0 <= int(cy) < masks.shape[0] and 0 <= int(cx) < masks.shape[1]:
                            val = masks[int(cy), int(cx)]
                            if val > 0: masks_ocupadas.add(val)
                    except: pass

            if len(pts_neg) > 0:
                for i, pt in enumerate(pts_neg):
                    cx, cy = pt[0], pt[1]
                    
                    patch = get_smart_patch(img_bgr, cx, cy, PATCH_SIZE)
                    fname = f"{base_name}_ID{i}_neg.jpg"
                    cv2.imwrite(str(out_phase_dir / 'neg' / fname), patch)
                    contadores['neg'] += 1
                    
                    try:
                        if 0 <= int(cy) < masks.shape[0] and 0 <= int(cx) < masks.shape[1]:
                            val = masks[int(cy), int(cx)]
                            if val > 0: masks_ocupadas.add(val)
                    except: pass

            props = regionprops(masks)
            bg_idx = 0
            
            for p in props:
                if p.label not in masks_ocupadas:
                    y_cent, x_cent = p.centroid
                    
                    patch = get_smart_patch(img_bgr, x_cent, y_cent, PATCH_SIZE)
                    fname = f"{base_name}_BG{bg_idx}_BG.jpg"
                    cv2.imwrite(str(out_phase_dir / 'BG' / fname), patch)
                    
                    contadores['BG'] += 1
                    bg_idx += 1
            
            pbar.set_postfix(pos=contadores['pos'], neg=contadores['neg'], bg=contadores['BG'])

        print(f"Fase {phase.upper()} terminada.")
        print(f"Pos: {contadores['pos']} | Neg: {contadores['neg']} | BG: {contadores['BG']}")

    print("\nProceso completo. Dataset guardado en:", OUTPUT_ROOT)

if __name__ == "__main__":
    generar_dataset_bcdata()