import sys
import os
import json
import cv2
import numpy as np
import warnings
import logging
import contextlib
from pathlib import Path
from tqdm.auto import tqdm
from skimage.measure import regionprops
from cellpose import models, core, io

BASE_DIR = Path("/media/HDD2/SHIDC-B-Ki-67/256x256 cropped images/train256")
OUTPUT_DIR = Path("/media/HDD2/SHIDC-B-Ki-67/SD_KI67_Patch_SAM_2") 

PATCH_SIZE = 19


MODEL_TYPE = 'cpsam'
GPU = True

INFERENCE_CONFIG = {
    'diameter': 15,       
    'channels': [0, 0],    
    'invert': True,          
    'cellprob_threshold': -3.0, 
    'flow_threshold': 0.9,    
    'min_size': 5         
}


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def get_smart_patch(img, cx, cy, size):
    """
    Extrae un parche de (size x size) centrado en (cx, cy).
    Usa padding espejo si se sale de la imagen.
    """
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

def load_gt_points(json_path):
    """Carga puntos del JSON robustamente."""
    if not json_path.exists():
        return []
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        if isinstance(data, list): return data
        if isinstance(data, dict): return list(data.values())[0] if data else []
    except: return []
    return []

def generar_dataset_sd():
    io.logger_setup()
    logging.getLogger("cellpose").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    
    for c in ['background', 'ki67_pos', 'ki67_neg']:
        ensure_dir(OUTPUT_DIR / c)

    print(f"Iniciando Minería de Datos SD")
    print(f"Modelo: {MODEL_TYPE}")
    print(f"Estrategia: UltraSensible (Prob={INFERENCE_CONFIG['cellprob_threshold']}, Flow={INFERENCE_CONFIG['flow_threshold']})")
    print(f"Parche: {PATCH_SIZE}x{PATCH_SIZE}px")
    print(f"Salida: {OUTPUT_DIR}")

    try:
        print("Cargando modelo...")
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            model = models.CellposeModel(gpu=GPU, pretrained_model=MODEL_TYPE)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    img_files = sorted(list(BASE_DIR.glob("*.jpg")))
    print(f"Encontradas {len(img_files)} imágenes.")

    contadores = {'pos': 0, 'neg': 0, 'bg': 0}

    pbar = tqdm(img_files, desc="Generando Parches")
    
    for img_path in pbar:
        base_name = img_path.stem
        
        if (OUTPUT_DIR / 'background' / f"{base_name}_BG0_background.jpg").exists():
            continue

        img_rgb = io.imread(str(img_path))
        if img_rgb is None: continue

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        gt_points = load_gt_points(img_path.with_suffix('.json'))

        try:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                masks, _, _ = model.eval(img_rgb, 
                                         diameter=INFERENCE_CONFIG['diameter'],
                                         channels=INFERENCE_CONFIG['channels'],
                                         invert=INFERENCE_CONFIG['invert'],
                                         cellprob_threshold=INFERENCE_CONFIG['cellprob_threshold'],
                                         flow_threshold=INFERENCE_CONFIG['flow_threshold'],
                                         min_size=INFERENCE_CONFIG['min_size']
                                        ) 
        except Exception as e:
            pbar.write(f"Fallo inferencia en {base_name}: {e}")
            continue

        masks_ocupadas = set()

        for i, pt in enumerate(gt_points):
            lid = pt.get('label_id')
            if lid not in [1, 2]: continue 
            
            cx, cy = pt['x'], pt['y']

            patch = get_smart_patch(img_bgr, cx, cy, PATCH_SIZE)
            
            clase = 'ki67_pos' if lid == 1 else 'ki67_neg'
            fname = f"{base_name}_ID{i}_{clase}.jpg"
            

            cv2.imwrite(str(OUTPUT_DIR / clase / fname), patch)
            
            if lid == 1: contadores['pos'] += 1
            else: contadores['neg'] += 1

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
                
                fname = f"{base_name}_BG{bg_idx}_background.jpg"
                cv2.imwrite(str(OUTPUT_DIR / 'background' / fname), patch)
                
                contadores['bg'] += 1
                bg_idx += 1

        pbar.set_postfix(Pos=contadores['pos'], Bg=contadores['bg'])

    print("\nDataset Generado Exitosamente.")
    print(f"Positivos (Ki67+): {contadores['pos']}")
    print(f"Negativos (Ki67-): {contadores['neg']}")
    print(f"Background (Falsos Pos): {contadores['bg']}")
    print(f"Total Parches: {sum(contadores.values())}")
    print(f"Ubicación: {OUTPUT_DIR}")

if __name__ == "__main__":
    generar_dataset_sd()