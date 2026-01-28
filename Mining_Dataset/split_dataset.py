import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm 

def split_dataset(source_dir, dest_dir, split_ratio=0.8, seed=42):
    """
    Divide un dataset en Train y Validation manteniendo la estructura de subcarpetas.
    """
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    classes = ["background", "ki67_neg", "ki67_pos"]

    random.seed(seed)

    for class_name in classes:
        current_source = source_path / class_name
        
        if not current_source.exists():
            print(f"Advertencia: No se encontró la carpeta {current_source}")
            continue

        files = [f for f in current_source.iterdir() if f.is_file()]
        random.shuffle(files)
        
        split_point = int(len(files) * split_ratio)
        
        train_files = files[:split_point]
        val_files = files[split_point:]
        
        print(f"Procesando '{class_name}': {len(files)} total -> {len(train_files)} Train, {len(val_files)} Val")

        def copy_files(file_list, dataset_type):
            target_dir = dest_path / dataset_type / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
        
            for file_path in tqdm(file_list, desc=f"Copiando a {dataset_type}/{class_name}", leave=False):
                shutil.copy2(file_path, target_dir / file_path.name)
        copy_files(train_files, "Train")
        copy_files(val_files, "Validation")

    print("\n--- ¡Proceso Completado! ---")
    print(f"Dataset creado en: {dest_path.resolve()}")

if __name__ == "__main__":
    SOURCE = "SHIDC-B-Ki-67/SD_KI67_Patch_SAM"
    DESTINATION = "SHIDC-B-Ki-67/SD_KI67_Patch_SAM_TV"
    

    split_dataset(SOURCE, DESTINATION, split_ratio=0.8)