# Instrucciones de Entornos de Ejecución

Este repositorio utiliza múltiples entornos de Conda con el fin de aislar dependencias y asegurar la reproducibilidad de los experimentos realizados durante el desarrollo del proyecto.

Los archivos de definición de entornos se encuentran en la carpeta `environments/` y están escritos en formato `.yml`.  
Para utilizar estos entornos es **requisito contar con Miniconda o Anaconda instalado**.

---

## Convención de nombres

Todos los archivos de entorno siguen la convención:
```bash
environment_<NOMBRE_ENTORNO>.yml
```
El nombre real del entorno de Conda que debe activarse corresponde a la parte posterior a `environment_`.

**Ejemplo:**

Archivo:
```bash
environment_biomarker_ki67.yml
```
Entorno a crear / activar:
```bash
biomarker_ki67
```

---

## Entornos disponibles

### 1. `biomarker_mvp`
**Archivo:** `environment_biomarker_mvp.yml`  

Este entorno se utiliza para:
- Ejecutar la **interfaz gráfica final (MVP)**
- Realizar inferencia completa del pipeline
- Clasificación de núcleos Ki-67
- Exportación de máscaras y centroides

Incluye:
- PyTorch
- PySide6
- ConvNeXt-Tiny
- Cellpose-SAM (descarga de pesos automática en la primera ejecución)

---

### 2. `ki67_modern`
**Archivo:** `environment_ki67_modern.yml`  

Este entorno se utiliza para:
- Pruebas experimentales con **Cellpose SAM**
- Evaluación de configuraciones de segmentación
- Generación de parches para entrenamiento

Está separado del entorno de la interfaz para evitar conflictos de dependencias.

---

### 3. `cellpose_v3`
**Archivo:** `environment_cellpose_v3.yml`  

Este entorno se utiliza exclusivamente para:
- Pruebas con **Cellpose v3**
- Comparación de comportamiento frente a Cellpose SAM
- Experimentos de segmentación independientes

---

## Creación de un entorno

Para crear cualquiera de los entornos, ejecutar:

```bash
conda env create -f environments/environment_<NOMBRE>.yml
``
