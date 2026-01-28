# Biomarker Ki-67 – Pipeline Modular e Interfaz de Inferencia

Este repositorio contiene el código desarrollado para un sistema modular de visión por computador orientado a la detección y cuantificación del biomarcador Ki-67 en imágenes histopatológicas de cáncer de mama.

El proyecto incluye tanto código experimental utilizado durante la etapa de desarrollo como la implementación final de una interfaz gráfica que integra el pipeline de inferencia. El sistema fue concebido como un **prototipo de investigación (MVP)** y **no como una herramienta clínica**.

---

## Contenido del repositorio

En este repositorio se encuentran:

- Scripts de prueba y evaluación para:
  - Cellpose 3
  - Cellpose SAM
- Código de entrenamiento y evaluación de distintos backbones CNN
- Interfaz gráfica final para inferencia y visualización
- Definiciones de entornos Conda utilizados durante el desarrollo

**No se incluyen modelos entrenados ni datasets**, los cuales deben obtenerse por separado desde sus fuentes oficiales.

---

## Entornos de ejecución

Todos los entornos utilizados en el proyecto se encuentran definidos como archivos `.yml` dentro de la carpeta `environments/`.

> Es necesario tener **Miniconda o Anaconda** instalado para ejecutar cualquier parte del proyecto.

Se utilizaron tres entornos principales:

| Uso | Entorno |
|----|--------|
| Pruebas con Cellpose 3 | `cellpose3.yml` |
| Pruebas con Cellpose SAM | `cellpose_sam.yml` |
| Interfaz y pipeline final | `biomarker_mvp.yml` |

Para crear un entorno:
```bash
conda env create -f environments/<nombre_entorno>.yml
```

## Licencias y uso de terceros

Este proyecto integra herramientas y modelos de terceros con fines
**exclusivamente académicos y de investigación**.

Principales componentes utilizados:

- **PyTorch** (BSD-style License)
- **Cellpose** (BSD 3-Clause License, HHMI)
- **ConvNeXt** (MIT License, Meta)
- **ResNeXt** (BSD License, Facebook)
- **EfficientNet-V2** (Apache License 2.0)

Los pesos preentrenados utilizados corresponden a modelos entrenados en ImageNet
y se obtienen directamente desde PyTorch.

Los datasets **SHIDC-B-Ki-67** y **BCData** son públicos y se utilizan respetando
sus términos de uso.  
Este repositorio **no redistribuye imágenes, anotaciones ni pesos de modelos**.

Para mayor detalle, ver el archivo [`LICENCIAS.md`](LICENCIAS.md).


## Pesos del modelo ConvNeXt-Tiny

Los pesos del modelo ConvNeXt-Tiny entrenado para detección de Ki-67
se encuentran disponibles en el siguiente enlace:

👉 [Descargar pesos ConvNeXt-Tiny (Google Drive)](ENLACE_AQUI)

### Detalles del entrenamiento
- Arquitectura: ConvNeXt-Tiny (MIT License)
- Inicialización: pesos preentrenados en ImageNet
- Fine-tuning sobre: BCData (dataset público)
- Uso: académico y de investigación

Estos pesos se proveen únicamente con fines de investigación.
No se redistribuyen imágenes ni anotaciones de los datasets utilizados.
