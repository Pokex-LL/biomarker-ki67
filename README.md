# Biomarker Ki-67 – Pipeline Modular e Interfaz de Inferencia

Este repositorio contiene el código desarrollado para un sistema modular de visión por computador orientado a la detección y cuantificación del biomarcador Ki-67 en imágenes histopatológicas de cáncer de mama.

El proyecto incluye tanto código experimental utilizado durante la etapa de desarrollo como la implementación final de una interfaz gráfica que integra el pipeline de inferencia. El sistema fue concebido como un **prototipo de investigación (MVP)** y **no como una herramienta clínica**.

---

## Contenido del repositorio

En este repositorio se incluyen:

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

> Es necesario contar con **Miniconda o Anaconda** para ejecutar cualquier componente del proyecto.

Se utilizaron tres entornos principales:

| Uso | Entorno |
|----|--------|
| Pruebas con Cellpose 3 | `cellpose3.yml` |
| Pruebas con Cellpose SAM | `cellpose_sam.yml` |
| Interfaz y pipeline final | `biomarker_mvp.yml` |

Para más información, véase  
[`Environments/Instructions.md`](Environments/Instructions.md)

---

## Ajustes de rutas y ejecución

Los códigos contienen rutas y configuraciones por defecto correspondientes al entorno de desarrollo original.  
Estas deben ser adaptadas según la estructura de directorios y el entorno del usuario.

En algunos directorios se incluyen archivos de texto plano con instrucciones explícitas para la ejecución desde terminal, cuando corresponde.

---

## Licencias y uso de software de terceros

Este proyecto integra herramientas y modelos de terceros con fines **exclusivamente académicos y de investigación**.

Principales componentes utilizados:

- **PyTorch** — BSD-style License  
- **Cellpose** — BSD 3-Clause License (Howard Hughes Medical Institute)  
- **ConvNeXt** — MIT License (Meta Platforms, Inc.)  
- **ResNeXt** — BSD License (Facebook, Inc.)  
- **EfficientNet-V2** — Apache License 2.0  

Los pesos preentrenados utilizados corresponden a modelos entrenados en ImageNet y se obtienen directamente desde PyTorch.

Los datasets **SHIDC-B-Ki-67** y **BCData** son públicos y se utilizan respetando estrictamente sus términos de uso.  
Este repositorio **no redistribuye imágenes, anotaciones ni datasets completos**.

Para mayor detalle, véase el archivo  
[`LICENCIAS.md`](LICENCIAS.md)

---

## Pesos del modelo ConvNeXt-Tiny

Los pesos del modelo ConvNeXt-Tiny entrenado para la detección del biomarcador Ki-67 se encuentran disponibles en el siguiente enlace:

**[Descargar pesos ConvNeXt-Tiny (Google Drive)](https://usmcl-my.sharepoint.com/:f:/g/personal/claudio_zanetta_usm_cl/IgDlkGJdo5d3QI3QftXHUvAuAUbNkXzSstxwzuVLMEDW0Tw?e=xTpnng)**

### Detalles del entrenamiento

- Arquitectura: ConvNeXt-Tiny (MIT License)  
- Inicialización: pesos preentrenados en ImageNet  
- Fine-tuning sobre: BCData (dataset público)  
- Uso: académico y de investigación  

Los pesos compartidos fueron entrenados por el autor de este proyecto y **no redistribuyen imágenes ni anotaciones de los datasets utilizados**.

---

## Nota final

Este repositorio corresponde a un prototipo académico orientado a investigación y validación experimental.  
No ha sido validado para uso clínico y no pretende reemplazar la evaluación realizada por un patólogo.

