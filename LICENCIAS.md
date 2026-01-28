# Licencias y uso de software, modelos y datasets de terceros

Este repositorio integra herramientas, arquitecturas y conjuntos de datos de terceros con fines
exclusivamente **académicos y de investigación**.  
El autor **no reclama propiedad** sobre ninguno de los componentes listados a continuación y respeta
las licencias y términos de uso originales.

---

## Framework principal

### PyTorch
- Licencia: BSD-style License
- Copyright © Facebook, Inc., DeepMind Technologies, NYU, NEC Laboratories America,
  Idiap Research Institute y otros contribuidores
- Sitio oficial: https://pytorch.org/

PyTorch se utiliza como framework base para entrenamiento e inferencia.
No se redistribuye código modificado de PyTorch en este repositorio.

---

## Segmentación de núcleos

### Cellpose
- Licencia: BSD 3-Clause License
- Copyright © 2020 Howard Hughes Medical Institute (HHMI)
- Sitio oficial: https://www.cellpose.org/

Cellpose se utiliza como segmentador general de núcleos celulares.
Los pesos del modelo se descargan automáticamente desde las fuentes oficiales
y no se redistribuyen en este repositorio.

---

## Arquitecturas de clasificación

### ConvNeXt
- Licencia: MIT License
- Copyright © Meta Platforms, Inc. and affiliates

ConvNeXt se utiliza como backbone clasificador.
Los pesos preentrenados en ImageNet se obtienen mediante PyTorch.
El modelo entrenado por el autor utiliza únicamente datasets públicos.

---

### ResNeXt
- Licencia: BSD License
- Copyright © 2017 Facebook, Inc.

ResNeXt fue utilizado como arquitectura comparativa durante la etapa experimental.
Se respetan las condiciones de redistribución establecidas en la licencia BSD.

---

### EfficientNet-V2
- Licencia: Apache License 2.0
- Copyright © Google / contribuciones académicas

EfficientNet-V2 fue utilizado exclusivamente con fines comparativos.
Los pesos preentrenados en ImageNet se obtienen mediante PyTorch.

---

## Pesos preentrenados (ImageNet)

Todas las arquitecturas CNN (ConvNeXt, ResNeXt, EfficientNet) fueron inicializadas
con pesos preentrenados en ImageNet, conforme a las licencias de PyTorch y de cada arquitectura.
No se redistribuyen pesos propietarios ni modificados.

---

## Datasets utilizados

### SHIDC-B-Ki-67
- Publicación: Negahbani et al., *Scientific Reports*, 2021
- Uso permitido: **Investigación no comercial y fines educativos**
- Términos de uso oficiales:
  - Uso exclusivo para investigación y educación
  - Sin garantías de exactitud clínica
  - Acceso condicionado a aceptación explícita de los términos
  - Obligación de citar el trabajo original

Referencia:
@article{negahbani2021pathonet,
title={PathoNet introduced as a deep neural network backend for evaluation of Ki-67 and tumor-infiltrating lymphocytes in breast cancer},
author={Negahbani et al.},
journal={Scientific Reports},
year={2021}
}
Este repositorio **no redistribuye imágenes ni anotaciones de SHIDC**.

---

### BCData
- Publicación:
  Huang et al., *MICCAI 2020*
- Uso permitido: Investigación académica
- DOI: https://doi.org/10.1007/978-3-030-59722-1_28

Referencia:
Huang, Z. et al. (2020). BCData: A Large-Scale Dataset and Benchmark for Cell Detection and Counting.
MICCAI 2020, LNCS 12265.
Este repositorio **no redistribuye datos de BCData**.

---

## Uso académico y fair use

El uso de todos los componentes se realiza bajo principios de:
- Investigación académica
- Reproducibilidad científica
- Fair use

Este proyecto **no tiene fines comerciales** ni pretende sustituir
evaluaciones clínicas realizadas por patólogos.

---

## Nota final

Si algún componente requiere atribuciones adicionales o presenta restricciones
no mencionadas explícitamente, se agradece reportarlo mediante un *issue*
para su corrección inmediata.



