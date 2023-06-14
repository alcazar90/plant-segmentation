## Roadmap


1. TODO: escribir metodología de Edward
1. TODO: escribir metodología de SegFormer
1. TODO: escribir metodología de Segment Anything (SAM)
1. Correr SegFormer y ver resultados para subset `cwt` en modo segmentación (normal, normal_cut, noise)
1.  Modificar `PlantDataset` para integrar observaciones (imágenes) de `cwt`, `dead` en un solo dataset y repetir los pasos (1)-(3)
10. Agregar otros modelos para resolver los task de semantic segmentation y instance segmentation (e.g. Mask2Former, SAM)
1. Entrenar enfocandonos en detección y no detección. Usar `cwt` y `dead` para segmentació.
1. Reportar métricas, confusion matrices, IoU, de qué clases necesitamos más datos
1. Método de comparación de los resultados en (2) con metodología Edward
1.  Agregar otros modelos para resolver los task de semantic segmentation y instance segmentation (e.g. Mask2Former, SAM)

## Bitacora

### `single-segmentation.ipynb`

En `single-segmentation.ipynb` se esta corriendo un modelo solo para 
detectar el label `normal`. En la transformación de máscaras binarias,
el label `normal` se convierte en `1` y el resto de labels se convierten
en 0. 

![Segmentation](./assets/single-segmentation-overfitting-a-batch-preds.png)

![Segmentation](./assets/single-segmentation-overfitting-a-batch-truth.png)


## Dependencias

### Segment Anything

- Instalación de [Segment Anything](https://github.com/facebookresearch/segment-anything), que permite utilizar los modelos de segmentación de imágenes entrenados por Meta de manera fácil. Para instalar la instalación se deben seguir los siguientes pasos:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

- Los _checkpoints_ con los modelos se encuentran en el directorio `ckpt`.