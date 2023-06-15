## Roadmap



## Bitácora

### `single-segmentation.ipynb`

El archivo `single-segmentation.ipynb` se puede entrenar un modelo sobre el conjunto de datos `cwt`, solo para detectar el label `normal`. En la creación del _target_ a partir de las máscaras, se utiliza `get_binary_mask` para convertir las masks asociadas a label `normal` en `1` y el resto de labels se convierten en 0. 

En la siguiente imagen se puede ver el resultado de la segmentación _overfitteando_ un batch de 4 imágenes con label `normal`:

<center>
<img src="./assets/single-segmentation-overfitting-a-batch-without-training.png" alt="lalal" width="600"/>

<img src="./assets/single-segmentation-overfitting-a-batch-preds.png" alt="lalal" width="600"/>


<img src="./assets/single-segmentation-overfitting-a-batch-truth.png" alt="lalal" width="600"/>
</center>

Importante verificar luego del entrenamiento sobre el conjunto de datos completos, la influencia de elementos como `normal-cut` o `noise`.

Se deben tener la capacidad de computar la métrica de IoU para cada predicción, así luego ocuparlo tanto en el conjunto de validación y pruebas, para clasificar las predicciones dado cierto _threshold_ como correctas e incorrectas. Esto permitirá computar otras métricas como _precision_ y _recall_.

### `playground.ipynb`

En este archivo esta el desarrollo y prueba general de clases y funciones auxiliares, tanto para el conjunto de datos, su entrenamiento y evaluación.


**TODO:**

1. Escribir metodología de Edward
1. Escribir metodología de SegFormer
1.  Modificar `PlantDataset` para integrar observaciones (imágenes) de `cwt`, `dead` en un solo dataset y repetir los pasos (1)-(3)
1. Entrenar enfocandonos en detección y no detección. Usar `cwt` y `dead` para segmentación.
    - Entrenar un modelo para `cwt`
    - Entrenar un modelo para `dead`
1. Reportar métricas, confusion matrices, IoU, de qué clases necesitamos más datos
1. Método de comparación de los resultados en (2) con metodología Edward
1. Agregar otros modelos para resolver los task de semantic segmentation y instance segmentation (e.g. Mask2Former, SAM)

## Dependencias

### Segment Anything

- Instalación de [Segment Anything](https://github.com/facebookresearch/segment-anything), que permite utilizar los modelos de segmentación de imágenes entrenados por Meta de manera fácil. Para instalar la instalación se deben seguir los siguientes pasos:

    ```bash
    pip install git+https://github.com/facebookresearch/segment-anything.git
    ```

- Los _checkpoints_ con los modelos se encuentran en el directorio `ckpt`.