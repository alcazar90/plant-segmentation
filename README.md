## Roadmap


1. Correr SegFormer y ver resultados para subset `cwt` en modo segmentación (normal, normal_cut, noise)
2. Reportar métricas, confusion matrices, IoU, de qué clases necesitamos más datos
3. Método de comparación de los resultados en (2) con metodología Edward
4. Modificar `PlantDataset` para integrar observaciones (imágenes) de `cwt`, `dead` en un solo dataset y repetir los pasos (1)-(3)
5. Agregar otros modelos para resolver los task de semantic segmentation y instance segmentation (e.g. Mask2Former, SAM)
