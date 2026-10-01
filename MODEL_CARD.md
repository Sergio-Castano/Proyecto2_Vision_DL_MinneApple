# Model Card — MinneApple

**Versión:** 1.0 · **Fecha:** 27 de septiembre de 2026 (cifras completas; revisión final y congelación el 1 de octubre)
**Institución:** Universidad Autónoma de Occidente — Maestría en Inteligencia Artificial y Ciencia de Datos
**Repositorio:** sin remoto todavía (trabajo local; ver README)

> **Estado de este documento:** cifras completas. El test oficial se evaluó **una sola vez** (27-sep-2026) con el modelo final
> elegido antes en validación. Cada cifra sale de `reports/` (evaluación en test, bootstrap de la ablación, cuantización y tiempos).

## 1. Descripción del problema

Los agricultores necesitan estimar el rendimiento de un huerto de manzanas antes de la cosecha, para planificar la logística de
recolección, almacenamiento y transporte. Hacerlo a ojo es impreciso y genera desperdicio o falta de capacidad de transporte.

MinneApple recibe una foto de un árbol de manzanas y:
1. **Detecta** cada manzana visible con una caja (cuenta el número de frutos).
2. **Segmenta** cada manzana, píxel a píxel, y la **aísla** del fondo (hojas, ramas, cielo).
3. Mide su **diámetro** en píxeles (y en milímetros/peso si el usuario aporta una escala) y muestra un **color descriptivo**
   (rojo o verde/amarillo, calculado sobre la imagen, **no es una clasificación de madurez**: el dataset no trae esa etiqueta).
4. **Explica** sus detecciones con Grad-CAM y compara un bloque de self-attention (C2PSA) con y sin él.

**Uso previsto.** Herramienta académica para estudiar detección, segmentación de instancias y mecanismos de atención sobre imágenes
de huerto, y para dar una **estimación aproximada** de conteo y tamaño de fruta visible en una foto.

**Uso no previsto.** Estimar el rendimiento total de un huerto a partir de una sola foto (solo cuenta lo visible en el encuadre,
no lo oculto por hojas ni las hileras del fondo), decisiones comerciales o logísticas sin verificación humana, o cualquier uso
fuera de manzanos (el modelo se entrenó con un solo huerto y una sola especie).

## 2. Dataset

**MinneApple** (Häni, Roy, Isler, 2019). Data Repository for the University of Minnesota (DRUM), DOI `10.13020/8ecp-3r13`
(handle `11299/206575`). Artículo: *MinneApple: A Benchmark Dataset for Apple Detection and Segmentation*, IEEE RA-L, 2020,
DOI `10.1109/LRA.2020.2965061`.

**Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 US (CC BY-NC-SA 3.0 US).** Uso no comercial, con cita y
obras derivadas bajo la misma licencia. Ver detalle y comandos de descarga en el [README](README.md#datos-minneapple).

| Componente | Volumen |
|---|---|
| Imágenes de entrenamiento con máscaras de instancia | 670 (10 videos), 1280×720 px |
| Imágenes de test oficial con etiquetas (publicadas en 2022) | 331 (7 videos/hileras, de otro año), 1280×720 px |

**Procedencia.** University of Minnesota Horticultural Research Center, video de celular (Samsung Galaxy S4), 2015-2016, cara
soleada y sombreada de las hileras.

**Partición.** Train (670 imágenes) se divide en train/val **por video completo** (2 videos para val, 93 imágenes, 13,9 %): un
mismo video tiene fotogramas casi idénticos, así que partir por imagen habría dejado fotogramas casi iguales en las dos
particiones. Se verificó por código que ningún video cae en dos particiones. El **test oficial** (331 imágenes)
es de otras hileras y otro año: se usa una sola vez, al final.

**Anotación.** Máscaras de instancia hechas con VGG Image Annotator por los autores del dataset; **cubren solo las manzanas del
árbol en primer plano**: las manzanas caídas al suelo y las de árboles del fondo no están anotadas (verificado por inspección
visual, 26-sep-2026). Esto pone un techo a las métricas: parte de lo que el modelo detecta y no cuenta como acierto puede ser
una manzana real sin etiqueta.

**Tamaño de los objetos.** Entre el 75 % y el 89 % de las manzanas mide menos de 32×32 px en la imagen original (lado mediano
30-35 px): son objetos pequeños, y es la principal dificultad del dataset.

**Color.** No hay etiquetas de madurez ni de manzanas podridas. Con un descriptor propio (fracción de píxeles rojos de la
máscara, sin validar contra ninguna etiqueta), el 56,6 % de las manzanas de entrenamiento y el 35,5 % de test son
«verde/amarillo»: la mezcla de color cambia entre train y test.

## 3. Arquitecturas

| Tarea | Modelo | Notas |
|---|---|---|
| Detección y segmentación de instancias | **YOLO11s-seg**, con self-attention **C2PSA** en la capa 10 del backbone, a **1280 px** | Fine-tuning desde pesos COCO; una sola clase (`apple`); elegido sobre nano (+0,020 de mAP de caja en validación, confirmado con 2 semillas) |
| Self-attention | C2PSA (bloque de atención posicional tipo transformer, incluido en YOLO11) | Ablación: variante idéntica con la capa 10 reemplazada por `nn.Identity` |
| Interpretabilidad | Grad-CAM implementado a mano, sobre la capa 16 (paso 8) | Capa elegida por el mayor cociente de enfoque sobre las manzanas, de entre 3 capas candidatas |
| Optimización | Cuantización INT8 estática (ONNX Runtime, QDQ por canal, cabeza en FP32) | −62 % de tamaño y 1,2× más rápido en CPU; aceptada en validación, **supera la tolerancia de calidad en test**; la app usa el `.pt` (necesita gradientes para Grad-CAM, que el ONNX no permite) |

**Modelo final: `small` a 1280 px** (checkpoint `colab_1280_small_s0`, entrenado en Google Colab con una GPU T4 por falta de VRAM en la GPU
local de 4 GB a esa resolución; verificado en la GPU local: 332 ms de mediana por foto, dentro del presupuesto de la demo). Se prefirió la
semilla 0 sobre la semilla 1 (que dio un resultado incluso mejor) para no elegir el modelo final mirando cuál semilla ganó por azar.

## 4. Métricas de desempeño

Test (331 imágenes), evaluado una sola vez el 27-sep-2026. Modelo final: `colab_1280_small_s0` (YOLO11s-seg + C2PSA, 1280 px).

| Métrica | Valor (test) |
|---|---|
| Detección: mAP@0.5:0.95 (caja) | **0,480** |
| Detección: mAP@0.5 (caja) | 0,819 |
| Segmentación: mAP@0.5:0.95 (máscara) | **0,348** |
| Segmentación: Dice de la unión | 0,782 (IoU 0,648) |
| Error de conteo por imagen (MAE) | 5,4 manzanas (de ~37 en promedio; sesgo −2,8, subcuenta ligera) |
| Self-attention: diferencia con/sin C2PSA (Dice, bootstrap pareado) | **+0,017, confirmada** (IC95: 0,014 a 0,019) |
| Self-attention: diferencia con/sin C2PSA (error de conteo, bootstrap pareado) | **1,9 manzanas menos de error, confirmada** (IC95: −2,29 a −1,53) |
| Self-attention: diferencia con/sin C2PSA (mAP de caja, media de 3 semillas) | +0,006 (0,448 frente a 0,442; no se hizo bootstrap sobre el mAP) |
| Optimización: tamaño (ONNX FP32 → INT8) | 39,16 MB → 14,93 MB (**−61,9 %**) |
| Optimización: impacto en mAP de caja / máscara (INT8 − FP32) | Validación: −0,017 / −0,011 (cumple la tolerancia de −0,02). **Test: −0,044 / −0,025 (no cumple)**; en test predice +9,5 manzanas de más por foto (IC95 8,5 a 10,5) |
| Tiempos: red sola, mediana (CPU i5-8300H / GPU GTX 1050) | 846 ms / 120 ms (**GPU 7,0× más rápida**); ONNX en CPU: FP32 813 ms, INT8 659 ms |
| Tiempos: la app con una foto (CPU / GPU) | 990 ms / 323 ms; con Grad-CAM 1.530 ms / 923 ms (`reports/benchmark.csv`) |

**Umbrales de operación** (elegidos en validación, sin mirar el test): confianza 0,35, NMS 0,7.

## 5. Alcances

Funciona bien con fotos de árboles de manzana tomadas de frente, a una distancia y ángulo similares a las del dataset (huerto en
espaldera, cámara horizontal). Detecta y segmenta manzanas de tamaño mediano a grande con buena fiabilidad: con el umbral de la app (confianza 0,35, 1280 px),
el modelo final encuentra el **85 %** de las manzanas de 32 a 48 px, el **81 %** de las mayores y el **73 %** de las de 20 a 32 px, con un
15 % de falsas alarmas entre sus predicciones (validación, `reports/diagnostico_margen_colab_1280_small_s0.json`).

## 6. Limitaciones

- **Manzanas diminutas (< 20 px):** con el umbral de la app (0,35), el modelo final solo encuentra **1 de cada 4** (24,5 %, 51 de 208 en
  validación). Es la principal limitación y explica buena parte de la subcuenta (en test cuenta 2,8 manzanas de menos por foto). Bajar el
  umbral encuentra más a cambio de más falsas alarmas (en la ronda 1, con 0,25, el nano encontraba la mitad).
- **Anotación incompleta del dataset:** manzanas del suelo y de hileras del fondo no están etiquetadas; el modelo puede
  detectarlas y contarse como «falsa alarma» sin serlo.
- **Un solo huerto, una sola especie, cámara de celular de 2015-2016.** No se ha probado en otros huertos, variedades,
  condiciones de luz muy distintas, ni con cámaras de otra calidad.
- **Sin etiquetas de madurez:** el color mostrado en la app es descriptivo (fracción de píxeles rojos), no una clasificación de
  madurez ni de calidad comercial.
- **El modelo detecta peor las manzanas verde/amarillo que las rojas:** recall 0,61 frente a 0,86 en validación (0,25 de diferencia), y la
  brecha no se explica por tamaño (se repite igual o peor en manzanas grandes). Es un hallazgo exploratorio: el criterio para considerar
  una corrida adicional (menos saturación de color) se fijó antes de medir, y sí se cumplió; aun así se decidió no lanzarla porque no
  cambiaría el modelo final ya elegido, así que queda como limitación abierta para trabajo futuro.
- **Falla mucho en fotos de cerca, sin importar el color: es un problema de escala.** El modelo solo reconoce manzanas del
  tamaño con el que entrenó (20-40 px en fotos de huerto a 1280×720). Probado con dos fotos reales de manzanas verdes en primer
  plano (una rama de cerca y una canasta), donde las manzanas miden 100-180 px (5 a 8 veces más grandes): en la foto de la rama,
  **ninguna** de las 6-8 manzanas grandes y nítidas se detectó en ningún umbral de confianza; los pocos cuadros que aparecen
  caen sobre puntos borrosos del fondo, del tamaño «correcto» mal aplicado. La brecha de color (arriba) se suma, pero aquí el
  efecto dominante es el zoom/la distancia, no el color. La app cuenta solo lo que se ve **a la distancia de un huerto**; una
  foto de producto o de cerca no funciona.
- **Diámetro y peso son estimaciones:** dependen de una escala (mm/px) que da el usuario; sin escala solo se muestran píxeles.
  El peso asume una densidad típica (0,8 g/cm³) y una forma esférica, no medidas reales.
- **Concurrencia:** la app usa un modelo compartido con un candado (`Lock`); con varios usuarios a la vez, el último espera el
  tiempo de los demás (no se han probado más de 4 usuarios simultáneos).
- **La versión comprimida (INT8) no generaliza tan bien como la original:** en test pierde más calidad de la tolerada y «ve» manzanas
  que no existen (+9,5 por foto), sobre todo en fotos de otras hileras. No se usa en la app; si se quisiera usar, habría que recalibrarla
  con fotos más variadas y validarla con datos nuevos.
- **El aporte de la self-attention (C2PSA) es real pero pequeño** (+0,017 de Dice; +0,006 de mAP de caja en la media de 3 semillas, sin
  bootstrap), y se midió con el modelo nano a 1024 px: no se repitió la ablación con el modelo final (small a 1280 px).
- **Los criterios de la app se eligieron en validación**, que comparte origen con el entrenamiento; en fotos de otro año u otro huerto
  el desempeño puede bajar (el test oficial, de otras hileras y otro año, es la mejor estimación disponible).

## 7. Consideraciones éticas y de sesgo

- **No reemplaza una inspección humana** para decisiones logísticas o comerciales: es una estimación de apoyo.
- **Sesgo geográfico y de variedad:** entrenado en un solo huerto de Minnesota (EE. UU.); no se sabe cómo generaliza a otras
  variedades de manzana, huertos o climas.
- **Uso de datos:** licencia no comercial (CC BY-NC-SA 3.0 US); este proyecto es académico y no se distribuye con fines de lucro.
- **Sin datos personales:** las fotos del dataset son de huertos, sin personas identificables.

---
*Documento generado como parte del Proyecto 2 de Visión Computacional con Deep Learning (UAO). Ver también: [README](README.md) e
[informe.md](informe.md).*
