# MinneApple: detección, segmentación y atención para estimar la cosecha de manzana

Proyecto 2 de Visión Computacional con Deep Learning (Maestría en IA y Ciencia de Datos, UAO). Sobre fotos de árboles de manzana, el sistema **detecta y cuenta** cada manzana,
**segmenta** cada una (para aislarla del fondo y medir su diámetro) y usa **mecanismos de atención** (self-attention C2PSA y Grad-CAM). Se despliega en Streamlit con un Cloudflare Tunnel.
**Sustentación: 3 de octubre de 2026.**

> Modelo final: `colab_1280_small_s0` (YOLO11s-seg + C2PSA, 1280 px), evaluado en test una sola vez. Cifras de detección, segmentación,
> atención, optimización y tiempos en `reports/`, [`MODEL_CARD.md`](MODEL_CARD.md) e [`informe.md`](informe.md). Los pesos entrenados ya
> están incluidos y el proyecto corre sin reentrenar nada (sección «Pesos entrenados», abajo).

## Qué resuelve

| Tarea | Cómo | Métrica |
|---|---|---|
| Detección y conteo | YOLO11n-seg (cajas), una clase: `apple` | mAP@0.5:0.95 (estilo COCO), mAP@0.5, error de conteo por imagen |
| Segmentación de instancias | Misma cabeza de máscaras; recorte de cada manzana sobre la imagen original | mAP de máscara, Dice e IoU |
| Atención | C2PSA (self-attention de YOLO11), con ablación «con» y «sin»; Grad-CAM a mano | Comparación con bootstrap pareado entre semillas |

Además de cada manzana se muestra su diámetro (en píxeles; en mm y peso aproximado solo si el usuario da una escala) y un **color descriptivo** rojo o verde/amarillo. No hay etiquetas de madurez en el dataset,
así que el color **no** se presenta como madurez ni se evalúa como tarea.

## Estructura

```
app/             aplicación Streamlit (streamlit_app.py), análisis con Lock (analisis.py) y medidas (medidas.py: aislamiento, diámetro, color, peso)
configs/         yolo11n-seg-sin-c2psa.yaml (variante de la ablación)
notebooks/       notebook de sustentación (secciones a-f, ejecutado) y un notebook de respaldo para entrenar en Colab (1280 px)
reports/         cifras y análisis (eda.json, color_dataset.json, eval/, ablacion_test.json, benchmark.csv, int8_vs_fp32...)
runs/            los 3 checkpoints que usan el notebook y la app: best.pt, args.yaml y el log de cada uno
scripts/         datos, entrenamiento y evaluación (ver «Comandos en orden»)
requirements.txt dependencias con las versiones reales
--- no se versionan (se descargan o se generan; ver las secciones de abajo) ---
data/            el dataset MinneApple: se descarga con los comandos de «Datos: MinneApple»
models/          pesos preentrenados de COCO: los descarga scripts/smoke_test.py
```

## Datos: MinneApple

- **Fuente:** Häni, N., Roy, P., Isler, V. (2019). *MinneApple: A Benchmark Dataset for Apple Detection and Segmentation*. Data Repository for the University of Minnesota (DRUM),
  <https://doi.org/10.13020/8ecp-3r13> (handle <http://hdl.handle.net/11299/206575>). Artículo: IEEE Robotics and Automation Letters, 2020, <https://doi.org/10.1109/LRA.2020.2965061>.
- **Licencia: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 US (CC BY-NC-SA 3.0 US).** Hay que citar la fuente, no se puede usar con fines comerciales y las obras derivadas se comparten con la misma licencia.
  **Los datos no están en este repositorio**: cada persona los descarga.
- **Qué se usa:** `detection.tar.gz` (670 imágenes de entrenamiento con máscaras de instancia y 331 de test) y `test_data.zip` (etiquetas del test, publicadas en 2022). `counting.tar.gz` no hace falta.

| Archivo | Tamaño | MD5 |
|---|---|---|
| `detection.tar.gz` | 1,8 GB | `d24a2f70144f1a3c5a52ffef66c50630` |
| `test_data.zip` | 1,25 GB | `0c24727552bc3e94cad6d8f1c851c887` |

Desde la raíz del repositorio (crea `data/raw/` si no existe; comandos para Git Bash, Linux o macOS; en PowerShell usa `curl.exe` y `certutil -hashfile <archivo> MD5`):

```bash
mkdir -p data/raw && cd data/raw
curl -L -A "Mozilla/5.0" -o detection.tar.gz "https://conservancy.umn.edu/server/api/core/bitstreams/3ef26f04-6467-469b-9857-f443ffa1bb61/content"
curl -L -A "Mozilla/5.0" -o test_data.zip    "https://conservancy.umn.edu/server/api/core/bitstreams/7856047c-1f23-43ef-a50d-9327ec2ae0ff/content"
md5sum detection.tar.gz test_data.zip        # deben coincidir con la tabla
tar -xzf detection.tar.gz
python -m zipfile -e test_data.zip test_data
cd ../..
```

Debe quedar `data/raw/detection/{train,test}` y `data/raw/test_data/test_data/{detection,segmentation,counting}`. Si el comando `curl` falla, la página del dataset (<https://conservancy.umn.edu/items/e1bb4015-e92a-4295-822c-d21d277ecfbd>) permite la descarga manual desde el navegador.
El test (331 fotos) es el oficial publicado por los autores en 2022 (etiquetas COCO JSON y máscaras), de otras hileras y otro año que train: es
más creíble que un test recortado del propio train y permite comparar con la cifra del artículo original. Train (670 fotos) se divide en train/val
**por secuencia de video, no por foto suelta** (los fotogramas de un mismo video son casi idénticos; partir por foto dejaría fotogramas casi
iguales en las dos particiones), verificado por código que ningún video cae en dos particiones.

## Preparar el entorno

Requiere **Python 3.12** y, para entrenar, una **GPU NVIDIA** (se entrenó en una GTX 1050 de 4 GB con CUDA 12.4).

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows;  en Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test.py           # comprueba versiones y GPU, y descarga los pesos COCO a models/
```

`requirements.txt` fija las dependencias directas a las versiones reales del entorno de entrenamiento (torch 2.6.0+cu124, ultralytics 8.4.149, onnxruntime 1.30.0, streamlit 1.63.0…). La resolución de dependencias se comprobó
para Windows y Linux, pero **no se ha probado una instalación limpia completa**; las dependencias indirectas pueden variar y las cifras pueden diferir en el tercer decimal. Sin GPU NVIDIA hay que quitar el sufijo `+cu124` de torch y torchvision.

## Comandos en orden

Todos se ejecutan desde la raíz, con el entorno activo.

| # | Comando | Qué hace |
|---|---|---|
| 1 | `python scripts/smoke_test.py` | Comprueba el entorno y descarga `models/yolo11n-seg.pt` |
| 2 | `python scripts/preparar_datos.py` | Parte train/val por video, convierte las máscaras a polígonos YOLO-seg y escribe `data/yolo/`, `data/splits.csv` y `reports/eda.json`. **Borra y recrea `data/yolo/`**, no lo ejecutes mientras se entrena. Usa enlaces duros: `data/raw` y `data/yolo` deben estar en la misma unidad |
| 3 | `python scripts/color_del_dataset.py` | (Opcional) proporción de manzanas rojas y verde/amarillo por partición → `reports/color_dataset.json` |
| 4 | `python scripts/entrenar.py --variante c2psa --semilla 0` | Entrena una corrida en `runs/c2psa_s0/` (40 épocas, 1024 px, lote 4; unos 2,2 min por época en la GTX 1050). Variante sin atención: `--variante sin_c2psa` |
| 5 | `python scripts/evaluar.py --nombre c2psa_s0 --particion val --conf 0.15 0.25 0.35 0.45` | mAP, Dice/IoU y error de conteo en val; sirve para elegir el umbral. El mAP de máscara usa la referencia a resolución completa (`mask_ratio=1`): la del entrenamiento (`mask_ratio=8`) infravalora el mAP en unos 0,06 y no es representativa |
| 6 | `python scripts/evaluar.py --nombre c2psa_s0 --particion test --conf <umbral>` | Igual en test. **Se ejecuta una sola vez**, con el umbral elegido en val |
| 7 | `streamlit run app/streamlit_app.py` | Aplicación: subir foto, cámara o ejemplo de validación; conteo, máscaras, manzanas aisladas, diámetro, color, peso aproximado (con escala) y Grad-CAM. Busca los pesos en `models/minneapple.pt` o en `MINNEAPPLE_PESOS`; `MINNEAPPLE_DEVICE=cpu` fuerza CPU. Pruebas: `python app/analisis.py`, `python app/gradcam.py` y `python app/prueba_concurrencia.py` |
| 8 | `python scripts/gradcam_capas.py` | (Análisis) barre 3 capas candidatas en 20 imágenes de validación y elige la de mayor concentración de atención sobre las manzanas reales (cociente de enfoque 54,0 en la capa 16, frente a 47,9 y 0,9 en las otras dos) → `reports/gradcam_capas.json` |
| 9 | `jupyter nbconvert --to notebook --execute --inplace notebooks/sustentacion.ipynb --ExecutePreprocessor.timeout=900` | Ejecuta el notebook y lo guarda con sus salidas. Necesita los datos preparados y las corridas en `runs/`. Secciones a-f completas |
| 10 | `python scripts/optimizar.py --excluir-cabeza` | Modelo final por defecto: exporta a ONNX FP32, forma cuadrada 1280×1280 (necesaria porque Ultralytics no calcula el mAP de un ONNX exportado con forma rectangular), y cuantiza a INT8 estático con 100 imágenes de train, con la cabeza en FP32 (sin esa opción, todo en INT8: mezcla en la misma escala de 8 bits coordenadas de hasta 1.280 px y confianzas de 0 a 1, y da mAP 0) → `reports/optimizacion_colab_1280_small_s0_sincabeza.json`. Se compara en el mismo motor con `python scripts/evaluar.py --nombre <etiqueta> --pesos <onnx> --particion val --imgsz 1280 --conf 0.35 --device cpu`; `python scripts/comparar_int8.py` compara INT8 y FP32 imagen por imagen |
| 11 | `python scripts/benchmark.py` | Tiempos CPU y GPU (forward, pipeline y app; mediana y p95 de 50 mediciones, hardware exacto) → `reports/benchmark.csv`. **Se niega a medir si hay un entrenamiento en marcha** (una GPU o CPU ocupada falsearía los tiempos); `--prueba` ensaya con 3 iteraciones sin escribir nada |
| — | `notebooks/colab_entrenamiento_1280.ipynb` (se ejecuta en Google Colab, no en local) | Entrenamiento de respaldo a 1280 px nativo con una GPU T4 (16 GB): no cabe en la GTX 1050 local (4 GB). Autocontenido (no depende de tener el resto del código a mano); reconstruye la misma partición por video que el entrenamiento local, para que los dos sean comparables. Instrucciones dentro del notebook |

Las 6 corridas de la ablación (2 variantes × semillas 0, 1 y 2) las lanza `bash scripts/cola_entrenamiento.sh` (unas 9 a 10 horas estimadas; con el portátil enchufado y sin suspender; usa el `python` del entorno activo, o `PYTHON=ruta/al/python.exe bash scripts/cola_entrenamiento.sh` si no está activado en esa terminal). 40 épocas y **sin parada temprana**, para que las 6 corridas sean directamente comparables entre sí (una parada distinta en cada una sesgaría la comparación); `mask_ratio=8` durante el entrenamiento por memoria (las métricas finales se recalculan con `mask_ratio=1`, más preciso, ver el paso 5); lote 4 por la VRAM de la GTX 1050 (4 GB). Los aumentos de datos son los predeterminados de Ultralytics (mosaico, escala, traslación, HSV…), sin ajustar a mano para este dataset.

**Probado:** app + Cloudflare Tunnel, desde un celular real (28-sep), sin errores.

## Pesos entrenados

El repositorio incluye los 3 checkpoints que hacen falta para que el notebook y la app corran sin reentrenar nada:
`runs/colab_1280_small_s0/` (modelo final), `runs/c2psa_s0/` y `runs/sin_c2psa_s0/` (los dos de la ablación), con su `best.pt`, `args.yaml` y
el registro de entrenamiento de cada uno — unos 34 MB en total. El resto de las cifras (61 archivos en `reports/`, incluida cada métrica de
este README, la model card y el informe) también está incluido.

Se conservan solo las corridas que usan el notebook y la app; otras semillas y variantes de tamaño o de aumentos que se probaron durante el
desarrollo no hacen falta para reproducir los resultados reportados. Si se quiere reentrenar algo desde cero, los comandos de la sección de
arriba son los mismos que se usaron para obtener estas cifras.

## Licencias y atribución

- **Datos:** CC BY-NC-SA 3.0 US (arriba). El uso es académico y no comercial.
- **Ultralytics (YOLO11):** AGPL-3.0.
- **Código de este repositorio:** licencia aún sin definir.
