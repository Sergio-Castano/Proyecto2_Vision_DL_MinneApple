"""Recall de detección por color (rojo frente a verde/amarillo), análisis exploratorio.

Solo validación (nunca test). Cada manzana real se clasifica con el mismo descriptor de color que usa la app
(`app/medidas.py:color_descriptivo`, fracción de píxeles rojos de la máscara) y se empareja con la mejor predicción libre de
IoU >= 0,5 (misma lógica que `scripts/diagnostico_margen.py`), en orden de confianza. Umbral y resolución: los de la app
(conf 0,35, 1280 px).

Criterio fijado antes de medir: si el recall de verde/amarillo queda más de 0,05 por debajo del de rojo, se considera una
corrida exploratoria (c2psa, semilla 0, hsv_s=0.3, hsv_v=0.3) — no se lanza aquí, solo se señala.

Uso: python scripts/recall_por_color.py [--nombre colab_1280_small_s0] [--conf 0.35] [--imgsz 1280] [--device 0]
Salida: reports/recall_color_<nombre>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from medidas import color_descriptivo  # noqa: E402
from diagnostico_margen import iou  # noqa: E402 - misma función de emparejamiento que el recall por tamaño

MASCARAS = RAIZ / "data" / "raw" / "detection" / "train" / "masks"
IMAGENES = RAIZ / "data" / "raw" / "detection" / "train" / "images"
UMBRAL_ALERTA = 0.05  # si verde/amarillo queda más de esto por debajo de rojo, correspondería una corrida exploratoria


def instancias_reales(nombre_archivo: str):
    """boxes (Nx4), colores (lista de 'rojo'/'verde/amarillo') de cada manzana real de una imagen."""
    m = np.array(Image.open(MASCARAS / nombre_archivo))
    img = np.array(Image.open(IMAGENES / nombre_archivo))[:, :, ::-1]  # RGB -> BGR, como espera color_descriptivo
    boxes, colores = [], []
    for i in np.unique(m)[1:]:
        mask_i = m == i
        ys, xs = np.nonzero(mask_i)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        colores.append(color_descriptivo(img, mask_i)[0])
    return np.array(boxes, float).reshape(-1, 4), colores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", default="colab_1280_small_s0")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    modelo = YOLO(str(RAIZ / "runs" / a.nombre / "weights" / "best.pt"))
    imgs = sorted((RAIZ / "data" / "yolo" / "images" / "val").glob("*.png"))

    acierto = {"rojo": [0, 0], "verde/amarillo": [0, 0]}
    for ruta in imgs:
        gt, colores = instancias_reales(ruta.name)
        r = modelo.predict(str(ruta), imgsz=a.imgsz, conf=a.conf, max_det=300, device=a.device, verbose=False)[0]
        pr = r.boxes.xyxy.cpu().numpy()
        emparejada = np.zeros(len(gt), bool)
        if len(gt) and len(pr):
            m = iou(gt, pr)
            confs = r.boxes.conf.cpu().numpy()
            for j in np.argsort(-confs):
                libres = np.where(~emparejada & (m[:, j] >= 0.5))[0]
                if len(libres):
                    k = libres[np.argmax(m[libres, j])]
                    emparejada[k] = True
        for c, encontrada in zip(colores, emparejada):
            acierto[c][0] += int(encontrada)
            acierto[c][1] += 1

    recall = {c: v[0] / v[1] for c, v in acierto.items()}
    diferencia = recall["rojo"] - recall["verde/amarillo"]  # positivo => verde/amarillo peor
    salida = {
        "modelo": a.nombre, "particion": "val", "conf": a.conf, "imgsz": a.imgsz,
        "rojo": {"encontradas": acierto["rojo"][0], "total": acierto["rojo"][1], "recall": round(recall["rojo"], 4)},
        "verde_amarillo": {"encontradas": acierto["verde/amarillo"][0], "total": acierto["verde/amarillo"][1],
                            "recall": round(recall["verde/amarillo"], 4)},
        "diferencia_rojo_menos_verde": round(diferencia, 4),
        "umbral_alerta_decision_0007": UMBRAL_ALERTA,
        "dispara_corrida_exploratoria": bool(diferencia > UMBRAL_ALERTA),
    }
    (RAIZ / "reports" / f"recall_color_{a.nombre}.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(salida, indent=2, ensure_ascii=False))
    if salida["dispara_corrida_exploratoria"]:
        print("\n>>> Verde/amarillo queda más de 0,05 por debajo de rojo: correspondería la corrida "
              "exploratoria (c2psa, semilla 0, hsv_s=0.3, hsv_v=0.3). No se lanza automáticamente.")


if __name__ == "__main__":
    main()
