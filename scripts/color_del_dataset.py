"""Distribución de color (rojo o verde/amarillo) de las manzanas etiquetadas, por partición, con el mismo descriptor que usa la app.

Es un análisis descriptivo: no hay etiquetas de madurez ni de color, así que esto solo dice cuánto pesan las manzanas
no rojas en los datos, no si el descriptor acierta.

Salida: reports/color_dataset.json
Uso: python scripts/color_del_dataset.py
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from medidas import color_descriptivo  # noqa: E402

RAW = ROOT / "data" / "raw"
CARPETAS = {  # partición -> (imágenes, máscaras)
    "train": (RAW / "detection/train/images", RAW / "detection/train/masks"),
    "val": (RAW / "detection/train/images", RAW / "detection/train/masks"),
    "test": (RAW / "test_data/test_data/segmentation/images", RAW / "test_data/test_data/segmentation/masks"),
}
BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]  # fracción de píxeles rojos de la máscara


def main():
    conteo = {p: Counter() for p in CARPETAS}
    fracs = {p: [] for p in CARPETAS}
    for fila in csv.DictReader(open(ROOT / "data" / "splits.csv", encoding="utf-8")):
        p = fila["particion"]
        imgs, masks = CARPETAS[p]
        img = cv2.imread(str(imgs / fila["archivo"]))
        m = np.array(Image.open(masks / fila["archivo"]))
        for inst in np.unique(m)[1:]:
            color, frac = color_descriptivo(img, m == inst)
            conteo[p][color] += 1
            fracs[p].append(frac)
    out = {}
    for p, c in conteo.items():
        n = sum(c.values())
        assert n > 0 and c["rojo"] + c["verde/amarillo"] == n
        hist = np.histogram(fracs[p], bins=BINS)[0]
        out[p] = {"manzanas": n, "rojo": c["rojo"], "verde_amarillo": c["verde/amarillo"],
                  "pct_verde_amarillo": round(100 * c["verde/amarillo"] / n, 1),
                  "hist_fraccion_roja": dict(zip(["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"], map(int, hist)))}
    (ROOT / "reports" / "color_dataset.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
