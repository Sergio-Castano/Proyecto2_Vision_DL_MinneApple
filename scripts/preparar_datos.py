"""Prepara MinneApple para YOLO-seg: partición por video, máscaras -> polígonos, EDA y comprobaciones.

Entrada (data/raw, descargado de la UMN):
  detection/train/{images,masks}                  670 imágenes, máscaras de instancia (valor = id, 0 = fondo)
  test_data/test_data/segmentation/{images,masks} 331 imágenes de test oficial, mismo formato
Salida:
  data/yolo/{images,labels}/{train,val,test}   imágenes (enlace duro) y etiquetas YOLO-seg
  data/yolo/minneapple.yaml                    configuración de datos para Ultralytics
  data/splits.csv                              archivo, grupo (video), partición
  reports/eda.json                             tamaños y conteos por partición

Uso: python scripts/preparar_datos.py
"""

import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics.data.converter import merge_multi_segment

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "yolo"

# Validación: dos videos completos (frames vecinos de un mismo video no pueden caer en train y en val).
# 59 + 34 = 93 de 670 imágenes (13,9 %), 34,6 manzanas por imagen, cerca del test oficial (37,1).
# Descartado el primer intento con 20150919_174151: 98 manzanas por imagen (23 % de las de train), val poco representativo.
VAL_GROUPS = {"20150921_131234", "20150921_131729"}


def group_of(name: str) -> str:
    """'20150919_174151_image1.png' -> '20150919_174151' (secuencia de video); test -> 'dataset1_back'."""
    return name.rsplit("_", 1)[0]


def mask_to_polygons(mask: np.ndarray):
    """Una línea YOLO-seg por instancia. Devuelve (polígonos normalizados, estadísticas por instancia)."""
    h, w = mask.shape
    polys, stats = [], []
    for inst in np.unique(mask)[1:]:
        m = (mask == inst).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c.reshape(-1, 2) for c in cnts if len(c) >= 3]
        if not cnts:  # instancia de 1-2 píxeles: no forma polígono
            stats.append({"descartada": True})
            continue
        seg = np.concatenate(merge_multi_segment(cnts)) if len(cnts) > 1 else cnts[0]
        polys.append(seg / [w, h])
        x, y, bw, bh = cv2.boundingRect(m)
        stats.append({"area": int(m.sum()), "w": bw, "h": bh, "partes": len(cnts)})
    return polys, stats


def polygon_iou(mask: np.ndarray, polys) -> float:
    """IoU entre la unión de máscaras original y la rasterización de los polígonos (fidelidad de la conversión)."""
    h, w = mask.shape
    canvas = np.zeros_like(mask, dtype=np.uint8)
    for p in polys:
        cv2.fillPoly(canvas, [np.round(p * [w, h]).astype(np.int32)], 1)
    gt = mask > 0
    return float((gt & (canvas > 0)).sum() / max((gt | (canvas > 0)).sum(), 1))


def main():
    sources = {
        "train": (RAW / "detection/train/images", RAW / "detection/train/masks"),
        "test": (RAW / "test_data/test_data/segmentation/images", RAW / "test_data/test_data/segmentation/masks"),
    }
    if OUT.exists():
        shutil.rmtree(OUT)
    rows, eda, ious, por_grupo = [], {}, {}, {}
    for src, (img_dir, mask_dir) in sources.items():
        for img_path in sorted(img_dir.glob("*.png")):
            g = group_of(img_path.name)
            split = "test" if src == "test" else ("val" if g in VAL_GROUPS else "train")
            mask = np.array(Image.open(mask_dir / img_path.name))
            polys, stats = mask_to_polygons(mask)
            for sub in ("images", "labels"):
                (OUT / sub / split).mkdir(parents=True, exist_ok=True)
            os.link(img_path, OUT / "images" / split / img_path.name)  # enlace duro: no duplica 2 GB
            lines = ["0 " + " ".join(f"{v:.6f}" for v in p.reshape(-1)) for p in polys]
            (OUT / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            rows.append({"archivo": img_path.name, "grupo": g, "particion": split})
            e = eda.setdefault(split, {"imagenes": 0, "por_imagen": [], "inst": []})
            e["imagenes"] += 1
            e["por_imagen"].append(len(stats))
            e["inst"] += stats
            ious.setdefault(split, []).append(polygon_iou(mask, polys))
            pg = por_grupo.setdefault(g, {"particion": split, "imagenes": 0, "manzanas": 0})
            pg["imagenes"] += 1
            pg["manzanas"] += sum(1 for s in stats if not s.get("descartada"))

    # Comprobación de validez: ningún video en dos particiones.
    groups = {}
    for r in rows:
        groups.setdefault(r["grupo"], set()).add(r["particion"])
    shared = {g: s for g, s in groups.items() if len(s) > 1}
    assert not shared, f"grupos en más de una partición: {shared}"

    with open(ROOT / "data" / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["archivo", "grupo", "particion"])
        w.writeheader()
        w.writerows(rows)

    (OUT / "minneapple.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: apple\n", encoding="utf-8"
    )

    report = {}
    for split, e in eda.items():
        descartadas = sum(1 for s in e["inst"] if s.get("descartada"))
        inst = [s for s in e["inst"] if not s.get("descartada")]
        areas = np.array([s["area"] for s in inst])
        sides = np.array([max(s["w"], s["h"]) for s in inst])
        report[split] = {
            "imagenes": e["imagenes"],
            "grupos": sorted({r["grupo"] for r in rows if r["particion"] == split}),
            "manzanas": len(inst),
            "descartadas_sin_poligono": descartadas,
            "manzanas_por_imagen": {"media": round(float(np.mean(e["por_imagen"])), 1), "min": int(np.min(e["por_imagen"])),
                                    "max": int(np.max(e["por_imagen"]))},
            "area_px": {"mediana": float(np.median(areas)), "p10": float(np.percentile(areas, 10)),
                        "p90": float(np.percentile(areas, 90))},
            "lado_mayor_px_mediana": float(np.median(sides)),
            "frac_pequenas_coco_lt_32x32": round(float((areas < 32 * 32).mean()), 3),
            "frac_instancias_partidas": round(float(np.mean([s["partes"] > 1 for s in inst])), 3),
            "iou_poligono_vs_mascara": {"media": round(float(np.mean(ious[split])), 4),
                                        "min": round(float(np.min(ious[split])), 4)},
            "por_video": {g: {"imagenes": v["imagenes"], "manzanas": v["manzanas"],
                              "manzanas_por_imagen": round(v["manzanas"] / v["imagenes"], 1)}
                          for g, v in sorted(por_grupo.items()) if v["particion"] == split},
        }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "eda.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    assert report["test"]["manzanas"] + report["test"]["descartadas_sin_poligono"] == 12285, "el test oficial tiene 12.285 manzanas (JSON COCO de la UMN)"
    assert all(r["iou_poligono_vs_mascara"]["media"] > 0.9 for r in report.values()), "conversión a polígonos infiel"
    print("OK: particiones sin grupos compartidos; conversión verificada")


if __name__ == "__main__":
    main()
