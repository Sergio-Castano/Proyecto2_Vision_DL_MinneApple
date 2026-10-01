"""Evalúa un checkpoint sobre val o test: mAP (Ultralytics), Dice/IoU de la unión de máscaras y error de conteo por imagen.

Protocolo: el umbral de confianza se elige en val; test se evalúa UNA vez con el umbral elegido.
Salidas en reports/eval/: <nombre>_<particion>.json (resumen) y <nombre>_<particion>_por_imagen.csv (para bootstrap pareado).

Uso:
    python scripts/evaluar.py --nombre c2psa_s0 --particion val --conf 0.15 0.25 0.35 0.45
    python scripts/evaluar.py --nombre c2psa_s0 --particion test --conf 0.25
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MASCARAS = {  # verdad de terreno a resolución completa (val sale de las máscaras de train)
    "val": RAW / "detection" / "train" / "masks",
    "test": RAW / "test_data" / "test_data" / "segmentation" / "masks",
}


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return (1.0, 1.0) if union == 0 else (2 * inter / total, inter / union)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", required=True, help="corrida en runs/ (usa weights/best.pt)")
    ap.add_argument("--particion", choices=["val", "test"], required=True)
    ap.add_argument("--conf", type=float, nargs="+", default=[0.25])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--mask-ratio", type=int, default=1, help="resolución de la máscara de referencia del mAP (1 = completa; el mask_ratio=8 del entrenamiento infravalora el mAP ~0,06)")
    ap.add_argument("--pesos", default=None, help="ruta a un .pt (por defecto runs/<nombre>/weights/best.pt)")
    ap.add_argument("--iou-nms", type=float, default=0.7, help="umbral de NMS: dos cajas se fusionan si su IoU lo supera")
    ap.add_argument("--sufijo", default=None, help="para no pisar el reporte al barrer --iou-nms con varios valores")
    a = ap.parse_args()

    pesos = a.pesos or str(ROOT / "runs" / a.nombre / "weights" / "best.pt")
    model = YOLO(pesos)
    datos = str(ROOT / "data" / "yolo" / "minneapple.yaml")
    m = model.val(data=datos, split=a.particion, imgsz=a.imgsz, batch=4, device=a.device, plots=False,
                  verbose=False, workers=0, mask_ratio=a.mask_ratio, project=str(ROOT / "runs" / "val_tmp"), exist_ok=True)
    resumen = {
        "pesos": pesos, "particion": a.particion, "imgsz": a.imgsz, "mask_ratio": a.mask_ratio, "iou_nms": a.iou_nms,
        "caja": {"mAP50-95": m.box.map, "mAP50": m.box.map50, "P": m.box.mp, "R": m.box.mr},
        "mascara": {"mAP50-95": m.seg.map, "mAP50": m.seg.map50, "P": m.seg.mp, "R": m.seg.mr},
        "por_umbral": {},
    }

    imagenes = sorted((ROOT / "data" / "yolo" / "images" / a.particion).glob("*.png"))
    filas = []
    for conf in a.conf:
        for img in imagenes:
            gt_inst = np.array(Image.open(MASCARAS[a.particion] / img.name))
            gt = gt_inst > 0
            r = model.predict(str(img), imgsz=a.imgsz, conf=conf, iou=a.iou_nms, max_det=300, retina_masks=True,
                              device=a.device, verbose=False)[0]
            n_pred = 0 if r.masks is None else len(r.masks)
            pred = np.zeros_like(gt) if r.masks is None else r.masks.data.any(0).cpu().numpy().astype(bool)
            if pred.shape != gt.shape:  # por si la máscara sale con otra resolución
                pred = np.array(Image.fromarray(pred.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST), bool)
            d, i = dice_iou(pred, gt)
            filas.append({"conf": conf, "imagen": img.name, "n_gt": len(np.unique(gt_inst)) - 1,
                          "n_pred": n_pred, "dice": d, "iou": i})
        f = [x for x in filas if x["conf"] == conf]
        err = np.array([x["n_pred"] - x["n_gt"] for x in f])
        resumen["por_umbral"][str(conf)] = {
            "dice_union": float(np.mean([x["dice"] for x in f])), "iou_union": float(np.mean([x["iou"] for x in f])),
            "conteo_MAE": float(np.abs(err).mean()), "conteo_sesgo": float(err.mean()),
            "conteo_total_pred": int(sum(x["n_pred"] for x in f)), "conteo_total_gt": int(sum(x["n_gt"] for x in f)),
        }

    out = ROOT / "reports" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    base = f"{a.nombre}_{a.particion}" + (f"_{a.sufijo}" if a.sufijo else "")
    (out / f"{base}.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / f"{base}_por_imagen.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
