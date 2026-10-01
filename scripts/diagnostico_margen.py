"""Diagnóstico sin reentrenar del margen de mejora: solo validación, nunca test.

1. Recall por tamaño de manzana: empareja cajas reales y predichas (IoU >= 0,5, de mayor a menor confianza) con el umbral dado.
2. mAP de caja y de máscara al inferir con 1024 px (como se entrenó) y con 1280 px (resolución nativa del dataset), mismo modelo.

Uso: python scripts/diagnostico_margen.py [--nombre c2psa_s0] [--conf 0.25] [--device 0]     Salida: reports/diagnostico_margen_<nombre>.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]
MASCARAS = RAIZ / "data" / "raw" / "detection" / "train" / "masks"
TRAMOS = [(0, 20), (20, 32), (32, 48), (48, 10_000)]  # lado mayor de la caja real, en px de la imagen original


def cajas_reales(ruta_mascara: Path) -> np.ndarray:
    m = np.array(Image.open(ruta_mascara))
    out = []
    for i in np.unique(m)[1:]:
        ys, xs = np.nonzero(m == i)
        out.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.array(out, float).reshape(-1, 4)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", default="c2psa_s0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    modelo = YOLO(str(RAIZ / "runs" / a.nombre / "weights" / "best.pt"))
    imgs = sorted((RAIZ / "data" / "yolo" / "images" / "val").glob("*.png"))

    salida = {"modelo": a.nombre, "particion": "val", "conf": a.conf, "recall_por_tamano": {}, "mAP_por_resolucion": {}}
    for imgsz in (1024, 1280):
        acierto = {t: [0, 0] for t in TRAMOS}  # [encontradas, total]
        fp = n_pred = 0
        for ruta in imgs:
            gt = cajas_reales(MASCARAS / ruta.name)
            r = modelo.predict(str(ruta), imgsz=imgsz, conf=a.conf, max_det=300, device=a.device, verbose=False)[0]
            pr = r.boxes.xyxy.cpu().numpy()
            usadas = np.zeros(len(pr), bool)
            emparejada = np.zeros(len(gt), bool)
            if len(gt) and len(pr):
                m = iou(gt, pr)
                for j in np.argsort(-r.boxes.conf.cpu().numpy()):  # cada predicción, de más a menos confiada, a su mejor caja libre
                    libres = np.where(~emparejada & (m[:, j] >= 0.5))[0]
                    if len(libres):
                        k = libres[np.argmax(m[libres, j])]
                        emparejada[k] = usadas[j] = True
            lado = np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]) if len(gt) else np.zeros(0)
            for t in TRAMOS:
                sel = (lado >= t[0]) & (lado < t[1])
                acierto[t][0] += int(emparejada[sel].sum())
                acierto[t][1] += int(sel.sum())
            fp += int((~usadas).sum())
            n_pred += len(pr)
        salida["recall_por_tamano"][str(imgsz)] = {
            f"{t[0]}-{t[1] if t[1] < 10_000 else 'mas'} px": {"encontradas": v[0], "total": v[1], "recall": round(v[0] / max(v[1], 1), 3)}
            for t, v in acierto.items()}
        salida["recall_por_tamano"][str(imgsz)]["falsas_alarmas"] = fp
        salida["recall_por_tamano"][str(imgsz)]["predicciones"] = n_pred
        m = modelo.val(data=str(RAIZ / "data" / "yolo" / "minneapple.yaml"), split="val", imgsz=imgsz, batch=1, device=a.device,
                       mask_ratio=1, plots=False, verbose=False, workers=0, project=str(RAIZ / "runs" / "val_tmp"), exist_ok=True)
        salida["mAP_por_resolucion"][str(imgsz)] = {"caja_50_95": round(m.box.map, 4), "caja_50": round(m.box.map50, 4),
                                                   "mascara_50_95": round(m.seg.map, 4), "mascara_50": round(m.seg.map50, 4)}
        print(imgsz, json.dumps(salida["recall_por_tamano"][str(imgsz)], ensure_ascii=False), salida["mAP_por_resolucion"][str(imgsz)], flush=True)
    (RAIZ / "reports" / f"diagnostico_margen_{a.nombre}.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
