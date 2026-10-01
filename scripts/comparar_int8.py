"""Compara, imagen por imagen, el ONNX INT8 con el ONNX FP32 en la misma partición.

Descriptivo: no cambia ninguna decisión (el INT8 se aceptó en validación). Bootstrap pareado por imagen como en
comparar_ablacion.py (10.000 remuestras, IC95 por percentiles), y el desglose por conjunto de origen (prefijo del nombre).

Uso: python scripts/comparar_int8.py [--particion test]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def leer(nombre):
    with open(ROOT / "reports" / "eval" / f"{nombre}_por_imagen.csv", encoding="utf-8") as f:
        return {r["imagen"]: r for r in csv.DictReader(f) if r["conf"] == "0.35"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", default="test")
    a = ap.parse_args()
    fp32, int8 = leer(f"final_onnx_fp32_{a.particion}"), leer(f"final_onnx_int8_sincabeza_{a.particion}")
    assert fp32.keys() == int8.keys(), "las dos evaluaciones deben cubrir las mismas imágenes"
    imgs = sorted(fp32)
    err = lambda r: int(r["n_pred"]) - int(r["n_gt"])  # noqa: E731
    difs = {  # INT8 − FP32, por imagen
        "dice": np.array([float(int8[i]["dice"]) - float(fp32[i]["dice"]) for i in imgs]),
        "error_conteo_abs": np.array([abs(err(int8[i])) - abs(err(fp32[i])) for i in imgs], float),
        "manzanas_de_mas": np.array([int(int8[i]["n_pred"]) - int(fp32[i]["n_pred"]) for i in imgs], float),
    }
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(imgs), (10_000, len(imgs)))
    res = {"particion": a.particion, "n_imagenes": len(imgs), "diferencia": "INT8 - FP32", "metricas": {}, "por_conjunto": {}}
    for k, d in difs.items():
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        res["metricas"][k] = {"media": float(d.mean()), "ic95": [float(lo), float(hi)]}
    grupos = defaultdict(list)
    for j, i in enumerate(imgs):
        grupos[i.rsplit("_", 1)[0]].append(j)
    for g, js in sorted(grupos.items()):
        res["por_conjunto"][g] = {"n": len(js), "gt_medio": float(np.mean([int(fp32[imgs[j]]["n_gt"]) for j in js])),
                                  **{k: float(d[js].mean()) for k, d in difs.items()}}
    salida = ROOT / "reports" / f"int8_vs_fp32_{a.particion}.json"
    salida.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
