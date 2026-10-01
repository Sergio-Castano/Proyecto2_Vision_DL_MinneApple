"""Compara C2PSA frente a sin C2PSA (ablación de self-attention).

Criterio (fijado ANTES de ver el resultado):
  - mAP de caja y de máscara: media ± desviación entre las 3 semillas de cada variante (NO se remuestrea; el mAP ya agrega
    sobre todas las imágenes, remuestrear imágenes con reemplazo no da una distribución muestral válida de un ranking).
  - Dice de la unión y error de conteo: bootstrap PAREADO por imagen. Para cada semilla i, c2psa_s{i} y sin_c2psa_s{i}
    se evaluaron sobre las MISMAS imágenes; se calcula la diferencia por imagen (c2psa − sin_c2psa) y se juntan las
    diferencias de las 3 semillas (3 × n_imagenes). Se remuestrea esa lista con reemplazo 10.000 veces, se promedia
    cada remuestra, y el intervalo de confianza del 95 % son los percentiles 2,5 y 97,5. "Confirmada" solo si el
    intervalo queda entero por encima (o por debajo) de cero.
  - No se adopta ni se descarta C2PSA por este resultado: el enunciado exige el bloque y se reporta lo que aporte,
    aunque sea nada.

Uso:
    python scripts/comparar_ablacion.py --particion val    # dry run / exploratorio, para probar el script (NO es el resultado final)
    python scripts/comparar_ablacion.py --particion test   # resultado final; solo se ejecuta UNA VEZ
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEMILLAS = (0, 1, 2)
N_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 42  # reproducible


def cargar_json(nombre: str, particion: str) -> dict:
    ruta = ROOT / "reports" / "eval" / f"{nombre}_{particion}.json"
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta}: corre primero 'python scripts/evaluar.py --nombre {nombre} --particion {particion} --conf 0.35'")
    return json.loads(ruta.read_text(encoding="utf-8"))


def cargar_csv(nombre: str, particion: str) -> dict:
    """imagen -> fila, para emparejar c2psa_s{i} y sin_c2psa_s{i} por nombre de archivo (por si el orden difiere)."""
    ruta = ROOT / "reports" / "eval" / f"{nombre}_{particion}_por_imagen.csv"
    return {r["imagen"]: r for r in csv.DictReader(open(ruta, encoding="utf-8"))}


def media_desv(variante: str, particion: str, campo_grupo: str, campo: str) -> dict:
    vals = [cargar_json(f"{variante}_s{s}", particion)[campo_grupo][campo] for s in SEMILLAS]
    return {"valores": [round(v, 4) for v in vals], "media": float(np.mean(vals)), "desv": float(np.std(vals, ddof=1))}


def bootstrap_pareado(diferencias: np.ndarray, n: int = N_BOOTSTRAP, seed: int = SEMILLA_BOOTSTRAP) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diferencias), size=(n, len(diferencias)))
    medias = diferencias[idx].mean(axis=1)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return {"diferencia_media": float(diferencias.mean()), "ic95_lo": float(lo), "ic95_hi": float(hi),
            "n_diferencias": len(diferencias), "confirmada": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", choices=["val", "test"], required=True)
    a = ap.parse_args()
    if a.particion == "val":
        print("AVISO: 'val' es un dry run para probar el script. El resultado que cuenta para el proyecto es en 'test', una sola vez.\n")

    resumen = {"particion": a.particion, "semillas": list(SEMILLAS), "n_bootstrap": N_BOOTSTRAP}

    # mAP: media ± desviación entre semillas, sin remuestrear (ya agrega sobre todas las imágenes; ver el criterio arriba).
    resumen["mAP"] = {}
    for variante in ("c2psa", "sin_c2psa"):
        resumen["mAP"][variante] = {
            "caja_50_95": media_desv(variante, a.particion, "caja", "mAP50-95"),
            "mascara_50_95": media_desv(variante, a.particion, "mascara", "mAP50-95"),
        }
    for campo in ("caja_50_95", "mascara_50_95"):
        c, s = resumen["mAP"]["c2psa"][campo]["media"], resumen["mAP"]["sin_c2psa"][campo]["media"]
        resumen["mAP"][f"diferencia_{campo}"] = round(c - s, 4)

    # Dice y conteo: bootstrap pareado por imagen, semilla a semilla.
    dif_dice, dif_error_conteo = [], []
    for s in SEMILLAS:
        con = cargar_csv(f"c2psa_s{s}", a.particion)
        sin = cargar_csv(f"sin_c2psa_s{s}", a.particion)
        assert con.keys() == sin.keys(), f"semilla {s}: las imágenes evaluadas no coinciden entre variantes"
        for img, fc in con.items():
            fs = sin[img]
            dif_dice.append(float(fc["dice"]) - float(fs["dice"]))
            error_c = abs(int(fc["n_pred"]) - int(fc["n_gt"]))
            error_s = abs(int(fs["n_pred"]) - int(fs["n_gt"]))
            dif_error_conteo.append(error_c - error_s)

    resumen["dice_union"] = bootstrap_pareado(np.array(dif_dice))
    resumen["dice_union"]["lectura"] = "diferencia = Dice(c2psa) - Dice(sin_c2psa); positivo => con C2PSA es mejor"
    resumen["error_conteo_absoluto"] = bootstrap_pareado(np.array(dif_error_conteo))
    resumen["error_conteo_absoluto"]["lectura"] = "diferencia = error(c2psa) - error(sin_c2psa); NEGATIVO => con C2PSA es mejor (menos error)"
    resumen["lectura"] = "'confirmada' exige que el IC95 no cruce el cero; ver 'lectura' de cada métrica para el signo que significa mejora"

    out = ROOT / "reports" / f"ablacion_{a.particion}{'_exploratorio' if a.particion == 'val' else ''}.json"
    out.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print("\nescrito:", out)


if __name__ == "__main__":
    main()
