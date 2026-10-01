"""Entrena YOLO11-seg sobre MinneApple, con o sin C2PSA (ablación de self-attention) y en escala nano o small.

Las variantes c2psa/sin_c2psa se construyen desde YAML y cargan los pesos COCO de la misma escala: todo es
igual salvo la capa 10 (C2PSA frente a Identity). Sin parada temprana, para que las corridas sean comparables.

Uso:
    python scripts/entrenar.py --variante c2psa --semilla 0                                  # base nano
    python scripts/entrenar.py --variante c2psa --escala s --batch 2 --nombre small_s0        # T1: small en vez de nano
    python scripts/entrenar.py --variante c2psa --scale-aug 0.25 --nombre scale025_s0          # T2: menos zoom de aumento
    python scripts/entrenar.py --variante sin_c2psa --semilla 0 --epocas 1                     # prueba
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ARQUITECTURAS_SIN_C2PSA = {"n": str(ROOT / "configs" / "yolo11n-seg-sin-c2psa.yaml")}  # solo nano tiene la variante sin atención


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", choices=("c2psa", "sin_c2psa"), required=True)
    ap.add_argument("--escala", choices=("n", "s"), default="n", help="tamaño del modelo: nano (base) o small (T1)")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1024)  # nativo 1280 no cabe en 4 GB con máscaras finas
    ap.add_argument("--scale-aug", type=float, default=0.5, help="hiperparámetro 'scale' de aumentos (zoom); 0.25 = T2")
    ap.add_argument("--nombre", default=None)
    a = ap.parse_args()

    if a.variante == "sin_c2psa":
        if a.escala != "n":
            raise SystemExit("solo existe la variante sin_c2psa en escala nano (configs/yolo11n-seg-sin-c2psa.yaml)")
        arquitectura = ARQUITECTURAS_SIN_C2PSA["n"]
    else:
        arquitectura = f"yolo11{a.escala}-seg.yaml"  # YAML de Ultralytics: la escala se toma del nombre del archivo

    model = YOLO(arquitectura).load(str(ROOT / "models" / f"yolo11{a.escala}-seg.pt"))
    model.train(
        data=str(ROOT / "data" / "yolo" / "minneapple.yaml"),
        epochs=a.epocas,
        imgsz=a.imgsz,
        batch=a.batch,
        scale=a.scale_aug,
        seed=a.semilla,
        deterministic=True,
        patience=0,  # sin parada temprana: una parada distinta en cada corrida sesgaría la comparación de la ablación
        cache="disk",  # "ram" no es determinista (aviso de Ultralytics); la ablación necesita reproducibilidad
        workers=2,
        mask_ratio=8,  # máscaras de entrenamiento a 1/8: con 4 se desborda la GPU a la RAM
        max_det=300,  # hay imágenes con 123 manzanas
        project=str(ROOT / "runs"),
        name=a.nombre or f"{a.variante}_{a.escala}_s{a.semilla}",
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
