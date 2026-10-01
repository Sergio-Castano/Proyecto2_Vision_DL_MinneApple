"""Prueba de concurrencia: varios hilos analizan a la vez con un solo `Analizador`.

Cada resultado (conteo y mapa de Grad-CAM) se compara con una referencia calculada antes, en serie. Con el candado deben coincidir todos.
`--sin-candado` quita el `Lock` para mostrar qué pasaría sin él (puede fallar con excepciones o cerrar el proceso).

Uso:  python app/prueba_concurrencia.py [--hilos 4] [--rondas 2] [--sin-candado]      (MINNEAPPLE_DEVICE=cpu si la GPU está ocupada)
"""

import argparse
import contextlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisis import RAIZ, Analizador, cargar_imagen  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--rondas", type=int, default=2)
    ap.add_argument("--sin-candado", action="store_true")
    a = ap.parse_args()

    analizador = Analizador()
    analizador.calentar()
    rutas = sorted((RAIZ / "data" / "yolo" / "images" / "val").glob("*.png"))
    rutas = rutas[:: max(1, len(rutas) // a.hilos)][: a.hilos]  # imágenes distintas: si los mapas se cruzan, se nota
    imgs = [cargar_imagen(p.read_bytes()) for p in rutas]
    ref = [analizador.analizar(im, con_gradcam=True) for im in imgs]
    print("referencia en serie:", [r["n"] for r in ref], "manzanas por imagen", flush=True)

    if a.sin_candado:
        analizador._lock = contextlib.nullcontext()

    def trabajo(k):
        return k % len(imgs), analizador.analizar(imgs[k % len(imgs)], con_gradcam=True)

    distintos = errores = 0
    with ThreadPoolExecutor(a.hilos) as ex:
        futuros = [ex.submit(trabajo, k) for k in range(a.hilos * a.rondas)]
        for f in futuros:
            try:
                i, r = f.result()
            except Exception as e:  # noqa: BLE001 - se cuenta y se informa
                errores += 1
                print("  excepción:", type(e).__name__, str(e)[:90])
                continue
            if not (r["n"] == ref[i]["n"] and np.allclose(r["cam"], ref[i]["cam"], atol=1e-4)):
                distintos += 1
    total = a.hilos * a.rondas
    print(f"{total} análisis con {a.hilos} hilos ({'SIN candado' if a.sin_candado else 'con candado'}): "
          f"{distintos} resultados distintos de la referencia, {errores} excepciones")
    if not a.sin_candado:
        assert distintos == 0 and errores == 0, "el candado no evitó resultados cruzados"
        print("OK")


if __name__ == "__main__":
    main()
