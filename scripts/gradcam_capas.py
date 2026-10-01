"""Elige la capa del Grad-CAM: cociente de enfoque medio de las capas 16, 19 y 22 en 20 imágenes de validación.

Criterio fijado antes de medir: gana la capa de mayor cociente medio; si la diferencia es < 5 %, la 16. Solo validación, nunca test.
También cuenta cuántas anclas con confianza >= conf hay en cada escala, que explica por qué la capa 22 puede dar un mapa vacío.

Uso: python scripts/gradcam_capas.py [--nombre c2psa_s0] [--device cpu]     Salida: reports/gradcam_capas.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
from analisis import cargar_imagen  # noqa: E402
from gradcam import enfoque, grad_cam, preparar  # noqa: E402

CAPAS = (16, 19, 22)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", default="c2psa_s0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    net = YOLO(str(RAIZ / "runs" / a.nombre / "weights" / "best.pt")).model.float().to(a.device).eval()
    rutas = sorted((RAIZ / "data" / "yolo" / "images" / "val").glob("*.png"))
    rutas = [rutas[i] for i in np.linspace(0, len(rutas) - 1, a.n).round().astype(int)]
    filas, por_escala = [], []
    for ruta in rutas:
        img = cargar_imagen(ruta.read_bytes())
        gt = np.array(Image.open(RAIZ / "data" / "raw" / "detection" / "train" / "masks" / ruta.name)) > 0
        x, _ = preparar(img)
        with torch.no_grad():
            conf_ancla = net(x.to(a.device))[0][0][0, 4].cpu().numpy()
        h, w = x.shape[2] // 8, x.shape[3] // 8  # anclas de paso 8, 16 y 32 en ese orden
        cortes = np.cumsum([0, h * w, (h // 2) * (w // 2), (h // 4) * (w // 4)])
        por_escala.append([int((conf_ancla[cortes[i]:cortes[i + 1]] >= a.conf).sum()) for i in range(3)])
        for capa in CAPAS:
            cam, n = grad_cam(net, img, capa=capa, conf=a.conf)
            d, area, q = enfoque(cam, gt)
            filas.append({"imagen": ruta.name, "capa": capa, "anclas": n, "vacio": bool(cam.max() == 0), "dentro": d, "area": area, "cociente": q})
        print(ruta.name, [round(f["cociente"], 1) for f in filas[-3:]], "anclas por escala", por_escala[-1], flush=True)

    resumen = {}
    for capa in CAPAS:
        f = [x for x in filas if x["capa"] == capa]
        resumen[str(capa)] = {"cociente_medio": float(np.mean([x["cociente"] for x in f])), "energia_dentro_media": float(np.mean([x["dentro"] for x in f])),
                              "mapas_vacios": int(sum(x["vacio"] for x in f)), "imagenes": len(f)}
    mejor = max(CAPAS, key=lambda c: resumen[str(c)]["cociente_medio"])
    elegida = 16 if resumen["16"]["cociente_medio"] >= 0.95 * resumen[str(mejor)]["cociente_medio"] else mejor
    salida = {"pesos": a.nombre, "conf": a.conf, "particion": "val", "resumen": resumen, "elegida": elegida,
              "anclas_por_escala_medias": np.mean(por_escala, axis=0).round(1).tolist(), "filas": filas}
    (RAIZ / "reports" / "gradcam_capas.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in salida.items() if k != "filas"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
