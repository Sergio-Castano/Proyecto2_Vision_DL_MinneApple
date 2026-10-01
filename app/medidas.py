"""Aislamiento y medidas por manzana a partir de su máscara (§4.2 del enunciado: aislar el objeto sobre la imagen original).

`medir` es una función pura (imagen + máscara -> diccionario) para poder probarla sin modelo. Correr este archivo ejecuta su autoprueba.

Diámetro: diámetro equivalente del círculo con la misma área de la máscara (sqrt(4A/pi)). Con `mm_por_px` se convierte a mm y a un peso
aproximado de esfera x densidad. Es una ESTIMACIÓN: la manzana está parcialmente oculta por hojas, la escala cambia con la distancia a la
cámara y la densidad (0,8 g/cm3) es un valor típico, no medido.
"""

import math

import cv2
import numpy as np

DENSIDAD_G_CM3 = 0.8  # densidad típica de la manzana; se declara en la app y en la model card


def color_descriptivo(img_bgr: np.ndarray, mask: np.ndarray) -> tuple[str, float]:
    """'rojo' o 'verde/amarillo' según la fracción de píxeles rojos de la máscara. Solo descriptivo: NO es madurez ni se evalúa."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[mask]
    if len(hsv) == 0:
        return "sin datos", 0.0
    h, s = hsv[:, 0].astype(int), hsv[:, 1].astype(int)
    frac_rojo = float((((h < 12) | (h > 165)) & (s > 80)).mean())  # el rojo cruza el 0 del círculo de tonos
    return ("rojo" if frac_rojo >= 0.4 else "verde/amarillo"), frac_rojo


def medir(img_bgr: np.ndarray, mask: np.ndarray, mm_por_px: float | None = None) -> dict:
    """Aísla una manzana y mide su tamaño. `mask` es booleana, del tamaño de la imagen."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("máscara vacía")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    recorte = np.dstack([img_bgr[y0:y1, x0:x1], (mask[y0:y1, x0:x1] * 255).astype(np.uint8)])  # BGRA: fuera de la manzana, transparente
    area = int(mask.sum())
    d_px = math.sqrt(4 * area / math.pi)
    color, frac_rojo = color_descriptivo(img_bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1])  # solo el recorte: evita convertir la imagen entera por manzana
    out = {"recorte_bgra": recorte, "caja": (int(x0), int(y0), int(x1), int(y1)), "area_px": area,
           "diametro_px": d_px, "color": color, "fraccion_roja": frac_rojo}
    if mm_por_px:
        d_mm = d_px * mm_por_px
        out["diametro_mm"] = d_mm
        out["peso_g_aprox"] = math.pi / 6 * (d_mm / 10) ** 3 * DENSIDAD_G_CM3  # esfera: (pi/6) d^3, d en cm -> cm3 x g/cm3
    return out


def _autoprueba():
    img = np.zeros((200, 300, 3), np.uint8)
    yy, xx = np.mgrid[:200, :300]
    roja = (yy - 100) ** 2 + (xx - 80) ** 2 <= 20**2
    verde = (yy - 100) ** 2 + (xx - 220) ** 2 <= 20**2
    img[roja] = (0, 0, 220)  # BGR rojo
    img[verde] = (60, 200, 90)  # BGR verde
    r = medir(img, roja, mm_por_px=2.0)
    assert abs(r["diametro_px"] - 40) < 1.5, r["diametro_px"]
    assert r["color"] == "rojo" and medir(img, verde)["color"] == "verde/amarillo"
    assert r["recorte_bgra"].shape[2] == 4 and r["recorte_bgra"][..., 3].sum() // 255 == roja.sum()  # alfa = máscara
    assert (r["recorte_bgra"][..., 3][~roja[r["caja"][1]:r["caja"][3], r["caja"][0]:r["caja"][2]]] == 0).all()
    # d = 40 px x 2 mm/px = 80 mm = 8 cm -> (pi/6) * 8^3 * 0.8 = 214,5 g
    assert abs(r["peso_g_aprox"] - math.pi / 6 * 8**3 * 0.8) < 3, r["peso_g_aprox"]
    assert "diametro_mm" not in medir(img, roja)  # sin escala no se inventan milímetros
    print("autoprueba de medidas OK:", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k != "recorte_bgra"})


if __name__ == "__main__":
    _autoprueba()
