"""Análisis de una foto: manzanas (cajas y máscaras) y medidas por manzana.

Un único `Analizador` se comparte entre todas las sesiones de Streamlit. La inferencia va detrás de un `Lock`: sin él, varios
usuarios a la vez con hooks compartidos devuelven mapas cruzados sin avisar (comprobado con `app/prueba_concurrencia.py`).

Autoprueba (con los pesos de una corrida y una imagen de validación):
    python app/analisis.py [imagen.png]
"""

import io
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gradcam import enfoque, grad_cam, superponer  # noqa: E402
from medidas import medir  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
LADO_MAX = 1280  # las fotos de celular (12 MP) se reducen a este lado mayor: es el tamaño con que se entrenó y evita esperas de varios segundos
CONF_POR_DEFECTO = 0.35  # elegido en val con evaluar.py: minimiza el error de conteo


def paleta(n: int) -> np.ndarray:
    """n colores BGR distinguibles entre sí (y legibles sobre fondo claro u oscuro), uno por instancia."""
    return np.random.default_rng(0).integers(60, 255, (max(n, 1), 3))


def buscar_pesos() -> Path:
    # Modelo final: small + C2PSA a 1280 px, semilla 0 (mejor mAP en validación que nano, confirmado con 2 semillas). Antes era c2psa_s0 (nano, 1024 px).
    candidatos = [os.environ.get("MINNEAPPLE_PESOS"), RAIZ / "models" / "minneapple.pt",
                  RAIZ / "runs" / "colab_1280_small_s0" / "weights" / "best.pt", RAIZ / "runs" / "c2psa_s0" / "weights" / "best.pt"]
    for c in candidatos:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        "No se encontraron los pesos del modelo. Descarga el release del repositorio en models/minneapple.pt, "
        "o define MINNEAPPLE_PESOS con la ruta de un .pt (ver README, «Pesos entrenados»)."
    )


def cargar_imagen(datos: bytes) -> np.ndarray:
    """Bytes de una foto (PNG, JPG…) -> BGR uint8, respetando la rotación EXIF de los celulares."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(datos))).convert("RGB")
    return np.ascontiguousarray(np.array(img)[:, :, ::-1])


def reducir(img: np.ndarray, lado_max: int = LADO_MAX) -> tuple[np.ndarray, float]:
    """Reduce (nunca amplía) para que el lado mayor no pase de `lado_max`. Devuelve (imagen, factor <= 1)."""
    f = min(1.0, lado_max / max(img.shape[:2]))
    return (img if f == 1.0 else cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)), f


class Analizador:
    def __init__(self, pesos=None, device=None, imgsz: int = 1280):  # 1280: el modelo final se entrenó y se mide a esa resolución
        self.pesos = Path(pesos) if pesos else buscar_pesos()
        self.device = device or os.environ.get("MINNEAPPLE_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.model = YOLO(str(self.pesos))
        self._lock = threading.Lock()

    def calentar(self):
        """Una pasada en vacío al arrancar: el primer visitante no paga la carga de kernels."""
        self.analizar(np.full((720, 1280, 3), 114, np.uint8))

    def analizar(self, img_bgr: np.ndarray, conf: float = CONF_POR_DEFECTO, mm_por_px: float | None = None,
                 con_gradcam: bool = False) -> dict:
        """`mm_por_px`: milímetros por píxel de la imagen ORIGINAL (None = sin escala, no se inventan mm)."""
        t0 = time.perf_counter()
        img, factor = reducir(img_bgr)
        with self._lock:  # el Grad-CAM registra un hook en el modelo compartido: también va dentro del candado
            r = self.model.predict(img, imgsz=self.imgsz, conf=conf, iou=0.7, max_det=300, retina_masks=True,
                                   device=self.device, verbose=False)[0]
            cam = grad_cam(self.model.model, img, conf=conf)[0] if con_gradcam else None
        masks = np.zeros((0,) + img.shape[:2], bool) if r.masks is None else r.masks.data.cpu().numpy().astype(bool)
        confs = np.zeros(0) if r.boxes is None else r.boxes.conf.cpu().numpy()
        mm = mm_por_px / factor if mm_por_px else None  # un píxel de la imagen reducida cubre 1/factor píxeles originales
        instancias, mascaras_validas = [], []
        for m, c in zip(masks, confs):
            if m.any():
                d = medir(img, m, mm)
                d["conf"] = float(c)
                instancias.append(d)
                mascaras_validas.append(m)
        union = masks.any(0) if len(masks) else np.zeros(img.shape[:2], bool)
        colores = paleta(len(instancias))  # un color por instancia: Ultralytics r.plot() coloreaba por CLASE (una sola: "apple"),
        # así que todas las cajas salían del mismo color y parecía segmentación semántica en vez de por instancia
        anotada = img.copy()
        for m, color in zip(mascaras_validas, colores):
            anotada[m] = (0.75 * img[m] + 0.25 * color).astype(np.uint8)  # tinte suave: se ve que es una manzana, no un parche de color
        aisladas = (img * union[..., None]).astype(np.uint8)  # todas las manzanas, con su color real, sobre fondo negro
        for d, m, color in zip(instancias, mascaras_validas, colores):
            color_t = tuple(int(c) for c in color)
            x0, y0, x1, y1 = d["caja"]
            cv2.rectangle(anotada, (x0, y0), (x1 - 1, y1 - 1), color_t, 2)
            contornos, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(aisladas, contornos, -1, color_t, 2)  # separa visualmente las manzanas que se tocan
        res = {
            "imagen": img, "factor": factor, "anotada": anotada, "instancias": instancias, "n": len(instancias),
            "aisladas": aisladas,
            "con_escala": mm is not None, "dispositivo": self.device,
        }
        if cam is not None:  # (energía dentro de las manzanas detectadas, fracción de imagen que ocupan, cociente)
            res.update(cam=cam, gradcam=superponer(img, cam), enfoque=enfoque(cam, union))
        res["ms"] = 1000 * (time.perf_counter() - t0)
        return res


if __name__ == "__main__":
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((RAIZ / "data" / "yolo" / "images" / "val").glob("*.png"))[0]
    a = Analizador()
    a.calentar()
    r = a.analizar(cargar_imagen(ruta.read_bytes()), mm_por_px=0.5)
    assert r["n"] > 0, "no detectó ninguna manzana en una imagen de validación"
    for d in r["instancias"]:  # el recorte es la máscara: alfa > 0 sólo dentro de la manzana
        assert (d["recorte_bgra"][..., 3] > 0).sum() == d["area_px"]
        assert d["peso_g_aprox"] > 0 and d["diametro_mm"] > 0
    assert r["aisladas"].shape == r["imagen"].shape and r["anotada"].shape == r["imagen"].shape
    print(f"OK: {ruta.name}: {r['n']} manzanas en {r['ms']:.0f} ms ({r['dispositivo']}); pesos {a.pesos.name}")
