"""Grad-CAM a mano para YOLO11-seg.

Objetivo del gradiente: la suma de las confianzas de las anclas de la clase `apple` que superan `conf` (antes de NMS). Mapa = ReLU(Σ_k α_k A_k), con α_k el promedio
espacial del gradiente sobre la activación A de una capa que alimenta la cabeza de segmentación (16, 19 o 22: pasos 8, 16 y 32). Se calcula en float32 y con un hook
que se quita siempre; debe llamarse con el candado del `Analizador` puesto (los hooks son estado compartido del modelo).

Autoprueba (pesos de una corrida, imágenes de validación):  python app/gradcam.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CAPA_POR_DEFECTO = 16  # capa con mayor cociente de enfoque sobre las manzanas, de 3 candidatas (barrido en 20 imágenes de validación)
RELLENO = 114  # el mismo gris que usa el LetterBox de Ultralytics


def preparar(img_bgr: np.ndarray, imgsz: int = 1024, stride: int = 32):
    """Igual que el LetterBox de Ultralytics con auto=True: lado mayor a `imgsz`, relleno hasta múltiplo de `stride`, centrado.
    Devuelve el tensor (1,3,H,W) RGB en [0,1] y la geometría (top, left, alto, ancho) de la imagen dentro del lienzo."""
    h0, w0 = img_bgr.shape[:2]
    r = imgsz / max(h0, w0)
    nw, nh = round(w0 * r), round(h0 * r)
    img = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w0, h0) else img_bgr
    dw, dh = (imgsz - nw) % stride, (imgsz - nh) % stride
    top, left = round(dh / 2 - 0.1), round(dw / 2 - 0.1)
    lienzo = np.full((nh + dh, nw + dw, 3), RELLENO, np.uint8)
    lienzo[top:top + nh, left:left + nw] = img
    x = torch.from_numpy(lienzo[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
    return x, (top, left, nh, nw)


def grad_cam(net, img_bgr: np.ndarray, capa: int = CAPA_POR_DEFECTO, conf: float = 0.25, imgsz: int = 1024):
    """`net`: el `nn.Module` de YOLO (`YOLO(...).model`). Devuelve (mapa en [0,1] del tamaño de `img_bgr`, número de anclas usadas).
    Si ninguna ancla supera `conf`, el mapa es de ceros y el número es 0."""
    dev = next(net.parameters()).device
    x, (top, left, nh, nw) = preparar(img_bgr, imgsz)
    x = x.to(dev).requires_grad_(True)  # los pesos vienen congelados: sin esto no hay grafo hasta las activaciones
    guardado = []
    gancho = net.model[capa].register_forward_hook(lambda mod, inp, out: guardado.append(out))
    try:
        with torch.enable_grad():
            net.eval()
            salida = net(x)
            confianza = salida[0][0][0, 4]  # (anclas,): confianza de `apple`, ya con sigmoide
            elegidas = confianza >= conf
            n = int(elegidas.sum())
            if n == 0:
                return np.zeros(img_bgr.shape[:2], np.float32), 0
            A = guardado[0]
            (g,) = torch.autograd.grad(confianza[elegidas].sum(), A)
    finally:
        gancho.remove()
    cam = F.relu((g.mean((2, 3), keepdim=True) * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0, top:top + nh, left:left + nw]
    cam = cv2.resize(cam.detach().cpu().numpy(), (img_bgr.shape[1], img_bgr.shape[0]))
    m = cam.max()
    return (cam / m if m > 0 else cam).astype(np.float32), n


def superponer(img_bgr: np.ndarray, cam: np.ndarray, alfa: float = 0.85, gamma: float = 0.5) -> np.ndarray:
    """Mapa de calor sobre la imagen; donde el mapa vale 0 la imagen queda intacta.
    `gamma` < 1 realza los valores bajos: el mapa se concentra en picos pequeños sobre las manzanas y, sin realce, casi no se ve.
    Solo cambia la visualización; el mapa y las métricas (`enfoque`) usan `cam` sin modificar."""
    v = cam**gamma
    calor = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    a = (alfa * v)[..., None]
    return (img_bgr * (1 - a) + calor * a).astype(np.uint8)


def enfoque(cam: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """(fracción de la energía del mapa dentro de las manzanas, fracción de la imagen que ocupan, cociente).
    Cociente > 1: el mapa se concentra en las manzanas más de lo que ocuparían al azar."""
    e = float(cam.sum())
    dentro = float((cam * mask).sum() / e) if e > 0 else 0.0
    area = float(mask.mean())
    return dentro, area, (dentro / area if area > 0 else 0.0)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analisis import RAIZ, Analizador, cargar_imagen
    from PIL import Image

    a = Analizador(device="cpu")
    net = a.model.model.float().eval()
    ruta = sorted((RAIZ / "data" / "yolo" / "images" / "val").glob("*.png"))[3]
    img = cargar_imagen(ruta.read_bytes())
    gt = np.array(Image.open(RAIZ / "data" / "raw" / "detection" / "train" / "masks" / ruta.name)) > 0
    for capa in (16, 19):  # la 22 solo alimenta las anclas de paso 32: con manzanas de ~30 px su mapa sale vacío
        cam, n = grad_cam(net, img, capa=capa)
        assert cam.shape == img.shape[:2] and np.isfinite(cam).all() and cam.max() == 1.0 and n > 0, (capa, n, cam.max())
        d, ar, q = enfoque(cam, gt)
        assert q > 1.0, f"capa {capa}: el mapa no se concentra en las manzanas (cociente {q:.2f})"
        print(f"capa {capa}: {n} anclas | energía dentro de las manzanas {d:.2f} frente a {ar:.2f} de área -> cociente {q:.1f}")
    assert superponer(img, cam).shape == img.shape
    sin = grad_cam(net, img, conf=0.999)  # sin anclas: mapa vacío, sin fallar
    assert sin[1] == 0 and sin[0].max() == 0
    print("OK: Grad-CAM de", ruta.name)
