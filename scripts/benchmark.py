"""Tiempos de inferencia en CPU y GPU.

Casos: PyTorch (.pt) en CPU y GPU, y ONNX Runtime FP32 e INT8 en CPU (el onnxruntime instalado es solo de CPU). Etapas:
  forward   la red sola, misma entrada cuadrada 1x3xSxS para todos los formatos (S = --imgsz, 1280 para el modelo final)
  pipeline  preproceso + red + NMS + máscaras (Ultralytics), cada motor con su forma nativa (columna forma_entrada)
  app       Analizador.analizar de la app, sin y con Grad-CAM (solo .pt)
Método: calentamiento, 50 mediciones por caso (mediana y p95), cuda.synchronize() en GPU, casos intercalados en orden aleatorio en cada ronda,
mismo número de hilos en PyTorch y en el forward de ONNX Runtime (el pipeline de Ultralytics crea su propia sesión de ORT con los hilos por defecto).
Cada fila lleva el hardware exacto. El CSV se sobrescribe: una sola corrida por método.

Se niega a medir si hay un entrenamiento en marcha (GPU > 15 % o CPU > 30 %), salvo con --prueba (n=3, no escribe en reports/).

Uso (modelo final):
      python scripts/benchmark.py [--nombre colab_1280_small_s0] [--imgsz 1280] [--int8 best_int8_sincabeza.onnx] [--n 50] [--hilos 4]
                                  [--dispositivos cpu gpu] [--out reports/benchmark.csv]
"""

import argparse
import copy
import csv
import platform
import random
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil
import torch
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app"))
from analisis import Analizador  # noqa: E402
from gradcam import preparar  # noqa: E402  - el mismo LetterBox de Ultralytics: da la forma real de entrada del .pt

METODO = "v2"  # versión del método de medición: si cambia, cambia esta etiqueta (v2: modelo final, 1280 px, conf 0,35)


def nombre_cpu() -> str:
    if platform.system() == "Windows":
        try:
            return subprocess.check_output(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], text=True).strip()
        except Exception:  # noqa: BLE001
            pass
    return platform.processor() or platform.machine()


def ocupado() -> bool:
    """¿Hay un entrenamiento u otra carga que falsearía las mediciones?"""
    try:
        gpu = int(subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True).split()[0])
    except Exception:  # noqa: BLE001 - sin nvidia-smi no hay GPU que vigilar
        gpu = 0
    return gpu > 15 or psutil.cpu_percent(interval=2.0) > 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", default="colab_1280_small_s0")  # modelo final (small + C2PSA, 1280 px; mejor mAP en validación)
    ap.add_argument("--imgsz", type=int, default=1280)  # debe coincidir con el lado del ONNX exportado
    ap.add_argument("--conf", type=float, default=0.35)  # umbral de la app: minimiza el error de conteo en validación
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--hilos", type=int, default=psutil.cpu_count(logical=False) or 4)
    ap.add_argument("--dispositivos", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"])
    ap.add_argument("--int8", default="best_int8_sincabeza.onnx")  # INT8 con la cabeza en FP32: la única variante que no dio mAP 0
    ap.add_argument("--out", default=None)
    ap.add_argument("--prueba", action="store_true")
    a = ap.parse_args()
    if a.prueba:
        a.n = 3
    elif ocupado():
        sys.exit("La CPU o la GPU están ocupadas (¿hay un entrenamiento en marcha?): las cifras saldrían falsas. Espera a que termine o usa --prueba.")
    gpu = "gpu" in a.dispositivos and torch.cuda.is_available()
    if "gpu" in a.dispositivos and not gpu:
        print("aviso: no hay GPU CUDA, se omiten los casos de GPU")
    out = Path(a.out) if a.out else (None if a.prueba else RAIZ / "reports" / "benchmark.csv")

    pesos = RAIZ / "runs" / a.nombre / "weights"
    pt, onnx_fp32, onnx_int8 = pesos / "best.pt", pesos / "best.onnx", pesos / a.int8
    S = a.imgsz
    x = np.random.default_rng(0).random((1, 3, S, S), dtype=np.float32)
    xt = torch.from_numpy(x)
    img = cv2.imread(str(sorted((RAIZ / "data" / "yolo" / "images" / "val").glob("*.png"))[3]))  # 720x1280 vertical, como todo el dataset
    alto, ancho = preparar(img, S)[0].shape[2:]
    cuadrada, auto = f"1x3x{S}x{S}", f"auto ({alto}x{ancho})"  # auto: la forma que usa el .pt con esta foto (rectangular)
    torch.set_num_threads(a.hilos)
    tareas = []

    def agregar(formato, motor, disp, etapa, forma, fn, hilos, sync=None):
        tareas.append({"formato": formato, "motor": motor, "dispositivo": disp, "etapa": etapa, "forma_entrada": forma, "hilos": hilos,
                       "fn": fn, "sync": sync, "t": []})

    def forward(net, entrada):
        def f():
            with torch.inference_mode():
                net(entrada)
        return f

    def predecir(modelo, dispositivo):
        return lambda: modelo.predict(img, imgsz=S, conf=a.conf, retina_masks=True, device=dispositivo, verbose=False)

    net = YOLO(str(pt)).model.fuse(verbose=False).float().eval()
    if "cpu" in a.dispositivos:
        agregar("pt", "pytorch", "cpu", "forward", cuadrada, forward(net, xt), a.hilos)
        agregar("pt", "pytorch", "cpu", "pipeline", auto, predecir(YOLO(str(pt)), "cpu"), a.hilos)
        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = a.hilos
        for etiqueta, ruta in (("onnx-fp32", onnx_fp32), ("onnx-int8", onnx_int8)):
            if not ruta.exists():
                print(f"aviso: falta {ruta.name}; se omite (ejecuta scripts/optimizar.py)")
                continue
            sesion = ort.InferenceSession(str(ruta), sess_options=opciones, providers=["CPUExecutionProvider"])
            nombre_entrada = sesion.get_inputs()[0].name
            agregar(etiqueta, "onnxruntime", "cpu", "forward", cuadrada, lambda s=sesion, n=nombre_entrada: s.run(None, {n: x}), a.hilos)
            agregar(etiqueta, "onnxruntime", "cpu", "pipeline", f"{S}x{S}", predecir(YOLO(str(ruta), task="segment"), "cpu"), "defecto")
        app_cpu = Analizador(pesos=str(pt), device="cpu", imgsz=S)
        app_cpu.calentar()
        agregar("pt", "pytorch", "cpu", "app", auto, lambda: app_cpu.analizar(img, conf=a.conf), a.hilos)
        agregar("pt", "pytorch", "cpu", "app+gradcam", auto, lambda: app_cpu.analizar(img, conf=a.conf, con_gradcam=True), a.hilos)
    if gpu:
        net_g, xg = copy.deepcopy(net).cuda(), xt.cuda()
        agregar("pt", "pytorch", "gpu", "forward", cuadrada, forward(net_g, xg), None, torch.cuda.synchronize)
        agregar("pt", "pytorch", "gpu", "pipeline", auto, predecir(YOLO(str(pt)), 0), None, torch.cuda.synchronize)
        app_gpu = Analizador(pesos=str(pt), device="cuda:0", imgsz=S)
        app_gpu.calentar()
        agregar("pt", "pytorch", "gpu", "app", auto, lambda: app_gpu.analizar(img, conf=a.conf), None, torch.cuda.synchronize)
        agregar("pt", "pytorch", "gpu", "app+gradcam", auto, lambda: app_gpu.analizar(img, conf=a.conf, con_gradcam=True), None, torch.cuda.synchronize)

    for t in tareas:  # calentamiento
        for _ in range(2 if a.prueba else 5):
            t["fn"]()
        if t["sync"]:
            t["sync"]()
    for _ in range(a.n):
        random.shuffle(tareas)  # orden aleatorio en cada ronda: ningún caso se beneficia siempre de medirse después de otro
        for t in tareas:
            if t["sync"]:
                t["sync"]()
            t0 = time.perf_counter()
            t["fn"]()
            if t["sync"]:
                t["sync"]()
            t["t"].append(1000 * (time.perf_counter() - t0))

    cpu = host = nombre_cpu()
    filas = []
    for t in sorted(tareas, key=lambda t: (t["etapa"], t["dispositivo"], t["formato"])):
        filas.append({"metodo": METODO, "fecha": date.today().isoformat(), "modelo": a.nombre, "formato": t["formato"], "motor": t["motor"],
                      "dispositivo": t["dispositivo"], "hardware": torch.cuda.get_device_name(0) if t["dispositivo"] == "gpu" else cpu,
                      "cpu_host": host, "etapa": t["etapa"], "forma_entrada": t["forma_entrada"], "hilos": t["hilos"] if t["hilos"] is not None else "",
                      "n": len(t["t"]), "mediana_ms": round(float(np.median(t["t"])), 2), "p95_ms": round(float(np.percentile(t["t"], 95)), 2),
                      "torch": torch.__version__, "onnxruntime": ort.__version__})
    for f in filas:
        print(f"{f['etapa']:12s} {f['dispositivo']:3s} {f['formato']:10s} {f['forma_entrada']:16s} mediana {f['mediana_ms']:9.2f} ms   p95 {f['p95_ms']:9.2f} ms   (n={f['n']})")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
        print("escrito:", out)
    else:
        print("(prueba: no se escribió ningún CSV)")


if __name__ == "__main__":
    main()
