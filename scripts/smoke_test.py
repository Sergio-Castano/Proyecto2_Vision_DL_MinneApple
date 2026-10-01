"""Prueba de humo del entorno: versiones, GPU y un forward de YOLO11n-seg a 1280 px.
Crea models/ y descarga allí los pesos COCO (yolo11n-seg.pt) si faltan: es el primer paso tras instalar.

Uso (con el entorno activo):
    python scripts/smoke_test.py
"""

import platform
from pathlib import Path

import onnxruntime
import streamlit
import torch
import torchvision
import ultralytics
from ultralytics import YOLO

print(f"python {platform.python_version()} | torch {torch.__version__} | torchvision {torchvision.__version__}")
print(f"ultralytics {ultralytics.__version__} | onnxruntime {onnxruntime.__version__} | streamlit {streamlit.__version__}")
print(f"CPU: {platform.processor()}")
assert torch.cuda.is_available(), "CUDA no disponible"
print(f"GPU: {torch.cuda.get_device_name(0)}")

PESOS = Path(__file__).resolve().parents[1] / "models" / "yolo11n-seg.pt"
PESOS.parent.mkdir(exist_ok=True)
model = YOLO(str(PESOS))  # la primera vez lo descarga de los releases oficiales de Ultralytics
torch.cuda.reset_peak_memory_stats()
res = model.predict(torch.rand(1, 3, 736, 1280), imgsz=1280, device=0, verbose=False)
assert res[0].boxes is not None
print(f"forward 1280 ok | memoria pico GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
print("SMOKE TEST OK")
