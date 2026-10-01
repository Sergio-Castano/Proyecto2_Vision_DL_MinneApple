"""Exporta el modelo a ONNX FP32 (forma fija y cuadrada) y lo cuantiza a INT8 estático con ONNX Runtime.

Salida, junto a los pesos: best.onnx (FP32), best_int8.onnx (o best_int8_sincabeza.onnx con --excluir-cabeza) y un JSON con tamaños y opciones.
Después se comparan en el mismo motor con:  python scripts/evaluar.py --nombre <etiqueta> --pesos <onnx> --particion val --imgsz <lado> --device cpu

Uso (modelo final):
    python scripts/optimizar.py --nombre colab_1280_small_s0 [--imgsz 1280] [--n-calib 100] [--excluir-cabeza]
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

RAIZ = Path(__file__).resolve().parents[1]
# Calibración por tramos de 2 imágenes: a 1280 px guardar las salidas intermedias de las 100 no cabe en RAM. ORT calcula el min/max de
# cada tramo y los fusiona (mínimo de los mínimos, máximo de los máximos), así que el resultado es idéntico a calibrar las 100 de una vez.
# NO usar CalibMaxIntermediateOutputs: en onnxruntime 1.30 descarta los datos al llegar al límite sin calcular su rango.
TRAMO_CALIB = 2


def preparar(bgr: np.ndarray, lado: int) -> np.ndarray:
    """Igual que la inferencia: letterbox a la forma fija lado x lado, BGR -> RGB, [0,1], (1,3,H,W) float32.
    Cuadrada porque Ultralytics no calcula el mAP (val) de un ONNX no cuadrado."""
    x = LetterBox(new_shape=(lado, lado), auto=False, stride=32)(image=bgr)[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    return np.ascontiguousarray(x)


class Lector(CalibrationDataReader):
    """Entrega imágenes reales de train de una en una (sin cargarlas todas en memoria).
    `__len__` y `set_range` son los que pide ORT para calibrar por tramos (CalibStridedMinMax)."""

    def __init__(self, rutas, entrada, lado):
        self.rutas, self.entrada, self.lado = list(rutas), entrada, lado
        self.set_range(0, len(self.rutas))

    def __len__(self):
        return len(self.rutas)

    def set_range(self, start_index: int, end_index: int):
        self._it = ({self.entrada: preparar(cv2.imread(str(r)), self.lado)} for r in self.rutas[start_index:end_index])

    def get_next(self):
        return next(self._it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", default="colab_1280_small_s0")  # modelo final (small + C2PSA, 1280 px)
    ap.add_argument("--imgsz", type=int, default=1280, help="lado de la entrada cuadrada del ONNX; debe coincidir con el de evaluación")
    ap.add_argument("--n-calib", type=int, default=100)
    ap.add_argument("--excluir-cabeza", action="store_true", help="no cuantiza los nodos de la cabeza (/model.23/*): cuantizarlos junto con el resto mezcla escalas de coordenadas y confianzas y da mAP 0")
    ap.add_argument("--reporte", default=None)
    a = ap.parse_args()

    pesos = RAIZ / "runs" / a.nombre / "weights" / "best.pt"
    carpeta = pesos.parent
    reporte = Path(a.reporte) if a.reporte else RAIZ / "reports" / f"optimizacion_{a.nombre}.json"
    fp32 = Path(YOLO(str(pesos)).export(format="onnx", imgsz=a.imgsz, opset=17, dynamic=False, simplify=True, device="cpu")).resolve()
    print("ONNX FP32:", fp32.name, f"{fp32.stat().st_size / 2**20:.1f} MB", flush=True)

    origen, preproceso = fp32, False
    try:  # inferencia de formas y optimizaciones que recomienda ONNX Runtime antes de cuantizar
        pp = carpeta / "best_pp.onnx"
        quant_pre_process(str(fp32), str(pp))
        origen, preproceso = pp, True
    except Exception as e:  # noqa: BLE001 - se sigue sin preproceso y se deja constancia en el reporte
        print("aviso: el preproceso falló, se cuantiza el ONNX tal cual:", type(e).__name__, str(e)[:100])

    entrada = ort.InferenceSession(str(origen), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    imgs = sorted((RAIZ / "data" / "yolo" / "images" / "train").glob("*.png"))
    rutas = [imgs[i] for i in np.linspace(0, len(imgs) - 1, a.n_calib).round().astype(int)]
    excluir = [n.name for n in onnx.load(str(origen)).graph.node if n.name.startswith("/model.23/")] if a.excluir_cabeza else []
    if a.excluir_cabeza:
        assert excluir, "no se encontraron nodos /model.23/*: revisar los nombres del ONNX"
    int8 = carpeta / ("best_int8_sincabeza.onnx" if a.excluir_cabeza else "best_int8.onnx")

    t0 = time.perf_counter()
    quantize_static(str(origen), str(int8), Lector(rutas, entrada, a.imgsz), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8, nodes_to_exclude=excluir,
                    extra_options={"CalibStridedMinMax": TRAMO_CALIB})
    seg = time.perf_counter() - t0

    # el modelo cuantizado debe cargar y dar salidas finitas con una imagen real
    sesion = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    salidas = sesion.run(None, {entrada: preparar(cv2.imread(str(rutas[0])), a.imgsz)})
    assert all(np.isfinite(s).all() for s in salidas), "el INT8 devuelve valores no finitos"
    info = {
        "corrida": a.nombre, "forma_alto_ancho": [a.imgsz, a.imgsz], "opset": 17, "n_calibracion": len(rutas),
        "calibracion": f"MinMax, imágenes de train repartidas por igual, por tramos de {TRAMO_CALIB} (mismo resultado que todas a la vez)",
        "formato": "QDQ, por canal, pesos QInt8, activaciones QUInt8", "excluye_cabeza": a.excluir_cabeza, "nodos_excluidos": len(excluir),
        "preproceso_ort": preproceso, "segundos_cuantizacion": round(seg, 1), "onnx": onnx.__version__, "onnxruntime": ort.__version__,
        "tamano_mb": {"pt": round(pesos.stat().st_size / 2**20, 2), "onnx_fp32": round(fp32.stat().st_size / 2**20, 2), "onnx_int8": round(int8.stat().st_size / 2**20, 2)},
        "archivos": {"fp32": fp32.name, "int8": int8.name},
        "nota_tamano": "El .pt está en FP16 (formato de checkpoint de Ultralytics): la comparación de la optimización es onnx_fp32 frente a onnx_int8.",
    }
    info["reduccion_int8_vs_fp32"] = round(1 - info["tamano_mb"]["onnx_int8"] / info["tamano_mb"]["onnx_fp32"], 3)
    reporte.parent.mkdir(parents=True, exist_ok=True)
    reporte.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
