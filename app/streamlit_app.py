"""Interfaz Streamlit: sube una foto (o usa la cámara) y muestra manzanas detectadas, conteo, manzanas aisladas y medidas.

Uso (con el entorno activo, desde la raíz):  streamlit run app/streamlit_app.py
Pesos: models/minneapple.pt, o la variable MINNEAPPLE_PESOS. Dispositivo: MINNEAPPLE_DEVICE (por defecto la GPU si existe).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analisis import CONF_POR_DEFECTO, RAIZ, Analizador, cargar_imagen  # noqa: E402

st.set_page_config(page_title="MinneApple", layout="wide")


@st.cache_resource(show_spinner="Cargando el modelo…")
def obtener_analizador() -> Analizador:
    a = Analizador()
    a.calentar()
    return a


def rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sobre_gris(bgra: np.ndarray) -> np.ndarray:
    """Recorte con alfa -> RGB sobre gris claro, para que se vea igual en tema claro y oscuro."""
    a = bgra[..., 3:4] / 255.0
    return (rgb(bgra[..., :3]) * a + 225 * (1 - a)).astype(np.uint8)


st.title("MinneApple: detección, segmentación y conteo de manzanas")
st.caption("Herramienta académica (Proyecto 2, Visión Computacional, UAO). Cuenta solo las manzanas **visibles en la foto**; no estima el rendimiento del huerto completo.")

try:
    analizador = obtener_analizador()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Ajustes")
    conf = st.slider("Confianza mínima", 0.05, 0.90, CONF_POR_DEFECTO, 0.05, key="conf",
                     help="Más alta: menos manzanas pero más seguras. Más baja: cuenta más, con más falsas alarmas.")
    escala = st.number_input("Escala: mm por píxel de la foto original (0 = sin escala)", min_value=0.0, value=0.0, step=0.05,
                             format="%.3f", key="escala")
    with st.expander("¿Cómo calculo la escala?"):
        st.write("Mide en la foto, en píxeles, un objeto del que conozcas el tamaño real. Escala = tamaño real en mm ÷ tamaño en píxeles "
                 "(ej.: una manzana de 70 mm que ocupa 140 px → 0,5 mm/px). Sin escala solo se muestran píxeles.")
    usar_cam = st.checkbox("Calcular Grad-CAM (explicación visual, algo más lento)", value=True, key="gradcam")
    st.caption(f"Modelo: `{analizador.pesos.name}` · dispositivo: `{analizador.device}`")

carpeta_ejemplos = RAIZ / "data" / "yolo" / "images" / "val"
ejemplos = sorted(p.name for p in carpeta_ejemplos.glob("*.png"))[:30] if carpeta_ejemplos.exists() else []
origen = st.radio("Origen de la imagen", ["Subir imagen", "Cámara"] + (["Ejemplo (validación)"] if ejemplos else []), horizontal=True, key="origen")

datos, nombre = None, None
if origen == "Subir imagen":
    f = st.file_uploader("Foto de un árbol de manzana", type=["png", "jpg", "jpeg"], key="subida")
    if f:
        datos, nombre = f.getvalue(), f.name
elif origen == "Cámara":
    f = st.camera_input("Toma una foto", key="camara")
    if f:
        datos, nombre = f.getvalue(), "cámara"
else:
    nombre = st.selectbox("Imagen de ejemplo", ejemplos, key="ejemplo")
    datos = (carpeta_ejemplos / nombre).read_bytes()

if datos is None:
    st.info("Sube una imagen, usa la cámara o elige un ejemplo para empezar.")
    st.stop()

try:
    r = analizador.analizar(cargar_imagen(datos), conf=conf, mm_por_px=escala or None, con_gradcam=usar_cam)
except Exception as e:  # noqa: BLE001 - la app no debe caerse con una imagen rara
    st.error(f"No se pudo analizar la imagen: {e}")
    st.stop()

inst = r["instancias"]
if r["n"] == 0:
    st.warning("No se detectaron manzanas con esta confianza. Prueba a bajarla o con otra foto.")
    st.stop()

tab_det, tab_aisl, tab_cam, tab_info = st.tabs(["Detección y conteo", "Manzanas aisladas", "Atención (Grad-CAM)", "Acerca de"])

with tab_det:
    c1, c2, c3, c4 = st.columns(4)
    d = np.array([i["diametro_px"] for i in inst])
    c1.metric("Manzanas detectadas", r["n"])
    c2.metric("Diámetro mediano", f"{np.median(d):.0f} px" + (f" ≈ {np.median([i['diametro_mm'] for i in inst]):.0f} mm" if r["con_escala"] else ""))
    c3.metric("Rojas / verde-amarillas", f"{sum(i['color'] == 'rojo' for i in inst)} / {sum(i['color'] != 'rojo' for i in inst)}")
    c4.metric("Tiempo de análisis", f"{r['ms']:.0f} ms")
    if r["con_escala"]:
        st.caption(f"Peso total aproximado de las manzanas visibles: **{sum(i['peso_g_aprox'] for i in inst) / 1000:.1f} kg** (estimación: esfera × 0,8 g/cm³; "
                   "las manzanas tapadas por hojas se miden más pequeñas de lo real).")
    st.image(rgb(r["anotada"]), caption=f"{nombre}: manzanas detectadas" + (f" (imagen reducida a {r['factor']:.0%})" if r["factor"] < 1 else ""))
    hist, bordes = np.histogram(d, bins=12)
    st.bar_chart({"manzanas": hist.tolist()}, x_label="diámetro (px, de menor a mayor)", y_label="manzanas")

with tab_aisl:
    st.image(rgb(r["aisladas"]), caption="Todas las manzanas aisladas del fondo (máscara de segmentación sobre la imagen original)")
    n_ver = st.slider("Manzanas a mostrar (las más grandes primero)", 1, min(60, r["n"]), min(12, r["n"]), key="n_ver")
    orden = sorted(inst, key=lambda i: -i["area_px"])[:n_ver]
    cols = st.columns(6)
    for k, i in enumerate(orden):
        etiqueta = f"{i['diametro_px']:.0f} px" + (f" · {i['diametro_mm']:.0f} mm · ~{i['peso_g_aprox']:.0f} g" if r["con_escala"] else "") + f" · {i['color']}"
        cols[k % 6].image(sobre_gris(i["recorte_bgra"]), caption=etiqueta)
    st.caption("El color es **descriptivo** (fracción de píxeles rojos dentro de la máscara); no es madurez ni se evaluó como tarea.")

with tab_cam:
    if not usar_cam:
        st.info("Activa «Calcular Grad-CAM» en la barra lateral para ver esta explicación.")
    else:
        dentro, area, cociente = r["enfoque"]
        st.image(rgb(r["gradcam"]), caption="Grad-CAM: zonas que más aportan a la confianza de las detecciones (más rojo = más aporte)")
        st.metric("Atención sobre las manzanas detectadas", f"{dentro:.0%}",
                  help="Fracción de la energía del mapa que cae dentro de las máscaras de las manzanas detectadas.")
        st.caption(f"Las manzanas detectadas ocupan el {area:.1%} de la imagen y reciben el {dentro:.0%} de la atención ({cociente:.0f} veces lo esperado al azar).")
        st.markdown("Grad-CAM (implementado a mano, sobre la capa 16 de la red: la que mejor concentra la atención sobre las manzanas, de 3 candidatas) muestra **dónde mira** el modelo, no por qué decide como lo hace. "
                    "Es una explicación visual de esta imagen y no prueba que el modelo entienda qué es una manzana.")

with tab_info:
    st.markdown(
        "- **Modelo:** YOLO11n-seg (detección y segmentación de instancias) con self-attention C2PSA, ajustado sobre MinneApple.\n"
        "- **Datos:** MinneApple (Häni, Roy, Isler, 2019; CC BY-NC-SA 3.0 US): un solo huerto de Minnesota, fotos de celular de 2015-2016. Uso académico, no comercial.\n"
        "- **Limitaciones:** no anota las manzanas del suelo ni de las hileras del fondo; las manzanas muy tapadas o menores de ~15 px se pierden; "
        "el diámetro y el peso son estimaciones y dependen de la escala que indiques; no hay etiquetas de madurez, así que no se clasifica madurez.\n"
        "- Detalle completo en la model card del repositorio (pendiente de publicar)."
    )
