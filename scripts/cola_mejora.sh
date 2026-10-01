#!/bin/bash
# Cola de los tiros de mejora T1 (small) y T2 (scale reducido). Uno tras otro; si uno falla, sigue con el otro.
# Uso (con el entorno activo, desde la raíz): bash scripts/cola_mejora.sh
# Si no tienes el entorno activado en esta terminal, define PYTHON=ruta/al/python.exe antes de llamar al script.
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python}"

lanzar() {
  nombre="$1"; shift
  if [ -f "runs/$nombre/weights/last.pt" ] && grep -q "40 epochs completed" "runs/$nombre.log" 2>/dev/null; then
    echo "$(date '+%F %H:%M') $nombre ya terminada, se omite"; return
  fi
  echo "$(date '+%F %H:%M') inicio $nombre"
  "$PY" scripts/entrenar.py --nombre "$nombre" "$@" > "runs/$nombre.log" 2>&1
  echo "$(date '+%F %H:%M') fin $nombre (salida $?)"
}

lanzar small_s0     --variante c2psa --escala s --batch 2 --semilla 0
lanzar scale025_s0  --variante c2psa --scale-aug 0.25 --semilla 0
echo "$(date '+%F %H:%M') COLA DE MEJORA TERMINADA"
