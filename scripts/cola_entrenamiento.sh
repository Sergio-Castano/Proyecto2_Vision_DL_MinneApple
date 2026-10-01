#!/bin/bash
# Cola de las 6 corridas (ablación: con/sin C2PSA, 3 semillas). Cada una escribe su log en runs/; si una falla, sigue con la siguiente.
# Uso (con el entorno activo, desde la raíz): bash scripts/cola_entrenamiento.sh
# Si no tienes el entorno activado en esta terminal, define PYTHON=ruta/al/python.exe antes de llamar al script.
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python}"
for semilla in 0 1 2; do
  for variante in c2psa sin_c2psa; do
    nombre="${variante}_s${semilla}"
    if [ -f "runs/$nombre/weights/last.pt" ] && grep -q "40 epochs completed" "runs/$nombre.log" 2>/dev/null; then
      echo "$(date '+%F %H:%M') $nombre ya terminada, se omite"; continue
    fi
    echo "$(date '+%F %H:%M') inicio $nombre"
    "$PY" scripts/entrenar.py --variante "$variante" --semilla "$semilla" > "runs/$nombre.log" 2>&1
    echo "$(date '+%F %H:%M') fin $nombre (salida $?)"
  done
done
echo "$(date '+%F %H:%M') COLA TERMINADA"
