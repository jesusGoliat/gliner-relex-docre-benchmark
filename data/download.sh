#!/usr/bin/env bash
# Descarga los datos externos en data/external/ (no versionados) en las revisiones exactas usadas en la evaluación.
# Uso: bash data/download.sh   (desde la raíz del repositorio)
set -euo pipefail
mkdir -p data/external && cd data/external
clone_at() { [ -d "$2" ] || { git clone -q "$1" "$2" && git -C "$2" checkout -q "$3"; }; }
clone_at https://github.com/tonytan48/Re-DocRED.git Re-DocRED ccfb54f5ddf5836027c87badda10f6dfc56efaac                                  # train/dev/test revisados + evaluation.py oficial
clone_at https://github.com/mntlra/DocRE-StructuralConsistency.git DocRE-StructuralConsistency ee9de0f8cf9f2df3b061d72e13568d6361cc8dd0   # rel_info + tablas dominio/rango
echo "OK: $(ls)"
