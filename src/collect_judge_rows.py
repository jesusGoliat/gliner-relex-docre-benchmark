"""Convierte las líneas JSON de run_judge.py (RUNS/.../*.jsonl) en un CSV con la unión de columnas (en orden de aparición).
Uso: python src/collect_judge_rows.py ENTRADA.jsonl SALIDA.csv [--tag TAG]"""
import sys, json, csv

src, dst = sys.argv[1], sys.argv[2]
tag = sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else None
rows = [json.loads(l) for l in open(src) if l.strip()]
if tag is not None:
    rows = [r for r in rows if r.get('tag') == tag]
cols = []
for r in rows:
    for k in r:
        if k not in cols: cols.append(k)
with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} filas -> {dst}')
