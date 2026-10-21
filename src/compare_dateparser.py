"""Compara las métricas de test recalculadas con `dateparser` instalado (RUNS/test) con las oficiales (results/test).
Entradas: results/test/{baseline_metrics.csv, J0_rows.csv, A1_A2_summary.csv, A1_A2_bootstrap.csv} y los mismos ficheros en RUNS/test.
Salida: RUNS/test/dateparser_comparison.csv (una fila por sistema/N/subconjunto/métrica) y un resumen impreso."""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import C1, RUNS

OFF = os.path.join(C1, 'results', 'test'); NEW = os.path.join(RUNS, 'test')


def rows(path):
    return list(csv.DictReader(open(path)))


out = []
for fn, key, cols in (('baseline_metrics.csv', ('system', 'n', 'sub'), ('f1_typed', 'ign_f1_typed', 'p_typed', 'r_typed', 'f1_untyped', 'f1_soft_typed', 'n_pred')),
                      ('J0_rows.csv', ('n', 'sub'), ('judge_f1_typed', 'judge_ign_f1_typed', 'judge_p_typed', 'judge_r_typed', 'base_f1', 'valid_rate', 'calls'))):
    a = {tuple(r[k] for k in key): r for r in rows(os.path.join(OFF, fn))}
    b = {tuple(r[k] for k in key): r for r in rows(os.path.join(NEW, fn))}
    for k in sorted(a):
        for c in cols:
            x, y = float(a[k][c]), float(b[k][c])
            out.append(dict(file=fn, key='/'.join(k), metric=c, official=x, recomputed=y, diff=y - x))
with open(os.path.join(NEW, 'dateparser_comparison.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, out[0].keys()); w.writeheader(); w.writerows(out)
for fn in ('A1_A2_summary.csv', 'A1_A2_bootstrap.csv'):
    print('==', fn)
    for r1, r2 in zip(rows(os.path.join(OFF, fn)), rows(os.path.join(NEW, fn))):
        print({k: (round(float(v), 3) if k not in ('metric',) else v) for k, v in r1.items()}, '->', {k: (round(float(v), 3) if k not in ('metric',) else v) for k, v in r2.items()})
for m in ('f1_typed', 'judge_f1_typed'):
    d = [r['diff'] for r in out if r['metric'] == m]
    print(f'{m}: máx |Δ| = {max(abs(v) for v in d):.4f}  media Δ = {sum(d) / len(d):+.4f}  (n={len(d)})')
