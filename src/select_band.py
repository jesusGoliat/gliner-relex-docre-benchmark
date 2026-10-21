"""Elección de la banda [lo, hi) del juez por tamaño N con el juez real en D (dev, 50 documentos).
Regla: entre las bandas evaluadas en los subconjuntos s0 y s1, la de mayor F1 medio del juez. Las ejecuciones repetidas con la misma
(n, sub, lo, hi) comparten caché de llamadas; se conserva la primera.
Uso: python src/select_band.py results/selection/judge_band_selection_D50.csv"""
import sys
import pandas as pd

d = pd.read_csv(sys.argv[1]).drop_duplicates(['n', 'sub', 'lo', 'hi'])
g = d.groupby(['n', 'lo', 'hi']).agg(runs=('sub', 'count'), q_per_doc=('q_per_doc', 'mean'), base_f1=('base_f1', 'mean'), judge_f1=('judge_f1', 'mean'),
                                     delta=('delta', 'mean'), valid_rate=('valid_rate', 'mean'), trunc=('trunc', 'sum'), band=('band', 'sum'))
print(g.round(3).to_string())
for n in sorted(d['n'].unique()):
    x = g.loc[n]
    assert (x.runs == 2).all(), 'faltan ejecuciones (s0 y s1) para alguna banda'
    print(f'N={n}: banda elegida (lo, hi) = {x.judge_f1.idxmax()}  F1 juez medio {x.judge_f1.max():.2f}')
