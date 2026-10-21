"""Análisis de TEST: B0 vs J0 (A1: N=10, A2: N=50). Por subconjunto (media ± desv. sobre 5) y bootstrap pareado POR DOCUMENTO
(10 000 remuestreos, semilla 0, mismos documentos remuestreados en los 5 subconjuntos; Δ = media sobre subconjuntos del F1 e2e estricto con tipo).
Entradas en RUNS/test/ (baseline_metrics.csv, J0_rows.csv, J0_counts_*.npz); escribe A1_A2_summary.csv y A1_A2_bootstrap.csv en RUNS/test/."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from data import RUNS


def f1v(c):   # c: (B,3) sumas -> F1 (%)
    p = c[:, 1] / np.maximum(c[:, 0], 1e-9); r = c[:, 1] / np.maximum(c[:, 2], 1e-9)
    return 100 * np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)


def boot(cj, cb, B=10000, seed=0):
    """cj, cb: listas (por subconjunto) de arrays (D,3). Δ medio J-B por remuestreo de documentos común a todos los subconjuntos."""
    D = cj[0].shape[0]; idx = np.random.default_rng(seed).integers(0, D, size=(B, D))
    dj = np.mean([f1v(c[idx].sum(1)) for c in cj], 0); db = np.mean([f1v(c[idx].sum(1)) for c in cb], 0)
    d = dj - db
    obs = np.mean([f1v(c.sum(0, keepdims=True))[0] for c in cj]) - np.mean([f1v(c.sum(0, keepdims=True))[0] for c in cb])
    return dict(delta=obs, ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)), p_one_sided=float((d <= 0).mean()))


if __name__ == '__main__':
    base = pd.read_csv(f'{RUNS}/test/baseline_metrics.csv'); jt = pd.read_csv(f'{RUNS}/test/J0_rows.csv')
    jt = jt.drop_duplicates(['n', 'sub'], keep='last')
    rows, brow = [], []
    for n in (10, 50):
        b = base[(base.system == 'B0') & (base.n == n)].sort_values('sub'); j = jt[jt.n == n].sort_values('sub')
        if len(j) < 5: print(f'N={n}: solo {len(j)}/5 subconjuntos con juez'); continue
        for name, col_b, col_j in (('F1 e2e estricto+tipo', 'f1_typed', 'judge_f1_typed'), ('Ign F1 e2e (train completo)', 'ign_f1_typed', 'judge_ign_f1_typed'),
                                   ('F1 estricto sin tipo', 'f1_untyped', 'judge_f1_untyped'), ('F1 blando+tipo', 'f1_soft_typed', 'judge_f1_soft_typed'),
                                   ('Precisión', 'p_typed', 'judge_p_typed'), ('Recall', 'r_typed', 'judge_r_typed')):
            rows.append(dict(N=n, metric=name, B0_mean=b[col_b].mean(), B0_sd=b[col_b].std(ddof=1), J0_mean=j[col_j].mean(), J0_sd=j[col_j].std(ddof=1),
                             delta_mean=(j[col_j].values - b[col_b].values).mean(), subsets_J0_better=int((j[col_j].values > b[col_b].values).sum())))
        cj = [np.load(f'{RUNS}/test/J0_counts_N{n}_s{k}.npz')['judge'] for k in range(5)]
        cb = [np.load(f'{RUNS}/test/J0_counts_N{n}_s{k}.npz')['base'] for k in range(5)]
        brow.append(dict(N=n, **boot(cj, cb)))
    pd.DataFrame(rows).to_csv(f'{RUNS}/test/A1_A2_summary.csv', index=False); pd.DataFrame(brow).to_csv(f'{RUNS}/test/A1_A2_bootstrap.csv', index=False)
    print(pd.DataFrame(rows).round(2).to_string()); print(pd.DataFrame(brow).round(4).to_string())
