"""A3: pseudo-etiquetas del pool con el extractor ftB (N=10, subconjunto k).
  --arm ctrl : control SIN LLM: relaciones con S >= rt* (la decisión del propio extractor, mismo umbral elegido en D).
  --arm judge: sure (S >= hi) + opciones elegidas por el juez en la banda [lo, hi) (mismo juez, prompt y fusión que A1; respaldo si no parsea).
Entidades pseudo = las predichas por el extractor (clustering por reglas). Misma población de documentos en ambos brazos.
Salida: RUNS/pseudo_labels/{arm}_N10_s{k}_P{P}.json (ejemplos GLiNER) y RUNS/pseudo_labels/stats_{arm}.csv (precisión/recall de las pseudo-etiquetas vs el gold
del pool, SOLO análisis, nunca selección).
Uso: python src/a3_pseudo.py --arm ctrl --P 500 | --arm judge --lo 0.1 --hi 0.9 --P 500 [--subs 0 1]"""
import os, sys, json, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data import load, PREP, RUNS
from relex_post import build_pred, rel_scores, score_tensor
from relex_io import LABEL2TYPE, to_train_relex
from relex_router import band_split, fuse_hard, mask_to_preds
from eval_e2e import re_e2e

ap = argparse.ArgumentParser()
ap.add_argument('--arm', required=True, choices=['ctrl', 'judge']); ap.add_argument('--P', type=int, default=500)
ap.add_argument('--subs', type=int, nargs='+', default=[0, 1, 2, 3, 4]); ap.add_argument('--lo', type=float); ap.add_argument('--hi', type=float)
a = ap.parse_args()
sel = {(r['n'], r['sub']): r for r in json.load(open(f'{RUNS}/selection/ftB_baseline_dev_D50.json'))}
os.makedirs(f'{RUNS}/pseudo_labels', exist_ok=True)
if a.arm == 'judge':
    import llm_router as LR
    assert a.lo is not None and a.hi is not None
rows = []
for k in a.subs:
    best = sel[(10, k)]['best']; rt = best['rel_thr']
    R = json.load(open(f'{RUNS}/raw/pool_N10_s{k}_P{a.P}.json')); run = R['runs'][0]
    pool = load(os.path.join(PREP, 'redocred', f'pool_N10_s{k}_P{a.P}.json'))
    P, S = [], []
    for d, e, r in zip(pool, run['ents'], run['rels']):
        pd, w = build_pred(d, e, run['cfg']['threshold'], True, LABEL2TYPE); P.append(pd); S.append(score_tensor(len(pd['vertexSet']), rel_scores(d, w, r)))
    if a.arm == 'ctrl':
        masks = [s >= rt for s in S]; extra = {}
    else:
        train = load(os.path.join(PREP, 'redocred', f'train_N10_s{k}.json'))
        sure, routed = [], []
        for pd, s in zip(P, S):
            su, ro, _ = band_split(pd, s, a.lo, a.hi, 5); sure.append(su); routed.append(ro)
        dec = LR.run_many(P, routed, train)
        res = [fuse_hard(su, s, dc, rt) for su, s, dc in zip(sure, S, dec)]
        masks = [m for m, _ in res]; extra = dict(parse_fail=sum(nf for _, nf in res), questions=sum(len(r) for r in routed),
                                                  usd_total=LR.total_spend_usd(), calls=LR.SPEND['calls'], cached=LR.SPEND['cached'])
    ex = []
    for pd, m in zip(P, masks):
        h, t, c = np.nonzero(m)
        from data import PIDS
        d = {**pd, 'labels': [{'h': int(x), 't': int(y), 'r': PIDS[int(z)]} for x, y, z in zip(h, t, c)]}
        ex.append(to_train_relex(d))
    json.dump(ex, open(f'{RUNS}/pseudo_labels/{a.arm}_N10_s{k}_P{a.P}.json', 'w'))
    preds = [p for d, m in zip(pool, masks) for p in mask_to_preds(d['title'], m)]
    q = re_e2e(P, pool, preds, 'strict', True)
    row = dict(arm=a.arm, sub=k, P=a.P, lo=a.lo, hi=a.hi, rt=rt, n_pseudo_rels=int(sum(m.sum() for m in masks)), pseudo_p=q['p'], pseudo_r=q['r'], pseudo_f1=q['f1'],
               n_ex=len(ex), n_rel_examples=sum(len(e['relations']) for e in ex), **extra)
    rows.append(row); print(json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in row.items()}))
fn = f'{RUNS}/pseudo_labels/stats_{a.arm}.csv'; new = not os.path.exists(fn)
with open(fn, 'a') as f:
    w = csv.DictWriter(f, rows[0].keys())
    if new: w.writeheader()
    w.writerows(rows)
