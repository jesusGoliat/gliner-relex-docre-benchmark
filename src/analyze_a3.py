"""A3 en test: estudiantes SIN LLM en inferencia. B0 (N gold) vs control sin LLM (N gold + pseudo del extractor) vs A3 (N gold + pseudo con juez), N=10, 5 subconjuntos.
Escribe RUNS/test/A3_metrics.csv, A3_summary.csv, A3_bootstrap.csv. Bootstrap pareado por documento como analyze_test.py; ganancia además sobre
tripletas 'no vistas' (cuyo par de nombres no aparece en el gold del pool de ese subconjunto)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from data import load_red, load, RUNS, PREP
from relex_post import build_pred, rel_scores, score_tensor
from relex_io import LABEL2TYPE
from relex_router import mask_to_preds
from eval_test import metrics
from eval_e2e import per_doc_counts_e2e, _map_preds
from eval import train_facts
from analyze_test import boot

P_POOL = int(os.environ.get('POOL_P', '1000'))


def student(arm, k):
    R = json.load(open(f'{RUNS}/raw/a3B{arm}_test_N10_s{k}.json')); docs = load_red('test'); run = R['runs'][0]; rt = R['best']['rel_thr']
    P, S = [], []
    for d, e, r in zip(docs, run['ents'], run['rels']):
        pd_, w = build_pred(d, e, run['cfg']['threshold'], True, LABEL2TYPE); P.append(pd_); S.append(score_tensor(len(pd_['vertexSet']), rel_scores(d, w, r)))
    return docs, P, S, rt


def unseen_recall(docs, P, preds, pool_facts):
    """Recall (%) solo sobre tripletas gold 'no vistas' (ningún par de nombres h-t con esa relación en el gold del pool)."""
    keys, gl = _map_preds(P, docs, preds, 'strict', True)
    ok = {(t, g) for t, g in keys if g}
    # reconstruir qué tripletas gold se aciertan: recomputar con match
    from eval_e2e import match_entities
    pdoc = {d['title']: d for d in P}; gdoc = {d['title']: d for d in docs}; hit = tot = 0
    byt = {}
    for x in preds: byt.setdefault(x['title'], []).append(x)
    for t, gd in gdoc.items():
        mm = match_entities(pdoc[t], gd, 'strict', True); found = set()
        for x in byt.get(t, []):
            h, tt = mm.get(x['h_idx']), mm.get(x['t_idx'])
            if h is not None and tt is not None: found.add((h, x['r'], tt))
        for l in gd['labels']:
            seen = any((a['name'], b['name'], l['r']) in pool_facts for a in gd['vertexSet'][l['h']] for b in gd['vertexSet'][l['t']])
            if not seen:
                tot += 1; hit += (l['h'], l['r'], l['t']) in found
    return 100 * hit / max(tot, 1), tot


if __name__ == '__main__':
    rows, counts, un = [], {}, {}
    for arm in ('ctrl', 'judge'):
        for k in range(5):
            docs, P, S, rt = student(arm, k)
            preds = [p for d, s in zip(docs, S) for p in mask_to_preds(d['title'], s >= rt)]
            m = metrics(P, docs, preds)
            pool = load(os.path.join(PREP, 'redocred', f'pool_N10_s{k}_P{P_POOL}.json')); pf = train_facts(pool)
            ur, nt = unseen_recall(docs, P, preds, pf)
            rows.append(dict(system=f'A3_{arm}', sub=k, rt=rt, recall_unseen=ur, n_unseen=nt, **m)); counts[(arm, k)] = per_doc_counts_e2e(P, docs, preds, 'strict', True)
            print(arm, k, round(m['f1_typed'], 2), round(ur, 2), flush=True)
    base = pd.read_csv(f'{RUNS}/test/baseline_metrics.csv'); b = base[(base.system == 'B0') & (base.n == 10)].sort_values('sub')
    bc = np.load(f'{RUNS}/test/baseline_counts.npz')
    cb = [bc[f'B0_N10_s{k}'] for k in range(5)]
    for k in range(5):
        r = b[b['sub'] == k].iloc[0].to_dict(); r.update(system='B0', sub=k); rows.append({kk: r.get(kk) for kk in rows[0]})
    df = pd.DataFrame(rows); df.to_csv(f'{RUNS}/test/A3_metrics.csv', index=False)
    summ = df.groupby('system')[['f1_typed', 'ign_f1_typed', 'p_typed', 'r_typed', 'f1_untyped', 'f1_soft_typed', 'recall_unseen']].agg(['mean', 'std']).round(2)
    summ.to_csv(f'{RUNS}/test/A3_summary.csv'); print(summ.to_string())
    cc = [counts[('ctrl', k)] for k in range(5)]; cj = [counts[('judge', k)] for k in range(5)]
    brow = [dict(cmp='A3(judge) - B0', **boot(cj, cb)), dict(cmp='A3(judge) - control(sin LLM)', **boot(cj, cc)), dict(cmp='control(sin LLM) - B0', **boot(cc, cb))]
    pd.DataFrame(brow).to_csv(f'{RUNS}/test/A3_bootstrap.csv', index=False); print(pd.DataFrame(brow).round(4).to_string())
