"""Barrido zero-shot (etiquetas v0) en D=50 de dev. Lee RUNS/raw/zs_dev_D50.json; escribe RUNS/selection/zs_dev_D50.csv.
Por configuración (entidad thr, adyacencia thr, flat) y umbral de relación: F1/P/R e2e estricto con tipo (principal), sin tipo y blando."""
import os, sys, json, csv, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import RUNS, load_red
from relex_post import build_pred, rel_scores, to_preds
from eval_e2e import re_e2e, entity_metrics

raw_path = os.path.join(RUNS, 'raw', 'zs_dev_D50.json')
R = json.load(open(raw_path)); dev = load_red('dev'); docs = [dev[i] for i in R['idx']]
print('sha256 raw', hashlib.sha256(open(raw_path, 'rb').read()).hexdigest()[:16], 'docs', len(docs))
rows = []
for run in R['runs']:
    c = run['cfg']; te = c['threshold']
    pdocs, scored = [], []
    for d, e, r in zip(docs, run['ents'], run['rels']):
        pd, where = build_pred(d, e, te, flat=True)       # aplanado de menciones solapadas (gana mayor score, luego más largo)
        pdocs.append(pd); scored.append(rel_scores(d, where, r))
    em = entity_metrics(pdocs, docs)
    for rt in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        preds = [p for d, sc in zip(docs, scored) for p in to_preds(d['title'], sc, rt)]
        a = re_e2e(pdocs, docs, preds, 'strict', True); b = re_e2e(pdocs, docs, preds, 'strict', False); s = re_e2e(pdocs, docs, preds, 'soft', True)
        rows.append(dict(ent_thr=te, adj_thr=c['adjacency_threshold'], flat_ner=c['flat_ner'], rel_thr=rt,
                         f1_typed=a['f1'], p_typed=a['p'], r_typed=a['r'], f1_untyped=b['f1'], f1_soft_typed=s['f1'],
                         n_pred=a['n_pred'], mention_typed_f1=em['mention_typed_f1'], cluster_strict_f1=em['cluster_strict_f1']))
os.makedirs(os.path.join(RUNS, 'selection'), exist_ok=True)
with open(os.path.join(RUNS, 'selection', 'zs_dev_D50.csv'), 'w') as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
rows.sort(key=lambda x: -x['f1_typed'])
for r in rows[:8]:
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
print('--- peor/mediana f1_typed:', round(rows[-1]['f1_typed'], 2), round(float(np.median([r['f1_typed'] for r in rows])), 2))
