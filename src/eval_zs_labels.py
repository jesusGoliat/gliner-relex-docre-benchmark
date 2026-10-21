"""Barrido zero-shot de las variantes de etiquetas de entidad v1 (descripciones) y v3 (23 descriptores finos) en D=50 de dev.
Lee RUNS/raw/zs_labels_D50.json; escribe RUNS/selection/zs_labels_D50.csv (F1 e2e estricto con tipo, sin tipo y tipo/mención por umbral)."""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_red, RUNS
from relex_post import build_pred, rel_scores, to_preds
from relex_io import LABEL_VARIANTS
from eval_e2e import entity_metrics, re_e2e

R = json.load(open(f'{RUNS}/raw/zs_labels_D50.json')); dev = load_red('dev'); docs = [dev[i] for i in R['idx']]
rows = []
for v, o in R['variants'].items():
    l2t = LABEL_VARIANTS[v][1]
    for run in o['runs']:
        te = run['cfg']['threshold']; pds = []; sc = []
        for d, e, r in zip(docs, run['ents'], run['rels']):
            pd, w = build_pred(d, e, te, True, l2t); pds.append(pd); sc.append(rel_scores(d, w, r))
        em = entity_metrics(pds, docs)
        for rt in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            pr = [p for d, s in zip(docs, sc) for p in to_preds(d['title'], s, rt)]
            a = re_e2e(pds, docs, pr, 'strict', True); b = re_e2e(pds, docs, pr, 'strict', False)
            rows.append(dict(variant=v, ent_thr=te, rel_thr=rt, f1_typed=a['f1'], p=a['p'], r=a['r'], f1_untyped=b['f1'], n_pred=a['n_pred'],
                             mention_typed_f1=em['mention_typed_f1'], type_acc=em['type_acc'], cluster_strict_f1=em['cluster_strict_f1']))
os.makedirs(f'{RUNS}/selection', exist_ok=True)
w = csv.DictWriter(open(f'{RUNS}/selection/zs_labels_D50.csv', 'w'), rows[0].keys()); w.writeheader(); w.writerows(rows)
