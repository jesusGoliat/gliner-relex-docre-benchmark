"""Métricas de TEST de B0 y Z0 (configuración congelada). Escribe RUNS/test/baseline_metrics.csv y los conteos por documento (npz) para el bootstrap.
Métricas: e2e estricto con tipo (principal), estricto sin tipo, blando con tipo, Ign F1 e2e (train_revised completo), mención/entidad."""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data import load_red, RUNS
from relex_post import build_pred, rel_scores
from relex_io import LABEL_VARIANTS
from relex_router import mask_to_preds
from eval_e2e import re_e2e, per_doc_counts_e2e, ign_e2e, entity_metrics
from runs import load_run

os.makedirs(f'{RUNS}/test', exist_ok=True)
train = load_red('train')


def metrics(P, docs, preds):
    a = re_e2e(P, docs, preds, 'strict', True); em = entity_metrics(P, docs)
    return dict(f1_typed=a['f1'], p_typed=a['p'], r_typed=a['r'], n_pred=a['n_pred'], f1_untyped=re_e2e(P, docs, preds, 'strict', False)['f1'],
                f1_soft_typed=re_e2e(P, docs, preds, 'soft', True)['f1'], ign_f1_typed=ign_e2e(P, docs, preds, train)['ign_f1'],
                mention_typed_f1=em['mention_typed_f1'], cluster_strict_f1=em['cluster_strict_f1'], type_acc=em['type_acc'])


if __name__ == '__main__':
    rows, counts = [], {}
    for n in (10, 50):
        for k in range(5):
            docs, P, S, best = load_run(n, k, 'test'); rt = best['rel_thr']
            preds = [p for d, s in zip(docs, S) for p in mask_to_preds(d['title'], s >= rt)]
            m = metrics(P, docs, preds); rows.append(dict(system='B0', n=n, sub=k, rt=rt, ent_thr=best['ent_thr'], dev_f1=best['f1_typed'], **m))
            counts[f'B0_N{n}_s{k}'] = per_doc_counts_e2e(P, docs, preds, 'strict', True)
            print('B0', n, k, round(m['f1_typed'], 2), round(m['ign_f1_typed'], 2), flush=True)
    R = json.load(open(f'{RUNS}/raw/zs_test_Z0.json')); run = R['runs'][0]; docs = load_red('test')
    P, sc = [], []
    for d, e, r in zip(docs, run['ents'], run['rels']):
        pd, w = build_pred(d, e, 0.3, True, LABEL_VARIANTS['v3'][1]); P.append(pd); sc.append(rel_scores(d, w, r))
    preds = [p for d, s in zip(docs, sc) for p in [{'title': d['title'], 'h_idx': h, 't_idx': t, 'r': pid} for h, t, pid, v in s if v >= 0.35]]
    m = metrics(P, docs, preds); rows.append(dict(system='Z0', n=0, sub=0, rt=0.35, ent_thr=0.3, dev_f1=15.0, **m)); counts['Z0'] = per_doc_counts_e2e(P, docs, preds, 'strict', True)
    print('Z0', round(m['f1_typed'], 2), round(m['ign_f1_typed'], 2))
    with open(f'{RUNS}/test/baseline_metrics.csv', 'w') as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    np.savez(f'{RUNS}/test/baseline_counts.npz', **counts)
