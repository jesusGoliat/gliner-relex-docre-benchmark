"""Selección con D=50 de dev: barrido (entidad thr x relación thr) sobre salidas crudas de Relex. Código único para local y Modal."""
import numpy as np
from relex_post import build_pred, rel_scores, to_preds
from eval_e2e import re_e2e, entity_metrics

REL_GRID = [round(x, 2) for x in np.arange(0.10, 0.951, 0.05)]


def d_indices(n_dev=500, d=50, seed=1234):
    return sorted(np.random.default_rng(seed).permutation(n_dev)[:d].tolist())


def sweep(docs, runs, label2type, tag=None, rel_grid=REL_GRID):
    """runs: [{'cfg': {...,'threshold': te}, 'ents': [...], 'rels': [...]}] sobre `docs`. -> filas con métricas e2e (estricto con tipo = principal)."""
    rows = []
    for run in runs:
        te = run['cfg']['threshold']
        pds, sc = [], []
        for d, e, r in zip(docs, run['ents'], run['rels']):
            pd, w = build_pred(d, e, te, True, label2type)
            pds.append(pd); sc.append(rel_scores(d, w, r))
        em = entity_metrics(pds, docs)
        for rt in rel_grid:
            pr = [p for d, s in zip(docs, sc) for p in to_preds(d['title'], s, rt)]
            a = re_e2e(pds, docs, pr, 'strict', True)
            rows.append(dict(tag=tag, ent_thr=te, rel_thr=rt, f1_typed=a['f1'], p_typed=a['p'], r_typed=a['r'], n_pred=a['n_pred'],
                             f1_soft_typed=re_e2e(pds, docs, pr, 'soft', True)['f1'], mention_typed_f1=em['mention_typed_f1'],
                             cluster_strict_f1=em['cluster_strict_f1'], type_acc=em['type_acc']))
    return rows


def best(rows):
    """argmax de f1_typed; empate -> menor tag (checkpoint más temprano), luego umbrales más bajos."""
    return max(rows, key=lambda r: (round(r['f1_typed'], 9), -(r['tag'] or 0), -r['ent_thr'], -r['rel_thr']))
