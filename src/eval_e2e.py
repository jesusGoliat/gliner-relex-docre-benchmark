"""Evaluación end-to-end con entidades predichas.
Documentos predichos y gold alineados por posición (mismo title)."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from eval import dedup


def _ment_sets(doc):
    return [frozenset((m['sent_id'], m['pos'][0], m['pos'][1]) for m in e) for e in doc['vertexSet']]


def match_entities(pred_doc, gold_doc, mode='strict', typed=False):
    """dict pred_idx -> gold_idx (1 a 1). strict: mismo conjunto de menciones; soft: Jaccard>=0.5 + húngaro.
    typed=True: además exige el mismo tipo (tipo principal de la entidad predicha == tipo gold)."""
    P, G = _ment_sets(pred_doc), _ment_sets(gold_doc)
    pt = [_ent_type(pred_doc, i) for i in range(len(P))]
    gt = [_ent_type(gold_doc, j) for j in range(len(G))]
    if mode == 'strict':
        gi = {}
        for j, g in enumerate(G):
            gi.setdefault(g, []).append(j)   # entidades gold duplicadas: se emparejan en orden
        out = {}
        for i, p in enumerate(P):
            cand = [j for j in gi.get(p, []) if not typed or gt[j] == pt[i]]
            if cand:
                out[i] = cand[0]; gi[p].remove(cand[0])
        return out
    if not P or not G:
        return {}
    J = np.zeros((len(P), len(G)))
    for i, p in enumerate(P):
        for j, g in enumerate(G):
            u = len(p | g)
            J[i, j] = len(p & g) / u if u and (not typed or pt[i] == gt[j]) else 0.0
    r, c = linear_sum_assignment(-J)
    return {int(i): int(j) for i, j in zip(r, c) if J[i, j] >= 0.5}


def _prf(tp, npred, ngold):
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    return 100 * p, 100 * r, (100 * 2 * p * r / (p + r) if p + r else 0.0)


def _ent_type(doc, i):
    t2 = doc.get('types2')
    return t2[i][0] if t2 else doc['vertexSet'][i][0]['type']


def entity_metrics(pred_docs, gold_docs):
    """F1 (%) de menciones (strict, con y sin tipo), de clusters (strict, soft) y exactitud de tipo."""
    mp = mg = mtp = 0; tp_t = 0; np_t = ng_t = 0
    cl = {m: [0, 0, 0] for m in ('strict', 'soft')}   # tp, npred, ngold
    acc_n = acc_ok = 0
    for pd, gd in zip(pred_docs, gold_docs):
        assert pd['title'] == gd['title'], (pd['title'], gd['title'])
        ps = {(m['sent_id'], m['pos'][0], m['pos'][1]) for e in pd['vertexSet'] for m in e}
        gs = {(m['sent_id'], m['pos'][0], m['pos'][1]) for e in gd['vertexSet'] for m in e}
        mp += len(ps); mg += len(gs); mtp += len(ps & gs)
        pt = {(m['sent_id'], m['pos'][0], m['pos'][1], m['type']) for e in pd['vertexSet'] for m in e}
        gt = {(m['sent_id'], m['pos'][0], m['pos'][1], m['type']) for e in gd['vertexSet'] for m in e}
        np_t += len(pt); ng_t += len(gt); tp_t += len(pt & gt)
        for mode in cl:
            mt = match_entities(pd, gd, mode)
            cl[mode][0] += len(mt); cl[mode][1] += len(pd['vertexSet']); cl[mode][2] += len(gd['vertexSet'])
            if mode == 'strict':
                for i, j in mt.items():
                    acc_n += 1; acc_ok += _ent_type(pd, i) == gd['vertexSet'][j][0]['type']
    out = {}
    out['mention_p'], out['mention_r'], out['mention_f1'] = _prf(mtp, mp, mg)
    out['mention_typed_p'], out['mention_typed_r'], out['mention_typed_f1'] = _prf(tp_t, np_t, ng_t)
    for mode, (tp, a, b) in cl.items():
        out[f'cluster_{mode}_p'], out[f'cluster_{mode}_r'], out[f'cluster_{mode}_f1'] = _prf(tp, a, b)
    out['type_acc'] = 100 * acc_ok / acc_n if acc_n else 0.0
    return out


def _map_preds(pred_docs, gold_docs, preds, mode, typed=False):
    """-> (lista dedup de claves (title, ...), conjunto de claves correctas, n_gold por doc, claves por doc)."""
    gl = {gd['title']: {(l['h'], l['r'], l['t']) for l in gd['labels']} for gd in gold_docs}
    pdoc = {d['title']: d for d in pred_docs}
    gdoc = {d['title']: d for d in gold_docs}
    mm = {}
    seen, keys, ok = set(), [], []
    for x in dedup(preds):
        t = x['title']
        if t not in gdoc:
            continue
        if t not in mm:
            mm[t] = match_entities(pdoc[t], gdoc[t], mode, typed)
        h, tt = mm[t].get(x['h_idx']), mm[t].get(x['t_idx'])
        good = h is not None and tt is not None and (h, x['r'], tt) in gl[t]
        k = (t, h, x['r'], tt) if good else (t, 'p', x['h_idx'], x['t_idx'], x['r'])   # dedup tras emparejar
        if k in seen:
            continue
        seen.add(k); keys.append((t, good))
    return keys, gl


def re_e2e(pred_docs, gold_docs, preds, mode='strict', typed=False):
    """P/R/F1 (%) de RE end-to-end. Denominador = todas las tripletas gold."""
    keys, gl = _map_preds(pred_docs, gold_docs, preds, mode, typed)
    n_gold = sum(len(v) for v in gl.values())
    n_pred = len(keys); n_ok = sum(g for _, g in keys)
    p, r, f = _prf(n_ok, n_pred, n_gold)
    return dict(f1=f, p=p, r=r, n_pred=n_pred, n_correct=n_ok, n_gold=n_gold)


def per_doc_counts_e2e(pred_docs, gold_docs, preds, mode='strict', typed=False):
    """(n_pred, n_correct, n_gold) por doc en el orden de gold_docs; compatible con eval.paired_bootstrap."""
    keys, gl = _map_preds(pred_docs, gold_docs, preds, mode, typed)
    pos = {d['title']: i for i, d in enumerate(gold_docs)}
    c = np.zeros((len(gold_docs), 3))
    for d in gold_docs:
        c[pos[d['title']], 2] = len(gl[d['title']])
    for t, g in keys:
        c[pos[t], 0] += 1; c[pos[t], 1] += g
    return c


def ign_e2e(pred_docs, gold_docs, preds, train_docs, mode='strict', typed=True):
    """Ign F1 (%) de extremo a extremo, análogo al oficial: una tripleta correcta (entidades emparejadas con gold y (h,r,t) en gold) se
    descuenta si algún par de nombres de mención (h_gold, t_gold) con esa relación aparece en el train COMPLETO (`train_docs`).
    Precisión sin esos aciertos: (ok - ok_en_train) / (n_pred - ok_en_train + 1e-5); recall = ok / n_gold. Con entidades gold coincide con
    eval.evaluate(...)['ign_f1'] (test_eval_e2e_ign.py)."""
    from eval import train_facts
    facts = train_facts(train_docs)
    gl = {gd['title']: {(l['h'], l['r'], l['t']) for l in gd['labels']} for gd in gold_docs}
    pdoc = {d['title']: d for d in pred_docs}; gdoc = {d['title']: d for d in gold_docs}
    mm, seen = {}, set()
    n_pred = n_ok = n_in_train = 0
    for x in dedup(preds):
        t = x['title']
        if t not in gdoc:
            continue
        if t not in mm:
            mm[t] = match_entities(pdoc[t], gdoc[t], mode, typed)
        h, tt = mm[t].get(x['h_idx']), mm[t].get(x['t_idx'])
        good = h is not None and tt is not None and (h, x['r'], tt) in gl[t]
        k = (t, h, x['r'], tt) if good else (t, 'p', x['h_idx'], x['t_idx'], x['r'])
        if k in seen:
            continue
        seen.add(k); n_pred += 1
        if good:
            n_ok += 1
            vs = gdoc[t]['vertexSet']
            if any((a['name'], b['name'], x['r']) in facts for a in vs[h] for b in vs[tt]):
                n_in_train += 1
    n_gold = sum(len(v) for v in gl.values())
    rec = n_ok / n_gold if n_gold else 0.0
    p_ign = (n_ok - n_in_train) / (n_pred - n_in_train + 1e-5)
    ign = 2 * p_ign * rec / (p_ign + rec) if p_ign + rec else 0.0
    return dict(ign_f1=100 * ign, p_ign=100 * p_ign, r=100 * rec, n_in_train=n_in_train)
