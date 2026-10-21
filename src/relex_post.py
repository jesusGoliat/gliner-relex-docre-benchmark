"""Salida cruda de GLiNER-Relex (offsets de caracteres) -> documento predicho (entidades por clustering) y puntuaciones
S[h,t,r] entre entidades (máximo de la puntuación de relación sobre los pares de menciones). Sin dependencias de gliner."""
import numpy as np
from relex_io import LABEL2TYPE, NAME2PID, char_offsets, span_tokens
from ent_cluster import cluster
from ent_build import build_doc
from data import PIDS, REL2ID


def mentions_from_raw(doc, ents, ent_thr, flat=True, label2type=LABEL2TYPE):
    """-> (menciones, char_span -> índice de mención). Une la misma span duplicada (mayor score); si flat, quita solapes
    dentro de la oración (gana mayor score, luego más largo), ."""
    offs = char_offsets(doc)
    cand = {}
    for e in ents:
        if e['score'] < ent_thr or e['label'] not in label2type:
            continue
        sp = span_tokens(offs, e['start'], e['end'])
        if sp is None:
            continue
        old = cand.get(sp)
        if old is None or e['score'] > old[0]:
            cand[sp] = (float(e['score']), label2type[e['label']])
    ms = [{'sent_id': s, 'pos': [i, j], 'name': ' '.join(doc['sents'][s][i:j]), 'type': t, 'score': sc, 'src': 'relex',
           'tprobs': {t: 1.0}} for (s, i, j), (sc, t) in cand.items()]
    if flat:
        ms.sort(key=lambda m: (-m['score'], -(m['pos'][1] - m['pos'][0]), m['sent_id'], m['pos'][0]))
        keep, by = [], {}
        for m in ms:
            i, j = m['pos']
            if any(i < b and a < j for a, b in by.get(m['sent_id'], [])):
                continue
            by.setdefault(m['sent_id'], []).append((i, j)); keep.append(m)
        ms = keep
    ms.sort(key=lambda m: (m['sent_id'], m['pos'][0], m['pos'][1]))
    return ms


def build_pred(doc, ents, ent_thr, flat=True, label2type=LABEL2TYPE):
    """-> (doc predicho sin labels, {(sent_id,i,j): índice de entidad})."""
    ms = mentions_from_raw(doc, ents, ent_thr, flat, label2type)
    for k, m in enumerate(ms):
        m['mid'] = k
    es = cluster(ms, doc['sents'])
    pred = build_doc(doc, es)
    where = {}
    for ei, e in enumerate(es):
        for m in e['mentions']:
            where[(m['sent_id'], m['pos'][0], m['pos'][1])] = ei
    return pred, where


def rel_scores(doc, where, rels):
    """Lista de (h, t, pid, score) por par de entidades (máximo sobre menciones y duplicados)."""
    offs = char_offsets(doc)
    best = {}
    for r in rels:
        pid = NAME2PID.get(r['relation'])
        a = span_tokens(offs, r['head']['start'], r['head']['end'])
        b = span_tokens(offs, r['tail']['start'], r['tail']['end'])
        if pid is None or a is None or b is None or a not in where or b not in where:
            continue
        h, t = where[a], where[b]
        if h == t:
            continue
        k = (h, t, pid)
        best[k] = max(best.get(k, 0.0), float(r['score']))
    return [(h, t, p, s) for (h, t, p), s in best.items()]


def to_preds(title, scored, rel_thr):
    return [{'title': title, 'h_idx': h, 't_idx': t, 'r': p} for h, t, p, s in scored if s >= rel_thr]


def score_tensor(n_ent, scored):
    S = np.zeros((n_ent, n_ent, len(PIDS)), dtype=np.float32)
    for h, t, p, s in scored:
        S[h, t, REL2ID[p] - 1] = s
    return S
