"""Evaluador de C1: reimplementación fiel de `official_evaluate` de Re-DocRED/evaluation.py
(F1, P, R, Ign F1) sin depender de train_distant.json ni meta/rel2id.json.

Predicciones: lista de {'title', 'h_idx', 't_idx', 'r'} (r = P-id), como en el oficial.
Ign F1: descuenta los hechos (nombre_mención_h, nombre_mención_t, r) vistos en los documentos de
entrenamiento que se pasen (en poco recurso, los N documentos; también se puede pasar el train completo).
"""


def train_facts(train_docs):
    facts = set()
    for d in train_docs:
        vs = d['vertexSet']
        for l in d['labels']:
            for n1 in vs[l['h']]:
                for n2 in vs[l['t']]:
                    facts.add((n1['name'], n2['name'], l['r']))
    return facts


def dedup(preds):
    seen, out = set(), []
    for x in preds:
        k = (x['title'], x['h_idx'], x['t_idx'], x['r'])
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def evaluate(preds, gold_docs, train_docs=()):
    """Devuelve dict con f1, p, r, ign_f1 (en %), n_pred, n_gold, n_correct."""
    std, t2vs = set(), {}
    for d in gold_docs:
        t2vs[d['title']] = d['vertexSet']
        for l in d['labels']:
            std.add((d['title'], l['r'], l['h'], l['t']))
    facts = train_facts(train_docs)
    sub = dedup(preds)
    n_gold = len(std)
    if not sub:
        return dict(f1=0.0, p=0.0, r=0.0, ign_f1=0.0, n_pred=0, n_gold=n_gold, n_correct=0)
    correct = correct_in_train = 0
    for x in sub:
        title, h, t, r = x['title'], x['h_idx'], x['t_idx'], x['r']
        if title not in t2vs:
            continue
        if (title, r, h, t) in std:
            correct += 1
            vs = t2vs[title]
            if any((n1['name'], n2['name'], r) in facts for n1 in vs[h] for n2 in vs[t]):
                correct_in_train += 1
    p = correct / len(sub)
    rec = correct / n_gold
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    p_ign = (correct - correct_in_train) / (len(sub) - correct_in_train + 1e-5)
    ign = 2 * p_ign * rec / (p_ign + rec) if p_ign + rec else 0.0
    return dict(f1=100 * f1, p=100 * p, r=100 * rec, ign_f1=100 * ign,
                n_pred=len(sub), n_gold=n_gold, n_correct=correct)


def gold_as_preds(gold_docs):
    return [{'title': d['title'], 'h_idx': l['h'], 't_idx': l['t'], 'r': l['r']}
            for d in gold_docs for l in d['labels']]
