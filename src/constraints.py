"""Restricciones de dominio/rango: tablas de Menotti et al. (DocRE-StructuralConsistency) y tablas
aprendidas de N documentos. allowed(doc) -> (E, E, 96) bool."""
import os, json
import numpy as np
from data import EXT, REL2ID

RULES = os.path.join(EXT, 'DocRE-StructuralConsistency', 'data', 'rules')
MENOTTI_DR = json.load(open(os.path.join(RULES, 'domain_range_rules.json')))
TYPES = ['PER', 'ORG', 'LOC', 'TIME', 'NUM', 'MISC']


def ent_types(doc):
    return [e[0]['type'] for e in doc['vertexSet']]


def table_menotti():
    t = np.zeros((len(TYPES), len(TYPES), 96), dtype=bool)
    for k, rels in MENOTTI_DR.items():
        a, b = k.split('+')
        for r in rels:
            t[TYPES.index(a), TYPES.index(b), REL2ID[r] - 1] = True
    return t


def table_from_docs(docs):
    t = np.zeros((len(TYPES), len(TYPES), 96), dtype=bool)
    for d in docs:
        ty = ent_types(d)
        for l in d['labels']:
            t[TYPES.index(ty[l['h']]), TYPES.index(ty[l['t']]), REL2ID[l['r']] - 1] = True
    return t


def allowed(doc, table):
    if 'types2' not in doc:
        ti = np.array([TYPES.index(x) for x in ent_types(doc)])
        return table[ti[:, None], ti[None, :]]
    # con types2: permitida si ALGÚN par de tipos (types2[h] x types2[t]) es válido
    E = len(doc['types2'])
    T = np.zeros((E, len(TYPES)), dtype=bool)
    for i, ts in enumerate(doc['types2']):
        for x in ts:
            T[i, TYPES.index(x)] = True
    out = np.zeros((E, E, table.shape[-1]), dtype=bool)
    for a in range(len(TYPES)):
        for b in range(len(TYPES)):
            out |= (T[:, a][:, None] & T[:, b][None, :])[:, :, None] & table[a, b]
    return out
