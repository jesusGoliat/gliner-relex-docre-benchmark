"""Carga de salidas crudas de Relex (ajustado, ftB) -> documentos predichos y puntuaciones S[h,t,r]."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_red, RUNS
from relex_post import build_pred, rel_scores, score_tensor
from relex_io import LABEL2TYPE


def load_run(n, k, split='dev'):
    """Salida cruda de B0 (N=n, subconjunto k) en `split` -> (docs gold, docs predichos, S por documento, selección en D)."""
    R = json.load(open(f'{RUNS}/raw/ftB_{split}_N{n}_s{k}.json')); dev = load_red(split)
    docs = [dev[i] for i in R['idx']]; run = R['runs'][0]; te = run['cfg']['threshold']
    P, S = [], []
    for d, e, r in zip(docs, run['ents'], run['rels']):
        pd, w = build_pred(d, e, te, True, LABEL2TYPE); P.append(pd); S.append(score_tensor(len(pd['vertexSet']), rel_scores(d, w, r)))
    return docs, P, S, R['best']
