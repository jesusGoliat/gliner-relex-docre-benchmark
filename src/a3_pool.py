"""A3: pool de documentos sin etiquetar para el autoentrenamiento (N=10). Selección independiente del modelo, con semilla 500+k.
Excluye train_N10_s{k}, y todo dev/test por TÍTULO; mide el solapamiento de TRIPLETAS FACTUALES (nombre_h, nombre_t, r) del pool (con su gold,
solo para análisis) con dev y test. Escribe data/subsets/redocred/pool_N10_s{k}_P{P}.json (no versionado) y RUNS/pseudo_labels/pool_overlap.csv.
Uso: python src/a3_pool.py --P 500 [--subs 0 1 2 3 4]"""
import os, sys, json, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data import load_red, load, PREP, RUNS
from eval import train_facts


def overlap(pool, ref):
    """Fracción de tripletas gold de `ref` cuyo par de nombres (con la misma relación) aparece en las tripletas gold de `pool`."""
    facts = train_facts(pool); tot = hit = 0
    for d in ref:
        for l in d['labels']:
            tot += 1
            hit += any((a['name'], b['name'], l['r']) in facts for a in d['vertexSet'][l['h']] for b in d['vertexSet'][l['t']])
    return hit, tot


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--P', type=int, default=500); ap.add_argument('--subs', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    tr, dev, test = load_red('train'), load_red('dev'), load_red('test')
    ban_ev = {d['title'] for d in dev + test}
    rows = []
    for k in a.subs:
        n10 = load(os.path.join(PREP, 'redocred', f'train_N10_s{k}.json'))
        ban = ban_ev | {d['title'] for d in n10}
        avail = [i for i, d in enumerate(tr) if d['title'] not in ban]
        sel = sorted(np.random.default_rng(500 + k).choice(avail, a.P, replace=False).tolist())
        pool = [tr[i] for i in sel]
        assert len({d['title'] for d in pool}) == len(pool) and not ({d['title'] for d in pool} & ban)
        json.dump(pool, open(os.path.join(PREP, 'redocred', f'pool_N10_s{k}_P{a.P}.json'), 'w'))
        for name, ref in (('dev', dev), ('test', test)):
            h, t = overlap(pool, ref)
            rows.append(dict(sub=k, P=a.P, ref=name, overlap_triples=h, ref_triples=t, frac=h / t))
        print(f's{k}: pool {len(pool)} de {len(avail)} disponibles; solapamiento dev {rows[-2]["frac"]:.3f} test {rows[-1]["frac"]:.3f}')
    os.makedirs(os.path.join(RUNS, 'pseudo_labels'), exist_ok=True)
    with open(os.path.join(RUNS, 'pseudo_labels', 'pool_overlap.csv'), 'w') as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
