"""Verificación de métricas: eval.evaluate debe dar EXACTAMENTE lo mismo que official_evaluate de Re-DocRED/evaluation.py.
Ign F1 estándar = re_f1_ignore_train_annotated con train_file = train_revised.json (train completo, 3 053 docs).
(El oficial también calcula Ign contra train_distant.json, que no está en el repo: se pasa una lista vacía y se ignora esa salida.)
Uso: python tests/test_eval_official.py"""
import os, sys, json, shutil, tempfile, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from data import EXT, REL2ID, PIDS, load_red
from eval import evaluate, gold_as_preds

TOL = 1e-9


def official(preds, dev_docs, train_docs):
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, 'meta'))
    json.dump(REL2ID, open(os.path.join(tmp, 'meta', 'rel2id.json'), 'w'))
    json.dump(train_docs, open(os.path.join(tmp, 'train_revised.json'), 'w'))
    json.dump([], open(os.path.join(tmp, 'train_distant.json'), 'w'))
    json.dump(dev_docs, open(os.path.join(tmp, 'dev_revised.json'), 'w'))
    cwd = os.getcwd()
    os.chdir(tmp)                     # el oficial lee meta/rel2id.json al importarse
    sys.path.insert(0, os.path.join(EXT, 'Re-DocRED'))
    try:
        sys.modules.pop('evaluation', None)
        import evaluation as ev
        f1, _, ign_ann, _, p, r = ev.official_evaluate([dict(x) for x in preds], tmp, 'train_revised.json', 'dev_revised.json')
    finally:
        os.chdir(cwd); sys.path.pop(0); shutil.rmtree(tmp)
    return dict(f1=100 * f1, p=100 * p, r=100 * r, ign_f1=100 * ign_ann)


def compare(name, preds, dev, train):
    a, b = evaluate(preds, dev, train), official(preds, dev, train)
    worst = max(abs(a[k] - b[k]) for k in ('f1', 'p', 'r', 'ign_f1'))
    print(f'{name:28s} nuestro F1={a["f1"]:.6f} Ign={a["ign_f1"]:.6f} | oficial F1={b["f1"]:.6f} Ign={b["ign_f1"]:.6f} | max|d|={worst:.2e}')
    assert worst < TOL, (name, a, b)


if __name__ == '__main__':
    dev = load_red('dev'); train = load_red('train')
    gold = gold_as_preds(dev)
    rng = random.Random(0)
    compare('gold', gold, dev, train)
    compare('gold, train vacío (sin Ign)', gold, dev, [])
    compare('50% gold', [x for x in gold if rng.random() < .5], dev, train)
    noisy = [dict(x, r=rng.choice(PIDS)) if rng.random() < .3 else x for x in gold]
    compare('gold con 30% de ruido en r', noisy, dev, train)
    allp = [{'title': d['title'], 'h_idx': h, 't_idx': t, 'r': rng.choice(PIDS)}
            for d in dev for h in range(len(d['vertexSet'])) for t in range(len(d['vertexSet'])) if h != t and rng.random() < .02]
    compare('aleatorio', allp + gold[:300], dev, train)
    compare('duplicados', gold[:500] + gold[:500], dev, train)
    compare('vacío', [], dev, train)
    print('OK: eval.evaluate == official_evaluate en todos los casos')
