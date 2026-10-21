"""ign_e2e con entidades gold == eval.evaluate (que a su vez == oficial, ver test_eval_official.py). Uso: python tests/test_eval_e2e_ign.py"""
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from data import load_red, PIDS
from eval import evaluate, gold_as_preds
from eval_e2e import ign_e2e
dev, train = load_red('dev'), load_red('train'); rng = random.Random(3)
pdocs = [{**d, 'labels': []} for d in dev]; g = gold_as_preds(dev)
for name, pr in [('gold', g), ('70% + ruido', [x for x in g if rng.random() < .7] + [dict(x, r=rng.choice(PIDS)) for x in g[:2000]])]:
    a = evaluate(pr, dev, train)['ign_f1']; b = ign_e2e(pdocs, dev, pr, train)['ign_f1']
    print(f'{name}: eval.evaluate Ign={a:.6f} ign_e2e={b:.6f}'); assert abs(a - b) < 1e-6
print('OK')
