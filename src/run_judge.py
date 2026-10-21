"""Juez LLM (J0) sobre los candidatos de la banda [lo, hi) del extractor ajustado B0, para un tamaño N y un subconjunto.
Modos:
  * selección de banda: --split dev (D = 50 documentos de dev);
  * evaluación: --split test (requiere ALLOW_TEST=1; configuración congelada; guarda métricas completas y conteos por documento);
  * control de memoria del LLM: --first K (solo los K primeros documentos) con/sin --mask-entities (entidades PER/ORG/LOC/MISC como ENT<i> en el prompt).
Cada ejecución añade una línea JSON a RUNS/{selection|test|masking_control}/ (ver abajo) y la imprime.
Uso: set -a; source .env; set +a; C1_MAX_USD=9 C1_DI_CONCURRENCY=200 python src/run_judge.py --n 10 --sub 0 --lo 0.1 --hi 0.7 --split dev"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data import load, PREP, RUNS
import llm_router as LR
from runs import load_run
from relex_router import band_split, fuse_hard, mask_to_preds, mask_pred_doc
from eval_e2e import re_e2e, per_doc_counts_e2e

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, required=True); ap.add_argument('--sub', type=int, required=True)
ap.add_argument('--lo', type=float, required=True); ap.add_argument('--hi', type=float, required=True)
ap.add_argument('--topk', type=int, default=5); ap.add_argument('--tag', default='')
ap.add_argument('--split', default='dev', choices=['dev', 'test'])
ap.add_argument('--first', type=int, default=0, help='usar solo los K primeros documentos (0 = todos)')
ap.add_argument('--mask-entities', action='store_true', help='control de memoria del LLM: enmascara PER/ORG/LOC/MISC en el prompt')
a = ap.parse_args()
assert a.split == 'dev' or os.environ.get('ALLOW_TEST') == '1', 'test solo con la configuración congelada (ALLOW_TEST=1)'
docs, P, S, best = load_run(a.n, a.sub, a.split)
if a.first: docs, P, S = docs[:a.first], P[:a.first], S[:a.first]
train = load(os.path.join(PREP, 'redocred', f'train_N{a.n}_s{a.sub}.json'))
rt = best['rel_thr']
sure, routed, stats = [], [], []
for pd, s in zip(P, S):
    su, ro, st = band_split(pd, s, a.lo, a.hi, a.topk); sure.append(su); routed.append(ro); stats.append(st)
q = sum(len(r) for r in routed); nopt = sum(sum(len(c) for _, _, c in r) for r in routed)
t0 = time.time(); s0 = dict(LR.SPEND)
PJ = [mask_pred_doc(p) for p in P] if a.mask_entities else P
dec = LR.run_many(PJ, routed, train)       # documentos predichos (con tipos) como documentos del prompt: entidades por nombre
secs = time.time() - t0
masks, fails = [], 0
for su, s, dc in zip(sure, S, dec):
    m, nf = fuse_hard(su, s, dc, rt); masks.append(m); fails += nf
base_p = [p for d, s in zip(docs, S) for p in mask_to_preds(d['title'], s >= rt)]
judge_p = [p for d, m in zip(docs, masks) for p in mask_to_preds(d['title'], m)]
b = re_e2e(P, docs, base_p, 'strict', True); j = re_e2e(P, docs, judge_p, 'strict', True)
failed = LR.SPEND.get('failed', 0); calls = LR.SPEND['calls'] - s0['calls']; cached = LR.SPEND['cached'] - s0['cached']
row = dict(split=a.split, tag=a.tag, mask=int(a.mask_entities), first=a.first, n=a.n, sub=a.sub, lo=a.lo, hi=a.hi, topk=a.topk, docs=len(docs),
           questions=q, options=nopt, q_per_doc=q / len(docs),
           base_f1=b['f1'], judge_f1=j['f1'], delta=j['f1'] - b['f1'], base_p=b['p'], base_r=b['r'], judge_p=j['p'], judge_r=j['r'],
           parse_fail_questions=fails, failed_calls=failed, valid_rate=1 - fails / max(q, 1), calls=calls, cached_calls=cached,
           usd_session=LR.spend_usd(), usd_total_cache=LR.total_spend_usd(), secs=secs, rt=rt,
           sure=sum(x['sure'] for x in stats), trunc=sum(x['trunc'] for x in stats), band=sum(x['band'] for x in stats),
           model=LR.KEY_MODEL, prompt=LR.PROMPT)
if a.split == 'test' and not a.first:     # métricas completas + conteos por documento (para el bootstrap)
    from eval_test import metrics
    row.update({f'judge_{k}': v for k, v in metrics(P, docs, judge_p).items()})
    os.makedirs(f'{RUNS}/test', exist_ok=True)
    np.savez(f'{RUNS}/test/J0_counts_N{a.n}_s{a.sub}.npz', judge=per_doc_counts_e2e(P, docs, judge_p, 'strict', True),
             base=per_doc_counts_e2e(P, docs, base_p, 'strict', True))
line = json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
print(line)
out = f'{RUNS}/masking_control/judge_mask.jsonl' if a.first else (f'{RUNS}/selection/judge_runs_dev.jsonl' if a.split == 'dev' else f'{RUNS}/test/judge_runs_test.jsonl')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'a') as f: f.write(line + '\n')
