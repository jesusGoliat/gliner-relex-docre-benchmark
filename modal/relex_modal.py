"""GLiNER-Relex en Modal: inferencia y ajuste fino. Cada modo se lanza con (desde la raíz, con los secretos cargados):
  set -a; source .env; set +a; modal run [--detach] modal/relex_modal.py --mode MODO
Modos:
  zs_dev        Z0, etiquetas v0, rejilla (entidad x adyacencia x flat) en D=50 de dev          -> RUNS/raw/zs_dev_D50.json
  zs_labels     Z0, variantes de etiquetas v1 y v3 en D=50 de dev                                 -> RUNS/raw/zs_labels_D50.json
  zs_test       Z0 congelada en test (ALLOW_TEST=1)                                               -> RUNS/raw/zs_test_Z0.json
  train_b0      B0: ajuste fino N in {10,50} x 5 subconjuntos, checkpoint/umbrales elegidos en D  -> RUNS/selection/ftB_baseline_dev_D50.json
  infer_dev     salidas crudas de B0 en D=50 de dev                                               -> RUNS/raw/ftB_dev_N*_s*.json
  infer_test    salidas crudas de B0 en test (ALLOW_TEST=1)                                       -> RUNS/raw/ftB_test_N*_s*.json
RUNS = RELEX_RUNS_DIR (def. <repo>/runs). Los modelos viven en el volumen 'relex-outputs' (/models).
GPU de inferencia: RELEX_GPU (def. A100-40GB); GPU de ajuste fino: RELEX_FT_GPU (def. H100)."""
import os, sys, json, time
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
RUNS = os.environ.get('RELEX_RUNS_DIR', os.path.join(ROOT, 'runs'))
GPU = os.environ.get('RELEX_GPU', 'A100-40GB')
MODEL = os.environ.get('RELEX_MODEL', 'knowledgator/gliner-relex-large-v1.0')
GLINER = os.environ.get('RELEX_GLINER', 'gliner==0.2.29')

image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install(GLINER, 'torch', 'numpy<2.3', 'huggingface_hub', 'accelerate>=1.1.0', 'scipy', 'dateparser==1.4.3')
         .env({'HF_HOME': '/root/hf', 'TOKENIZERS_PARALLELISM': 'false'})
         .run_commands(f"python -c \"from gliner import GLiNER; GLiNER.from_pretrained('{MODEL}')\""))
EXTD = os.path.join(ROOT, 'data', 'external')
image = (image
         .add_local_dir(os.path.join(ROOT, 'src'), '/c1/src', ignore=['__pycache__'])
         .add_local_dir(os.path.join(ROOT, 'data', 'subsets', 'redocred'), '/c1/data/subsets/redocred')
         .add_local_dir(os.path.join(EXTD, 'DocRE-StructuralConsistency', 'data', 'meta'), '/c1/data/external/DocRE-StructuralConsistency/data/meta')
         .add_local_file(os.path.join(EXTD, 'Re-DocRED', 'data', 'dev_revised.json'), '/c1/data/external/Re-DocRED/data/dev_revised.json'))
app = modal.App('relex', image=image)
vol = modal.Volume.from_name('relex-outputs', create_if_missing=True)

FT_GPU = os.environ.get('RELEX_FT_GPU', 'H100')
FT_BS = {10: 5, 50: 10}      # tamaño de lote del ajuste fino de B0 (<= N)
FT_LR = 2e-5


@app.function(gpu=FT_GPU, timeout=4 * 3600, volumes={'/out': vol}, memory=32768)
def ft_run(n: int, sub: int, steps: list, tag: str, seed: int = 42, extra: str = '', bs_override: int = 0):
    """Ajusta Relex con train_N{n}_s{sub}, evalúa checkpoints en D=50 de dev, elige por argmax(F1 e2e estricto con tipo),
    guarda el mejor en /out/models/{tag}_N{n}_s{sub} y devuelve la tabla completa."""
    import sys, shutil, torch, gliner
    sys.path.insert(0, '/c1/src')
    from transformers import TrainerCallback
    from gliner import GLiNER
    from data import load, load_red, PREP
    from relex_io import to_train_relex, to_text, ENT_LABELS, REL_NAMES, LABEL2TYPE
    from ft_select import sweep, best, d_indices
    bs = bs_override or FT_BS[n]
    train = load(os.path.join(PREP, 'redocred', f'train_N{n}_s{sub}.json'))
    assert bs <= len(train), 'lote > N'
    ex = [to_train_relex(d) for d in train]
    n_gold = len(ex)
    if extra:                      # pseudo-etiquetas (A3 / control): ejemplos GLiNER ya convertidos, en el volumen
        vol.reload(); ex = ex + json.load(open(f'/out/{extra}'))
    dev = load_red('dev'); idx = d_indices(len(dev)); docs = [dev[i] for i in idx]; texts = [to_text(d) for d in docs]
    t0 = time.time()
    m = GLiNER.from_pretrained(MODEL).to('cuda')
    work = f'/tmp/ft_{tag}_{n}_{sub}'; shutil.rmtree(work, ignore_errors=True)

    class SaveAt(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step in steps:
                control.should_save = True
            return control
    targs = m.create_training_args(output_dir=work, learning_rate=FT_LR, others_lr=FT_LR, lr_scheduler_type='constant', warmup_ratio=0.0,
                                   per_device_train_batch_size=bs, max_steps=max(steps), bf16=True, save_steps=10**9,
                                   save_total_limit=len(steps) + 1, logging_steps=10, dataloader_num_workers=0, report_to='none',
                                   seed=seed, data_seed=seed)
    from gliner.model import _get_trainer_classes
    Trainer, _ = _get_trainer_classes()
    tk = dict(model=m, args=targs, train_dataset=ex, eval_dataset=None, data_collator=m._create_data_collator(), callbacks=[SaveAt()])
    import transformers
    from packaging import version
    tk['tokenizer' if version.parse(transformers.__version__) < version.parse('5.0.0') else 'processing_class'] = m.data_processor.transformer_tokenizer
    trainer = Trainer(**tk)
    trainer.train()
    losses = [(h['step'], h['loss']) for h in trainer.state.log_history if 'loss' in h]
    train_sec = time.time() - t0
    del trainer, m; torch.cuda.empty_cache()
    rows, outs = [], {}
    cfgs = [dict(threshold=te, adjacency_threshold=0.5, relation_threshold=0.05, flat_ner=True, batch_size=8) for te in (0.3, 0.5)]
    for st in steps:
        mm = GLiNER.from_pretrained(f'{work}/checkpoint-{st}').to('cuda').eval()
        runs = []
        for c in cfgs:
            e_, r_ = mm.inference(texts=texts, labels=ENT_LABELS, relations=REL_NAMES, return_relations=True, **c)
            runs.append(dict(cfg=c, ents=e_, rels=r_))
        rows += sweep(docs, runs, LABEL2TYPE, tag=st)
        del mm; torch.cuda.empty_cache()
    b = best(rows)
    dest = f'/out/models/{tag}_N{n}_s{sub}'
    shutil.rmtree(dest, ignore_errors=True); os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(f'{work}/checkpoint-{b["tag"]}', dest)
    json.dump(dict(n=n, sub=sub, best=b, steps=steps, bs=bs, lr=FT_LR, seed=seed, n_gold=n_gold, n_extra=len(ex) - n_gold, extra=extra, rows=rows), open(dest + '.json', 'w'))
    vol.commit()
    return dict(n=n, sub=sub, best=b, n_gold=n_gold, n_extra=len(ex) - n_gold, rows=rows, losses=losses[::5], train_sec=train_sec, total_sec=time.time() - t0,
                versions=dict(gliner=gliner.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(0)))


@app.function(gpu=GPU, timeout=3 * 3600, volumes={'/out': vol})
def infer_grid(texts: list, ent_labels, rel_labels: list, configs: list, model_ref: str = ''):
    """Inferencia cruda para varias configuraciones (cada una: dict de kwargs de inference). model_ref vacío = modelo base;
    '/out/...' = modelo guardado en el volumen. Devuelve [{'cfg', 'ents', 'rels', 'sec'}]."""
    import torch, gliner
    from gliner import GLiNER
    m = GLiNER.from_pretrained(model_ref or MODEL).to('cuda').eval()
    res = []
    for cfg in configs:
        t0 = time.time()
        ents, rels = m.inference(texts=texts, labels=ent_labels, relations=rel_labels, return_relations=True, **cfg)
        res.append(dict(cfg=cfg, ents=ents, rels=rels, sec=time.time() - t0))
    return dict(gliner=gliner.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), runs=res)


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, 'w'), default=str)


def _d_indices(n_dev):
    import numpy as np
    return sorted(np.random.default_rng(1234).permutation(n_dev)[:50].tolist())     # D = 50 documentos de dev (semilla 1234)


def _require_test():
    assert os.environ.get('ALLOW_TEST') == '1', 'test solo con la configuración congelada (ALLOW_TEST=1)'


@app.local_entrypoint()
def main(mode: str):
    from data import load_red
    from relex_io import to_text, ENT_LABELS, REL_NAMES, LABEL_VARIANTS
    if mode == 'zs_dev':
        dev = load_red('dev'); idx = _d_indices(len(dev))
        cfgs = [dict(threshold=te, adjacency_threshold=ta, relation_threshold=0.05, flat_ner=fl, batch_size=8)
                for te in (0.3, 0.5) for ta in (0.3, 0.5, 0.65) for fl in (True, False)]
        out = infer_grid.remote([to_text(dev[i]) for i in idx], ENT_LABELS, REL_NAMES, cfgs)
        _save(f'{RUNS}/raw/zs_dev_D50.json', dict(idx=idx, **out))
    elif mode == 'zs_labels':
        dev = load_red('dev'); idx = _d_indices(len(dev))
        cfgs = [dict(threshold=te, adjacency_threshold=0.5, relation_threshold=0.05, flat_ner=True, batch_size=8) for te in (0.3, 0.5)]
        outs = {v: infer_grid.remote([to_text(dev[i]) for i in idx], LABEL_VARIANTS[v][0], REL_NAMES, cfgs) for v in ('v1', 'v3')}
        _save(f'{RUNS}/raw/zs_labels_D50.json', dict(idx=idx, variants=outs))
    elif mode == 'zs_test':
        _require_test()
        te = load_red('test')
        out = infer_grid.remote([to_text(d) for d in te], LABEL_VARIANTS['v3'][0], REL_NAMES,
                                [dict(threshold=0.3, adjacency_threshold=0.5, relation_threshold=0.05, flat_ner=True, batch_size=8)])
        _save(f'{RUNS}/raw/zs_test_Z0.json', out)
    elif mode == 'train_b0':
        ST = {10: [20, 40, 80, 160, 320, 480], 50: [25, 50, 100, 200, 400, 800]}      # checkpoints evaluados en D
        res = list(ft_run.starmap([(n, k, ST[n], 'ftB', 42) for n in (10, 50) for k in range(5)], return_exceptions=True))
        _save(f'{RUNS}/selection/ftB_baseline_dev_D50.json', [r if isinstance(r, dict) else str(r) for r in res])
    elif mode in ('infer_dev', 'infer_test'):
        split = 'dev' if mode == 'infer_dev' else 'test'
        if split == 'test': _require_test()
        sel = json.load(open(f'{RUNS}/selection/ftB_baseline_dev_D50.json'))
        docs_all = load_red(split)
        idx = _d_indices(len(docs_all)) if split == 'dev' else list(range(len(docs_all)))
        texts = [to_text(docs_all[i]) for i in idx]
        calls = [infer_grid.spawn(texts, ENT_LABELS, REL_NAMES,
                                  [dict(threshold=r['best']['ent_thr'], adjacency_threshold=0.5, relation_threshold=0.05, flat_ner=True, batch_size=8)],
                                  f"/out/models/ftB_N{r['n']}_s{r['sub']}") for r in sel]
        for r, c in zip(sel, calls):
            _save(f"{RUNS}/raw/ftB_{split}_N{r['n']}_s{r['sub']}.json", dict(idx=idx, n=r['n'], sub=r['sub'], best=r['best'], **c.get()))
    else:
        raise SystemExit(f'modo desconocido: {mode}')
    print('OK', mode)
