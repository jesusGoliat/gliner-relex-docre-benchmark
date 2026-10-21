"""Registra, desde la misma imagen de Modal usada en los experimentos (capas en caché), las versiones exactas de las dependencias instaladas y la
revisión del modelo base descargada al construirla, y calcula el sha256 de los checkpoints guardados en el volumen 'relex-outputs'.
Uso: set -a; source .env; set +a; modal run modal/inspect_artifacts.py
Salidas (RUNS = RELEX_RUNS_DIR, def. <repo>/runs): RUNS/environment/modal_env.json y RUNS/environment/checkpoints.csv."""
import os, sys, json, csv, glob, hashlib, subprocess
import modal
vol = modal.Volume.from_name('relex-outputs')
if modal.is_local():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from relex_modal import image, RUNS          # misma definición de imagen que los experimentos (capas en caché)
else:
    image = None                                  # dentro del contenedor la imagen ya está construida

app = modal.App('relex-inspect', image=image)
MODELS = [f'ftB_N{n}_s{k}' for n in (10, 50) for k in range(5)] + [f'a3B{a}_N10_s{k}' for a in ('ctrl', 'judge') for k in range(5)]


@app.function(timeout=900)
def env_info():
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout.splitlines()
    snaps = sorted(glob.glob('/root/hf/**/models--knowledgator--gliner-relex-large-v1.0/snapshots/*', recursive=True))
    return dict(python=sys.version, pip_freeze=freeze, model_snapshots=snaps)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


@app.function(volumes={'/out': vol}, cpu=4, timeout=3 * 3600)
def hash_checkpoints():
    from concurrent.futures import ThreadPoolExecutor
    vol.reload()
    files = [(m, os.path.join(r, f)) for m in MODELS for r, _, fs in os.walk(f'/out/models/{m}') for f in fs]
    def one(a):
        m, p = a
        return dict(model=m, file=os.path.relpath(p, f'/out/models/{m}'), bytes=os.path.getsize(p), sha256=_sha(p))
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, files))


@app.local_entrypoint()
def main():
    info = env_info.remote()
    rows = hash_checkpoints.remote()
    os.makedirs(f'{RUNS}/environment', exist_ok=True)
    json.dump(info, open(f'{RUNS}/environment/modal_env.json', 'w'), indent=1)
    w = csv.DictWriter(open(f'{RUNS}/environment/checkpoints.csv', 'w', newline=''), ['model', 'file', 'bytes', 'sha256']); w.writeheader(); w.writerows(rows)
    print('OK', len(info['pip_freeze']), 'paquetes;', len(rows), 'ficheros de checkpoints;', len({r['model'] for r in rows}), 'modelos')
