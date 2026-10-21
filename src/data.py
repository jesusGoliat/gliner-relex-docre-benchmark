"""Rutas y datos de Re-DocRED (formato interno de documentos).
Re-DocRED: {title, sents, vertexSet[[{name,pos,sent_id,type}]], labels[{r,h,t,evidence}]}"""
import json, os

C1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # raíz del repositorio
EXT = os.path.join(C1, 'data', 'external')                                # descargas (data/download.sh)
PREP = os.path.join(C1, 'data', 'subsets')                                # subconjuntos de entrenamiento N=10/50 (versionados)
RED_DIR = os.path.join(EXT, 'Re-DocRED', 'data')
# Salidas de NUEVAS ejecuciones. results/ contiene los resultados oficiales y el código nunca escribe en él.
RUNS = os.environ.get('RELEX_RUNS_DIR', os.path.join(C1, 'runs'))

REL_INFO = json.load(open(os.path.join(EXT, 'DocRE-StructuralConsistency', 'data', 'meta', 'rel_info.json')))
PIDS = list(REL_INFO)                                   # 96 P-ids, orden de rel_info.json
REL2ID = {'Na': 0, **{p: i + 1 for i, p in enumerate(PIDS)}}
assert len(PIDS) == 96 and len({v for v in REL_INFO.values()}) == 96


def load(path):
    return json.load(open(path))


def load_red(split):
    """split in {train, dev, test}"""
    return load(os.path.join(RED_DIR, f'{split}_revised.json'))
