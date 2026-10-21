"""Entrada/salida de GLiNER-Relex sobre documentos Re-DocRED (sin dependencias de gliner: se prueba en local).
- ENT_LABELS: nombre de etiqueta de entidad por tipo de Re-DocRED (D3: 6 tipos); REL_NAMES: las 96 relaciones por nombre.
- to_text / to_train_relex: documento -> texto (tokens unidos por espacio) / ejemplo de ajuste fino de GLiNER (tokenized_text, ner, relations).
- char_offsets / span_tokens: conversión de offsets de caracteres de la salida a (sent_id, i, j) de Re-DocRED."""
from data import REL_INFO, PIDS

TYPE2LABEL = {'PER': 'person', 'ORG': 'organization', 'LOC': 'location', 'TIME': 'time', 'NUM': 'number', 'MISC': 'miscellaneous'}
LABEL2TYPE = {v: k for k, v in TYPE2LABEL.items()}
ENT_LABELS = list(TYPE2LABEL.values())
REL_NAMES = [REL_INFO[p] for p in PIDS]
NAME2PID = {REL_INFO[p]: p for p in PIDS}


def flat_tokens(doc):
    return [w for s in doc['sents'] for w in s]


def to_text(doc):
    return ' '.join(flat_tokens(doc))


def char_offsets(doc):
    """[(sent_id, tok_idx, c0, c1)] sobre to_text(doc)."""
    offs, c = [], 0
    for sid, s in enumerate(doc['sents']):
        for k, w in enumerate(s):
            offs.append((sid, k, c, c + len(w))); c += len(w) + 1
    return offs


def span_tokens(offs, cs, ce):
    """(sent_id, i, j) con j exclusivo; None si el span está vacío o cruza oraciones."""
    hit = [(s, k) for s, k, a, b in offs if a < ce and b > cs]
    if not hit or len({s for s, _ in hit}) > 1:
        return None
    return hit[0][0], hit[0][1], hit[-1][1] + 1


def to_train_relex(doc):
    """Ejemplo de ajuste fino: ner=[ini, fin_inclusivo, etiqueta] por mención; relations=[idx_ner_h, idx_ner_t, nombre]
    expandidas a TODOS los pares de menciones (cabeza x cola) de cada tripleta gold de la entidad."""
    base, ner, idx = [0], [], {}
    for s in doc['sents']:
        base.append(base[-1] + len(s))
    for e, ent in enumerate(doc['vertexSet']):
        for k, m in enumerate(ent):
            idx[(e, k)] = len(ner)
            ner.append([base[m['sent_id']] + m['pos'][0], base[m['sent_id']] + m['pos'][1] - 1, TYPE2LABEL[m['type']]])
    rels = []
    for l in doc['labels']:
        for a in range(len(doc['vertexSet'][l['h']])):
            for b in range(len(doc['vertexSet'][l['t']])):
                rels.append([idx[(l['h'], a)], idx[(l['t'], b)], REL_INFO[l['r']]])
    return {'tokenized_text': flat_tokens(doc), 'ner': ner, 'relations': rels}


# ---- variantes de etiquetas de entidad Se eligen en D=50 de dev ----
DESC_V1 = {   # etiqueta -> descripción (la API acepta dict etiqueta->descripción)
    'person': 'a human individual',
    'organization': 'a company, institution, sports team, band or other group of people',
    'location': 'a country, city, region, geographical feature, building or other physical place',
    'time': 'a date, year or other time expression',
    'number': 'a number or quantity',
    'miscellaneous': 'a work of art, book, film, song, event, award, product, language, nationality or religion',
}
V3_TYPES = {   # 23 descriptores finos -> tipo
    'PER': ['person'], 'ORG': ['organization', 'company', 'institution', 'sports team', 'musical band'],
    'LOC': ['location', 'country', 'city', 'state or province', 'geographical feature', 'building or facility'],
    'MISC': ['work of art', 'book, film, album or song', 'event', 'nationality or religion', 'award', 'product', 'language'],
    'TIME': ['date', 'year'], 'NUM': ['number', 'quantity']}
LABEL_VARIANTS = {
    'v0': (ENT_LABELS, LABEL2TYPE),
    'v1': (DESC_V1, LABEL2TYPE),
    'v3': ([d for ds in V3_TYPES.values() for d in ds], {d: t for t, ds in V3_TYPES.items() for d in ds}),
}
