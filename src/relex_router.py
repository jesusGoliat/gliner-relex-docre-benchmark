"""Enrutador del juez para GLiNER-Relex. Banda explícita [lo, hi) sobre la puntuación de relación y top-k explícito con la tasa de recorte
reportada. Fusión dura con respaldo al extractor si la respuesta no se parsea."""
import numpy as np
from data import PIDS
from constraints import allowed, table_menotti

_M = None
def menotti():
    global _M
    if _M is None: _M = table_menotti()
    return _M


def band_split(pdoc, S, lo, hi, topk=5):
    """-> (sure (E,E,96) bool, routed [(h,t,[cols])] , stats). Candidatos solo si el tipo de h/t admite la relación (Menotti)."""
    E = S.shape[0]
    off = ~np.eye(E, dtype=bool)[:, :, None]
    al = allowed(pdoc, menotti()) & off if E else np.zeros_like(S, dtype=bool)
    sure = (S >= hi) & al
    band = (S >= lo) & (S < hi) & al
    n_band = int(band.sum()); n_sure = int(sure.sum()); n_low = int(((S > 0) & (S < lo) & al).sum()); n_blocked = int(((S >= lo) & ~al).sum())
    sb = np.where(band, S, -1.0)
    idx = np.argsort(-sb, -1)[..., :topk]
    keep = np.zeros_like(band); np.put_along_axis(keep, idx, True, -1)
    kept = band & keep
    routed = [(int(h), int(t), sorted(np.nonzero(kept[h, t])[0].tolist(), key=lambda c: -S[h, t, c]))
              for h, t in zip(*np.nonzero(kept.any(-1)))]
    return sure, routed, dict(sure=n_sure, band=n_band, routed_opts=int(kept.sum()), trunc=n_band - int(kept.sum()),
                              below_lo=n_low, blocked_by_types=n_blocked, pairs=len(routed))


def fuse_hard(sure, S, dec, th_fallback):
    """sure + opciones elegidas por el juez en los pares enrutados. Si la respuesta no se pudo parsear (ok=False), se conserva la
    decisión del extractor (S >= th_fallback) para esas opciones."""
    m = sure.copy(); n_fail = 0
    for h, t, cols, ch, conf, ok, dist in dec:
        if ok:
            for c in ch: m[h, t, c] = True
        else:
            n_fail += 1
            for c in cols:
                if S[h, t, c] >= th_fallback: m[h, t, c] = True
    return m, n_fail


def mask_to_preds(title, mask):
    h, t, c = np.nonzero(mask)
    return [{'title': title, 'h_idx': int(a), 't_idx': int(b), 'r': PIDS[int(k)]} for a, b, k in zip(h, t, c)]


MASK_TYPES = {'PER', 'ORG', 'LOC', 'MISC'}


def mask_pred_doc(pdoc):
    """Control de memoria del LLM: sustituye cada mención de entidades PER/ORG/LOC/MISC por un único token 'ENT<i>' (i = índice de la entidad)
    y recalcula las posiciones; TIME y NUM no se enmascaran (llevan información necesaria para las relaciones de fecha/cantidad).
    Devuelve un documento con las mismas entidades (mismo orden) y los textos modificados, solo para construir el prompt del juez."""
    sents = [list(s) for s in pdoc['sents']]
    ments = []   # (sent_id, i, j, ent_idx, mention_idx)
    for e, ent in enumerate(pdoc['vertexSet']):
        for k, m in enumerate(ent):
            if m['type'] in MASK_TYPES: ments.append((m['sent_id'], m['pos'][0], m['pos'][1], e, k))
    shift = {}   # sent_id -> lista (inicio_original, fin_original, nuevo token) ordenada
    for sid, i, j, e, k in sorted(ments):
        shift.setdefault(sid, []).append((i, j, f'ENT{e}'))
    new_sents, remap = [], {}
    for sid, s in enumerate(sents):
        out, pos, mp = [], 0, {}
        reps = sorted(shift.get(sid, []))
        ri = 0
        while pos < len(s):
            if ri < len(reps) and reps[ri][0] == pos:
                i, j, tok = reps[ri]; mp[i] = len(out); out.append(tok); pos = j
                while ri < len(reps) and reps[ri][0] < pos: ri += 1   # menciones solapadas dentro del mismo span: se omiten
            else:
                mp[pos] = len(out); out.append(s[pos]); pos += 1
        mp[len(s)] = len(out)
        new_sents.append(out); remap[sid] = (mp, reps)
    vs = []
    for e, ent in enumerate(pdoc['vertexSet']):
        nent = []
        for m in ent:
            mp, reps = remap[m['sent_id']]
            if m['type'] in MASK_TYPES:
                st = mp.get(m['pos'][0]); nm = f'ENT{e}'
                nent.append({**m, 'name': nm, 'pos': [st, st + 1] if st is not None else m['pos']})
            else:
                a, b = mp.get(m['pos'][0]), mp.get(m['pos'][1])
                nent.append({**m, 'pos': [a, b]} if a is not None and b is not None else dict(m))
        vs.append(nent)
    return {**pdoc, 'sents': new_sents, 'vertexSet': vs}
