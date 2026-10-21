"""Agrupa menciones en entidades con reglas (nombre, alias PER, acrónimos/contención, fechas, números).
Entrada: menciones {sent_id, pos:[i,j), name, type, score, src, tprobs?}. Salida: entidades [{mentions, type, types2, tprobs}]
(type = argmax de tprobs; types2 = [type] + segundo tipo si su P >= 0.20)."""
import re, collections

SOFT = {'ORG', 'LOC', 'MISC'}
MONTH = {m: i + 1 for i, m in enumerate('january february march april may june july august september october november december'.split())}
MONTH.update({k[:3]: v for k, v in list(MONTH.items())})
MONTH['sept'] = 9
WORDNUM = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve'.split())}


def compat(a, b):
    return a == b or (a in SOFT and b in SOFT)


def key_name(name):
    """Minúsculas, sin puntuación ni artículo inicial."""
    t = re.sub(r"['’]s\b", '', name.lower())
    t = re.sub(r'[^\w\s]', ' ', t).split()
    if len(t) > 1 and t[0] in ('the', 'a', 'an'):
        t = t[1:]
    return ' '.join(t)


def time_key(name):
    """Forma canónica (año, mes, día) con las componentes presentes; dateparser para el resto."""
    low = name.lower().replace(' ,', ',')
    m = re.search(r'\b(1[0-9]{3}|20[0-9]{2})(0s|s)?\b', low)
    if re.fullmatch(r'\d{4}s', low.strip()):
        return ('dec', int(low[:4]))
    toks = re.findall(r'[a-z]+|\d+', low)
    mon = next((MONTH[t] for t in toks if t in MONTH), None)
    day = next((int(t) for t in toks if t.isdigit() and len(t) <= 2 and 1 <= int(t) <= 31), None)
    if m and mon is None and day is None and len(toks) <= 2:
        return ('d', int(m.group(1)), None, None)
    try:
        import dateparser
        dt = dateparser.parse(name, languages=['en'], settings={
            'PREFER_DAY_OF_MONTH': 'first', 'PREFER_DATES_FROM': 'past',
            'RELATIVE_BASE': __import__('datetime').datetime(2000, 1, 1)})
    except Exception:
        dt = None
    if dt is None or not m:        # sin año explícito o no parseable: solo nombre normalizado
        return ('s', key_name(name))
    return ('d', dt.year, dt.month if mon or re.search(r'\d+/\d+|-\d+-', name) else None,
            dt.day if day is not None else None)


def num_key(name):
    low = name.lower().replace(',', '')
    toks = re.findall(r'\d+(?:\.\d+)?|[a-z]+|[%$€£]', low)
    val, rest = None, []
    for t in toks:
        if val is None and (re.fullmatch(r'\d+(?:\.\d+)?', t)):
            val = float(t)
        elif val is None and t in WORDNUM:
            val = float(WORDNUM[t])
        elif t not in ('the', 'a', 'an'):
            rest.append(t.rstrip('s') if len(t) > 3 else t)
    return ('n', val, tuple(rest)) if val is not None else ('s', key_name(name))


def _acro(toks):
    return ''.join(t[0] for t in toks if t[0].isupper()).lower()


class UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)


def _tokname(m):
    return re.sub(r"['’]s$", '', m['name']).split()


def cluster(mentions, sents=None):
    # 1. deduplicar por posición (queda la de mayor score)
    best = {}
    for m in mentions:
        k = (m['sent_id'], m['pos'][0], m['pos'][1])
        if k not in best or m.get('score', 1) > best[k].get('score', 1):
            best[k] = m
    ms = sorted(best.values(), key=lambda m: (m['sent_id'], m['pos'][0], m['pos'][1]))
    n = len(ms)
    uf = UF(n)

    # 2. regla (a)/(d)/(e): misma clave de nombre + tipo compatible
    groups = collections.defaultdict(list)
    for i, m in enumerate(ms):
        if m['type'] == 'TIME':
            k = ('T',) + time_key(m['name'])
        elif m['type'] == 'NUM':
            k = ('N',) + num_key(m['name'])
        else:
            k = ('X', key_name(m['name']))
        groups[k].append(i)
    for k, idx in groups.items():
        for a in idx[1:]:
            for b in idx:
                if b >= a: break
                if compat(ms[a]['type'], ms[b]['type']):
                    uf.u(a, b); break      # une con el primero compatible (sin encadenar tipos incompatibles)

    # 2b. PER multi-token con mismo nombre de pila y apellido (ignora Jr.): "Henrik Angell" ~ "Henrik August Angell"
    def pkey(m):
        t = [x for x in _tokname(m) if x.strip('.').lower() not in ('jr',)]
        return (t[0].lower(), t[-1].lower()) if len(t) >= 2 else None
    byk = collections.defaultdict(list)
    for i, m in enumerate(ms):
        if m['type'] == 'PER' and pkey(m):
            byk[pkey(m)].append(i)
    for idx in byk.values():
        for a in idx[1:]:
            uf.u(a, idx[0])
    # 2c. demónimos LOC ("Cambodian"->"Cambodia", "Vietnamese"->"Vietnam")
    locs = {key_name(m['name']): i for i, m in enumerate(ms) if m['type'] == 'LOC'}
    for i, m in enumerate(ms):
        if m['type'] == 'LOC':
            k = key_name(m['name'])
            for suf in ('ese', 'ian', 'an', 'n'):
                if k.endswith(suf) and k[:-len(suf)] in locs and len(k) > len(suf) + 3:
                    uf.u(i, locs[k[:-len(suf)]]); break

    # 3. reglas (b) y (c): nombres cortos hacia largos (solo si el candidato es único)
    def comps():
        c = collections.defaultdict(list)
        for i in range(n):
            c[uf.f(i)].append(i)
        return c
    cs = comps()
    rep = {r: ms[idx[0]] for r, idx in cs.items()}      # nombre representativo del componente
    # nombre más largo de cada componente
    longest = {r: max((ms[i] for i in idx), key=lambda m: len(_tokname(m))) for r, idx in cs.items()}
    for r, m in rep.items():
        if m['type'] not in ('PER', 'ORG', 'LOC', 'MISC'):
            continue
        short = _tokname(longest[r]) if False else _tokname(m)
        cand = set()
        for r2, L in longest.items():
            if r2 == r or not compat(m['type'], L['type']):
                continue
            lt = _tokname(L)
            if len(lt) <= len(short):
                continue
            if m['type'] == 'PER' and L['type'] == 'PER':
                if len(short) == 1 and short[0][:1].isupper() and (short[0] == lt[-1] or short[0] == lt[0]):
                    cand.add(r2)
            elif m['type'] != 'PER' and L['type'] == m['type']:     # contención/acrónimo: mismo tipo
                core = [t.lower().strip('.') for t in short]
                if len(short) == 1 and len(short[0]) >= 2 and short[0].isupper() and short[0].isalpha() \
                        and short[0].lower() == _acro(lt):
                    cand.add(r2)
                elif m['type'] != 'LOC' and any(t[:1].isupper() for t in short) and sum(len(t) for t in short) >= 2:
                    ll = [t.lower().strip('.') for t in lt]
                    if any(ll[i:i + len(core)] == core for i in range(len(ll) - len(core) + 1)):
                        cand.add(r2)
        if len(cand) == 1:
            uf.u(next(iter(cs[r])), cs[next(iter(cand))][0])

    # 4. salida
    out = []
    for idx in comps().values():
        mm = [ms[i] for i in idx]
        tp, w = collections.defaultdict(float), 0.0
        for m in mm:
            s = m.get('score', 1.0)
            for t, p in (m.get('tprobs') or {m['type']: 1.0}).items():
                tp[t] += s * p
            w += s
        tp = {t: v / w for t, v in tp.items()}
        order = sorted(tp, key=lambda t: -tp[t])
        t2 = [order[0]] + ([order[1]] if len(order) > 1 and tp[order[1]] >= 0.20 else [])
        out.append({'mentions': mm, 'type': order[0], 'types2': t2, 'tprobs': tp})
    out.sort(key=lambda e: (e['mentions'][0]['sent_id'], e['mentions'][0]['pos'][0]))
    return out
