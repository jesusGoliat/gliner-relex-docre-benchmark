"""Juez LLM sobre los candidatos de la banda (DeepInfra, openai/gpt-oss-120b, solo prompting).
El LLM decide los pares enrutados entre sus opciones o ninguna. Una llamada por documento (troceada si hay muchos pares); salida JSON con
elección(es) y confianza 1-5. Caché en disco por hash del prompt: la misma llamada no se paga dos veces. Lleva la cuenta de tokens y de USD.
Variables de entorno: C1_LLM (def. deepinfra:openai/gpt-oss-120b), C1_REASONING (def. medium), C1_MAX_USD (tope de gasto acumulado, 0 = sin tope),
C1_DI_CONCURRENCY, C1_DI_TIMEOUT, C1_DI_WALL, C1_LLM_CACHE (ruta de la caché; def. runs/llm_cache.jsonl)."""
import os, json, hashlib, threading, time, random
from concurrent.futures import ThreadPoolExecutor
from data import REL_INFO, PIDS, REL2ID, RUNS

_LLM = os.environ.get('C1_LLM', 'deepinfra:openai/gpt-oss-120b')
assert _LLM.startswith('deepinfra:'), 'la evaluación usa un único servidor del juez: DeepInfra'
MODEL = _LLM.split(':', 1)[1]
REASONING = os.environ.get('C1_REASONING', 'medium')
KEY_MODEL = f'{MODEL}@deepinfra[{REASONING or "default"}]'      # forma parte de la clave de caché
PRICE = {'sync': (0.037, 0.17)}   # USD por millón de tokens (entrada, salida): tarifa publicada por DeepInfra para gpt-oss-120b
CACHE = os.environ.get('C1_LLM_CACHE', os.path.join(RUNS, 'llm_cache.jsonl'))
LETTERS = 'abcdefghijklmnopqrstuvwxy'
MAX_Q = 30
_lock = threading.Lock()
_cache = None
SPEND = {'in': 0, 'out': 0, 'calls': 0, 'cached': 0}
MAX_USD = float(os.environ.get('C1_MAX_USD', '0') or 0)   # tope de gasto acumulado (caché + sesión); 0 = sin tope. Se lanza RuntimeError al alcanzarlo
_BASE_USD = None


def total_spend_usd():
    """Gasto acumulado = llamadas ya cacheadas (de ejecuciones previas) + sesión actual."""
    global _BASE_USD
    if _BASE_USD is None:
        a, b = PRICE['sync']
        _BASE_USD = sum(r.get('in', 0) * a + r.get('out', 0) * b for r in _load_cache().values()) / 1e6 - 0.0
        _BASE_USD -= (SPEND['in'] * a + SPEND['out'] * b) / 1e6   # lo ya sumado en esta sesión no estaba en la caché al cargarla
    a, b = PRICE['sync']
    return _BASE_USD + (SPEND['in'] * a + SPEND['out'] * b) / 1e6


SYSTEM_V3 = (
    "You are an expert annotator for document-level relation extraction following the DocRED / Re-DocRED guidelines "
    "(Wikidata relations). Each question gives a HEAD entity, a TAIL entity and candidate relations proposed by an "
    "automatic model; many candidates are wrong. A relation R holds if the document states the fact (HEAD, R, TAIL) or it "
    "follows from the document by common-sense reasoning, read as 'TAIL is the R of HEAD' (e.g. (Paris, country, France)). "
    "Direction matters. DocRED conventions: an entity may be mentioned by a nationality/demonym (e.g. 'Canadian' denotes "
    "Canada, so (Alberta, country, Canadian) holds); geographic facts propagate (a city in a province in a country has "
    "that country); dates and numbers are entities too. Do not accept a fact just because both entities appear together: "
    "the document must support that specific relation between them. Study the solved examples, which show the annotation "
    "style including rejected candidates. Several options may hold; choose all that hold, or none. For every chosen option "
    "cite supporting sentence ids. Answer ONLY with JSON: {\"answers\": [{\"q\": <int>, \"choices\": [<letters>], "
    "\"evidence\": [<sentence ids>], \"confidence\": <1-5>}]} with one entry per question."
)
SYSTEM = SYSTEM_V3
PROMPT = 'v3'


_cache_lock = threading.Lock()
def _with_wall(fn, timeout):
    """Ejecuta fn en un hilo daemon; si tarda más de timeout s, se abandona (no bloquea la salida del proceso)."""
    import queue
    q = queue.Queue(1)
    threading.Thread(target=lambda: q.put(_try(fn)), daemon=True).start()
    try:
        ok, v = q.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(f'sin respuesta en {timeout:.0f} s')
    if not ok: raise v
    return v


def _try(fn):
    try: return True, fn()
    except Exception as e: return False, e


def _load_cache():
    # se publica la caché solo cuando está completa: antes otro hilo podía ver un dict a medio llenar (falsos fallos de caché)
    global _cache
    with _cache_lock:
        if _cache is None:
            c = {}
            if os.path.exists(CACHE):
                for line in open(CACHE):
                    try:
                        r = json.loads(line); c[r['key']] = r
                    except Exception:
                        pass
            _cache = c
    return _cache


def ent_desc(doc, i):
    names = []
    for m in doc['vertexSet'][i]:
        if m['name'] not in names: names.append(m['name'])
    return f"E{i} \"{names[0]}\"" + (f" (also: {', '.join(names[1:4])})" if len(names) > 1 else '') + f" [{doc['vertexSet'][i][0]['type']}]"


def doc_text(doc):
    return '\n'.join(f"[S{k}] " + ' '.join(s) for k, s in enumerate(doc['sents']))


def examples_qa(train_docs, relcols, k=8, seed=0):
    """Ejemplos resueltos en formato pregunta/respuesta a partir de los N documentos de entrenamiento:
    verdaderos (gold) y rechazados (par compatible por tipo, relación de las opciones, sin esa etiqueta)."""
    from constraints import table_from_docs, ent_types, TYPES
    rng = random.Random(seed)
    want = set(relcols)
    tab = table_from_docs(train_docs)
    pos, neg = [], []
    for d in train_docs:
        ty = ent_types(d); E = len(ty)
        lab = {(l['h'], l['t']): set() for l in d['labels']}
        for l in d['labels']: lab[(l['h'], l['t'])].add(REL2ID[l['r']] - 1)
        ev = {(l['h'], l['t']): l.get('evidence', []) for l in d['labels']}
        nm = lambda i: d['vertexSet'][i][0]['name']
        for (h, t), rs in lab.items():
            for r in rs & want:
                sid = (ev.get((h, t)) or [d['vertexSet'][h][0]['sent_id']])[0]
                pos.append(f"- Text: \"{' '.join(d['sents'][sid])}\" | HEAD '{nm(h)}' [{ty[h]}] TAIL '{nm(t)}' [{ty[t]}] | candidate: {REL_INFO[PIDS[r]]} => TRUE")
        for _ in range(30):
            h, t = rng.randrange(E), rng.randrange(E)
            if h == t: continue
            cand = [r for r in want if tab[TYPES.index(ty[h]), TYPES.index(ty[t]), r] and r not in lab.get((h, t), set())]
            if not cand: continue
            r = rng.choice(cand)
            shared = sorted({m['sent_id'] for m in d['vertexSet'][h]} & {m['sent_id'] for m in d['vertexSet'][t]})
            if not shared: continue
            neg.append(f"- Text: \"{' '.join(d['sents'][shared[0]])}\" | HEAD '{nm(h)}' [{ty[h]}] TAIL '{nm(t)}' [{ty[t]}] | candidate: {REL_INFO[PIDS[r]]} => FALSE")
    rng.shuffle(pos); rng.shuffle(neg)
    return pos[:k // 2 + k % 2] + neg[:k // 2]


def build_prompt(doc, questions, examples):
    """questions: lista de (h, t, [cols])."""
    assert all(len(cols) <= len(LETTERS) for _, _, cols in questions), 'más opciones que letras'
    ents = sorted({x for h, t, _ in questions for x in (h, t)})
    lines = [f"Document:\n{doc_text(doc)}", '', 'Entities:'] + [ent_desc(doc, e) for e in ents]
    if examples:
        lines += ['', 'Solved examples from other documents:'] + examples
    lines += ['', 'Questions:']
    for q, (h, t, cols) in enumerate(questions):
        opts = '; '.join(f"({LETTERS[j]}) {REL_INFO[PIDS[c]]}" for j, c in enumerate(cols))
        lines.append(f"Q{q}: HEAD=E{h} {doc['vertexSet'][h][0]['name']!r} [{doc['vertexSet'][h][0]['type']}] "
                     f"TAIL=E{t} {doc['vertexSet'][t][0]['name']!r} [{doc['vertexSet'][t][0]['type']}] | options: {opts}")
    return '\n'.join(lines)


def call(prompt, client, temperature=0.0, seed=0):
    key = hashlib.sha256(f'{KEY_MODEL}|{temperature}|{seed}|{SYSTEM}|{prompt}'.encode()).hexdigest()
    c = _load_cache()
    if key in c:
        with _lock: SPEND['cached'] += 1
        return c[key]['content']
    if MAX_USD and total_spend_usd() >= MAX_USD:
        raise RuntimeError(f'tope de gasto del juez alcanzado: ${total_spend_usd():.3f} >= ${MAX_USD}')
    extra = {'reasoning_effort': REASONING} if REASONING else {}
    t0 = time.time()
    n429 = 0
    attempt = -1
    while True:
        attempt += 1
        try:
            req = lambda: client.chat.completions.create(
                model=MODEL, temperature=temperature, seed=seed, response_format={'type': 'json_object'},
                messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}], **extra)
            r = _with_wall(req, float(os.environ.get('C1_DI_WALL', '300')))   # límite de reloj: una conexión colgada se abandona y se reintenta
            break
        except Exception as e:
            if '429' in str(e) or 'ate limit' in str(e):      # límite de la API: espera larga con jitter y hasta 12 reintentos propios (no gastan los normales)
                n429 += 1
                if n429 <= 12:
                    attempt -= 1
                    time.sleep(min(120, 5 * 2 ** min(n429, 5)) * (0.5 + random.random())); continue
            if attempt == 5: raise
            time.sleep(2 ** attempt + random.random())
    content = r.choices[0].message.content
    rec = {'key': key, 'content': content, 'in': r.usage.prompt_tokens, 'out': r.usage.completion_tokens, 'lp': None,
           'model': f'deepinfra:{MODEL}', 'reasoning': REASONING or 'default', 'secs': time.time() - t0}
    with _lock:
        SPEND['in'] += rec['in']; SPEND['out'] += rec['out']; SPEND['calls'] += 1
        c[key] = rec
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, 'a') as f: f.write(json.dumps(rec) + '\n')
    return content


def parse(content, questions):
    """-> dict q -> (set(cols elegidas), confianza, dist) ; dist = {} (sin logprobs)"""
    out = {}
    try:
        ans = json.loads(content).get('answers', [])
    except Exception:
        ans = []
    for i, a in enumerate(ans):
        try:
            q = int(a['q']); cols = questions[q][2]
            ch = {cols[LETTERS.index(x.strip().lower()[0])] for x in a.get('choices', []) if x and x.strip().lower()[0] in LETTERS[:len(cols)]}
            out[q] = (ch, int(a.get('confidence', 3)), {})
        except Exception:
            continue
    return out


def route_doc(doc, routed, train_docs, client, n_examples=6, temperature=0.0, seed=0, ex_seed=None):
    """routed: lista de (h, t, cols). Devuelve lista de (h, t, cols, elegidas(set), confianza, ok, dist)."""
    res = []
    for i in range(0, len(routed), MAX_Q):
        qs = routed[i:i + MAX_Q]
        relcols = sorted({c for _, _, cols in qs for c in cols})
        es = seed if ex_seed is None else ex_seed
        ex = examples_qa(train_docs, relcols, k=max(n_examples, 8), seed=es) if n_examples else []
        prompt = build_prompt(doc, qs, ex)
        try:
            content = call(prompt, client, temperature, seed)
        except Exception as e:      # una llamada que agota reintentos no tumba la ejecución: esos pares quedan sin parsear (ok=False)
            if 'tope de gasto' in str(e): raise
            content = ''
            with _lock:
                SPEND['failed'] = SPEND.get('failed', 0) + 1
                if SPEND['failed'] <= 5: print(f'[juez] llamada fallida: {type(e).__name__}: {str(e)[:200]}', file=__import__('sys').stderr, flush=True)
        dec = parse(content, qs)
        for q, (h, t, cols) in enumerate(qs):
            ch, conf, dist = dec.get(q, (set(), 0, {}))
            res.append((h, t, cols, ch, conf, q in dec, dist))
    return res


def run_many(docs, routed_per_doc, train_docs, **kw):
    from openai import OpenAI
    client = OpenAI(base_url='https://api.deepinfra.com/v1/openai', api_key=os.environ['DEEPINFRA_API_KEY'],
                    timeout=float(os.environ.get('C1_DI_TIMEOUT', '600')), max_retries=0)   # cola larga: cortar y reintentar
    workers = int(os.environ.get('C1_DI_CONCURRENCY', '64'))   # hilos simultáneos (límite publicado de DeepInfra: 200 por modelo)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda a: route_doc(a[0], a[1], train_docs, client, **kw) if a[1] else [],
                           zip(docs, routed_per_doc)))


def spend_usd():
    a, b = PRICE['sync']
    return (SPEND['in'] * a + SPEND['out'] * b) / 1e6
