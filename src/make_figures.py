"""Genera los diagramas y gráficos del README en figures/. Las cifras de los gráficos se leen de results/test/ (no están escritas a mano);
los esquemas (flujo, protocolo, banda del juez, diseño de comparaciones) son figuras explicativas estáticas.
Uso: python src/make_figures.py   (requiere matplotlib, numpy y pandas)"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'figures'); os.makedirs(OUT, exist_ok=True)
plt.rcParams['font.family'] = 'DejaVu Sans'
NAVY, BLUE, GREY, LIGHT, ORANGE, GREEN = '#1F3A5F', '#2E6DB4', '#8A8F98', '#EAF1F8', '#C8741A', '#2E7D4F'
LORANGE, LGREEN, LGREY = '#FDF3E6', '#EAF6EE', '#F2F3F5'


def new(w, h, xl=100, yl=None):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, xl); ax.set_ylim(0, yl or xl * h / w); ax.axis('off'); return fig, ax


def box(ax, x, y, w, h, txt, fc=LIGHT, ec=NAVY, fs=7.2, bold=False, ls=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.0', fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=NAVY, linespacing=ls, fontweight='bold' if bold else 'normal')


def arrow(ax, p1, p2, color=NAVY, ls='-', lw=1.1, style='-|>', rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=9, lw=lw, color=color, linestyle=ls, connectionstyle=f'arc3,rad={rad}'))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches='tight', pad_inches=0.05, facecolor='white'); plt.close(fig)


def rows(path):
    return list(csv.DictReader(open(os.path.join(ROOT, path))))


# ---------------------------------------------------------------- 1. Pipelines
fig, ax = new(10, 5.0, 100, 50)
ax.text(1, 48, 'A. Z0, B0 y J0 — extracción y evaluación (por cada subconjunto de entrenamiento)', fontsize=8.6, fontweight='bold', color=NAVY, va='top')
box(ax, 1, 28, 11, 13, 'Documento\nsin anotar')
arrow(ax, (12.4, 34.5), (15, 34.5))
box(ax, 15.3, 28, 17, 13, 'Extractor\nGLiNER-Relex\n(menciones y\nrelaciones + puntuación)', fs=6.8)
arrow(ax, (32.6, 34.5), (35.2, 34.5))
box(ax, 35.5, 28, 19, 13, 'Menciones → entidades\n(reglas de correferencia)\nS(h,t,r) = máx. sobre\npares de menciones', fs=6.8)
arrow(ax, (54.9, 36), (58.5, 39))
arrow(ax, (54.9, 33), (58.5, 30.5))
box(ax, 58.8, 36, 20, 6.5, 'Z0 / B0:  S ≥ τ', fc=LGREY, ec=GREY, fs=7)
box(ax, 58.8, 25.5, 20, 9.6, 'J0:  S ≥ hi  acepta\nlo ≤ S < hi  → juez LLM\nS < lo  descarta', fc=LORANGE, ec=ORANGE, fs=6.7)
arrow(ax, (79.2, 39), (82.5, 36.5)); arrow(ax, (79.2, 30.5), (82.5, 33.5))
box(ax, 82.8, 28, 16, 13, 'Evaluación\nvs. gold: estricta\ny con tipo\nF1 · Ign F1', fc=LGREEN, ec=GREEN, fs=6.8)
ax.plot([0.5, 99.5], [24, 24], color='#C9D3DF', lw=0.8)
ax.text(1, 22.5, 'B. A3-ctrl y A3 — autoentrenamiento (N = 10), comparten pool, hiperparámetros y pasos', fontsize=8.6, fontweight='bold', color=NAVY, va='top')
box(ax, 1, 4, 13, 12.5, '1 000 documentos\nsin anotar\n(pool; sin solape\ncon dev/test)', fs=6.5)
arrow(ax, (14.4, 10.2), (17, 10.2))
box(ax, 17.3, 4, 14, 12.5, 'B0 (N = 10)\nentidades y S\nsobre el pool', fs=6.8)
arrow(ax, (31.6, 12.5), (35.2, 14.5)); arrow(ax, (31.6, 8), (35.2, 6.5))
box(ax, 35.5, 11.2, 22, 6.2, 'A3: S ≥ hi + juez en la banda', fc=LORANGE, ec=ORANGE, fs=6.7)
box(ax, 35.5, 2.6, 22, 6.2, 'A3-ctrl: S ≥ τ (sin LLM)', fc=LGREY, ec=GREY, fs=6.7)
arrow(ax, (57.9, 14.3), (61.5, 11.5)); arrow(ax, (57.9, 5.7), (61.5, 8.5))
box(ax, 61.8, 4, 20, 12.5, 'Reentrenar desde el\ncheckpoint base con\nN gold + pseudo-\netiquetas', fs=6.7)
arrow(ax, (82.2, 10.2), (84.8, 10.2))
box(ax, 85.1, 4, 14, 12.5, 'Estudiante\n(sin LLM)\n→ evaluación\nen test', fc=LGREEN, ec=GREEN, fs=6.7)
save(fig, 'fig1_pipelines.png')

# ---------------------------------------------------------------- 2. Protocolo de datos
fig, ax = new(10, 3.5, 100, 35)
ax.text(1, 33.6, 'Qué datos se usan para qué: ninguna decisión se toma con el conjunto de test', fontsize=8.8, fontweight='bold', color=NAVY, va='top')
cols = [(1, '1  ENTRENAR', 'Subconjuntos de N = 10 o 50\ndocumentos gold (5 por cada N)\n\n(solo A3) + pool de 1 000 docs\nde train sin etiquetar', LIGHT, NAVY),
        (35, '2  SELECCIONAR', 'D = 50 documentos de dev\n(el mismo para todos los sistemas)\n\ncheckpoint · umbrales · conjunto\nde etiquetas · banda del juez', LORANGE, ORANGE),
        (69, '3  EVALUAR (una vez)', 'Test completo: 500 documentos\n\nconfiguración congelada\nantes de medir', LGREEN, GREEN)]
for x, ttl, txt, fc, ec in cols:
    ax.add_patch(FancyBboxPatch((x, 8.5), 30, 20, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec=ec, lw=1.2))
    ax.text(x + 15, 26, ttl, ha='center', fontsize=8.6, fontweight='bold', color=ec)
    ax.text(x + 15, 16.3, txt, ha='center', va='center', fontsize=7.4, color=NAVY, linespacing=1.35)
arrow(ax, (31.8, 18.5), (34.4, 18.5)); arrow(ax, (65.8, 18.5), (68.4, 18.5))
ax.text(50, 4.6, 'Los 450 documentos de dev restantes no se usan para elegir nada · Z0, B0 y J0 usan como máximo los N gold + D · '
        'el pool de A3 excluye dev/test por título', ha='center', fontsize=6.9, color='#444444')
ax.text(50, 1.0, 'Resultados: media ± desviación sobre 5 subconjuntos · bootstrap pareado por documento (10 000 remuestreos)', ha='center', fontsize=6.9, color='#444444')
save(fig, 'fig3_protocolo.png')

# ---------------------------------------------------------------- 3. Banda del juez
fig, ax = new(10, 3.3, 100, 33)
ax.text(1, 32, 'Cómo decide cada sistema según la puntuación S(h,t,r) del extractor', fontsize=8.8, fontweight='bold', color=NAVY, va='top')
def bar(y, label, segs):
    ax.text(0.5, y + 2.3, label, fontsize=7.6, fontweight='bold', color=NAVY, ha='left', va='center')
    for a, b, c, t, tc, hatch in segs:
        x0, x1 = 22 + a * 74, 22 + b * 74
        ax.add_patch(plt.Rectangle((x0, y), x1 - x0, 4.6, fc=c, ec='white', lw=1.2, hatch=hatch))
        if t: ax.text((x0 + x1) / 2, y + 2.3, t, ha='center', va='center', fontsize=6.8, color=tc, fontweight='bold')
bar(22, 'Z0 / B0', [(0, 0.25, '#D9DCE1', 'rechaza', '#333333', None), (0.25, 0.95, '#C5D3E4', 'umbral τ: rechaza si S < τ, acepta si S ≥ τ', '#1F3A5F', '//'), (0.95, 1, '#9DB4CF', '', '', None)])
ax.text(22 + 0.6 * 74, 28.3, 'τ se elige en D por subconjunto: Z0 usa 0.35; B0 entre 0.25 y 0.95', fontsize=6.5, color='#444444', ha='center')
bar(13.5, 'J0, N = 10', [(0, 0.1, '#D9DCE1', '', '', None), (0.1, 0.7, '#F0B97A', 'el juez LLM decide  (lo = 0.1, hi = 0.7)', '#5A3300', None), (0.7, 1, '#7DBB92', 'acepta', '#0F4A27', None)])
bar(5.5, 'J0, N = 50', [(0, 0.05, '#D9DCE1', '', '', None), (0.05, 0.9, '#F0B97A', 'el juez LLM decide  (lo = 0.05, hi = 0.9)', '#5A3300', None), (0.9, 1, '#7DBB92', 'acepta', '#0F4A27', None)])
ax.text(22 + 0.05 * 74, 8.55, 'descarta', fontsize=5.8, color='#555555', ha='center'); ax.text(22 + 0.05 * 74, 17.0, 'descarta', fontsize=5.8, color='#555555', ha='center')
for v in (0, 0.25, 0.5, 0.75, 1): ax.text(22 + v * 74, 2.6, f'{v:g}', ha='center', fontsize=6.5, color='#555555')
ax.text(22 + 0.5 * 74, 0.2, 'puntuación S del extractor (gris: se descarta, S < lo)', ha='center', fontsize=6.5, color='#555555')
save(fig, 'fig2_banda_juez.png')

# ---------------------------------------------------------------- 4. Resultados (leídos de results/)
j0 = rows('results/test/J0_rows.csv'); a3 = rows('results/test/A3_metrics.csv'); bm = rows('results/test/baseline_metrics.csv')
def per(n, key): return [float(r[key]) for r in sorted([r for r in j0 if r['n'] == str(n)], key=lambda r: int(r['sub']))]
def sys_(name): return [float(r['f1_typed']) for r in sorted([r for r in a3 if r['system'] == name], key=lambda r: int(float(r['sub'])))]
z0 = float([r for r in bm if r['system'] == 'Z0'][0]['f1_typed'])
groups = [('N = 10\nB0  vs  J0', per(10, 'base_f1'), per(10, 'judge_f1')),
          ('N = 50\nB0  vs  J0', per(50, 'base_f1'), per(50, 'judge_f1')),
          ('N = 10 autoentrenamiento\nA3-ctrl  vs  A3', sys_('A3_ctrl'), sys_('A3_judge'))]
fig, ax = plt.subplots(figsize=(7.6, 3.9)); w = 0.36
for i, (lab, a, b) in enumerate(groups):
    for off, vals, col in ((-w / 2, a, GREY), (w / 2, b, BLUE)):
        m, sd = np.mean(vals), np.std(vals, ddof=1)
        ax.bar(i + off, m, w, color=col, zorder=3, alpha=0.95)
        ax.errorbar(i + off, m, yerr=sd, color='black', capsize=3, lw=0.9, zorder=5)
        ax.scatter([i + off] * len(vals), vals, s=14, color='white', edgecolor='black', lw=0.7, zorder=6)
        ax.text(i + off, m / 2, f'{m:.1f}', ha='center', va='center', fontsize=8.5, color='white', fontweight='bold', zorder=4)
    d = np.mean(b) - np.mean(a)
    ax.text(i, max(max(a), max(b)) + 2.2, f'+{d:.2f} F1\n{sum(y > x for x, y in zip(a, b))}/5 subconjuntos', ha='center', fontsize=7.6, color=NAVY, fontweight='bold', linespacing=1.15)
ax.axhline(z0, color=ORANGE, ls='--', lw=1.2, zorder=2)
ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
ax.set_ylim(0, 42); ax.set_ylabel('F1 de extremo a extremo en test (%)', fontsize=8.5, color=NAVY); ax.tick_params(axis='y', labelsize=8)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.grid(axis='y', color='#DDDDDD', lw=0.5, zorder=0)
ax.legend(handles=[Patch(color=GREY, label='sistema base (B0 / A3-ctrl)'), Patch(color=BLUE, label='con juez LLM (J0 / A3)'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', markersize=5, label='un subconjunto'),
                   Line2D([0], [0], color=ORANGE, ls='--', lw=1.2, label=f'Z0 zero-shot: {z0:.1f}')],
          fontsize=7.2, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
ax.set_title('Barras: media; barras de error: desviación típica; puntos: cada subconjunto', fontsize=7, color='#555555', pad=26, loc='left')
save(fig, 'fig4_resultados.png')

# ---------------------------------------------------------------- 5. Diseño de comparaciones
fig, ax = new(10, 3.6, 100, 36)
ax.text(1, 34.8, 'Qué comparaciones son controladas (cambia un solo factor) y cuáles no', fontsize=8.8, fontweight='bold', color=NAVY, va='top')
box(ax, 1, 20, 11, 7, 'Z0', fc=LGREY, ec=GREY, bold=True); box(ax, 28, 20, 14, 7, 'B0  (N = 10)', bold=True); box(ax, 28, 7, 14, 7, 'B0  (N = 50)', bold=True)
box(ax, 60, 20, 14, 7, 'J0  (N = 10)', fc=LORANGE, ec=ORANGE, bold=True); box(ax, 60, 7, 14, 7, 'J0  (N = 50)', fc=LORANGE, ec=ORANGE, bold=True)
box(ax, 85, 20, 14, 7, 'A3-ctrl', fc=LGREY, ec=GREY, bold=True); box(ax, 85, 7, 14, 7, 'A3', fc=LORANGE, ec=ORANGE, bold=True)
arrow(ax, (42.6, 23.5), (59.6, 23.5), color=GREEN, lw=2.4); arrow(ax, (42.6, 10.5), (59.6, 10.5), color=GREEN, lw=2.4)
arrow(ax, (92, 19.6), (92, 14.4), color=GREEN, lw=2.4)
ax.text(51.2, 25.1, 'añade el juez\n(mismo extractor)', ha='center', fontsize=6.7, color=GREEN, fontweight='bold', linespacing=1.1)
ax.text(51.2, 12.1, 'añade el juez\n(mismo extractor)', ha='center', fontsize=6.7, color=GREEN, fontweight='bold', linespacing=1.1)
ax.text(90.6, 17, 'solo cambia la fuente de\nlas pseudo-etiquetas', ha='right', va='center', fontsize=6.4, color=GREEN, fontweight='bold', linespacing=1.1)
arrow(ax, (12.6, 23.5), (27.6, 23.5), color=GREY, ls=(0, (4, 3)), lw=1.2)
ax.text(20, 25.1, 'cambia ajuste\ny etiquetas', ha='center', fontsize=6.3, color='#555555', linespacing=1.1)
arrow(ax, (35, 14.4), (35, 19.6), color=GREY, ls=(0, (4, 3)), lw=1.2, style='<|-|>')
ax.text(33.4, 17, 'cambia N y el lote', ha='right', va='center', fontsize=6.3, color='#555555')
ax.text(50, 1.4, 'B0 y los estudiantes A3 se entrenaron con hiperparámetros distintos: no se comparan entre sí', ha='center', fontsize=6.6, color='#555555')
ax.legend(handles=[Line2D([0], [0], color=GREEN, lw=2.4, label='comparación controlada: sostiene conclusiones'),
                   Line2D([0], [0], color=GREY, lw=1.2, ls=(0, (4, 3)), label='cambia más de un factor: solo descriptiva')],
          fontsize=6.8, loc='upper left', frameon=False, bbox_to_anchor=(0.0, 0.93))
save(fig, 'fig5_comparaciones.png')
print('figuras generadas en', OUT, sorted(os.listdir(OUT)))
