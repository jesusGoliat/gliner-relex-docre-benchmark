"""Prueba de regresión sin GPU, API ni datos externos: regenera A1_A2_summary.csv y A1_A2_bootstrap.csv con src/analyze_test.py a partir de las
métricas y los conteos por documento incluidos en results/test, y comprueba que coinciden byte a byte con los ficheros oficiales.
Uso: python tests/test_reproduce_tables.py"""
import os, sys, shutil, filecmp, subprocess, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = ('A1_A2_summary.csv', 'A1_A2_bootstrap.csv')
with tempfile.TemporaryDirectory() as tmp:
    shutil.copytree(os.path.join(ROOT, 'results', 'test'), os.path.join(tmp, 'test'))
    for f in OUT:
        os.remove(os.path.join(tmp, 'test', f))                     # se regeneran desde cero
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'analyze_test.py')], env=dict(os.environ, RELEX_RUNS_DIR=tmp),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    for f in OUT:
        same = filecmp.cmp(os.path.join(tmp, 'test', f), os.path.join(ROOT, 'results', 'test', f), shallow=False)
        print(f'{f}: {"idéntico al oficial" if same else "DIFIERE"}')
        assert same, f
print('OK: las tablas de J0 frente a B0 se regeneran idénticas desde los conteos incluidos')
