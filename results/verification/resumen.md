# Verificación de reproducibilidad (2026-10-21)

Cada etapa se volvió a ejecutar con el código de este repositorio, en una carpeta de salida aparte (`RELEX_RUNS_DIR`), y se comparó con los ficheros de `results/`.
Entorno local: el mismo que produjo las cifras de §6 (sin `dateparser`). Los logs de las dos primeras filas están en esta carpeta.

| Etapa (script) | Entradas | Resultado de la comparación |
|---|---|---|
| Equivalencia con el evaluador oficial (`tests/test_eval_official.py`) | datos de Re-DocRED | 7 casos, diferencia 0.00 en F1, P, R e Ign F1 (`test_eval_official.log`) |
| Ign F1 de extremo a extremo (`tests/test_eval_e2e_ign.py`) | datos de Re-DocRED | `ign_e2e` = `eval.evaluate` con entidades gold (`test_eval_e2e_ign.log`) |
| Elección de banda (`src/select_band.py`) | `judge_band_selection_D50.csv` | bandas N=10 → (0.1, 0.7), N=50 → (0.05, 0.9) (`select_band.log`) |
| Selección zero-shot (`src/eval_zs_dev.py`) | salida cruda zero-shot | `zs_dev_D50.csv` idéntico byte a byte |
| Pool de A3 (`src/a3_pool.py`) | datos de Re-DocRED | 5 pools y `pool_overlap.csv` idénticos byte a byte |
| Pseudo-etiquetas A3-ctrl (`src/a3_pseudo.py --arm ctrl`) | salidas crudas del pool | 5 ficheros de pseudo-etiquetas y `stats_ctrl` idénticos byte a byte |
| Métricas de test de B0 y Z0 (`src/eval_test.py`) | salidas crudas de test | `baseline_metrics.csv` y conteos por documento idénticos |
| J0 en test (`src/run_judge.py --split test`, 10 configuraciones) | salidas crudas de test + caché del juez | 0 llamadas nuevas al juez; 29 columnas de métricas por fila y conteos por documento idénticos |
| Resumen y bootstrap de J0 (`src/analyze_test.py`) | resultados anteriores | `A1_A2_summary.csv` y `A1_A2_bootstrap.csv` idénticos byte a byte |
| Métricas de A3 (`src/analyze_a3.py`) | salidas crudas de test de A3 | `A3_metrics.csv`, `A3_summary.csv` y `A3_bootstrap.csv` idénticos byte a byte |
| Regeneración de las tablas de J0 sin datos externos (`tests/test_reproduce_tables.py`) | conteos y métricas de `results/test/` | `A1_A2_summary.csv` y `A1_A2_bootstrap.csv` idénticos; ejecutada también en una copia del repositorio sin `data/external` |
| Inferencia en Modal, zero-shot (`modal/relex_modal.py --mode zs_dev`) | GPU A100 | entidades y relaciones idénticas en las 12 configuraciones |
| Inferencia en Modal con los 10 modelos B0 (`--mode infer_dev`) | checkpoints guardados | salidas en D idénticas en los 10 modelos |

**Excepción — pseudo-etiquetas de A3 con juez (`src/a3_pseudo.py --arm judge`).** Regeneradas desde la caché del juez difieren de las usadas para entrenar en 38 documentos de 5 000 (0.23 % de las relaciones): en la generación original, algunas llamadas fallaron por límite de la API y esas preguntas conservaron la decisión del extractor; al regenerar, 38 llamadas nuevas (unos 0.02 USD) las respondieron. Por eso los ficheros de pseudo-etiquetas originales se archivan junto con los artefactos y no se derivan solo de la caché.

**No cubierto:** el ajuste fino (`train_b0`, `train_a3`) no se repitió; no es reproducible al bit (la misma semilla dio diferencias de hasta ±3 F1 entre ejecuciones).
