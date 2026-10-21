# Extracción de relaciones a nivel de documento con GLiNER-Relex en Re-DocRED (pocos datos, extremo a extremo)

Registro oficial de la evaluación: código, protocolo, métricas y resultados finales.

Atribución y licencias de terceros (datos de Re-DocRED incluidos): `NOTICE.md`.

## 1. Visión general

**Objetivo.** Medir la extracción de relaciones a nivel de documento de extremo a extremo (texto → entidades → relaciones) con **GLiNER-Relex** cuando solo hay **N = 10** o **N = 50** documentos de entrenamiento etiquetados, y medir el efecto de un **juez LLM** que vuelve a decidir únicamente los candidatos *dudosos* del extractor, así como el de usar ese juez para etiquetar documentos no etiquetados y reentrenar (autoentrenamiento).

**Benchmark.** Re-DocRED (Tan et al., EMNLP 2022): 96 relaciones de Wikidata, 6 tipos de entidad (PER, ORG, LOC, TIME, NUM, MISC); train 3 053 / dev 500 / test 500 documentos. Los conjuntos de entrenamiento con pocos datos son los 5 subconjuntos publicados por cada N (`data/subsets/redocred/train_N{10,50}_s{0..4}.json`). Todas las cifras se miden en el **split de test completo (500 documentos)**.

**Pipelines de la evaluación oficial**

| Id | Pipeline | LLM en inferencia | Datos |
|---|---|---|---|
| **Z0** | GLiNER-Relex large-v1.0 sin entrenamiento adicional (zero-shot) | no | ninguno |
| **B0** | GLiNER-Relex ajustado con los N documentos gold | no | N documentos gold |
| **J0** | B0 + juez LLM sobre los candidatos dudosos | sí (`openai/gpt-oss-120b`) | N documentos gold (también ejemplos del juez) |
| **A3-ctrl** | Estudiante entrenado con N gold + 1 000 documentos sin etiquetar con pseudo-etiquetas del propio extractor (sin LLM) | no | N=10 gold + pool sin etiquetas |
| **A3** | Estudiante entrenado con N gold + los mismos 1 000 documentos con pseudo-etiquetas del juez | no (el LLM solo se usa para etiquetar el pool) | N=10 gold + pool sin etiquetas |

GLiNER-Relex no produce correferencia; las entidades se construyen agrupando menciones predichas con reglas. Por ello todas las cifras son de **extremo a extremo**: incluyen detección de menciones, tipado, agrupación de entidades y clasificación de relaciones. No son comparables con cifras publicadas que parten de entidades gold.

## 2. Pipelines evaluados

Componentes comunes (`src/`):

1. **Extractor** (`modal/relex_modal.py`): `knowledgator/gliner-relex-large-v1.0` mediante `gliner==0.2.29` (`UniEncoderSpanRelexGLiNER`). Cada documento entra completo como un único texto (tokens unidos por espacios; el `max_len` del modelo es 2 048 palabras, el documento más largo de dev tiene 481 tokens y el de test 510). Devuelve spans de entidad (offsets de carácter, etiqueta, puntuación) y relaciones entre spans (`head`, `tail`, nombre de la relación, puntuación) para los 96 nombres de relación (`rel_info.json`).
2. **Agregación mención → entidad** (`relex_post.py`, `ent_cluster.py`, `ent_build.py`): los spans se convierten a posiciones de token de Re-DocRED (`relex_io.span_tokens`); las menciones solapadas dentro de una oración se aplanan (gana la de mayor puntuación, luego la más larga); las menciones se agrupan en entidades con correferencia por reglas (nombre normalizado, alias de personas, acrónimos/contención, fechas, números); el tipo de la entidad es el argmax ponderado por puntuación. La puntuación de una relación entre entidades es **S[h, t, r] = máximo sobre los pares de menciones** de la puntuación de la relación entre spans.
3. **Decisión**: Z0 y B0 predicen cada (h, t, r) con `S[h, t, r] ≥ rt`.

**Z0** — zero-shot. Etiquetas de entidad: 23 descriptores finos asignados a los 6 tipos (`relex_io.V3_TYPES`, conjunto "v3"); relaciones: los 96 nombres. Configuración: umbral de entidad 0.3, `flat_ner=True`, `adjacency_threshold=0.5`, umbral de relación `rt = 0.35`.

**B0** — ajustado. Ejemplos de entrenamiento (`relex_io.to_train_relex`): un span NER por mención gold (6 etiquetas: `person, organization, location, time, number, miscellaneous`) y, por cada tripleta gold, una relación entre **cada par** de menciones cabeza/cola. Hiperparámetros (iguales en todas las ejecuciones de B0): tasa de aprendizaje 2e-5 (constante, sin warm-up), bf16, semilla 42, tamaño de lote 5 (N=10) / 10 (N=50), partiendo del checkpoint preentrenado, en una H100 por cada (N, subconjunto). Se guardan checkpoints en pasos fijos (N=10: 20, 40, 80, 160, 320, 480; N=50: 25, 50, 100, 200, 400, 800); el checkpoint, el umbral de entidad ∈ {0.3, 0.5} y el umbral de relación `rt` ∈ {0.10, 0.15, …, 0.95} se eligen por subconjunto en el conjunto de selección D (ver §3) maximizando la métrica principal (empates: checkpoint más temprano, luego umbrales más bajos). Los valores elegidos están en `results/test/baseline_metrics.csv` (`rt`, `ent_thr`) y en `results/selection/ftB_baseline_dev_D50.json`.

**J0** — B0 + juez (`relex_router.py`, `llm_router.py`, `run_judge.py`). Para cada documento y cada par ordenado de entidades, los candidatos son las relaciones con `S ≥ lo` cuyo par de tipos (cabeza, cola) permite la tabla dominio/rango de Menotti et al. (`DocRE-StructuralConsistency`):

* `S ≥ hi` → se acepta directamente;
* `lo ≤ S < hi` → se envía al juez (como máximo 5 opciones por par, ordenadas por puntuación);
* `S < lo` → se descarta.

El juez (`openai/gpt-oss-120b` en DeepInfra, `reasoning_effort=medium`, temperatura 0, semilla 0, salida JSON) recibe el documento con identificadores de oración, las entidades implicadas, **hasta 8 ejemplos resueltos tomados de los N documentos de entrenamiento** (candidatos verdaderos y rechazados) y hasta 30 preguntas por llamada; para cada pregunta elige cualquier subconjunto de opciones o ninguna (prompt de sistema `SYSTEM_V3` en `llm_router.py`). La fusión es *dura*: en los pares enviados, la predicción es exactamente la elección del juez; si una respuesta no se puede interpretar (o la llamada falla), para esas opciones se conserva la decisión del extractor (`S ≥ rt`). Bandas, elegidas en D con el juez real: **N=10 → [0.1, 0.7)**, **N=50 → [0.05, 0.9)**. El extractor es exactamente B0 (mismos checkpoints y umbrales).

**A3-ctrl y A3** — autoentrenamiento (`a3_pool.py`, `a3_pseudo.py`, `modal/relex_modal.py`), solo N=10. Para cada subconjunto k se toma un **pool de 1 000 documentos de `train_revised` sin etiquetas** (semilla `default_rng(500 + k)`), excluyendo los 10 documentos de entrenamiento de ese subconjunto y todos los de dev/test por título. B0 (N=10, mismo subconjunto) se aplica al pool y produce las entidades predichas y S. Pseudo-etiquetas:

* **A3-ctrl**: las relaciones del propio extractor, `S ≥ rt` (el umbral elegido en D para ese subconjunto de B0). No interviene ningún LLM.
* **A3**: `S ≥ hi` más las opciones elegidas por el juez en la banda [0.1, 0.7) — mismo juez, prompt, banda y fusión que J0 — con 1 voto (temperatura 0) y el mismo respaldo ante respuestas no interpretables.

Los documentos pseudo-etiquetados usan las entidades predichas (relaciones expandidas a todos los pares de menciones, como en B0). El estudiante se entrena **desde el checkpoint preentrenado** con los N documentos gold + los 1 000 pseudo-etiquetados: tasa de aprendizaje 2e-5 (constante), bf16, semilla 42, tamaño de lote 10; checkpoints en {100, 200, 400, 800, 1600, 3200} pasos; checkpoint y umbrales elegidos en D con la misma regla que B0. Ambos brazos usan el mismo pool, los mismos hiperparámetros y los mismos pasos; solo difieren las pseudo-etiquetas. No se aplica rebalanceo de clases. En inferencia ninguno de los dos usa LLM.

## 3. Metodología de evaluación

**Datos y particiones**

* Entrenamiento: `train_N{10,50}_s{0..4}` (5 subconjuntos por N).
* **Conjunto de selección D**: 50 documentos de dev, `numpy.random.default_rng(1234).permutation(500)[:50]` (ordenados). **Todas** las decisiones (umbrales, checkpoint, conjunto de etiquetas, banda del juez) se tomaron solo en D; el mismo D se usa para todos los pipelines.
* **Test**: los 500 documentos de test; cada configuración congelada se evaluó en test; no se tomó ninguna decisión con test.
* Supervisión igual: Z0, B0 y J0 usan como máximo los N documentos gold de entrenamiento más D para seleccionar. A3-ctrl y A3 usan además documentos de train **sin etiquetas**.
* Control de memoria del LLM: el juez se volvió a ejecutar en los 100 primeros documentos de test con las menciones PER/ORG/LOC/MISC sustituidas por `ENT<i>` en el prompt (TIME/NUM se mantienen), para N=10 (5 subconjuntos) y N=50 (subconjuntos 0 y 1).

**Protocolo.** Para cada pipeline: se obtienen las salidas crudas del modelo (Modal), se construyen offline los documentos predichos y S, se aplica la configuración congelada y se puntúa contra el gold. Estadística: media ± desviación típica sobre los 5 subconjuntos; **bootstrap pareado por documento** (10 000 remuestreos, semilla 0; los mismos documentos remuestreados en los 5 subconjuntos y el estadístico es la media sobre subconjuntos de la diferencia de F1). Los subconjuntos comparten dev/test, por lo que los intervalos reflejan el muestreo de documentos y no la variabilidad entre subconjuntos de entrenamiento.

**Criterios de comparación.** Los pipelines se comparan sobre los mismos documentos de test, con los mismos checkpoints del extractor (J0 frente a B0), el mismo servidor, modelo y prompt del juez, y la métrica principal descrita abajo. A3 se compara con A3-ctrl, que tiene los mismos datos, hiperparámetros y pasos. No se comparan cifras con otras publicaciones.

**Consideraciones necesarias para interpretar los resultados**

* Los valores elegidos en dev son optimistas: B0 con N=10 promedia 25.9 de F1 en D (columna `dev_f1` de `baseline_metrics.csv`) frente a 23.6 en test. Solo se reportan como resultados las cifras de test.
* Las respuestas del juez se guardan en caché por hash del prompt. Las cifras de J0 en test se obtuvieron con la caché del conjunto completo de llamadas; las llamadas que habían fallado por límite de la API (HTTP 429) en una pasada parcial previa de la misma configuración se volvieron a emitir. La tasa mínima de respuestas válidas en las diez ejecuciones de J0 es 98.8 % (`valid_rate` en `results/test/J0_rows.csv`).
* El juez usa la tabla dominio/rango de Menotti solo como filtro de candidatos, y sus ejemplos provienen de los N documentos de entrenamiento.
* `adjacency_threshold` no tiene efecto en la salida de `gliner==0.2.29` en la rejilla zero-shot (puntuaciones idénticas para 0.3, 0.5 y 0.65 en `results/selection/zs_dev_D50.csv`); se fija en 0.5.
* Las pseudo-etiquetas de A3 se generan con una sola votación (temperatura 0). Las pseudo-etiquetas con juez usadas para entrenar proceden de una generación en la que algunas llamadas fallaron por límite de la API y conservaron la decisión del extractor; regenerarlas desde la caché difiere en 38 de 5 000 documentos (0.23 % de las relaciones), por lo que los ficheros originales deben conservarse (`results/verification/resumen.md`). El solapamiento de tripletas entre el pool y test es del 7.3–7.9 % (`results/pseudo_labels/pool_overlap.csv`); por eso se reporta además el recall sobre tripletas "no vistas" (ver §4).

## 4. Métricas

Todas las métricas de tripletas son **micro-promedios sobre el split de test** y se expresan en %.

* **Métrica principal — F1 de tripletas de extremo a extremo, estricto y con tipo** (`eval_e2e.re_e2e(mode='strict', typed=True)`). Una entidad predicha empareja con una entidad gold si sus conjuntos de menciones `{(sent_id, inicio, fin)}` son idénticos **y** los tipos coinciden (uno a uno; los duplicados se emparejan en orden). Una tripleta predicha (h, r, t) es correcta si ambas entidades emparejan con entidades gold y (h_gold, r, t_gold) es una tripleta gold; los duplicados tras el emparejamiento cuentan una vez; toda otra predicción es un falso positivo; el denominador del recall es **todas** las tripletas gold (incluidas aquellas cuyas entidades no se predijeron).
  P = TP / #pred, R = TP / #gold, F1 = 2PR / (P + R).
* **F1 estricto sin tipo**: igual sin la condición de tipo. **F1 blando con tipo**: emparejamiento por Jaccard de los conjuntos de menciones ≥ 0.5 con asignación húngara uno a uno, exigiendo tipo.
* **Ign F1** (`eval_e2e.ign_e2e`): como en el script oficial de Re-DocRED, los aciertos cuya terna (nombre de mención cabeza, nombre de mención cola, relación) aparece en el **train completo** (`train_revised.json`, 3 053 documentos) se descuentan de la precisión: `P_ign = (TP − TP_train) / (#pred − TP_train + 1e-5)`; el recall no cambia; `Ign F1 = 2·P_ign·R / (P_ign + R)`.
* **Recall sobre tripletas no vistas** (`analyze_a3.unseen_recall`, solo A3): recall sobre las tripletas gold de test cuya terna (nombre cabeza, nombre cola, relación) no aparece en las tripletas gold del pool de ese subconjunto.
* **Métricas de entidad** (`entity_metrics`): F1 de menciones con tipo (`mention_typed_f1`), F1 estricto de clusters (`cluster_strict_f1`) y exactitud de tipo de las entidades emparejadas (`type_acc`).
* **Verificación contra el evaluador oficial.** Con entidades gold, `eval.evaluate` reproduce `official_evaluate` del repositorio de Re-DocRED (`evaluation.py`) en F1, P, R e Ign F1; `tests/test_eval_official.py` comprueba 7 casos (gold, subconjuntos, ruido, aleatorio, duplicados, vacío; con y sin hechos de train) con tolerancia 1e-9, y `tests/test_eval_e2e_ign.py` comprueba que `ign_e2e` coincide con `eval.evaluate` cuando las entidades son gold. Ambos se ejecutaron con el código de este repositorio y pasaron todos los casos (logs en `results/verification/`).

Justificación: el F1 micro y el Ign F1 son las medidas estándar de DocRE (Yao et al., 2019; Tan et al., 2022); el emparejamiento de extremo a extremo y con tipo es necesario porque el extractor predice sus propias entidades, y el emparejamiento estricto con tipo evita acreditar relaciones asociadas a entidades de tipo incorrecto o con límites erróneos (la variante blanda se reporta para mostrar el efecto de los errores de límites y de agrupación).

## 5. Entorno experimental

| Componente | Valor |
|---|---|
| Análisis local / cliente del juez | Python 3.12.3; numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, openai 2.41.1, modal 1.6.2, scikit-learn 1.8.0 (solo lo necesita el evaluador oficial de Re-DocRED en las pruebas) (`requirements.txt`); 4 núcleos de CPU, ≈7.6 GiB de RAM, sin GPU. En este entorno **no** estaba instalado `dateparser` |
| Contenedor de Modal | Python 3.11.12 (`debian_slim`); versiones exactas en `requirements-modal.txt`, entre ellas `gliner==0.2.29`, `torch==2.14.1` (CUDA 13.0), `transformers==5.16.1`, `accelerate==1.15.0`, `numpy==2.2.6`, `scipy==1.17.1`, `safetensors==0.8.0`, `tokenizers==0.23.3`, `dateparser==1.4.3` (la imagen fija con `==` solo `gliner` y `dateparser`; el resto son las versiones que resolvió `pip` al construirla) |
| GPU (Modal) | NVIDIA A100-SXM4-40GB (inferencia), NVIDIA H100 80GB HBM3 (ajuste fino) |
| Modelo | `knowledgator/gliner-relex-large-v1.0`, revisión `4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290` (la descargada al construir la imagen; `modal/inspect_artifacts.py` la lee de la imagen; coincide con la revisión actual de Hugging Face, sin modificaciones desde 2026-03-22) |
| Juez | API de DeepInfra compatible con OpenAI, `openai/gpt-oss-120b`, `reasoning_effort=medium`, temperatura 0, semilla 0, `response_format=json_object`; límite de reloj de 300 s por llamada; hasta 12 reintentos con espera ante HTTP 429; hasta 200 llamadas simultáneas por proceso (`C1_DI_CONCURRENCY`) |
| Datos | Re-DocRED, commit `ccfb54f5ddf5836027c87badda10f6dfc56efaac`; DocRE-StructuralConsistency, commit `ee9de0f8cf9f2df3b061d72e13568d6361cc8dd0` (`data/download.sh`) |
| Fechas | todas las ejecuciones: 2026-10-20 / 2026-10-21 |

**Nota sobre el entorno.** La elección de checkpoints y umbrales de B0 se hizo en el contenedor de Modal, que tiene `dateparser` instalado (lo usa `ent_cluster.time_key` para agrupar fechas), mientras que todas las evaluaciones offline de §6 se ejecutaron en el entorno local, que no lo tenía. Recalculadas con `dateparser==1.4.3`, las métricas de test de Z0, B0 y J0 difieren como máximo 0.03 F1 por subconjunto y el efecto del juez no cambia (+2.23 → +2.23 con N=10; +2.07 → +2.08 con N=50); ver `results/robustness_dateparser/`.

## 6. Resultados finales

Split de test, 500 documentos; F1 = F1 de tripletas de extremo a extremo, estricto y con tipo (%). Media ± desviación típica sobre los 5 subconjuntos de entrenamiento. Fuente: `results/test/*.csv`.

### 6.1 Z0, B0 y J0

| Pipeline | N | F1 | Ign F1 | Precisión | Recall | F1 sin tipo | F1 blando con tipo |
|---|---|---|---|---|---|---|---|
| Z0 (zero-shot) | – | 16.84 | 15.63 | 18.60 | 15.38 | 18.29 | 27.33 |
| B0 | 10 | 23.62 ± 2.01 | 22.36 ± 1.88 | 25.93 ± 3.03 | 21.75 ± 1.43 | 25.16 ± 2.00 | 33.94 ± 3.10 |
| **J0** | 10 | **25.85 ± 1.62** | 24.42 ± 1.42 | 26.00 ± 2.86 | 26.18 ± 3.34 | 27.56 ± 1.71 | 36.99 ± 2.63 |
| B0 | 50 | 31.34 ± 0.71 | 29.87 ± 0.44 | 33.74 ± 2.16 | 29.50 ± 2.48 | 32.85 ± 1.19 | 42.97 ± 0.96 |
| **J0** | 50 | **33.41 ± 0.15** | 31.68 ± 0.32 | 31.82 ± 2.10 | 35.45 ± 2.46 | 35.04 ± 0.41 | 45.48 ± 0.21 |

**Efecto del juez (J0 − B0)**, bootstrap pareado por documento (`A1_A2_bootstrap.csv`):

| N | ΔF1 | IC 95 % | Δ Ign F1 | Subconjuntos con J0 > B0 |
|---|---|---|---|---|
| 10 | +2.23 | [1.88, 2.60] | +2.06 | 5 / 5 |
| 50 | +2.07 | [1.64, 2.52] | +1.81 | 5 / 5 |

(ninguno de los 10 000 remuestreos tuvo una diferencia no positiva). La ganancia procede del recall (+4.4 / +6.0); la precisión no cambia para N=10 y baja 1.9 puntos para N=50.

**Por subconjunto (F1, test)** — de `results/test/J0_rows.csv`:

| N | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| 10: B0 → J0 | 24.44 → 27.26 | 20.38 → 23.59 | 23.42 → 24.82 | 25.79 → 27.34 | 24.07 → 26.26 |
| 50: B0 → J0 | 31.55 → 33.18 | 31.56 → 33.36 | 31.60 → 33.50 | 30.10 → 33.55 | 31.91 → 33.49 |

**Control de memoria del LLM** (100 primeros documentos de test; ΔF1 = juez − B0; `results/masking_control/judge_mask.csv`):

| N / subconjunto | 10/s0 | 10/s1 | 10/s2 | 10/s3 | 10/s4 | 50/s0 | 50/s1 |
|---|---|---|---|---|---|---|---|
| nombres visibles | +2.61 | +2.86 | +0.79 | +0.27 | +1.39 | +0.48 | +1.09 |
| nombres enmascarados | +0.41 | +1.37 | +0.66 | +0.57 | +1.66 | +1.22 | −0.74 |

Con 100 documentos las estimaciones son ruidosas; la ganancia del juez se reduce pero no se elimina al ocultar los nombres de las entidades en N=10 (media de las cinco filas: +1.58 visibles frente a +0.93 enmascarados).

### 6.2 Autoentrenamiento (A3), N = 10

| Pipeline | F1 | Ign F1 | Precisión | Recall | F1 sin tipo | F1 blando con tipo | Recall en tripletas no vistas |
|---|---|---|---|---|---|---|---|
| A3-ctrl (pseudo-etiquetas del extractor, sin LLM) | 24.11 ± 1.61 | 22.82 ± 1.48 | 26.89 ± 3.67 | 22.11 ± 2.03 | 25.51 ± 1.59 | 35.07 ± 2.37 | 21.72 ± 1.93 |
| **A3** (pseudo-etiquetas del juez) | **26.53 ± 1.54** | 25.15 ± 1.30 | 28.60 ± 1.22 | 24.96 ± 3.09 | 27.92 ± 1.72 | 37.89 ± 2.05 | 24.62 ± 2.98 |

**A3 − A3-ctrl** (`A3_bootstrap.csv`): ΔF1 = **+2.42**, IC 95 % [1.99, 2.86]; ningún remuestreo de 10 000 con diferencia no positiva. F1 por subconjunto (test):

| | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| A3-ctrl | 24.64 | 21.49 | 23.89 | 25.74 | 24.79 |
| A3 | 26.66 | 24.18 | 26.21 | 27.22 | 28.39 |

Calidad de las pseudo-etiquetas del pool frente a su gold (solo análisis, nunca usada para seleccionar; `results/pseudo_labels/stats_*_P1000.csv`): A3-ctrl precisión 19.2–25.8 %, recall 23.2–27.3 %; A3 precisión 19.6–26.6 %, recall 23.9–32.3 %. Respuestas del juez no interpretables al etiquetar el pool: 1.0–2.1 % de las preguntas.

## 7. Decisiones técnicas

* **Emparejamiento estricto, con tipo y de extremo a extremo** como métrica principal; el blando se reporta como vista secundaria de los errores de límites/agrupación.
* **Puntuación de relación a nivel de entidad = máximo sobre pares de menciones**; **las relaciones de entrenamiento se expanden a todos los pares de menciones** de cada tripleta gold, de forma coherente con esa agregación.
* **Conjuntos de etiquetas**: zero-shot usa 23 descriptores finos asignados a los 6 tipos (v3), elegido en D; los modelos ajustados usan los 6 nombres simples (el modelo los aprende de los datos).
* **Banda del juez y top-k explícitos**; el número de candidatos recortados por top-k se registra (`trunc` en `J0_rows.csv`: 16 de 32 657 candidatos para N=10/s4, 45 de 36 320 para N=50/s0, 2 de 22 913 para N=50/s4, 0 en el resto). El filtro de tipos (tabla de Menotti) se aplica antes del enrutado.
* **Respaldo ante salida inválida del juez**: se conserva la decisión del extractor (ningún par se pierde por un fallo de interpretación).
* **Un único servidor y prompt del juez** en todas las comparaciones; temperatura 0; ejemplos solo de los N documentos de entrenamiento.
* **Selección solo en D** (50 documentos de dev, igual para todos los pipelines); empates resueltos hacia el checkpoint más temprano y los umbrales más bajos; el test está protegido en el código (`ALLOW_TEST=1`).
* **Umbral de Z0**: el umbral de relación 0.35 se eligió en D; el barrido fino (pasos de 0.05) que lo respalda está en `results/selection/zs_labels_fine_D50.csv` (F1 en D = 15.00 para `v3`, umbral de entidad 0.3, umbral de relación 0.35). La constante `dev_f1 = 15.0` de Z0 en `baseline_metrics.csv` proviene de esa selección.
* **Extremo de la rejilla**: para N=50/s3 el umbral de relación elegido es el mayor valor de la rejilla (0.95).
* **Autoentrenamiento**: pseudo-etiquetas con un voto, sin rebalanceo de clases; el pool se filtra de dev/test por título y el solapamiento de tripletas se mide y reporta en lugar de filtrarse.

## 8. Limitaciones conocidas

* **Incertidumbre.** Los intervalos de confianza (bootstrap por documento) miden solo la variabilidad por muestreo de documentos de test, condicionada a los modelos ya entrenados. No incluyen la variabilidad de volver a entrenar: con la misma semilla se observaron diferencias de hasta ≈3 F1 entre ejecuciones del ajuste fino, y no se entrenó con varias semillas. Los 5 subconjuntos comparten el mismo test.
* **Selección con 50 documentos.** Checkpoint, umbrales, etiquetas y banda se eligieron en D (50 documentos) entre muchas configuraciones candidatas; las cifras en D son optimistas (B0 con N=10: 25.9 en D frente a 23.6 en test). Evaluar en un test independiente evita trasladar ese optimismo a las cifras finales, pero no se estudió cuánto cambiaría la configuración elegida con otros 50 documentos.
* **Reproducibilidad del entrenamiento.** Las métricas se reproducen exactamente a partir de las salidas guardadas (`results/verification/resumen.md`), pero el ajuste fino no es reproducible al bit y no se repitió. Los checkpoints solo existen en el volumen de Modal donde se entrenaron (sus sha256 están en `artifacts/CHECKPOINTS.csv`).
* **Control de memoria del LLM.** Se aplicó a los 100 primeros documentos de test (no una muestra aleatoria) y, para N=50, a 2 subconjuntos. Solo se enmascaran las menciones del documento evaluado; los ejemplos resueltos del prompt (tomados de los N documentos de entrenamiento) no se enmascaran. Por tanto no demuestra que el juez sea independiente del conocimiento factual previo.
* **Autoentrenamiento (A3).** Solo N=10, un único pool de 1 000 documentos y una sola votación. El pool comparte con test el 7.3–7.9 % de las tripletas (por nombres). La ganancia de recall de A3 sobre A3-ctrl es similar en el conjunto completo (+2.85) y en las tripletas no vistas (+2.90), lo que no apunta a que se concentre en hechos repetidos, pero no se separó el F1 entre tripletas vistas y no vistas.
* **Dependencia de la agrupación de entidades.** El F1 estricto está limitado por la correferencia por reglas: el F1 estricto de clusters de entidades en test es 72–80 % (columna `cluster_strict_f1` de `baseline_metrics.csv`).
* **Hiperparámetros.** Tasa de aprendizaje, tamaño de lote y semilla se fijaron a priori y no se exploraron; solo se seleccionaron checkpoint y umbrales.
* **Entorno.** Diferencia entre el entorno de selección y el de evaluación respecto a `dateparser`: ver la nota de §5.

## 9. Estructura del repositorio

```
README.md
NOTICE.md                    atribución y licencia de los datos de terceros incluidos
requirements.txt             entorno local (versiones verificadas)
requirements-modal.txt       dependencias exactas del contenedor de Modal y revisión del modelo base
data/
  download.sh                clona Re-DocRED y DocRE-StructuralConsistency en los commits evaluados
  subsets/redocred/          train_N{10,50}_s{0..4}.json — los 10 conjuntos de entrenamiento con pocos datos
modal/relex_modal.py         funciones de Modal: inferencia, ajuste fino, selección de checkpoint/umbrales en D (modos en su docstring)
modal/inspect_artifacts.py   registra las versiones del contenedor y el sha256 de los checkpoints del volumen de Modal
src/
  data.py                    rutas, carga de Re-DocRED, mapas de las 96 relaciones; RUNS = directorio de salida de las ejecuciones nuevas (nunca results/)
  relex_io.py                conjuntos de etiquetas (v0, v1, v3), documento → texto, offsets token/carácter, ejemplos de entrenamiento
  relex_post.py              salida cruda del modelo → documento predicho y S[h,t,r]
  ent_cluster.py, ent_build.py   agrupación por reglas mención → entidad y construcción del documento predicho
  constraints.py             tabla de tipos dominio/rango (Menotti et al.) usada como filtro de candidatos
  eval.py                    métricas con entidades gold (reimplementación del script oficial)
  eval_e2e.py                métricas de extremo a extremo: emparejamiento estricto/blando, con tipo, Ign F1, métricas de entidad, conteos por documento
  ft_select.py               barrido de selección en D (umbrales de entidad × relación), regla de desempate
  runs.py                    carga las salidas crudas de B0 en (docs gold, docs predichos, S, configuración elegida)
  relex_router.py            banda del juez (aceptado / enviado / descartado), fusión dura con respaldo, enmascarado de entidades
  llm_router.py              juez: prompt, ejemplos, cliente de DeepInfra, caché, límite de tasa, tope de gasto
  run_judge.py               ejecuciones del juez: selección de banda (dev), evaluación en test y control de enmascarado
  collect_judge_rows.py      líneas JSON de run_judge → CSV
  select_band.py             regla de elección de banda en D
  eval_zs_dev.py, eval_zs_labels.py, eval_zs_labels_fine.py   tablas de selección zero-shot en D
  eval_test.py               métricas de test de B0 y Z0
  compare_dateparser.py      compara las métricas de test recalculadas con dateparser frente a las oficiales
  analyze_test.py            tablas J0 frente a B0 y bootstrap pareado
  a3_pool.py                 pool sin etiquetar de A3 y solapamiento de tripletas con dev/test
  a3_pseudo.py               pseudo-etiquetas de A3-ctrl y A3 y su calidad
  analyze_a3.py              métricas de test de A3 y A3-ctrl, recall en tripletas no vistas y bootstrap
tests/                       pruebas de equivalencia de las métricas (script oficial; Ign F1) y de regeneración de las tablas de J0
artifacts/MANIFEST.csv       manifiesto (tamaño y sha256) de los artefactos que se distribuyen aparte
artifacts/CHECKPOINTS.csv    sha256 de los ficheros de los 20 checkpoints entrenados (B0 y A3), que viven en el volumen de Modal
results/                     RESULTADOS OFICIALES (solo lectura; el código nunca escribe aquí)
  test/                      baseline_metrics.csv (Z0, B0), J0_rows.csv, J0_counts_*.npz y baseline_counts.npz (conteos por documento),
                             A1_A2_summary.csv, A1_A2_bootstrap.csv, A3_metrics.csv, A3_summary.csv, A3_bootstrap.csv
  selection/                 tablas de selección en D (ver §6 y §7), incluidas las de A3
  pseudo_labels/             solapamiento pool/dev/test y calidad de las pseudo-etiquetas (P=1 000)
  robustness_dateparser/     métricas de test recalculadas con dateparser instalado y su comparación con las oficiales (§5)
  masking_control/judge_mask.csv
  verification/              logs y resumen de la verificación de reproducibilidad
```

## 10. Uso

**Verificación de reproducibilidad realizada:** `results/verification/resumen.md` (qué etapas se volvieron a ejecutar y con qué resultado).

**Consultar los resultados oficiales** (sin ejecutar nada): `results/test/` (cifras finales), `results/selection/` (cómo se eligieron las configuraciones), `results/pseudo_labels/`, `results/masking_control/`, `results/robustness_dateparser/`.

**Recalcular las tablas resumen de J0 a partir de los conteos por documento incluidos** (sin experimentos nuevos; no llama a ningún modelo ni API; no necesita los datos externos). La prueba `python tests/test_reproduce_tables.py` lo hace en un directorio temporal y comprueba que el resultado es idéntico al oficial:

```bash
pip install -r requirements.txt
mkdir -p runs && cp -r results/test runs/test        # el código escribe en runs/, nunca en results/
python src/analyze_test.py                            # regenera A1_A2_summary.csv y A1_A2_bootstrap.csv en runs/test
```

**Comparar el efecto de `dateparser`** (tras recalcular con él en `runs/test`): `python src/compare_dateparser.py`.

**Comprobar la implementación de las métricas contra el evaluador oficial** (requiere los datos):

```bash
bash data/download.sh
python tests/test_eval_official.py
python tests/test_eval_e2e_ign.py
```

**Volver a ejecutar la evaluación** (ejecuciones nuevas: requiere una cuenta de Modal con crédito de GPU y una clave de DeepInfra en `.env`; los resultados se escriben en `runs/` o en `$RELEX_RUNS_DIR`). Orden de las etapas:

1. `bash data/download.sh`.
2. Selección zero-shot: `modal run modal/relex_modal.py --mode zs_dev` y `--mode zs_labels`; después `python src/eval_zs_dev.py`, `python src/eval_zs_labels.py` y `python src/eval_zs_labels_fine.py`.
3. B0: `--mode train_b0` (escribe `runs/selection/ftB_baseline_dev_D50.json`) y después `--mode infer_dev`.
4. Selección de banda del juez en D: `python src/run_judge.py --split dev --n N --sub K --lo LO --hi HI` sobre la rejilla de bandas {(0.05,0.9), (0.1,0.9), (0.2,0.9), (0.1,0.7), (0.2,0.7)}, subconjuntos 0 y 1, N ∈ {10, 50}; después `python src/collect_judge_rows.py runs/selection/judge_runs_dev.jsonl runs/selection/judge_band_selection_D50.csv` y `python src/select_band.py runs/selection/judge_band_selection_D50.csv`.
5. Test (solo configuraciones congeladas; `ALLOW_TEST=1`): `--mode infer_test`, `--mode zs_test`, `python src/eval_test.py`, después `python src/run_judge.py --split test --n N --sub K --lo LO --hi HI` para los 10 pares (N, subconjunto) con las bandas elegidas, `collect_judge_rows.py` sobre `runs/test/judge_runs_test.jsonl` → `runs/test/J0_rows.csv` y `python src/analyze_test.py`.
6. Control de enmascarado: `python src/run_judge.py --split test --first 100 [--mask-entities] ...`.
7. A3 (N=10): `python src/a3_pool.py --P 1000`; `POOL_P=1000 modal run modal/relex_modal.py --mode infer_pool`; `python src/a3_pseudo.py --arm ctrl --P 1000` y `python src/a3_pseudo.py --arm judge --P 1000 --lo 0.1 --hi 0.7`; subir `runs/pseudo_labels` al volumen de Modal (`modal volume put relex-outputs runs/pseudo_labels /a3`); `ARM=ctrl POOL_P=1000 modal run modal/relex_modal.py --mode train_a3` y lo mismo con `ARM=judge`; en test (`ALLOW_TEST=1`) `ARM=ctrl|judge modal run ... --mode infer_test_a3` y `POOL_P=1000 python src/analyze_a3.py`.

**Descargar y verificar los artefactos** (desde una carpeta vacía):

```bash
gh release download artefactos-v1 -R jesusGoliat/gliner-relex-docre-benchmark
mkdir pseudo_labels && mv ctrl_N10_*.json judge_N10_*.json pseudo_labels/
sha256sum -c MANIFEST.sha256        # los 12 ficheros deben dar OK
```

Después, en una copia del repositorio: `llm_cache.jsonl` → `runs/llm_cache.jsonl`; `tar xzf raw_outputs.tgz -C runs`; `pseudo_labels/*.json` → `runs/pseudo_labels/`.

8. Registro del entorno de Modal y hashes de los checkpoints: `modal run modal/inspect_artifacts.py` (escribe `runs/environment/`).

Entorno del juez: `set -a; source .env; set +a` (claves `DEEPINFRA_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`), `C1_MAX_USD` (tope de gasto) y `C1_DI_CONCURRENCY` (200 en las ejecuciones de test reportadas). Las salidas crudas del modelo (≈1.1 GB sin comprimir), la caché de respuestas del juez, las pseudo-etiquetas originales de A3 y los checkpoints entrenados no se incluyen en este repositorio; `artifacts/MANIFEST.csv` lista los tres primeros (12 ficheros, 178 MB) con tamaño, sha256 y ubicación de uso, y se distribuyen en la release [`artefactos-v1`](https://github.com/jesusGoliat/gliner-relex-docre-benchmark/releases/tag/artefactos-v1); los checkpoints no están archivados fuera del volumen de Modal donde se entrenaron; las respuestas del juez proceden de una API en vivo y pueden diferir si no se dispone de la caché.
