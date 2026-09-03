# Handoff: contaminación de la clase ELL

Sesión del 2026-09-03. Punto de partida: muchas estrellas sin período coherente
reciben clase ELL con probabilidad alta.

## Estado en una línea

El ELL espurio lo pone la **CNN**, no el BRF, y ninguna perturbación interna
del clasificador lo detecta. Lo único que lo separa es `power` del
periodograma y una señal nueva, `p_LPV` bajo la normalización `rank`, que está
verificada pero **no explicada**.

---

## 1. Etiquetas humanas

`catalogs/notas_periodos.csv` — 45 pares (TIC, sector) revisados a mano
mirando todos los candidatos de período doblados en fase.

- **11 estrellas con período confirmado**, 28 sin período coherente.
- **6 ambiguas, sin usar**: TIC 16187387, 41903679, 71935430, 116065031,
  137113608, 153304135. Tienen `elegida='n'` con nota "periodico" y el parser
  no distinguió si "n" era "ninguno" o "el panel n" (`n` es también la
  etiqueta del panel 14). **Resolverlas es la acción de mayor rendimiento por
  estrella**: 4 de las 6 tienen `max(power) <= 0.30`, por debajo de todas las
  periódicas confirmadas (mínimo 0.513), así que si "n" era "el panel n" son
  contraejemplos fuertes del corte de power.
- Solo **2 ELL reales** en todo el conjunto (TIC 12921082 s82 y 362792232
  s41). Todo lo que se diga sobre "no matar las ELL buenas" descansa en dos
  estrellas.

Herramienta de anotación: `scripts/annotate_folds.py` (interactiva, guarda
incremental, retoma donde quedó). NO usar `conda run` sin
`--no-capture-output`: cierra stdin.

## 2. Qué se descartó, con números

Cinco intentos de detectar el error desde el clasificador. **Los cinco
fallaron**, siempre por la misma razón: el error no es el modelo dudando, es
el modelo convencido.

| vía | resultado | script |
|---|---|---|
| σ y `prob/σ` del MC-dropout | contaminantes indistinguibles de reales (σ≈0.003 en ambos) | `prob_snr.py` |
| voto e inestabilidad MC | AUC 0.536 = azar | `bimodal.py` |
| desacuerdo del ensemble de 7 checkpoints | los falsos tienen MÁS acuerdo (6.0) que los reales (5.0) | `ensemble_abstain.py` |
| morfología de la distribución MC (bimodalidad) | AUC 0.24–0.46 para las métricas de forma | `mc_shape.py` |
| consistencia entre 9 normalizaciones (argmax) | contaminantes y reales son ELL en las mismas 8 de 9 | `norm_matrix.py` |

Otros hechos medidos:

- **El BRF no es la fuente de ELL**: de 38 ELL finales en estrellas sin
  período, 36 ya eran ELL en la CNN. Balance neto del BRF sobre ELL: −4.
  Lo que sí hace el BRF es matar LPV (103 → 32 picos con `p_LPV > 0.3`).
- **La afirmación del docstring de `apply_vote_stability_gate`** ("el BRF
  aterriza en una constante sesgada hacia ELL, 0.51 contra 0.23") **está mal
  atribuida**. Barriendo amplitud ×200 y `per` ×30 sobre los picos CNN-ELL, el
  BRF se queda en ELL 0.59–0.65: es aproximadamente la identidad, no una
  fuente. Reescribir ese docstring y el de `scripts/step_cnn_only.py`.
- **MC-dropout es un parche**: mc20 vs mc200 coinciden en el 98.1% de los
  picos; 200 pasadas no mejora el acierto (0.667 → 0.641).
- **`results/cnn_mc200.npz` y `cnn_mc20.npz` están corridos sobre
  `Number_DST`**, que es `config.DEFAULT_MODEL` y uno de los peores
  checkpoints: 45 ELL falsos contra 24 de `batchBalanced_Number_ELL`. El
  sesgo varía por factor 2.5 entre checkpoints (`per_model.py`).
- **La elección de `HIST_NORM` mueve entre el 4% y el 67% de las clases** del
  catálogo (acuerdo con `log`: 0.96 en `power_0.33`, 0.33 en `quantized`). Es
  una incertidumbre sistemática grande y no está documentada.

## 3. Lo que sí funciona

### `power` del periodograma

```
max(power) por estrella   N   min   mediana   max
sin periodo              28  0.072   0.274   0.755
con periodo              11  0.513   0.737   0.971
```

AUC 0.971, contra 0.779 de la probabilidad máxima de la CNN. Un corte
`max(power) >= 0.5` solo, sin CNN ni BRF, da acc 0.923 con recall 1.00 contra
0.667 del pipeline completo.

**Pero no resuelve ELL**: al subir el umbral la contaminación se vuelve más
pura en ELL, no menos (78.9% de los contaminantes son ELL sin corte, 100% con
`power >= 0.5`). Los 2 sobrevivientes son estrellas que el humano describió
como **multiperiódicas**, no como ruido — altura de pico alta y período
incoherente. Un umbral de altura no puede distinguir eso.

### `p_LPV` bajo la normalización `rank` — LO NUEVO, sin explicar

```
p_LPV(rank) en los picos que log llama ELL
  5 ELL REALES        : todos exactamente 0.0000
  41 CONTAMINANTES    : todos > 0.0006  (mediana 0.0489)

sobre TODOS los picos etiquetados: AUC 0.878  (N=223 con periodo vs 482 sin)
correlacion con power: r = -0.079   <- informacion INDEPENDIENTE
```

Es la hipótesis original del usuario confirmada: la señal de LPV **sí** está
en los contaminantes; bajo `log` está aplastada a 1e-4 y solo aparece cuando
la normalización destruye la información de densidad.

**Verificado que no es artefacto del desempate** (`verify_rank.py`). El
`rank_norm` original rompe empates por orden de índice, lo que podría inyectar
estructura espuria. Con empates al azar (dos semillas) y con rango promediado:

```
      variante | ceros reales ceros contam | AUC ELL  AUC todos
          rank |          5/5         0/41 |   1.000      0.878
  rank_random1 |          5/5         0/41 |   1.000      0.887
  rank_random2 |          4/5         0/41 |   1.000      0.885
  rank_average |          4/5         0/41 |   1.000      0.887
```

Tampoco es un proxy de la ocupación del histograma (r=0.24) ni de la entropía
(r=0.31).

Reglas, a nivel estrella:

```
                          regla  TP FN FP TN   acc  ELL_falsos  ELL_reales
          sin filtro (baseline)  11  0 15 13 0.615          19           3
                   power >= 0.5   9  2  2 26 0.897           2           2
               p_LPV(rank) == 0   7  4  0 28 0.897           0           3
    power>=0.5 O p_LPV(rank)==0  10  1  2 26 0.923           2           3

AUC a nivel estrella: max_power 0.971 | min p_LPV(rank) 0.948 | suma de rangos 0.974
```

`p_LPV(rank) == 0` es **la única regla que deja cero contaminantes ELL
conservando las 3 reales**.

## 4. Por qué NO adoptarlo todavía

1. **No se sabe por qué funciona.** Bajo `rank` la red está en régimen
   degenerado (843 de 857 picos en Rndm) y `p_LPV` vale 0.05 en mediana. Qué
   codifica esa cantidad residual es una pregunta abierta. Sin mecanismo, no
   hay razón para que sobreviva a un cambio de modelo o de grilla.
2. **Un corte en "exactamente cero"** casi siempre indica una degeneración
   numérica, no un umbral físico. El respaldo real es el AUC 0.878 sobre
   223 vs 482, no el 1.000 sobre 5 vs 41.
3. Cuesta 4 verdaderos positivos (TP 11 → 7).

## 5. Siguiente sesión, en orden

1. **Resolver las 6 estrellas ambiguas** (§1). Es lo más barato y lo que más
   información aporta por estrella.
2. **Entender el mecanismo de `p_LPV(rank)`**: mirar qué features activa, o
   qué tienen en común los folds con `p_LPV = 0`. Sin esto no se puede
   defender en un paper.
3. **Test de armónicos de Gomel+2023** sobre el fold: ajustar 3 armónicos y
   exigir `A1/A2 < 1`, `A3/A2 < 0.3`, `A2/A2_err > 10`, `A1/A1_err > 3`.
   Es lo único pendiente que **no** es una perturbación del clasificador —
   mide la forma de la curva directamente. Ver
   `docs/UMBRALES_PERIODICIDAD_LITERATURA.md` §4.
4. **Umbrales de la literatura** (mismo doc): LPH de McQuillan (`>= 0.15`),
   S/N >= 5 en amplitud con ventana local de 1 d⁻¹ sobre el espectro
   residual (Burssens+2020), altura-en-sigmas de Gomel (`> 12`) como única
   escala común LS↔ACF.
5. **Reescribir los docstrings mal atribuidos** (§2).
6. Considerar correr el pipeline sobre `batchBalanced_Number_ELL` en vez de
   `Number_DST`, o abandonar el dropout por el ensemble de 7.

## 6. Archivos

**Datos generados** (todo en `results/`):
- `cnn_mc20.npz`, `cnn_mc200.npz` — MC-dropout sobre Number_DST
- `cnn_mc.npz` — ensemble determinístico de los 7 checkpoints (1 pasada c/u)
- `clasificacion_review20.csv` — con `med_`/`q1_`/`q3_` por grupo
- `clasificacion_mean20.csv` — variante BRF(media), sin dispersión
- `norm_compare/cnn_input_*.npz` y `cnn_mc_*.npz` — 12 normalizaciones
  (log, min_max, power_0.5, power_0.33, column, quantized, poisson, rank,
  binary, rank_random1, rank_random2, rank_average)

**PDFs** (`results/figures/`): `lc_prob_review_median.pdf`,
`phasefold_review_median.pdf`, `top_folds.pdf`, `mc_shape_20.pdf`

**Scripts nuevos**: `scripts/annotate_folds.py`, `scripts/build_top3_folds.py`,
`scripts/step_brf_mean.py`, y el análisis en `scripts/analysis_ell/`.

**Docs**: `docs/UMBRALES_PERIODICIDAD_LITERATURA.md`.

**Entorno**:
```
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate CNN_TESS
export PYTHONPATH=src
export MSV_BRF=$HOME/Dropbox/MassiveStarVariability/models/balanced_random_forest_model.joblib
# la CNN necesita tf_env y los pesos locales:
MSV_WEIGHTS=$HOME/ViT_VariableStars/pretrained/keras_checkpoints PYTHONPATH=src \
  /opt/anaconda3/envs/tf_env/bin/python scripts/step_cnn.py <in.npz> <out.npz>
```

**Trampa conocida**: `msv.features.phase_fold_hist2d_*` aplica `.T[::-1]`
DESPUÉS de normalizar. Saltearlo le entrega a la CNN el histograma transpuesto
y todo colapsa a Rndm. Control obligatorio al construir cubos: reconstruir
`log` y verificar que reproduce `results/cnn_input.npz['X']` bit a bit.
