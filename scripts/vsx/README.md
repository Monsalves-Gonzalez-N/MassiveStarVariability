# Selección de período sobre el benchmark VSX

## El script de producción

`selector_periodo.py` — único script del pipeline. Entra una fila por pico,
sale una fila por estrella: el período elegido y su clase, o `multiperiodica`.

```
PYTHONPATH=src python scripts/vsx/selector_periodo.py
```

### El pipeline

```
1. run_peaks              LS + ACF, todos los picos del periodograma
2. step_cnn               una pasada determinista, checkpoint Number_M,
                          hist2d con normalizacion log, 8 clases -> 5 grupos
3. filtro Rndm            clase != Rndm     corta ~78 % de los picos
4. probe_peaks            --filtro-cnn --sin-subarmonico      ~3.5 min
5. gate multiperiodicas   RF por estrella, corte 0.30
6. selector de periodo    11 features, boosting, top-1 sin umbral
7. clasificacion          RF por estrella, ECL/ELL/PULS
```

Filtrar con la CNN **antes** de sondear baja el costo de ~12.4 min a 3.4 min sin
cambiar ningún valor: `extract_components` depende solo de la curva y `probe`
solo del período candidato.

### El embudo, en números

```
filtro Rndm       9646 -> 2113 picos (27 -> 5.9 por estrella)
                  pierde el período verdadero en 4 estrellas de 294

gate (0.30)       conserva 275/294 ok (93.5 %)
                  saca 56/65 multiperiódicas (86.2 %)

selección         acierto top-1 sobre las ok del catálogo   90.2 %
                  períodos correctos / total de ok          84.4 %
                  contaminación del catálogo                 3.2 %
                  hasta un factor 2                         92.2 %

clasificación     88.9 ± 0.5 % (balanceado 64.3), 281 estrellas
                  referencia: clase CNN en el pico de mayor SNR, 62.8 %
```

### Por qué el gate va antes y no como umbral de abstención

La versión anterior elegía período y se abstenía bajo un umbral. Conseguía 91.9 %
de acierto, pero **tirando estrellas buenas**: descartaba 59 de 293 `ok` para
sacar 64 de 66 multiperiódicas. Un modelo entrenado *para* detectar
multiperiódicas es mucho más eficiente — a igual retención de `ok`:

```
conserva ok    gate RF   abstención emergente
   94.9         84.8           54.5
   90.1         92.4           74.2
   85.0         95.5           86.4
```

Separar las dos decisiones sube los períodos correctos de **73.4 % a 82.3 %** del
total, a cambio de 3.2 % de contaminación en vez de 0.8 %.

### Las 11 features del selector

| bloque | features |
|---|---|
| periodograma | `ciclos`, `per_rel`, `power`, `width` |
| sonda (1 pasada) | `snr`, `snr_breger`, `a2_a1`, `amplitude_ppt` |
| CNN | `log_pLPV`, `p_Rndm`, `p_E` |

El modo de falla se invierte por familia y es la física esperada: **ECL falla
hacia P/2** (los dos eclipses se superponen al doblar a la mitad), **PULS falla
hacia 2P** (elige el subarmónico). Por eso el catálogo lleva una columna `nota`
con la dirección de la ambigüedad cuando el segundo candidato está en P/2 o 2P:
1 de cada 3 eclipsantes tiene un candidato en P/2 pisándole los talones, contra
3 % de las pulsantes.

## Qué se probó y quedó fuera

| pieza | efecto | por qué fuera |
|---|---|---|
| 2ª pasada de la sonda (`snr_half`) | +0.6 | `a2_a1` ya mide el factor 2, gratis, en la misma pasada |
| `per` absoluto | +0.1 | prior sobre los períodos publicados por VSX, no una regla |
| MC-dropout (20×8) + PCA(8) | +0.6 selección, **−7.5 abstención** | obliga a dos configuraciones distintas |
| 8 clases finas en vez de 5 grupos | +0.4 selección, **−6 abstención** | mismo canje malo |
| BalancedRF | −1 | el desbalance por pico es 40/60, no lo necesita |
| features de multiperiodicidad | −1.5 | versión discretizada de lo que `n_picos`/`power_rel` ya tenían |
| 14 features de amplitud y par 2P | 0 | redundantes con `a2_a1` |
| `amplitude`, `irregular` | −21 abstención | constantes dentro de la estrella: no pueden elegir pico |
| abstención por umbral del selector | −8.9 en períodos correctos | tira estrellas buenas para conseguir pureza; la reemplaza el gate |
| `p_LPV` | 0 | `log_pLPV` es su transformación monótona; un árbol no distingue |
| 7 features de contexto | 0 | importancia < 0.006 |

Regla general de la sesión: **más features bajan la abstención**. Todo lo que
hace al selector más confiado le quita la capacidad de callarse.

## El registro de las ablaciones

- `arbol_features.py` — comparación de conjuntos de features y de modelos
  (árbol, RF, BRF, SVM, boosting), con pesos por familia de VSX.
- `reducir_features.py` — poda por importancia, conjuntos anidados top-k. Es de
  donde salen las 11.
- `arbol_abstencion.py` — banco de pruebas de la abstención. Flags `--once`,
  `--mc`, `--grupos`, `--con-globales`, `--entrena-unconstrained`.
- `arbol_mc.py` — el resumen invariante al orden del MC-dropout y el PCA.
- `clase_estrella.py` — clasificar la ESTRELLA con el conjunto de picos
  aplanado. Su `vector_por_estrella` es el que usan el gate y el paso 3.
- `gate_multiperiodica.py` — la comparación gate dedicado vs abstención
  emergente que motivó reordenar el pipeline.
- `resumen.py` — el embudo completo en números, para el paper.
- `comparar_checkpoints.py` — el pipeline completo con cada uno de los 7
  checkpoints. De aquí sale `DEFAULT_MODEL = Number_M`.
- `build_cubo_norm.py` — rehace el cubo de la CNN con otra normalización del
  hist2d (validado: reproduce bit a bit el cubo original con `log`).
- `build_peaks_norm.py` — cambia solo el bloque de la CNN en `peaks_con_snr`,
  para comparar normalizaciones o checkpoints sin recalcular nada más.
- `plot_misclasificadas.py` — PDF de las estrellas donde el período elegido no
  es el de VSX, dobladas a los dos. OJO: `results/vsx/curves.pkl` tiene el flujo
  en CEROS; hay que leer de `lc.parquet`.
- `arbol_seleccion.py`, `barrido_snr.py`, `barrido_ancho_acf.py`,
  `test_recall_picos.py` — anteriores a esta sesión.

## La normalización y el checkpoint

Las dos decisiones se midieron corriendo el pipeline completo, no se heredaron.

**`log` sobre `min_max`** (`config.HIST_NORM`). `min_max` conserva más picos
(recall del filtro 99.0 % contra 97.6 %) y sube el acierto del período 1.4
puntos, pero rompe el acoplamiento clase-período que es lo que hace
interpretable el catálogo: el AUC de `log_pLPV` para distinguir el pico correcto
baja de 0.632 a 0.578. Y en el período correcto empeora la clase de las
pulsantes — 68 correctas contra 80 con `log`, porque manda más a `LPV`. Se
queda `log`.

**`Number_M` sobre `Number_ELL`** (`config.DEFAULT_MODEL`). Los 7 checkpoints
sobre el mismo cubo y la misma sonda, 10 semillas:

```
                          top-1   correctos/ok   clase  balanc.  pierde por Rndm
Number_ELL (el anterior)  87.3       82.3        88.9    62.9         8
Number_M                  91.6       85.4        88.9    64.3         4
batchBalanced_Number_DST  90.7       83.6        89.1    63.8         8
```

`Number_M` es el único que está arriba o cerca del mejor en todas las métricas a
la vez, y las pérdidas por `Rndm` (número determinista, sin CV) se reducen a la
mitad. Que el checkpoint de Miras sea el mejor es consistente con que
`log_pLPV` sea la feature #1 del selector: el pipeline se apoya sobre todo en
distinguir lo que NO es una variable de período largo.

**Advertencia**: elegir el mejor de 7 sobre el mismo benchmark de 294 estrellas
es selección sobre el conjunto de prueba. Las 10 semillas controlan la varianza
del CV, no ésa. Confirmar con muestra independiente.

## Detalles de entorno

Los pesos de la CNN están hardcodeados a la máquina Linux en `config.py:61`. En
el Mac:

```
MSV_WEIGHTS=/Users/bhianca/ViT_VariableStars/pretrained/keras_checkpoints
```

`selector_periodo.py` y las ablaciones corren en `CNN_TESS`; solo `step_cnn.py`
necesita `tf_env`.

## Dos cosas que siguen abiertas

- **15 ELL** es el techo real. Ningún esquema de pesos lo arregla; hay que
  ampliar la muestra desde VSX. Con RF el clasificador las borra (recall 1.3 %);
  con BalancedRF las recupera a 66.7 % pero con ~18 % de precisión.
- **Validación anidada pendiente.** El gate y el clasificador comparten features
  y muestra, y hoy cada uno se valida con su propio CV. El número end-to-end
  (82.3 %) puede ser algo optimista hasta que se valide el pipeline completo.
- Las **fases armónicas** del ajuste de `prewhiten` se descartan hoy (solo se
  usa `a2_a1`). Son lo que separa ELL de ECL por geometría, y es la feature con
  mejor retorno esperado antes de tocar el modelo.
