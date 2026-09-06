# Validar el pipeline contra la golden sample

Muestra externa de clasificación humana (`catalogs/golden_sample.csv`, 1081 TIC
de 5 papers). Objetivo final: el plot amplitud vs período con clase e
irregularidad, con los cortes calibrados contra alguien que no seamos nosotros.

**Decidido el 2026-09-04:** nos quedamos con los 134 TIC que ya tienen curva de
luz. Los 947 que faltan no están ni en la muestra padre
(`1_MassiveXTessV8.csv`), así que no hay `dataURI` y bajarlos es una query nueva
a MAST. Si en algún momento se hace, el subset que importa son 210 (ECL 29,
ELL 6, PULS 80, ROT 95), no los 947.

## Estado

Hecho:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate CNN_TESS
export PYTHONPATH=src

python scripts/build_lc_parquet.py --dir golden --out results/golden/lc.parquet
python scripts/run_peaks.py --lc results/golden/lc.parquet --out results/golden/peaks.parquet
python scripts/step_cnn_export.py results/golden/peaks.parquet results/golden/cnn_input.npz
MSV_WEIGHTS=$HOME/ViT_VariableStars/pretrained/keras_checkpoints PYTHONPATH=src /opt/anaconda3/envs/tf_env/bin/python scripts/step_cnn.py results/golden/cnn_input.npz results/golden/cnn_probs.npz --model Number_ELL
python scripts/step_clasificar_una_red.py results/golden/cnn_input.npz results/golden/cnn_probs.npz results/golden/clasificacion.csv --curves results/golden/curves.pkl

python scripts/collapse_comb.py             # -> results/golden/peaks_collapsed.parquet
python scripts/step_cnn_export.py results/golden/peaks_collapsed.parquet results/golden/cnn_input_collapsed.npz
MSV_WEIGHTS=$HOME/ViT_VariableStars/pretrained/keras_checkpoints PYTHONPATH=src /opt/anaconda3/envs/tf_env/bin/python scripts/step_cnn.py results/golden/cnn_input_collapsed.npz results/golden/cnn_probs_collapsed.npz --model Number_ELL
python scripts/step_clasificar_una_red.py results/golden/cnn_input_collapsed.npz results/golden/cnn_probs_collapsed.npz results/golden/clasificacion_collapsed.csv --curves results/golden/curves.pkl

python scripts/golden/build_truth.py        # -> catalogs/golden_truth.csv
python scripts/golden/match_periods.py      # -> results/golden/match_*.csv
python scripts/golden/build_2p_review.py    # -> results/figures/golden_review_2P.pdf

python scripts/golden/prewhiten.py                 # -> results/golden/prewhiten_*
python scripts/step_cnn_export.py results/golden/prewhiten_peaks.parquet results/golden/prewhiten_cnn_input.npz
MSV_WEIGHTS=$HOME/ViT_VariableStars/pretrained/keras_checkpoints PYTHONPATH=src /opt/anaconda3/envs/tf_env/bin/python scripts/step_cnn.py results/golden/prewhiten_cnn_input.npz results/golden/prewhiten_cnn_probs.npz --model Number_ELL
python scripts/step_clasificar_una_red.py results/golden/prewhiten_cnn_input.npz results/golden/prewhiten_cnn_probs.npz results/golden/prewhiten_clasificacion.csv
python scripts/golden/build_prewhiten_review.py    # -> results/figures/golden_prewhiten.pdf + prewhiten_resumen.csv
python scripts/golden/plot_star_folds.py --tic 42889751   # -> results/figures/folds_TIC42889751.pdf
python scripts/golden/plot_prewhiten_steps.py --tic 42889751 --sector 6 --n-max 10

python scripts/golden/build_probe_targets.py       # -> results/golden/probe_targets.csv
python scripts/golden/prewhiten.py --all --veredicto results/golden/probe_targets.csv --out-dir results/golden/probe_all
python scripts/golden/score_periods.py             # -> results/golden/score_periods.csv
```

`plot_prewhiten_steps.py` es la explicación de la operación: una fila por
iteración con el espectro del residuo, la curva de corte `snr_min` x ruido
local, el pico elegido y el modelo que se le resta. Está para dejar claro que
**no se seleccionan puntos**: en cada instante el brillo es la suma de todas las
variaciones, así que no hay un subconjunto de la curva que sea una de ellas. Lo
que se corta es el espectro.

**`--n-max` es un tope, no un criterio de convergencia.** El default de 6 trunca
en las estrellas con grupos ricos: TIC 42889751 s6 da 8 componentes si se lo
deja llegar (la novena queda en SNR 3.8 y ahí sí para sola). Afecta poco a las
sondas — sacar más componentes baja el ruido y sube un poco los SNR — pero al
mirar una estrella conviene correrla con `--n-max 10` y ver dónde para.

`plot_star_folds.py` es el detalle de UNA estrella: una página por sector con
un panel por variación — la línea roja es esa variación sola, y los puntos
grises son el residuo, que no es otra variación sino lo que ninguna componente
coherente explica. Con `--only-model` se va la nube y queda solo la curva. La
última página enfrenta los sectores entre sí. Esa última es el discriminante entre rotación y pulsación, que en un
solo sector no se puede separar: una señal rotacional vuelve en la misma
frecuencia con la misma amplitud, un grupo de frecuencias devuelve la envolvente
y no sus miembros.

Cada sonda se pliega en cuatro variantes: `raw` (la curva cruda), `iso` (sin las
otras variaciones pero conservando las conmensurables), `alone` (la componente
completamente sola, sin ni siquiera su grupo armónico — solo para las
extraídas) y `model` (el modelo sin ruido, ilustración de la forma).

`prewhiten.py` sin argumentos corre las 15 estrella-sector marcadas
`multi periodico`; con `--all --out-dir results/golden/prewhiten_all` corre las
59 del caso 2P completo. De `step_clasificar_una_red.py` acá se leen `clase`,
`prob` y `log_pLPV` por fila: `irregular`, `reportado` y `nivel` mezclan las
filas `:raw` con las `:iso` de la misma estrella y no significan nada.

134 TIC, 806 estrella-sector, 12044 picos. 40 TIC con `period_usable` (eran
80 antes de sacar las Be, §10).

Pendiente: llenar el formulario del PDF y correr

```bash
python scripts/golden/read_veredicto.py     # -> results/golden/veredicto_2P.csv
```

### El PDF de revisión es un formulario

`build_2p_review.py` escribe el PDF con matplotlib y después le agrega widgets
AcroForm con PyMuPDF (matplotlib no escribe formularios). Preview los llena y
los guarda con cmd+S; `read_veredicto.py` los lee de vuelta.

Cada fila tiene sólo dos paneles — la curva en tiempo y el fold en el período
**publicado**. Nuestro período y nuestra clase **no se muestran a propósito**:
el juicio se ancla si se ve la respuesta del pipeline al lado. Las cuatro
opciones se traducen a quién tiene el período bueno:

```
periodico        -> golden     el fold cierra a ese período
multi periodico  -> multi      hay varias frecuencias reales; la publicada puede
                               ser una, pero "acertar" no es la pregunta
irregular        -> ninguno    no hay señal coherente ahí
half period      -> nuestro    dos ciclos distintos superpuestos: el verdadero es el doble
```

Regenerar el PDF **no pierde lo contestado**: se leen las respuestas del
archivo anterior y se reponen por nombre de campo, y avisa si alguna quedó
huérfana. `--fresh` las descarta. Guardar en Preview antes de regenerar.

`read_veredicto.py` agrega a nivel estrella y marca `conflicto` cuando dos
sectores de la misma no coinciden — que es información, no un error: un sector
corto puede no mostrar lo que muestra uno largo.

Trampa al leer los widgets: un checkbox sin marcar vuelve como el **string**
`"Off"`, que en Python es verdadero. Hay que comparar contra el estado, no
evaluar el valor.

## Los resultados que ya salieron

### 1. El periodograma está bien; la regla de selección no

```
Q1  el período del paper ESTÁ entre nuestros picos       87.7%   (316 estrella-sector)
      ECL 100%   ROT 94%   PULS 91%   BE 80%
Q2  la regla min p_LPV lo ELIGE                          21.8%   (de los 220 alcanzables)
```

Son dos fallas separadas y hay que reportarlas separadas: un porcentaje único
mezcla un periodograma sano con una regla rota.

**Ese 21.8% (y el 31.3% de §9) está mal medido y no hay que citarlo.** Cuenta
como error dos cosas que no lo son — comparar contra períodos que no existen, y
llamar equivocado un armónico que la fotometría no puede desmentir. La versión
corregida está en §10; el número honesto es **50.5% / 63.7%**.

### 2. El alias 2P es el modo de falla dominante, y puede que sea del paper

De los 241 picos reportados con período publicado: 48 en 1:1, **59 en 2P**,
129 en "otro". Y los 8 que el pipeline marca `nivel alta` son **todos** 2P.

**En los 29 TIC del caso 2P, el período del paper estaba entre nuestros
candidatos en los 29.** No es que no lo encontráramos: la regla lo vio y
prefirió el doble.

Se parte en dos hipótesis que el fold visual separa:

- **nuestra clase es E** (10 TIC): binaria eclipsante con eclipses desiguales.
  El paper reporta el período fotométrico y nosotros el orbital, que es el
  correcto. TIC 42889751 es el caso de manual: en 2.354 d se ven dos mínimos de
  profundidad distinta, en 1.1765 d se superponen en uno.
- **nuestra clase es ELL** (19 TIC): sobre estrellas que el paper llama ROT,
  PULS o BE. Ahí el doblado es el mecanismo por el que entra la contaminación
  ELL, no una corrección — una elipsoidal tiene dos máximos por órbita, así que
  doblar el período de cualquier modulación la disfraza de ELL.

Evidencia de que la ambigüedad también existe en la literatura: de las 3
estrellas donde dos papers reportan períodos distintos para el mismo TIC, 2
difieren por un factor exacto de 2 (spread 1.966 y 1.993).

### 3. La irregularidad valida, pero sólo contra SLF

```
AUC SLF(20) vs ECL+PULS(24)      0.775
AUC SLF+BE(81) vs ECL+PULS(24)   0.517
```

Mediana de `irregular`: SLF 0.55, ROT 0.42, PULS 0.40, BE 0.31, ECL 0.15.

**BE no es proxy de irregular** y meterla borra la señal: Labadie-Bartz reporta
período rotacional o pulsacional para 40 de las 61 BE que tenemos. La
predicción correcta es "SLF alto", no "SLF y BE alto".

### 4. El período publicado está en los datos; el nuestro no

Prewhitening iterativo (`msv.prewhiten`): se extrae la frecuencia más fuerte
con sus armónicos, se resta, y se repite sobre el residuo hasta SNR 4. Después
se sondean tres frecuencias impuestas — la publicada, la nuestra y, para las Be,
los centros `fg1`/`fg2` de sus grupos — ajustando una sinusoide exactamente ahí
sobre la curva con las OTRAS variaciones removidas. Sondear separa dos cosas que
el fold confunde: si el período **está** en los datos, y si **se ve** al plegar.

Sobre las 59 estrella-sector del caso 2P:

```
                       SNR>=4        mediana SNR   mediana A
periodo publicado      38/59              8.9        1.46 ppt
periodo nuestro        10/59              1.3        0.51 ppt
```

Y las 10 que sí tienen potencia en el fundamental de nuestro período no son
cualquiera:

```
clase del paper    con potencia en 2P
ECL                     8/11   (73%)
BE                      1/17   ( 6%)
ROT                     1/21   ( 5%)
PULS                    0/7
AMBIGUOUS               0/3
```

**El doblado es correcto cuando la estrella es una eclipsante de verdad y
espurio cuando no.** En las 49 restantes el fundamental de nuestro período está
vacío (A2/A1 entre 1.3 y 31): toda la potencia vive en su primer armónico, que
es la definición de un subarmónico. Esto da la regla que la Etapa 2 pedía, con
un número en vez de un ojo: **ajustar fundamental + armónicos en el período
reportado; si el fundamental no llega a SNR 4 y el primer armónico domina, el
período está doblado.**

Ojo con el corte por clase NUESTRA, que es el que estaba escrito antes: de
nuestras 25 estrella-sector con clase E solo 8 tienen potencia real en 2P (32%),
y las 8 son las tres eclipsantes del paper. La clase E no distingue.

**TIC 42889751 ya no es el caso de manual de eclipses desiguales.** Es una Be
con `signals=G` y `Ns=5` grupos, y la evidencia en contra es de tres tipos:
el fundamental en 2.354 d tiene SNR 0.48-1.03; las frecuencias extraídas no se
repiten entre sectores (s6: 0.868/0.828/0.916/0.784 /d; s33:
0.796/0.840/0.875/0.754 /d) sino solo su envolvente; y la "profundidad del
eclipse" plegando en nuestro período pasa de 45.7 a 10.1 ppt entre s6 y s33.
Una geometría de eclipses no cambia 4.5 veces en dos años; una amplificación
temporal de un grupo de frecuencias es justo lo que Labadie-Bartz+ 2022
describen. Los dos mínimos de distinta profundidad son el batido entre los 4-5
miembros no resueltos del grupo g1.

### 5. Por qué el fold no cierra aunque el período esté

Tres motivos distintos, y el espectro de amplitud los separa:

- **El "período" de las Be no es un período.** `build_golden_sample.py` toma
  `period = 1/fg1` de Labadie-Bartz+ 2022, y `fg1` es en el ReadMe "Central
  frequency of g1": el centro de un GRUPO de frecuencias apiñadas
  (`signals=G`, `Ns` = 2 a 5 grupos por estrella). El prewhitening lo
  reproduce: en TIC 42889751 s33 cinco de seis componentes caen entre 1.10 y
  1.33 d, tres de ellas dentro de una resolución de Rayleigh. No hay una señal
  coherente que plegar.
- **La misma tabla publica el segundo grupo, y es exactamente el doble.** En
  las 13 Be del caso 2P, `fg2/fg1` = 1.89 a 2.06 y `fg1/f_nuestro` = 1.93 a
  2.04. La estrella tiene potencia en `fg1` y en `2*fg1`; nosotros elegimos
  `fg1/2`. Doblar convierte el par (f, 2f) en (f/2, f), o sea dos máximos por
  ciclo, o sea una elipsoidal. Ese es el mecanismo de contaminación ELL
  completo, con los números del propio paper.
- **Cuidado con leer las componentes de un grupo como modos.** Blanquear una
  estructura ancha no devuelve su contenido: la deja cortada en un peine de
  picos separados por una resolución de Rayleigh, porque cada resta deja un
  residuo que la iteración siguiente encuentra un elemento de resolución más
  allá. En TIC 42889751 la separación mediana entre componentes vecinas es
  **1.01 Rayleigh en los dos sectores**. Marcan dónde está la joroba; no son
  cinco modos independientes. Lo que sí es real es que la joroba es más ancha
  que la resolución, que es exactamente por lo que no hay un período que
  plegar.
- **La señal está pero enterrada.** TIC 48217508: el período de rotación de
  Campelo (2.785 d, A=1.19 ppt, SNR 15.8) convive con una pulsación de 0.279 d
  (A=1.02, SNR 34) y otra de 0.0957 d — que es el `P_lit` = 0.095 d que la
  propia Tabla 2 de Campelo cita de otro paper. Las tres están en el mismo
  sector. El fold en 2.785 d sobre la curva cruda lleva las otras dos encima
  como dispersión.

### 6. Aislar una variación a la vez sí cambia lo que dice la red

Cada sonda se pliega dos veces, sobre la curva cruda y sobre la curva con las
otras variaciones removidas, y las dos imágenes van a la CNN. Una componente
conmensurable con la sonda no se remueve: el primer armónico de una elipsoidal
es lo que le da sus dos máximos, y quitarlo borraría la forma que el fold tiene
que mostrar.

En las 15 estrella-sector multiperiódicas la clase cambia en 28 de 94 sondas,
10 pasan a `Pulsating`, y la mediana de `log_pLPV` sube de 0.27 a 1.03. Casos
concretos: TIC 48217508 en el período de Campelo pasa de `Pulsating` 4.0 a
`Pulsating` 6.4; TIC 276653683 PW1 pasa de `LPV` 0.2 a `Pulsating` 3.5.

No es una mejora universal: sobre las 59 del caso 2P completo la mediana baja
de 1.07 a 0.82. Aislar ayuda donde efectivamente hay varias variaciones, que es
lo que se esperaba y no más.

**El número más incómodo está acá.** En TIC 48217508 s26 el fold crudo en
nuestro período da `ELL` con `log_pLPV` 12.05 — por encima de `LOG_PLPV_MIN` y
por lo tanto `nivel alta` — sobre una frecuencia cuyo fundamental tiene SNR
0.85. La confianza del pipeline es máxima justo donde no hay señal.

### 7. Por qué falla el pipeline

El periodograma no es el problema (§1: 87.7% de recall). Falla en los dos pasos
siguientes, y son tres cosas encadenadas.

**El ACF fabrica los candidatos equivocados y nadie colapsa el peine.** De los
59 picos reportados en 2P, **55 vienen del ACF**; de los 48 correctos en 1:1, 38
vienen del LS.

```
origen del pico reportado    ACF    LS
1:1                           10    38
2P                            55     4
```

No es un bug del ACF: la autocorrelación de una señal periódica tiene picos en
P, 2P, 3P por construcción, eso es lo que hace un ACF. El problema es que esos
subarmónicos entran a la lista de candidatos como si fueran períodos
independientes. `msv.peaks.candidate_periods` existe exactamente para colapsar
el peine en su fundamental — y **no está en este camino**: solo la usan
`build_phasefold_input.py` y `validate_peak_selection.py`. `run_peaks.py` emite
un candidato por cada pico del ACF.

**La regla de selección ordena por cómo se ve la IMAGEN, y la imagen se ve mejor
en el período equivocado.** Enfrentando el período publicado contra el nuestro
con el criterio del pipeline (`max log_pLPV`):

```
                              prefiere el NUESTRO
las 59 estrella-sector             56  (95%)
las 38 donde el publicado
tiene senal real (SNR>=4)          37  (97%)

mediana log_pLPV   publicado 0.20    nuestro 7.80
mediana SNR        publicado 8.9     nuestro 1.3
```

El mecanismo se verifica quitando las otras variaciones: plegado en el período
publicado sobre la curva cruda, `log_pLPV` mediano es 0.20 — o sea que la red lo
ve como una LPV, que es su clase de "nube sin forma". Aislando, la mediana sube
a 2.09 y la clase se mueve de `LPV` a `E`/`Pulsating` en 28 de 59. **El fold del
período verdadero está borroneado por las otras variaciones, y la regla lee
"borroso" como "no es un período".** El fold del doble, en cambio, tiene dos
máximos y se ve estructurado.

**Y en ningún punto de la cadena se pregunta si hay potencia en esa
frecuencia.** Sobre 90 picos reportados elegidos AL AZAR (no del conjunto 2P):

```
el periodo reportado tiene SNR>=4 en su fundamental    36/90   (40%)
tiene A2/A1 > 2 (el armonico domina)                   28/90   (31%)

Spearman(log_pLPV, SNR del fundamental)   -0.126   p=0.24   NO significativa
```

Cuidado con cómo se lee esa correlación. Una versión anterior de esta sección
decía que `log_pLPV` está **anticorrelacionado** con la señal, con rho = -0.275.
Ese número estaba medido sobre `veredicto_2P.csv`, que es por construcción el
conjunto donde el pipeline falló: condicionar en el fracaso y después reportar
la correlación no prueba nada. En la muestra al azar la correlación no es
significativa. Lo correcto es más débil y más preciso: **`log_pLPV` no está
invertido, es CIEGO a si la frecuencia existe.** Eso es compatible con que
funcione como discriminador de clase, que es para lo que se validó (§ nota
abajo), y no como detector de período.

Lo que no depende de ninguna selección es el 40%: **tres de cada cinco períodos
que el pipeline reporta no tienen potencia en su fundamental.**

En una frase: el pipeline elige períodos preguntándole a un clasificador
entrenado con OGLE —estrellas de un solo modo y baselines de años— cuál fold se
parece más a una variable, sobre candidatos que un peine fabricó, sin medir
nunca si la frecuencia existe. En una estrella de un solo modo eso funciona. En
las masivas multiperiódicas de esta muestra se invierte.

Lo que lo arreglaría, en orden:

1. ~~Meter `candidate_periods` en el camino~~ — hecho, §9:
   `scripts/collapse_comb.py`. Q2 de 21.8% a 31.3%, los 2P a la mitad.
2. Agregar el test del fundamental como VETO: un candidato cuyo fundamental no
   llega a SNR 4 no se reporta. En el conjunto 2P rechaza nuestro período en 49
   de 59 y deja en pie el publicado en 38; en la muestra al azar tocaría al 60%
   de lo reportado.
3. Ordenar los que sobreviven por SNR de amplitud, que es físico, en vez de por
   `p_LPV`, que es la apariencia de una imagen. La red para la CLASE, no para
   el período.

### 8. Por qué `log_pLPV` funcionó antes y acá no

La validación original (`catalogs/notas_periodos.csv`) tiene **16 estrella-sector
con período confirmado a ojo — E 9, ELL 5, Pulsating 4, LPV 1 — y 29
descartadas** con notas del tipo "multiperiodico", "no se ve el periodo". O sea
que se validó sobre las de un solo modo, después de sacar a mano justamente las
estrellas donde el problema aparece. Ahí `log_pLPV` sube de 8/11 a 10/11 y eso
sigue siendo cierto: no está roto, está validado en un dominio que no es este.

**`irregular` (el `p_LPV` medio de la estrella) NO mide multiperiodicidad.**
Es la pregunta natural — usarlo para mandar las uniperiódicas por un camino y
las multiperiódicas al prewhitening — y sale al revés. Sobre las mismas 90 al
azar:

```
irregular      n   componentes   fraccion      amplitud
                   (mediana)     coherente     (mag)
<0.15         16       7.0         0.91         0.047
0.15-0.35     23       5.0         0.79         0.035
0.35-0.60     31       2.0         0.49         0.035
>0.60         20       2.0         0.53         0.037

Spearman(irregular, n componentes)  -0.323  p=0.002
   parcial, controlando amplitud    -0.308  p=0.003
Spearman(amplitud, irregular)       -0.109  p=0.31   (no es brillo)
```

**`irregular` bajo no quiere decir "un solo período": quiere decir "mucha señal
coherente"**, que es exactamente una pulsante multiperiódica. Tiene sentido
físico — LPV es la clase de las curvas borroneadas, así que `p_LPV` mide
estocasticidad, no cuenta frecuencias. Lo que sí trackea es la fracción
coherente: 0.91 -> 0.49 al subir `irregular`. Es el mismo eje del AUC 0.775
contra SLF de §3.

El triage correcto usa dos variables, y ninguna necesita a la red:

```
irregular alto                      -> estocastica: no hay periodo que buscar
irregular bajo + una componente
  se lleva la varianza coherente    -> uniperiodica: ese es el periodo
irregular bajo + varias componentes -> multiperiodica: prewhitening
```

### 9. El colapso del peine, aplicado

`scripts/collapse_comb.py` mete `msv.peaks.candidate_periods` en el camino: la
serie armónica declarada se colapsa en su fundamental y sus múltiplos dejan de
ser candidatos. **Colapsar a secas rompe las eclipsantes**, y es el motivo por
el que hay un segundo paso. En TIC 220197273 s6 el ACF tiene el peine completo
(0.604, 1.208, 1.813, 2.438...), la serie se declara en P0 = 0.6080 —el período
FOTOMÉTRICO, que es el que reporta la literatura— y el orbital de 1.2083 d, con
8.2 ppt en su fundamental, desaparece. Lo mismo en TIC 337886863: colapsa a
2.846 y mata el 5.699, que tiene SNR 14.4.

Por eso después del colapso se le pregunta a la curva por cada período que el
peine se llevó puesto: se ajusta una sinusoide exactamente ahí y se mide el SNR
de su FUNDAMENTAL. El que tiene potencia propia vuelve. **La regla es nunca
borrar un período que exista**, y funciona porque la asimetría entre eclipses
desiguales es justamente lo que le da potencia al fundamental en P_orb:

```
                     SNR(2*P0)          decision
TIC 337886863 s58  ECL      9.06     se emite 2*P0
TIC 220197273 s6   ECL      5.19     se emite 2*P0
TIC 42889751  s6   Be       1.17     solo P0
TIC 48217508  s26  ROT      0.92     solo P0
TIC 238381092 s62  ROT      0.02     solo P0

rescate por potencia propia (SNR >= 4)   probados   rescatados
picos absorbidos por la serie                2205          370
P0/2 (fotometrico), no era pico               161           70
2*P0 (orbital), no era pico                    32            7
```

Resultado sobre las 316 estrella-sector con período publicado:

```
                              antes    despues
candidatos por estrella-sector 19.4       14.3
Q1  el periodo publicado esta  87.7%      84.2%
Q1  existe ademas un 2P        61.1%      37.0%

Q2  reportado en 1:1              48         62      +29%
Q2  reportado en 2P               59         30      -49%
Q2  en 2P con `nivel alta`         8          2
Q2  condicionado a Q1          21.8%      31.3%
```

**La caída de Q1 no es una pérdida.** De las 11 estrella-sector que dejaron de
tener el período publicado entre sus candidatos, **10 no tenían potencia ahí**
(SNR mediana 1.07); la restante es TIC 449459169 s38, donde el fundamental del
ACF (0.9120) difiere del publicado (0.96) en exactamente 5.0% y queda justo
afuera de la tolerancia. Q1 mide "arrastramos un candidato compatible", no
"detectamos la señal": bajar a costa de dejar de arrastrar candidatos vacíos es
lo que se quería.

**Lo que el colapso NO arregla** es la regla de selección. Q2 sigue en 31.3% y
el modo de falla dominante ya no es 2P (30) sino `otro` (105 de 198): la regla
elige algo que no es ni el período publicado ni un armónico simple suyo. Eso es
el punto 3 de §7 y sigue abierto.

### 10. La métrica corregida (`score_periods.py`)

`match_periods.py` puntúa binario contra `period_gold`, y eso hace dos supuestos
que la muestra no sostiene. Los dos inflan el error, y los dos se arreglan sin
tocar el pipeline.

**Supuesto 1: que el período publicado existe.** Ya salían SLF y NOISY. Faltaba
sacar las **Be**, cuyo `period` es `1/fg1` de Labadie-Bartz+ 2022 — "Central
frequency of g1", el centro de un GRUPO de frecuencias no resueltas, no una
señal coherente (§5, y el prewhitening lo reproduce). Eran **40 de los 80 TIC
usables y 119 de las 316 estrella-sector**: la mitad del test medía otra cosa.
Quedan marcadas en `period_is_group` en `golden_truth.csv` y se pueden seguir
mirando aparte. Con eso `period_usable` baja a 40 TIC (PULS 19, ROT 15, ECL 4,
AMBIGUOUS 2) y Q1 sube de 84.2% a **90.9%**.

Y no alcanza con filtrar por clase: sondeando el fundamental del período
publicado en cada estrella-sector, en **53 de 144 no hay potencia ahí**
(PULS 35 de 78). El período está publicado para la estrella, no para este
sector. Contra un número que no está en los datos no se puede acertar, así que
esas salen del denominador — igual que SLF.

**Supuesto 2: que reportar 2P es equivocarse.** Una curva de doble onda es
fotométricamente **idéntica** plegada en P y en 2P: si los dos semiciclos son
iguales, los armónicos impares valen cero y **no existe medición que separe las
dos hipótesis**. La única evidencia posible a favor de doblar es que los
semiciclos difieran — eclipses de distinta profundidad, dos grupos de manchas
desiguales — o sea potencia en el fundamental de 2P. Por eso el armónico se
adjudica con la sonda y no con el paper. Ojo con la conclusión de §4: que el
fundamental nuestro esté vacío **no prueba que el paper tenga razón**, prueba
que no hay evidencia para doblar y que la parsimonia se queda con el publicado.
La diferencia importa porque decide en qué casilla cae, no cuál gana.

Cinco veredictos, y los tres del medio son los que antes se contaban todos como
error:

```
acierto          1:1 con el publicado
doblado_ok       armonico del publicado, y NUESTRO fundamental tiene SNR >= 4
doblado_sin_ev   armonico del publicado, sin potencia propia: misma senal,
                 armonico equivocado (no es un error de deteccion)
fallo            otra frecuencia, y el publicado si esta en los datos
sin_senal        el publicado no tiene potencia en este sector -> fuera
```

Sobre las 144 estrella-sector reportadas con período comparable (36 TIC):

```
class_gold  acierto  doblado_ok  doblado_sin_ev  fallo  sin_senal  total
AMBIGUOUS         1           0               2      2          1      6
ECL               4           7               0      2          1     14
PULS             19           0               4     20         35     78
ROT              14           1               6      9         16     46
todas            38           8              12     33         53    144

sobre las 91 donde el periodo publicado ESTA en los datos:
  periodo correcto                     46   50.5%
  la senal correcta (algun armonico)   58   63.7%
```

Contra el 31.3% de §9. La diferencia no es una recalibración: son 53 filas donde
la pregunta no estaba definida y 20 donde la respuesta era "indecidible" o
"nuestra".

El `doblado_ok` se concentra donde tenía que concentrarse: **7 de las 8 son
ECL**, y en ECL nuestro fundamental tiene señal en 93% contra 36% del publicado.
Ahí el doblado es la corrección de manual — el paper reporta el fotométrico y
nosotros el orbital. Fuera de ECL el doblado no tiene respaldo en 12 de 13.

El corte SNR ≥ 4 no es un borde:

```
corte   decididas   correcto   senal correcta   doblado_ok
>=3           104      47.1%            57.7%           11
>=4            91      50.5%            63.7%            8
>=5            77      55.8%            72.7%            5
>=6            71      57.7%            77.5%            3
```

Subir el corte sube las dos tasas porque saca sectores marginales, no porque
mueva la adjudicación.

**Lo que esto NO rescata** son los 33 `fallo`: reportamos una frecuencia sin
relación con la publicada estando la publicada en los datos. Ese es el modo de
falla real y es el mismo de §7 punto 3 — la regla ordena por apariencia de
imagen y nunca pregunta si la frecuencia existe. La corrección de la métrica no
lo toca.

Para correrlo:

```bash
python scripts/golden/build_truth.py            # period_usable ya sin Be
python scripts/golden/build_probe_targets.py    # -> results/golden/probe_targets.csv
python scripts/golden/prewhiten.py --all --veredicto results/golden/probe_targets.csv --out-dir results/golden/probe_all
python scripts/golden/score_periods.py          # -> results/golden/score_periods.csv
```

`build_probe_targets.py` existe porque `prewhiten.py` se manejaba con
`veredicto_2P.csv`, que es por construcción el conjunto donde el pipeline ya
había fallado: sondear solo ahí es el mismo error de condicionar en el fracaso
que se corrigió en §7.

## La cadena de 4 clases

`docs/PLAN_clasificacion_4clases.md`, implementado. Cuatro clases —ECL, ELL,
Pulsante, Irregular— y dos preguntas separadas: el prewhitening contesta si la
frecuencia existe y cuántas hay, la CNN qué forma tiene el fold que sobrevive.

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate CNN_TESS
export PYTHONPATH=src

# 1. descriptores de estructura + triage      -> descriptores.csv
python scripts/step_descriptores.py --lc results/golden/lc.parquet \
    --out results/golden/descriptores.csv \
    --componentes results/golden/descriptores_componentes.csv

# 2. sonda del fundamental de CADA candidato  -> probe_peaks_all.csv
python scripts/probe_peaks.py \
    --clasificacion results/golden/clasificacion_collapsed.csv \
    --lc results/golden/lc.parquet --out results/golden/probe_peaks_all.csv

# 3. veto + orden por SNR + adjudicacion P/2P -> clasificacion_veto.csv
python scripts/step_clasificar_una_red.py results/golden/cnn_input_collapsed.npz \
    results/golden/cnn_probs_collapsed.npz results/golden/clasificacion_veto.csv \
    --probe results/golden/probe_peaks_all.csv --curves results/golden/curves.pkl

# 4. la clase final                            -> clase_final.csv
python scripts/step_clase_final.py \
    --clasificacion results/golden/clasificacion_veto.csv \
    --descriptores results/golden/descriptores.csv \
    --out results/golden/clase_final.csv \
    --out-estrella results/golden/clase_final_estrella.csv

# tests 1 y 3
python scripts/golden/validar_4clases.py
# test 2: volver a sondear el periodo NUEVO contra el publicado
python scripts/golden/build_probe_targets.py \
    --clasificacion results/golden/clasificacion_veto.csv
python scripts/golden/prewhiten.py --all \
    --veredicto results/golden/probe_targets.csv --out-dir results/golden/probe_all
python scripts/golden/score_periods.py
```

`step_descriptores.py` emite el triage en dos columnas y hay que elegir una con
`--estructura-col`:

- `estructura` es la regla literal del plan — irregular si TODAS las
  componentes caen en un mismo grupo no resuelto.
- `estructura_apinado` relaja el "todas": basta con que ALGÚN grupo tenga 3
  componentes dentro de una resolución de Rayleigh.

### Lo que salió, corrido el 2026-09-05

810 estrella-sector, 2586 componentes extraídas, 9651 candidatos sondeados.

```
estructura        regla literal   con `estructura_apinado`
multiperiodica          560              517
sin_senal               128              128
uniperiodica            119              119
irregular                 3               46
```

**La rama `irregular` de la regla literal casi no se dispara, y no atrapa el
caso que la motiva.** El plan pide "todas las componentes en UN cluster con
cluster_size >= 3"; TIC 42889751 tiene un grupo de 3 componentes dentro de una
resolución de Rayleigh más 3 sueltas —4 grupos— y cae en `multiperiodica` en
los dos sectores. `estructura_apinado` relaja el "todas" y sí lo atrapa. Cuál
de las dos usar es una decisión abierta: sobre el test 3 la diferencia es un
empate (BE→Irregular 22.2% → 25.9%, PULS→Pulsante 70.2% → 63.0%).

El veto:

```
candidatos que pasan SNR >= 4              2946/9651   30.5%
estrella-sector sin ningun candidato          91/806
estrella-sector que reportan periodo         715/806
de esos, adjudicados a 2P                    122/715
```

Clase final (regla literal, folds crudos): ECL 11, ELL 9, Pulsante 554,
Irregular 236 sobre 810 estrella-sector; 574 períodos reportados.

**Test 1 — sanidad.** 49.0% -> **100.0%**. Por construcción, pero es la prueba
de que el veto está conectado. Los reportes bajan de 463 a 574 sobre un
denominador distinto: la regla vieja sólo reportaba si la clase era periódica,
la nueva no mira la clase.

**Test 2 — `score_periods.py`.**

```
                                    antes           ahora
periodo correcto              46/91  50.5%    121/140  86.4%
la senal correcta (armonico)  58/91  63.7%    121/140  86.4%
fallo                            33               19
```

Los denominadores no son el mismo conjunto —el veto cambia qué
estrella-sector reportan— así que la comparación es entre pipelines, no entre
dos medidas de las mismas 91 filas. `doblado_sin_ev` desaparece por
construcción: nuestro fundamental siempre tiene SNR >= 4, así que un armónico
del publicado es siempre `doblado_ok`.

**Test 3 — la clase contra la etiqueta del paper**, que no usa ningún período
publicado:

```
                        estrella-sector        por TIC
BE     -> Irregular      42/189   22.2%       3/61    4.9%
SLF    -> Irregular      22/30    73.3%       6/12   50.0%
PULS   -> Pulsante      292/416   70.2%      15/29   51.7%
ECL    -> ECL             7/48    14.6%       2/5    40.0%
```

SLF y PULS salen donde el plan predice. **Las Be no**: 145 de 189 caen en
`Pulsante`, porque tienen dos o más grupos resueltos y algún candidato con
fundamental real, así que el triage las llama multiperiódicas. Que la clase
`Irregular` recoja las Be era el objetivo declarado del plan y **no se cumple**
con el triage tal como está escrito.

El 14.6% de ECL es engañoso y no hay que citarlo: 30 de las 48 estrella-sector
ECL son **una sola estrella**, TIC 30317301, cuyo fold crudo la red lee `LPV`
en casi todos sus sectores. Por TIC son 2 de 5.

**El riesgo de ROT no se materializó.** El plan esperaba que las 15 ROT
cayeran en ELL; caen 41 estrella-sector en `Pulsante`, 26 en `Irregular`, 1 en
`ECL` y **ninguna en ELL**. La ELL final son 9 estrella-sector y vienen de PULS
(4), BE (2), AMBIGUOUS (2) y SLF (1). La contaminación ELL que motivó el plan
se apagó en el veto: el 79.4% de los 141 picos que la regla vieja reportaba
como ELL no tenían fundamental, y los reportes ELL caen a 11 picos y 9
estrella-sector.

**Crudo vs aislado: el test 2 ya no puede decidirlo.** Con el veto puesto el
período se elige por SNR del fundamental y la CNN deja de participar en la
selección, así que las dos variantes reportan exactamente los mismos 715
períodos y el test 2 da idéntico. La única diferencia está en la clase:

```
                     crudo   aislado
ECL final               11        17
ELL final                9        21
  de esas, sobre PULS     4        12
Pulsante               554       536
```

Aislar recupera eclipsantes (ECL sube de 2/5 a 3/5 TIC) **y triplica la ELL
sobre pulsantes**, que es exactamente la contaminación que el plan quiere
apagar. Con `Number_ELL` sobre folds crudos la ELL queda en 9 filas; recomendar
aislar por defecto necesitaría un argumento mejor que un ECL más sobre n=5.

Archivos: `descriptores.csv`, `descriptores_componentes.csv`,
`probe_peaks_all.csv`, `clasificacion_veto.csv`, `clase_final.csv`,
`clase_final_estrella.csv`, `score_periods_veto.csv`, y la variante aislada en
`peaks_aislados.parquet` / `clase_final_aislados.csv`.

### Decisiones que quedaron abiertas

1. **Qué hace `Irregular` con las Be.** Hoy 145 de 189 estrella-sector Be caen
   en `Pulsante`. O el triage se endurece (¿cuántos grupos, de qué ancho, hacen
   una Be?), o la clase `Irregular` no es la que recoge las Be y hay que
   decirlo.
2. **`estructura` o `estructura_apinado`.** Ver arriba: empate en el test 3.
3. **Crudo o aislado.** Ver arriba; el test 2 no arbitra.
4. **La uniperiódica cuya red dice LPV/Rndm.** El plan no la define; hoy cae
   en `Irregular` (67 + 7 estrella-sector). La alternativa es `Pulsante`, que
   contradice la definición ("multiperiódica coherente").
5. **ROT.** El riesgo previsto no apareció, así que la salida barata —aceptar
   que ELL incluye rotación y reportar la fracción— hoy se reporta sola: 0 de
   68 estrella-sector ROT caen en ELL.

Los pasos 1 y 2 corren el prewhitening por separado sobre las mismas curvas
(~5 y ~20 minutos sobre las 810 estrella-sector de la golden). Es trabajo
duplicado y se puede fusionar si alguna vez molesta; hoy la separación deja
que `probe_peaks.py` siga sirviendo para auditar la cadena vieja.

## Qué queda

- **Etapa 1** — adjudicar el 2P. Ya no bloquea la métrica (§10 lo adjudica con
  la sonda y deja `doblado_sin_ev` como categoría propia). El PDF sigue abierto para las 18 TIC sin
  contestar, pero la sonda del fundamental ya lo adjudica sola en las 59: ECL
  73% contra BE/ROT/PULS 3%. Lo que falta del PDF es confirmar a ojo las 10 con
  potencia real, no las 49 sin.
- **Etapa 2** — el test de armónico ya está escrito (`msv.prewhiten.probe`).
  Falta meterlo en la regla de selección: rechazar el candidato cuyo
  fundamental no llega a SNR 4, y comparar contra `min p_LPV`, `max prob`,
  potencia y prominencia sobre la misma tabla.
- **Etapa 3** — recalibrar `-log10 p_LPV > 12`, `irregular < 0.05` para ELL y el
  piso de 4 ciclos con la ROC de la muestra externa.
- **Etapa 4** — multi-sector: 42 estrellas con >= 5 sectores, 17 con >= 10.
  Concatenar y ver si `log_pLPV` sube al ganar ciclos. Rompe la circularidad del
  piso de 4 ciclos y no depende de la golden.
- **Etapa 5** — el plot amplitud vs período.

## Notas de dato

`period_usable` en `golden_truth.csv` saca del test de período SLF, NOISY, **Be**
y las estrellas donde dos papers difieren más de 5%. El criterio es el mismo en
los cuatro casos y no es dificultad: son filas donde "acertar el período" no es
una pregunta con respuesta. En variabilidad estocástica un "período" es una
escala de tiempo; en las Be es el centro de un grupo de frecuencias
(`period_is_group`, §10); con dos papers en desacuerdo no hay contra qué
arbitrar. El filtro por clase no basta — hay que sondear sector por sector, y
53 de 144 no tienen potencia en el período publicado (§10).

Tolerancia del armónico: **5% relativo sobre el cociente**, igual que
`analysis_ell/_common.load_peak_truth`. Con tolerancia absoluta de 0.05 sobre un
cociente de 2 el test queda en 2.5% y se pierden 15 de los 59 alias.

Los FITS: 810 en `~/Dropbox/MassiveStarVariability/raw/golden/`, hidratados
(432 MB). Para revertir el movimiento:

```bash
mv ~/Dropbox/MassiveStarVariability/raw/golden/*.fits ~/Dropbox/MassiveStarVariability/raw/download_paralell/ && rmdir ~/Dropbox/MassiveStarVariability/raw/golden
```

## Composición de la muestra

```
              con curva   con período   period_usable   sectores (mediana)
BE                   61            40              40          2
PULS                 29            19              19          8
ROT                  15            15              15          4
SLF                  12            12               0          2
AMBIGUOUS             8             2               2        6.5
ECL                   5             4               4          3
OTHER                 4             0               0        2.5
```

La única ELL de la muestra quedó etiquetada ECL al colapsar por prioridad de
vocabulario. **La golden no sirve para validar la clase ELL directamente** —
sirve para mostrar dónde se mete, que es el 2P sobre ROT/PULS/BE.
