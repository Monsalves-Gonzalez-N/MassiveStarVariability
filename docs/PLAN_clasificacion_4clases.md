# Plan: clasificar en 4 clases con prewhitening + CNN

Decidido el 2026-09-05. Reemplaza la cadena actual de selección de período, que
elige el candidato de mayor `log_pLPV` sin preguntar nunca si la frecuencia
existe.

## Objetivo

Cuatro clases y nada más:

```
ECL          eclipsante
ELL          elipsoidal
Pulsante     multiperiódica coherente
Irregular    sin período: grupos de frecuencias, ruido rojo, Be
```

Las Be y la variabilidad estocástica de baja frecuencia **no se clasifican**:
caen en `Irregular` por construcción y no se les reporta período. Eso es una
decisión de alcance, no una limitación — en variabilidad estocástica un período
no es una cantidad que se pueda acertar o errar.

## La estructura de la decisión

El error de la versión actual es pedirle a la CNN que elija el período. La red
se entrenó con OGLE para contestar "¿qué forma tiene este fold?", y sobre eso
acierta: en los 16 picos con período confirmado a ojo da 11/11 E y 3/3 ELL. Lo
que no sabe es si la frecuencia existe — medido sobre 6488 candidatos, un pico
que llama ELL tiene MENOS potencia real (76% con el fundamental vacío) que uno
que llama LPV (61%), y ninguna agregación de sus probabilidades correlaciona con
el número de componentes de la estrella (rho ≈ 0).

Entonces:

```
prewhitening  ->  ¿hay período? ¿cuántos?     (existencia y multiplicidad)
CNN           ->  ¿qué forma tiene ese fold?  (ECL vs ELL vs pulsante)
```

Ninguna de las dos hace el trabajo de la otra.

## Costo

Medido sobre las curvas reales de la muestra golden:

```
                                        por estrella-sector
clean_lightcurve                             < 10 ms
extract_components (2-6 componentes)      60 - 450 ms
probe, por candidato                       11 -  44 ms
espectro completo, N=11763, 60207 freqs        39 ms

total con 14 candidatos                   ~0.25 s tipico, ~1.1 s peor caso
```

Mil estrella-sector son cuatro minutos. No es el régimen del GP que se descartó
al principio: aquello es O(N³) sobre >10⁴ puntos, esto son ~6 iteraciones de
Lomb-Scargle. Si en algún momento hiciera falta, cortar la grilla en 25 c/d
(las clases del paper viven por debajo de 10) baja el espectro de 39 a 11 ms;
hoy no hace falta.

## Los pasos

### 1. Prewhitening por estrella-sector — EXISTE

`msv.prewhiten.extract_components`. Extrae la frecuencia más fuerte con sus 3
armónicos, la resta, y repite sobre el residuo hasta SNR 4. Se ajusta el modelo
armónico completo y no una sinusoide sola: a un eclipse al que se le resta solo
el fundamental le queda la estructura armónica en el residuo, y vuelve en las
iteraciones siguientes como componentes espurias en 2f y 3f — que es
exactamente el peine que se está tratando de matar. Es el motivo por el que no
se usa SMURFS, que ajusta una sinusoide por iteración.

Salida por componente: frecuencia, amplitud, armónicos, SNR, y el modelo
evaluado en el tiempo.

### 2. Descriptores de estructura — FALTA UN SCRIPT

De las mismas componentes, sin costo extra:

- `cluster_components(components, 1/T)`: etiqueta como un solo grupo las
  componentes separadas por menos de una resolución de Rayleigh. Dos frecuencias
  a menos de 1/T no están resueltas: lo que hay es UNA estructura ancha, no dos
  períodos.
- `n_clusters` = número de grupos resueltos.
- `cluster_size` = cuántas componentes tiene cada uno.
- `coherent_fraction` = varianza explicada por las componentes / varianza total.

Falta un script que emita esto para toda la muestra. `prewhiten.py` ya lo
calcula pero está orientado a revisión visual (pliega, arma parquet de picos,
guarda curvas); hace falta un modo que solo escriba los descriptores.

### 3. Triage de estructura — FALTA

```
sin componentes sobre SNR 4                    -> sin variabilidad coherente
todas las componentes en UN cluster
  con cluster_size >= 3                        -> IRREGULAR
>= 2 clusters resueltos                        -> multiperiódica
1 cluster resuelto (1-2 componentes)           -> uniperiódica
```

Esta es la parte que la CNN no puede hacer y el prewhitening sí. También es la
que separa `Irregular` de `Pulsante`, que era la duda abierta: varias
frecuencias SEPARADAS son modos independientes; varias frecuencias APIÑADAS son
una joroba ancha que el propio prewhitening cortó en pedazos. En TIC 42889751 la
separación mediana entre componentes vecinas es 1.01 Rayleigh en los dos
sectores — no son cinco modos, es un grupo.

### 4. Veto del fundamental — FALTA

`probe_peaks.py` (escrito, `results/golden/probe_peaks.csv`) da por candidato el
SNR del fundamental y `a2_a1`. La regla:

```
1. descartar todo candidato con SNR < 4
2. de los que sobreviven, reportar el de mayor SNR (no el de mayor log_pLPV)
3. si no sobrevive ninguno, la estrella no reporta período
```

El punto 3 es una respuesta que hoy no existe y hace falta.

**El veto no es opcional**: ELL es una de las cuatro clases y hoy el 76% de los
picos que la red llama ELL no tienen fundamental. Medido sobre la golden, los
reportes ELL caen de 141 a 42 (−70%) bajo una regla que nunca mira la clase.

Nota sobre el SNR: es `A₁ / mediana del espectro del residuo en ±1 c/d`,
convención de Breger+ 1993 con la mediana en lugar de la media. La media es lo
literal de Breger, pero se infla con las frecuencias reales que quedan dentro de
la ventana: medido, cambiar a media pierde 21 de 146 detecciones reales (−14%),
casi todas en PULS (69.4% → 55.2%) y ninguna en ECL. La AUC real-vs-espurio es
la misma (0.779 vs 0.776), así que es un punto de operación más estricto sobre
la misma curva, no una mejora. Se documenta que el umbral 4 con mediana equivale
a ~3.76 en unidades estrictas de Breger.

### 5. La CNN sobre lo que sobrevive — EXISTE, FALTA UNA DECISIÓN

Se pliega la componente dominante (y las demás que pasen el veto) y la red da
E / ELL / Pulsating / LPV / Rndm.

Decisión abierta: fold sobre la curva cruda o sobre la curva con las otras
variaciones removidas (`isolate`, conservando las conmensurables). Medido:
aislar sube la mediana de `log_pLPV` de 0.27 a 1.03 en las 15 estrella-sector
multiperiódicas, y la baja de 1.07 a 0.82 sobre el conjunto completo de 59.
Ayuda donde hay varias variaciones y estorba donde no — lo que sugiere aislar
solo cuando el triage del paso 3 dice multiperiódica.

### 6. Adjudicación P vs 2P — REGLA ESCRITA, FALTA APLICARLA

Para la componente reportada, comparar el SNR del fundamental en P y en 2P:

```
SNR(fundamental en 2P) >= 4   -> reportar 2P  (eclipses desiguales: el orbital)
si no                         -> reportar P   (parsimonia)
```

Limitación que hay que dejar escrita en el paper: **una doble onda simétrica es
fotométricamente idéntica en P y en 2P**. Si los dos semiciclos son iguales los
armónicos impares valen cero y ninguna medición separa las hipótesis. La regla
elige el período parsimonioso cuando los datos no distinguen; no es una prueba
de que 2P esté mal.

Medido sobre 709 picos periódicos que fallan el veto: 25.7% son el doblado
(reportamos 2× el real), 5.8% el halvado, y **68.5% no tienen señal ni en f, ni
en 2f, ni en f/2**. El problema grande no es el armónico, es el vacío.

### 7. Clase final — FALTA

```
sin candidato que pase el veto                       -> Irregular (sin período)
triage = un grupo apiñado                            -> Irregular
la CNN dice E sobre la componente dominante          -> ECL
la CNN dice ELL sobre la componente dominante        -> ELL
triage = multiperiódica y la CNN no dice E/ELL       -> Pulsante
```

ECL/ELL tienen prioridad sobre Pulsante cuando el fold de la dominante tiene esa
forma, por el mismo criterio de vocabulario que usa `build_truth.py`: es la
etiqueta físicamente más específica, y una eclipsante con pulsaciones sigue
siendo una eclipsante.

## Validación

Tres tests, en orden de independencia:

1. **Sanidad, por construcción.** El % de períodos reportados con fundamental
   SNR ≥ 4 tiene que pasar de 49% a ~100%. Si no, el veto está mal conectado.
2. **La golden, con `score_periods.py`.** Hoy 50.5% de período correcto y 63.7%
   de señal correcta sobre las 91 estrella-sector donde el período publicado
   está en los datos. Ver si sube.
3. **El test externo que NO depende de ningún período publicado**: de los 134
   TIC con curva, la golden tiene 61 BE y 12 SLF. **Qué fracción cae en
   `Irregular`.** Ese test no usa períodos de la literatura, solo la etiqueta de
   clase, y mide justamente la clase nueva. Simétricamente, 29 PULS y 5 ECL
   deberían caer en `Pulsante` y `ECL`.

## Riesgo conocido: no hay bucket para ROT

La golden tiene 15 ROT con curva, y la rotación es uniperiódica y casi
sinusoidal: **va a caer en ELL**. Es exactamente el canal por el que entra la
contaminación que motivó todo esto — de las 19 estrella-sector donde nuestra
clase es ELL sobre 2P, el paper las llama ROT, PULS o BE.

Dos salidas, hay que elegir una antes de escribir el paper:

- **Agregar ROT como quinta clase.** Necesita un discriminante que la CNN no
  tiene, y el único que se conoce es multi-sector: una señal rotacional vuelve
  en la MISMA frecuencia con la misma amplitud en cada sector; un grupo de
  frecuencias devuelve la envolvente y no sus miembros. Hay 42 estrellas con ≥5
  sectores y 17 con ≥10.
- **Aceptar que ELL incluye rotación** y decirlo con el número: qué fracción de
  las ROT de la golden cae en ELL. Es honesto y no requiere trabajo nuevo.

## Orden de trabajo

```
1. script de descriptores (paso 2)                      -> descriptores.csv
2. triage de estructura (paso 3)                        -> columna `estructura`
3. veto + orden por SNR en step_clasificar_una_red.py   -> paso 4
4. correr los tres tests de validación
5. decidir crudo vs aislado (paso 5) con el test 2
6. decidir qué hacer con ROT
```

Los pasos 1-3 son el trabajo real; el 4 es correr lo que ya existe.

## Lo que NO se hace

- No se instala SMURFS ni Pyriod. El prewhitening propio ajusta armónicos, que
  es un requisito de una muestra con eclipsantes. Si el referee lo pide, se
  valida contra SMURFS sobre 20-30 estrellas y se cita el acuerdo.
- No se toca el FAP de la LS ni la banda de Bartlett del ACF. No filtran estos
  picos porque no hacen esta pregunta: un máximo del ACF en 2P es correlación
  REAL, no una falsa alarma, y el FAP de Baluev es contra ruido blanco mientras
  estas estrellas tienen ruido rojo. El arreglo es el colapso del peine
  (`collapse_comb.py`, hecho) y el veto, no un umbral más apretado.
- No se usa `irregular` (el p_LPV medio) como medida de multiperiodicidad. Mide
  estocasticidad, no cuenta frecuencias: `irregular` bajo quiere decir "mucha
  señal coherente", que es una pulsante multiperiódica.

## Estado de la implementación — 2026-09-05

Implementado y corrido sobre la golden. Los números están en
`scripts/golden/README.md`, sección "La cadena de 4 clases"; acá sólo qué
código es cuál y en qué se apartó del plan.

```
paso 2  descriptores       src/msv/structure.py + scripts/step_descriptores.py
paso 3  triage             msv.structure.triage -> columna `estructura`
paso 4  veto + orden SNR   scripts/step_clasificar_una_red.py --probe
paso 5  crudo vs aislado   scripts/build_folds_aislados.py
paso 6  P vs 2P            step_clasificar_una_red.py -> `armonico`, `per_reportado`
paso 7  clase final        scripts/step_clase_final.py
tests 1 y 3                scripts/golden/validar_4clases.py
test 2                     build_probe_targets.py -> prewhiten.py -> score_periods.py
```

Los tres tests:

```
1  periodos reportados con fundamental SNR >= 4    49.0% -> 100.0%
2  periodo correcto (score_periods.py)             50.5% ->  86.4%
3  BE  -> Irregular    22.2%      SLF  -> Irregular  73.3%
   PULS -> Pulsante    70.2%      ECL  -> ECL        14.6%  (n=5 TIC)
```

Cuatro puntos donde el plan no alcanzaba y hubo que decidir:

1. **La rama `irregular` del triage no se dispara** (3 de 810) y no atrapa
   TIC 42889751, que es el caso que la motiva: tiene un grupo de 3 componentes
   no resueltas más 3 sueltas, o sea 4 grupos, y "todas las componentes en UN
   cluster" es falso. Se dejó la regla literal como default y la variante
   —basta con que ALGÚN grupo tenga 3— en la columna `estructura_apinado`,
   seleccionable con `--estructura-col`. En el test 3 empatan.

2. **`Pulsante` se decide por estructura, no por la forma del fold.** El paso 7
   dice "triage = multiperiódica y la CNN no dice E/ELL -> Pulsante", y eso
   incluye a la red diciendo `LPV`, que es lo que dice en 406 de 810
   estrella-sector sobre el candidato que el veto elige. Exigir `Pulsating`
   manda a `Irregular` casi todo lo que tiene señal (Pulsante 554 -> 142); está
   en `--cnn-estricta` por si alguna vez se quiere.

3. **La uniperiódica cuya red dice LPV/Rndm no está en el plan.** Cae en
   `Irregular` (74 estrella-sector), marcada en `regla`. `Pulsante` no le
   corresponde: la clase está definida como multiperiódica coherente.

4. **El paso 5 ya no se puede decidir con el test 2.** Con el veto puesto la
   CNN no participa en la selección del período, así que crudo y aislado
   reportan los MISMOS 715 períodos y el test 2 da idéntico. Sobre la clase,
   aislar sube ECL de 11 a 17 estrella-sector y ELL de 9 a 21, de las cuales
   las que están sobre estrellas PULS pasan de 4 a 12. Sigue abierto, pero el
   argumento a favor de aislar ahora tiene que ser sobre la clase y pagar esa
   contaminación.

Y el riesgo que no apareció: **ninguna de las 68 estrella-sector ROT cae en
ELL** (41 `Pulsante`, 26 `Irregular`, 1 `ECL`). La decisión sobre ROT no urge.

Lo que sí quedó sin cumplirse es el objetivo declarado de la clase `Irregular`:
**145 de 189 estrella-sector BE caen en `Pulsante`**, porque tienen dos o más
grupos resueltos y algún candidato con fundamental real. Endurecer el triage
para las Be es el trabajo que sigue.
