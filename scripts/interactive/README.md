# Banco interactivo de whitening

Corre el camino de `run_peaks.py` — `ls_periodogram` + `acf_periodogram`,
`select_peaks`, `candidate_periods` — clasifica **cada candidato** con la CNN,
y deja blanquear de la curva los que uno decida para volver a correr el
periodograma sobre lo que queda.

El motivo: la CNN fue entrenada con estrellas cuya curva **es** una sola
variación. En una multiperiódica el fold en cualquier período arrastra las
otras como dispersión, y la red lee esa dispersión como `LPV`. Sacar lo que ya
se juzgó incoherente deja ver lo que había debajo.

## Cómo correrlo

```
MSV_WEIGHTS=$HOME/ViT_VariableStars/pretrained/keras_checkpoints PYTHONPATH=src /opt/anaconda3/envs/CNN_TESS/bin/python -m streamlit run scripts/interactive/app.py
```

## Por qué son dos procesos

Los envs de este Mac no se pueden juntar: `CNN_TESS` tiene astropy, scipy y
statsmodels pero no TensorFlow, y `tf_env` tiene Keras 3 pero no puede
`import msv`. Todo hasta la imagen 32×32 es numpy puro, así que lo único que
cruza es `model.predict`.

```
app.py (CNN_TESS)  ──32x32 por stdin──►  predict_server.py (tf_env)
  plotly, msv      ◄──probs por stdout── checkpoint cargado una sola vez
```

Arranque completo 4.7 s; cada predict después, ~70 ms.

`predict_server.py` duplica `CLASS_NAMES` y la arquitectura de
`msv.classify_brf.make_model` en vez de importarlas, porque no puede importar
`msv`. **Si cambia la arquitectura allá, hay que espejarla acá** o el
checkpoint deja de cargar posicionalmente.

## Qué muestra

1. **Los dos periodogramas**, eje de período log: LS con su `window` y su nivel
   de FAP de Baluev, ACF con su banda de Bartlett.
2. **Cada candidato como marcador coloreado por la clase que le dio la red**, y
   con la forma según el `kind` de `candidate_periods` (estrella =
   fundamental, triángulo = subarmónico, círculo = aislado). Ahí se ve el
   periodograma con cada pico etiquetado `E` / `ELL` / `LPV` / `Rndm`.
3. **La curva de luz**, original en gris y blanqueada en negro.
4. **Tabla editable de candidatos** con `ciclos`, `snr` del fundamental,
   clase, `p_clase` y `-log10 p_LPV`. La columna `blanquear` viene pre-marcada
   en los `Rndm` y `LPV`, y se corrige a mano — ese checkbox es el punto.
5. **Blanquear y recalcular**: resta fundamental + armónicos en los períodos
   marcados, recalcula los dos periodogramas sobre la curva blanqueada,
   reselecciona picos y reclasifica. Con deshacer y reset.
6. **Fold + entrada 32×32 + probabilidades** del pico seleccionado.
7. **Veredicto** a `results/golden/veredicto_whitening.csv`, una fila por
   candidato por iteración.

## Lo que el lazo hace mal si nadie lo frena

Probado sobre **TIC 337886863** (eclipsante, s16+17+18, 75.7 d, 27.5 ciclos del
período publicado), blanqueando automáticamente todo lo `Rndm`/`LPV`:

**Iteración 0.** El ACF arma el peine y lo colapsa bien:

```
5.692061 fundamental ACF  snr  5.5  P/Ppub 2.0668   E  p=1.000
2.846030 subarmonico ACF  snr 20.2  P/Ppub 1.0334   E  p=0.998
```

Encuentra el período publicado (2.846 contra 2.754 d, +3.3%) y su alias 2P, y
la red dice `E` en los dos. **La sonda los separa limpio: SNR 20.2 en el
verdadero contra 5.5 en el doblado** — ordenar por SNR habría elegido bien.

El LS en paralelo colapsó a **otro fundamental**: 0.9483 d = P/3, y 0.7117 =
P/4. La potencia de una eclipsante se va a los armónicos, no al orbital, así
que `label_harmonics` declaró fundamental a un armónico. **Las dos fuentes se
contradicen y nadie las reconcilia**, porque cada una pasa por
`candidate_periods` por separado.

**Iteración 2, sin guard.** El lazo blanqueó 0.474, 0.948, 0.712 y 1.140 —
todos armónicos del orbital verdadero, porque la red ve cada fold por separado
y los llama `LPV`/`Rndm`. Le arrancó pedazos al eclipse:

```
2.846962 LS   ELL  p=0.702
2.846734 ACF  LPV  p=0.627
```

Lo que era `E` con p=1.000 quedó `ELL` y `LPV`. **Sacarle los armónicos a un
eclipse deja una onda suave con dos máximos por ciclo, que es exactamente una
elipsoidal: es el mecanismo de la contaminación ELL reproducido en vivo.**

**Con el guard** (`parte_de`, abajo) el eclipse sobrevive las tres vueltas:
`E` con p = 0.998 / 0.994 / 0.996 y SNR estable en 20-23.

## Las tres defensas de la tabla

- **`parte_de`** — el período coherente del que este candidato es armónico,
  vía `prewhiten.commensurate` con `max_order=8` (el 4 por defecto de `msv` no
  reconoce un P/6, y el peine de una eclipsante llega hondo). Un armónico de
  una variación real no se pre-marca: blanquearlo no saca una variación, le
  arranca un pedazo a esa.
- **`ciclos`** = baseline/P. Abajo de 4 no hay período que acertar.
- **`snr`** del fundamental vía `prewhiten.probe`. Abajo de 4 (Breger) la
  frecuencia no se distingue del ruido. Es la pregunta que el pipeline nunca
  hace: la clase se lee de un fold que existe haya o no haya señal.

Ninguna filtra sola — están para decidir cuándo parar, y la selección sigue
siendo manual.

## Sobre TIC 464295672 = HD 90273 (la estrella anterior)

Era `SLF+SPB?` de Burssens+2020, o sea que el token primario es **SLF, que por
definición no es periódico**, y su `period` es `1/nu_dom`, que
`GOLDEN_SAMPLE.md` §3 ya advierte que no es el período de clasificación.

Nuestro LS igual lo encuentra: primer candidato sobre s9, P = 1.71365 d contra
1.72117 publicado, 0.4%. Sale `LPV` p = 0.994 — **y esa clase es correcta**,
porque no hay período que encontrar. Con el criterio de Breger la estrella no
tiene señal coherente en ninguna de sus tres épocas (0 componentes sobre SNR 4;
el `nu_dom` da SNR 2.09 / 0.22 / 1.85).

Sirve como control negativo. Y destapó un defecto del benchmark: `period_usable`
saca las SLF puras pero no las compuestas, porque la prioridad
`ECL > ELL > BE > ROT > PULS > SLF` colapsa `PULS|SLF` a `PULS`. **21 de las 80
estrellas del test de período (26%) entran con una "verdad" que es el `nu_dom`
de una estrella que el mismo paper llama estocástica.**
