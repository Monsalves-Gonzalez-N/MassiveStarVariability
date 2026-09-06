# Handoff: contaminación de la clase ELL

Sesión del 2026-09-04. Reescribe la versión del 2026-09-03, que contenía tres
errores (§0). Punto de partida: muchas estrellas sin período coherente reciben
clase ELL con probabilidad alta.

## Estado en una línea

La señal está en **`p_LPV` leída en la CNN**, antes del BRF, y con **un solo
checkpoint**: `Number_ELL` bajo `log`. Da AUC 0.988 a nivel de estrella,
acierta la clase 15/16 (el ensemble de 7 más el BRF acierta 14/16) y, usada
para elegir qué pico reportar, sube de 8/11 a 10/11 estrellas con el período
correcto. El BRF, el ensemble, el dropout, `power` y el gate `prob >= 0.8` no
hacen falta.

---

## 0. Correcciones a la versión anterior

### 0.1 La verdad estaba a nivel de estrella, no de pico

`has_period` marcaba como "ELL real" a **cualquier** pico de una estrella
periódica. Con eso, los dos picos ELL de TIC 89761288 s55 contaban como reales
— pero esa estrella es una **E de 6.4444 d** y sus picos están en 3.232 y
3.236 d, o sea **P/2**. Doblada a la mitad, una eclipsante apila primario y
secundario y parece una elipsoidal: es justo el alias que la normalización
`log` se eligió para evitar (ver el comentario de `config.HIST_NORM`).

Contando bien, contra los períodos que el humano eligió (tolerancia 5%):

```
                        antes    ahora
ELL reales                  5        3   (TIC 12921082 x2, 362792232)
ELL contaminantes          41       43
```

Sobre los 16 picos con período humano confirmado el pipeline no se equivoca de
clase: 11/11 E correctas, 3/3 ELL correctas, 2 Pulsating perdidas a Rndm. Todo
el problema está en los picos espurios, no en los confirmados.

La función que hace el match está en `scripts/analysis_ell/_common.py`
(`load_peak_truth`).

### 0.2 El test de σ NO había fallado

La tabla decía «σ y `prob/σ` del MC-dropout: contaminantes indistinguibles de
reales (σ≈0.003 en ambos)». El propio script, corrido sin cambios, daba:

```
prob/sigma        AUC 0.820
(prob-0.5)/sigma  AUC 0.826
q1                AUC 0.828
p16               AUC 0.846
power             AUC 0.733
```

La fila estaba mal leída. σ **sí** separa, y mejor que `power`.

### 0.3 `rank` no es especial; sólo se lo midió donde hay 3 datos

`p_LPV(rank)` se validó únicamente en la comparación 3-vs-43. Medido también
sobre 16 picos confirmados contra 689 espurios — donde sí hay estadística —
`rank` es **el peor** de la familia:

```
p_LPV, por normalizacion       AUC 3v43   AUC 16v689   p (16v689)
  min_max                         0.919        0.972      5e-11
  quantized                       0.826        0.949      4e-10
  power_0.5                       0.884        0.929      2e-09
  log                             0.798        0.900      2e-08
  rank                            0.977        0.818      6e-06   <-
```

`min_max` es la normalización con la que se entrenaron los pesos, así que su
`p_LPV` es el que está calibrado. El 0.977 de `rank` es suerte con N=3.

Esa tabla está medida **después del BRF**, que es el otro problema (§0.4).

### 0.4 Todo se midió después del BRF

Los cinco tests descartados y los cuatro nuevos leyeron las probabilidades a
la salida del BRF. El BRF tiene 500 árboles: no puede representar nada por
debajo de 2e-3, y promediado sobre 7 pasadas su piso es 2.9e-4. La CNN dice
`p_LPV` desde 1.5e-18. Todo el rango informativo cae debajo del piso y se
aplasta a cero.

Leída en la CNN, `p_LPV(log)` sube de AUC 0.740 a **0.958** a nivel de
estrella. Detalle y mecanismo en §2. Esto también explica de dónde salía
`p_LPV(rank) == 0`: era el piso de cuantización, no un umbral físico.

---

## 1. Los tests, ordenados

Los scripts viven en `scripts/analysis_ell/`, agrupados por tema, con un
`README.md` que lista cada uno y su conclusión. **Los de la sesión del
2026-09-03 se borraron**: medían con el argmax, usaban verdad a nivel de
estrella y leían después del BRF, así que sus números no son comparables. Lo
que aportaron queda acá.

### Lo que se probó en la sesión anterior y por qué falló

Los cinco comparten un defecto: **colapsan el eje de pasadas con el argmax o con
la media antes de mirar las otras clases**, y **leen después del BRF**.

| test | qué medía | veredicto |
|---|---|---|
| σ, `prob/σ` MC | dispersión de la clase ganadora | **funcionaba** (AUC 0.82–0.85); el doc lo reportaba mal (§0.2) |
| voto e inestabilidad MC | argmax por pasada | falla (AUC 0.54) — es el argmax |
| desacuerdo del ensemble | cuántos de 7 votan ELL | falla (falsos 6.0 contra reales 5.0) — es el argmax |
| forma de la distribución MC | bimodalidad | falla (AUC 0.24–0.46) |
| consistencia entre normalizaciones | argmax en 9 normas | falla (8 de 9 iguales) — es el argmax |

### Los tests vigentes

```
01_verdad/              verdad a nivel de pico; clase contra periodo
02_donde_esta_la_senal/ CNN contra BRF, cuantizacion, vector completo, colas
03_descartados/         normalizaciones, power, p_Rndm, contraste del pico
04_confianza/           saturacion del argmax y de la probabilidad
05_irregular/           el % irregular por estrella
06_final/               eleccion del checkpoint y el pipeline adoptado
```

### Qué encontró mirar el vector completo

Medianas en los mismos picos que el pipeline llama ELL:

```
nivel        grupo   N     ELL  Pulsating       E     LPV    Rndm
  CNN     ELL real   3  0.9998     0.0000  0.0002  0.0000  0.0000
  CNN contaminante  43  0.8078     0.0001  0.0088  0.0007  0.0045
```

Y el runner-up, que el argmax tira entero: en las ELL reales la segunda clase
es siempre **E o Pulsating**, nunca Rndm ni LPV; en los contaminantes es Rndm
en 14 de 43 y LPV en 4.

### Qué encontró mirar las colas

`max p_Rndm` entre las 7 pasadas del ensemble, en los picos ELL:

```
                     min        mediana       max
3 ELL reales     1.1e-09       4.0e-09    4.0e-08
43 ELL falsos    6.2e-10       8.8e-01    1.0e+00      <- 7 ordenes de magnitud
```

La media lo aplasta (0.0045) y el voto no lo ve (los contaminantes tienen
*más* acuerdo). Sólo sobrevive si se mira el máximo entre pasadas.

---

## 2. Antes o después del BRF: la corrección principal

Todos los tests anteriores leyeron las probabilidades **después del BRF**. Es
ahí donde se pierde la señal.

```
                         AUC 3v43   AUC 16v689   AUC estrella   p (estrella)
p_LPV CNN (log)             0.984        0.938          0.958          2e-05
p_LPV BRF (log)             0.798        0.900          0.740           0.02
max p_Rndm CNN (binary)     0.969        0.928          0.861          0.001
power                       0.682        0.808          0.743           0.03
```

### Por qué el BRF la borra

El BRF tiene **500 árboles**, así que la probabilidad más chica que puede
representar es 1/500 = 2e-3; promediada sobre las 7 pasadas, su piso es
2.9e-4. La CNN usa un rango mucho más ancho:

```
p_LPV sobre los 857 picos      valores unicos      ceros    min no cero
  CNN (log)                               856          0        1.5e-18
  BRF (log)                               506         76        2.9e-04
  CNN (binary)                            830          0        3.1e-19
  BRF (binary)                            373        117        2.9e-04
  BRF (rank)                              329        139        2.9e-04
```

**Esto explica `p_LPV(rank) == 0`**, que la versión anterior daba por
verificado pero sin mecanismo. No era un umbral físico: era el piso de
cuantización del bosque. `rank` sólo era la normalización que empujaba a más
picos por debajo de ese piso (139 ceros contra 76 de `log`). La cantidad real
es la de la CNN, que tiene 18 órdenes de magnitud de rango.

### Qué nivel usar para cada cosa

| cantidad | nivel | por qué |
|---|---|---|
| la **clase** y su probabilidad | BRF | es lo que el pipeline reporta |
| `p_LPV`, `p_Rndm`, colas | **CNN** | el BRF no tiene resolución debajo de 2.9e-4 |
| σ entre pasadas | BRF o CNN | da parecido en los dos (0.29 vs 0.29) |

`p_LPV` bajo `log` no cuesta nada: `log` ya es la normalización de producción,
así que el número sale de la corrida que el pipeline hace igual.

## 3. Lo que importa: que la confianza sea honesta

El objetivo no es borrar contaminantes sino que el número que el catálogo
imprime signifique algo. Si el modelo dice ELL con dispersión alta, la
estrella queda marcada y se revisa; el caso que hace daño es **seguro y
equivocado**.

Unidad de medida: **el pico que el catálogo reporta por estrella** (el de mayor
probabilidad entre las clases periódicas con `prob >= 0.8`). La pureza a nivel
de pico es 7% por construcción — una estrella periódica igual tiene ~20 picos
espurios — así que no responde la pregunta. A nivel de estrella hay 26
reportadas de las 39 revisadas: 8 correctas, 18 erradas.

### La probabilidad del catálogo no es honesta

```
                      CORRECTA mediana   ERRADA mediana
prob (BRF)                      0.995            0.991    <- no separa nada
sigma entre pasadas             0.010            0.097    x10
p_LPV CNN (log)               1.6e-09          1.7e-05    x10000
max p_Rndm CNN (binary)       1.4e-07          1.6e-03    x12000
power                           0.619            0.319
```

Cortar por `prob >= 0.99` sube la pureza de 31% a 36%. Es decir: casi nada.

### La regla

Sobre la corrida de producción (`log`), en la salida de la CNN, sin
periodograma y sin BRF:

```
max sobre las 7 pasadas de p_LPV(CNN, log)  <  1.8e-8
```

```
regla                                    correctas  erradas  pureza  margen  costo
p_LPV(log) max pasadas < 1.8e-8                6/8     0/18    100%   x10.9  ninguno
p_LPV(log) media pasadas < 3.5e-9              6/8     0/18    100%    x7.4  ninguno
max p_LPV sobre 9 normas x pasadas < 3e-4      7/8     0/18    100%    x1.9  9 corridas CNN
power>=0.5 Y p_LPV(log)<1e-5                   6/8     1/18     86%       -  periodograma
p_LPV(log) < 1e-5                              8/8     7/18     53%       -  ninguno
prob>=0.99 (lo que hay hoy)                    5/8     9/18     36%       -  -
sin filtro                                     8/8    18/18     31%       -  -
```

El corte cae en un hueco limpio:

```
ultima CORRECTA conservada   339568213_12   1.7e-08
primer ERROR                 384805438_58   1.9e-07     margen x10.9
```

### Qué son las "7 pasadas": checkpoints, no dropout

Las 7 pasadas de `norm_compare/cnn_mc_log.npz` son los **7 checkpoints** de
`config.MODELS`, una pasada determinista cada uno. **No** son MC-dropout, que
sólo existe para `Number_DST` (`cnn_mc20.npz`, `cnn_mc200.npz`).

Y el dropout no conviene:

```
agregado              AUC estrella   correctas sin error   margen
ens7 max                     0.958                   6/8     10.9
ens7 media                   0.958                   6/8      7.4
dropout200 media             0.951                   6/8      1.4
dropout200 max               0.951                   6/8      1.3
dropout20 media              0.910                   3/8      6.7
```

Mismo recall, margen 8 veces peor. La razón se ve abajo: el dropout corre sobre
`Number_DST`, que es una de las peores redes para LPV, y ninguna cantidad de
pasadas lo arregla.

### Qué red lee mejor LPV

Una pasada determinista por checkpoint, normalización `log`:

```
checkpoint                 AUC estrella      p    correctas sin error   margen
Number_ELL                        0.979   4e-06                   5/8     29.9
Number_M                          0.944   4e-05                   3/8    445.7
batchBalanced_Number_ELL          0.944   4e-05                   4/8     20.0
Number_DST     <- el default      0.910  0.0002                   2/8       -
batchBalanced_Number_M            0.903  0.0003                   5/8     19.3
batchBalanced_Number_DST          0.889  0.0005                   3/8      6.4
Number_CEP                        0.854   0.002                   1/8       -
```

**`Number_ELL` es claramente la mejor** y **`Number_DST`, que es
`config.DEFAULT_MODEL`, está entre las peores.** Tiene sentido: el nombre del
checkpoint es la clase sobre la que se balanceó el entrenamiento.

Pero el máximo sobre los 7 sigue siendo el mejor punto de operación:

```
                     correctas sin error   margen   AUC estrella
max sobre los 7                      6/8     10.9          0.958
solo Number_ELL                      5/8     29.9          0.979
media de ELL+M                       6/8      4.2          0.979
```

`Number_ELL` sola tiene mejor AUC y mejor margen pero pierde
TIC 405216722 (6.6e-12, apenas encima de su corte). El máximo sobre los 7 la
conserva y todavía deja un factor 11 de aire. Es otra vez la misma lección: el
AUC mide orden, no separabilidad a umbral fijo.

Se barrieron los 254 subconjuntos de checkpoints (7 modelos × media/max). Hay
combinaciones que llegan a **7/8 sin errores**, pero todas con margen ×1.3, que
sobre 26 estrellas es ruido. No adoptarlas.

### p_Rndm no aporta

La frontera de `p_LPV` sola y la del plano `(p_LPV, p_Rndm)` son **idénticas**
en todos los niveles de recall:

```
correctas conservadas   8/8   7/8   6/8   5/8   4/8
erradas, solo p_LPV       4     2     0     0     0
erradas, solo p_Rndm     16     4     4     3     2
erradas, las dos          4     2     0     0     0
```

`p_Rndm` por sí sola es mucho peor y no agrega ninguna separación que `p_LPV`
no dé ya. El eje que hay que mirar es `p_LPV`.

### Corrección: `log` sí tiene umbral que separa

Una versión intermedia de este análisis concluyó que `log` "ordena bien pero
no tiene umbral que separe, pureza máxima 70%". **Es falso**, y el error fue de
método: el barrido empezaba en 1e-6, y el punto de operación está en 1e-8.
Con una grilla fina sobre los valores observados, `log` sola llega a 6/8 con
cero errores y margen ×10.9.

Las nueve normalizaciones conservan una detección más (7/8) pero con margen
×1.9 y a costa de 9 corridas de CNN. El barrido entre normalizaciones sirvió
para saber que el contraste no compra separación adicional; no hace falta
adoptarlo.

### Un valor global por estrella: el % irregular

Todo lo anterior mira **un pico aislado**. La información de que la estrella es
irregular está en los otros picos, y es la que falta astrofísicamente.

**Definición.** Cada pico tiene un vector de probabilidad de la CNN que suma 1.
El promedio de esos vectores sobre todos los picos de la estrella-sector suma 1
también, así que ya es una distribución sobre clases a nivel de estrella: no
hay que reescalar nada para que quede en [0,1]. El **% irregular** es la masa
que cae en LPV.

```
                          AUC     p       mediana CORRECTA   mediana ERRADA
p_LPV medio (masa cruda)  0.868  0.001                2.9%            24.4%
p_LPV / (1 - p_Rndm)      0.819  0.005               18.8%            46.5%
p_LPV + p_Rndm            0.160    1                 85.6%            65.3%   <- se invierte
```

**Rndm no cuenta como irregular.** Toda estrella tiene ~19 picos basura que la
red llama Rndm correctamente, así que `p_Rndm` es una línea de base común y
sumarla invierte el score. Sólo LPV mide variabilidad irregular.

**Validación contra las notas humanas** (39 estrellas):

```
regimen                 N   mediana   min     max
1 periodo confirmado   16      8.0%   0.0%   53.3%
2 multiperiodico        3     38.2%  36.9%   38.9%
3 ninguno periodico     3     16.9%   9.8%   21.0%
4 no se ve el periodo  23     29.3%   1.8%   67.2%
```

Las tres que el humano llamó "multiperiódico" caen en 36.9 / 38.2 / 38.9%: un
grupo de 2 puntos de ancho, sin que nada lo forzara. No es un proxy del número
de picos (r=−0.37) ni del ruido (r=−0.40).

**Cómo usarlo: como número reportado, no como segundo corte.** Combinado con el
corte del pico llega a 7/8 sin errores en vez de 6/8, pero con márgenes ×1.7 y
×1.4 — sobre 26 estrellas eso es ruido. Como número reportado no cuesta nada y
se lee directo:

```
 12921082_82   1.773 d   ELL con  0% irregular   CORRECTA
339568213_12   5.792 d     E con  2% irregular   CORRECTA
400166377_72   2.220 d     E con  6% irregular   CORRECTA
384805438_58   2.935 d   ELL con 20% irregular   ERRADA    "no se ve el periodo claro"
 13785212_41   3.646 d   ELL con 38% irregular   ERRADA    "multiperiodico"
  8301691_59   1.394 d   ELL con 39% irregular   ERRADA    "multi periodico"
331664814_11   0.514 d  Puls con 58% irregular   ERRADA    "no se ve un periodo claro"
```

**Los dos números responden preguntas distintas y no hay que fundirlos.**
`p_LPV` del pico reportado dice *si ese período es real*; el % irregular dice
*cuánta de la variabilidad de la estrella es irregular*. Una estrella puede
tener las dos cosas: TIC 89761288 es una E confirmada a 6.444 d **con 32%
irregular**, y eso no es un error sino una descripción correcta.

### El % irregular se lee distinto según la clase

Condicionado a la clase que el pipeline reporta, el mismo número se comporta de
forma opuesta:

```
clase        N (ok/err)     AUC   mediana OK   mediana ERR   corte limpio
ELL            2 / 13     1.000         0.0%         24.2%   SI  (0% | 9.8%)
E              6 /  4     0.667         4.8%         17.1%   no
Pulsating      0 /  1         -            -         57.9%   -
```

**En ELL es un criterio de validación.** Las 2 ELL confirmadas tienen
exactamente 0.0% irregular; las 13 espurias van de 9.8% a 62%. Sin solapamiento.
Tiene sentido físico: una elipsoidal es **una** modulación continua y suave por
distorsión de marea. No convive con potencia irregular — si la hay, el "ELL" es
una lectura errónea de otra cosa.

**En E es una descripción, no un criterio.** Un eclipse es un evento geométrico
agudo que sobrevive a estar superpuesto sobre otra cosa: manchas, pulsaciones,
viento. TIC 89761288 es una E confirmada a 6.444 d **con 32% irregular** y es
correcta. Que una eclipsante tenga variabilidad irregular no invalida el
eclipse; es una propiedad del objeto.

```
regla de lectura
  ELL con  >10% irregular   ->  sospechosa, revisar
  ELL con   ~0% irregular   ->  limpia
  E   con   X% irregular    ->  no dice nada sobre la clase; es descripcion
                                del objeto ("eclipsante con 32% irregular")
```

Advertencia: son **2** ELL confirmadas. El AUC 1.000 es sobre 2 contra 13.

### Los dos modos de falla son independientes

```
                     periodo correcto
clase correcta      no      si
      no            16       0
      si             2       8
```

Sólo 2 de 26 son "clase bien, período mal", y las dos son alias 2P exactos de
eclipsantes (TIC 24985783 y 269818126). El % irregular no las ve, y no tiene por
qué: describe la variabilidad, no la elección del período.

En la otra dirección pasa lo mismo y el comportamiento de la red es el correcto:
la contaminación ELL de TIC 89761288 en P/2 es **la red negándose a llamar
eclipse a un fold de medio período**. A P/2 primario y secundario se apilan y la
curva es genuinamente elipsoidal. La clase que devuelve describe bien lo que ve;
lo que está mal es el período con que se dobló.

### El argmax se queda; lo que se cae es el número que lo acompaña

**El argmax no es el problema.** Sobre los 16 picos con período humano
confirmado acierta la clase 14 veces (11/11 E, 3/3 ELL; pierde 2 Pulsating a
Rndm). A nivel de estrella, 10/11.

Los tests viejos no fallaron por *elegir* con el argmax sino por **medir** con
él: contar en cuántas normalizaciones o checkpoints coincide el ganador tira el
vector entero. Elegir la clase con el argmax y describirla con el vector son
cosas distintas.

**El espectro de estrella no sirve para elegir la clase**, sólo para
describirla:

```
                acierto de la clase
argmax del pico          10/11
argmax del espectro       7/11
```

Falla justo donde importa: para TIC 12921082 y 362792232, las dos ELL
confirmadas, el espectro tiene más masa en Pulsating (16%, 10%) que en ELL (8%,
10%), porque los ~19 picos basura de la estrella la aportan. El espectro
describe la estrella; el pico elige la clase.

**Lo que sí hay que sacar es `prob >= 0.8`.** Es un corte sobre la cantidad que
ya no discrimina (0.995 contra 0.991, §3) y cuesta 3 de los 16 picos
confirmados:

```
gate                                    picos  confirm.  TP FN FP TN    acc
sin gate                                  238     16/16  11  0 26  2  0.333
prob >= 0.8  (el actual)                   52     13/16  11  0 15 13  0.615
prob >= 0.99                               16      6/16   6  5  8 20  0.667
prob>=0.8 Y ELL: irregular<5%              31     13/16  11  0  6 22  0.846
p_LPV(pico) < 1.8e-8                        8      8/16   8  3  0 28  0.923
```

Dos puntos de operación, según lo que se quiera:

- **recall 1.00**: `prob>=0.8` + `ELL con irregular<5%` → 11/11 estrellas
  periódicas, FP de 15 a 6, acc 0.615 → 0.846
- **pureza 1.00**: `p_LPV(pico) < 1.8e-8` → 0 FP, pero pierde 3 estrellas

Y si se quiere conservar un número de confianza por pico en vez de la
probabilidad lineal, el margen en log-odds tiene rango donde la probabilidad
no lo tiene:

```
score                          AUC 16v689     p5    p50    p95
BRF prob lineal (top1)              0.781  0.407  0.802  0.994   <- saturado
BRF margen log10 top1/top3          0.855  0.399   1.37   3.24
CNN -log10 p_LPV                    0.938  0.099   1.25   6.06
```

### La fila del catálogo, propuesta

```
TIC        sector  periodo  clase  irregular%  -log10 p_LPV  nivel
12921082       82    1.773    ELL         0%           8.3   alta
362792232      41    4.451    ELL         0%           9.0   alta
89761288       55    6.444      E        32%           7.2   alta
384805438      58    2.935    ELL        20%           7.2   baja
 8301691       59    1.394    ELL        39%           2.2   baja
```

- **clase**: el argmax, sin cambios
- **irregular%**: masa en LPV del espectro de la estrella-sector (§3)
- **-log10 p_LPV**: la confianza del pico, sin saturar
- **nivel**: `alta` si la clase no es ELL o el irregular es < 5%
- la probabilidad del BRF **no se reporta**: no significa nada

### El pipeline final: una sola red

Los 7 checkpoints midiendo las tres funciones a la vez, cada uno solo, CNN sin
BRF, normalización `log`:

```
checkpoint                clase_ok_16  p_LPV AUC estrella  irreg AUC ELL  N_ELL
Number_ELL                      15/16              0.979          1.000     44
Number_M                        15/16              0.944          0.946     69
batchBalanced_Number_ELL        15/16              0.944          1.000     33
batchBalanced_Number_M          15/16              0.903          1.000     63
Number_DST      <- el default   14/16              0.910          1.000     58
batchBalanced_Number_DST        14/16              0.889          1.000     57
Number_CEP                      14/16              0.854          1.000     54

referencia: ensemble de 7 + BRF ->  14/16
```

**`Number_ELL` sola.** Empata en clase con otras tres y les gana claro en
`p_LPV`. Y le gana al ensemble de 7 más el BRF en acierto de clase: 15/16
contra 14/16. El BRF no hace falta.

### Con la probabilidad saturada, el pico se elige por p_LPV

Elegir el pico reportado por "mayor probabilidad de clase" es casi arbitrario
cuando esa probabilidad tiene mediana 0.996. Elegirlo por **menor `p_LPV`**:

```
regla de seleccion del pico          estrellas periodicas con el pico correcto
mayor probabilidad de clase                                             9/11
menor p_LPV                                                            10/11
pipeline viejo (ens7 + BRF + prob>=0.8)                                 8/11
```

Y arregla dos de los tres **alias 2P**: TIC 269818126 pasa de 6.720 a 3.3611 y
TIC 24985783 de 7.530 a 3.7662, los dos correctos. El período doblado tiene más
`p_LPV` que el verdadero, así que la misma cantidad que mide irregularidad
también desempata armónicos.

### Números finales

```
-log10 p_LPV(Number_ELL, log)      AUC 0.988   (10 correctas vs 25 erradas)
corte limpio  > 12.0               7/10 correctas, 0 erradas, margen 1.5 decadas
```

### La fila del catálogo

```
TIC        sector  periodo   clase  p_clase  irregular%  -log10 pLPV
316187504      56   1.2732       E    1.000          8%         23.4
362792232      41   4.4513     ELL    1.000          0%         14.5
 12921082      82   1.7731     ELL    1.000          0%         13.4
 89761288      55   6.4444       E    0.999         48%         10.7
384805438      58   2.9352     ELL    0.997         38%          7.2
179639066      28   4.7292    Puls    0.587         74%          0.4
```

- **clase**: argmax de `Number_ELL`
- **p_clase**: su softmax. **Se reporta pero no se filtra con ella**: mediana
  0.996. Sirve sólo como bandera de una cara — las 5 estrellas con
  `p_clase < 0.7` son las 5 erradas, pero por encima de eso no distingue nada.
- **irregular%**: masa en LPV promediada sobre todos los picos de la
  estrella-sector. Se lee según la clase (ELL >10% es sospechoso, en E es
  descripción).
- **-log10 pLPV**: la confianza del pico, la única cantidad con rango.

Lo que desaparece: el BRF, el ensemble de checkpoints, el MC-dropout, `power`,
las otras normalizaciones y el gate `prob >= 0.8`.

### El muestreo: ELL es un imán para períodos submuestreados

Un sector de TESS dura ~26.5 d, así que un período de 10 d se dobla con 2.6
ciclos y uno de 1.4 d con 19. La CNN se entrenó con OGLE, cuyos baselines son
de años: **nunca vio un fold de 3 ciclos**. Con tan pocos ciclos, una tendencia
lenta o ruido rojo se dobla en una modulación suave de uno o dos máximos, que
es exactamente la firma de una elipsoidal.

El efecto está medido y es grande:

```
banda de P   frac ELL        ciclos       frac ELL
(0, 2]          0.019        (0, 3]          0.079
(2, 4]          0.066        (3, 4]          0.192   <- pico
(4, 6]          0.093        (4, 6]          0.075
(6, 8]          0.104        (6, 10]         0.087
(8, 10]         0.143        (10, 20]        0.050
(10, 30]        0.098        (20, 250]       0.006
```

A 3–4 ciclos, **1 de cada 5 picos sale ELL**; por encima de 20, 1 de cada 170.
Factor 32. En ciclos el efecto es mucho más nítido que en período, que es lo
esperable si la causa es el muestreo y no la física.

### El piso de ciclos, y por qué no es un score

Como **score** el número de ciclos va al revés (AUC 0.358): el periodograma
produce basura sobre todo en períodos cortos, que son los que más ciclos
cubren, así que "más ciclos" correlaciona con "pico corto y espurio".

Como **piso** funciona y no cuesta nada:

```
                            confirmados   espurios
< 3   ciclos (P > 8.8 d)          0/16      69/689
< 3.5 ciclos (P > 7.6 d)          0/16      91/689
< 4   ciclos (P > 6.6 d)          0/16     114/689
< 4.5 ciclos (P > 5.9 d)          3/16     134/689   <- ya cuesta
```

**Adoptado: `baseline/P >= 4`.** Saca 4 de las 40 estrellas reportadas, todas
con P > 7 d y menos de 3.5 ciclos, y ninguna de ellas era correcta. La
selección del pico y la frontera de confianza no cambian (10/11 y 7/10 con
margen 1.5).

No es un umbral ajustado: con 2 ciclos no se establece un período, diga lo que
diga el clasificador, y el fold está fuera de la distribución de entrenamiento.

**Caveat**: los picos confirmados llegan hasta 6.44 d = 4.1 ciclos y ni uno
más. Puede ser física o puede ser el mismo efecto — con un sector no se puede
confirmar visualmente un período largo, así que nunca se etiquetó uno. El piso
es parcialmente circular con cómo se construyó la verdad. El test que lo rompe
es multi-sector: alargar el baseline y ver si el mismo período largo sube su
`-log10 p_LPV` al ganar ciclos.

### Probado y descartado: elegir por p_clase con desempate por p_LPV

Es más fácil de defender ("la clase manda, `p_LPV` desempata") pero pierde.
Sobre `Number_ELL` sola las dos reglas empatan (10/11), así que la comparación
se hizo promediando sobre los 7 checkpoints: una diferencia de una estrella
sobre 11 no se resuelve con una muestra.

```
regla                              pico correcto      por checkpoint
max p_clase                        62/77  (80.5%)     7 8 10 9 8 10 10
p_clase, empate 0.01 -> p_LPV      67/77  (87.0%)     9 10 10 9 9 10 10
p_clase, empate 0.05 -> p_LPV      68/77  (88.3%)     9 10 10 9 10 10 10
p_clase, empate 0.20 -> p_LPV      68/77  (88.3%)     9 10 10 9 10 10 10
max -log10 p_LPV (adoptada)        70/77  (90.9%)     10 10 10 9 10 11 10
```

El desempate mejora sobre `p_clase` sola, pero el gradiente es monótono: cuanto
más peso lleva `p_LPV`, mejor. Y es más reproducible — elige el mismo pico en
el 79.8% de los pares (estrella, checkpoint) contra 55.7%.

El mecanismo: **no hay un solo empate exacto** en `p_clase` (0 de 32 estrellas),
así que el desempate nunca dispara sin tolerancia explícita. Y lo que decide
`p_clase` son diferencias de 0.002 sobre valores de 0.99, o sea ruido de float
cerca de 1; `p_LPV` recorre 18 órdenes de magnitud.

Nota de método: evaluar esto sobre un checkpoint no tenía resolución. Aunque el
pipeline final use una sola red, **las comparaciones entre reglas conviene
hacerlas sobre las siete.**

### Lo que NO funcionó de la normalización por ciclos

Multiplicar `-log10 p_LPV` por un factor de ciclos para elegir el pico llega a
**11/11** con `ciclos/(ciclos+6)`, y la estrella que arregla es la que faltaba:
TIC 12675729 pasa de 2.882 d (9.0 ciclos, el alias 2P) a 1.440 d (18.0 ciclos,
la elección humana). Pero el 11/11 sale sólo para C = 6 y 7 de 14 valores
probados: una ventana de dos, para ganar una estrella. No se adopta.

Encoger `p_clase` hacia el prior por ciclos **la empeora**: des-satura (la
fracción por encima de 0.99 pasa de 38% a 0%) pero el AUC baja de 0.900 a 0.72.
Mismo motivo que arriba: el factor es monótono en 1/P e infla los picos cortos,
que son mayoritariamente basura.

### El que se escapaba

Con la regla basada en `power`, la única que quedaba segura y equivocada era:

```
384805438_58   ELL 2.935 d   power=0.549   p_LPV_CNN(log)=9.4e-09
```

Las 9 normalizaciones menos `rank` la llaman ELL con `p_ELL >= 0.998`. Su
periodograma es una **escalera armónica**: 2.935 d con power 0.549 y 1.459 d
con 0.530, más 4.345, 5.808 y 7.320 entre 0.44 y 0.48. Ningún pico domina.

El máximo entre pasadas bajo `log` **sí la atrapa**: 1.9e-07 contra 1.7e-08 de
la última correcta conservada. Es justo el primer error de la lista, o sea que
la regla la deja afuera por un factor 11.

Se probó además el contraste `power_1/power_2` como eje extra: **no generaliza**
(AUC 0.465, azar). Atrapa a esta estrella y a nada más. No adoptarlo.

### Hallazgo aparte: el alias 2P

Tres de las 18 erradas no son contaminación sino **período duplicado exacto**
en estrellas genuinamente periódicas:

```
12675729_82    reportado 2.882 d   humano 1.440   razon 2.002
24985783_66    reportado 7.530 d   humano 3.766   razon 1.999
269818126_57   reportado 6.720 d   humano 3.361   razon 1.999
```

La clase (E) es correcta y el período es el doble. `log` se eligió porque ata
la clase al período, pero eso se validó contra **P/2** (`config.HIST_NORM`),
nunca contra 2P. Es un problema sistemático distinto del de las ELL y
probablemente más fácil: doblada a 2P una eclipsante muestra dos eclipses por
ciclo, que es una firma buscable.

## 4. Lo que sigue

1. **Resolver el alias 2P** (§3): tres de las 18 erradas son el período
   duplicado en estrellas correctas. Es el error más frecuente después de la
   contaminación ELL y tiene firma buscable.
2. **Resolver las 6 estrellas ambiguas** (TIC 16187387, 41903679, 71935430,
   116065031, 137113608, 153304135). Siguen sin usar y son 7 picos ELL sin
   etiqueta. Sigue siendo lo más barato por estrella.
   Herramienta: `scripts/annotate_folds.py` (no usar `conda run` sin
   `--no-capture-output`: cierra stdin).
3. **Ampliar la muestra.** El nivel de estrella son 8 correctas y 18 erradas;
   el de picos ELL, 3 contra 43. Los cortes tienen aire (`p_LPV_CNN` separa
   1.6e-09 de 1.7e-05) pero 8 es 8.
4. **Test de armónicos de Gomel+2023** sobre el fold: `A1/A2 < 1`,
   `A3/A2 < 0.3`, `A2/A2_err > 10`, `A1/A1_err > 3`. Sigue siendo la única vía
   pendiente que mide la forma de la curva directamente, sin pasar por el
   clasificador. Ver `docs/UMBRALES_PERIODICIDAD_LITERATURA.md` §4.
5. **Sacar `PROB_MIN` y el gate `prob >= 0.8`** del catálogo (§3): cortan
   sobre un número saturado (0.995 contra 0.991) y cuestan 3 de los 16 picos
   confirmados. Reemplazar por el punto de operación elegido.
6. **Reescribir los docstrings mal atribuidos**: `apply_vote_stability_gate`
   ("el BRF aterriza en una constante sesgada hacia ELL") y
   `scripts/step_cnn_only.py`. El BRF es aproximadamente la identidad sobre
   ELL (0.59–0.65 barriendo amplitud ×200 y `per` ×30), no la fuente.
7. **Revisar `DEFAULT_MODEL`.** `Number_DST` es mala por dos vías
   independientes: 45 ELL falsos contra 24 de `batchBalanced_Number_ELL`
   (medido en la sesión anterior), y es de las peores leyendo `p_LPV`
   (AUC 0.910 contra
   0.979 de `Number_ELL`, §3). Como el MC-dropout corre sobre ella, arrastra
   las dos.

## 5. Abierto: la probabilidad de clase

`p_clase` se reporta en la tabla pero **no se filtra con ella** y no se fija un
umbral: con 26 estrellas revisadas su mediana es 0.996 y lo único medible es
que las 5 con `p_clase < 0.7` son las 5 erradas. Fijar su comportamiento
requiere una muestra mayor. Hasta entonces es una columna informativa.

## 6. Hechos que siguen valiendo de la versión anterior

- **El BRF no es la fuente de ELL**: de 38 ELL finales en estrellas sin
  período, 36 ya eran ELL en la CNN. Balance neto del BRF sobre ELL: −4. Lo
  que sí hace es matar LPV (103 → 32 picos con `p_LPV > 0.3`).
- El BRF **tampoco** destruye la dispersión: con las mismas 20 pasadas, σ de
  `p_ELL` pasa de 0.29 (CNN) a 0.29 (BRF) en los contaminantes. Lo que el
  el test de σ original perdía era otra cosa, no la compresión del BRF.
- **MC-dropout es un parche**: mc20 vs mc200 coinciden en el 98.1% de los
  picos. Para la cola de Rndm, 20 pasadas dan AUC 0.771 contra 0.790 de 200:
  no hace falta correr 200.
- **`HIST_NORM` mueve entre el 4% y el 67% de las clases** del catálogo. Sigue
  sin documentarse como incertidumbre sistemática.
- **Trampa conocida**: `msv.features.phase_fold_hist2d_*` aplica `.T[::-1]`
  DESPUÉS de normalizar. Saltearlo entrega el histograma transpuesto y todo
  colapsa a Rndm. Control obligatorio al construir cubos: reconstruir `log` y
  verificar que reproduce `results/cnn_input.npz['X']` bit a bit.

## 7. Archivos

**Pipeline** (reemplaza a `step_brf.py` y `step_cnn_only.py`):

```
python scripts/step_clasificar_una_red.py results/cnn_input.npz results/cnn_mc.npz results/clasificacion_una_red.csv --curves results/phasefold_curves.pkl
python scripts/build_review_una_red.py results/clasificacion_una_red.csv
python scripts/build_top3_folds.py results/clasificacion_una_red.csv --curves results/phasefold_curves.pkl --outname top_folds_una_red.pdf --stat mean
```

`step_clasificar_una_red.py` mantiene el contrato de columnas de `step_brf.py`
(`sigma`, `s_`, `med_`, `q1_`, `q3_`, `v_`) para que los PDF sigan corriendo,
pero **con una pasada determinista esas columnas son degeneradas**: sigma vale
0 y la mediana vale la media. No es que la dispersión sea chica, no existe. La
confianza es `log_pLPV`.

**PDFs** (`results/figures/`):
- `review_una_red.pdf` — una estrella por fila: curva, fold en el período
  reportado, el hist2d que entra a la CNN, y `-log10 p_LPV` de los 12 mejores
  candidatos con el umbral marcado. Título en rojo si el nivel es `baja`.
- `top_folds_una_red.pdf` — los 3 mejores candidatos doblados lado a lado para
  arbitrar el período a mano, ordenados por `p_LPV` entre las clases
  periódicas. Es donde se ve el alias 2P: en TIC 12675729 el panel de 2.8819 d
  muestra dos mínimos por ciclo y el de 1.4398 d uno solo.
- `legacy_mc_brf/` — los del enfoque anterior, conservados para comparar.

**Scripts** — `scripts/analysis_ell/`, ver su `README.md` para la tabla
completa de qué mide cada uno:

```
_common.py              carga de picos, verdad a nivel de pico, AUC
build_all_norms.py      los cubos en las 9 normalizaciones
01_verdad/              peak_truth, clase_vs_periodo
02_donde_esta_la_senal/ cnn_vs_brf, brf_quantization, brf_damping,
                        prob_geometry, pass_structure, rndm_tail, significance
03_descartados/         17 scripts: power, otras normalizaciones, p_Rndm,
                        contraste del pico, versiones intermedias de la regla
04_confianza/           star_confidence, saturacion_argmax, reemplazar_gate,
                        residual_failure
05_irregular/           global_irregular, reporte_irregular, valida_irregular,
                        irregular_por_clase
06_final/               lpv_por_modelo, lpv_subset_modelos, una_sola_red,
                        pipeline_una_red
```

**Entorno**:
```
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate CNN_TESS
export PYTHONPATH=src:scripts/analysis_ell
export MSV_BRF=$HOME/Dropbox/MassiveStarVariability/models/balanced_random_forest_model.joblib
```
La CNN necesita `tf_env` y los pesos locales:
```
MSV_WEIGHTS=$HOME/ViT_VariableStars/pretrained/keras_checkpoints PYTHONPATH=src /opt/anaconda3/envs/tf_env/bin/python scripts/step_cnn.py <in.npz> <out.npz>
```
