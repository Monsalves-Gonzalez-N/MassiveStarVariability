# Umbrales de detección de periodicidad en la literatura (ACF, LS, SLF, ELL)

Recopilado el 2026-09-03 por búsqueda bibliográfica dirigida. Motivación: los
umbrales del pipeline (`FAP_ALPHA`, `ACF_PROMINENCE_K_FAP`, `ACF_WIDTH_FRAC`)
se fijaron por calibración interna, y la banda de Bartlett del ACF más la FAP
analítica de Baluev del LS suponen ruido no correlacionado — supuesto que las
estrellas OB violan por su variabilidad estocástica de baja frecuencia (SLF).

Convención: **[V]** verificado leyendo el texto del paper; **[NV]** no
verificado o no encontrado. Lo marcado [NV] NO debe citarse.

---

## 1. ACF: el umbral de altura de pico

### McQuillan, Aigrain & Mazeh 2013, MNRAS 432, 1203

arXiv:1303.6787 · doi:10.1093/mnras/stt536

ACF estándar (Sec. 2, Ec. 1), normalizada por la suma total de cuadrados, de
modo que **r_0 = 1**:

    r_k = sum_{i=1}^{N-k} (x_i - xbar)(x_{i+k} - xbar) / sum_{i=1}^{N} (x_i - xbar)^2

**El umbral (Sec. 3.3, Ec. 4)** [V]:

    h_P > MAX(0.15, sigma_P / 51 dias)

`h_P` = altura local del pico; `sigma_P` = incertidumbre del período en días.
Recupera el 91% de las detecciones con 10% de tasa de falsa alarma.

Detalles operativos [V]:
- Curva normalizada por la mediana de cada cuarto, menos 1.
- Huecos rellenados con interpolación lineal + ruido gaussiano blanco. (NO se
  dejan huecos: relevante para el bug de `fill_gaps` del ACF de este repo.)
- Solo lags `k < N/2`.
- ACF suavizada con kernel gaussiano de ventana 56 lags, FWHM 18 lags.
- Incertidumbre (Ec. 3): `sigma_P = 1.483 * MAD / sqrt(N-1)`, MAD sobre las
  separaciones entre picos consecutivos.
- Sobreviven 1570 de 2483 M dwarfs (63.2%).

### McQuillan, Mazeh & Aigrain 2014, ApJS 211, 24

arXiv:1402.5694 · doi:10.1088/0067-0049/211/2/24

**Local Peak Height (LPH)**, Apéndice A.3 [V]: *"la altura del pico
seleccionado con respecto a la media de los valles a ambos lados"*, o sea

    LPH = r(pico) - 0.5 * [r(valle izquierdo) + r(valle derecho)]

Es una medida **local**, y por eso es insensible al envelope decreciente que
el ruido rojo impone sobre la ACF. Es la diferencia conceptual con un umbral
de altura absoluta o con una banda global tipo Bartlett.

Esquema AutoACF de dos etapas [V]:

1. **Coherencia multi-segmento** (Apéndice A.2): el período de la curva
   completa debe coincidir con el de **al menos 2 de 4 segmentos**, dentro del
   **10%**, aceptando también **P/2 y 2P** como coincidencia. Retiene el 99.3%
   de la clase "Periodic" pero deja demasiados falsos positivos por sí sola.
2. **Peso por LPH** (Apéndice A.3):
   - `LPH_line = 0.0027 * P + 0.126`  (P en días; en P->0 vale 0.126, cerca
     del 0.15 de 2013)
   - `w1 = 1 - exp[-(LPH - LPH_line)]`, negativos a 0
   - `w2` penaliza la distancia a la secuencia esperada en T_eff-período-LPH
   - `w = (w1 + C*w2)/C` con **C = 0.15**
   - **corte final `w > 0.25`**
   - rango permitido **0.2-70 d**

Período definido como la pendiente de un ajuste de recta a las posiciones de
los picos vs número de pico (hasta 4 picos, intercepto forzado en el origen).

Rendimiento (Tabla 6): selecciona el **86.2%** de la clase "Periodic" visual
con **0.1%** de falsos positivos. Sobreviven 34.030 de 133.030 estrellas
(25.6%).

### Ceillier et al. 2017 / Santos et al. 2019, ApJS 244, 21

arXiv:1908.05222 · doi:10.3847/1538-4365/ab3b56, Sec. 3.2 [V]

Umbrales, combinando wavelet (GWPS), ACF y composite spectrum (CS):

    G_ACF >= 0.2     altura ABSOLUTA del pico de ACF en P_rot
    H_ACF >= 0.3     altura del pico menos la media de los dos minimos locales
                     (la MISMA cantidad que el LPH de McQuillan)
    H_CS  >= 0.15

más: los tres períodos coinciden dentro de **2 sigma**, y dentro del **20%**
entre distintos filtros de la curva. Selecciona automáticamente ~60%; el resto
va a inspección visual.

> **Dos números independientes y concordantes para el corte de altura local de
> pico de ACF: 0.15 (McQuillan 2013) y 0.3 (Ceillier 2017 / Santos 2019).**
> Ambos sobre ACF normalizada con r_0 = 1.

---

## 2. No existe una escala común LS <-> ACF

### VanderPlas 2018, ApJS 236, 16

arXiv:1703.09824 · doi:10.3847/1538-4365/aab766

**Normalización (Sec. 7.5)** [V]: la normalización "standard" de astropy es
`P_norm(f) = 1 - chi2(f)/chi2_0`, adimensional y acotada en [0,1]; la PSD es
`P(f) = 0.5*(chi2_0 - chi2(f))`, en unidades de amplitud al cuadrado. Advierte
(nota al pie 8, Sec. 7.4.2) que **"para distintas normalizaciones del
periodograma, la forma de esta distribución cambia"**.

**FAP (Sec. 7.4.2)** [V]:
- La base analítica (Scargle 1982) supone **ruido gaussiano blanco**:
  `P_single(Z) = 1 - exp(-Z)`.
- El método de N_eff **subestima** la FAP del bootstrap; Baluev la
  **sobreestima**.
- **"el bootstrap tampoco es universalmente aplicable: por ejemplo, no da
  cuenta correctamente de los casos en que el ruido en las observaciones está
  correlacionado"**.
- La FAP responde "¿qué probabilidad tiene un proceso sin periodicidad de dar
  un pico así?" y **"enfáticamente no responde"** "¿qué probabilidad hay de
  que este dataset sea periódico?".
- Recomendación explícita: la única vía fiable de fijar un umbral es
  **inyección-recuperación** de señales simuladas con la ventana y el ruido
  reales.

**Consecuencia**: no hay conversión publicada entre `power_LS` y `power_ACF`, y
probablemente no pueda haberla. Mezclarlos en un mismo umbral no tiene
respaldo. La literatura usa **umbrales separados por diagnóstico + requisito
de acuerdo**.

### Requisitos de acuerdo publicados [V]

| Trabajo | Criterio | Referencia |
|---|---|---|
| McQuillan+2014 | completo vs segmento, **10%**, aceptando P/2 y 2P; >=2 de 4 segmentos | Ap. A.2 |
| Santos+2019 | GWPS = ACF = CS dentro de **2 sigma**; **20%** entre filtros | Sec. 3.2 |
| Santos+2021 | P_rot,ML dentro del **15%** del P_rot final | Sec. 3 |
| Fetherolf+2023 | reproducibilidad entre sectores TESS | Sec. 3.2 |

### La única cantidad adimensional común: Gomel et al. 2023

**"Frequencygram peak" > 12** — *la altura del pico relativa a la media del
frecuencigrama, en unidades de la desviación estándar del frecuencigrama*. Es
adimensional e independiente de la normalización, así que **se puede computar
igual para LS y para ACF**. Es lo más cercano a una escala común que existe.

### Fetherolf et al. 2023, AJ 165, 71 — el más parecido a este pipeline

arXiv:2208.11721 · doi:10.3847/1538-3881/acacf3. TESS Prime Mission, 2-min,
LS + ACF juntas, Sec. 2.4 [V]:

- Búsqueda en 0.01-1.5 d y 1-13 d.
- Doble sinusoide si el 2do pico de LS tiene **potencia normalizada > 0.1**;
  se acepta si mejora chi2_nu **>= 25%**.
- Se prueba ACF si los ajustes sinusoidales son malos (**chi2_nu > 100**), y
  **se acepta si la correlación más fuerte es > 0.5**.
- **Corte mínimo del catálogo: potencia LS normalizada > 0.01** (por debajo la
  variabilidad suele ser sistemática de la nave).
- Vetos: regiones densas del plano potencia-período (bins 50x50 en log con >=7
  estrellas); pico en el borde superior de 13 d; ajusta mejor una recta.
- Resultado: 68.497 estrellas variables.

---

## 3. Estrellas OB y variabilidad estocástica de baja frecuencia (SLF)

### El modelo — Bowman et al. 2020, A&A 640, A36

arXiv:2006.03012 · doi:10.1051/0004-6361/202038224, Sec. 2.3, Ec. 2 [V]:

    alpha(nu) = alpha_0 / [1 + (nu/nu_char)^gamma] + C_w

Ajustado por MCMC al espectro de amplitud **residual**, en **0.1 <= nu <= 360
d^-1**. Por debajo de 0.1 d^-1 dominan los sistemáticos instrumentales; para
TESS eso es ~12 d, medio sector.

Discriminantes físicos: IGWs de convección nuclear predicen
**0.8 <= gamma <= 3**; convección subsuperficial da **gamma >= 3.25**
(Couston et al. 2018).

### Cómo se separa un pico coherente del ruido rojo [V]

**El orden importa: pre-whitening iterativo de las frecuencias coherentes
PRIMERO, ajuste del fondo rojo al residuo DESPUÉS.**

Criterio de significancia (Bowman+2020, Sec. 2.2): *"el criterio estándar de
significancia de amplitud en el pre-whitening iterativo, que define como
significativas a las frecuencias con **una razón señal-ruido en amplitud (S/N)
mayor que cuatro** (Breger et al. 1993)"*. Es **S/N en AMPLITUD, no en
potencia**.

Cómo se mide el ruido local con fondo rojo — Bowman et al. 2019, NatAst 3,
760 (arXiv:1905.02120), Methods [V]:

> *"El valor de S/N en amplitud se calculó como el cociente entre la amplitud
> de la frecuencia extraída y **el promedio de los residuos en el espectro de
> amplitud dentro de una ventana de frecuencia simétrica de 1 d^-1 alrededor
> de la frecuencia extraída**. Este enfoque conservador (a diferencia de
> calcular el ruido usando una ventana a alta frecuencia) aseguró que no
> estuviéramos sobreajustando o sobreextrayendo la señal."*

O sea: ventana **estrecha**, **centrada en el pico**, sobre el espectro
**residual**. Una ventana a alta frecuencia subestima groseramente el ruido a
baja frecuencia cuando el fondo es rojo.

### Burssens et al. 2020, A&A 639, A81 — el más detallado

arXiv:2005.09658 · doi:10.1051/0004-6361/202037700. 98 OB, TESS S1-13.

**Adoptan S/N >= 5, no 4** (Sec. 2.3, Ap. A.1) [V]: *"Baran et al. (2015)
revisitaron este criterio. Concluyeron que se recomienda un umbral más alto
(S/N >= 5) para datasets que duran menos de unos pocos meses, ya que el número
típicamente mucho mayor de puntos aumenta la probabilidad de detección de
frecuencias espurias."*

**Tamaño de la ventana de ruido** (Ap. A.1) [V]: **1 d^-1 en estrellas donde
la SLF es prominente** (enanas masivas M > 20 Msol, gigantes OB); **5 d^-1**
en las de menor masa. Con ventana chica la densidad de modos g infla el ruido;
con ventana grande el fondo rojo queda mal representado.

Otros cortes [V]:
- Resolución en frecuencia `nu_res = 1/DeltaT`; frecuencias no resueltas entre
  sí eliminadas con **Loumos & Deeming (1978): 1.5/DeltaT**.
- **"frecuencias nu <~ 0.1 d^-1 son dudosas, dado que están muestreadas solo
  ~2 (4) veces para un dataset TESS de 1 (2) sectores"**.
- Errores corregidos por ruido correlacionado con Schwarzenberg-Czerny (2003).

Separación rot vs SLF (Sec. 2.3): rotacional = *"una única frecuencia y sus
(sub-)armónicos"*. Admiten la degeneración: *"existe cierta degeneración entre
la clasificación SLF y la SPB"*.

### Balona et al. 2019, MNRAS 485, 3457

arXiv:1902.09470 · doi:10.1093/mnras/stz586 [V]

- Umbral general: **falsa alarma <= 1e-3**.
- Umbral efectivo de la muestra ROT: FAP **< 1e-6** y **amplitud del pico /
  ruido de fondo > 10** siempre. Amplitud típica ~135 ppm.
- Firma ROT: *"un pico significativo, aislado, de baja amplitud"* — en la
  práctica el fundamental más su primer armónico.
- Ruido del periodograma de amplitud TESS: ~10 ppm (brillantes), ~30 ppm a
  V=8, ~100 ppm a V=10, ~200 ppm a V=12.

### Pedersen et al. 2019, ApJL 872, L9

arXiv:1901.07576. Clasificación visual independiente por varios autores, con
categoría explícita de **"uncertain classification"**. [NV] No define umbrales
numéricos propios.

### Labadie-Bartz et al. 2022, AJ 163, 226

arXiv:2010.13905. [NV] **No define umbrales numéricos**: clasificación por
inspección visual. No usar como fuente de umbrales.

### Barraza et al. 2022, ApJ 924, 2

doi:10.3847/1538-4357/ac3335 · arXiv:2202.01022. **Pendiente**: la búsqueda
bibliográfica no localizó el paper y sus criterios quedaron sin revisar. El
DOI y el arXiv ID están confirmados en `docs/GOLDEN_SAMPLE.md`.

---

## 4. Variables elipsoidales

### Gomel et al. 2023, A&A 674, A19 — el criterio ELL más cuantitativo

arXiv:2206.06032 · doi:10.1051/0004-6361/202243626, Sec. 2 [V]

Ajustan **tres armónicos** de la frecuencia orbital y definen
`A_i = sqrt(a_ic^2 + a_is^2)`.

Filtro de calidad del período:

    numero de transitos G limpios > 25
    0.25 < P < 2.5 d
    P / P_err > 10
    "frequencygram peak" > 12    (altura en unidades de sigma del frecuencigrama)

Filtro de **forma** ELL:

    0.33 < A2 / (rango de G) < 0.6
    A2 / A2_err > 10
    A1/A1_err > 3  o  A3/A3_err > 3   <- "minimos NO iguales"
    A1/A2 < 1  y  A3/A2 < 0.3         <- razon de armonicos tipica de ELL

De ~20 millones de candidatos quedan 22.914 sistemas; tras q_min > 0.5, 6.306.

> **Este es un test de FORMA sobre el fold, ortogonal a la fuerza del pico.**
> Directamente computable sobre los folds de este repo.

### Que ELL se sobre-asigna: advertencias publicadas [V]

- Gomel+2023 Sec. 5.1: binarias de contacto y elipsoidales *"no es fácil
  diferenciar entre las dos, ya que ambas modulaciones se extienden sobre todo
  el período binario y tienen formas similares"*. Mitigación: excluir
  **P < 0.25 d** (Rucinski 2010) y exigir mínimos desiguales.
- Contaminación medida: ~0.5% de RR Lyrae en el catálogo; admiten
  *"un nivel de contaminación desconocido"*.
- **Desacuerdo documentado entre catálogos**: Gaia DR3 4042390512917208960 es
  elipsoidal para Gaia y OGLE, y binaria de contacto eclipsante para ASAS-SN.
- **La degeneración ROT<->ELL es exactamente un factor 2 en período**
  (Sec. 5.2): dos fuentes clasificadas ROT en Gaia DR2 con la mitad del
  período resultaron elipsoidales.
- Balona+2019 Sec. 3: *"Los efectos de marea no pueden distinguirse fácilmente
  de la modulación rotacional. La ambigüedad entre las clasificaciones ELL y
  ROT puede romperse si hay modulación significativa de amplitud o de
  frecuencia."*

[NV] No se encontró una tasa publicada de falsos positivos de la clase ELL.

---

## 5. Abstenerse de clasificar es práctica estándar

| Trabajo | Mecanismo | Ref. |
|---|---|---|
| McQuillan+2014 | `nan` en P_rot/sigma_P/LPH/w; publican `w` *"para que los usuarios elijan su propio umbral"* | Sec. 2, Ap. A |
| Burssens+2020 | `?` para incierta (`SLF?`, `rot?`), `cont.`, **`PQ` (poor quality)**, `+` para tipos coexistentes | Sec. 2.3 |
| Balona+2019 | **guion** si es constante o no hay asignación posible; `?` para incierta; **barra `ROT/ELL`** si dos clases son igualmente aceptables | Sec. 4 |
| Pedersen+2019 | categoría "uncertain classification" | Sec. 2.2 |
| Santos+2019 | para señales Type 2 y 3 *"no proveemos un período"* | Sec. 3.2 |
| Fetherolf+2023 | la estrella no entra al catálogo | Sec. 2.4 |

**Patrón común**: ningún catálogo fuerza una clase para toda estrella. Todos
tienen (a) categoría de abstención, (b) un score continuo publicado para que
el usuario elija su corte, o ambas.

---

## 6. Aplicación a este pipeline

### Con solución establecida en la literatura

1. **Mezclar `power` de LS y de ACF en un umbral único**: no hacerlo, y no hay
   conversión. Usar umbrales separados (`LPH_ACF >= 0.15`, `power_LS > 0.01`)
   más requisito de acuerdo dentro del 10-20% admitiendo P/2 y 2P. O bien la
   altura-en-sigmas de Gomel+2023, que sí es común a los dos.
2. **Ruido rojo SLF que imita periodicidad**: pre-whitening primero, S/N >= 5
   en amplitud, ruido medido en ventana de 1 d^-1 sobre el espectro residual.
   No reportar FAP analíticas para OB.
3. **ELL sobre-asignada**: test de armónicos de Gomel+2023 sobre el fold
   (A1/A2 < 1, A3/A2 < 0.3, A2/A2_err > 10, A1/A1_err > 3). Si falla, degradar
   a `ROT/ELL` o a la clase de abstención.
4. **Abstenerse**: clase explícita + publicar los scores continuos.

### Sin solución establecida

- **No existe un umbral de altura de pico de ACF calibrado para OB con TESS.**
  Los 0.15 / 0.3 son de estrellas frías tipo solar en Kepler, con años de
  baseline y ruido casi blanco. Trasladarlos a TESS/OB es una extrapolación, y
  hay que decirlo así.
- **No existe escala común LS<->ACF** más allá de la altura-en-sigmas.
- **La vía rigurosa para fijar el umbral propio es inyección-recuperación**
  (VanderPlas Sec. 7.4.2): inyectar sinusoides de amplitud conocida sobre
  curvas OB reales, que ya traen el SLF puesto, y medir la curva ROC. Las 39
  estrellas revisadas a mano son una versión pequeña de eso; McQuillan usó
  16.789 estrellas inspeccionadas visualmente para fijar C y w_thres.

### Advertencia sobre el corte P > 10 d

La recomendación de descartar nu < 0.1 d^-1 tiene tres referencias detrás
(Burssens+2020, Bowman+2020, Fetherolf+2023) y es sólida en general, PERO los
contaminantes ELL medidos en `results/clasificacion_review20.csv` tienen
períodos de 2.9 a 7.1 d, todos por debajo de 10 d. Ese corte no ataca el falso
positivo que motivó esta búsqueda.

---

## Fuentes

- McQuillan, Aigrain & Mazeh 2013, MNRAS 432, 1203 — arXiv:1303.6787 · doi:10.1093/mnras/stt536
- McQuillan, Mazeh & Aigrain 2014, ApJS 211, 24 — arXiv:1402.5694 · doi:10.1088/0067-0049/211/2/24
- Angus et al. 2018, MNRAS 474, 2094 — arXiv:1706.05459 · doi:10.1093/mnras/stx2109
- VanderPlas 2018, ApJS 236, 16 — arXiv:1703.09824 · doi:10.3847/1538-4365/aab766
- Bowman et al. 2019, NatAst 3, 760 — arXiv:1905.02120 · doi:10.1038/s41550-019-0768-1
- Bowman et al. 2020, A&A 640, A36 — arXiv:2006.03012 · doi:10.1051/0004-6361/202038224
- Burssens et al. 2020, A&A 639, A81 — arXiv:2005.09658 · doi:10.1051/0004-6361/202037700
- Balona et al. 2019, MNRAS 485, 3457 — arXiv:1902.09470 · doi:10.1093/mnras/stz586
- Pedersen et al. 2019, ApJL 872, L9 — arXiv:1901.07576 · doi:10.3847/2041-8213/ab01e1
- Labadie-Bartz et al. 2022, AJ 163, 226 — arXiv:2010.13905 · doi:10.3847/1538-3881/ac5abd
- Santos et al. 2019, ApJS 244, 21 — arXiv:1908.05222 · doi:10.3847/1538-4365/ab3b56
- Santos et al. 2021, ApJS 255, 17 — arXiv:2107.02217 · doi:10.3847/1538-4365/ac033f
- Fetherolf et al. 2023, AJ 165, 71 — arXiv:2208.11721 · doi:10.3847/1538-3881/acacf3
- Gomel et al. 2023, A&A 674, A19 — arXiv:2206.06032 · doi:10.1051/0004-6361/202243626
- Breger et al. 1993, A&A 271, 482 (origen de S/N=4; citado, no leído)
- Ceillier et al. 2017, A&A 605, A111 (origen de G_ACF/H_ACF/H_CS; citado vía Santos+2019, no leído)
- Baran et al. 2015 (origen de S/N=5 para datasets cortos; citado vía Burssens+2020, no leído)
- Loumos & Deeming 1978 (criterio de resolución 1.5/DeltaT; citado vía Burssens+2020, no leído)
- Rucinski 2010 (P < 0.25 d para binarias de contacto; citado vía Gomel+2023, no leído)
- Schwarzenberg-Czerny 2003 (corrección por ruido correlacionado; citado vía Burssens+2020, no leído)
