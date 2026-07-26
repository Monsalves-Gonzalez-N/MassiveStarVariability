# build_catalog_v2.py — Especificación

Ruta alternativa de clasificación ("path 2") para testear la robustez de la distribución de clases frente al criterio de selección. Paralelo a `build_catalog.py`; no lo reemplaza.

## Objetivo

Generar un catálogo unificado LS+ACF donde la clase final por TIC se decide mediante una **cascada jerárquica**:

1. Clasificar candidatos **no periódicos** a nivel TIC (`No_periodo`, `Rndm`, `Irregular`).
2. Para el resto, clasificar por **voto de sectores** (moda) sobre los peaks periódicos.

## Entradas

- `data/catalogos/CNN_RF_prediction_LS_log.csv`
- `data/catalogos/CNN_RF_prediction_ACF_log.csv`

Columnas relevantes: `TIC`, `sector_list`, `final_class`, `CNN+RF_max_prob`, `power`, `period`.

Tagear cada fuente al cargar:
- `ls["source"] = "LS"`, `ls["cube_idx"] = ls.index`
- `acf["source"] = "ACF"`, `acf["cube_idx"] = acf.index`

## Parámetros

- `threshold = 0.8` — umbral de probabilidad sobre `CNN+RF_max_prob`. Exponerlo como input para facilitar cambios.
- `rng = np.random.default_rng(seed=42)` — usado solo en los desempates explícitos (ver Paso 2).

## Clases posibles

- **No periódicas**: `No_periodo`, `Rndm`, `LPV`.
- **Periódicas**: el resto (ej. eclipsante, rotational, etc.) — se decide por voto de sectores en el Paso 2.
- **Derivadas**: `Irregular` (mezcla Rndm+LPV sobre threshold), `Unconstrained` (sector sin peak sobre threshold), `Mixed` (empate en la moda del Paso 2).

---

## Paso 1 — Clasificación de candidatos a NO periódicos (granularidad TIC)

Se evalúa cada TIC como un todo (no por sector). Las reglas se aplican **en orden jerárquico**: la primera que matchea gana y el TIC sale del pipeline (no entra al Paso 2).

### 1.1 — `No_periodo`
Si el TIC no tiene ningún peak (ni en LS ni en ACF) → `final_class_tic = "No_periodo"`.

No aplica threshold de probabilidad: `No_periodo` significa ausencia de períodos detectados, no hay `CNN+RF_max_prob` que evaluar.

### 1.2 — `Rndm`
Si el TIC cumple **ambas** condiciones:
- **Todos** los peaks (LS + ACF, todos los sectores, a cualquier probabilidad) tienen `final_class == "Rndm"`, **y**
- Al menos un peak tiene `CNN+RF_max_prob ≥ threshold`.

→ `final_class_tic = "Rndm"`.

### 1.3 — `Irregular`
Si el TIC cumple **todas** las condiciones:
- **Todos** los peaks tienen `final_class ∈ {"Rndm", "LPV"}` (a cualquier probabilidad), **y**
- Al menos un peak es `LPV` (caso contrario caería en 1.2), **y**
- Al menos un peak tiene `CNN+RF_max_prob ≥ threshold`.

→ `final_class_tic = "Irregular"`.

### 1.4 — Fall-through
Si el TIC no matchea 1.1–1.3, pasa al Paso 2. Esto incluye:
- TICs con al menos un peak de clase periódica (aunque sea bajo threshold).
- TICs con solo peaks Rndm/LPV pero **ninguno** sobre el threshold.

En ambos casos, el Paso 2 puede terminar clasificándolos como `Unconstrained` si ningún sector aporta votos.

> **Criterio de diseño**: la presencia de *cualquier* peak periódico (incluso con baja probabilidad) impide clasificar el TIC como Rndm/Irregular en el Paso 1. Se prefiere mandar al Paso 2 y dejar que el mecanismo de `Unconstrained` maneje la incertidumbre.

---

## Paso 2 — Clasificación de candidatos a PERIÓDICOS (voto por sector)

Solo procesa TICs que no fueron clasificados en el Paso 1.

**Pre-filtro**: eliminar de la muestra todos los peaks con `final_class ∈ {"No_periodo", "Rndm", "LPV"}`. Los votos se construyen solo con peaks de clases periódicas.

### 2.1 — Voto por sector

Para cada grupo `(TIC, sector_list)`:

**Caso A — Sector sin peaks sobre threshold**
Si ningún peak del sector (ACF ni LS) tiene `CNN+RF_max_prob ≥ threshold`:
- Clasificar el sector como `Unconstrained`.
- Conservar el peak top de ACF y el peak top de LS (con su probabilidad real) para output.
- Si los peaks top de ACF y LS tienen la misma `CNN+RF_max_prob`, elegir uno aleatoriamente con `rng`.
- Un sector `Unconstrained` aporta **0 votos** a la moda.

**Caso B — Sector con al menos un peak sobre threshold**

1. Separar los peaks del sector en subgrupos **ACF** y **LS**.

2. **Peak ganador ACF**: dentro del subgrupo ACF del sector, seleccionar el peak con mayor `CNN+RF_max_prob`. Desempate: mayor `power`.

3. **Peak ganador LS**: dentro del subgrupo LS del sector, seleccionar el peak con mayor `CNN+RF_max_prob`. Desempate: menor `power`.

4. **Comparar ACF vs LS**:
   - Si `prob(ACF) > prob(LS)` → el sector aporta **1 voto** con la clase del peak ACF.
   - Si `prob(LS) > prob(ACF)` → el sector aporta **1 voto** con la clase del peak LS.
   - Si `prob(ACF) == prob(LS)` → el sector aporta **2 votos**: la clase del peak ACF y la clase del peak LS (pueden ser iguales o distintas).

5. Conservar los peaks ganadores (ACF y/o LS, según corresponda) para output.

### 2.2 — Voto por TIC (moda de los votos de sector)

Una vez procesados todos los sectores de un TIC:

- **Si todos los sectores son `Unconstrained`** → `final_class_tic = "Unconstrained"`.
- **Si al menos un sector aporta votos**:
  - Calcular la moda de todos los votos (sectores `Unconstrained` no aportan; sectores en empate ACF=LS aportan 2 votos).
  - Si hay **moda única** → `final_class_tic = <clase moda>`.
  - Si hay **empate en la moda** → `final_class_tic = "Mixed"`.

> Ejemplo: TIC con 3 sectores, donde el sector 1 es `Unconstrained`, el sector 2 aporta 1 voto "Eclipsante" (prob 0.99), y el sector 3 aporta 1 voto "Eclipsante". Moda = Eclipsante. El sector problemático no descalifica.

---

## Paso 3 — Salidas

Guardar en `data/catalogos/`:

- `catalog_path2_main.csv` — todas las filas relevantes (peaks ganadores de cada sector y peaks top de sectores `Unconstrained`) con columna añadida `final_class_tic` = clase decidida a nivel TIC. Se conservan `period`, `power`, `source`, `cube_idx` y demás columnas originales por sector para permitir visualización posterior.
- `catalog_path2_mixed.csv` — subset con los TICs cuyo `final_class_tic == "Mixed"`.
- `catalog_path2_{clase}.csv` — un CSV por cada valor distinto de `final_class_tic` (excluyendo `Mixed`), siguiendo el patrón de `build_catalog.py`. Incluye `No_periodo`, `Rndm`, `Irregular`, `Unconstrained`, y cada clase periódica observada.

## Resumen en consola

Imprimir al final:
- Total de TICs y filas en `catalog_path2_main`.
- `value_counts()` de `final_class_tic`.
- Número de TICs en `Mixed`.
- Número de TICs en `Unconstrained`.

## Notas de implementación

- La única fuente de aleatoriedad está en el Caso A del Paso 2 (desempate ACF vs LS en sectores `Unconstrained`). Usar `np.random.default_rng(seed=42)` por consistencia con path 1.
- El voto `Unconstrained` **no cuenta** para la moda (aporta 0 votos). Solo si *todos* los sectores de un TIC son `Unconstrained`, el TIC queda clasificado como `Unconstrained`.
- En empates ACF=LS dentro de un sector (Caso B, sub-caso 4), el sector aporta **2 votos** (uno por cada peak ganador). Esto puede afectar la moda.
- **No** hacer comparación con path 1 en este script. Esa comparación es un análisis posterior, fuera de alcance.

