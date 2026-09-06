# Tests de la contaminación ELL

Cada script imprime una tabla y no escribe nada. Todos importan `_common.py`,
que carga los picos y la **verdad a nivel de pico** (`load_peak_truth`).

```
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate CNN_TESS
export PYTHONPATH=src:scripts/analysis_ell
export MSV_BRF=$HOME/Dropbox/MassiveStarVariability/models/balanced_random_forest_model.joblib
python scripts/analysis_ell/06_final/pipeline_una_red.py
```

Las conclusiones están en `docs/HANDOFF_contaminacion_ELL.md`. Los tests de la
sesión del 2026-09-03 se borraron: medían con el argmax, usaban verdad a nivel
de estrella y leían después del BRF, así que sus números no son comparables con
estos. Lo que aportaron está resumido en §0 y §1 del handoff.

## 01_verdad — cómo se etiqueta

| script | qué mide | conclusión |
|---|---|---|
| `peak_truth.py` | verdad a nivel de pico contra nivel de estrella | 3 ELL reales, no 5: los picos "ELL" de TIC 89761288 son P/2 de una eclipsante |
| `clase_vs_periodo.py` | clase correcta contra período correcto | sólo 2 de 26 son clase-bien/período-mal, y son alias 2P |

## 02_donde_esta_la_senal — a qué nivel leer

| script | qué mide | conclusión |
|---|---|---|
| `cnn_vs_brf.py` | p_ELL, p_LPV, p_Rndm antes y después del BRF | **la clave**: p_LPV pasa de AUC 0.740 (BRF) a 0.958 (CNN) |
| `brf_quantization.py` | rango dinámico de cada nivel | el BRF tiene 500 árboles: piso 2.9e-4, la CNN llega a 1.5e-18 |
| `brf_damping.py` | σ entre pasadas antes y después del BRF | el BRF **no** comprime la dispersión (0.29 → 0.29) |
| `prob_geometry.py` | el vector completo, sin argmax | dónde cae la masa residual; el runner-up de las ELL reales nunca es Rndm ni LPV |
| `pass_structure.py` | estadísticos entre pasadas | las colas separan donde la media no |
| `rndm_tail.py` | la cola de Rndm es artefacto del max de 200? | no: la escalera de cuantiles es monótona, y 20 pasadas bastan |
| `significance.py` | Mann-Whitney exacto | acota el N=3 de las ELL confirmadas |

## 03_descartados — lo que se probó y NO se adoptó

| script | qué probó | por qué no |
|---|---|---|
| `norm_sweep.py` | p_LPV y cola de Rndm en 9 normalizaciones | `rank` ganaba sólo en la muestra chica; `log` alcanza |
| `lpv_subsets.py` | los 511 subconjuntos de normalizaciones | 9 corridas de CNN para pasar de 6/8 a 7/8, con peor margen |
| `lpv_rndm_log.py`, `lpv_rndm_frontier.py` | el plano (p_LPV, p_Rndm) | p_Rndm no aporta: las fronteras son idénticas |
| `replace_power.py` | ~200 scores de la CNN que reemplacen a `power` | el reemplazo resultó ser p_LPV otra vez |
| `peak_crowding.py` | contraste `power_1/power_2` como cuarto eje | AUC 0.465, azar: atrapa una estrella y nada más |
| `gate_rules.py`, `ell_gate_scan.py`, `ell_only_gate.py`, `final_gate.py`, `operating_point.py`, `gap_check.py` | gates basados en `power` y en la cola de Rndm bajo `binary` | funcionan (86–100% de pureza) pero `p_LPV` sola los supera sin periodograma |
| `lpv_cnn.py`, `lpv_only.py`, `lpv_final.py`, `tier_choice.py` | versiones intermedias de la regla | superadas por `06_final` |
| `confidence_audit.py` | pureza contra confianza a nivel de **pico** | denominador equivocado: la pureza es 7% por construcción |
| `seleccion_por_pclase.py` | elegir el pico por `p_clase` con desempate por `p_LPV` | 88.3% contra 90.9% de `p_LPV` sola, y la mitad de reproducible; no hay ni un empate exacto |

## 04_confianza — el número que acompaña a la clase

| script | qué mide | conclusión |
|---|---|---|
| `star_confidence.py` | pureza contra confianza a nivel de estrella | `prob` no separa (0.995 contra 0.991); σ y la cola sí |
| `saturacion_argmax.py` | cuánta información queda en el argmax y en la prob | el argmax acierta 14/16; la probabilidad satura |
| `reemplazar_gate.py` | el gate `prob >= 0.8` contra los no saturados | cuesta 3 de los 16 picos confirmados |
| `residual_failure.py` | la estrella que escapa, y el alias 2P | escalera armónica; 3 errores son período duplicado exacto |

## 05_irregular — el valor global por estrella

| script | qué mide | conclusión |
|---|---|---|
| `global_irregular.py` | candidatos a valor global | p_LPV medio sobre todos los picos; p_LPV+p_Rndm se invierte |
| `reporte_irregular.py` | el formato "clase con xx% irregular" | mediana 2.9% en las correctas contra 24.4% en las erradas |
| `valida_irregular.py` | el % irregular contra las notas humanas | las 3 "multiperiódico" caen en 36.9 / 38.2 / 38.9% |
| `irregular_por_clase.py` | condicionado a la clase | en ELL es criterio (AUC 1.000), en E es descripción |

## 06_final — el pipeline adoptado

| script | qué mide | conclusión |
|---|---|---|
| `lpv_por_modelo.py` | p_LPV por checkpoint, y dropout contra ensemble | `Number_ELL` es la mejor; el dropout corre sobre la peor |
| `lpv_subset_modelos.py` | los 254 subconjuntos de checkpoints | los que llegan a 7/8 tienen margen ×1.3: ruido |
| `una_sola_red.py` | los 7 checkpoints en las tres funciones | `Number_ELL` sola: clase 15/16, mejor que ensemble+BRF |
| `pipeline_una_red.py` | el pipeline final | selección del pico por menor p_LPV: 10/11 |

## build_all_norms.py

Regenera los cubos de entrada en las 9 normalizaciones. Sólo hace falta para
`03_descartados`; el pipeline final usa `log`, que es `results/cnn_input.npz`.
Control obligatorio: reconstruir `log` y verificar que reproduce
`results/cnn_input.npz['X']` bit a bit (`msv.features` aplica `.T[::-1]`
DESPUÉS de normalizar).
