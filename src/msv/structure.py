"""Descriptores de estructura y triage: ¿hay período, y cuántos?

La CNN contesta "¿qué forma tiene este fold?" y sobre eso acierta. Lo que no
sabe es si la frecuencia EXISTE: medido sobre 6488 candidatos, un pico que la
red llama ELL tiene menos potencia real en su fundamental que uno que llama
LPV. Las dos preguntas se separan acá:

    prewhitening  ->  ¿hay período? ¿cuántos?     (este módulo)
    CNN           ->  ¿qué forma tiene ese fold?

Los descriptores salen de las componentes que `prewhiten.extract_components`
ya devuelve, sin costo extra. El único que necesita una decisión es el
agrupamiento: dos frecuencias separadas por menos de una resolución de
Rayleigh 1/T **no están resueltas**. Lo que hay ahí es UNA estructura ancha, y
el peine de componentes que el prewhitening deja no son sus miembros sino
marcas de dónde está la joroba (README §5). Por eso un grupo apiñado de tres o
más componentes es `irregular` y no una pulsante multiperiódica: varias
frecuencias SEPARADAS son modos independientes; varias APIÑADAS son una sola
estructura que el propio prewhitening cortó en pedazos.
"""
import numpy as np

from msv.prewhiten import cluster_components

# Componentes en un mismo grupo no resuelto a partir de las cuales la
# estructura es una joroba ancha y no un conjunto de modos. Con dos no alcanza:
# un fundamental y una componente vecina mal restada son dos.
CLUSTER_SIZE_IRREGULAR = 3

SIN_SENAL = "sin_senal"
IRREGULAR = "irregular"
MULTIPERIODICA = "multiperiodica"
UNIPERIODICA = "uniperiodica"


def describe_structure(components, relative_flux, residual, baseline):
    """Descriptores de estructura de una estrella-sector.

    `components` y `residual` son la salida de `extract_components` sobre
    `relative_flux` (ppt); `baseline` es el span temporal en días, que fija la
    resolución de Rayleigh con la que se agrupa.
    """
    rayleigh = 1.0 / float(baseline)
    frequencies = np.array([component["frequency"] for component in components],
                           float)
    labels = cluster_components(components, rayleigh)
    sizes = {label: labels.count(label) for label in set(labels)}

    total_variance = float(np.var(relative_flux))
    coherent_fraction = (float(1.0 - np.var(residual) / total_variance)
                         if total_variance > 0 else np.nan)

    if len(frequencies) > 1:
        separations = np.diff(np.sort(frequencies)) / rayleigh
        median_separation = float(np.median(separations))
        min_separation = float(separations.min())
    else:
        median_separation = np.nan
        min_separation = np.nan

    if components:
        dominant = int(np.argmax([component["amplitude"]
                                  for component in components]))
    else:
        dominant = None

    return {
        "n_components": len(components),
        "n_clusters": len(sizes),
        "max_cluster_size": max(sizes.values()) if sizes else 0,
        "cluster_sizes": " ".join(str(sizes[label]) for label in sorted(sizes)),
        "coherent_fraction": coherent_fraction,
        "rayleigh_cpd": rayleigh,
        "baseline": float(baseline),
        "std_ppt": float(np.std(relative_flux)),
        "residual_ppt": float(np.std(residual)),
        "separacion_mediana_rayleigh": median_separation,
        "separacion_minima_rayleigh": min_separation,
        "dominante": dominant + 1 if dominant is not None else np.nan,
        "frecuencia_dominante": (float(frequencies[dominant])
                                 if dominant is not None else np.nan),
        "labels": labels,
    }


def triage(descriptors, solo_un_grupo=True):
    """Estructura de la estrella-sector a partir de sus descriptores.

        sin componentes sobre SNR 4                  -> sin_senal
        todo en UN grupo con >= 3 componentes        -> irregular
        >= 2 grupos resueltos                        -> multiperiodica
        1 grupo resuelto con 1-2 componentes         -> uniperiodica

    Ninguna rama mira la clase de la red: esta es la parte que el prewhitening
    puede contestar y la CNN no.

    `solo_un_grupo=True` es la regla literal del plan: la estrella es irregular
    si TODAS sus componentes caen en un mismo grupo no resuelto. Medida sobre
    la golden esa rama casi no se dispara (3 de 810), y no atrapa el caso que
    la motiva: TIC 42889751 tiene un grupo de 3 componentes dentro de una
    resolucion de Rayleigh, mas 3 sueltas, o sea 4 grupos, y cae en
    `multiperiodica`. Con `solo_un_grupo=False` basta con que ALGUN grupo tenga
    3 componentes no resueltas para llamarla irregular, que es lo que dice el
    razonamiento del plan — un apinamiento es una joroba que el prewhitening
    corto en pedazos, y eso no deja de ser cierto porque la estrella ademas
    tenga otra frecuencia aparte.
    """
    if descriptors["n_components"] == 0:
        return SIN_SENAL
    apinado = descriptors["max_cluster_size"] >= CLUSTER_SIZE_IRREGULAR
    if apinado and (descriptors["n_clusters"] == 1 or not solo_un_grupo):
        return IRREGULAR
    if descriptors["n_clusters"] >= 2:
        return MULTIPERIODICA
    return UNIPERIODICA
