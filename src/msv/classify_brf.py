"""Clasificación CNN (MC-dropout) → BRF con gate de incertidumbre.

TensorFlow se importa de forma LAZY: solo `make_model`/`load_cnn`/
`cnn_mc_probs` lo requieren. El resto (features BRF, gate) funciona en el
env base sin TF sobre probabilidades CNN precomputadas.

Métricas de incertidumbre por peak (sobre `n_iter` pasadas MC con Dropout
activo, ver VALIDACION_Y_COMPARACION_MODELOS.md):
  - sigma_top    σ MC de la probabilidad CNN de la clase top (media MC).
  - entropy      entropía predictiva del softmax CNN medio.
  - sigma_brf    σ entre pasadas de la prob BRF de la clase ganadora.
  - instability  fracción de pasadas cuya clase BRF difiere de la ganadora.

El estimador central es la MEDIA entre pasadas, no la mediana. Las medianas
por clase no suman 1 (mínimo observado 0.435 sobre 857 picos), así que no
forman una distribución, y σ es la dispersión alrededor de la media: la
mediana MC de la clase ganadora está apilada contra 1 y `mediana + σ` excede
1 en el 89% de los picos. El intervalo se reporta como percentiles p16/p84,
que sí respetan la asimetría (semiancho inferior 0.085 contra 0.003 arriba).

Gate: peaks con sigma_top > SIGMA_MAX se relabelan a 'Rndm' (o se descartan)
antes de la cascada Path-2.

Nota: `per` aquí es SIEMPRE el período detectado (LS/ACF). La columna
`per_ogle` (referencia OGLE) nunca debe sobreescribirse con él.
"""
import numpy as np
import pandas as pd

from .config import (BRF_MODEL, CLASS_GROUPS, CLASS_NAMES, SIGMA_MAX,
                     WEIGHTS_DIR)


def _get_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError as ex:
        raise ImportError(
            "TensorFlow no está instalado en este entorno. La predicción CNN "
            "corre en el env con TF (ver environment.txt); el resto de msv "
            "funciona sin TF."
        ) from ex


def make_model():
    """CNN 2D-hist (idéntica a Paper_OGLE/CNN_2dhist_function.make_model)."""
    tf = _get_tf()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=(32, 32, 1), activation="relu", padding="same"),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=0.1),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )
    return model


def _load_tf_checkpoint(model, path):
    """Copia los pesos de un checkpoint TF (cp.ckpt) capa por capa.

    Keras 3 dejó de aceptar ese formato en `load_weights` (solo `.keras`,
    `.weights.h5` y los `.h5` legacy), pero `tf.train.load_checkpoint` sigue
    leyéndolo. Las variables vienen como `layer_with_weights-<i>/{kernel,bias}`
    numeradas en el orden de las capas CON pesos, que es el mismo orden en que
    las expone el modelo, así que el mapeo es posicional.
    """
    tf = _get_tf()
    reader = tf.train.load_checkpoint(path)
    layers = [layer for layer in model.layers if layer.weights]
    for position, layer in enumerate(layers):
        prefix = f"layer_with_weights-{position}/"
        layer.set_weights([
            reader.get_tensor(prefix + "kernel/.ATTRIBUTES/VARIABLE_VALUE"),
            reader.get_tensor(prefix + "bias/.ATTRIBUTES/VARIABLE_VALUE"),
        ])


def load_cnn(name_or_path):
    """Construye la CNN y carga pesos: nombre de training (en WEIGHTS_DIR)
    o path directo a un checkpoint."""
    import os
    path = str(name_or_path)
    if os.sep not in path:
        path = str(WEIGHTS_DIR / path / "cp.ckpt")
    model = make_model()
    try:
        model.load_weights(path)
    except (ValueError, OSError):
        _load_tf_checkpoint(model, path)
    return model


def load_brf(path=BRF_MODEL):
    import joblib
    return joblib.load(path)


def cnn_mc_probs(model, X, n_iter=50, batch_size=512):
    """`n_iter` pasadas con Dropout activo (training=True) -> (n_iter, N, 8)."""
    N = len(X)
    out = np.empty((n_iter, N, len(CLASS_NAMES)), dtype=np.float32)
    for k in range(n_iter):
        for i0 in range(0, N, batch_size):
            batch = X[i0:i0 + batch_size]
            out[k, i0:i0 + len(batch)] = model(batch, training=True).numpy()
    return out


def cnn_ensemble_probs(models, X, batch_size=512):
    """Una pasada DETERMINÍSTICA por modelo -> (n_models, N, 8).

    Alternativa a MC-dropout: el desacuerdo entre entrenamientos independientes
    (los 7 checkpoints de config.MODELS) en vez de la sensibilidad a apagar
    neuronas. Devuelve el mismo shape que `cnn_mc_probs`, así que el resto del
    pipeline (aggregate_mc, group_probs) no cambia.
    """
    N = len(X)
    out = np.empty((len(models), N, len(CLASS_NAMES)), dtype=np.float32)
    for k, model in enumerate(models):
        for i0 in range(0, N, batch_size):
            batch = X[i0:i0 + batch_size]
            out[k, i0:i0 + len(batch)] = model(batch, training=False).numpy()
    return out


def cnn_mc_predict(model, X, n_iter=50, batch_size=512):
    """Media y σ de las pasadas MC: (N,8), (N,8)."""
    preds = cnn_mc_probs(model, X, n_iter=n_iter, batch_size=batch_size)
    return preds.mean(0), preds.std(0)


def brf_features(cnn_probs, per, amplitud, brf):
    """Arma las 10 features del BRF (8 probs CNN + amplitud + per) en el
    orden EXACTO de brf.feature_names_in_."""
    feat = pd.DataFrame(np.asarray(cnn_probs), columns=CLASS_NAMES)
    feat["amplitud"] = np.asarray(amplitud, dtype=float)
    feat["per"] = np.asarray(per, dtype=float)
    return feat[list(brf.feature_names_in_)]


def brf_mc_probs(p_mc, per, amplitud, brf):
    """BRF sobre cada pasada MC -> (n_iter, N, 8), NaN donde `amp`/`per` no valen.

    Separado de `aggregate_mc` porque el detalle por clase (mediana y σ de las
    8 probabilidades, no solo la de la ganadora) es lo que se grafica.
    """
    p_mc = np.asarray(p_mc, dtype=np.float32)
    n_iter, N, _ = p_mc.shape
    per = np.asarray(per, dtype=float)
    amp = np.asarray(amplitud, dtype=float)

    valid = np.isfinite(amp) & np.isfinite(per)      # amp NaN = flujo <= 0
    out = np.full((n_iter, N, len(CLASS_NAMES)), np.nan)
    if not valid.any():
        return out, valid

    idx_v = np.where(valid)[0]
    nv = len(idx_v)
    flat = p_mc[:, idx_v, :].reshape(n_iter * nv, len(CLASS_NAMES))
    feat = brf_features(flat, np.tile(per[idx_v], n_iter),
                        np.tile(amp[idx_v], n_iter), brf)
    out[:, idx_v, :] = brf.predict_proba(feat).reshape(n_iter, nv, -1)
    return out, valid


def group_probs(pp, class_groups=None):
    """Suma las columnas de la SALIDA del BRF según `config.CLASS_GROUPS`.

    Agrupar antes del argmax importa: una pasada con M=0.30, DST=0.25, RR=0.10
    contra ELL=0.32 da Pulsating 0.65, y se perdería tomando el argmax primero.
    Devuelve (probs_agrupadas, nombres_de_grupo).
    """
    groups = class_groups or CLASS_GROUPS
    names = list(dict.fromkeys(groups[name] for name in CLASS_NAMES))
    stacked = np.stack(
        [np.asarray(pp)[..., [i for i, name in enumerate(CLASS_NAMES)
                              if groups[name] == group]].sum(axis=-1)
         for group in names], axis=-1)
    return stacked, names


def vote_fractions(per_pass):
    """Fracción de pasadas en que cada clase es el argmax -> (N, n_clases).

    Complementa a la media: la media puntúa cuán alto se instala una clase,
    el voto cuán seguido gana. Las dos se separan cuando una clase ocupa un
    nivel medio-alto muy parejo sin dominar ninguna pasada, que es el caso de
    ELL bajo el BRF (media 0.673 contra voto 0.715, frente a Pulsating con
    media 0.546 y voto 0.850).
    """
    per_pass = np.asarray(per_pass)
    n_classes = per_pass.shape[-1]
    winners = np.where(np.isnan(per_pass), -np.inf, per_pass).argmax(axis=-1)
    return np.stack([(winners == index).mean(axis=0)
                     for index in range(n_classes)], axis=-1)


def apply_vote_stability_gate(scores, votes, names, gated_class="ELL",
                              vote_min=0.5):
    """Le quita el label a `gated_class` cuando no gana la mayoría de pasadas.

    El pico cede al subcampeón, no a 'Rndm': la evidencia dice que ELL no es
    la clase, no que el pico sea ruido (de hecho la mayoría termina en Rndm
    por sí sola, pero unos pocos van a E o Pulsating).

    Es un gate y no un reescalado del score porque penalizar solo a ELL
    multiplicando su probabilidad la dejaría medida en otra escala que el
    resto de las clases, y el argmax entre escalas distintas no significa
    nada. Como umbral, la mayoría simple es el único punto que no hay que
    calibrar: `vote_min=0.5` es "ELL gana más de la mitad de las pasadas".

    Justificación física: el BRF aporta `amplitud` y `per`, las dos fuera del
    rango de entrenamiento, y en ese régimen aterriza en una constante que no
    es neutra (ELL 0.51 contra Pulsating 0.23) — un corrimiento fijo hacia
    ELL. Sacando el BRF la inestabilidad de ELL cae de 0.265 a 0.143.
    """
    if gated_class not in names:
        return np.asarray(scores).argmax(axis=1)
    scores = np.array(scores, dtype=float, copy=True)
    index = names.index(gated_class)
    unstable = np.asarray(votes)[:, index] < vote_min
    scores[unstable, index] = -np.inf
    return scores.argmax(axis=1)


def aggregate_mc(p_mc, per, amplitud, brf):
    """BRF sobre cada pasada MC de la CNN -> clase e incertidumbre por peak.

    `p_mc` es (n_iter, N, 8), la salida cruda de `cnn_mc_probs`. Separada de
    `classify_peaks` porque la CNN y el BRF no caben en el mismo env: los
    scripts step_cnn / step_brf se pasan justamente este array.
    """
    p_mc = np.asarray(p_mc, dtype=np.float32)
    n_iter, N, _ = p_mc.shape
    mc_mean = p_mc.mean(0)
    mc_std = p_mc.std(0)
    top = mc_mean.argmax(1)
    sigma_top = mc_std[np.arange(N), top]
    entropy = -(mc_mean * np.log(mc_mean + 1e-12)).sum(1)

    pp, valid = brf_mc_probs(p_mc, per, amplitud, brf)

    brf_class = np.array([None] * N, dtype=object)
    brf_prob = np.full(N, np.nan)
    sigma_brf = np.full(N, np.nan)
    instability = np.full(N, np.nan)

    if valid.any():
        idx_v = np.where(valid)[0]
        nv = len(idx_v)
        pp_v = pp[:, idx_v, :]
        pp_mean = pp_v.mean(axis=0)                            # (nv,8)
        pred = pp_mean.argmax(1)                               # clase ganadora por media
        cls_per_pass = pp_v.argmax(2)                          # (n_iter,nv)

        brf_class[idx_v] = [CLASS_NAMES[c] for c in pred]
        brf_prob[idx_v] = pp_mean[np.arange(nv), pred]
        sigma_brf[idx_v] = pp_v[:, np.arange(nv), pred].std(0)
        instability[idx_v] = (cls_per_pass != pred[None, :]).mean(0)

    return {"brf_class": brf_class, "brf_prob": brf_prob,
            "instability": instability, "sigma_brf": sigma_brf,
            "entropy": entropy, "sigma_top": sigma_top}


def classify_peaks(peaks_df, model, brf, X=None, amp=None, n_iter=30,
                   batch_size=512):
    """Pipeline completo por peak: CNN MC-dropout → BRF por pasada → tabla.

    `peaks_df` requiere TIC, sector, source, per, power, prominence, width;
    si trae `hist2d` (schema peaks.parquet) y `amplitude`, X/amp se derivan
    de ahí — si no, pasar `X` (N,32,32,1) y `amp` (N,) de features.build_cube.

    Devuelve DataFrame con el schema de brf_mc_peaks_*.csv:
      TIC, sector, source, per, power, prominence, width, amplitude,
      brf_class, brf_prob, instability, sigma_brf, entropy, sigma_top
    """
    peaks_df = peaks_df.reset_index(drop=True)
    N = len(peaks_df)

    if X is None:
        if "hist2d" not in peaks_df.columns:
            raise ValueError("peaks_df sin columna hist2d: pasar X explícito "
                             "(ver features.build_cube)")
        X = np.stack([np.asarray(h, dtype=np.float32).reshape(32, 32)
                      for h in peaks_df["hist2d"].values])[..., np.newaxis]
    if amp is None:
        if "amplitude" not in peaks_df.columns:
            raise ValueError("peaks_df sin columna amplitude: pasar amp explícito")
        amp = peaks_df["amplitude"].to_numpy(dtype=float)
    per = peaks_df["per"].to_numpy(dtype=float)

    p_mc = cnn_mc_probs(model, X, n_iter=n_iter, batch_size=batch_size)  # (n_iter,N,8)
    metrics = aggregate_mc(p_mc, per, amp, brf)

    out = peaks_df[["TIC", "sector", "source", "per", "power",
                    "prominence", "width"]].copy()
    out["amplitude"] = amp
    for column, values in metrics.items():
        out[column] = values
    return out


def apply_gate(df_pred, sigma_max=SIGMA_MAX, mode="relabel",
               sigma_col="sigma_top"):
    """Gate de incertidumbre: peaks con σ > sigma_max se relabelan a 'Rndm'
    (mode='relabel') o se descartan (mode='drop'), ANTES de Path-2."""
    d = df_pred.copy()
    bad = d[sigma_col] > sigma_max
    if mode == "relabel":
        d.loc[bad, "brf_class"] = "Rndm"
    elif mode == "drop":
        d = d[~bad].reset_index(drop=True)
    else:
        raise ValueError(f"mode='{mode}' no soportado")
    return d
