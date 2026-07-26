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

Gate: peaks con sigma_top > SIGMA_MAX se relabelan a 'Rndm' (o se descartan)
antes de la cascada Path-2.

Nota: `per` aquí es SIEMPRE el período detectado (LS/ACF). La columna
`per_ogle` (referencia OGLE) nunca debe sobreescribirse con él.
"""
import numpy as np
import pandas as pd

from .config import BRF_MODEL, CLASS_NAMES, SIGMA_MAX, WEIGHTS_DIR


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


def load_cnn(name_or_path):
    """Construye la CNN y carga pesos: nombre de training (en WEIGHTS_DIR)
    o path directo a un checkpoint."""
    import os
    path = str(name_or_path)
    if os.sep not in path:
        path = str(WEIGHTS_DIR / path / "cp.ckpt")
    model = make_model()
    model.load_weights(path)
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

    # --- CNN: pasadas MC ----------------------------------------------------
    p_mc = cnn_mc_probs(model, X, n_iter=n_iter, batch_size=batch_size)  # (n_iter,N,8)
    mc_mean = p_mc.mean(0)
    mc_std = p_mc.std(0)
    top = mc_mean.argmax(1)
    sigma_top = mc_std[np.arange(N), top]
    entropy = -(mc_mean * np.log(mc_mean + 1e-12)).sum(1)

    # --- BRF sobre cada pasada MC (un solo batch plano) ----------------------
    valid = np.isfinite(amp) & np.isfinite(per)      # amp NaN = flujo <= 0
    brf_class = np.array([None] * N, dtype=object)
    brf_prob = np.full(N, np.nan)
    sigma_brf = np.full(N, np.nan)
    instability = np.full(N, np.nan)

    if valid.any():
        idx_v = np.where(valid)[0]
        nv = len(idx_v)
        flat = p_mc[:, idx_v, :].reshape(n_iter * nv, len(CLASS_NAMES))
        feat = brf_features(flat,
                            np.tile(per[idx_v], n_iter),
                            np.tile(amp[idx_v], n_iter), brf)
        pp = brf.predict_proba(feat).reshape(n_iter, nv, -1)   # (n_iter,nv,8)

        pp_median = np.median(pp, axis=0)                      # (nv,8)
        pred = pp_median.argmax(1)                             # clase ganadora por mediana
        cls_per_pass = pp.argmax(2)                            # (n_iter,nv)

        brf_class[idx_v] = [CLASS_NAMES[c] for c in pred]
        brf_prob[idx_v] = pp_median[np.arange(nv), pred]
        sigma_brf[idx_v] = pp[:, np.arange(nv), pred].std(0)
        instability[idx_v] = (cls_per_pass != pred[None, :]).mean(0)

    out = peaks_df[["TIC", "sector", "source", "per", "power",
                    "prominence", "width"]].copy()
    out["amplitude"] = amp
    out["brf_class"] = brf_class
    out["brf_prob"] = brf_prob
    out["instability"] = instability
    out["sigma_brf"] = sigma_brf
    out["entropy"] = entropy
    out["sigma_top"] = sigma_top
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
