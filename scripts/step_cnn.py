#!/usr/bin/env python
"""Paso 2/3 (env tf_env): pasadas de la CNN sobre el .npz de entrada.

Por defecto corre el ENSEMBLE: una pasada determinística por cada checkpoint de
`config.MODELS`. Los 7 son el mismo paper, el mismo test y validation set, y
difieren en la estrategia de balanceo, así que su desacuerdo mide algo real
—"esta estrella cae distinto según cómo balancees"— en vez de la sensibilidad
de una red a apagar neuronas al azar. Con `--mc-dropout` se recupera el método
viejo. `sigma` e `instability` pasan a significar desacuerdo ENTRE MODELOS.

tf_env no puede `import msv` (le faltan astropy/statsmodels), así que se cargan
`msv.config` y `msv.classify_brf` directamente por path, sin ejecutar el
`__init__` del paquete. Es un shim, no una copia del código.

    MSV_WEIGHTS=~/ViT_VariableStars/pretrained/keras_checkpoints \
      python scripts/step_cnn.py results/cnn_input.npz results/cnn_mc.npz --model Number_DST
"""
import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"


def load_msv_module(name):
    """Importa `msv.<name>` sin ejecutar `msv/__init__.py`."""
    if "msv" not in sys.modules:
        package = types.ModuleType("msv")
        package.__path__ = [str(SRC / "msv")]
        sys.modules["msv"] = package
    spec = importlib.util.spec_from_file_location(f"msv.{name}", SRC / "msv" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"msv.{name}"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_in", help=".npz de step_cnn_export.py")
    parser.add_argument("npz_out", help=".npz con p_mc (n_iter, N, 8)")
    parser.add_argument("--model", default=None, help="nombre del checkpoint en WEIGHTS_DIR")
    parser.add_argument("--mc-dropout", action="store_true",
                        help="MC-dropout sobre UN checkpoint (config.MC_ITER "
                             "pasadas) en vez del ensemble. Es el método viejo: "
                             "en TIC 12675729 sus pasadas son bimodales y la "
                             "mediana de 20 sale a suerte")
    parser.add_argument("--n-iter", type=int, default=None, help="default config.MC_ITER")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    config = load_msv_module("config")
    classify_brf = load_msv_module("classify_brf")
    model_name = args.model or config.DEFAULT_MODEL
    n_iter = args.n_iter or config.MC_ITER

    data = np.load(args.npz_in, allow_pickle=True)
    cube = data["X"]

    if args.model and not args.mc_dropout:
        # Una pasada determinista de UN checkpoint: es lo que consume
        # step_clasificar_una_red.py. Antes `--model` se ignoraba sin
        # `--mc-dropout` y cargaba igual los 7.
        models = [classify_brf.load_cnn(model_name)]
        probabilities = classify_brf.cnn_ensemble_probs(models, cube,
                                                        batch_size=args.batch_size)
        label, n_iter = model_name, 1
    elif not args.mc_dropout:
        models = [classify_brf.load_cnn(name) for name in config.MODELS]
        probabilities = classify_brf.cnn_ensemble_probs(models, cube,
                                                        batch_size=args.batch_size)
        label = "+".join(config.MODELS)
        n_iter = len(config.MODELS)
    else:
        model = classify_brf.load_cnn(model_name)
        probabilities = classify_brf.cnn_mc_probs(model, cube, n_iter=n_iter,
                                                  batch_size=args.batch_size)
        label = model_name

    np.savez_compressed(args.npz_out, p_mc=probabilities.astype(np.float32),
                        model=label, n_iter=n_iter)
    print(f"-> {args.npz_out}   p_mc{probabilities.shape}  modelo {label}")


if __name__ == "__main__":
    main()
