"""CNN inference over stdin/stdout, so the app can stay in an env without TensorFlow.

The pipeline is split across two conda envs on this machine: `msv` (astropy,
statsmodels, scipy) has no TensorFlow, and `tf_env` (Keras 3) cannot import
`msv` at all. Everything up to the 32x32 image is pure numpy, so the only piece
that has to live on the other side is `model.predict`. This runs there, keeps
the checkpoint warm, and answers one batch per request in milliseconds.

Protocol, both directions: 8-byte little-endian length, then a .npy payload.
  in   float32 (n, 32, 32, 1)
  out  float32 (n, len(CLASS_NAMES))
An empty request (length 0) shuts the server down.

Run by `app.py` as a subprocess; standalone only for debugging:
    /opt/anaconda3/envs/tf_env/bin/python predict_server.py <checkpoint_dir>
"""
import io
import struct
import sys

import numpy as np

# Duplicated from msv.config rather than imported: this process cannot import
# msv (no astropy/sklearn in tf_env). The app asserts the two agree at startup.
CLASS_NAMES = ["ELL", "M", "CEP", "DST", "E", "LPV", "RR", "Rndm"]
INPUT_SHAPE = (32, 32, 1)


def make_model():
    """Byte-for-byte the architecture of `msv.classify_brf.make_model`.

    Duplicated instead of imported for the same reason as CLASS_NAMES: this
    process cannot import msv. Any change there has to be mirrored here or the
    checkpoint stops loading positionally.
    """
    import tensorflow as tf

    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=INPUT_SHAPE, activation="relu", padding="same"),
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
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])


def load_weights(model, checkpoint):
    """Mirror of `msv.classify_brf.load_cnn`: Keras 3 no longer reads cp.ckpt,
    but `tf.train.load_checkpoint` does, and the variables are numbered in the
    order of the layers that have weights."""
    import tensorflow as tf

    try:
        model.load_weights(checkpoint)
        return
    except (ValueError, OSError):
        pass
    reader = tf.train.load_checkpoint(checkpoint)
    layers = [layer for layer in model.layers if layer.weights]
    for position, layer in enumerate(layers):
        prefix = f"layer_with_weights-{position}/"
        layer.set_weights([
            reader.get_tensor(prefix + "kernel/.ATTRIBUTES/VARIABLE_VALUE"),
            reader.get_tensor(prefix + "bias/.ATTRIBUTES/VARIABLE_VALUE"),
        ])


def read_message(stream):
    header = stream.read(8)
    if len(header) < 8:
        return None
    (length,) = struct.unpack("<Q", header)
    if length == 0:
        return None
    return np.load(io.BytesIO(stream.read(length)), allow_pickle=False)


def write_message(stream, array):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, np.float32), allow_pickle=False)
    payload = buffer.getvalue()
    stream.write(struct.pack("<Q", len(payload)))
    stream.write(payload)
    stream.flush()


def main():
    checkpoint = sys.argv[1]
    model = make_model()
    load_weights(model, checkpoint)
    # The first predict builds the graph and costs seconds; pay it before the
    # app can ask for anything, so every interactive click is warm.
    model.predict(np.zeros((1,) + INPUT_SHAPE, np.float32), verbose=0)
    sys.stderr.write("ready\n")
    sys.stderr.flush()

    while True:
        images = read_message(sys.stdin.buffer)
        if images is None:
            break
        write_message(sys.stdout.buffer,
                      model.predict(images.astype(np.float32), verbose=0))


if __name__ == "__main__":
    main()
