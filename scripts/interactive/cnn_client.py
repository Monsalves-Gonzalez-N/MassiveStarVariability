"""Client side of `predict_server.py`: keeps the TensorFlow subprocess warm.

Starting the server costs seconds (the TF import plus building the graph), so
it is started once per Streamlit session and reused for every click. It dies
with the parent.
"""
import io
import os
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

TF_PYTHON = os.environ.get("MSV_TF_PYTHON", "/opt/anaconda3/envs/tf_env/bin/python")
SERVER = Path(__file__).with_name("predict_server.py")


class PredictServer:
    def __init__(self, checkpoint, tf_python=TF_PYTHON):
        environment = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
        self.process = subprocess.Popen(
            [tf_python, str(SERVER), str(checkpoint)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment,
        )
        # TensorFlow writes its own warnings to stderr before the server gets
        # to speak, so the handshake has to skip lines rather than assume the
        # first one is ours. Reading to EOF on a live process would deadlock.
        noise = []
        while True:
            line = self.process.stderr.readline()
            if line.strip() == b"ready":
                break
            if not line:
                raise RuntimeError("predict_server no arranco:\n"
                                   + b"".join(noise).decode(errors="replace"))
            noise.append(line)
        self.startup_messages = b"".join(noise).decode(errors="replace")
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self):
        """A full stderr pipe blocks the server mid-predict; nobody reads it
        after the handshake, so it has to be emptied in the background."""
        for _ in iter(self.process.stderr.readline, b""):
            pass

    def predict(self, images):
        images = np.asarray(images, np.float32)
        if images.ndim == 3:
            images = images[..., np.newaxis]
        buffer = io.BytesIO()
        np.save(buffer, images, allow_pickle=False)
        payload = buffer.getvalue()
        self.process.stdin.write(struct.pack("<Q", len(payload)))
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        (length,) = struct.unpack("<Q", self.process.stdout.read(8))
        return np.load(io.BytesIO(self.process.stdout.read(length)),
                       allow_pickle=False)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(struct.pack("<Q", 0))
            self.process.stdin.flush()
            self.process.wait(timeout=10)


if __name__ == "__main__":
    server = PredictServer(sys.argv[1])
    probabilities = server.predict(np.random.rand(3, 32, 32).astype(np.float32))
    print(probabilities.shape, probabilities.sum(axis=1))
    server.close()
