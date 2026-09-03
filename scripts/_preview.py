"""Abrir los PDFs de revisión en Preview.

Preview cachea la versión vieja de un archivo que ya tiene abierto: si se
regenera el PDF y se hace `open` sin más, se ve el anterior. Por eso se cierra
primero.
"""
import platform
import subprocess


def open_in_preview(*paths):
    if platform.system() != "Darwin":
        return
    subprocess.run(["osascript", "-e", 'tell application "Preview" to quit'],
                   check=False)
    subprocess.run(["open", "-a", "Preview", *[str(path) for path in paths]],
                   check=False)
