"""Restart the source-built provider client with the same storage root."""
import os
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory() as directory:
    env = dict(os.environ, X2W_STORAGE_PATH=".")
    for operation in ("write", "read"):
        subprocess.run([sys.argv[1], operation], cwd=directory, env=env, check=True)
