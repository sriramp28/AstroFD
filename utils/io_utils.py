# utils/io_utils.py
import json
import os
from datetime import datetime

def make_run_dir(base="results", unique=False):
    """
    Create a run directory:
      unique=False -> results/YYYY-MM-DD/
      unique=True  -> results/YYYY-MM-DD/HH-MM-SS/
    """
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    if unique:
        path = os.path.join(base, date, now.strftime("%H-%M-%S"))
    else:
        path = os.path.join(base, date)
    os.makedirs(path, exist_ok=True)
    return path

def write_run_config(run_dir, settings):
    path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(path):
        return
    try:
        with open(path, "w") as f:
            json.dump(settings, f, indent=2, sort_keys=True, default=str)
    except Exception:
        return
