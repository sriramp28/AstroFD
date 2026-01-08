#!/usr/bin/env python3
# utils/backend.py
import numpy as np


def get_backend(name: str):
    if name is None:
        return np, "numpy"
    key = str(name).lower()
    if key in ("np", "numpy"):
        return np, "numpy"
    if key in ("cp", "cupy"):
        try:
            import cupy as cp
        except Exception as exc:
            raise RuntimeError(f"cupy backend requested but unavailable: {exc}") from exc
        return cp, "cupy"
    raise ValueError(f"Unknown backend: {name}")


def to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:
        pass
    return np.asarray(arr)
