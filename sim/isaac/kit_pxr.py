"""Make `pxr` importable in any Python: usd-core on the Mac; on the GPU host Isaac's own, which
may need the Kit app running first. Call `ensure_pxr()` before `from pxr import ...`."""

_app = None


def ensure_pxr():
    """Returns the SimulationApp it started (or None). Close it with `close()` at the end."""
    global _app
    try:
        import pxr.Usd  # noqa: F401
        return None
    except ImportError:
        from isaacsim import SimulationApp
        _app = SimulationApp({'headless': True})
        return _app


def close(code=0):
    """End the process: through the app if one was started (it exits 0 itself: judge by files)."""
    import os
    import sys
    sys.stdout.flush()
    if _app is not None:
        _app.close()
        os._exit(code)
    sys.exit(code)
