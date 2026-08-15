"""StereoCraft - turn a single photo into a high-resolution side-by-side 3D image."""

__version__ = "1.0.0"

__all__ = ["Settings", "Converter", "convert"]

_HOMES = {name: ".pipeline" for name in __all__}


def __getattr__(name):
    # Lazy so that `import stereocraft` stays cheap (torch takes a couple of seconds).
    if name in _HOMES:
        from importlib import import_module

        return getattr(import_module(_HOMES[name], __name__), name)
    raise AttributeError(name)
