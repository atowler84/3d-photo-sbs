"""sbs3d - turn a single photo into a high-resolution side-by-side 3D image."""

__version__ = "1.0.0"

__all__ = ["Settings", "Converter", "convert"]


def __getattr__(name):
    # Lazy so that `import sbs3d` stays cheap (torch takes a couple of seconds).
    if name in __all__:
        from importlib import import_module

        return getattr(import_module(".pipeline", __name__), name)
    raise AttributeError(name)
