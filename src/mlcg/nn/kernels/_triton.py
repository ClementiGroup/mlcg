"""
Triton for the kernel modules, or a stand-in where it is not installed (a
CPU-only torch): the kernel modules then still import, and their ops run
their CPU versions.
"""

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:

    class _MissingTriton:
        """Any attribute or call gives the stand-in back, and as a decorator
        it returns the function unchanged."""

        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return self

    triton = tl = libdevice = _MissingTriton()
