# Reduced to the Cython extension build by Claude Opus 5.5 (claude-opus-5-5)
# while merging feat/noise_up into main; all package metadata lives in
# pyproject.toml.
from setuptools import setup, Extension

from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "mlcg.datasets.cython.*",
        ["src/mlcg/datasets/cython/*.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    ext_modules=cythonize(
        extensions, compiler_directives={"language_level": "3"}
    ),
)
