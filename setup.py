from setuptools import setup, find_packages, Extension
import re

from Cython.Build import cythonize
import numpy as np

NAME = "mlcg"
EXCLUDE_FOLDERS = [
    "opt_radius",
]

VERSION = "0.1.2"


packages=find_packages(where="src",exclude=EXCLUDE_FOLDERS)
print(packages)
extensions = [
    Extension(
        "*",
        ["src/mlcg/datasets/cython/*.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

cython_ext=cythonize(
        extensions, compiler_directives={"language_level": "3"}
)


setup(
    name=NAME,
    version=VERSION,
    packages=packages,
    zip_safe=True,
    python_requires=">=3.8",
    license="MIT",
    author="Fe"
    + "\u0301"
    + "lix Musil, Nick Charron, Yoayi Chen, Atharva Kelkar, Clark Templeton",
    #install_requires=install_requires,
    ext_modules=cython_ext,
)

"""
    scripts=[
        "scripts/mlcg-train.py",
        "scripts/mlcg-nvt_langevin.py",
        "scripts/mlcg-nvt_pt_langevin.py",
        "scripts/mlcg-combine_model.py",
        "scripts/mlcg-train_h5.py",
        "scripts/mlcg-train_h5_ng.py",
    ],
"""