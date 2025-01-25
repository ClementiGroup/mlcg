from setuptools import setup, Extension
from torch.utils import cpp_extension


VERSION = "0.0.1"

setup(
    name="mlcg_opt_radius",
    version=VERSION,
    ext_modules=[
        cpp_extension.CUDAExtension(
            "mlcg_opt_radius",
            ["cu/radius_sd.cu", "cu/exclusion_pairs.cpp"],
            # extra_compile_args={"cxx": ["-O0", "-g"]},
            # extra_link_args=["-O0", "-g"],
            py_limited_api=True,
        ),
        ],
    python_requires=">=3.9",
    license="MIT",
    author="Paul Mifsud, Yaoyi Chen",
    install_requires=["torch", "mlcg"],
    cmdclass={'build_ext': cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
