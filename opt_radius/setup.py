from setuptools import find_packages, setup, Extension
from torch.utils import cpp_extension


VERSION = "0.0.2"

setup(
    name="mlcg_opt_radius",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[
        cpp_extension.CUDAExtension(
            "mlcg_opt_radius.radius_opt",
            [
                f"mlcg_opt_radius/{src}"
                for src in ["binding.cpp", "radius.cu", "exclusion_pairs.cpp"]
            ],
            py_limited_api=True,
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
            optional=True,
        ),
    ],
    python_requires=">=3.9",
    license="MIT",
    author="Paul Mifsud, Yaoyi Chen",
    install_requires=["torch"],
    cmdclass={
        "build_ext": cpp_extension.BuildExtension.with_options(
            no_python_abi_suffix=True, use_ninja=False
        )
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
