import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext


class BuildExtConfig(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "mingw32":
            for ext in self.extensions:
                ext.extra_compile_args = ["-O3", "-march=native"]
                ext.extra_link_args = [
                    "-static-libgcc",
                    "-static-libstdc++",
                    "-Wl,-Bstatic,--whole-archive",
                    "-lwinpthread",
                    "-Wl,--no-whole-archive",
                ]
        super(BuildExtConfig, self).build_extensions()


setup(
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},
    package_data={"shtoolkit.shtrans": ["*.pyx", "*.pxd"]},
    ext_modules=cythonize(
        Extension(
            "shtoolkit.shtrans.*",
            ["src/shtoolkit/shtrans/*.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        language_level=3,
    ),
    cmdclass={"build_ext": BuildExtConfig},
)
