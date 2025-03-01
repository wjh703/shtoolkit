import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

extra_link_args_mingw32 = [
    "-static-libgcc",
    "-static-libstdc++",
    "-Wl,-Bstatic,--whole-archive",
    "-lwinpthread",
    "-Wl,--no-whole-archive",
]


class BuildExtConfig(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "mingw32":
            for e in self.extensions:
                e.extra_compile_args = ["-O3"]
                e.extra_link_args = extra_link_args_mingw32
        super(BuildExtConfig, self).build_extensions()


extensions = [
    Extension(
        "shtoolkit.shtrans.*",
        ["src/shtoolkit/shtrans/*.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]
setup(ext_modules=cythonize(extensions, language_level=3), cmdclass={"build_ext": BuildExtConfig})
