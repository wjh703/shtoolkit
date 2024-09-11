from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "shtoolkit.shtrans.*",
        ["src/shtoolkit/shtrans/*.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]
setup(ext_modules=cythonize(extensions, language_level=3))
