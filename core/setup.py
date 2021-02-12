from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension(
        "cython_core",
        ["cython_core.pyx"],
        extra_compile_args=['/openmp'],
        language = 'c++17',
        extra_link_args=['-openmp']
    )
]

setup(
    name='cython_core',
    ext_modules=cythonize(ext_modules),
)