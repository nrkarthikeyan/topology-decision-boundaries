from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "Fast distance matrix for supVR",
    ext_modules = cythonize('supVR.pyx'),  # accepts a glob pattern
)