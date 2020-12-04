from Cython.Build import cythonize
from distutils.core import setup
import numpy as np

setup(
    name= "conv module", ext_modules= cythonize("convolve.pyx"),
    include_dirs=[np.get_include()]
)