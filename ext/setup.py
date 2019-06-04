from setuptools import setup  # , find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("ce_denoise",
        sources=["ce_denoise.pyx", "ce_functions.c"],
        include_dirs=[numpy.get_include()])
    # Extension("ce_WignerVille",
        # sources=["ce_wvd.pyx", "ce_wvd_functions.c"],
        # include_dirs=[numpy.get_include()])
]

setup(
    name='ce_denoise',
    description='C extensions for denoising',
    ext_modules=cythonize(extensions)
)

# installation: python setup.py build_ext -i
