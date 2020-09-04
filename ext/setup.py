from setuptools import setup  # , find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("ce_denoise",
        sources=["ce_denoise.pyx", "ce_functions.c"],
        include_dirs=[numpy.get_include()]),
    Extension("SplitLauncher",
        sources=["SplitLauncher.pyx", "SplitWav.c"])#,
    # Extension("ce_detect",
    #     sources=["ce_detect.pyx", "detector.c"])
]

setup(
    name='ce_denoise',
    description='C extensions for denoising',
    ext_modules=cythonize(extensions)
)

# installation: python setup.py build_ext -i
