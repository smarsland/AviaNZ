from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
	cmdclass={'build_ext': build_ext},
	name = 'ce_denoise',
	description = 'C extensions for denoising',
	ext_modules=[
        Extension("ce_denoise",
		sources=["ce_denoise.pyx", "ce_functions.c"],
		include_dirs=[numpy.get_include()])
        #Extension("ce_WignerVille",
		#sources=["ce_wvd.pyx", "ce_wvd_functions.c"],
		#include_dirs=[numpy.get_include()])

]
)

# installation: python setup.py build_ext -i
