from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy

from Cython.Build import cythonize

extensions = [
	Extension(
		"ce_denoise",
		["ce_denoise.pyx", "ce_functions.c"],
		include_dirs=[numpy.get_include()]
	)
]

setup(
	name = 'ce_denoise',

	description = 'C extensions for denoising',

	packages = find_packages(),
	ext_modules = cythonize(extensions)
)
