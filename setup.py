from cx_Freeze import setup, Executable
import scipy, os, sys
import numpy.core._methods
import numpy.lib.format
import spectrum
import resampy
import numba
import llvmlite

includefiles_list=[]
spectrum_path = os.path.dirname(spectrum.__file__)
resampy_path = os.path.dirname(resampy.__file__)
numba_path = os.path.dirname(numba.__file__)
llvmlite_path = os.path.dirname(llvmlite.__file__)

includefiles_list.append(spectrum_path)
includefiles_list.append(resampy_path)
includefiles_list.append(numba_path)
includefiles_list.append(llvmlite_path)
includefiles_list.append('dmey.txt')
includefiles_list.append('AviaNZconfig.txt')
includefiles_list.append('sppInfo.txt')
#includefiles_list.append('AviaNZconfig_user.txt')
# Dependencies are automatically detected, but it might need
# fine tuning.
buildOptions = dict(
    packages = ['numpy.core._methods','numpy.linalg._umath_linalg','numpy.lib.format','AviaNZ_batch','colourMaps','Dialogs','Segment','SignalProc','SupportClasses','WaveletSegment'	,'WaveletFunctions','wavio'],
    excludes = ['tkinter'],
    includes = [] ,
    include_files = includefiles_list
) 
#import sys
#base = 'Win32GUI' if sys.platform=='win32' else None
base = 'Console'
 
executables = [
    Executable('AviaNZ.py', base=base, icon="img/Avianz.ico") 
]
 
setup(
    name='AviaNZ',
	author = "Stephen Marsland, Victoria University of Wellington (2018) with code by Nirosha Priyadarshani and Julius Juodakis",
	author_email=" stephen.marsland@vuw.ac.nz,  nirosha.priyadarshani@vuw.ac.nz",
    version = '1.1',
    description = 'AviaNZ Birdsong Analysis Software',
    options = dict(build_exe = buildOptions),
    executables = executables
)