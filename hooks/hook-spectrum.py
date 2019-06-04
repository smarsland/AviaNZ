#-----------------------------------------------------------------------------
# Hook for spectrum: Tested on Windows 10 x64 for Spectrum 0.7.5

from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_data_files

datas = copy_metadata("spectrum")
datas += collect_data_files("spectrum")
