#-----------------------------------------------------------------------------
# Hook for spectrum: Tested on Windows 10 x64 for Spectrum 0.7.5

from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all


alldatas, binaries, hiddenimports = collect_all("spectrum")
datas = []
# collect .pyd libraries and other data, just not .py
for d in alldatas:
  if not d[0].endswith(".py"):
    datas.append(d)

datas += copy_metadata("spectrum")

#datas += collect_data_files("spectrum")
