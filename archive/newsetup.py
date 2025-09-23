from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
buildOptions = dict(packages = [], excludes = [])

import sys
base = 'Win32GUI' if sys.platform=='win32' else None

executables = [
    Executable('AviaNZ.py', base=base)
]

setup(name='AviaNZ',
      version = '1.1',
      description = 'Birdsong Recognition',
      options = dict(build_exe = buildOptions),
      executables = executables)
