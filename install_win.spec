# -*- mode: python -*-

## --------------------------------- ##
# Spec file for installing via PyInstaller.
# Tested on 3.5dev0+d74052489 version.
# USAGE:
#  cd ext; python setup.py build_ext -i
#  pyinstaller install_win.spec

# Sufficient environment:
#  VC++ compiler 16.0
#  Visual Studio 2017
#  Microsoft Build Tools 2015
#  Windows SDK 7.1
#  .NET framework 4.7

# Sufficient conda environment:
# NOTE: this setup uses openBLAS instead of MKL
# Name                    Version                   Build  Channel
# altgraph                  0.16.1                     py_0    conda-forge
# blas                      1.1                    openblas    conda-forge
# ca-certificates           2019.1.23                     0
# certifi                   2019.3.9                 py37_0
# future                    0.17.1                py37_1000    conda-forge
# icu                       58.2                 ha66f8fd_1
# jpeg                      9b                   hb83a4c4_2
# libflang                  5.0.0           h6538335_20180525    conda-forge
# libpng                    1.6.37               h2a8f88b_0
# llvm-meta                 5.0.0                         0    conda-forge
# macholib                  1.11                       py_0    conda-forge
# numpy                     1.16.2          py37_blas_openblash442142e_0  [blas_openblas]  conda-forge
# openblas                  0.3.3             h535eed3_1001    conda-forge
# openmp                    5.0.0                    vc14_1    conda-forge
# openssl                   1.1.1c               he774522_1
# pefile                    2019.4.18                  py_0    conda-forge
# pip                       19.1.1                   py37_0    conda-forge
# pycrypto                  2.6.1           py37hfa6e2cd_1002    conda-forge
# pyinstaller               3.5.dev0+d74052489          pypi_0    pypi
# pyqt                      5.9.2            py37h6538335_2
# pyqt5                     5.12                     pypi_0    pypi
# pyqt5-sip                 4.19.17                  pypi_0    pypi
# pyqtgraph                 0.11.0.dev0+geb90616          pypi_0    pypi
# python                    3.7.3                hb12ca83_0    conda-forge
# pywin32                   224             py37hfa6e2cd_1000    conda-forge
# pywin32-ctypes            0.2.0                 py37_1000    conda-forge
# qt                        5.9.7            vc14h73c81de_0
# setuptools                41.0.1                   py37_0    conda-forge
# sip                       4.19.8           py37h6538335_0
# sqlite                    3.28.0               hfa6e2cd_0    conda-forge
# vc                        14.1                 h0510ff6_4
# vs2015_runtime            14.15.26706          h3a45250_4
# wheel                     0.33.4                   py37_0    conda-forge
# wincertstore              0.2                   py37_1002    conda-forge
# zlib                      1.2.11            h2fa13f4_1004    conda-forge

block_cipher = None

a = Analysis(['AviaNZ.py'],
             pathex=['c:\\Program Files (x86)\\Windows Kits\\10\\Redist\\ucrt\\DLLs', 'c:\\Users\\Julius\\Documents\\DARBO\\AviaNZ\\AviaNZ-master'],
             binaries=[],
             datas=[('Config/*txt', 'Config'), ('Filters/*txt', 'Filters'), ('Wavelets/*txt', 'Wavelets'), ('img/*', 'img')],
             hiddenimports=['cython', 'sklearn', 'sklearn.ensemble', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils', 'sklearn.utils._cython_blas', 'pywt._extensions._cwt'],
             hookspath=['.\\hooks'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

# note: this may have a bug requiring absolute path to ICON
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='AviaNZ',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=True,
          icon='c:\\Users\\Julius\\Documents\\DARBO\\AviaNZ\\AviaNZ-master\\img\\Avianz.ico')

# rename version below:
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='AviaNZv1.5.1')
