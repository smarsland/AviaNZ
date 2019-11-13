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
# packages in environment at C:\Users\Julius\.conda\envs\v2:
#
# Name                    Version                   Build  Channel
# altgraph                  0.16.1                   pypi_0    pypi
# attrs                     19.3.0                   pypi_0    pypi
# blas                      1.1                    openblas    conda-forge
# ca-certificates           2019.9.11            hecc5488_0    conda-forge
# certifi                   2019.6.16                py37_1    conda-forge
# future                    0.18.2                   pypi_0    pypi
# icu                       58.2                 ha66f8fd_1
# importlib-metadata        0.23                     pypi_0    pypi
# jpeg                      9b                   hb83a4c4_2
# jsonschema                3.1.1                    pypi_0    pypi
# libblas                   3.8.0           7_h8933c1f_netlib    conda-forge
# libcblas                  3.8.0           7_h8933c1f_netlib    conda-forge
# libflang                  5.0.0           h6538335_20180525    conda-forge
# liblapack                 3.8.0           7_h8933c1f_netlib    conda-forge
# liblapacke                3.8.0           7_h8933c1f_netlib    conda-forge
# libpng                    1.6.37               h2a8f88b_0
# llvm-meta                 5.0.0                         0    conda-forge
# m2w64-gcc-libgfortran     5.3.0                         6
# m2w64-gcc-libs            5.3.0                         7
# m2w64-gcc-libs-core       5.3.0                         7
# m2w64-gmp                 6.1.0                         2
# m2w64-libwinpthread-git   5.0.0.4634.697f757               2
# more-itertools            7.2.0                    pypi_0    pypi
# msys2-conda-epoch         20160418                      1
# numpy                   	1.17.3			py37hc71023c_0     conda-forge
# openblas                  0.3.7             h535eed3_1001    conda-forge
# openmp                    5.0.0                    vc14_1    conda-forge
# openssl                   1.1.1c               hfa6e2cd_0    conda-forge
# pefile                    2019.4.18                pypi_0    pypi
# pip                       19.3.1                   py37_0    conda-forge
# pyinstaller               3.5                      pypi_0    pypi
# pyqt                      5.9.2            py37h6538335_4    conda-forge
# pyqt5                     5.12                     pypi_0    pypi
# pyqt5-sip                 4.19.19                  pypi_0    pypi
# pyqtgraph                 0.11.0.dev0+gf2740f7          pypi_0    pypi
# pyrsistent                0.15.5                   pypi_0    pypi
# python                    3.7.3                h510b542_1    conda-forge
# pywin32-ctypes            0.2.0                    pypi_0    pypi
# qt                        5.9.7            vc14h73c81de_0
# setuptools                41.6.0                   py37_0    conda-forge
# sip                       4.19.8          py37h6538335_1000    conda-forge
# sqlite                    3.30.1               he774522_0
# vc                        14.1                 h0510ff6_4
# vs2015_runtime            14.16.27012          hf0eaf9b_0
# wheel                     0.33.6                   py37_0    conda-forge
# wincertstore              0.2                   py37_1002    conda-forge
# zipp                      0.6.0                    pypi_0    pypi
# zlib                      1.2.11               h62dcd97_3

block_cipher = None

a = Analysis(['AviaNZ.py'],
             pathex=['c:\\Program Files (x86)\\Windows Kits\\10\\Redist\\ucrt\\DLLs', 'c:\\Users\\Julius\\Documents\\DARBO\\AviaNZ\\AviaNZ-master'],
             binaries=[],
             datas=[('Config/*', 'Config'), ('Filters/*txt', 'Filters'), ('Wavelets/*txt', 'Wavelets'), ('img/*', 'img'), ('Sound Files/*', 'Sound Files')],
             hiddenimports=['cython', 'sklearn', 'sklearn.ensemble', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils', 'sklearn.utils._cython_blas', 'pywt._extensions._cwt'],
             hookspath=['.\\hooks'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

	 
a_s = Analysis(['SplitAnnotations.py'],
             pathex=['c:\\Program Files (x86)\\Windows Kits\\10\\Redist\\ucrt\\DLLs', 'c:\\Users\\Julius\\Documents\\DARBO\\AviaNZ\\AviaNZ-master'],
             binaries=[],
             hiddenimports=['cython'],
             hookspath=['.\\hooks'],
             datas=[('img/*', 'img')],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

MERGE((a, 'AviaNZ', 'AviaNZ'),
		(a_s, 'SplitAnnotations', 'AviaNZ-Splitter'))

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
			 
pyz_s = PYZ(a_s.pure, a_s.zipped_data,
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

exe_s = EXE(pyz_s,
          a_s.scripts,
          [],
          exclude_binaries=True,
          name='AviaNZ-Splitter',
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
               name='AviaNZv2.0')
coll_s = COLLECT(exe_s,
               a_s.binaries,
               a_s.zipfiles,
               a_s.datas,
               strip=False,
               upx=True,
               name='AviaNZv2.0s')
# Then need to move the other exe from v2.0s/ to the first dir