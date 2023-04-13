Welcome to AviaNZ, an open-source project for manual and automatic analysis of bio-acoustic recordings.

This software enables you to: 
* review and listen to wav files from acoustic field recorders, 
* segment and annotate the recordings, 
* train filters to recognise calls from particular species, 
* use filters that others have provided to batch process many files
* review annotations
* produce output in spreadsheet form, or as files ready for further statistical analyses

For more information about the project, see http://www.avianz.net

# Citation

If you use this software, please credit us in any papers that you write. An appropriate reference is:

```
@article{Marsland19,
  title = "AviaNZ: A future-proofed program for annotation and recognition of animal sounds in long-time field recordings",
  author = "{Marsland}, Stephen and {Priyadarshani}, Nirosha and {Juodakis}, Julius and {Castro}, Isabel",
  journal = "Methods in Ecology and Evolution",
  volume = 10,
  number = 8,
  pages = "1189--1195",
  year = 2019
}
```


# Installation

## Windows
Windows binaries are available at http://www.avianz.net.
To install from source, follow the Linux instructions.

## macOS
An installer script is available at http://www.avianz.net.
To install from source, follow the Linux instructions.

## Linux
No binaries are available. Install from the source as follows:
1. Download the source .zip of the latest release.
2. Extract (`unzip v2.0.zip`) and navigate to the extracted directory.
3. Ensure Python (3.6 or higher), pip and git are available on your system. On Ubuntu, these can be installed by running:  
```
sudo apt-get install python3.6
sudo apt-get install python3-pip
sudo apt-get install git
```
4. Install the required packages by running `pip3 install -r requirements.txt --user` at the command line. (On Ubuntu and some other systems, `python` and `pip` refer to the Python 2 versions. If you are sure these refer to version 3 of the language, use `python` and `pip` in steps 4-6.)  
5. Build the Cython extensions by running `cd ext; python3 setup.py build_ext -i; cd..`  
6. Done! Launch the software with `python3 AviaNZ.py`  

# Acknowledgements

AviaNZ is based on PyQtGraph and PyQt, and uses Librosa and Scikit-learn amongst others.

Development of this software was supported by the RSNZ Marsden Fund, and the NZ Department of Conservation.


# Addendum - installation on a Mac M1/M2 ARM plaform (current MacBook Pro, Mac Mini etc...) with native support for tensorflow-metal

The package can be installed using Apple native tensorflow support on a Mac M1/M2 ARM platform. This gives a significant performance improvement. 

For Apple M1/M2, the most stable version as at April 23 seems to be Python version 3.9

Use MiniForge3 -> https://github.com/conda-forge/miniforge 

1. set up and activate a conda environment: 
```
    conda create --name aviaNZ_ARM python=3.9.13
    conda activate aviaNZ_ARM
```
2. Install python package dependencies listed in requirements.txt using `conda install <package name>`. I suggest doing individually to check for issues - all should install with the exception of spectrum and PyQt5. Make sure that arm64 platform versions are installed.

3. Install spectrum with `python -m pip install spectrum` 

4. PyQt5 can be problematic as it is not in the conda-forge repo.

To get it working:

- Install brew using instructions as https://brew.sh/
- Make sure your system python is 3.9 (or use pyenv to set)
- `brew install pyqt@5`
- Create a symlink called PyQt5 in the conda site-packages: 
       `ln -s /opt/homebrew/lib/python3.9/site-packages/PyQt5 ~/miniforge3/envs/avianz_ARM/lib/python3.9/site-packages/PyQt5`
(Note this does create a fragile dependency, but seems to be the only place with an arm64 native version available - may change in future)

5. Install apple native tensorflow:

 follow instructions at: https://developer.apple.com/metal/tensorflow-plugin/ (ignore the earliest steps setting up condaforge - as we are using miniforge instead)

```
       conda install -s apple tensorflow-deps
       python -m pip install tensorflow-macos
       python -m pip install tensorflow-metal
```  

 Note: if this fails, then you may need to manually install the older, more stable versions: 
 follow instructions at https://developer.apple.com/forums/thread/706920
```
pip install https://files.pythonhosted.org/packages/4d/74/47440202d9a26c442b19fb8a15ec36d443f25e5ef9cf7bfdeee444981513/tensorflow_macos-2.8.0-cp39-cp39-macosx_11_0_arm64.whl
pip install https://files.pythonhosted.org/packages/d5/37/c48486778e4756b564ef844b145b16f3e0627a53b23500870d260c3a49f3/tensorflow_metal-0.4.0-cp39-cp39-macosx_11_0_arm64.whl
```
After all that, it should work OK. Note that you will need the code changes from this repo as the PyQt5 version changed some object dependencies.

To run:

'''
cd path/to/local/repo/
conda activate aviaNZ_ARM
python aviaNZ.py
'''
 