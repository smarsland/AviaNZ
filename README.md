<div align="center">

![Logo](img/AviaNZ_SW_V2.jpg)
### Welcome to AviaNZ, an open-source project for manual and automatic analysis of bio-acoustic recordings.

</div>

---

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

```python
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
