
xcode-select --install

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x Miniconda3-latest-MacOSX-x86_64.sh 

curl -OL https://github.com/smarsland/AviaNZ/archive/v1.5.1.zip
unzip v1.5.1.zip
cd AviaNZ-1.5.1

conda install --file condaenv.txt

pip install pyqt5 --user
pip install git+https://github.com/pyqtgraph/pyqtgraph --user

cd ext; python3 setup.py build_ext -i
cd ..


