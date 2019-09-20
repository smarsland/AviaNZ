
xcode-select --install

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x Miniconda3-latest-MacOSX-x86_64.sh 

conda install --file condaenv.txt

pip install pyqt5 --user
pip install git+https://github.com/pyqtgraph/pyqtgraph --user

curl avianz.zip 'http://avianz.net/docs/v1.5.1.zip'
unzip avianz.zip
cd avianz

cd ext; python3 setup.py build_ext -i
