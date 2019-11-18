
xcode-select --install

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x Miniconda3-latest-MacOSX-x86_64.sh 
./Miniconda3-latest-MacOSX-x86_64.sh

curl -OL https://github.com/smarsland/AviaNZ/archive/v2.0.zip
unzip v2.0.zip
cd AviaNZ-2.0

pip install -r requirements.txt --user

cd ext; python3 setup.py build_ext -i
cd ..


