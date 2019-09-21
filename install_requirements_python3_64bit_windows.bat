:: Assumes python3 with pip is installed
:: the Torch library is quite large (ca. 700 MB)

pip3 install --upgrade pip
pip3 install torch -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torchvision numpy scipy matplotlib