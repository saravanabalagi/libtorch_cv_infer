if [ -d "libs" ]; then 
    echo "libs already setup, skipping..."
    exit 0
fi

mkdir libs
cd libs

# download libtorch
wget https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.7.0%2Bcu101.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.7.0+cu101.zip
rm libtorch-cxx11-abi-shared-with-deps-1.7.0+cu101.zip

# download and build opencv
wget https://github.com/opencv/opencv/archive/4.5.0.tar.gz
tar -xvf 4.5.0.tar.gz
rm 4.5.0.tar.gz
mv opencv-4.5.0 opencv
cd opencv
mkdir -p build && cd build
cmake  ..
cmake --build .
cd ../..
