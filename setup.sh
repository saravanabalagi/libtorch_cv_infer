if [ -d "libs" ]; then 
    echo "libs already setup, skipping..."
    exit 0
fi

mkdir libs
cd libs

# download libtorch
wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu111.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.9.0+cu111.zip
rm libtorch-cxx11-abi-shared-with-deps-1.9.0+cu111.zip

# download and build opencv
wget -O opencv.tar.gz https://github.com/opencv/opencv/archive/4.5.3.tar.gz
tar -xvf opencv.tar.gz
rm opencv.tar.gz
mv opencv-4.5.3 opencv

wget -O opencv_contrib.tar.gz https://github.com/opencv/opencv_contrib/archive/4.5.3.tar.gz
tar -xvf opencv_contrib.tar.gz
rm opencv_contrib.tar.gz
mv opencv_contrib-4.5.3 opencv/opencv_contrib

cd opencv
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ..
make -j10
cd ../..
