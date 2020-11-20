sh setup.sh
rm -rf build

mkdir -p build
cd build
cmake ..
make -j6
cd ..

mv build/predict .

rm -rf build
