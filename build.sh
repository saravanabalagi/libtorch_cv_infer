sh setup.sh

[ $1="--clean" ] && rm -rf build
[ ! -d "build" ] && mkdir -p build

cd build
cmake ..
make -j6
cd ..
