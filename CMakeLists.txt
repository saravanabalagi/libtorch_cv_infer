cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(embedding-inference)

set(CMAKE_PREFIX_PATH "./libs/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

set(CMAKE_PREFIX_PATH "./libs/opencv/build")
find_package(OpenCV REQUIRED)

add_executable(embedding-inference main.cpp)

target_link_libraries(embedding-inference "${TORCH_LIBRARIES}")
target_link_libraries(embedding-inference "${OpenCV_LIBS}")

set_property(TARGET embedding-inference PROPERTY CXX_STANDARD 17)
set_property(TARGET embedding-inference PROPERTY OUTPUT_NAME predict)
