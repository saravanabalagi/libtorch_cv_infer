cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch-cv-inference)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0074 NEW)
endif(COMMAND cmake_policy)

set(Torch_ROOT "libs/libtorch")
find_package(Torch REQUIRED PATHS ${Torch_ROOT} NO_DEFAULT_PATH)

set(OpenCV_DIR "libs/opencv/build")
find_package(OpenCV REQUIRED imgcodecs PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)

add_executable(predict predict.cpp)
target_link_libraries(predict ${TORCH_LIBRARIES} ${OpenCV_LIBS} stdc++fs)
set_property(TARGET predict PROPERTY CXX_STANDARD 17)

add_executable(predict_dir predict_dir.cpp)
target_link_libraries(predict_dir ${TORCH_LIBRARIES} ${OpenCV_LIBS} stdc++fs)
set_property(TARGET predict_dir PROPERTY CXX_STANDARD 17)
