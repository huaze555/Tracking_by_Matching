cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(Tracking-by-Matching LANGUAGES CXX)

add_definitions(-fpermissive -fPIC -w)   # 预编译选项

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wunused-function -Wunused-variable -Wfatal-errors")
#set(CMAKE_BUILD_TYPE RELEASE)

set(TensorRT_DIR "/home/zxj/3rdParty/TensorRT-6.0.1.5")

find_package(CUDA REQUIRED)
find_package(OpenCV 4 REQUIRED)

FILE (GLOB DIR_SRCS "${PROJECT_SOURCE_DIR}/trt_lib/src/*.cpp" "${PROJECT_SOURCE_DIR}/trt_lib/cu/*.cu")

include_directories(${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/trt_lib/include ${TensorRT_DIR}/include
        ${PROJECT_SOURCE_DIR}/include)    # -I
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${TensorRT_DIR}/lib)     #  -L

cuda_add_executable(demo main.cpp TrackerByMatcher.cpp ${DIR_SRCS})
target_link_libraries(demo ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} cublas cudnn nvinfer nvinfer_plugin stdc++fs)