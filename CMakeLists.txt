cmake_minimum_required(VERSION 3.12)
project(dqn)

set(CMAKE_CXX_STANDARD 14)

#caffe
include_directories(/work/caffe-master/build/install/include)
link_directories(/work/caffe-master/build/install/lib)

#cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3")

#add_definitions(-DCPU_ONLY)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


include_directories(/usr/local/opt/lapack/include)
link_directories(/usr/local/opt/lapack/lib)

include_directories(/usr/local/opt/openblas/include)
link_directories(/usr/local/opt/openblas/lib)

# Boost
set(Boost_USE_STATIC_LIBS ON)  # only find static libs
set(Boost_USE_DEBUG_LIBS OFF) # ignore debug libs and
set(Boost_USE_RELEASE_LIBS ON)  # only find release libs
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS system program_options thread timer chrono REQUIRED)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif ()

message(STATUS ${Boost_LIBRARIES})




#main program
message(STATUS "${CMAKE_BINARY_DIR}")
message(STATUS "${CMAKE_SOURCE_DIR}")

file(GLOB PROTOTXT_FILES "${CMAKE_SOURCE_DIR}/*.prototxt")
message(STATUS ${PROTOTXT_FILES})
file(COPY ${PROTOTXT_FILES} DESTINATION ${CMAKE_BINARY_DIR})

add_executable(dqn main.cpp PlayEngine.cpp dqn.cpp)

target_link_libraries(dqn caffe caffeproto protobuf caffe glog gflags ${Boost_LIBRARIES} ${OpenCV_LIBS})


