cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Test project
PROJECT(Test C CXX)

# ---[ Flags
if(UNIX OR APPLE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpermissive")

SET(MTCNN_INCLUDE_DIRS "${CAFFE_WINDOWS_PATH}/windows/caffe.binding")

INCLUDE_DIRECTORIES(${MTCNN_INCLUDE_DIRS})

SET(TEST_LINK_LIBS ${OpenCV_LIBS})
list(APPEND TEST_LINK_LIBS ${CAFFE_WINDOWS_PATH}/lib/libcaffe-d.so)
list(APPEND TEST_LINK_LIBS "${CAFFE_WINDOWS_PATH}/windows/caffe.binding/lib/caffe_binding-d.so")

LINK_LIBRARIES(${PROJECT_NAME} ${TEST_LINK_LIBS})

ADD_EXECUTABLE(${PROJECT_NAME} test.cpp)

target_link_libraries(${PROJECT_NAME} ${TEST_LINK_LIBS})
