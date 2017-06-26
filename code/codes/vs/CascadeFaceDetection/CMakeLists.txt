cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ CascadeFaceDetection project
PROJECT(CascadeFaceDetection C CXX)

list(APPEND MTCNN_INCLUDE_DIRS "${CAFFE_WINDOWS_PATH}/windows/caffe.binding")

INCLUDE_DIRECTORIES(${MTCNN_INCLUDE_DIRS})

# ---[ Flags
if(UNIX OR APPLE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpermissive")

SET(SOURCE_FILES "")
list(APPEND SOURCE_FILES CascadeFaceDetection.cpp)
list(APPEND SOURCE_FILES python_module.cpp)

if("${OpenCV_VERSION}" VERSION_LESS "3.0.0")
  list(APPEND SOURCE_FILES pyboost_cv2_converter.cpp)
else()
  list(APPEND SOURCE_FILES pyboost_cv3_converter.cpp)
endif()

SET(CASCADE_LINK_LIBS,${OpenCV_LIBS})
list(APPEND CASCADE_LINK_LIBS "${CAFFE_WINDOWS_PATH}/lib/libcaffe.so")
list(APPEND CASCADE_LINK_LIBS "${CAFFE_WINDOWS_PATH}/windows/caffe.binding/lib/caffe_binding.so")

LINK_LIBRARIES(${PROJECT_NAME} ${CASCADE_LINK_LIBS})

ADD_LIBRARY(${PROJECT_NAME} ${SOURCE_FILES})


