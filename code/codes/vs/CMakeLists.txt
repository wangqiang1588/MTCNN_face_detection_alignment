cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ TestCascadeCNN project
PROJECT(TestCascadeCNN)

include(ExternalProject)

ExternalProject_Add(caffe-windows
	GIT_REPOSITORY https://github.com/wangqiang1588/caffe-windows.git
	GIT_TAG "ms"
	PREFIX "caffe"
	INSTALL_DIR ""
	PATCH_COMMAND ""
	CMAKE_ARGS "-DCUDA_NVCC_FLAGS='-D_FORCE_INLINES'"
	BUILD_IN_SOURCE 1 
	UPDATE_COMMAND ""
	INSTALL_COMMAND "")


SET(CAFFE_WINDOWS_PATH "${PROJECT_SOURCE_DIR}/caffe/src/caffe-windows")

SET(USE_OPENCV ON)
SET(USE_PYTHON ON)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(cmake/Utils.cmake)

# ---[ Dependencies
INCLUDE(cmake/Dependencies.cmake)

#message(STATUS "MTCNN_INCLUDE_DIRS='${MTCNN_INCLUDE_DIRS}'")

message(STATUS "OpenCV_LIBS='${OpenCV_LIBS}'")
message(STATUS "OpenCV_CONFIG_PATH='${OpenCV_CONFIG_PATH}'")


ADD_SUBDIRECTORY(CascadeFaceDetection)

ADD_SUBDIRECTORY(Test)

