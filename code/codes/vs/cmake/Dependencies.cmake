# These lists are later turned into target properties on main MTCNN library target
set(MTCNN_LINKER_LIBS "")
set(MTCNN_INCLUDE_DIRS "")
set(MTCNN_DEFINITIONS "")
set(MTCNN_COMPILE_OPTIONS "")

# ---[ OpenCV
if(USE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  list(APPEND MTCNN_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
  list(APPEND MTCNN_LINKER_LIBS ${OpenCV_LIBS})
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  list(APPEND MTCNN_DEFINITIONS -DUSE_OPENCV)
endif()

# ---[ Python
if(USE_PYTHON)
  if(NOT "${python_version}" VERSION_LESS "3.0.0")
    # use python3
    find_package(PythonInterp 3.0)
    find_package(PythonLibs 3.0)
    find_package(NumPy 1.7.1)
    # Find the matching boost python implementation
    set(version ${PYTHONLIBS_VERSION_STRING})

    STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
    find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
    set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

    while(NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
      STRING( REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version} )

      STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
      find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
      set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

      STRING( REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version} )
      if("${has_more_version}" STREQUAL "")
        break()
      endif()
    endwhile()
    if(NOT Boost_PYTHON_FOUND)
      find_package(Boost 1.46 COMPONENTS python)
    endif()
  else()
    # disable Python 3 search
    find_package(PythonInterp 2.7)
    find_package(PythonLibs 2.7)
    find_package(NumPy 1.7.1)
    find_package(Boost 1.46 COMPONENTS python)
  endif()
  if(PYTHONLIBS_FOUND AND NUMPY_FOUND AND Boost_PYTHON_FOUND)
    set(HAVE_PYTHON TRUE)
    list(APPEND MTCNN_DEFINITIONS -DWITH_PYTHON_LAYER)
    list(APPEND MTCNN_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
    list(APPEND MTCNN_LINKER_LIBS ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
  endif()
endif()

