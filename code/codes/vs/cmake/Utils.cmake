################################################################################################
# Clears variables from list
# Usage:
#   mtcnn_clear_vars(<variables_list>)
macro(mtcnn_clear_vars)
  foreach(_var ${ARGN})
    unset(${_var})
  endforeach()
endmacro()
