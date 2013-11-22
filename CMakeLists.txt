##########################################
# cube CMake

cmake_minimum_required (VERSION 2.8)
project (cube C CXX Fortran)

### include local modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

### version
set(cube_VERSION_MAJOR 0)
set(cube_VERSION_MINOR 8)

# also get the current git revision
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_REV)


#if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#  set(CMAKE_C_FLAGS "-std=gnu99")
#endif()

### compiler flags
set(CMAKE_C_FLAGS "-std=c11 -D_XOPEN_SOURCE=700")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pedantic")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0 -ggdb3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O2")


#########################################
# HDF-5
# FIND_PACKAGE (HDF5 NAMES hdf5)
find_package (HDF5 REQUIRED COMPONENTS C HL)
include_directories (${HDF5_INCLUDE_DIR})
set(LINK_LIBS ${LINK_LIBS} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

#########################################
# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIR})
set(LINK_LIBS ${LINK_LIBS} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

#the compute architectures we support

set(CUDA_ARCH              -gencode arch=compute_20,code=sm_20)
set(CUDA_CCARCH 2.0)

if(NOT (${CUDA_VERSION} LESS 5.0))
  set(CUDA_ARCH ${CUDA_ARCH} -gencode arch=compute_30,code=sm_30)
  set(CUDA_CCARCH ${CUDA_CCARCH} 3.0)
endif()
if(NOT (${CUDA_VERSION} LESS 5.5))
  set(CUDA_ARCH ${CUDA_ARCH} -gencode arch=compute_35,code=\"sm_35,compute_35\")
  set(CUDA_CCARCH ${CUDA_CCARCH} 3.5)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};${CUDA_ARCH})

if(APPLE)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ccbin /usr/bin/clang)
endif()

########################################
# OpenGL
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
set(LINK_LIBS ${LINK_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

#########################################
# OpenSSL
find_package(OpenSSL REQUIRED)
set(LINK_LIBS ${LINK_LIBS} ${OPENSSL_LIBRARIES})

#########################################
# BLAS Libraries
find_package(BLAS REQUIRED)
include_directories(${BLAS_INCLUDE_DIR})
set(LINK_LIBS ${LINK_LIBS} ${BLAS_LIBRARIES})

find_package(LAPACK REQUIRED)
include_directories(${LAPACK_INCLUDE_DIR})
set(LINK_LIBS ${LINK_LIBS} ${LAPACK_LIBRARIES})

if(NOT APPLE) #FIXME: OR NOT MKL OR NOT ACML
  find_library(LAPACKE_LIBRARY lapacke)
  set(LINK_LIBS ${LINK_LIBS} ${LAPACKE_LIBRARY})
endif()

########################################
# JSON Libraries
if(NOT APPLE)
  find_library(JSON_LIBRARIES "json")
else()
  set(JSON_LIBRARIES "json-c")
endif()

########################################
########################################
# The CUBE library
include_directories(lib)
include_directories(lib/kernel)
file(GLOB cube_SOURCES lib/*.c)
file(GLOB cube_INCLUDES lib/*.h)
file(GLOB cube_CUFILES lib/kernel/*.cu)

configure_file("${PROJECT_SOURCE_DIR}/lib/version.c.in"
               "${PROJECT_BINARY_DIR}/version.c")

cuda_add_library(cube SHARED
                 "${PROJECT_BINARY_DIR}/version.c"
                 ${cube_SOURCES}
		 ${cube_INCLUDES}
		 ${cube_CUFILES})

#add_library(cube SHARED ${cube_INCLUDES} ${cube_SOURCES})
target_link_libraries(cube ${LINK_LIBS})

set_target_properties(cube PROPERTIES
  FRAMEWORK TRUE
  FRAMEWORK_VERSION 1.0
  PUBLIC_HEADER "${cube_INCLUDES}")





#######################################
# The ICA binary

include_directories(bin)
set(ica_SOURCES bin/ica.c
		bin/sca.c
                bin/monitor_sca.c
                bin/monitor_oglui.c
                bin/monitor_stdout.c
		bin/cube_math_f.f90
                )

add_executable(ica ${ica_SOURCES})
target_link_libraries(ica cube ${JSON_LIBRARIES})

############
# playground

if(EXISTS "${CMAKE_SOURCE_DIR}/main.cu")
  cuda_add_executable(playground EXCLUDE_FROM_ALL main.cu)
  target_link_libraries(playground cube)
endif()

########################################
# Tests

enable_testing()
include_directories(tests)

add_executable(cube_test tests/test_main.c)
target_link_libraries(cube_test cube)

add_test("CubeMain" cube_test)

include(CTest)

########################################
# Misc

# etags
if(UNIX)
  add_custom_target(tags etags --members --declarations `find ${CMAKE_CURRENT_SOURCE_DIR} -name "*.h" -o -name "*.c" -o -name ".cu" ` )
endif()

########################################

MESSAGE(STATUS "READY. ")
MESSAGE(STATUS "===============================")
MESSAGE(STATUS "CUDA:    ${CUDA_VERSION}")
MESSAGE(STATUS "`-CA:    ${CUDA_CCARCH}")
MESSAGE(STATUS "HDF5:    ${HDF5_LIBRARIES}")
MESSAGE(STATUS "BLAS:    ${BLAS_LIBRARIES}")
MESSAGE(STATUS "JSON:    ${JSON_LIBRARIES}")
MESSAGE(STATUS "===============================")
