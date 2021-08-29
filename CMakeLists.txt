cmake_minimum_required(VERSION 3.5)
project(cutemodel)

set(CMAKE_CXX_STANDARD 11)
set(CUTEMODEL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(tflite)
include_directories(cutemodel STATIC ${CUTEMODEL_INCLUDE_DIR})

option(CUTEMODEL_RUN_EXAMPLE "Run CuteModel Example" OFF)

if (CUTEMODEL_RUN_EXAMPLE)
  add_executable(cutemodel example.cc
          ${CUTEMODEL_INCLUDE_DIR}/cutemodel/cute_model.cc)
else()
  add_library(cutemodel STATIC
          ${CUTEMODEL_INCLUDE_DIR}/cutemodel/cute_model.cc)
endif()

target_link_libraries(cutemodel PUBLIC tflite)
