project(tflite)
add_library(tflite INTERFACE)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/tensorflow/lite/tools/make/downloads)

target_include_directories(tflite INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${THIRD_PARTY_DIR}/flatbuffers/include
        )


file(GLOB TFLITE_LIBRARY_FILES "${CMAKE_CURRENT_SOURCE_DIR}/lib/*.dylib")

target_link_libraries(tflite INTERFACE ${TFLITE_LIBRARY_FILES})