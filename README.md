# Tensorflow Lite C++ API
 * Prebuilt Guide
 * Building from source


## Prebuilt Guide

Select branch according to your platform.
CuteModel usage: https://github.com/visualcamp/Tensorflow-Lite/tree/cutemodel


## Building from source

0. Get tensorflow 
    ```
    > git clone https://github.com/tensorflow/tensorflow
   ```

1. Go to tensorflow directory
    ```
    > cd tensorflow
   ```
2. Update Tensorflow (if needed)
    ```
   > git fetch origin
   > git pull
   ```

3. Download dependencies
   ```
   > ./tensorflow/lite/tools/make/download_dependencies.sh
   ```

4. Set configuration
    ```
   > ./configure
   ```
   * Follow it's instruction
   
5. Install appropriate version of [bazel](https://docs.bazel.build/versions/master/install.html)

6. Build Libraries
    * [Platform config](https://github.com/tensorflow/tensorflow/blob/master/.bazelrc#L91)
        * Android arm64-v8a: `--config=android_arm64`
        * Android armeabi-v7a: `--config=android_arm`
        * iOS: `--config=ios_fat`
        * No `--config`: detects current platform
        
    * iOS
        ```
        bazel build --config=ios_fat -c opt //tensorflow/lite/experimental/ios:TensorFlowLiteC_framework
        ```
        
    * Other Platform
        
        * CPU
            ```
            bazel build -c opt --cxxopt=--std=c++11 //tensorflow/lite/c:tensorflowlite_c
            ```
          
        * GPU
            ```
            bazel build -c opt //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
            ```
          
        * NNAPI (Android Only)
            ```
            bazel build -c opt --config=android_arm64 //tensorflow/lite/delegates/nnapi:nnapi_delegate
            bazel build -c opt --config=android_arm64 //tensorflow/lite/nnapi:nnapi_implementation
            bazel build -c opt --config=android_arm64 //tensorflow/lite/nnapi:nnapi_util
            ```
          
7. Link built library and headers to `CuteModel` using CMakeLists.txt

    ```
    tflite
       ├─cutemodel // Wrapper
       ├─include   // Tensorflow Lite headers
       ├─lib       // Built Libraries
       └─CMakeLists.txt
    ```

    * CMakeLists.txt (Android Example)
    ```
    cmake_minimum_required(VERSION 3.4.1)
    project(tflite)

    set (TFLITE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    set (TFLITE_LIB_PATH "${TFLITE_PATH}/lib/${ANDROID_ABI}")
    set (TFLITE_INCLUDE_PATH "${TFLITE_PATH}/include")
    set (CUTEMODEL_PATH "${TFLITE_PATH}/cutemodel")
    
    add_library(lib_tflite SHARED IMPORTED)
    set_target_properties(lib_tflite PROPERTIES IMPORTED_LOCATION ${TFLITE_LIB_PATH}/libtensorflowlite_c.so)
    
    add_library(lib_tflite_gpu SHARED IMPORTED)
    set_target_properties(lib_tflite_gpu PROPERTIES IMPORTED_LOCATION ${TFLITE_LIB_PATH}/libtensorflowlite_gpu_delegate.so)
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")
    
    add_library(tflite SHARED ${CUTEMODEL_PATH}/CuteModel.cpp)
    
    
    target_include_directories(tflite PUBLIC
            ${TFLITE_PATH}
            ${TFLITE_INCLUDE_PATH}
            ${TFLITE_INCLUDE_PATH}/tensorflow/lite/tools/make/downloads/flatbuffers/include
            ${TFLITE_INCLUDE_PATH}/tensorflow/lite/tools/make/downloads/absl
            ${TFLITE_INCLUDE_PATH}/tensorflow/lite/tools/make/downloads/absl/absl
            )
    
    target_link_libraries(tflite
            lib_tflite
            lib_tflite_gpu
            log
    )
    ```
    

8. Link `tflite` to your main CMake project.  

9. Use `CuteModel` to build and run models.
    * https://github.com/visualcamp/Tensorflow-Lite/tree/cutemodel
    
