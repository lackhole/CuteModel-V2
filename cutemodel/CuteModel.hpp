//
// Created by YongGyu Lee on 2020-03-26.
//

#ifndef CUTEMODEL_H
#define CUTEMODEL_H

#define USE_GPU_DELEGATE 1
#define USE_NNAPI_DELEGATE 0


#include <vector>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api.h"

#if USE_GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#if USE_NNAPI_DELEGATE
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"
#endif


namespace ct {
    
    class CuteModel {
    public:
    
        CuteModel() = default;
        ~CuteModel() = default;
        
        /** construct from model buffer */
        CuteModel(const void *buffer, size_t bufferSize);
        
        /** construct from model file */
        CuteModel(const std::string& path);
        
        /** Move con,op */
        CuteModel(CuteModel &&other) noexcept;
        CuteModel& operator=(CuteModel &&other) noexcept;
        
        /** build from model buffer */
        void buildFromBuffer(const void* buffer, size_t bufferSize);
        
        /** build from model file */
        void buildFromFile(const std::string& path);
        
        /**
         * set number of threads
         * if GPU delegate is used, setting a thread is meaningless
         */
        void setCpuNumThreads(int numThread = -1);
    

    #ifdef TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_H_
        /**
         * set GPU delegate
         * you can customize options; See TfLiteGpuDelegateOptionsV2's declared header
         */
        void setGpuDelegate(const TfLiteGpuDelegateOptionsV2 &gpuOptions = TfLiteGpuDelegateOptionsV2Default());
    #endif
    
    #ifdef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
        /**
         * set NNAPI delegate for Android devices
         */
        void setNnApiDelegate(const tflite::StatefulNnApiDelegate::Options &nnApiOptions = tflite::StatefulNnApiDelegate::Options());
    #endif
        
        /**
         * build the interpreter
         */
        void buildInterpreter();
        
        /** check if the interpreter is built */
        bool isBuilt() const noexcept;
        
        /** give input data to the model */
        template<typename Data, typename ...Rest> void setInput(Data &&data, Rest &&... data_rest) noexcept;
        template<typename Data> void setInput(Data &&data) noexcept;
        
        /** run an inference */
        void invoke() noexcept;
        
        /** get specific index of the output data */
        template<typename T> [[deprecated("use copyOutputToBuffer instead")]] void getOutput(int index, std::vector<T> &output) const noexcept;
        template<typename T> std::vector<T> getOutput(int index) const noexcept;
        
        /** copy output data to buffer '
         * IMPORTANT: buffer must be allocated
         */
        template<typename T> void copyOutputToBuffer(int index, T&& buffer) const noexcept;
        template<typename T> void copyOutputToBuffer(int index, T&& buffer, size_t bufferSize) const noexcept;
        
        /** return number of i/o tensors */
        auto inputTensorCount() const;
        auto outputTensorCount() const;
        
        /** return specific index of tensor's length of an element array */
        size_t inputTensorLength(int index) const;
        size_t outputTensorLength(int index) const;
        
        /** return specific index of a tensor */
        TfLiteTensor* inputTensor(int index);
        const TfLiteTensor* inputTensor(int index) const;
        const TfLiteTensor* outputTensor(int index) const;
        
        /** return model's information */
        std::string summary() const;
    
        /** return inferencing hardware options */
        std::string summarizeOptions() const;
        
    
    private:
        
        /** Tensorflow Lite members */
        std::unique_ptr<TfLiteModel, decltype(&TfLiteModelDelete)> model{nullptr, TfLiteModelDelete};
        std::unique_ptr<TfLiteInterpreterOptions, decltype(&TfLiteInterpreterOptionsDelete)> options{nullptr, TfLiteInterpreterOptionsDelete};
        std::unique_ptr<TfLiteInterpreter, decltype(&TfLiteInterpreterDelete)> interpreter{nullptr, TfLiteInterpreterDelete};
        
        #ifdef TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_H_
        /** Tensorflow Lite GPU delegate members */
        TfLiteGpuDelegateOptionsV2 gpuDelegateOptionsV2{};
        std::unique_ptr<TfLiteDelegate, decltype(&TfLiteGpuDelegateV2Delete)> gpuDelegate{nullptr, TfLiteGpuDelegateV2Delete};
        #endif
        
        #ifdef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
        /** Tensorflow Lite NNAPI delegate members */
        tflite::StatefulNnApiDelegate::Options nnApiDelegateOptions{};
        std::unique_ptr<tflite::StatefulNnApiDelegate> nnApiDelegate(nullptr);
        #endif
        
        /** auto incrementing input index */
        int input_data_index = 0;
        
//        /** i/o tensor byte storage */
//        std::vector<size_t> inputTensorByteArray;
//        std::vector<size_t> outputTensorByteArray;
        
        CuteModel(const CuteModel &other) = delete;
        CuteModel& operator=(const CuteModel &other) = delete;
    };
    
    
    template<typename Data>
    inline void CuteModel::setInput(Data &&data) noexcept {
        TfLiteTensorCopyFromBuffer(
                inputTensor(input_data_index),
                data,
                TfLiteTensorByteSize(inputTensor(input_data_index))
        );
        ++input_data_index;
    }
    
    template<typename Data, typename ...Rest>
    inline void CuteModel::setInput(Data &&data, Rest &&... data_rest) noexcept {
        TfLiteTensorCopyFromBuffer(
                inputTensor(input_data_index),
                data,
                TfLiteTensorByteSize(inputTensor(input_data_index))
        );
        ++input_data_index;
        setInput(data_rest...);
    }
    
    inline void CuteModel::invoke() noexcept {
        TfLiteInterpreterInvoke(interpreter.get());
        input_data_index = 0;
    }
    
    template<typename T>
    inline void CuteModel::getOutput(int index, std::vector<T> &output) const noexcept {
        static_assert(!std::is_class<T>::value, "vector must not contain classes");
        
        output.resize((TfLiteTensorByteSize(outputTensor(index)) / sizeof(T))); // std::vector 안에서 자동으로 크기 체크함
        TfLiteTensorCopyToBuffer(outputTensor(index), output.data(), TfLiteTensorByteSize(outputTensor(index)));
    }
    
    template <typename T>
    inline std::vector<T> CuteModel::getOutput(int index) const noexcept {
        static_assert(!std::is_class<T>::value, "vector must not contain classes");
        
        std::vector<T> output((TfLiteTensorByteSize(outputTensor(index)) / sizeof(T)));
        TfLiteTensorCopyToBuffer(outputTensor(index), output.data(), TfLiteTensorByteSize(outputTensor(index)));
        
        return std::move(output);
    }
    
    template<typename T>
    inline void CuteModel::copyOutputToBuffer(int index, T &&buffer) const noexcept {
        copyOutputToBuffer(index, buffer, TfLiteTensorByteSize(outputTensor(index)));
    }
    
    template<typename T>
    inline void CuteModel::copyOutputToBuffer(int index, T &&buffer, size_t bufferSize) const noexcept {
        TfLiteTensorCopyToBuffer(outputTensor(index), buffer, bufferSize);
    }
    
}

#endif //HELLO_LIBS_CUTEMODEL_H