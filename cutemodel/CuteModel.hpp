//
// Created by YongGyu Lee on 2020-03-26.
//

#ifndef CUTEMODEL_H
#define CUTEMODEL_H

#define USE_GPU_DELEGATE 0
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
    
    size_t elementByteSize(const TfLiteTensor *tensor);
    
    size_t tensorLength(const TfLiteTensor *tensor);
    
    class CuteModel {
    public:
        
        TfLiteModel *model = nullptr;
        TfLiteInterpreterOptions *options = nullptr;
        TfLiteInterpreter *interpreter = nullptr;
        
        
    
        /**
         * Constructor / Move Con/Op
         */

//    CuteModel() = delete;
        CuteModel() = default;
        
        ~CuteModel();
        
        CuteModel(void *buffer, size_t bufferSize);
        
        CuteModel(const std::string& path);
        
        CuteModel(CuteModel &&other);
        
        CuteModel &operator=(CuteModel &&other);
        
        /**
         *
         */
        
        
        void setCpuNumThreads(int numThread = -1);

#ifdef TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_H_
        TfLiteGpuDelegateOptionsV2 gpuDelegateOptionsV2{};
        TfLiteDelegate *gpuDelegate = nullptr;
        
        void setGpuDelegate(const TfLiteGpuDelegateOptionsV2 &gpuOptions = TfLiteGpuDelegateOptionsV2Default());
#endif

#ifdef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
        tflite::StatefulNnApiDelegate::Options nnApiDelegateOptions{};
        tflite::StatefulNnApiDelegate *nnApiDelegate = nullptr;
        
        void setNnApiDelegate(const tflite::StatefulNnApiDelegate::Options &nnApiOptions = tflite::StatefulNnApiDelegate::Options());
#endif
        
        TfLiteInterpreter* buildInterpreter();
        
         bool isBuilt() const;
        
        template<typename Data, typename ...Rest>
        void setInput(const Data &data, const Rest &... data_rest);
        
        template<typename Data>
        void setInput(const Data &data);
        
        void invoke();
        
        template<typename T>
        void getOutput(int index, std::vector<T> &output) const;
        
        template<typename T>
        void getOutput(std::vector<std::vector<T>> &output) const;
        
        template<typename T>
        std::vector<std::vector<T>> getOutput() const;
        
        template<typename T>
        std::vector<T> getOutput(int index) const;
        
        int32_t inputTensorCount() const;
        int32_t outputTensorCount() const;
        
        size_t inputTensorLength(int index) const;
        size_t outputTensorLength(int index) const;
        
        TfLiteTensor* inputTensor(int index);
        const TfLiteTensor* inputTensor(int index) const;
        
        const TfLiteTensor* outputTensor(int index) const;
        
        std::string summary() const;

        std::string summarizeOptions() const;
        
    
    private:
        int input_data_index = 0;
        
        void clear();
        
        CuteModel(const CuteModel &other) = delete;
        
        void operator=(const CuteModel &other) = delete;
    };
    
    
    template<typename Data>
    void CuteModel::setInput(const Data &data) {
        assert(("Too many input to model", input_data_index < inputTensorCount()));
        
        TfLiteTensorCopyFromBuffer(
                inputTensor(input_data_index),
                data,
                TfLiteTensorByteSize(inputTensor(input_data_index))
        );
        ++input_data_index;
    }
    
    template<typename Data, typename ...Rest>
    void CuteModel::setInput(const Data &data, const Rest &... data_rest) {
        assert(("Too many input to model", input_data_index < inputTensorCount()));
        
        TfLiteTensorCopyFromBuffer(
                inputTensor(input_data_index),
                data,
                TfLiteTensorByteSize(inputTensor(input_data_index))
        );
        ++input_data_index;
        setInput(data_rest...);
    }
    
    template<typename T>
    void CuteModel::getOutput(std::vector<std::vector<T>> &output) const {
        if(!output.empty())
            output.clear();
        
        output.resize(outputTensorCount());
        for(int i = 0; i < output.size(); ++i) {
            output[i].resize((TfLiteTensorByteSize(outputTensor(i)) / sizeof(T)));
            TfLiteTensorCopyToBuffer(outputTensor(i), output[i].data(),
                                     TfLiteTensorByteSize(outputTensor(i)));
        }
    }
    
    template<typename T>
    void CuteModel::getOutput(int index, std::vector<T> &output) const {
        if(!output.empty())
            output.clear();
        
        output.resize((TfLiteTensorByteSize(outputTensor(index)) / sizeof(T)));
        TfLiteTensorCopyToBuffer(outputTensor(index), output.data(),
                                 TfLiteTensorByteSize(outputTensor(index)));
    }
    
    template<typename T>
    std::vector<std::vector<T>> CuteModel::getOutput() const {
        std::vector<std::vector<T>> output(outputTensorCount());
        
        for (int i=0; i<output.size(); ++i){
            output[i].resize((TfLiteTensorByteSize(outputTensor(i)) / sizeof(T)));
            TfLiteTensorCopyToBuffer(outputTensor(i), output[i].data(),
                                     TfLiteTensorByteSize(outputTensor(i)));
        }
        
        return std::move(output);
    }
    
    template <typename T>
    std::vector<T> CuteModel::getOutput(int index) const {
        std::vector<T> output((TfLiteTensorByteSize(outputTensor(index)) / sizeof(T)));
        TfLiteTensorCopyToBuffer(outputTensor(index), output.data(),
                                 TfLiteTensorByteSize(outputTensor(index)));
        
        return std::move(output);
    }
    
}

#endif //HELLO_LIBS_CUTEMODEL_H