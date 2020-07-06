//
// Created by YongGyu Lee on 2020-03-26.
//

#include "cutemodel/CuteModel.hpp"
#include <sstream>
#include <cstring>

using namespace ct;

namespace ct {
    inline static size_t TfLiteTypeByteSize(const TfLiteTensor *tensor) noexcept;
    inline static size_t TfLiteTensorElementArrayLength(const TfLiteTensor *tensor);
}

CuteModel::CuteModel(const void *buffer, size_t bufferSize) :
    model(TfLiteModelCreate(buffer, bufferSize), TfLiteModelDelete),
    options(TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete)
{}

CuteModel::CuteModel(const std::string &path) :
    model(TfLiteModelCreateFromFile(path.c_str()), TfLiteModelDelete),
    options(TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete)
{}

void CuteModel::buildFromBuffer(const void *buffer, size_t bufferSize) {
    model.reset(TfLiteModelCreate(buffer, bufferSize));
    options.reset(TfLiteInterpreterOptionsCreate());
}

void CuteModel::buildFromFile(const std::string &path) {
    model.reset(TfLiteModelCreateFromFile(path.c_str()));
    options.reset(TfLiteInterpreterOptionsCreate());
}

CuteModel::CuteModel(CuteModel &&other) noexcept :
        model(std::move(other.model)),
        options(std::move(other.options)),
        interpreter(std::move(other.interpreter))
#ifdef TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_H_
        ,gpuDelegate(std::move(other.gpuDelegate))
#endif
#ifdef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
        ,nnApiDelegate(std::move(other.nnApiDelegate))
#endif
{}

CuteModel& CuteModel::operator=(CuteModel &&other) noexcept {
    if (model == other.model) return *this;
    
    model = std::move(other.model);
    options = std::move(other.options);
    interpreter = std::move(other.interpreter);
#ifdef TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_H_
    gpuDelegate = std::move(other.gpuDelegate);
#endif
#ifdef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
    nnApiDelegate = std::move(other.nnApiDelegate);
#endif
    
    return *this;
}

void CuteModel::setCpuNumThreads(int numThread) {
    TfLiteInterpreterOptionsSetNumThreads(options.get(), numThread);
}


#ifdef TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_H_
void CuteModel::setGpuDelegate(const TfLiteGpuDelegateOptionsV2 &gpuDelegate_) {
    gpuDelegateOptionsV2 = gpuDelegate_;
    gpuDelegate.reset(TfLiteGpuDelegateV2Create(&gpuDelegateOptionsV2));
    TfLiteInterpreterOptionsAddDelegate(options.get(), gpuDelegate.get());
}
#endif

#ifdef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
void CuteModel::setNnApiDelegate(const tflite::StatefulNnApiDelegate::Options &nnApiOptions) {
    nnApiDelegateOptions = nnApiOptions;
    nnApiDelegate.reset(new tflite::StatefulNnApiDelegate(nnApiDelegateOptions));
    TfLiteInterpreterOptionsAddDelegate(options.get(), nnApiDelegate.get());
}
#endif


void CuteModel::buildInterpreter() {
    interpreter.reset(TfLiteInterpreterCreate(model.get(), options.get()));
    
    if (interpreter == nullptr) return;

#ifdef TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_H_
    if(gpuDelegate == nullptr)
#endif
        TfLiteInterpreterAllocateTensors(interpreter.get());
}

bool CuteModel::isBuilt() const noexcept {
    return interpreter != nullptr;
}

auto CuteModel::inputTensorCount() const {
    return TfLiteInterpreterGetInputTensorCount(interpreter.get());
}

auto CuteModel::outputTensorCount() const {
    return TfLiteInterpreterGetOutputTensorCount(interpreter.get());
}

size_t CuteModel::inputTensorLength(int index) const {
    return TfLiteTensorElementArrayLength(inputTensor(index));
}

size_t CuteModel::outputTensorLength(int index) const {
    return TfLiteTensorElementArrayLength(outputTensor(index));
}

TfLiteTensor* CuteModel::inputTensor(int index) {
    return TfLiteInterpreterGetInputTensor(interpreter.get(), index);
}

const TfLiteTensor* CuteModel::inputTensor(int index) const {
    return TfLiteInterpreterGetInputTensor(interpreter.get(), index);
}

const TfLiteTensor* CuteModel::outputTensor(int index) const {
    return TfLiteInterpreterGetOutputTensor(interpreter.get(), index);
}

std::string CuteModel::summary() const {
    if(interpreter == nullptr)
        return "Interpreter is not built.";
    
    std::stringstream log;
    
    static decltype(auto) getTensorInfo = [](const TfLiteTensor* tensor) {
        std::stringstream log;
        
        log << TfLiteTensorName(tensor) << ' ';
        log << TfLiteTensorByteSize(tensor) << ' ';
        log << TfLiteTypeGetName(TfLiteTensorType(tensor)) << ' ';
    
        if(tensor->dims[0].size > 0) {
            log << tensor->dims[0].data[0];
            for(int s = 1; s < tensor->dims[0].size; ++s)
                log << 'x' << tensor->dims[0].data[s];
        }
        else
            log << "None";
        
        return log.str();
    };
    
    log << " Input Tensor\n";
    log << " Number / Name / Byte / Type / Size\n";
    for(int i=0; i<inputTensorCount(); ++i){
        log <<  "  #" << i << ' ' << getTensorInfo(this->inputTensor(i)) << '\n';
    }
    log << '\n';
    
    
    log << " Output Tensor\n";
    log << " Number / Name / Byte / Type / Size\n";
    for(int i=0; i<outputTensorCount(); ++i){
        log << "  #" << i << ' ' << getTensorInfo(this->outputTensor(i)) << '\n';
    }
    log << '\n';
    
    
    return log.str();
}

std::string CuteModel::summarizeOptions() const {
    
    std::stringstream log;
    #ifdef TENSORFLOW_LITE_C_C_API_INTERNAL_H_
    log << "Threads: ";
    if (options->num_threads == TfLiteInterpreterOptions::kDefaultNumThreads) log << "default\n";
    else log << options->num_threads << '\n';
    
    log << "Delegates: " << options->delegates.size() << '\n';
    log << "Use NNAPI: " << (options->use_nnapi ? "Yes" : "No");

    #ifdef TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_H_
    static decltype(auto) getInferencePriority = [](int priority) {
        switch(priority){
            case TFLITE_GPU_INFERENCE_PRIORITY_AUTO:             return std::string("Auto");
            case TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION:    return std::string("MaxPrecision");
            case TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY:      return std::string("MinLatency");
            case TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE: return std::string("MinMemory");
            default:                                             return std::string("Auto");
        }
    };
    
    if(gpuDelegate != nullptr && options->delegates[0]->data_ != NULL) {
        log << "\nGPU delegate Options:\n"    <<
               "    Precision Loss Allowed: " << (gpuDelegateOptionsV2.is_precision_loss_allowed ? "Yes\n" : "No\n") <<
               "    Inference Preference: "   << (gpuDelegateOptionsV2.inference_preference == TfLiteGpuInferenceUsage::TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER ? "Fast Single Answer\n" : "Sustained Speed\n") <<
               "    Inference Priority: "     <<
               getInferencePriority(gpuDelegateOptionsV2.inference_priority1) << ", " <<
               getInferencePriority(gpuDelegateOptionsV2.inference_priority2) << ", " <<
               getInferencePriority(gpuDelegateOptionsV2.inference_priority3);
    }
    #endif

    #ifdef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
    using ExecutionPreference = tflite::StatefulNnApiDelegate::Options::ExecutionPreference;
    static decltype(auto) getExecutionPreference = [](ExecutionPreference preference) {
        switch(preference){
            case ExecutionPreference::kUndefined:        return std::string("Undefined");
            case ExecutionPreference::kLowPower:         return std::string("LowPower");
            case ExecutionPreference::kFastSingleAnswer: return std::string("FastSingleAnswer");
            case ExecutionPreference::kSustainedSpeed:   return std::string("SustainedSpeed");
        }
    };
    if(nnApiDelegate != nullptr && options->delegates[0]->data_ != NULL){
        
        
        const auto checkChar = [](const char* str) -> std::string{
            if (str == nullptr)
                return "-";
            return std::string(str);
        };
        
        log <<
          "\nNNAPI delegate Options:" <<
          "\n    Execution Preference: " << getExecutionPreference(nnApiDelegateOptions.execution_preference) <<
          "\n  Accelerator Name: " << checkChar(nnApiDelegateOptions.accelerator_name) <<
          "\n  Cache Dir: " << checkChar(nnApiDelegateOptions.cache_dir) <<
          "\n  Model Token: " << checkChar(nnApiDelegateOptions.model_token) <<
          "\n  Disallow NNAPI CPU: " << (nnApiDelegateOptions.disallow_nnapi_cpu ? "Yes" : "No") <<
          "\n  Max Numper Delegate Partition: " << nnApiDelegateOptions.max_number_delegated_partitions;
    }
    #endif
    
    #endif //TENSORFLOW_LITE_C_C_API_INTERNAL_H_
    
    return log.str();
}

inline size_t ct::TfLiteTypeByteSize(const TfLiteTensor* tensor) noexcept{
    switch(TfLiteTensorType(tensor)){
        case kTfLiteNoType:     return static_cast<size_t>(0);
        case kTfLiteFloat32:    return sizeof(float);
        case kTfLiteInt32:      return sizeof(int32_t);
        case kTfLiteUInt8:      return sizeof(uint8_t);
        case kTfLiteInt64:      return sizeof(int64_t);
        case kTfLiteString:     return strlen(reinterpret_cast<const char*>(TfLiteTensorData(tensor)));
        case kTfLiteBool:       return sizeof(bool);
        case kTfLiteInt16:      return sizeof(int16_t);
        case kTfLiteComplex64:  return sizeof(TfLiteComplex64);
        case kTfLiteInt8:       return sizeof(int8_t);
        case kTfLiteFloat16:    return sizeof(TfLiteFloat16);
        case kTfLiteFloat64:    return sizeof(double);
    }
}

inline size_t ct::TfLiteTensorElementArrayLength(const TfLiteTensor *tensor) {
    return TfLiteTensorByteSize(tensor) / ct::TfLiteTypeByteSize(tensor);
}

