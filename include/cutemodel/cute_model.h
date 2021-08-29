//
// Created by YongGyu Lee on 2021/08/30.
//

#ifndef CUTE_MODEL_CUTE_MODEL_H_
#define CUTE_MODEL_CUTE_MODEL_H_

#include <cstddef>
#include <cstdint>

#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace cute {

class CuteModel {
 private:
  struct Impl;
  Impl* pImpl = nullptr;

  int input_index = 0;

  TfLiteStatus setInputInner(int index, const void* data);

 public:
  CuteModel();
  ~CuteModel();

  CuteModel(const CuteModel&) = delete;
  CuteModel& operator = (const CuteModel&) = delete;

  CuteModel(CuteModel&&) noexcept;
  CuteModel& operator=(CuteModel&&) noexcept;

  CuteModel& buildModelFromBuffer(const void* buffer, std::size_t buffer_size);
  CuteModel& buildModelFromFile(const std::string& path);

  CuteModel& setNumThreads(int num);

  CuteModel& addDelegate(TfLiteDelegate* delegate);

  void buildInterpreter();
  bool isInterpreterBuilt() const;

  template<class Input>
  void setInput(const Input input) {
    setInputInner(input_index++, input);
  }

  template<class Input, class ...Inputs>
  void setInput(const Input input, const Inputs ...inputs) {
    setInput(input);
    setInput(inputs...);
  }

  void copyOutput(int index, void* dst) const;
  template<typename T>
  std::vector<T> getOutput(int index) const {
    std::vector<T> output(output_tensor(index)->bytes / sizeof(T));
    copyOutput(index, output.data());
    return output;
  }

  TfLiteStatus invoke();

  // get specific input tensor
  TfLiteTensor* input_tensor(int index);
  const TfLiteTensor* input_tensor(int index) const;

  // get total numbers of input tensors
  std::int32_t input_tensor_count() const;

  // get total input tensors
  std::vector<TfLiteTensor*> input_tensor();
  std::vector<const TfLiteTensor*> input_tensor() const;

  // get specific output tensor
  const TfLiteTensor* output_tensor(int index) const;

  // get total numbers of output tensors
  std::int32_t output_tensor_count() const;

  // get total output tensors
  std::vector<const TfLiteTensor*> output_tensor() const;

  std::string summarize() const;
};

} // namespace cute

#endif // CUTE_MODEL_CUTE_MODEL_H_
