//
// Created by YongGyu Lee on 2021/08/30.
//

#include "cutemodel/cute_model.h"

#include <memory>
#include <string>
#include <vector>
#include <sstream>

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"

namespace cute {

struct CuteModel::Impl {
  std::unique_ptr<TfLiteModel, decltype(&TfLiteModelDelete)>
    model{nullptr, TfLiteModelDelete};
  std::unique_ptr<TfLiteInterpreterOptions, decltype(&TfLiteInterpreterOptionsDelete)>
    options{nullptr, TfLiteInterpreterOptionsDelete};
  std::unique_ptr<TfLiteInterpreter, decltype(&TfLiteInterpreterDelete)>
    interpreter{nullptr, &(TfLiteInterpreterDelete)};
};

CuteModel::CuteModel() {
  pImpl = new Impl();
}

CuteModel::~CuteModel() {
  delete pImpl;
  pImpl = nullptr;
}

CuteModel::CuteModel(CuteModel&& other) noexcept : pImpl(other.pImpl) {
  other.pImpl = nullptr;
}

CuteModel &CuteModel::operator=(CuteModel&& other) noexcept {
  if (this != &other) {
    pImpl = other.pImpl;
    other.pImpl = nullptr;
  }
  return *this;
}

CuteModel& CuteModel::buildModelFromFile(const std::string& path) {
  pImpl->model.reset(TfLiteModelCreateFromFile(path.c_str()));
  pImpl->options.reset(TfLiteInterpreterOptionsCreate());
  return *this;
}

CuteModel& CuteModel::buildModelFromBuffer(const void* buffer, std::size_t size) {
  pImpl->model.reset(TfLiteModelCreate(buffer, size));
  pImpl->options.reset(TfLiteInterpreterOptionsCreate());
  return *this;
}

CuteModel& CuteModel::setNumThreads(std::int32_t num) {
  TfLiteInterpreterOptionsSetNumThreads(pImpl->options.get(), num);
  return *this;
}


CuteModel& CuteModel::addDelegate(TfLiteDelegate *delegate) {
  TfLiteInterpreterOptionsAddDelegate(pImpl->options.get(), delegate);
  return *this;

}

void CuteModel::buildInterpreter() {
  pImpl->interpreter.reset(TfLiteInterpreterCreate(pImpl->model.get(), pImpl->options.get()));
}

bool CuteModel::isInterpreterBuilt() const {
  return pImpl->interpreter != nullptr;
}

TfLiteStatus CuteModel::setInputInner(int index, const void *data) {
  auto tensor = input_tensor()[index];
  return TfLiteTensorCopyFromBuffer(tensor, data, tensor->bytes); // This is always TfLiteStatus::kTfLiteOk
}

TfLiteStatus CuteModel::invoke() {
  input_index = 0;
  return TfLiteInterpreterInvoke(pImpl->interpreter.get());
}

void CuteModel::copyOutput(int index, void *dst) const {
  auto tensor = output_tensor()[index];
  TfLiteTensorCopyToBuffer(tensor, dst, TfLiteTensorByteSize(tensor));
}

TfLiteTensor *CuteModel::input_tensor(int index) {
  return TfLiteInterpreterGetInputTensor(pImpl->interpreter.get(), index);
}

const TfLiteTensor *CuteModel::input_tensor(int index) const {
  return TfLiteInterpreterGetInputTensor(pImpl->interpreter.get(), index);
}

std::int32_t CuteModel::input_tensor_count() const {
  return TfLiteInterpreterGetInputTensorCount(pImpl->interpreter.get());
}

std::vector<TfLiteTensor *> CuteModel::input_tensor() {
  const auto count = input_tensor_count();
  std::vector<TfLiteTensor *> tensors(count);
  for (int i = 0; i < count; ++i)
    tensors[i] = input_tensor(i);
  return tensors;
}
std::vector<const TfLiteTensor *> CuteModel::input_tensor() const {
  const auto count = input_tensor_count();
  std::vector<const TfLiteTensor *> tensors(count);
  for (int i = 0; i < count; ++i)
    tensors[i] = input_tensor(i);
  return tensors;
}

const TfLiteTensor *CuteModel::output_tensor(int index) const {
  return TfLiteInterpreterGetOutputTensor(pImpl->interpreter.get(), index);
}

std::int32_t CuteModel::output_tensor_count() const {
  return TfLiteInterpreterGetOutputTensorCount(pImpl->interpreter.get());
}

std::vector<const TfLiteTensor *> CuteModel::output_tensor() const {
  const auto count = output_tensor_count();
  std::vector<const TfLiteTensor *> tensors(count);
  for (int i = 0; i < count; ++i)
    tensors[i] = output_tensor(i);
  return tensors;
}

std::string CuteModel::summarize() const {
  static const auto write_array = [](std::stringstream& ss, TfLiteIntArray* array) -> std::stringstream& {
    if (array->size == 0) {
      ss << 0;
    } else {
      ss << array->data[0];
      for (int i = 1; i < array->size; ++i) {
        ss << 'x' << array->data[i];
      }
    }
    return ss;
  };

  if (!isInterpreterBuilt())
    return "Interpreter is not built";

  std::stringstream buffer;

  for (int i = 0; i < input_tensor_count(); ++i) {
    const auto tensor = input_tensor(i);
    buffer << "input" << i << ' '
    << tensor->name << ' '
    << tensor->bytes << ' '
    << TfLiteTypeGetName(tensor->type);
    write_array(buffer, tensor->dims) << '\n';
  }

  for (int i = 0; i < output_tensor_count(); ++i) {
    const auto tensor = output_tensor(i);
    buffer << "output" << i << ' '
    << tensor->name << ' '
    << tensor->bytes << ' '
    << TfLiteTypeGetName(tensor->type);
    write_array(buffer, tensor->dims) << '\n';
  }

  return buffer.str();
}

} // namespace cute
