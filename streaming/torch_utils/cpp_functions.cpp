#include <torch/extension.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/DLConvertor.h>
#include <ATen/Functions.h>

at::Tensor backward_weight(
    c10::ArrayRef<long int> weight_size,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    c10::ArrayRef<long int> padding,
    c10::ArrayRef<long int> stride,
    c10::ArrayRef<long int> dilation,
    int64_t groups) {

    auto grads = at::convolution_backward(
        grad_output,
        input,
        at::empty(weight_size, input.options()), // temporary weight
        c10::nullopt, // bias_sizes
        stride,
        padding,
        dilation,
        false,  // transposed
        std::vector<int64_t>{0, 0}, // output_padding
        groups,
        {false, true, false}); // output_mask

    return std::get<1>(grads); // grad_weight
}

at::Tensor backward_input(
    c10::ArrayRef<long int> input_size,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    c10::ArrayRef<long int> padding,
    c10::ArrayRef<long int> stride,
    c10::ArrayRef<long int> dilation,
    int64_t groups) {

    auto grads = at::convolution_backward(
        grad_output,
        at::empty(input_size, grad_output.options()), // temporary input
        weight,
        c10::nullopt, // bias_sizes
        stride,
        padding,
        dilation,
        false,  // transposed
        std::vector<int64_t>{0, 0}, // output_padding
        groups,
        {true, false, false}); // output_mask

    return std::get<0>(grads); // grad_input
}

void DLPack_Capsule_Destructor(PyObject* data) {
  HANDLE_TH_ERRORS
    DLManagedTensor * dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(data, "dltensor");
    if (dlMTensor) {
        dlMTensor->deleter(const_cast<DLManagedTensor*>(dlMTensor));
    } else {
        PyErr_Clear();
    }
  END_HANDLE_TH_ERRORS_RET()
}

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward", &backward_weight, "Conv backward_weight");
  m.def("backward_input", &backward_input, "Conv backward_input");
  m.def("to_dlpack_with_device_id", [](const at::Tensor& data, int64_t device_id) {
      DLManagedTensor* dlMTensor = at::toDLPack(data);
      dlMTensor->dl_tensor.device.device_id = device_id;
      auto capsule = py::capsule(dlMTensor, "dltensor", DLPack_Capsule_Destructor);
      return capsule;
  }, "Specify device_id in dlpack, for cupy to copy to right GPU");
}
