// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow_addons/custom_ops/layers/cc/kernels/deformable_conv2d_op.h"

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace addons {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {
template <typename Dtype>
Dtype bilinear_interpolate(EigenTensor<Dtype, 2> input_tensor, Dtype y,
                           Dtype x) {
  auto max_height = input_tensor.dimension(0);
  auto max_width = input_tensor.dimension(1);

  if (y <= -1 || max_height <= y || x <= -1 || max_width <= x) {
    return Dtype(0);
  }

  int y_low = floor(y);
  int x_low = floor(x);
  int y_high = y_low + 1;
  int w_high = x_low + 1;

  auto v1 = Dtype(0);
  if (y_low >= 0 && x_low >= 0) {
    v1 = input_tensor(y_low, x_low);
  }

  auto v2 = Dtype(0);
  if (y_low >= 0 && w_high <= max_width - 1) {
    v2 = input_tensor(y_low, w_high);
  }

  auto v3 = Dtype(0);
  if (y_high <= max_height - 1 && x_low >= 0) {
    v3 = input_tensor(y_high, x_low);
  }

  auto v4 = Dtype(0);
  if (y_high <= max_height - 1 && w_high <= max_width - 1) {
    v4 = input_tensor(y_high, w_high);
  }

  auto lh = y - y_low;
  auto lw = x - x_low;
  auto hh = 1 - lh;
  auto hw = 1 - lw;

  auto w1 = hh * hw;
  auto w2 = hh * lw;
  auto w3 = lh * hw;
  auto w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename Dtype>
void deformable_im2col(typename TTypes<Dtype, 4>::ConstTensor &_input_tensor,
                       typename TTypes<Dtype, 4>::ConstTensor &_offset_tensor,
                       typename TTypes<Dtype, 4>::ConstTensor &_mask_tensor,
                       typename TTypes<Dtype, 4>::Tensor &column_buffer_tensor,
                       DeformableConv2DParams &p, int32 b) {
  auto use_mask = _mask_tensor.dimension(0) > 0;
  auto num_kernels =
      p.input_channels * p.output_rows * p.output_cols * p.parallel_imgs;
  auto batches = p.input_batches / p.parallel_imgs;

  EigenTensor<Dtype, 4> input_tensor =
      _input_tensor
          .reshape(Shape5D({batches, p.parallel_imgs, p.input_channels,
                            p.input_rows, p.input_cols}))
          .chip(b, 0);

  EigenTensor<Dtype, 7> offset_tensor =
      _offset_tensor
          .reshape(
              Shape8D({batches, p.parallel_imgs, p.offset_groups, p.filter_rows,
                       p.filter_cols, 2, p.output_rows, p.output_cols}))
          .chip(b, 0);

  EigenTensor<Dtype, 6> mask_tensor =
      use_mask
          ? static_cast<EigenTensor<Dtype, 6>>(
                _mask_tensor
                    .reshape(Shape7D({batches, p.parallel_imgs, p.offset_groups,
                                      p.filter_rows, p.filter_cols,
                                      p.output_rows, p.output_cols}))
                    .chip(b, 0))
          : _mask_tensor.reshape(Shape6D({0, 0, 0, 0, 0, 0}));

  for (auto k = 0; k < num_kernels; k++) {
    const auto current_output_row = k % p.output_cols;
    const auto current_output_col = (k / p.output_cols) % p.output_rows;
    const auto current_batch =
        (k / (p.output_rows * p.output_cols)) % p.parallel_imgs;
    const auto current_input_channel =
        k / (p.output_rows * p.output_cols * p.parallel_imgs);
    const auto current_output_channel =
        current_input_channel * p.filter_rows * p.filter_cols;

    const auto group_index =
        current_input_channel / (p.input_channels / p.offset_groups);

    auto input_tensor_chipped =
        input_tensor.chip(current_batch, 0).chip(current_input_channel, 0);
    EigenTensor<Dtype, 5> offset_tensor_chipped =
        offset_tensor.chip(current_batch, 0).chip(group_index, 0);

    EigenTensor<Dtype, 4> mask_tensor_chipped =
        use_mask ? static_cast<EigenTensor<Dtype, 4>>(
                       mask_tensor.chip(current_batch, 0).chip(group_index, 0))
                 : mask_tensor.reshape(Shape4D({0, 0, 0, 0}));

    auto column_buffer_tensor_channel = current_output_channel;
    for (auto current_filter_row = 0; current_filter_row < p.filter_rows;
         current_filter_row++) {
      for (auto current_filter_col = 0; current_filter_col < p.filter_cols;
           current_filter_col++) {
        auto offset_h =
            offset_tensor_chipped(current_filter_row, current_filter_col, 0,
                                  current_output_row, current_output_col);
        auto offset_w =
            offset_tensor_chipped(current_filter_row, current_filter_col, 1,
                                  current_output_row, current_output_col);

        auto mask = use_mask ? mask_tensor_chipped(
                                   current_filter_row, current_filter_col,
                                   current_output_row, current_output_col)
                             : Dtype(1);

        auto y = (current_output_row * p.stride_rows - p.padding_rows) +
                 current_filter_row * p.dilation_rows + offset_h;
        auto x = (current_output_col * p.stride_cols - p.padding_cols) +
                 current_filter_col * p.dilation_cols + offset_w;

        column_buffer_tensor(column_buffer_tensor_channel, current_batch,
                             current_output_row, current_output_col) =
            mask * bilinear_interpolate<Dtype>(input_tensor_chipped, y, x);
        column_buffer_tensor_channel++;
      }
    }
  }
}

template <typename Dtype>
void compute_filter_grad(
    typename TTypes<Dtype, 4>::ConstTensor &input_tensor,
    typename TTypes<Dtype, 4>::ConstTensor &filter_tensor,
    typename TTypes<Dtype, 4>::ConstTensor &offset_tensor,
    typename TTypes<Dtype, 4>::ConstTensor &mask_tensor,
    typename TTypes<Dtype, 4>::ConstTensor &output_grad_tensor,
    typename TTypes<Dtype, 4>::Tensor &filter_grad_tensor,
    typename TTypes<Dtype, 4>::Tensor &column_buffer_tensor,
    DeformableConv2DParams &p) {
  const auto use_mask = mask_tensor.dimension(0) > 0;
  const auto batches = p.input_batches / p.parallel_imgs;

  auto filter_grad_tensor_reshaped = filter_grad_tensor.reshape(
      Shape5D({p.weight_groups, p.output_channels / p.weight_groups,
               p.filter_channels, p.filter_rows, p.filter_cols}));

  EigenTensor<Dtype, 5> output_grad_tensor_reshaped =
      output_grad_tensor
          .reshape(Shape5D({batches, p.parallel_imgs, p.output_channels,
                            p.output_rows, p.output_cols}))
          .shuffle(Shape5D({0, 2, 1, 3, 4}))
          .reshape(Shape5D({batches, p.weight_groups,
                            p.output_channels / p.weight_groups,
                            p.parallel_imgs * p.output_rows, p.output_cols}));

  // input_channels * filter_rows * filter_cols / weight_groups ==
  // filter_channels * filter_rows * filter_cols
  const auto elems = p.filter_channels * p.filter_rows * p.filter_cols;
  const auto rows = p.output_channels / p.weight_groups;
  const auto cols = p.parallel_imgs * p.output_rows * p.output_cols;

  auto column_buffer_tensor_reshaped =
      column_buffer_tensor.reshape(Shape3D({p.weight_groups, elems, cols}));

  for (auto b = 0; b < batches; b++) {
    auto output_grad_tensor_reshaped_batch =
        output_grad_tensor_reshaped.chip(b, 0);

    deformable_im2col<Dtype>(input_tensor, offset_tensor, mask_tensor,
                             column_buffer_tensor, p, b);

    for (auto g = 0; g < p.weight_groups; g++) {
      EigenTensor<Dtype, 2> column_buffer_mtx =
          column_buffer_tensor_reshaped.chip(g, 0).shuffle(Shape2D({1, 0}));

      EigenTensor<Dtype, 2> output_grad_mtx =
          output_grad_tensor_reshaped_batch.chip(g, 0).reshape(
              Shape2D({rows, cols}));

      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
          Eigen::IndexPair<int>(1, 0)};

      EigenTensor<Dtype, 2> mul =
          output_grad_mtx.contract(column_buffer_mtx, product_dims);

      filter_grad_tensor_reshaped.chip(g, 0).reshape(Shape2D({rows, elems})) +=
          mul;
    }
  }
}

template <typename Dtype>
struct DeformableConv2DFunctor<CPUDevice, Dtype> {
  Status operator()(OpKernelContext *context,
                    typename TTypes<Dtype, 4>::ConstTensor input_tensor,
                    typename TTypes<Dtype, 4>::ConstTensor filter_tensor,
                    typename TTypes<Dtype, 1>::ConstTensor bias_tensor,
                    typename TTypes<Dtype, 4>::ConstTensor offset_tensor,
                    typename TTypes<Dtype, 4>::ConstTensor mask_tensor,
                    typename TTypes<Dtype, 4>::Tensor column_buffer_tensor,
                    typename TTypes<Dtype, 4>::Tensor output_tensor,
                    DeformableConv2DParams &p) {
    output_tensor.setZero();

    const auto use_bias = bias_tensor.dimension(0) > 0;
    const auto batches = p.input_batches / p.parallel_imgs;

    auto filter_tensor_reshaped = filter_tensor.reshape(
        Shape5D({p.weight_groups, p.output_channels / p.weight_groups,
                 p.filter_channels, p.filter_rows, p.filter_cols}));

    // FIXME: Make copy for the better performance
    auto output_tensor_reshaped = output_tensor.reshape(
        Shape5D({batches, p.weight_groups, p.output_channels / p.weight_groups,
                 p.parallel_imgs * p.output_rows, p.output_cols}));

    // input_channels * filter_rows * filter_cols / weight_groups ==
    // filter_channels * filter_rows * filter_cols
    const auto elems = p.filter_channels * p.filter_rows * p.filter_cols;
    const auto rows = p.output_channels / p.weight_groups;
    const auto cols = p.parallel_imgs * p.output_rows * p.output_cols;

    auto column_buffer_tensor_reshaped =
        column_buffer_tensor.reshape(Shape3D({p.weight_groups, elems, cols}));

    for (auto b = 0; b < batches; b++) {
      auto output_tensor_reshaped_batch = output_tensor_reshaped.chip(b, 0);

      deformable_im2col<Dtype>(input_tensor, offset_tensor, mask_tensor,
                               column_buffer_tensor, p, b);

      for (auto g = 0; g < p.weight_groups; g++) {
        EigenTensor<Dtype, 2> filter_mtx =
            filter_tensor_reshaped.chip(g, 0).reshape(Shape2D({rows, elems}));
        EigenTensor<Dtype, 2> column_buffer_mtx =
            column_buffer_tensor_reshaped.chip(g, 0);

        auto mtx_shape = Shape2D({rows, cols});
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
            Eigen::IndexPair<int>(1, 0)};

        EigenTensor<Dtype, 2> mul =
            filter_mtx.contract(column_buffer_mtx, product_dims);

        output_tensor_reshaped_batch.chip(g, 0).reshape(mtx_shape) += mul;
      }
    }

    auto output_tensor_transposed =
        output_tensor_reshaped
            .reshape(Shape5D({batches, p.output_channels, p.parallel_imgs,
                              p.output_rows, p.output_cols}))
            .shuffle(Shape5D({0, 2, 1, 3, 4}))
            .reshape(Shape4D({p.input_batches, p.output_channels, p.output_rows,
                              p.output_cols}));

    output_tensor = output_tensor_transposed.eval();

    if (use_bias) {
      auto bias_tensor_broadcasted =
          bias_tensor.reshape(Shape4D({1, p.output_channels, 1, 1}))
              .broadcast(
                  Shape4D({p.input_batches, 1, p.output_rows, p.output_cols}));

      output_tensor += bias_tensor_broadcasted;
    }

    return Status::OK();
  }
};

template <typename Dtype>
struct DeformableConv2DGradFunctor<CPUDevice, Dtype> {
  Status operator()(OpKernelContext *context,
                    typename TTypes<Dtype, 4>::ConstTensor input_tensor,
                    typename TTypes<Dtype, 4>::ConstTensor filter_tensor,
                    typename TTypes<Dtype, 1>::ConstTensor bias_tensor,
                    typename TTypes<Dtype, 4>::ConstTensor offset_tensor,
                    typename TTypes<Dtype, 4>::ConstTensor mask_tensor,
                    typename TTypes<Dtype, 4>::ConstTensor output_grad_tensor,
                    typename TTypes<Dtype, 4>::Tensor input_grad_tensor,
                    typename TTypes<Dtype, 4>::Tensor filter_grad_tensor,
                    typename TTypes<Dtype, 1>::Tensor bias_grad_tensor,
                    typename TTypes<Dtype, 4>::Tensor offset_grad_tensor,
                    typename TTypes<Dtype, 4>::Tensor mask_grad_tensor,
                    typename TTypes<Dtype, 4>::Tensor column_buffer_tensor,
                    DeformableConv2DParams &p) {
    input_grad_tensor.setZero();
    filter_grad_tensor.setZero();

    const auto use_bias = bias_tensor.dimension(0) > 0;
    const auto use_mask = mask_tensor.dimension(0) > 0;

    filter_grad_tensor.setZero();
    compute_filter_grad<Dtype>(input_tensor, filter_tensor, offset_tensor,
                               mask_tensor, output_grad_tensor,
                               filter_grad_tensor, column_buffer_tensor, p);

    if (use_bias) {
      bias_grad_tensor.setConstant(Dtype(1));
      bias_grad_tensor *=
          output_grad_tensor.sum(Eigen::array<int, 3>({0, 2, 3}));
    }

    return Status::OK();
  }
};

}  // end namespace functor

template <typename Device, typename T>
class DeformableConv2DOp : public OpKernel {
 public:
  explicit DeformableConv2DOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
    OP_REQUIRES_OK(context, context->GetAttr("weight_groups", &weight_groups));
    OP_REQUIRES_OK(context, context->GetAttr("offset_groups", &offset_groups));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations));
    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    FormatFromString(data_format_str, &data_format);
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    const Tensor &filter_tensor = context->input(1);
    const Tensor &bias_tensor = context->input(2);
    const Tensor &offset_tensor = context->input(3);
    const Tensor &mask_tensor = context->input(4);

    const TensorShape &input_shape = input_tensor.shape();
    const TensorShape &filter_shape = filter_tensor.shape();
    const TensorShape &bias_shape = bias_tensor.shape();
    const TensorShape &offset_shape = offset_tensor.shape();
    const TensorShape &mask_shape = mask_tensor.shape();

    auto input_batches = input_shape.dim_size(0);
    auto input_channels = input_shape.dim_size(1);
    auto input_rows = input_shape.dim_size(2);
    auto input_cols = input_shape.dim_size(3);

    auto output_channels = filter_shape.dim_size(0);
    auto filter_channels = filter_shape.dim_size(1);
    auto filter_rows = filter_shape.dim_size(2);
    auto filter_cols = filter_shape.dim_size(3);

    auto dilation_rows = dilations[0];
    auto dilation_cols = dilations[1];

    auto stride_rows = strides[0];
    auto stride_cols = strides[1];

    auto parallel_imgs = get_parallel_imgs(input_batches);

    int64 output_rows, output_cols;
    int64 padding_rows, padding_cols;
    OP_REQUIRES_OK(
        context, GetWindowedOutputSizeV2(input_rows, filter_rows, dilation_rows,
                                         stride_rows, padding, &output_rows,
                                         &padding_rows));
    OP_REQUIRES_OK(
        context, GetWindowedOutputSizeV2(input_cols, filter_cols, dilation_cols,
                                         stride_cols, padding, &output_cols,
                                         &padding_cols));

    TensorShape column_buffer_shape({input_channels * filter_rows * filter_cols,
                                     parallel_imgs, output_rows, output_cols});
    Tensor column_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   column_buffer_shape,
                                                   &column_buffer_tensor));

    TensorShape output_shape = ShapeFromFormat(
        data_format, input_batches, output_rows, output_cols, output_channels);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    DeformableConv2DParams p{};
    p.input_batches = input_batches;
    p.input_channels = input_channels;
    p.input_rows = input_rows;
    p.input_cols = input_cols;
    p.filter_channels = filter_channels;
    p.filter_rows = filter_rows;
    p.filter_cols = filter_cols;
    p.padding_rows = padding_rows;
    p.padding_cols = padding_cols;
    p.stride_rows = stride_rows;
    p.stride_cols = stride_cols;
    p.dilation_rows = dilation_rows;
    p.dilation_cols = dilation_cols;
    p.output_channels = output_channels;
    p.output_rows = output_rows;
    p.output_cols = output_cols;
    p.parallel_imgs = parallel_imgs;
    p.weight_groups = weight_groups;
    p.offset_groups = offset_groups;

    functor::DeformableConv2DFunctor<Device, T> deformableConv2DFunc;
    Status s = deformableConv2DFunc(
        context, input_tensor.tensor<T, 4>(), filter_tensor.tensor<T, 4>(),
        bias_tensor.tensor<T, 1>(), offset_tensor.tensor<T, 4>(),
        mask_tensor.tensor<T, 4>(), column_buffer_tensor.tensor<T, 4>(),
        output_tensor->tensor<T, 4>(), p);

    OP_REQUIRES_OK(context, s);
  }

 private:
  std::vector<int32> strides;
  int32 weight_groups;
  int32 offset_groups;
  Padding padding;
  std::vector<int32> dilations;
  TensorFormat data_format;
};

template <typename Device, typename T>
class DeformableConv2DGradOp : public OpKernel {
 public:
  explicit DeformableConv2DGradOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
    OP_REQUIRES_OK(context, context->GetAttr("weight_groups", &weight_groups));
    OP_REQUIRES_OK(context, context->GetAttr("offset_groups", &offset_groups));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations));
    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    FormatFromString(data_format_str, &data_format);
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    const Tensor &filter_tensor = context->input(1);
    const Tensor &bias_tensor = context->input(2);
    const Tensor &offset_tensor = context->input(3);
    const Tensor &mask_tensor = context->input(4);
    const Tensor &output_grad_tensor = context->input(5);

    const TensorShape &input_shape = input_tensor.shape();
    const TensorShape &filter_shape = filter_tensor.shape();
    const TensorShape &bias_shape = bias_tensor.shape();
    const TensorShape &offset_shape = offset_tensor.shape();
    const TensorShape &mask_shape = mask_tensor.shape();

    auto input_batches = input_shape.dim_size(0);
    auto input_channels = input_shape.dim_size(1);
    auto input_rows = input_shape.dim_size(2);
    auto input_cols = input_shape.dim_size(3);

    auto output_channels = filter_shape.dim_size(0);
    auto filter_channels = filter_shape.dim_size(1);
    auto filter_rows = filter_shape.dim_size(2);
    auto filter_cols = filter_shape.dim_size(3);

    auto dilation_rows = dilations[0];
    auto dilation_cols = dilations[1];

    auto stride_rows = strides[0];
    auto stride_cols = strides[1];

    auto parallel_imgs = get_parallel_imgs(input_batches);

    int64 output_rows, output_cols;
    int64 padding_rows, padding_cols;
    OP_REQUIRES_OK(
        context, GetWindowedOutputSizeV2(input_rows, filter_rows, dilation_rows,
                                         stride_rows, padding, &output_rows,
                                         &padding_rows));
    OP_REQUIRES_OK(
        context, GetWindowedOutputSizeV2(input_cols, filter_cols, dilation_cols,
                                         stride_cols, padding, &output_cols,
                                         &padding_cols));

    TensorShape column_buffer_shape({input_channels * filter_rows * filter_cols,
                                     parallel_imgs, output_rows, output_cols});
    Tensor column_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   column_buffer_shape,
                                                   &column_buffer_tensor));

    TensorShape output_shape = ShapeFromFormat(
        data_format, input_batches, output_rows, output_cols, output_channels);

    Tensor *input_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, input_shape, &input_grad_tensor));
    Tensor *filter_grad_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, filter_shape,
                                                     &filter_grad_tensor));
    Tensor *bias_grad_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, bias_shape, &bias_grad_tensor));
    Tensor *offset_grad_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, offset_shape,
                                                     &offset_grad_tensor));
    Tensor *mask_grad_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, mask_shape, &mask_grad_tensor));

    DeformableConv2DParams p{};
    p.input_batches = input_batches;
    p.input_channels = input_channels;
    p.input_rows = input_rows;
    p.input_cols = input_cols;
    p.filter_channels = filter_channels;
    p.filter_rows = filter_rows;
    p.filter_cols = filter_cols;
    p.padding_rows = padding_rows;
    p.padding_cols = padding_cols;
    p.stride_rows = stride_rows;
    p.stride_cols = stride_cols;
    p.dilation_rows = dilation_rows;
    p.dilation_cols = dilation_cols;
    p.output_channels = output_channels;
    p.output_rows = output_rows;
    p.output_cols = output_cols;
    p.parallel_imgs = parallel_imgs;
    p.weight_groups = weight_groups;
    p.offset_groups = offset_groups;

    functor::DeformableConv2DGradFunctor<Device, T> deformableConv2DGradFunc;
    Status s = deformableConv2DGradFunc(
        context, input_tensor.tensor<T, 4>(), filter_tensor.tensor<T, 4>(),
        bias_tensor.tensor<T, 1>(), offset_tensor.tensor<T, 4>(),
        mask_tensor.tensor<T, 4>(), output_grad_tensor.tensor<T, 4>(),
        input_grad_tensor->tensor<T, 4>(), filter_grad_tensor->tensor<T, 4>(),
        bias_grad_tensor->tensor<T, 1>(), offset_grad_tensor->tensor<T, 4>(),
        mask_grad_tensor->tensor<T, 4>(), column_buffer_tensor.tensor<T, 4>(),
        p);

    OP_REQUIRES_OK(context, s);
  }

 private:
  std::vector<int32> strides;
  int32 weight_groups;
  int32 offset_groups;
  Padding padding;
  std::vector<int32> dilations;
  TensorFormat data_format;
};

// Register the CPU kernels.
#define REGISTER_DEFORMABLECONV2D_OP_CPU(T)                   \
  REGISTER_KERNEL_BUILDER(Name("Addons>DeformableConv2D")     \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<T>("T"),        \
                          DeformableConv2DOp<CPUDevice, T>)   \
  REGISTER_KERNEL_BUILDER(Name("Addons>DeformableConv2DGrad") \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<T>("T"),        \
                          DeformableConv2DGradOp<CPUDevice, T>)

TF_CALL_float(REGISTER_DEFORMABLECONV2D_OP_CPU);
#undef REGISTER_DEFORMABLECONV2D_OP_CPU

// Register the GPU kernels.
//#if GOOGLE_CUDA
//
//#define REGISTER_DEFORMABLECONV2D_OP_GPU(T)                   \
//  REGISTER_KERNEL_BUILDER(Name("Addons>DeformableConv2D")     \
//                              .Device(DEVICE_GPU)            \
//                              .TypeConstraint<T>("T"),       \
//                          DeformableConv2DOp<GPUDevice, T>)   \
//  REGISTER_KERNEL_BUILDER(Name("Addons>DeformableConv2DGrad") \
//                              .Device(DEVICE_GPU)            \
//                              .TypeConstraint<T>("T"),       \
//                          DeformableConv2DGradOp<GPUDevice, T>)
//
// TF_CALL_float(REGISTER_DEFORMABLECONV2D_OP_GPU);
//#undef REGISTER_DEFORMABLECONV2D_OP_GPU
//
//#endif  // GOOGLE_CUDA

}  // namespace addons
}  // namespace tensorflow
