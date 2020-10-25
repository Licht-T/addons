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

#ifndef TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_
#define TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/aggregate_ops.h"
#include "tensorflow/core/kernels/batch_matmul_op_impl.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace addons {
static const int kMaxParallelImgs = 32;

struct DeformableConv2DParams {
  int32 input_batches;
  int32 input_channels;
  int32 input_rows;
  int32 input_cols;
  int32 filter_channels;
  int32 filter_rows;
  int32 filter_cols;
  int32 padding_rows;
  int32 padding_cols;
  int32 stride_rows;
  int32 stride_cols;
  int32 dilation_rows;
  int32 dilation_cols;
  int32 output_channels;
  int32 output_rows;
  int32 output_cols;
  int32 parallel_imgs;
  int32 weight_groups;
  int32 offset_groups;
  int32 batches;
  bool use_mask;
  bool use_bias;
};

namespace functor {

template <typename Device, typename T>
struct SetZeroFunctor {
  void operator()(const Device &d, typename TTypes<T>::Flat out);
};

}  // namespace functor

template <typename Device, typename T>
Status TensorSetZero(OpKernelContext *ctx, Tensor *value) {
  functor::SetZeroFunctor<Device, T> set_zero_functor;
  set_zero_functor(ctx->template eigen_device<Device>(), value->flat<T>());
  return Status::OK();
}

template <typename Device, typename T>
Status AddToTensor(OpKernelContext *ctx, Tensor *sum, const Tensor *current,
                   const Tensor *add) {
  ::tensorflow::functor::Add2EigenImpl<Device, T>::Compute(
      ctx->template eigen_device<Device>(), sum->flat<T>(), current->flat<T>(),
      add->flat<T>());
  return Status::OK();
}

namespace functor {

template <typename Device, typename T>
struct DeformableConv2DFunctorBase {
  DeformableConv2DFunctorBase(const Tensor *_input_tensor,
                              const Tensor *_filter_tensor,
                              const Tensor *_bias_tensor,
                              const Tensor *_offset_tensor,
                              const Tensor *_mask_tensor,
                              Tensor *_column_buffer_tensor,
                              DeformableConv2DParams *_p)
      : input_tensor(_input_tensor->dtype()),
        filter_tensor(_filter_tensor->dtype()),
        bias_tensor(_bias_tensor->dtype()),
        offset_tensor(_offset_tensor->dtype()),
        mask_tensor(_mask_tensor->dtype()),
        column_buffer_tensor(_column_buffer_tensor->dtype()),
        p(*_p) {
    CHECK(input_tensor.CopyFrom(
        *_input_tensor,
        TensorShape({p.batches, p.parallel_imgs, p.input_channels, p.input_rows,
                     p.input_cols})));
    CHECK(filter_tensor.CopyFrom(
        *_filter_tensor,
        TensorShape({p.weight_groups, p.output_channels / p.weight_groups,
                     p.filter_channels * p.filter_rows * p.filter_cols})));
    CHECK(bias_tensor.CopyFrom(*_bias_tensor, _bias_tensor->shape()));

    CHECK(offset_tensor.CopyFrom(
        *_offset_tensor,
        TensorShape({p.batches, p.parallel_imgs, p.offset_groups, p.filter_rows,
                     p.filter_cols, 2, p.output_rows, p.output_cols})));

    if (p.use_mask) {
      CHECK(mask_tensor.CopyFrom(
          *_mask_tensor,
          TensorShape({p.batches, p.parallel_imgs, p.offset_groups,
                       p.filter_rows, p.filter_cols, p.output_rows,
                       p.output_cols})));
    } else {
      CHECK(mask_tensor.CopyFrom(*_mask_tensor,
                                 TensorShape({0, 0, 0, 0, 0, 0, 0})));
    }

    CHECK(column_buffer_tensor.CopyFrom(
        *_column_buffer_tensor,
        TensorShape({p.input_channels * p.filter_rows * p.filter_cols,
                     p.parallel_imgs, p.output_rows, p.output_cols})));
  }

  virtual Status operator()(OpKernelContext *context) = 0;

  EIGEN_DEVICE_FUNC T BilinearInterpolate(int32 b, int32 batch, int32 channel,
                                          T y, T x) {
    auto img = input_tensor.SubSlice(b)
                   .SubSlice(batch)
                   .SubSlice(channel)
                   .unaligned_shaped<T, 2>({p.input_rows, p.input_cols});

    auto max_height = img.dimension(0);
    auto max_width = img.dimension(1);

    if (y <= -1 || max_height <= y || x <= -1 || max_width <= x) {
      return T(0);
    }

    int y_low = floor(y);
    int x_low = floor(x);
    int y_high = y_low + 1;
    int w_high = x_low + 1;

    auto v1 = T(0);
    if (y_low >= 0 && x_low >= 0) {
      v1 = img(y_low, x_low);
    }

    auto v2 = T(0);
    if (y_low >= 0 && w_high <= max_width - 1) {
      v2 = img(y_low, w_high);
    }

    auto v3 = T(0);
    if (y_high <= max_height - 1 && x_low >= 0) {
      v3 = img(y_high, x_low);
    }

    auto v4 = T(0);
    if (y_high <= max_height - 1 && w_high <= max_width - 1) {
      v4 = img(y_high, w_high);
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

  void DeformableIm2Col(int32 b);

  Tensor input_tensor;
  Tensor filter_tensor;
  Tensor bias_tensor;
  Tensor offset_tensor;
  Tensor mask_tensor;
  Tensor column_buffer_tensor;
  DeformableConv2DParams p;
};

template <typename Device, typename T>
struct DeformableConv2DFunctor : public DeformableConv2DFunctorBase<Device, T> {
  using DeformableConv2DFunctorBase<Device, T>::input_tensor;
  using DeformableConv2DFunctorBase<Device, T>::filter_tensor;
  using DeformableConv2DFunctorBase<Device, T>::bias_tensor;
  using DeformableConv2DFunctorBase<Device, T>::offset_tensor;
  using DeformableConv2DFunctorBase<Device, T>::mask_tensor;
  using DeformableConv2DFunctorBase<Device, T>::column_buffer_tensor;
  using DeformableConv2DFunctorBase<Device, T>::p;

  DeformableConv2DFunctor(const Tensor *_input_tensor,
                          const Tensor *_filter_tensor,
                          const Tensor *_bias_tensor,
                          const Tensor *_offset_tensor,
                          const Tensor *_mask_tensor,
                          Tensor *_column_buffer_tensor, Tensor *_output_tensor,
                          DeformableConv2DParams *_p)
      : DeformableConv2DFunctorBase<Device, T>(
            _input_tensor, _filter_tensor, _bias_tensor, _offset_tensor,
            _mask_tensor, _column_buffer_tensor, _p),
        output_tensor(_output_tensor->dtype()) {
    CHECK(output_tensor.CopyFrom(*_output_tensor, _output_tensor->shape()));
  }

  Status operator()(OpKernelContext *context) {
    TF_RETURN_IF_ERROR(TensorSetZero<Device, T>(context, &output_tensor));

    // input_channels * filter_rows * filter_cols / weight_groups ==
    // filter_channels * filter_rows * filter_cols
    const auto elems = p.filter_channels * p.filter_rows * p.filter_cols;
    const auto rows = p.output_channels / p.weight_groups;
    const auto cols = p.parallel_imgs * p.output_rows * p.output_cols;

    Tensor output_tmp_tensor;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<T>::value,
        TensorShape({p.batches, p.output_channels, p.parallel_imgs,
                     p.output_rows, p.output_cols}),
        &output_tmp_tensor));

    Tensor output_tmp_mtx_tensor;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DataTypeToEnum<T>::value, TensorShape({p.weight_groups, rows, cols}),
        &output_tmp_mtx_tensor));

    Tensor output_tmp_tensor_reshaped(output_tmp_tensor.dtype());
    CHECK(output_tmp_tensor_reshaped.CopyFrom(
        output_tmp_tensor,
        TensorShape({p.batches, p.weight_groups, rows, cols})));

    Tensor column_buffer_tensor_reshaped(column_buffer_tensor.dtype());
    CHECK(column_buffer_tensor_reshaped.CopyFrom(
        column_buffer_tensor, TensorShape({p.weight_groups, elems, cols})));

    for (auto b = 0; b < p.batches; b++) {
      this->DeformableIm2Col(b);

      auto lhs = filter_tensor;
      auto rhs = column_buffer_tensor_reshaped;
      auto out = output_tmp_mtx_tensor;

      MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());

      LaunchBatchMatMul<Device, T>::Launch(context, lhs, rhs, false, false,
                                           false, false, bcast, &out);

      TF_RETURN_IF_ERROR(
          batch_util::CopyElementToSlice(out, &output_tmp_tensor_reshaped, b));
    }

    Tensor output_tensor_reshaped(output_tensor.dtype());
    CHECK(output_tensor_reshaped.CopyFrom(
        output_tensor,
        TensorShape({p.batches, p.parallel_imgs, p.output_channels,
                     p.output_rows, p.output_cols})));
    TransposeUsingEigen<Device, T, 5>(context->eigen_device<Device>(),
                                      output_tmp_tensor, {0, 2, 1, 3, 4},
                                      &output_tensor_reshaped);

    if (p.use_bias) {
      Eigen::DSizes<int32, 4> four_dims(1, p.output_channels, 1, 1);
      Eigen::DSizes<int32, 4> broadcast_dims(p.input_batches, 1, p.output_rows,
                                             p.output_cols);
      const Device &d = context->eigen_device<Device>();

      auto bias_tensor_broadcasted =
          bias_tensor.template tensor<T, 1>().reshape(four_dims).broadcast(
              broadcast_dims);

      output_tensor.tensor<T, 4>().device(d) += bias_tensor_broadcasted;
    }

    return Status::OK();
  }

  Tensor output_tensor;
};

template <typename Device, typename T>
struct DeformableConv2DGradFunctor
    : public DeformableConv2DFunctorBase<Device, T> {
  DeformableConv2DGradFunctor(
      const Tensor *_input_tensor, const Tensor *_filter_tensor,
      const Tensor *_bias_tensor, const Tensor *_offset_tensor,
      const Tensor *_mask_tensor, Tensor *_output_grad_tensor,
      Tensor *input_grad_tensor, Tensor *filter_grad_tensor,
      Tensor *bias_grad_tensor, Tensor *offset_grad_tensor,
      Tensor *mask_grad_tensor, Tensor *column_buffer_tensor,
      DeformableConv2DParams *p);

  Status operator()(OpKernelContext *context);

  Tensor output_grad_tensor;
  Tensor input_grad_tensor;
  Tensor filter_grad_tensor;
  Tensor bias_grad_tensor;
  Tensor offset_grad_tensor;
  Tensor mask_grad_tensor;
};

}  // namespace functor

template <typename Device, typename T, int NDIMS>
void TransposeUsingEigen(const Device &d, const Tensor &in,
                         const gtl::ArraySlice<int32> perm, Tensor *out) {
  Eigen::array<int, NDIMS> p;
  for (int i = 0; i < NDIMS; ++i) p[i] = perm[i];
  auto x = typename TTypes<T, NDIMS>::ConstTensor(
      reinterpret_cast<const T *>(in.tensor_data().data()),
      in.shape().AsEigenDSizes<NDIMS>());
  auto y = typename TTypes<T, NDIMS>::Tensor(
      reinterpret_cast<T *>(const_cast<char *>(out->tensor_data().data())),
      out->shape().AsEigenDSizes<NDIMS>());

  y.device(d) = x.shuffle(p);
}
}  // namespace addons
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_
