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
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace addons {
using Shape4D = Eigen::array<Eigen::DenseIndex, 4>;

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
struct DeformableConv2DFunctorBase {
  DeformableConv2DFunctorBase(const Tensor* _input_tensor,
                              const Tensor* _filter_tensor,
                              const Tensor* _bias_tensor,
                              const Tensor* _offset_tensor,
                              const Tensor* _mask_tensor,
                              Tensor* _column_buffer_tensor,
                              DeformableConv2DParams* _p)
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
    CHECK(bias_tensor.CopyFrom(*_bias_tensor, bias_tensor.shape()));

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

  virtual Status operator()(OpKernelContext* context) = 0;

  T BilinearInterpolate(int32 b, int32 batch, int32 channel, T y, T x) {
    auto img = input_tensor.SubSlice(b)
                   .SubSlice(batch)
                   .SubSlice(channel)
                   .tensor<T, 2>();

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

  void DeformableIm2Col(int32 b) {
    auto num_kernels =
        p.input_channels * p.output_rows * p.output_cols * p.parallel_imgs;

    auto column_buffer_eigen_tensor = column_buffer_tensor.tensor<T, 4>();

    for (auto k = 0; k < num_kernels; k++) {
      const auto current_output_col = k % p.output_cols;
      const auto current_output_row = (k / p.output_cols) % p.output_rows;
      const auto current_batch =
          (k / (p.output_rows * p.output_cols)) % p.parallel_imgs;
      const auto current_input_channel =
          k / (p.output_rows * p.output_cols * p.parallel_imgs);
      const auto current_output_channel =
          current_input_channel * p.filter_rows * p.filter_cols;

      const auto current_actual_batch = b * p.parallel_imgs + current_batch;

      const auto group_index =
          current_input_channel / (p.input_channels / p.offset_groups);

      const auto offset_eigen_tensor = offset_tensor.SubSlice(b)
                                           .SubSlice(current_batch)
                                           .SubSlice(group_index)
                                           .tensor<T, 5>();

      const auto mask_eigen_tensor =
          p.use_mask ? mask_tensor.SubSlice(b)
                           .SubSlice(current_batch)
                           .SubSlice(group_index)
                           .tensor<T, 4>()
                     : mask_tensor.shaped<T, 4>({0, 0, 0, 0});

      auto column_buffer_tensor_channel = current_output_channel;
      for (auto current_filter_row = 0; current_filter_row < p.filter_rows;
           current_filter_row++) {
        for (auto current_filter_col = 0; current_filter_col < p.filter_cols;
             current_filter_col++) {
          auto offset_h =
              offset_eigen_tensor(current_filter_row, current_filter_col, 0,
                                  current_output_row, current_output_col);
          auto offset_w =
              offset_eigen_tensor(current_filter_row, current_filter_col, 1,
                                  current_output_row, current_output_col);

          auto mask = p.use_mask ? mask_eigen_tensor(
                                       current_filter_row, current_filter_col,
                                       current_output_row, current_output_col)
                                 : T(1);

          auto y = (current_output_row * p.stride_rows - p.padding_rows) +
                   current_filter_row * p.dilation_rows + offset_h;
          auto x = (current_output_col * p.stride_cols - p.padding_cols) +
                   current_filter_col * p.dilation_cols + offset_w;

          column_buffer_eigen_tensor(column_buffer_tensor_channel,
                                     current_batch, current_output_row,
                                     current_output_col) =
              mask * BilinearInterpolate(b, current_actual_batch,
                                         current_input_channel, y, x);
          column_buffer_tensor_channel++;
        }
      }
    }
  }

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
  DeformableConv2DFunctor(const Tensor* _input_tensor,
                          const Tensor* _filter_tensor,
                          const Tensor* _bias_tensor,
                          const Tensor* _offset_tensor,
                          const Tensor* _mask_tensor,
                          Tensor* _column_buffer_tensor, Tensor* _output_tensor,
                          DeformableConv2DParams* _p);

  Status operator()(OpKernelContext* context);

  Tensor output_tensor;
};

template <typename Device, typename T>
struct DeformableConv2DGradFunctor
    : public DeformableConv2DFunctorBase<Device, T> {
  DeformableConv2DGradFunctor(
      const Tensor* _input_tensor, const Tensor* _filter_tensor,
      const Tensor* _bias_tensor, const Tensor* _offset_tensor,
      const Tensor* _mask_tensor, Tensor* _output_grad_tensor,
      Tensor* input_grad_tensor, Tensor* filter_grad_tensor,
      Tensor* bias_grad_tensor, Tensor* offset_grad_tensor,
      Tensor* mask_grad_tensor, Tensor* column_buffer_tensor,
      DeformableConv2DParams* p);

  Status operator()(OpKernelContext* context);

  Tensor output_grad_tensor;
  Tensor input_grad_tensor;
  Tensor filter_grad_tensor;
  Tensor bias_grad_tensor;
  Tensor offset_grad_tensor;
  Tensor mask_grad_tensor;
};

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_
