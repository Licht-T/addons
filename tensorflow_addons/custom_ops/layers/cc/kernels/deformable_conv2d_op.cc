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
//#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace addons {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename Dtype>
struct DeformableConv2DFunctor<CPUDevice, Dtype>
    : public DeformableConv2DFunctorBase<CPUDevice, Dtype> {
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::input_tensor;
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::filter_tensor;
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::bias_tensor;
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::offset_tensor;
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::mask_tensor;
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::column_buffer_tensor;
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::p;

  DeformableConv2DFunctor(const Tensor *_input_tensor, const Tensor *_filter_tensor,
                          const Tensor *_bias_tensor, const Tensor *_offset_tensor,
                          const Tensor *_mask_tensor, Tensor *_column_buffer_tensor,
                          Tensor *_output_tensor, DeformableConv2DParams *_p)
      : DeformableConv2DFunctorBase<CPUDevice, Dtype>(
            _input_tensor, _filter_tensor, _bias_tensor, _offset_tensor,
            _mask_tensor, _column_buffer_tensor, _p),
        output_tensor(_output_tensor->dtype()) {
    CHECK(output_tensor.CopyFrom(*_output_tensor, _output_tensor->shape()));
    output_tensor.tensor<Dtype, 5>().setZero();
  }

  Status operator()(OpKernelContext *context) {
    // input_channels * filter_rows * filter_cols / weight_groups ==
    // filter_channels * filter_rows * filter_cols
    const auto elems = p.filter_channels * p.filter_rows * p.filter_cols;
    const auto rows = p.output_channels / p.weight_groups;
    const auto cols = p.parallel_imgs * p.output_rows * p.output_cols;

    Tensor output_tensor_reshaped(output_tensor.dtype());
    CHECK(output_tensor_reshaped.CopyFrom(
        output_tensor,
        TensorShape({p.batches, p.weight_groups,
                     p.output_channels / p.weight_groups,
                     p.parallel_imgs * p.output_rows, p.output_cols})));

    Tensor column_buffer_tensor_reshaped(column_buffer_tensor.dtype());
    CHECK(column_buffer_tensor_reshaped.CopyFrom(
        column_buffer_tensor, TensorShape({p.weight_groups, elems, cols})));

    for (auto b = 0; b < p.batches; b++) {
      this->DeformableIm2Col(b);

      for (auto g = 0; g < p.weight_groups; g++) {
        auto filter_mtx =
            filter_tensor.SubSlice(g).template shaped<Dtype, 2>({rows, elems});
        auto column_buffer_mtx =
            column_buffer_tensor_reshaped.SubSlice(g).tensor<Dtype, 2>();

        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
            Eigen::IndexPair<int>(1, 0)};

        EigenTensor<Dtype, 2> mul =
            filter_mtx.contract(column_buffer_mtx, product_dims);

        output_tensor_reshaped.SubSlice(b).SubSlice(g).shaped<Dtype, 2>(
            {rows, cols}) += mul;
      }
    }

    auto output_tensor_transposed =
        output_tensor_reshaped.tensor<Dtype, 5>()
            .reshape(Shape5D({p.batches, p.output_channels, p.parallel_imgs,
                              p.output_rows, p.output_cols}))
            .shuffle(Shape5D({0, 2, 1, 3, 4}))
            .reshape(Shape4D({p.input_batches, p.output_channels, p.output_rows,
                              p.output_cols}));

    output_tensor.tensor<Dtype, 4>() = output_tensor_transposed.eval();

    if (p.use_bias) {
      auto bias_tensor_broadcasted =
          bias_tensor.template tensor<Dtype, 1>()
              .reshape(Shape4D({1, p.output_channels, 1, 1}))
              .broadcast(
                  Shape4D({p.input_batches, 1, p.output_rows, p.output_cols}));

      output_tensor.tensor<Dtype, 4>() += bias_tensor_broadcasted;
    }

    return Status::OK();
  }

  Tensor output_tensor;
};

template <typename Dtype>
struct DeformableConv2DGradFunctor<CPUDevice, Dtype>
    : public DeformableConv2DFunctorBase<CPUDevice, Dtype> {
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::input_tensor;
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::filter_tensor;
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::bias_tensor;
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::offset_tensor;
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::mask_tensor;
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::column_buffer_tensor;
  using DeformableConv2DFunctorBase<CPUDevice, Dtype>::p;

  DeformableConv2DGradFunctor(
      const Tensor *_input_tensor, const Tensor *_filter_tensor, const Tensor *_bias_tensor,
      const Tensor *_offset_tensor, const Tensor *_mask_tensor, Tensor *_output_grad_tensor,
      Tensor *_input_grad_tensor, Tensor *_filter_grad_tensor,
      Tensor *_bias_grad_tensor, Tensor *_offset_grad_tensor,
      Tensor *_mask_grad_tensor, Tensor *_column_buffer_tensor,
      DeformableConv2DParams *_p)
      : DeformableConv2DFunctorBase<CPUDevice, Dtype>(
            _input_tensor, _filter_tensor, _bias_tensor, _offset_tensor,
            _mask_tensor, _column_buffer_tensor, _p),
        output_grad_tensor(_output_grad_tensor->dtype()),
        input_grad_tensor(_input_grad_tensor->dtype()),
        filter_grad_tensor(_filter_grad_tensor->dtype()),
        bias_grad_tensor(_bias_grad_tensor->dtype()),
        offset_grad_tensor(_offset_grad_tensor->dtype()),
        mask_grad_tensor(_mask_grad_tensor->dtype()) {
    CHECK(output_grad_tensor.CopyFrom(
        *_output_grad_tensor,
        TensorShape({p.batches, p.parallel_imgs, p.output_channels,
                     p.output_rows, p.output_cols})));
    CHECK(input_grad_tensor.CopyFrom(*_input_grad_tensor,
                                     _input_grad_tensor->shape()));
    CHECK(filter_grad_tensor.CopyFrom(
        *_filter_grad_tensor,
        TensorShape({p.weight_groups, p.output_channels / p.weight_groups,
                     p.filter_channels, p.filter_rows, p.filter_cols})));
    CHECK(bias_grad_tensor.CopyFrom(*_bias_grad_tensor,
                                    _bias_grad_tensor->shape()));
    CHECK(offset_grad_tensor.CopyFrom(*_offset_grad_tensor,
                                      _offset_grad_tensor->shape()));
    CHECK(mask_grad_tensor.CopyFrom(*_mask_grad_tensor,
                                    _mask_grad_tensor->shape()));

    input_grad_tensor.tensor<Dtype, 4>().setZero();
    filter_grad_tensor.tensor<Dtype, 5>().setZero();
    column_buffer_tensor.template tensor<Dtype, 4>().setZero();
  }

  Status operator()(OpKernelContext *context) {
    ComputeInputOffsetMaskGrad();

    ComputeFilterGrad();

    if (p.use_bias) {
      auto bias_grad_eigen_tensor = bias_grad_tensor.tensor<Dtype, 1>();
      auto output_grad_eigen_tensor = output_grad_tensor.shaped<Dtype, 5>(
          {p.batches, p.output_channels, p.parallel_imgs, p.output_rows,
           p.output_cols});

      bias_grad_eigen_tensor.setConstant(Dtype(1));
      bias_grad_eigen_tensor *=
          output_grad_eigen_tensor.sum(Eigen::array<int, 4>({0, 2, 3, 4}));
    }

    return Status::OK();
  }

  void ComputeFilterGrad() {
    Tensor output_grad_tensor_reshaped(output_grad_tensor.dtype());
    CHECK(output_grad_tensor_reshaped.CopyFrom(
        output_grad_tensor,
        TensorShape({p.batches, p.weight_groups,
                     p.output_channels / p.weight_groups,
                     p.parallel_imgs * p.output_rows, p.output_cols})));

    // input_channels * filter_rows * filter_cols / weight_groups ==
    // filter_channels * filter_rows * filter_cols
    const auto elems = p.filter_channels * p.filter_rows * p.filter_cols;
    const auto rows = p.output_channels / p.weight_groups;
    const auto cols = p.parallel_imgs * p.output_rows * p.output_cols;

    // FIXME: 事前にTranspose
    Tensor column_buffer_tensor_reshaped(column_buffer_tensor.dtype());
    CHECK(column_buffer_tensor_reshaped.CopyFrom(
        column_buffer_tensor, TensorShape({p.weight_groups, elems, cols})));

    for (auto b = 0; b < p.batches; b++) {
      this->DeformableIm2Col(b);

      for (auto g = 0; g < p.weight_groups; g++) {
        EigenTensor<Dtype, 2> column_buffer_mtx =
            column_buffer_tensor_reshaped.SubSlice(g)
                .tensor<Dtype, 2>()
                .shuffle(Shape2D({1, 0}));

        auto output_grad_mtx = output_grad_tensor_reshaped.SubSlice(b)
                                   .SubSlice(g)
                                   .shaped<Dtype, 2>({rows, cols});

        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
            Eigen::IndexPair<int>(1, 0)};

        filter_grad_tensor.SubSlice(g).shaped<Dtype, 2>({rows, elems}) +=
            output_grad_mtx.contract(column_buffer_mtx, product_dims);
      }
    }
  }

  void ComputeInputOffsetMaskGrad() {
    Tensor output_grad_tensor_reshaped(output_grad_tensor.dtype());
    CHECK(output_grad_tensor_reshaped.CopyFrom(
        output_grad_tensor,
        TensorShape({p.batches, p.weight_groups,
                     p.output_channels / p.weight_groups,
                     p.parallel_imgs * p.output_rows, p.output_cols})));

    // input_channels * filter_rows * filter_cols / weight_groups ==
    // filter_channels * filter_rows * filter_cols
    const auto rows = p.filter_channels * p.filter_rows * p.filter_cols;
    const auto elems = p.output_channels / p.weight_groups;
    const auto cols = p.parallel_imgs * p.output_rows * p.output_cols;

    Tensor column_buffer_tensor_reshaped(column_buffer_tensor.dtype());
    CHECK(column_buffer_tensor_reshaped.CopyFrom(
        column_buffer_tensor, TensorShape({p.weight_groups, elems, cols})));

    for (auto b = 0; b < p.batches; b++) {
      for (int g = 0; g < p.weight_groups; g++) {
        // FIXME: 事前にTranspose
        EigenTensor<Dtype, 2> filter_mtx =
            filter_tensor.SubSlice(g)
                .template shaped<Dtype, 2>({elems, rows})
                .shuffle(Shape2D({1, 0}));
        auto output_grad_mtx = output_grad_tensor_reshaped.SubSlice(b)
                                   .SubSlice(g)
                                   .shaped<Dtype, 2>({elems, cols});

        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
            Eigen::IndexPair<int>(1, 0)};

        column_buffer_tensor_reshaped.SubSlice(g).tensor<Dtype, 2>() =
            filter_mtx.contract(output_grad_mtx, product_dims);
      }

      DeformableCol2ImForOffsetAndMask(b);

      DeformableCol2ImForInput(b);
    }
  }

  void DeformableCol2ImForOffsetAndMask(int32 b) {
    auto num_kernels = p.output_rows * p.output_cols * 2 * p.filter_rows *
                       p.filter_cols * p.offset_groups * p.parallel_imgs;
    auto offset_channels = 2 * p.filter_rows * p.filter_cols * p.offset_groups;

    const auto offset_eigen_tensor =
        offset_tensor.SubSlice(b).template tensor<Dtype, 7>();

    const auto mask_eigen_tensor =
        p.use_mask ? mask_tensor.SubSlice(b).template tensor<Dtype, 6>()
                   : mask_tensor.template shaped<Dtype, 6>({0, 0, 0, 0, 0, 0});

    const auto column_buffer_eigen_tensor =
        column_buffer_tensor.template shaped<Dtype, 6>(
            {p.input_channels, p.filter_rows, p.filter_cols, p.parallel_imgs,
             p.output_rows, p.output_cols});

    for (auto k = 0; k < num_kernels; k++) {
      auto offset_grad_value = Dtype(0);
      auto mask_grad_value = Dtype(0);

      const auto offset_channel_step = p.filter_rows * p.filter_cols;

      const auto current_output_col = k % p.output_cols;
      const auto current_output_row = (k / p.output_cols) % p.output_rows;
      const auto current_filter_col =
          (k / (2 * p.output_rows * p.output_cols)) % p.filter_cols;
      const auto current_filter_row =
          (k / (2 * p.output_rows * p.output_cols * p.filter_cols)) %
          p.filter_rows;
      const auto current_offset_channel =
          (k / (p.output_rows * p.output_cols)) % offset_channels;
      const auto current_batch =
          k / (p.output_rows * p.output_cols * offset_channels);

      auto current_actual_batch = b * p.parallel_imgs + current_batch;

      const auto current_offset_group =
          current_offset_channel / (2 * offset_channel_step);

      const auto channels_per_offset_group = p.input_channels / p.offset_groups;
      const auto offset_channel_diff =
          current_offset_channel -
          current_offset_group * 2 * offset_channel_step;
      const auto is_y_direction = offset_channel_diff % 2 == 0;

      for (auto selected_offset_channel = (offset_channel_diff / 2);
           selected_offset_channel <
           channels_per_offset_group * offset_channel_step;
           selected_offset_channel += offset_channel_step) {
        const auto selected_filter_col =
            selected_offset_channel % p.filter_cols;
        const auto selected_filter_row =
            (selected_offset_channel / p.filter_cols) % p.filter_rows;
        const auto input_channel_diff =
            (selected_offset_channel / (p.filter_cols * p.filter_rows));

        const auto offset_h = offset_eigen_tensor(
            current_batch, current_offset_group, selected_filter_row,
            selected_filter_col, 0, current_output_row, current_output_col);
        const auto offset_w = offset_eigen_tensor(
            current_batch, current_offset_group, selected_filter_row,
            selected_filter_col, 1, current_output_row, current_output_col);
        const auto mask =
            p.use_mask
                ? mask_eigen_tensor(current_batch, current_offset_group,
                                    selected_filter_row, selected_filter_col,
                                    current_output_row, current_output_col)
                : Dtype(1);

        const auto y = (current_output_row * p.stride_rows - p.padding_rows) +
                       selected_filter_row * p.dilation_rows + offset_h;
        const auto x = (current_output_col * p.stride_cols - p.padding_cols) +
                       selected_filter_col * p.dilation_cols + offset_w;

        const auto selected_input_channel =
            input_channel_diff +
            current_offset_group * channels_per_offset_group;

        auto filter_data = column_buffer_eigen_tensor(
            selected_input_channel, selected_filter_row, selected_filter_col,
            current_batch, current_output_row, current_output_col);

        const auto weight =
            GetCoordinateWeight(b, current_actual_batch, selected_input_channel,
                                y, x, is_y_direction);

        offset_grad_value += mask * weight * filter_data;

        if (is_y_direction) {
          mask_grad_value += filter_data * this->BilinearInterpolate(
                                               b, current_actual_batch,
                                               selected_input_channel, y, x);
        }
      }

      offset_grad_tensor.tensor<Dtype, 4>()(
          current_actual_batch, current_offset_channel, current_output_row,
          current_output_col) = offset_grad_value;

      if (p.use_mask && is_y_direction) {
        auto current_mask_channel =
            (current_offset_group * p.filter_rows + current_filter_row) *
                p.filter_cols +
            current_filter_col;

        mask_grad_tensor.tensor<Dtype, 4>()(
            current_actual_batch, current_mask_channel, current_output_row,
            current_output_col) = mask_grad_value;
      }
    }
  }

  void DeformableCol2ImForInput(int32 b) {
    auto num_kernels = p.input_channels * p.filter_rows * p.filter_cols *
                       p.output_rows * p.output_cols * p.parallel_imgs;

    const auto offset_eigen_tensor =
        offset_tensor.SubSlice(b).template tensor<Dtype, 7>();

    const auto mask_eigen_tensor =
        p.use_mask ? mask_tensor.SubSlice(b).template tensor<Dtype, 6>()
                   : mask_tensor.template shaped<Dtype, 6>({0, 0, 0, 0, 0, 0});

    const auto column_buffer_tensor_flattened =
        column_buffer_tensor.template shaped<Dtype, 1>({num_kernels});

    for (auto k = 0; k < num_kernels; k++) {
      const auto current_output_col = k % p.output_cols;
      const auto current_output_row = (k / p.output_cols) % p.output_rows;
      const auto current_batch =
          (k / (p.output_rows * p.output_cols)) % p.parallel_imgs;

      const auto current_filter_col =
          (k / (p.output_rows * p.output_cols * p.parallel_imgs)) %
          p.filter_cols;
      const auto current_filter_row = (k / (p.output_rows * p.output_cols *
                                            p.parallel_imgs * p.filter_cols)) %
                                      p.filter_rows;
      const auto current_channel =
          k / (p.output_rows * p.output_cols * p.parallel_imgs * p.filter_rows *
               p.filter_cols);

      const auto current_offset_group =
          current_channel / (p.input_channels / p.offset_groups);

      auto mask =
          p.use_mask ? mask_eigen_tensor(current_batch, current_offset_group,
                                         current_filter_row, current_filter_col,
                                         current_output_row, current_output_col)
                     : Dtype(1);

      auto offset_h = offset_eigen_tensor(
          current_batch, current_offset_group, current_filter_row,
          current_filter_col, 0, current_output_row, current_output_col);
      auto offset_w = offset_eigen_tensor(
          current_batch, current_offset_group, current_filter_row,
          current_filter_col, 1, current_output_row, current_output_col);

      const auto y = (current_output_row * p.stride_rows - p.padding_rows) +
                     current_filter_row * p.dilation_rows + offset_h;
      const auto x = (current_output_col * p.stride_cols - p.padding_cols) +
                     current_filter_col * p.dilation_cols + offset_w;

      for (auto dy = -1; dy <= 1; dy++) {
        for (auto dx = -1; dx <= 1; dx++) {
          auto current_input_row = int(y) + dy;
          auto current_input_col = int(x) + dx;
          if (p.input_rows > current_input_row && current_input_row >= 0 &&
              p.input_cols > current_input_col && current_input_col >= 0 &&
              std::abs(y - current_input_row) < 1 &&
              std::abs(x - current_input_col) < 1) {
            auto weight = (1.0 - std::abs(y - current_input_row)) *
                          (1.0 - std::abs(x - current_input_col));

            auto current_actual_batch = b * p.parallel_imgs + current_batch;

            input_grad_tensor.tensor<Dtype, 4>()(
                current_actual_batch, current_channel, current_input_row,
                current_input_col) +=
                mask * weight * column_buffer_tensor_flattened(k);
          }
        }
      }
    }
  }

  Dtype GetCoordinateWeight(int32 b, int32 batch, int32 channel, Dtype y,
                            Dtype x, bool is_y_direction) {
    const auto img = input_tensor.SubSlice(b)
                         .SubSlice(batch)
                         .SubSlice(channel)
                         .template tensor<Dtype, 2>();

    auto max_height = img.dimension(0);
    auto max_width = img.dimension(1);

    int y_low = floor(y);
    int x_low = floor(x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    bool valid_y_low = max_height > y_low && y_low >= 0;
    bool valid_y_high = max_height > y_high && y_high >= 0;
    bool valid_x_low = max_width > x_low && x_low >= 0;
    bool valid_x_high = max_width > x_high && x_high >= 0;

    auto v_yx = Dtype(0);
    if (valid_y_low && valid_x_low) {
      v_yx = img(y_low, x_low);
    }

    auto v_yX = Dtype(0);
    if (valid_y_low && valid_x_high) {
      v_yX = img(y_low, x_high);
    }

    auto v_Yx = Dtype(0);
    if (valid_y_high && valid_x_low) {
      v_Yx = img(y_high, x_low);
    }

    auto v_YX = Dtype(0);
    if (valid_y_high && valid_x_high) {
      v_YX = img(y_high, x_high);
    }

    if (is_y_direction) {
      auto dx = x - x_low;
      return (v_YX - v_yX) * dx + (v_Yx - v_yx) * (1 - dx);
    } else {
      auto dy = y - y_low;
      return (v_YX - v_Yx) * dy + (v_yX - v_yx) * (1 - dy);
    }
  }

  Tensor output_grad_tensor;
  Tensor input_grad_tensor;
  Tensor filter_grad_tensor;
  Tensor bias_grad_tensor;
  Tensor offset_grad_tensor;
  Tensor mask_grad_tensor;
};

}  // end namespace functor

template <typename Device, typename T>
class DeformableConv2DOpBase : public OpKernel {
 public:
  explicit DeformableConv2DOpBase(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
    OP_REQUIRES_OK(context, context->GetAttr("weight_groups", &weight_groups));
    OP_REQUIRES_OK(context, context->GetAttr("offset_groups", &offset_groups));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations));
    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    FormatFromString(data_format_str, &data_format);

    p = DeformableConv2DParams{};
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    const Tensor &filter_tensor = context->input(1);

    const Tensor &bias_tensor = context->input(2);
    const Tensor &mask_tensor = context->input(4);

    const TensorShape &input_shape = input_tensor.shape();
    const TensorShape &filter_shape = filter_tensor.shape();

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

    auto parallel_imgs = GetParallelImgs(input_batches);

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
    p.batches = p.input_batches / p.parallel_imgs;
    p.use_mask = mask_tensor.NumElements() > 0;
    p.use_bias = bias_tensor.NumElements() > 0;
  }

  int GetParallelImgs(int n) {
    for (auto k = kMaxParallelImgs; k > 1; --k) {
      if (n % k == 0) {
        return k;
      }
    }
    return 1;
  }

 protected:
  TensorFormat data_format;
  DeformableConv2DParams p;

 private:
  std::vector<int32> strides;
  int32 weight_groups;
  int32 offset_groups;
  Padding padding;
  std::vector<int32> dilations;
};

template <typename Device, typename T>
class DeformableConv2DOp : public DeformableConv2DOpBase<Device, T> {
  using DeformableConv2DOpBase<Device, T>::data_format;
  using DeformableConv2DOpBase<Device, T>::p;

 public:
  explicit DeformableConv2DOp(OpKernelConstruction *context)
      : DeformableConv2DOpBase<Device, T>(context) {}

  void Compute(OpKernelContext *context) override {
    DeformableConv2DOpBase<Device, T>::Compute(context);

    const Tensor &input_tensor = context->input(0);
    const Tensor &filter_tensor = context->input(1);
    const Tensor &bias_tensor = context->input(2);
    const Tensor &offset_tensor = context->input(3);
    const Tensor &mask_tensor = context->input(4);

    TensorShape column_buffer_shape(
        {p.input_channels * p.filter_rows * p.filter_cols, p.parallel_imgs,
         p.output_rows, p.output_cols});
    Tensor column_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   column_buffer_shape,
                                                   &column_buffer_tensor));

    TensorShape output_shape =
        ShapeFromFormat(data_format, p.input_batches, p.output_rows,
                        p.output_cols, p.output_channels);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    functor::DeformableConv2DFunctor<Device, T> deformableConv2DFunc(
        &input_tensor, &filter_tensor,
        &bias_tensor, &offset_tensor,
        &mask_tensor, &column_buffer_tensor,
        output_tensor, &p);
    Status s = deformableConv2DFunc(context);

    OP_REQUIRES_OK(context, s);
  }
};

template <typename Device, typename T>
class DeformableConv2DGradOp : public DeformableConv2DOpBase<Device, T> {
  using DeformableConv2DOpBase<Device, T>::data_format;
  using DeformableConv2DOpBase<Device, T>::p;

 public:
  explicit DeformableConv2DGradOp(OpKernelConstruction *context)
      : DeformableConv2DOpBase<Device, T>(context) {}

  void Compute(OpKernelContext *context) override {
    DeformableConv2DOpBase<Device, T>::Compute(context);

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

    TensorShape column_buffer_shape(
        {p.input_channels * p.filter_rows * p.filter_cols, p.parallel_imgs,
         p.output_rows, p.output_cols});
    Tensor column_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   column_buffer_shape,
                                                   &column_buffer_tensor));

    Tensor output_grad_tensor_reshaped;
    CHECK(output_grad_tensor_reshaped.CopyFrom(
        output_grad_tensor,
        TensorShape({p.batches, p.parallel_imgs, p.output_channels,
                     p.output_rows, p.output_cols})));

    TensorShape output_grad_tensor_transposed_shape(
        {p.batches, p.parallel_imgs, p.output_channels, p.output_rows,
         p.output_cols});
    Tensor output_grad_tensor_transposed;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          output_grad_tensor_transposed_shape,
                                          &output_grad_tensor_transposed));
    OP_REQUIRES_OK(
        context, DoTranspose(context->device(), output_grad_tensor_reshaped,
                             {0, 2, 1, 3, 4}, &output_grad_tensor_transposed));

    TensorShape output_shape =
        ShapeFromFormat(data_format, p.input_batches, p.output_rows,
                        p.output_cols, p.output_channels);

    Tensor *input_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, input_shape, &input_grad_tensor));
    Tensor *filter_grad_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, filter_shape,
                                                     &filter_grad_tensor));
    Tensor *bias_grad_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, bias_shape, &bias_grad_tensor));
    Tensor *offset_grad_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(3, offset_shape,
                                                     &offset_grad_tensor));
    Tensor *mask_grad_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(4, mask_shape, &mask_grad_tensor));

    functor::DeformableConv2DGradFunctor<Device, T> deformableConv2DGradFunc(
        &input_tensor, &filter_tensor, &bias_tensor, &offset_tensor,
        &mask_tensor, &output_grad_tensor_transposed, input_grad_tensor,
        filter_grad_tensor, bias_grad_tensor, offset_grad_tensor,
        mask_grad_tensor, &column_buffer_tensor, &p);
    Status s = deformableConv2DGradFunc(context);

    OP_REQUIRES_OK(context, s);
  }
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
TF_CALL_double(REGISTER_DEFORMABLECONV2D_OP_CPU);
#undef REGISTER_DEFORMABLECONV2D_OP_CPU

}  // namespace addons
}  // namespace tensorflow
