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
using Shape8D = Eigen::array<Eigen::DenseIndex, 8>;
using Shape7D = Eigen::array<Eigen::DenseIndex, 7>;
using Shape6D = Eigen::array<Eigen::DenseIndex, 6>;
using Shape5D = Eigen::array<Eigen::DenseIndex, 5>;
using Shape4D = Eigen::array<Eigen::DenseIndex, 4>;
using Shape3D = Eigen::array<Eigen::DenseIndex, 3>;
using Shape2D = Eigen::array<Eigen::DenseIndex, 2>;
using Shape1D = Eigen::array<Eigen::DenseIndex, 1>;

template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
using EigenTensor = Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>;
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
using EigenConstTensor =
    Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>;

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
};

namespace functor {

template <typename Device, typename T>
struct DeformableConv2DFunctor {
  Status operator()(OpKernelContext* context,
                    typename TTypes<T, 4>::ConstTensor input_tensor,
                    typename TTypes<T, 4>::ConstTensor filter_tensor,
                    typename TTypes<T, 1>::ConstTensor bias_tensor,
                    typename TTypes<T, 4>::ConstTensor offset_tensor,
                    typename TTypes<T, 4>::ConstTensor mask_tensor,
                    typename TTypes<T, 2>::Tensor column_buffer_tensor,
                    typename TTypes<T, 4>::Tensor output_tensor,
                    DeformableConv2DParams& p);
};

template <typename Device, typename T>
struct DeformableConv2DGradFunctor {
  Status operator()(OpKernelContext* context,
                    typename TTypes<T, 4>::ConstTensor input_tensor,
                    typename TTypes<T, 4>::ConstTensor filter_tensor,
                    typename TTypes<T, 1>::ConstTensor bias_tensor,
                    typename TTypes<T, 4>::ConstTensor offset_tensor,
                    typename TTypes<T, 4>::ConstTensor mask_tensor,
                    typename TTypes<T, 4>::ConstTensor output_grad_tensor,
                    typename TTypes<T, 4>::Tensor input_grad_tensor,
                    typename TTypes<T, 4>::Tensor filter_grad_tensor,
                    typename TTypes<T, 1>::Tensor bias_grad_tensor,
                    typename TTypes<T, 4>::Tensor offset_grad_tensor,
                    typename TTypes<T, 4>::Tensor mask_grad_tensor,
                    typename TTypes<T, 4>::Tensor column_buffer_tensor,
                    DeformableConv2DParams& p);
};

}  // namespace functor

static inline int get_parallel_imgs(int n) {
  for (auto k = kMaxParallelImgs; k > 1; --k) {
    if (n % k == 0) {
      return k;
    }
  }
  return 1;
}

}  // namespace addons
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_
