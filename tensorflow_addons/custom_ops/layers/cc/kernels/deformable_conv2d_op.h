/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLE_CONV_OP_H_
#define TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLE_CONV_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace addons {
namespace functor {

template <typename Device, typename T>
struct DeformableConv2DFunctor {
  Status operator()(OpKernelContext* context, typename TTypes<T, 4>::ConstTensor input_tensor,
                    typename TTypes<T, 4>::ConstTensor filter_tensor, typename TTypes<T, 1>::ConstTensor bias_tensor,
                    typename TTypes<T, 4>::ConstTensor offset_tensor, typename TTypes<T, 4>::ConstTensor mask_tensor,
                    typename TTypes<T, 2>::Tensor column_buffer_tensor, typename TTypes<T, 4>::Tensor output_tensor,
                    /* params */
                    int32 input_batches,
                    int32 input_channels,
                    int32 input_rows,
                    int32 input_cols,
                    int32 filter_channels,
                    int32 filter_rows,
                    int32 filter_cols,
                    int32 padding_rows,
                    int32 padding_cols,
                    int32 stride_rows,
                    int32 stride_cols,
                    int32 dilation_rows,
                    int32 dilation_cols,
                    int32 output_channels,
                    int32 output_rows,
                    int32 output_cols,
                    int32 n_parallel_imgs,
                    int32 weight_groups,
                    int32 offset_groups
                    );
};

//template <typename Device, typename T>
//struct DeformableConv2DGradFunctor {
//  Status operator()(OpKernelContext* context, const Tensor& input_a_t,
//                    const Tensor& input_b_t, const Tensor& topdiff_t,
//                    Tensor* output_a_gradient_t, Tensor* output_b_gradient_t,
//                    /* params */
//                    std::vector<int32> strides, int32 weight_groups,
//                    int32 offset_groups, bool no_bias,
//                    Padding padding, std::vector<int32> dilations,
//                    TensorFormat data_format, int32 n_parallel_imgs
//  );
//};

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLE_CONV_OP_H_
