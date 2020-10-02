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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/work_sharder.h"

#include "tensorflow_addons/custom_ops/layers/cc/kernels/deformable_conv2d_op.h"

namespace tensorflow {
    namespace addons {

        using CPUDevice = Eigen::ThreadPoolDevice ;
        using GPUDevice = Eigen::GpuDevice;

        using Shape5D = Eigen::array<Eigen::DenseIndex, 5>;
        using Shape3D = Eigen::array<Eigen::DenseIndex, 3>;
        using Shape2D = Eigen::array<Eigen::DenseIndex, 2>;

        template<typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex> using EigenTensor = Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>;

        static const int kMaxParallelImgs = 32;

        static int get_greatest_divisor_below_bound(int n) {
            for (int k = kMaxParallelImgs; k > 1; --k) {
                if (n % k == 0) {
                    return k;
                }
            }
            return 1;
        }

        namespace functor {
            template<typename Dtype>
            struct DeformableConv2DFunctor<CPUDevice, Dtype> {
                Status operator()(OpKernelContext *context, typename TTypes<Dtype, 4>::ConstTensor input_tensor,
                                  typename TTypes<Dtype, 4>::ConstTensor filter_tensor,
                                  typename TTypes<Dtype, 1>::ConstTensor bias_tensor,
                                  typename TTypes<Dtype, 4>::ConstTensor offset_tensor,
                                  typename TTypes<Dtype, 4>::ConstTensor mask_tensor,
                                  typename TTypes<Dtype, 2>::Tensor column_buffer_tensor,
                                  typename TTypes<Dtype, 4>::Tensor output_tensor,
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
                ) {
                    auto batches = input_batches / n_parallel_imgs;
                    auto input_tensor_reshaped = input_tensor.reshape(
                            Shape5D({batches, n_parallel_imgs, input_channels, input_rows, input_cols})
                    );
                    auto filter_tensor_reshaped = filter_tensor.reshape(
                            Shape5D({weight_groups, output_channels / weight_groups, filter_channels, filter_rows,
                                     filter_cols})
                    );

                    auto filter_area = filter_rows * filter_cols * offset_groups;
                    auto offset_tensor_reshaped = offset_tensor.reshape(
                            Shape5D({batches, n_parallel_imgs, 2 * filter_area, output_rows, output_cols})
                    );
                    auto mask_tensor_reshaped = mask_tensor.reshape(
                            Shape5D({batches, n_parallel_imgs, filter_area, output_rows, output_cols})
                    );

                    auto output_tensor_reshaped = output_tensor.reshape(
                            Shape5D({batches, weight_groups, output_channels / weight_groups,
                                     n_parallel_imgs * output_rows, output_cols})
                    );

                    // input_channels * filter_rows * filter_cols / weight_groups == filter_channels * filter_rows * filter_cols
                    int32 elems = filter_channels * filter_rows * filter_cols;
                    int32 cols = n_parallel_imgs * output_rows * output_cols;

                    auto column_buffer_tensor_reshaped = column_buffer_tensor.reshape(
                            Shape3D({weight_groups, elems, cols}));

                    for (int32 b = 0; b < batches; b++) {
                        auto input_tensor_reshaped_batch = input_tensor_reshaped.chip(b, 0);
                        auto offset_tensor_reshaped_batch = offset_tensor_reshaped.chip(b, 0);
                        auto output_tensor_reshaped_batch = output_tensor_reshaped.chip(b, 0);

                        for (int g = 0; g < weight_groups; g++) {
                            int32 rows = output_channels / weight_groups;

                            EigenTensor<Dtype, 2> filter_mtx = filter_tensor_reshaped.chip(g, 0).reshape(Shape2D({rows, elems}));
                            EigenTensor<Dtype, 2> column_buffer_mtx = column_buffer_tensor_reshaped.chip(g, 0);

                            auto mtx_shape = Shape2D({rows, cols});
                            Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };

                            EigenTensor<Dtype, 2> output_mtx = output_tensor_reshaped_batch.chip(g, 0).reshape(
                                    mtx_shape);
                            EigenTensor<Dtype, 2> mul = filter_mtx.contract(column_buffer_mtx, product_dims);

                            output_mtx = output_mtx + mul;
                        }
                    }
                }
            };

        }  // end namespace functor

        template<typename Device, typename T>
        class DeformableConv2DOp : public OpKernel {
        public:
            explicit DeformableConv2DOp(OpKernelConstruction *context)
                    : OpKernel(context) {

                OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
                OP_REQUIRES_OK(context, context->GetAttr("weight_groups", &weight_groups));
                OP_REQUIRES_OK(context, context->GetAttr("offset_groups", &offset_groups));
                OP_REQUIRES_OK(context, context->GetAttr("no_bias", &no_bias));
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

                auto n_parallel_imgs = get_greatest_divisor_below_bound(input_batches);

                auto filter_rows_eff = filter_rows + (filter_rows - 1) * (dilation_rows - 1);
                auto filter_cols_eff = filter_cols + (filter_cols - 1) * (dilation_cols - 1);

                int64 output_rows, output_cols;
                int64 padding_rows, padding_cols;
                OP_REQUIRES_OK(context,
                               GetWindowedOutputSizeV2(input_rows, filter_rows, dilation_rows, stride_rows, padding,
                                                       &output_rows, &padding_rows));
                OP_REQUIRES_OK(context,
                               GetWindowedOutputSizeV2(input_cols, filter_cols, dilation_cols, stride_cols, padding,
                                                       &output_cols, &padding_cols));

                TensorShape column_buffer_shape(
                        {input_channels * filter_rows * filter_cols, n_parallel_imgs * output_rows * output_cols});
                Tensor column_buffer_tensor;
                OP_REQUIRES_OK(context,
                               context->allocate_temp(DataTypeToEnum<T>::value,
                                                      column_buffer_shape, &column_buffer_tensor));

                TensorShape output_shape = ShapeFromFormat(
                        data_format, input_batches,
                        output_rows, output_cols, output_channels
                );
                Tensor *output_tensor = nullptr;
                OP_REQUIRES_OK(context,
                               context->allocate_output(0, output_shape, &output_tensor));

                functor::DeformableConv2DFunctor<Device, T> deformableConv2DFunc;
                Status s = deformableConv2DFunc(context, input_tensor.tensor<T, 4>(),
                                                filter_tensor.tensor<T, 4>(), bias_tensor.tensor<T, 1>(),
                                                offset_tensor.tensor<T, 4>(), mask_tensor.tensor<T, 4>(),
                                                column_buffer_tensor.tensor<T, 2>(), output_tensor->tensor<T, 4>(),
                                                input_batches,
                                                input_channels,
                                                input_rows,
                                                input_cols,
                                                filter_channels,
                                                filter_rows,
                                                filter_cols,
                                                padding_rows,
                                                padding_cols,
                                                stride_rows,
                                                stride_cols,
                                                dilation_rows,
                                                dilation_cols,
                                                output_channels,
                                                output_rows,
                                                output_cols,
                                                n_parallel_imgs,
                                                weight_groups,
                                                offset_groups);

                OP_REQUIRES_OK(context, s);
            }

        private:
            std::vector<int32> strides;
            int32 weight_groups;
            int32 offset_groups;
            bool no_bias;
            Padding padding;
            std::vector<int32> dilations;
            TensorFormat data_format;
        };

//template <typename Device, typename T>
//class DeformableConv2DGradOp : public OpKernel {
// public:
//  explicit DeformableConv2DGradOp(OpKernelConstruction* context)
//      : OpKernel(context) {
//      OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
//      OP_REQUIRES_OK(context, context->GetAttr("weight_groups", &weight_groups));
//      OP_REQUIRES_OK(context, context->GetAttr("offset_groups", &offset_groups));
//      OP_REQUIRES_OK(context, context->GetAttr("no_bias", &no_bias));
//      OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
//      OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations));
//      std::string data_format_str;
//      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
//      FormatFromString(data_format_str, &data_format);
//  }
//
//  void Compute(OpKernelContext* context) override {
//      const Tensor& input_tensor = context->input(0);
//      const Tensor& filter_tensor = context->input(1);
//      const Tensor& bias_tensor = context->input(2);
//      const Tensor& offset_tensor = context->input(3);
//      const Tensor& mask_tensor = context->input(4);
//
//      const TensorShape& input_shape = input_tensor.shape();
//      const TensorShape& filter_shape = filter_tensor.shape();
//      const TensorShape& bias_shape = bias_tensor.shape();
//      const TensorShape& offset_shape = offset_tensor.shape();
//      const TensorShape& mask_shape = mask_tensor.shape();
//  }
//
// private:
//    std::vector<int32> strides;
//    int32 weight_groups;
//    int32 offset_groups;
//    bool no_bias;
//    Padding padding;
//    std::vector<int32> dilations;
//    TensorFormat data_format;
//};

// Register the CPU kernels.
#define REGISTER_DEFORMABLECONV2D_OP_CPU(T)                   \
  REGISTER_KERNEL_BUILDER(Name("Addons>DeformableConv2D")     \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<T>("T"),       \
                          DeformableConv2DOp<CPUDevice, T>)   //\
//  REGISTER_KERNEL_BUILDER(Name("Addons>DeformableConv2DGrad") \
//                              .Device(DEVICE_CPU)            \
//                              .TypeConstraint<T>("T"),       \
//                          DeformableConv2DGradOp<CPUDevice, T>)

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
//TF_CALL_float(REGISTER_DEFORMABLECONV2D_OP_GPU);
//#undef REGISTER_DEFORMABLECONV2D_OP_GPU
//
//#endif  // GOOGLE_CUDA

    }  // namespace addons
}  // namespace tensorflow
