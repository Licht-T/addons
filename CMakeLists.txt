cmake_minimum_required(VERSION 3.16)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(addons LANGUAGES CUDA CXX)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(addons
        tensorflow_addons/custom_ops/activations/cc/kernels/gelu_op.cc
        tensorflow_addons/custom_ops/activations/cc/kernels/gelu_op.h
        tensorflow_addons/custom_ops/activations/cc/kernels/gelu_op_gpu.cu.cc
        tensorflow_addons/custom_ops/activations/cc/kernels/hardshrink_op.cc
        tensorflow_addons/custom_ops/activations/cc/kernels/hardshrink_op.h
        tensorflow_addons/custom_ops/activations/cc/kernels/hardshrink_op_gpu.cu.cc
        tensorflow_addons/custom_ops/activations/cc/kernels/lisht_op.cc
        tensorflow_addons/custom_ops/activations/cc/kernels/lisht_op.h
        tensorflow_addons/custom_ops/activations/cc/kernels/lisht_op_gpu.cu.cc
        tensorflow_addons/custom_ops/activations/cc/kernels/mish_op.cc
        tensorflow_addons/custom_ops/activations/cc/kernels/mish_op.h
        tensorflow_addons/custom_ops/activations/cc/kernels/mish_op_gpu.cu.cc
        tensorflow_addons/custom_ops/activations/cc/kernels/softshrink_op.cc
        tensorflow_addons/custom_ops/activations/cc/kernels/softshrink_op.h
        tensorflow_addons/custom_ops/activations/cc/kernels/softshrink_op_gpu.cu.cc
        tensorflow_addons/custom_ops/activations/cc/kernels/tanhshrink_op.cc
        tensorflow_addons/custom_ops/activations/cc/kernels/tanhshrink_op.h
        tensorflow_addons/custom_ops/activations/cc/kernels/tanhshrink_op_gpu.cu.cc
        tensorflow_addons/custom_ops/activations/cc/ops/gelu_op.cc
        tensorflow_addons/custom_ops/activations/cc/ops/hardshrink_op.cc
        tensorflow_addons/custom_ops/activations/cc/ops/lisht_op.cc
        tensorflow_addons/custom_ops/activations/cc/ops/mish_op.cc
        tensorflow_addons/custom_ops/activations/cc/ops/softshrink_op.cc
        tensorflow_addons/custom_ops/activations/cc/ops/tanhshrink_op.cc
        tensorflow_addons/custom_ops/image/cc/kernels/adjust_hsv_in_yiq_op.cc
        tensorflow_addons/custom_ops/image/cc/kernels/adjust_hsv_in_yiq_op.h
        tensorflow_addons/custom_ops/image/cc/kernels/adjust_hsv_in_yiq_op_gpu.cu.cc
        tensorflow_addons/custom_ops/image/cc/kernels/connected_components.cc
        tensorflow_addons/custom_ops/image/cc/kernels/connected_components.h
        tensorflow_addons/custom_ops/image/cc/kernels/euclidean_distance_transform_op.cc
        tensorflow_addons/custom_ops/image/cc/kernels/euclidean_distance_transform_op.h
        tensorflow_addons/custom_ops/image/cc/kernels/euclidean_distance_transform_op_gpu.cu.cc
        tensorflow_addons/custom_ops/image/cc/kernels/image_projective_transform_op.cc
        tensorflow_addons/custom_ops/image/cc/kernels/image_projective_transform_op.h
        tensorflow_addons/custom_ops/image/cc/kernels/image_projective_transform_op_gpu.cu.cc
        tensorflow_addons/custom_ops/image/cc/kernels/resampler_ops.cc
        tensorflow_addons/custom_ops/image/cc/kernels/resampler_ops.h
        tensorflow_addons/custom_ops/image/cc/kernels/resampler_ops_gpu.cu.cc
        tensorflow_addons/custom_ops/image/cc/ops/distort_image_ops.cc
        tensorflow_addons/custom_ops/image/cc/ops/image_ops.cc
        tensorflow_addons/custom_ops/image/cc/ops/resampler_ops.cc
        tensorflow_addons/custom_ops/layers/cc/kernels/correlation_cost_op.cc
        tensorflow_addons/custom_ops/layers/cc/kernels/correlation_cost_op.h
        tensorflow_addons/custom_ops/layers/cc/kernels/correlation_cost_op_gpu.cu.cc
        tensorflow_addons/custom_ops/layers/cc/ops/correlation_cost_op.cc
        tensorflow_addons/custom_ops/seq2seq/cc/kernels/beam_search_ops.cc
        tensorflow_addons/custom_ops/seq2seq/cc/kernels/beam_search_ops.h
        tensorflow_addons/custom_ops/seq2seq/cc/kernels/beam_search_ops_gpu.cu.cc
        tensorflow_addons/custom_ops/seq2seq/cc/ops/beam_search_ops.cc
        tensorflow_addons/custom_ops/text/cc/kernels/parse_time_kernel.cc
        tensorflow_addons/custom_ops/text/cc/kernels/skip_gram_kernels.cc
        tensorflow_addons/custom_ops/text/cc/ops/parse_time_op.cc
        tensorflow_addons/custom_ops/text/cc/ops/skip_gram_ops.cc
        tensorflow_addons/custom_ops/layers/cc/ops/deformable_conv2d_op.cc
        tensorflow_addons/custom_ops/layers/cc/kernels/deformable_conv2d_op.h
        tensorflow_addons/custom_ops/layers/cc/kernels/deformable_conv2d_op.cc
        )

target_include_directories(addons PUBLIC
        /home/rito/miniconda3/lib/python3.7/site-packages/tensorflow/include
        )
include_directories(addons PRIVATE
        ./
        )

target_compile_definitions(addons PUBLIC GOOGLE_CUDA)
