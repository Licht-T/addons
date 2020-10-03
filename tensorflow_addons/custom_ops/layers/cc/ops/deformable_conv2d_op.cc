// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {
    namespace addons {

        using ::tensorflow::shape_inference::InferenceContext;
        using ::tensorflow::shape_inference::ShapeHandle;
        using ::tensorflow::shape_inference::DimensionHandle;
        // --------------------------------------------------------------------------

        REGISTER_OP("Addons>DeformableConv2D")
                .Input("input: T")
                .Input("filter: T")
                .Input("bias: T")
                .Input("offset: T")
                .Input("mask: T")
                .Output("output: T")
                .Attr("strides: list(int)")
                .Attr("weight_groups: int")
                .Attr("offset_groups: int")
                .Attr("no_bias: bool = true")
                .Attr(GetPaddingAttrString())
                .Attr("dilations: list(int)")
                .Attr("data_format: { 'NCHW' }")
                .Attr("T: {half, bfloat16, float, double}")
                .SetShapeFn([](InferenceContext *c) {
                    ShapeHandle input_shape;
                    ShapeHandle filter_shape;
                    ShapeHandle offset_shape;
                    ShapeHandle mask_shape;
                    std::vector<int32> strides;
                    std::vector<int32> dilations;
                    std::string data_format_str;
                    TensorFormat data_format;
                    int32 weight_groups;
                    int32 offset_groups;
                    Padding padding;

                    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
                    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));
                    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &offset_shape));
                    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &mask_shape));
                    TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
                    TF_RETURN_IF_ERROR(c->GetAttr("weight_groups", &weight_groups));
                    TF_RETURN_IF_ERROR(c->GetAttr("offset_groups", &offset_groups));
                    TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
                    TF_RETURN_IF_ERROR(c->GetAttr("dilations", &dilations));
                    TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format_str));
                    FormatFromString(data_format_str, &data_format);
                    if (strides.size() != 2 || dilations.size() != 2) {
                        return errors::InvalidArgument("TODO");
                    }

                    DimensionHandle input_batches_dim = c->Dim(input_shape, 0);
                    DimensionHandle input_channels_dim = c->Dim(input_shape, 1);
                    DimensionHandle input_rows_dim = c->Dim(input_shape, 2);
                    DimensionHandle input_cols_dim = c->Dim(input_shape, 3);

                    DimensionHandle output_channels_dim = c->Dim(filter_shape, 0);
                    DimensionHandle filter_channels_dim = c->Dim(filter_shape, 1);
                    DimensionHandle filter_rows_dim = c->Dim(filter_shape, 2);
                    DimensionHandle filter_cols_dim = c->Dim(filter_shape, 3);

                    DimensionHandle offset_batches_dim = c->Dim(offset_shape, 0);
                    DimensionHandle offset_channels_dim = c->Dim(offset_shape, 1);
                    DimensionHandle offset_heights_dim = c->Dim(offset_shape, 2);
                    DimensionHandle offset_weights_dim = c->Dim(offset_shape, 3);

                    DimensionHandle mask_batches_dim = c->Dim(mask_shape, 0);
                    DimensionHandle mask_channels_dim = c->Dim(mask_shape, 1);
                    DimensionHandle mask_heights_dim = c->Dim(mask_shape, 2);
                    DimensionHandle mask_weights_dim = c->Dim(mask_shape, 3);

                    auto input_batches = InferenceContext::Value(input_batches_dim);
                    auto input_rows = InferenceContext::Value(input_rows_dim);
                    auto input_cols = InferenceContext::Value(input_cols_dim);
                    auto filter_rows = InferenceContext::Value(filter_rows_dim);
                    auto filter_cols = InferenceContext::Value(filter_cols_dim);

                    auto stride_rows = strides[0];
                    auto stride_cols = strides[1];
                    auto diration_rows = dilations[0];
                    auto diration_cols = dilations[1];

                    DimensionHandle tmp;
                    TF_RETURN_IF_ERROR(c->Divide(output_channels_dim, weight_groups, true, &tmp));
                    TF_RETURN_IF_ERROR(c->Divide(input_channels_dim, offset_groups, true, &tmp));

                    TF_RETURN_IF_ERROR(c->Multiply(filter_channels_dim, weight_groups, &tmp));
                    TF_RETURN_IF_ERROR(c->Merge(input_channels_dim, tmp, &tmp));

                    TF_RETURN_IF_ERROR(c->WithValue(offset_batches_dim, input_batches, &tmp));
                    TF_RETURN_IF_ERROR(c->WithValue(mask_batches_dim, input_batches, &tmp));

                    if (InferenceContext::ValueKnown(filter_rows_dim) && InferenceContext::ValueKnown(filter_cols_dim)) {
                        auto filter_area = filter_rows * filter_cols * offset_groups;
                        TF_RETURN_IF_ERROR(c->WithValue(mask_channels_dim, filter_area, &tmp));
                        TF_RETURN_IF_ERROR(c->WithValue(offset_channels_dim, 2 * filter_area, &tmp));
                    }

                    if (!InferenceContext::ValueKnown(input_rows_dim) || !InferenceContext::ValueKnown(input_cols_dim) ||
                        !InferenceContext::ValueKnown(filter_rows_dim) || !InferenceContext::ValueKnown(filter_cols_dim)) {
                        c->set_output(0, c->MakeShape(
                                {input_batches_dim, output_channels_dim, InferenceContext::kUnknownDim,
                                 InferenceContext::kUnknownDim}));
                        return Status::OK();
                    }

                    auto filter_rows_eff = filter_rows + (filter_rows - 1) * (diration_rows - 1);
                    auto filter_cols_eff = filter_cols + (filter_cols - 1) * (diration_cols - 1);

                    int64 output_rows, output_cols;
                    int64 padding_before, padding_after;
                    TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
                            input_rows, filter_rows_eff, stride_rows, padding, &output_rows,
                            &padding_before, &padding_after));
                    TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
                            input_cols, filter_cols_eff, stride_cols, padding, &output_cols,
                            &padding_before, &padding_after));


                    TF_RETURN_IF_ERROR(c->WithValue(mask_heights_dim, output_rows, &tmp));
                    TF_RETURN_IF_ERROR(c->WithValue(mask_weights_dim, output_cols, &tmp));
                    TF_RETURN_IF_ERROR(c->WithValue(offset_heights_dim, output_rows, &tmp));
                    TF_RETURN_IF_ERROR(c->WithValue(offset_weights_dim, output_cols, &tmp));

                    c->set_output(0, c->MakeShape({input_batches_dim, output_channels_dim, output_rows, output_cols}));

                    return Status::OK();
                })
                .Doc(R"doc(TODO)doc");

        REGISTER_OP("Addons>DeformableConv2DGrad")
                .Input("input: T")
                .Input("filter: T")
                .Input("bias: T")
                .Input("offset: T")
                .Input("mask: T")
                .Input("output_grad: T")
                .Output("input_grad: T")
                .Output("filter_grad: T")
                .Output("offset_grad: T")
                .Output("mask_grad: T")
                .Attr("strides: list(int) = [1, 1, 1, 1]")
                .Attr("weight_groups: int")
                .Attr("offset_groups: int")
                .Attr("no_bias: bool = true")
                .Attr(GetPaddingAttrString())
                .Attr("dilations: list(int) = [1, 1, 1, 1]")
                .Attr("data_format: { 'NCHW' }")
                .Attr("T: {half, bfloat16, float, double}")
                .SetShapeFn([](InferenceContext *c) {
                    c->set_output(0, c->input(0));
                    c->set_output(1, c->input(1));
                    c->set_output(2, c->input(2));
                    c->set_output(3, c->input(3));
                    c->set_output(4, c->input(4));
                    return Status::OK();
                })
                .Doc(R"doc(TODO)doc");

    }  // namespace addons
}  // namespace tensorflow