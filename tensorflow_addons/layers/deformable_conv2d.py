# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import typing

import tensorflow as tf
from typeguard import typechecked
from tensorflow_addons.utils import types
from tensorflow_addons.utils.resource_loader import LazySO
from tensorflow.python.keras.utils import conv_utils

_deformable_conv2d_ops_so = LazySO('custom_ops/layers/_deformable_conv2d_ops.so')


@typechecked
def _deformable_conv2d(
        input: tf.Tensor,
        filter: tf.Tensor,
        bias: tf.Tensor,
        offset: tf.Tensor,
        mask: tf.Tensor,
        strides: typing.Union[tuple, list],
        dilations: typing.Union[tuple, list],
        weight_groups: int,
        offset_groups: int,
        no_bias: bool,
        padding: str,
):
    with tf.name_scope('deformable_conv2d'):
        return _deformable_conv2d_ops_so.ops.addons_deformable_conv2d(
            input=input,
            filter=filter,
            bias=bias,
            offset=offset,
            mask=mask,
            strides=strides,
            weight_groups=weight_groups,
            offset_groups=offset_groups,
            no_bias=no_bias,
            padding=padding,
            data_format='NCHW',
            dilations=dilations,
        )


@tf.keras.utils.register_keras_serializable(package="Addons")
class DeformableConv2D(tf.keras.layers.Layer):
    @typechecked
    def __init__(
            self,
            filters: int,
            kernel_size: typing.Union[int, tuple, list] = (3, 3),
            strides: typing.Union[int, tuple, list] = (1, 1),
            padding: str = "valid",
            data_format: str = "channels_first",
            dilation_rate: typing.Union[int, tuple, list] = (1, 1),
            weight_groups: int = 1,
            offset_groups: int = 1,
            use_deformable_conv_bias: bool = False,
            use_filter_conv_bias: bool = False,
            use_mask_conv_bias: bool = False,
            kernel_initializer: types.Initializer = None,
            bias_initializer: types.Initializer = None,
            kernel_regularizer: types.Regularizer = None,
            bias_regularizer: types.Regularizer = None,
            kernel_constraint: types.Constraint = None,
            bias_constraint: types.Constraint = None,
            **kwargs
    ):
        super(DeformableConv2D, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.weight_groups = weight_groups
        self.offset_groups = offset_groups
        self.use_deformable_conv_bias = use_deformable_conv_bias
        self.use_filter_conv_bias = use_filter_conv_bias
        self.use_mask_conv_bias = use_mask_conv_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        if self.padding == 'causal':
            raise ValueError('Causal padding is not supported.')

        if self.data_format != 'channels_first':
            raise ValueError('`channels_last` data format is not supported.')

        self.conv_offset = tf.keras.layers.Conv2D(
            self.offset_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding=self.padding,
            use_bias=self.use_filter_conv_bias,
            data_format='channels_first',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        self.conv_mask = tf.keras.layers.Conv2D(
            self.offset_groups * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding=self.padding,
            use_bias=self.use_mask_conv_bias,
            data_format='channels_first',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        self.filter_weights = None
        self.bias_weights = None

    def build(self, input_shape):
        shape = (self.filters, input_shape[1] // self.weight_groups, self.kernel_size[0], self.kernel_size[1])

        self.filter_weights = self.add_weight(
            name='filter', shape=shape,
            initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint, trainable=True
        )
        self.bias_weights = self.add_weight(
            name='bias', shape=(self.filters,),
            initializer=self.bias_initializer, regularizer=self.bias_regularizer,
            constraint=self.bias_constraint, trainable=True
        )

        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()

        space = input_shape[2:]
        new_space = []

        for i, s in enumerate(space):
            new_dim = conv_utils.conv_output_length(
                s, self.kernel_size[i], padding=self.padding,
                stride=self.strides[i], dilation=self.dilation_rate[i]
            )
            new_space.append(new_dim)

        return tf.TensorShape([input_shape[0], self.filters] + new_space)

    def call(self, inputs, **kwargs):
        offset = self.conv_offset(inputs)
        mask = tf.keras.activations.sigmoid(self.conv_mask(inputs))

        return _deformable_conv2d(
            input=tf.convert_to_tensor(inputs),
            filter=tf.convert_to_tensor(self.filter_weights),
            bias=tf.convert_to_tensor(self.bias_weights),
            offset=tf.convert_to_tensor(offset),
            mask=tf.convert_to_tensor(mask),
            strides=self.strides,
            weight_groups=self.weight_groups,
            offset_groups=self.offset_groups,
            no_bias=not self.use_deformable_conv_bias,
            padding='SAME' if self.padding == 'same' else 'VALID',
            dilations=self.dilation_rate,
        )

    def get_config(self):
        config = {
            "kernel_size": self.kernel_size,
            "filters": self.filters,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "weight_groups": self.weight_groups,
            "offset_groups": self.offset_groups,
            "use_deformable_conv_bias": self.use_deformable_conv_bias,
            "use_filter_conv_bias": self.use_filter_conv_bias,
            "use_mask_conv_bias": self.use_mask_conv_bias,
        }
        base_config = super().get_config()
        return {**base_config, **config}
