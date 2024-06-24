# Copyright 2024 Lynden Wood

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras.layers import Layer
import tensorflow as tf
import scipy.signal as sps

class BandDecimate(Layer):
    def __init__(self,q,n=None,ftype='iir',axis=-1,zero_phase=True):
        self.q = q
        self.n = n
        self.ftype = ftype
        self.axis = axis
        self.zero_phase = zero_phase
        super().__init__()

    def ceil(self,a, b):
        return -1 * (-a // b)

    def NPDecimateAlias(self,input_array):
        n = self.n
        q = self.q
        axis = self.axis
        ftype = self.ftype
        zero_phase = self.zero_phase
        res = sps.decimate(input_array,q=q,n=n,axis=axis,ftype=ftype,zero_phase=zero_phase)
        return res

    def TFDecimate(self,input):
        q = self.q
        input_shape = input.shape.as_list()
        output_shape = input_shape
        output_shape[self.axis] = self.ceil(input_shape[self.axis],q)
        res = tf.numpy_function(self.NPDecimateAlias,[input],tf.float32)
        res = tf.ensure_shape(res,output_shape)
        return res

    def call(self, inputs:tf.Tensor):
        decimated_input = self.TFDecimate(inputs)
        return decimated_input