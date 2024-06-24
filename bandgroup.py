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

class Bandgroup(Layer):
    def __init__(self,n=3,method=2,name=None):
        super().__init__(trainable=False,name=name)
        self.groups = n
        self.method = method

    def call(self, inputs):
        n = self.groups
        n_elements = inputs.shape[1]//n*n 
        tensor_truncated = inputs[:,:n_elements]
        shape = tf.concat(values=[tf.shape(tensor_truncated)[:1], tf.TensorShape([inputs.shape[1]//n,n])],axis=0)
        bandgroup_chunk = tf.reshape( tensor_truncated , shape )
        bandgroup_stride = tf.transpose( bandgroup_chunk, (0,2,1) )
        if self.method == 1:
            output = bandgroup_chunk
        elif self.method == 2:
            output = bandgroup_stride
        else:
            output = inputs
        return output