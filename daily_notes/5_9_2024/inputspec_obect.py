from tensorflow.keras.layers import Layer, Input, Dense, InputSpec
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np

class XORLayer(Layer):
    def __init__(self, **kwargs):
        super(XORLayer, self).__init__(**kwargs)
        self.input_spec = InputSpec(shape=(None, 2))

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(2, 4), 
            initializer='uniform',
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(4,),  
            initializer='zeros',
            name='bias'
        )
        super(XORLayer, self).build(input_shape)

    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel) + self.bias
        x = tf.nn.relu(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4)

inputs = Input(shape=(2,))
x = XORLayer()(inputs)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np

class XORLayer(Layer):
    def __init__(self, **kwargs):
        super(XORLayer, self).__init__(**kwargs)
        self.input_spec = InputSpec(shape=(None, 2))

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(2, 3), 
            initializer='uniform',
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(3,),  
            initializer='zeros',
            name='bias'
        )
        super(XORLayer, self).build(input_shape)

    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel) + self.bias
        x = tf.nn.relu(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 3)

inputs = Input(shape=(2,))
x = XORLayer()(inputs)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
