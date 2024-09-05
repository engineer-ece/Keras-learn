import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, EinsumDense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

l = EinsumDense("ab,bc->ac",output_shape=3,bias_axes="c")
i = Input((2,))
o = l(i)
m = Model(inputs=i,outputs=o)
print(m.summary())
