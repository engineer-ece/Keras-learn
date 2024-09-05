import keras
from keras.layers  import Input,Dense
from keras.models import Model
from keras.regularizers import *
from keras.constraints import *

# vanishing gradient
# saturation
# non-linearity

i = Input((2,))
#h = Dense(3,activation='relu')(i)
#o = Dense(1, activation='sigmoid')(h)
r = Dense(
    5,
    activation='relu',
    use_bias=True,
    kernel_initializer='he_normal',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.01),
    bias_regularizer=l2(0.01),
    activity_regularizer=l2(0.01),
    kernel_constraint=max_norm(2.0),
    bias_constraint=max_norm(1.0),
    #lora_rank=4
)(i)
model = Model(inputs=i,outputs=r)

print(model.summary())
