### Overview of `Layer` Class Properties and Methods in `model.summary()`

In Keras, the `model.summary()` method provides a summary of the model, including the layer types, output shapes, and parameter counts. Here's how each property and method of the `Layer` class impacts `model.summary()`:

| **Aspect**               | **Description**                                                                                   | **Impact on `model.summary()`**                                                   | **Parameter Count Calculation**                               |
|--------------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|---------------------------------------------------------------|
| **Layer Class**          | Base class for all layers in Keras. Defines the basic functionality and properties for layers.   | Displays the type and name of each layer.                                          | Not directly shown in `model.summary()`.                      |
| **weights Property**     | Returns all weights of the layer, including trainable and non-trainable weights.                  | The parameter count in `model.summary()` includes all weights returned by this property. | Total count of weights (trainable and non-trainable).        |
| **trainable_weights Property** | Returns only the weights that are trainable.                                                        | Parameter count in `model.summary()` includes only trainable weights.               | Only trainable weights are considered in parameter count.    |
| **non_trainable_weights Property** | Returns weights that are not trainable.                                                               | These weights are included in the total parameter count but not updated during training. | Included in total parameter count.                           |
| **add_weight Method**    | Adds a new weight variable to the layer.                                                           | Weights added using this method are included in the parameter count.                | Parameter count increases based on the shape and number of weights added. |
| **trainable Property**   | Indicates whether the layer's weights are trainable or not.                                        | Affects which weights are counted as trainable in the `model.summary()`.             | Only trainable weights are counted towards parameter count.  |
| **get_weights Method**   | Retrieves the current weights of the layer.                                                        | Not directly shown in `model.summary()`.                                             | Not applicable to parameter count.                           |
| **set_weights Method**   | Sets the weights of the layer.                                                                     | Affects the weights but does not directly impact `model.summary()`.                   | Adjusts parameter count if weights change.                   |
| **get_config Method**    | Returns a dictionary containing the configuration of the layer.                                    | Not directly shown in `model.summary()`.                                             | Not applicable to parameter count.                           |
| **add_loss Method**      | Adds a loss tensor to the layer.                                                                   | Losses added are not shown in `model.summary()`.                                     | Not included in parameter count.                             |
| **losses Property**      | Returns a list of losses added to the layer.                                                        | Losses are tracked but not displayed in `model.summary()`.                           | Not included in parameter count.                             |

### Example Explanation

Here's an example to illustrate how the parameter count is calculated and how these aspects are shown in `model.summary()`:

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer

class CustomLayer(Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias

# Create a model with CustomLayer
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(20,)),
    CustomLayer(10)
])

# Print model summary
model.summary()
```

### Explanation of the Summary Output

1. **Layer Class**:
   - **Type**: `CustomLayer`
   - **Output Shape**: `(None, 10)`

2. **Weights Property**:
   - Includes all weights: `kernel` and `bias`.
   - **Parameter Count**: `input_dim * units (kernel) + units (bias)`

3. **trainable_weights Property**:
   - Includes only the `kernel` and `bias` weights.
   - **Parameter Count**: Same as above.

4. **non_trainable_weights Property**:
   - If there were any non-trainable weights, they would be listed here. In this case, there are none.

5. **add_weight Method**:
   - Adds `kernel` and `bias` weights.
   - **Parameter Count**: `kernel` (20 * 10) + `bias` (10) = 210 parameters.

6. **trainable Property**:
   - Both `kernel` and `bias` are trainable.

7. **get_weights Method**:
   - Retrieves current weights, not shown in summary.

8. **set_weights Method**:
   - Used to set weights, not shown in summary.

9. **get_config Method**:
   - Returns configuration dictionary, not shown in summary.

10. **add_loss Method**:
    - Adds custom loss, not shown in summary.

11. **losses Property**:
    - Returns added losses, not shown in summary.

In `model.summary()`, you see the number of parameters as the sum of weights from all layers, which includes trainable weights. Non-trainable weights, custom losses, and configurations are not displayed in `model.summary()` but are handled within the layer's methods and properties.


----------

