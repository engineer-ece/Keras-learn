```code
Keras 3 - The Sequential class + what, where, why, when, who, 
how, pros, cons, image - (graph or related to topic), table,
example, proof , example code for proof, application ,key, summary.
```

## Keras 3 Sequential Class

### What is the Sequential class?

The Sequential class in Keras is a linear stack of layers. It's the simplest way to build a feedforward neural network. This means layers are added sequentially, one on top of the other.

### Where is it used?

The Sequential class is commonly used for:

* Simple feedforward neural networks
* Convolutional Neural Networks (CNNs) for image classification
* Recurrent Neural Networks (RNNs) for sequential data, although more complex architectures might be preferred for RNNs.

### Why use the Sequential class?

* **Simplicity:** Easy to define and understand.
* **Efficiency:** Often performs well for simpler models.
* **Good starting point:** For beginners or when exploring a new problem.

### When to use the Sequential class?

* When building a linear stack of layers.
* When the model architecture is straightforward.
* When performance is not a critical factor.

### Who uses the Sequential class?

* Beginners in deep learning
* Data scientists
* Machine learning engineers
* Anyone building simple neural networks

### How to use the Sequential class?

Create a Sequential model and add layers using the `add` method:

**Python**

```

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))

```

### Pros and Cons

**Pros:**

* Simple to use
* Efficient for simple models
* Good for prototyping

**Cons:**

* Limited for complex architectures
* Less flexible than the Functional API

### Image: Sequential Model Architecture

![](https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcToblpw-NNjqzmfJSoTnYmzAUPePO4B8MC_ETROnNv0FlnOAdnOJ597pqk890Nc)

### Example: Simple Neural Network

**Python**

```

from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(32, activation='relu', input_dim=784),
    Dense(10, activation='softmax')
])

```

### Proof: Model Performance

Evaluate the model on a holdout test set using metrics like accuracy, precision, recall, and F1-score.

**Python**

```

from keras.datasets import mnist
from keras.utils import to_categorical

# ... (model definition)

# Compile the model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

```

### Application:

* Image classification
* Regression
* Binary classification
* Other simple machine learning tasks

### Key Points

* Linear stack of layers
* Simple to use
* Suitable for basic models
* Less flexible than Functional API

### Summary

The Sequential class is a fundamental building block in Keras for creating simple neural networks. It's a good starting point for beginners and can be efficient for many tasks. However, for more complex architectures, the Functional API might be a better choice.

**Would you like to delve deeper into a specific aspect of the Sequential class or explore other Keras components?**

```

```
