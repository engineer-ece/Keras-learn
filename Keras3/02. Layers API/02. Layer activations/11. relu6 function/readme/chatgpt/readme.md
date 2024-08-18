<body>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.15.2/katex.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.15.2/katex.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.15.2/contrib/auto-render.min.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            renderMathInElement(document.body, {
                delimiters: [
                    { left: "$$", right: "$$", display: true },
                    { left: "$", right: "$", display: false }
                ]
            });
        });
    </script>   
</body>

### Keras 3 - ReLU6 Function

#### What

The `ReLU6` function is a variant of the Rectified Linear Unit (ReLU) activation function. It is defined as:

$$\text{ReLU6}(x) = \min(\max(0, x), 6) $$

This function transforms input values by applying ReLU and then capping the output at 6.

#### Where

The `ReLU6` function is part of the Keras library in TensorFlow, used in deep learning models to control the range of activation values. It is particularly useful in neural network architectures designed for mobile and embedded devices.

#### Why

ReLU6 is used to:

- **Control Activation Values:** Limits the maximum output value to 6, preventing excessively large activation values that can cause instability.
- **Improve Efficiency:** Helps in creating models that are efficient and suitable for resource-constrained environments.
- **Enhance Training Stability:** Provides more stable training by avoiding extreme activation values.

#### When

ReLU6 is typically used in scenarios where:

- **Efficiency is Critical:** Such as in mobile and embedded systems where computational resources are limited.
- **Stable Training:** When you want to prevent exploding activation values and maintain stable gradients.

#### Who

ReLU6 was introduced by Google researchers as part of the MobileNet architecture, which is designed for efficient deep learning models on mobile and embedded devices.

#### How

In Keras, you can apply ReLU6 as an activation function in your neural network layers. Here’s how you can use it:

```python
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu6
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(64, activation=relu6, input_shape=(input_dim,)))
```

#### Pros

- **Prevents Exploding Activations:** Caps the maximum value of activations, reducing the risk of instability.
- **Efficient Computation:** Suitable for models running on mobile and embedded devices due to its controlled output range.
- **Stable Training:** Helps maintain more consistent training dynamics by avoiding extreme activation values.

#### Cons

- **Output Range Limitation:** Caps the activation values to a maximum of 6, which might affect the learning capacity if the network benefits from larger activation values.
- **Not Suitable for All Scenarios:** Might not be ideal for all types of neural networks, especially those that require a wider range of activation values.

#### Image

Here’s a graph comparing ReLU6 with the standard ReLU activation function:

![ReLU6 vs. ReLU](https://github.com/engineer-ece/Keras-learn/blob/edf75093c51220acf4d5e934fd3f5cce0ac6356c/Keras3/02.%20Layers%20API/02.%20Layer%20activations/11.%20relu6%20function/relu6_function.png)

In this graph, the ReLU6 function is similar to ReLU but is capped at 6.

#### Table

| Activation Function | Formula                 | Output Range    |
| ------------------- | ----------------------- | --------------- |
| ReLU                | $\max(0, x)$          | $[0, \infty)$ |
| ReLU6               | $\min(\max(0, x), 6)$ | $[0, 6]$      |

#### Example

Consider the following input values and their corresponding ReLU6 outputs:

- **Input Values:** $[-2, 0, 4, 8]$
- **ReLU6 Outputs:** $[0, 0, 4, 6]$

#### Proof

To verify the behavior of the ReLU6 function:

1. For $x < 0$: $\text{ReLU6}(x) = \min(\max(0, x), 6) = 0$
2. For $0 \leq x \leq 6$: $\text{ReLU6}(x) = \min(\max(0, x), 6) = x$
3. For $x > 6$: $\text{ReLU6}(x) = \min(\max(0, x), 6) = 6$

#### Example Code for Proof & Mode

Here’s Python code to visualize and prove the behavior of ReLU6:

```python
import numpy as np
import matplotlib.pyplot as plt

def relu6(x):
    return np.minimum(np.maximum(0, x), 6)

x = np.linspace(-2, 10, 100)
y = relu6(x)

plt.plot(x, y, label='ReLU6')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU6 Activation Function')
plt.legend()
plt.grid(True)
plt.show()
```

#### Application

- **MobileNets:** Used in MobileNet architectures to create efficient models for mobile devices.
- **Embedded Systems:** Suitable for applications where computational resources are limited and efficiency is important.

#### Key

The key aspect of ReLU6 is its ability to combine the benefits of the ReLU function with an upper bound on activation values, making it suitable for efficient and stable neural network models, especially in resource-constrained environments.

#### Summary

ReLU6 is a variant of the ReLU activation function that caps output values at 6, helping to prevent exploding activations and improve model efficiency and stability. It is particularly useful in mobile and embedded systems where computational resources are limited. While it enhances stability and efficiency, the capped output range may limit the flexibility of the model in certain scenarios.
