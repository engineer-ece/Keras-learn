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

[Constraints](https://www.youtube.com/watch?v=LiAHtgFyRkA)

<!-- Layer Weight Constraints Overview -->

## Layer Weight Constraints Table

| **Constraint Class** | **Use Cases**                                    | **Formula**                                           | **Formula Parameter Explanation**                              | **Tips and Tricks**                              | **Example Application in Process Flow**                            |
|----------------------|--------------------------------------------------|------------------------------------------------------|---------------------------------------------------------------|--------------------------------------------------|--------------------------------------------------------------------|
| **Constraint**       | Base class for custom constraints.               | N/A                                                  | N/A                                                           | Provides a framework for custom constraints.    | Implement custom constraints by inheriting from this class.         |
| **MaxNorm**          | Prevents large weights, stabilizes training.     | $$W_{\text{new}} = \frac{W_{\text{old}}}{\max(1, \frac{\|W_{\text{old}}\|_2}{\text{max\_value}})}$$ | **max_value**: Maximum norm value; **axis**: Axis for norm calculation | Useful for controlling weight growth; adjust max_value for tighter control. | Apply in layers prone to large weight values, such as dense layers. |
| **MinMaxNorm**       | Keeps weights within a specified range.          | $$W_{\text{new}} = \text{min\_value} + \left( \frac{\text{max\_value} - \text{min\_value}}{\max(\text{epsilon}, \|W_{\text{old}}\|_2)} \right) W_{\text{old}}$$ | **min_value**: Minimum norm; **max_value**: Maximum norm; **rate**: Interpolation rate; **axis**: Axis for norm calculation | Stabilizes weights by ensuring they stay within bounds; tweak rate for smoother transitions. | Use when weights need to be constrained within a specific range.    |
| **NonNeg**           | Ensures weights are non-negative.                | $$W_{\text{new}} = \max(0, W_{\text{old}})$$         | None                                                          | Forces non-negative weights; beneficial for interpretability. | Apply in models where non-negative weights are essential (e.g., RNNs). |
| **UnitNorm**         | Normalizes weight vectors to unit norm.          | $$W_{\text{new}} = \frac{W_{\text{old}}}{\|W_{\text{old}}\|_2}$$ | **axis**: Axis for normalization                                | Preserves direction of weights while controlling magnitude; useful in specific layer types. | Apply when consistent weight norms are crucial, such as in embedding layers. |
