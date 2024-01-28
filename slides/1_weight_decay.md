---
layout: chapter-title

# the image source
image: /img_dPTmgiHqZzSsjTU55qIO.jpg

# a custom class name to the content
class: my-cool-content-on-the-right
---

## Norms and Weight Decay

<span style="color:DimGray; font-size: 11px; position:absolute; right:20px; bottom:20px;">Image credit: Midjourney 6.0<br> prompt: ‘Norms and Weight Decay'
</span>

---

# Norms and Weight Decay (see [d2l.ai](https://d2l.ai/chapter_linear-regression/))

- How to overcome the **overfitting** of a linear model?
  - Use more data
  - Use one of the regularization techniques:
    - $\ell_1$ or Lasso regression<br>
    $\sum\limits_{i=1}^n (Y_i - \sum\limits_{j=1}^p X_{ij}\beta_j)^2 + \lambda\sum\limits_{j=1}^p \lvert \beta_j \rvert$
    - $\ell_2$ or Ridge regression<br>
    $\sum\limits_{i=1}^n (Y_i - \sum\limits_{j=1}^p X_{ij}\beta_j)^2 + \lambda\sum\limits_{j=1}^p \beta_j^2$

---

# Norms and Weight Decay: Model Complexity

- Consider simplest function $f_0 = 0$
  - Therefore we can measure the **complexity** of a function by the distance of its parameters from zero

- Complexity of a linear function $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ is measured by some norm of its weight vector, e.g., $\| \mathbf{w} \|^2$

  - Recall that:
    - $\mathbf{x}^{(i)}$ are the features
    - $y^{(i)}$ is the label for any data example $i$
    - $(\mathbf{w}, b)$ are the weight and bias parameters

---

# Norms and Weight Decay: $\ell_2$ Regularization

- Here we minimize **just** *training error* (the sum of the prediction loss):
$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2$$

- But assuming $\ell_2$ regularization, we minimize **both** training error and the penalty term:
$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2 + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

---

# Norms and Weight Decay: $\ell_2$ Regularization


<br>

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2 + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

- For $\lambda = 0$, we recover our original loss function
- For $\lambda > 0$, we restrict the size of $\| \mathbf{w} \|$
<br>
<br>

***
<br>

- Note 1: we need $\frac{1}{2}$ to simplify the derivative of a quadratic function
- Note 2: usage of $\|\mathbf{w}\|^2$ instead of $\|\mathbf{w}\|$ is more preferable because the corresponding derivative is easy to compute: the sum of derivatives equals the derivative of the sum

---

# Norms and Weight Decay: why not $\ell_1$?

- $\ell_2$ norm places an outsize penalty on large components of the weight vector
  - This biases our learning algorithm towards models with weight evenly distributed across a larger number of features
  - In practice, this might make them more robust to measurement error in a single variable

- $\ell_1$ penalties lead to models that concentrate weights on a small set of features
by clearing the other weights to zero
  - This gives us an effective method for *feature selection*

---

# Norms and Weight Decay for Minibatch SGD

<br>

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)
\end{aligned}$$

- We update $\mathbf{w}$ based on the amount by which our estimate differs from the observation
  - And also shrink the size of $\mathbf{w}$ towards zero

- That is why the method is sometimes called "**weight decay**":
  - Given the penalty term alone, our optimization algorithm *decays*
the weight at each step of training

- Finally, weight decay offers us a mechanism for continuously adjusting the complexity of a function