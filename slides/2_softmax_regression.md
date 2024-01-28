---
layout: chapter-title

# the image source
image: /img_VNOFvV7oFGy3ZuIbSW90.jpg

# a custom class name to the content
class: my-cool-content-on-the-right
---

## Linear Neural Networks for Classification

<span style="color:DimGray; font-size: 11px; position:absolute; right:20px; bottom:20px;">Image credit: Midjourney 6.0<br> prompt: ‘Linear Neural Networks for Classification'
</span>

---

# Linear Model

* Ex. We want to classify images of 3 animals "cat", "chicken" and "dog"<br> using predictors $x_1, x_2, x_3, x_4$

* Using *one-hot encoding* we can represent this data as $\footnotesize y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}$

* We need a model with multiple outputs, one per class:<br>
$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3
\end{aligned}
$

<div>
  <figure>
    <img src="/softmaxreg.svg" style="width: 400px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
      <a href="https://d2l.ai/chapter_linear-classification/softmax-regression.html">d2l.ai Fig. 4.1.1 Softmax regression is a single-layer neural network</a>
    </figcaption>
  </figure>
</div>


---

# Linear Model
<br>

$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3
\end{aligned}
$
* What are the problems with this vector-valued regression problem?
	* There is no guarantee that the outputs $o_i$ sum up to $1$ in the way we expect probabilities to behave
	* There is no guarantee that the outputs $o_i$ are even nonnegative
* We need somehow modify the outputs $o_i$:
	* [Probit model](https://en.wikipedia.org/wiki/Probit_model): $\mathbf{y} = \mathbf{o} + \boldsymbol{\epsilon}$, where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$
	* Softmax model

---
layout: chapter-title

# the image source
image: /img_d5kS6PBkNg9kQV66WxxV.jpg

# a custom class name to the content
class: my-cool-content-on-the-right
---

## Softmax Regression

<span style="color:DimGray; font-size: 11px; position:absolute; right:20px; bottom:20px;">Image credit: Midjourney 6.0<br> prompt: ‘Softmax Regression'
</span>

---

# The Softmax

* The Softmax model is based on the use of an exponential function $P(y = i) \propto e^{o_i}$ and on the *normalization*:<br>
$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})$, where $\hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}$

* Due to preserved ordering in the Softmax:<br> $\argmax\limits_j \hat y_j = \argmax\limits_j o_j$


---

# The Softmax: Vectorization

* Recall that the data matrix and the model used are both vectorized:
	* Minibatch $\mathbf{X} \in \mathbb{R}^{n \times d}$ and $q$ output classes
	* Model $\mathbf{O} = \mathbf{X} \mathbf{W} + \mathbf{b}$
		* Weights satisfy $\mathbf{W} \in \mathbb{R}^{d \times q}$
		* Bias satisfies $\mathbf{b} \in \mathbb{R}^{1\times q}$
* After applying the Softmax:<br>
$\hat{\mathbf{Y}} = \mathrm{softmax}(\mathbf{O})$

* The softmax operation itself can be computed *rowwise*.

---

# Softmax and Cross-Entropy Loss

#### Inserting the softmax into multi-class cross-enthropy $l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum\limits_{j=1}^q y_j \log \hat{y}_j$ we obtain:
$$ {0|1|2|all}
\small
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j \\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j
\end{aligned}
$$

#### The loss derivatives with respect to any logit $o_j$ are expressed by elements in the one-hot label vector: $\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum\limits_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j$