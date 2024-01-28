---
layout: chapter-title

# the image source
image: /img_ZA9C3TFvUSncIIo3EaG2.jpg

# a custom class name to the content
class: my-cool-content-on-the-right
---

## Loss Functions

<span style="color:DimGray; font-size: 11px; position:absolute; right:20px; bottom:20px;">Image credit: Midjourney 6.0<br> prompt: ‘Loss Function'
</span>

---

# Loss Function
<div></div>

A loss function is a function $L:(z,y)\in \R \times Y \rightarrow L(z,y) \in R$ that takes as inputs the predicted value $z$ corresponding to the real data value $y$ and outputs how different they are.

<div>
  <figure>
    <img src="/Loss_functions_cs-229.png" style="width: 750px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right: 60px; top: 60px"><br>Image source:
      <a href="https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-supervised-learning">by Shervine Amidi</a>
    </figcaption>
  </figure>
</div>

---

# Cross Entropy
<div></div>

<small>
  <small>

| Petal Width | Sepal Width | Species    | $"p"$  | Cross Entropy      |
|-------------|-------------|------------|--------|--------------------|
| **0.04**        | **0.42**        | **Setosa**     |  **0.57**  | $\bm{-\mathrm{log}("p")}$ **= 0.56** |
| 1.0         | 0.54        | Virginica  |  0.58  | $-\mathrm{log}("p")$ = 0.54 |
| 0.50        | 0.37        | Versicolor |  0.52  | $-\mathrm{log}("p")$ = 0.65 |

</small>
</small>

<div>
  <figure>
    <img src="/Cross_entropy_1.svg" style="width: 680px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right: 60px; top: 60px"><br>Example inspired by:
      <a href="https://www.youtube.com/watch?v=6ArSys5qHAU">Josh Starmer's video</a>
    </figcaption>
  </figure>
</div>

---

# Cross Entropy
<div></div>

<small>
  <small>

| Petal Width | Sepal Width | Species    | $"p"$  | Cross Entropy      |
|-------------|-------------|------------|--------|--------------------|
| 0.04        | 0.42        | Setosa     |  0.57  | $-\mathrm{log}("p")$ = 0.56 |
| **1.0**         | **0.54**        | **Virginica**  |  **0.58**  | $\bm{-\mathrm{log}("p")}$ **= 0.54** |
| 0.50        | 0.37        | Versicolor |  0.52  | $-\mathrm{log}("p")$ = 0.65 |

</small>
</small>

<div>
  <figure>
    <img src="/Cross_entropy_2.svg" style="width: 680px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right: 60px; top: 60px"><br>Example inspired by:
      <a href="https://www.youtube.com/watch?v=6ArSys5qHAU">Josh Starmer's video</a>
    </figcaption>
  </figure>
</div>

---

# Cross Entropy
<div></div>

<small>
  <small>

| Petal Width | Sepal Width | Species    | $"p"$  | Cross Entropy      |
|-------------|-------------|------------|--------|--------------------|
| 0.04        | 0.42        | Setosa     |  0.57  | $-\mathrm{log}("p")$ = 0.56 |
| 1.0         | 0.54        | Virginica  |  0.58  | $-\mathrm{log}("p")$ = 0.54 |
| **0.50**        | **0.37**        | **Versicolor** |  **0.52**  | $\bm{-\mathrm{log}("p")}$ **= 0.65** |

</small>
</small>

<div>
  <figure>
    <img src="/Cross_entropy_3.svg" style="width: 680px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right: 60px; top: 60px"><br>Example inspired by:
      <a href="https://www.youtube.com/watch?v=6ArSys5qHAU">Josh Starmer's video</a>
    </figcaption>
  </figure>
</div>

---

# Cross Entropy
<div></div>

<small>
  <small>

| Petal Width | Sepal Width | Species    | $"p"$  | Cross Entropy      |
|-------------|-------------|------------|--------|--------------------|
| 0.04        | 0.42        | Setosa     |  0.57  | $\bm{-\mathrm{log}("p")}$ **= 0.56** |
| 1.0         | 0.54        | Virginica  |  0.58  | $\bm{-\mathrm{log}("p")}$ **= 0.54** |
| 0.50        | 0.37        | Versicolor |  0.52  | $\bm{-\mathrm{log}("p")}$ **= 0.65** |

</small>
</small>

<span style="margin-left: 400px;">Total Cross Entropy = 0.56 + 0.54 + 0.65 = 1.75</span>

<div>
  <figure>
    <img src="/Log_loss.png" style="width: 250px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image source:
      <a href="https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html">ml-cheatsheet.readthedocs.io</a>
    </figcaption>
  </figure>
</div>

---

# Usage of common Loss Functions
<div></div>

[Mean Absolute Error (MAE)](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss) Loss: $L(x, y) = |x - y|$

```python {all}
# MAE Loss
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
mae_loss = nn.L1Loss()
output = mae_loss(input, target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(1.2850, grad_fn=<L1LossBackward>)
```

#### When could it be used?

* Regression problems. It is considered to be more robust to outliers.

<span style="color:grey"><small> Slides 23-29 are based on the [PyTorch documentation](https://pytorch.org/docs/stable/nn.html) and on the [neptune.ai guide](https://neptune.ai/blog/pytorch-loss-functions).</small></span>
---

# Usage of common Loss Functions
<div></div>

[Mean Squared Error (MSE)](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) Loss: $L(x, y) = (x - y)^2$

```python {all}
# MSE Loss
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
mse_loss = nn.MSELoss()
output = mse_loss(input, target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(2.3280, grad_fn=<MseLossBackward>)
```

#### When could it be used?

* Regression problems. MSE is the default loss function for most Pytorch regression problems

---

# Usage of common Loss Functions
<div></div>

[Negative Log-Likelihood (NLL)](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) Loss: $L(x, y) = \{l_1,...,l_N\}^T$, where $l_N = -w_{y_n}x_{n,y_n}$. Softmax required!

```python {all}
# NLL Loss
import torch
import torch.nn as nn

# size of input (N x C) is = 3 x 5
input = torch.randn(3, 5, requires_grad=True)
# every element in target should have 0 <= value < C
target = torch.tensor([1, 0, 4])
m = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()
output = nll_loss(m(input), target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(2.9472, grad_fn=<NllLossBackward>)
```

#### When could it be used?

* Multi-class classification problems

---

# Usage of common Loss Functions
<div></div>

[Cross Entropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) Loss: $L(x, y) = -[y \cdot \mathrm{log}(x) + (1 - y) \cdot \mathrm{log}(1 - x)]$

```python {all}
# Cross Entropy Loss
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
cross_entropy_loss = nn.CrossEntropyLoss()
output = cross_entropy_loss(input, target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(1.0393, grad_fn=<NllLossBackward>)
```

#### When could it be used?

* Binary classification tasks (default loss for classification in PyTorch)

---

# Usage of common Loss Functions
<div></div>

[Hinge Embedding](https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html#torch.nn.HingeEmbeddingLoss) Loss: $L(x,y) = \begin{cases}
        x, \phantom{-1 <{}} \phantom{-1 <{}} \phantom{-1 <{}} \mathrm{\textcolor{grey}{if}~} y = 1 \\
        \mathrm{max}\{0, \Delta - x\}, \phantom{-1 <{}} \mathrm{\textcolor{grey}{~if}~} y = -1
      \end{cases}$

```python {all}
# Hinge Embedding Loss
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
hinge_loss = nn.HingeEmbeddingLoss()
output = hinge_loss(input, target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(1.2183, grad_fn=<MeanBackward0>)
```

#### When could it be used?

* Classification problems, especially when determining if two inputs are dissimilar or similar
* Learning nonlinear embeddings or semi-supervised learning tasks

---

# Usage of common Loss Functions
<div></div>

[Margin Ranking](https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html#torch.nn.MarginRankingLoss) Loss: $L(x_1, x_2, y) = \mathrm{max}(0, -y \cdot (x_1 - x_2) + \mathrm{margin})$

```python {all}
# Margin Ranking Loss
import torch
import torch.nn as nn

input_one = torch.randn(3, requires_grad=True)
input_two = torch.randn(3, requires_grad=True)
target = torch.randn(3).sign()

ranking_loss = nn.MarginRankingLoss()
output = ranking_loss(input_one, input_two, target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(1.3324, grad_fn=<MeanBackward0>)
```

#### When could it be used?

* Ranking problems

---

# Usage of common Loss Functions
<div></div>

[Kullback-Leibler Divergence (KLD)](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss) Loss: $L(x, y) = y\cdot(\mathrm{log}y - x)$

```python {all}
# Kullback-Leibler Divergence Loss
import torch
import torch.nn as nn

input = torch.randn(2, 3, requires_grad=True)
target = torch.randn(2, 3)
kl_loss = nn.KLDivLoss(reduction = 'batchmean')
output = kl_loss(input, target)
output.backward()

print('output: ', output)
```

```python {all}
output:  tensor(0.8774, grad_fn=<DivBackward0>)
```

#### When could it be used?

* Approximating complex functions
* Multi-class classification tasks
* If you want to make sure that the distribution of predictions is similar to that of training data

---

# Best practices for Loss Functions (<small>source: [CoTAI lecture](https://hackmd.io/@gianghoangcotai/ryCqF_uO8)</small>)

#### Limitations of loss functions:
A loss function, more or less, cannot totally reflect the our objectives when training a model, in essence. In fact, we have some prior knowledge about “What we want to optimize” and we try to model our prior knowledge by designing some loss function by hand.<br>
* Practical uses of loss functions:
  * Use a composite loss function, i.e. a composition of many different loss functions, to train your model.
* Designing new loss functions:
  * What is the aspect you want the model to learn to optimize, e.g. to address the problem of class imbalance, etc.
  * Try to mathematically model your objective by a function, whose inputs are the predicted segmentation mask and the corresponding ground-truth segmentation mask.