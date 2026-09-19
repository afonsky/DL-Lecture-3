---
layout: center
---

# Backup Slides

---
zoom: 0.8
---

# Expert Systems vs Connectionism

| | **Symbolic AI (Expert Systems)** | **Neural Networks (Connectionism)** |
|---|---|---|
| **Core idea** | Intelligence = manipulation of human-readable symbols and logical rules | Intelligence = learning patterns from data<br> via networks of simple units |
| **Knowledge** | Hand-crafted by domain experts<br> as explicit "if-then" rules | Learned automatically from examples,<br> stored implicitly in weights |
| **Representation** | Interpretable: you can read and audit every rule | Opaque: a trained network is a matrix of numbers — a "black box" |
| **Strengths** | Works well for narrow, well-defined domains | Handles noisy, high-dimensional data;<br> generalizes to unseen examples |
| **Weaknesses** | Brittle: fails on cases not covered by rules; doesn't scale (knowledge bottleneck) | Needs large datasets; hard to interpret; limited theory in that era |

---
zoom: 0.66
---

# Activation Functions: Formulas and Derivatives

| Name | $\sigma(x)$ | $\sigma'(x)$ | Notes |
|---|---|---|---|
| **ReLU** | $\max(x, 0)$ | $\mathbb{1}[x>0]$ | derivative at 0 taken as 0 |
| **Leaky ReLU** | $\max(0,x) + \alpha\min(0,x)$ | $1$ or $\alpha$ | $\alpha = 0.01$ by default |
| **PReLU** | same, $\alpha$ learned | $1$ or $\alpha$ | [He et al. (2015)](https://arxiv.org/abs/1502.01852) |
| **ELU** | $x$ or $\alpha(e^x-1)$ | $1$ or $\alpha e^{x}$ | smooth, mean closer to 0 |
| **Sigmoid** | $\dfrac{1}{1+e^{-x}}$ | $\sigma(x)\,(1-\sigma(x))$ | max derivative $0.25$ at $x=0$ |
| **Tanh** | $\dfrac{1-e^{-2x}}{1+e^{-2x}}$ | $1 - \tanh^2(x)$ | $\tanh(x) + 1 = 2\,\text{sigmoid}(2x)$ |
| **GELU** | $x\,\Phi(x)$ | $\Phi(x) + x\,\varphi(x)$ | $\Phi$ = standard normal CDF |
| **SiLU / Swish** | $x\,\text{sigmoid}(\beta x)$ | $\sigma(\beta x)(1+\beta x(1-\sigma(\beta x)))$ | $\beta = 1$ in PyTorch |
| **Softplus** | $\log(1+e^{x})$ | $\text{sigmoid}(x)$ | a smooth ReLU |

<span class="refs">Read: [d2l.ai 5.1.2](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html#activation-functions) · Docs: [torch.nn non-linear activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)</span>

---
zoom: 0.72
---

# The Chain Rule on a Graph, in General

For tensors $\mathsf{Y} = f(\mathsf{X})$ and $\mathsf{Z} = g(\mathsf{Y})$:

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \operatorname{prod}\!\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right)$$

where $\operatorname{prod}$ means "multiply, after whatever transposes and reshapes are needed". For vectors it is just a matrix product.

### Two rules that cover every graph
* **Along a path**: multiply the local derivatives
* **Over paths**: if a value is used in several places, **add** the gradients arriving from each use

### Reverse mode vs forward mode
| | cost | good when |
|---|---|---|
| **Reverse mode** (backprop) | one pass per **output** | many inputs, one output — i.e. any loss function |
| **Forward mode** | one pass per **input** | few inputs, many outputs |

A neural network has millions of parameters and exactly one scalar loss, so reverse mode wins by a factor of millions.

<span class="refs">Read: [d2l.ai 5.3.3](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html#backpropagation) · [Baydin et al., Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)</span>

---
zoom: 0.78
---

# Xavier Initialization: Where the Formula Comes From

For a layer without a nonlinearity, $o_i = \sum_{j=1}^{n_\mathrm{in}} w_{ij}x_j$, with weights drawn independently with mean 0 and variance $\sigma^2$, and inputs with mean 0 and variance $\gamma^2$:

$$
\mathbb{E}[o_i] = \sum_j \mathbb{E}[w_{ij}]\,\mathbb{E}[x_j] = 0,
\qquad
\operatorname{Var}[o_i] = \sum_j \mathbb{E}[w_{ij}^2]\,\mathbb{E}[x_j^2] = n_\mathrm{in}\sigma^2\gamma^2
$$

<br>

* To keep the **forward** signal from changing scale we want $n_\mathrm{in}\sigma^2 = 1$
* The same argument on the **backward** pass wants $n_\mathrm{out}\sigma^2 = 1$
* Both are impossible at once, so Xavier splits the difference:

$$\tfrac{1}{2}(n_\mathrm{in} + n_\mathrm{out})\,\sigma^2 = 1 \quad\Longleftrightarrow\quad \sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}$$

* For a uniform distribution $U(-a, a)$ with variance $a^2/3$, this becomes $U\!\left(-\sqrt{\tfrac{6}{n_\mathrm{in}+n_\mathrm{out}}},\ \sqrt{\tfrac{6}{n_\mathrm{in}+n_\mathrm{out}}}\right)$
* ReLU zeroes roughly half the units, which halves the variance — hence He's extra factor of 2

<span class="refs">Read: [d2l.ai 5.4.2.2](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html#xavier-initialization) · [Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html)</span>

---
zoom: 0.8
---

# Dropout from Scratch

```python
def dropout_layer(X, p):
    assert 0 <= p <= 1
    if p == 1:                       # everything is dropped
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > p).float()
    return mask * X / (1.0 - p)      # rescale so E[h'] = h

X = torch.arange(8, dtype=torch.float32).reshape(1, 8)
print(dropout_layer(X, 0.0))   # tensor([[0., 1., 2., 3., 4., 5., 6., 7.]])
print(dropout_layer(X, 0.5))   # tensor([[0., 0., 4., 6., 0., 0., 0., 14.]])  <- one draw
print(dropout_layer(X, 1.0))   # tensor([[0., 0., 0., 0., 0., 0., 0., 0.]])
```

<br>

* The mask is redrawn **on every forward pass**, so every minibatch sees a different subnetwork
* The division by $1-p$ is what lets you leave the test-time network untouched ("inverted dropout"); this is what PyTorch does
* In a model you must guard it yourself — `if self.training: h = dropout_layer(h, p)` — which is exactly what `nn.Dropout` handles for you

<span class="refs">Read: [d2l.ai 5.6.2](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html#implementation-from-scratch)</span>

---
zoom: 0.72
---

# Counting Parameters and Activations

For a fully connected layer $n_\mathrm{in} \to n_\mathrm{out}$: $\ n_\mathrm{in} \cdot n_\mathrm{out} + n_\mathrm{out}$ parameters.

| Network | Parameters | Activations stored per example |
|---|---|---|
| `Linear(784, 10)` *(Lecture 2)* | $7{,}850$ | 10 |
| `Linear(784,256) + Linear(256,10)` | $200{,}960 + 2{,}570 = 203{,}530$ | $256 + 10 = 266$ |
| `784→256→256→10` | $203{,}530 + 65{,}792 = 269{,}322$ | $522$ |

<br>

* With `batch_size=256` and float32 the third network's activations cost $256 \times 522 \times 4 \approx 0.5$ MB — trivial here, but the same arithmetic on a transformer with a 4096-token context fills an 80 GB GPU
* Parameters are stored once; **activations are stored per example in the batch**
* AdamW keeps two extra tensors the size of the parameters: plan for ~4× the parameter memory

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)   # count them
```

<span class="refs">Read: [d2l.ai 5.3.4](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html#training-neural-networks)</span>
