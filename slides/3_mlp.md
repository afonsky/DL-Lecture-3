---
layout: center
---

<center>

# Multilayer Perceptrons

# Adding a hidden layer
</center>

---
zoom: 0.9
---

# Where a Linear Model Runs Out

### A linear model assumes **monotonicity**: raising a feature always pushes the output the same way

<figure>
  <img src="/linear_limits.svg" style="width: 800px !important; margin: 0 auto;">
</figure>

<div class="grid grid-cols-2 gap-10">
<div>

* **Left**: not monotone at all — risk rises on *both* sides of 37 °C
* **Right**: monotone, but the same \$50k is worth **+0.057** or **+0.0003**

</div>
<div>

<v-click>

* Fixable by hand — square it, use $|t-37|$: that is **feature engineering**
* CIFAR example: Which hand-made feature says "dog"? Inverting an image keeps its class

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 5.1.1](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html#hidden-layers) · Watch: [A. Ng, Why Non-linear Activation Functions](https://www.youtube.com/watch?v=NkOv_k7r6no)</span>

<!--
Both panels are fixable by feature engineering - ask the room how, they will get it.
Then the pixel question: should brightening pixel (13,17) always make "dog" more likely?
Inverting an image preserves the category, so no single pixel has a fixed meaning, and the
significance of a pixel depends on its neighbours. Nobody can hand-craft that, so we learn it.
-->

---
zoom: 0.98
---

# The XOR Problem, Solved

#### Minsky & Papert's 1969 example takes **one hidden layer of two ReLU units**

<figure>
  <img src="/xor_mlp.svg" style="width: 870px !important; margin: 0 auto;">
</figure>

<div class="text-center" style="font-size: 0.78em">

$h_1 = \mathrm{ReLU}(x_1 + x_2), \qquad h_2 = \mathrm{ReLU}(x_1 + x_2 - 1), \qquad o = h_1 - 2 h_2$

</div>

<v-click>

#### The hidden layer does not classify — it **re-describes** the data so that a line works. The output layer is still the linear model of Lecture 2.

</v-click>

<span class="refs">Watch: [Y. LeCun, NYU Deep Learning](https://atcold.github.io/NYU-DLSP21/) · [StatQuest: The Essential Main Ideas of Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw) · Play: [TensorFlow Playground](https://playground.tensorflow.org/)</span>

<!--
Evaluate the network out loud on (1,1): h1 = 2, h2 = 1, o = 2 - 2 = 0 -> class 0.
Then on (0,1): h1 = 1, h2 = 0, o = 1 -> class 1.
Panel (b) is the money shot: the two y=1 points land on the *same* point.
-->

---
zoom: 0.93
---

# Incorporating Hidden Layers

<div class="grid grid-cols-[3fr_4fr] gap-8">
<div>

$$\mathbf{H} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \quad \mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$$

$\mathbf{H}$ is the **hidden representation**;<br> both layers are fully connected (`nn.Linear`).

<br>

### Vocabulary
* **Width** = units per layer
* **Depth** = number of layers
* The input layer computes nothing<br> → this is a **2-layer** network
* MLP = stacked fully connected layers

<br>

<span class="refs">Read: [d2l.ai 5.1.1.2](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html#incorporating-hidden-layers)</span>
</div>
<div>
  <figure>
    <img src="/mlp.svg" style="width: 390px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 6px">Image source:
      <a href="https://d2l.ai/chapter_multilayer-perceptrons/mlp.html">d2l.ai Fig. 5.1.1 An MLP with a hidden layer of five hidden units</a>
    </figcaption>
  </figure>

<br>

<v-click>

#### Counting parameters

$28\times28=784 \to 256 \to 10$, as in [MNIST](https://en.wikipedia.org/wiki/MNIST_database):

$784\times256 + 256 = 200\,960$

$256\times10 + 10 = 2\,570$

$203\,530$ parameters, 99 % of them in the 1st layer

</v-click>
</div>
</div>

---
zoom: 0.99
---

# Two Linear Layers Are Still One Linear Layer

$$
\mathbf{O} = (\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)}
           = \mathbf{X}\underbrace{\mathbf{W}^{(1)}\mathbf{W}^{(2)}}_{\mathbf{W}} + \underbrace{\mathbf{b}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}}_{\mathbf{b}}
$$


<div class="grid grid-cols-[3fr_2fr] gap-8">
<div>

<br>

<v-clicks>

* An affine function of an affine function is affine:<br> **we gained nothing but parameters**

* Worse: $\operatorname{rank}(\mathbf{W}^{(1)}\mathbf{W}^{(2)}) \le \min(d, h, q)$<br> — a narrow hidden layer is a **bottleneck**

</v-clicks>
</div>
<div>

<v-click>
<br>

#### The missing ingredient
A nonlinear **activation** $\sigma$, applied elementwise:

$$
\begin{aligned}
\mathbf{H} &= \sigma\big(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}\big) \\
\mathbf{O} &= \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}
\end{aligned}
$$

Now the layers cannot be merged, and depth starts to pay.

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 5.1.1.3](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html#from-linear-to-nonlinear)</span>

<!--
The single most important algebraic fact of the lecture. If they remember one thing:
without sigma, depth costs parameters and buys nothing.
-->

---
zoom: 0.98
---

# What a ReLU Network Computes

<figure>
  <img src="/relu_piecewise.svg" style="width: 830px !important; margin: 0 auto;">
</figure>

<div class="grid grid-cols-2 gap-10">
<div>

* One ReLU unit = **one kink**: the network is **piecewise linear**
* More units → more pieces → closer fit

</div>
<div>

<v-click>

* Enough kinks pass through every training point, noise included — **overfitting** *(Lecture 2)*
* Outside the data it is just a straight line — **bad extrapolation**

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 5.1, Exercise 4](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html#exercises) · [Montúfar et al., On the Number of Linear Regions of Deep Neural Networks](https://arxiv.org/abs/1402.1869)</span>

---
zoom: 0.99
---

# How Powerful Is One Hidden Layer?

<br>

<div class="grid grid-cols-2 gap-10">
<div>

### Universal approximation *(Lecture 1)*
One hidden layer, enough units → **any** continuous function on a compact domain.

### What it does **not** say
<v-clicks>

* How many units "enough" is
* That gradient descent will **find** them
* That the result will **generalize**

</v-clicks>
</div>
<div>

### So why go deep?
<v-clicks>

* **Exponentially fewer** units deep than wide
* A hierarchy: edges → parts → objects *(Lecture 1)*
* And depth is what makes training hard

</v-clicks>
<br>
<v-click>

> *"It can express any computable program. Coming up with a program that meets your specifications is the hard part."*
> <small>— d2l.ai 5.1.1.4, on neural networks and the C language</small>

<!--
Cybenko (1989) and Hornik et al. (1989); links on the refs line. Stress the gap the theorem
leaves: existence is not construction, and "enough" can be absurdly many units.
-->

</v-click>
</div>
</div>

<br>

<span class="refs">Read: [d2l.ai 5.1.1.4](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html#universal-approximators) · [M. Nielsen, A visual proof](http://neuralnetworksanddeeplearning.com/chap4.html) · Watch: [Y. Abu-Mostafa, Learning From Data, Lec. 10](https://work.caltech.edu/lectures.html)</span>
