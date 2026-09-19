---
zoom: 0.9
---

# Activation Functions

<figure>
  <img src="/activations.svg" style="width: 900px !important; margin: 0 auto;">
</figure>

<div class="grid grid-cols-2 gap-10">
<div>

* Applied **elementwise**, right after the affine transform

</div>
<div>

* The **orange curve** is the factor a gradient is multiplied by on the way back. Where it is ≈ 0, learning stops

</div>
</div>

<span class="refs">Read: [d2l.ai 5.1.2](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html#activation-functions) · Watch: [StatQuest: Neural Networks Pt. 3, ReLU in Action](https://www.youtube.com/watch?v=68BZ5f7P94E)</span>

<!--
Point at the orange curves one by one. Sigmoid: 0.25 at best, and ~0 almost everywhere else.
ReLU: exactly 0 or exactly 1, nothing in between. That single fact is most of this lecture.
-->

---
zoom: 0.95
---

# Why ReLU Became the Default

<div class="grid grid-cols-2 gap-10">
<div>

### $\operatorname{ReLU}(x) = \max(x, 0)$

<v-clicks>

* **Derivative 0 or 1** — a gradient is blocked or passed on unchanged, never *shrunk*
* **Cheap**: one comparison, no $\exp$
* **Sparse**: about half the units output exactly 0

</v-clicks>
</div>
<div>

### The failure mode: dying ReLU
<v-clicks>

* Input always negative → gradient 0 **forever**
* Usually caused by too large a learning rate
* Fix: `LeakyReLU(0.01)` or `ELU`

</v-clicks>

<v-click>

#### Sigmoid and tanh are not obsolete
They moved from hidden layers to **outputs** and to **gates** in LSTMs and GRUs.

</v-click>
</div>
</div>

<!--
Say, don't show: shrinking is exactly what kills deep sigmoid networks - that is the whole
story of the next section. ReLU is not differentiable at 0; everyone takes the derivative to
be 0 there and nothing bad happens. Dying ReLU is also caused by a large negative bias.
-->

<span class="refs">Papers: [Nair & Hinton (2010)](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) · [Glorot, Bordes & Bengio (2011), Deep Sparse Rectifier Networks](https://proceedings.mlr.press/v15/glorot11a.html) · [He et al. (2015), PReLU](https://arxiv.org/abs/1502.01852)</span>

---
zoom: 0.7
---

# Which Activation Should I Use?

| Where | Use | Why |
|---|---|---|
| Hidden layers of an MLP or CNN | **ReLU** | fast, no saturation for $x>0$ — the safe default |
| Hidden layers of a transformer | **GELU** or **SiLU / Swish** | smooth near 0, small but consistent gains at scale |
| You suspect dead units | **LeakyReLU**, **ELU** | a small slope for $x<0$ keeps the gradient alive |
| Output, binary classification | **sigmoid** — in `BCEWithLogitsLoss` | a logit becomes a probability *(Lecture 2)* |
| Output, multi-class | **softmax** — in `CrossEntropyLoss` | probabilities over $C$ classes *(Lecture 2)* |
| Output, regression | **none** | the prediction must be free to take any value |

<v-clicks>

* **Never** put softmax or sigmoid in the model *and* use `CrossEntropyLoss` / `BCEWithLogitsLoss` — they include it
* The activation is rarely what to tune first

</v-clicks>

<span class="refs">Papers: [Hendrycks & Gimpel (2016), GELU](https://arxiv.org/abs/1606.08415) · [Ramachandran, Zoph & Le (2017), Swish](https://arxiv.org/abs/1710.05941) · Docs: [torch.nn activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)</span>

---
zoom: 0.95
---

# Five Minutes in the Playground

<div class="grid grid-cols-[3fr_2fr] gap-8">
<div>

### Three experiments on [playground.tensorflow.org](https://playground.tensorflow.org/)

<v-clicks>

1. **Kill the nonlinearity.** *Circle*, Activation = **Linear**, four hidden layers → the boundary stays a straight line
2. **Bring ReLU back.** Same network, **ReLU** → the boundary bends
3. **Break it.** *Spiral*, learning rate 3 → diverges; 0.001 → nothing happens

</v-clicks>
</div>
<div>

<v-click>

### What to look for
* Each hidden unit shows the region it responds to — **the features the network invented**
* The output layer just combines them linearly

</v-click>
</div>
</div>

<span class="refs">Try also: [A. Karpathy, ConvNetJS 2-D classification demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html)</span>

<!--
Do experiment 1 live - 30 seconds, and it is the thing students remember. It is exactly the
collapse we proved two slides ago. In experiment 2, hover over a hidden unit to see the
half-plane it contributes. Experiment 3 replays Lecture 2's learning-rate picture, live.
Also worth saying: deeper is not automatically better - with few examples the spiral overfits
visibly. If time is short, show experiment 1 only and assign the rest.
-->
