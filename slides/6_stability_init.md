---
layout: center
---

<center>

# Making Deep Networks Trainable

# Stability and initialization
</center>

---
zoom: 0.82
---

# Vanishing and Exploding Gradients

<div class="grid grid-cols-[2fr_3fr] gap-6">
<div>

Going back through $L$ layers multiplies $L$ Jacobians:

$$\frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(\ell)}} = \mathbf{M}^{(L)} \cdots \mathbf{M}^{(\ell+1)} \mathbf{v}^{(\ell)}$$

A long product either **collapses to 0** or **runs away**.

<v-clicks>

* Sigmoid's derivative never exceeds **0.25** → at most $0.25^{10}\approx 10^{-6}$ over ten layers
* **Vanishing**: early layers stop learning, the loss just stalls
* **Exploding**: the loss becomes `NaN`. Remedy: `clip_grad_norm_`

</v-clicks>
</div>
<div>
<figure>
  <img src="/vanishing_gradients.svg" style="width: 580px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 4px">
    Right: measured on a 20-layer, 100-unit-wide network with the same random initialization for both activations; median over 12 draws.
  </figcaption>
</figure>

<br>

<v-click>

#### This is why the 1986 promise took twenty more years: backpropagation was correct, but the gradient never reached the bottom of a deep sigmoid network.

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 5.4.1](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) · [Hochreiter (1991)](https://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf) · [Bengio, Simard & Frasconi (1994)](https://ieeexplore.ieee.org/document/279181)</span>

<!--
Tie this back to the history section explicitly: Rumelhart-Hinton-Williams 1986 gave the
algorithm; sigmoid plus bad initialization meant it did not scale. ReLU and He init fixed it.
-->

---
zoom: 0.92
---

# Never Start All the Weights at the Same Value

<figure>
  <img src="/symmetry_breaking.svg" style="width: 800px !important; margin: 0 auto;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; text-align: center">
    The same 2–8–1 ReLU network, trained twice with full-batch gradient descent. Each curve is one hidden unit's incoming weight.
  </figcaption>
</figure>

<div class="grid grid-cols-2 gap-10">
<div>

<v-clicks>

* Equal weights → equal activations → **equal gradients** → still equal after the update
* A layer of width 8 behaves like width **1**, forever

</v-clicks>
</div>
<div>

<v-clicks>

* Random init is what tells the units apart; **biases may start at zero**
* So the question is not *whether* to randomize, but **at what scale**

</v-clicks>
</div>
</div>

<span class="refs">Read: [d2l.ai 5.4.1.3](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html#breaking-the-symmetry)</span>

<!--
Walk the left panel: eight lines are drawn, you see one, because gradient descent
cannot distinguish units it started identical. Dropout would break the tie; plain SGD never does.
Then: the scale question is the next slide.
-->

---
zoom: 0.84
---

# Xavier and He Initialization

<div class="grid grid-cols-[3fr_4fr] gap-6">
<div>

**Goal:** keep the variance of the activations — and of the gradients — constant across layers.

$\operatorname{Var}[o] = n_\mathrm{in}\sigma^2\gamma^2$, so we want $n_\mathrm{in}\sigma^2 \approx 1$ forward and $n_\mathrm{out}\sigma^2 \approx 1$ backward. You cannot have both:

* **Xavier / Glorot** (tanh, sigmoid): $\sigma^2 = 2/(n_\mathrm{in} + n_\mathrm{out})$
* **He / Kaiming** (ReLU): $\sigma^2 = 2/n_\mathrm{in}$

</div>
<div>
<figure>
  <img src="/init_scales.svg" style="width: 500px !important;">
</figure>

```python
for m in model.modules():                 # PyTorch defaults are
    if isinstance(m, nn.Linear):          # sensible; override when
        nn.init.kaiming_normal_(          # you need to
            m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)
```

<v-click>

#### Over 20 layers: half the He scale → $10^{-6}$, 1.4× the He scale → $10^{2}$. Only the initial scale differs.

<!--
The extra factor 2 in He compensates for ReLU zeroing roughly half the signal.
PyTorch defaults are already sensible; you override when you need to.
-->

</v-click>
</div>
</div>

<span class="refs">Papers: [Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) · [He et al. (2015)](https://arxiv.org/abs/1502.01852) · Read: [d2l.ai 5.4.2](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html#parameter-initialization) · Docs: [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) · The derivation is in the backup slides</span>

---
zoom: 0.88
---

# What Actually Made Deep Networks Trainable

<div class="grid grid-cols-2 gap-10">
<div>

### Covered today
* **ReLU** — the shrinking factor at every layer
* **Xavier / He init** — the signal scale at step 0
* **Gradient clipping** — one bad batch killing the run

### Coming later
* **Batch / layer norm** — scale drift *during* training
* **Residual connections** — a gradient path around the layers
* **Adam / AdamW**, **warmup and LR decay**

</div>
<div>

<v-click>

#### None of these changes what the network can *represent*.
They change whether gradient descent can **find** good weights.

</v-click>

<v-click>

### Two symptoms
* Loss flat from epoch 0 → vanishing gradients, dead units, tiny learning rate
* Loss becomes `NaN` → exploding gradients, huge learning rate

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 5.4 Summary](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html#summary) · [Y. LeCun et al., Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) · [Y. Bengio, Practical Recommendations](https://arxiv.org/abs/1206.5533)</span>
