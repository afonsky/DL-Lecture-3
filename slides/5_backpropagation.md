---
layout: center
---

<center>

# Forward and Backward Propagation

# What `loss.backward()` really does
</center>

---
zoom: 0.95
---

# Forward Propagation

<div class="grid grid-cols-[3fr_3fr] gap-2">
<div>

**Forward propagation** = compute and **store** every intermediate value, input → loss:

$$
\mathbf{z} = \mathbf{W}^{(1)}\mathbf{x}, \quad
\mathbf{h} = \phi(\mathbf{z}), \quad
\mathbf{o} = \mathbf{W}^{(2)}\mathbf{h}, \quad \\
L = l(\mathbf{o}, y)
$$

with weight decay *(Lecture 2)*:<br> $J = L + s$, $\ s = \frac{\lambda}{2}(\|\mathbf{W}^{(1)}\|_\mathrm{F}^2 + \|\mathbf{W}^{(2)}\|_\mathrm{F}^2)$.

<v-clicks>

* A **computational graph** makes the dependencies explicit: squares are values, circles are operations
* *Store* is doing real work here — every value must survive until the backward pass

</v-clicks>

<!--
Squares are variables, circles are operators; data flows right and up. Everything drawn here
has to stay in memory, which is the subject of the memory slide in a few minutes.
-->
</div>
<div>

<br>
  <figure>
    <img src="/forward.svg" style="width: 480px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 8px">Image source:
      <a href="https://d2l.ai/chapter_multilayer-perceptrons/backprop.html">d2l.ai Fig. 5.3.1 Computational graph of forward propagation</a>
    </figcaption>
  </figure>

<br>
<br>
<br>
<br>
<br>

<span class="refs">Read: [d2l.ai 5.3.1–5.3.2](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html)</span>
</div>
</div>


---
zoom: 0.99
---

# Why Not Just Nudge Every Weight?

<div class="grid grid-cols-2 gap-10">
<div>

### The obvious method: finite differences
$$\frac{\partial L}{\partial w_i} \approx \frac{L(w_i + \varepsilon) - L(w_i - \varepsilon)}{2\varepsilon}$$

<v-clicks>

* Correct, trivial:<br> **two forward passes per parameter**
* Our $203\,530$-parameter MLP:<br> $\approx 407\,000$ passes per step
* A 7-billion-parameter model:<br> $1.4\times10^{10}$ per step

</v-clicks>
</div>
<div>

### What backpropagation costs
<v-clicks>

* **One** backward pass, **all** the derivatives, ~2× a forward pass — whatever the parameter count

</v-clicks>

<v-click>

#### This is not an optimization — it is why deep learning exists.

</v-click>

<v-click>

#### Still useful as a gradient check
`torch.autograd.gradcheck`, on a tiny input, when you write a custom layer.

</v-click>
</div>
</div>

<br>

<span class="refs">Read: [d2l.ai 2.5 Automatic Differentiation](https://d2l.ai/chapter_preliminaries/autograd.html) · [CS231n, Backpropagation](https://cs231n.github.io/optimization-2/)</span>

<!--
Finite differences are also numerically delicate: too small an epsilon gives rounding noise,
too large gives the wrong slope. And before autodiff, papers spent pages deriving update rules
by hand, and every architecture change meant redoing them.
-->

---
zoom: 0.87
---

# Backpropagation = the Chain Rule, Backwards

#### At every node: **incoming gradient × local derivative = outgoing gradient**

<figure>
  <img src="/backprop_chain.svg" style="width: 900px !important; margin: 0 auto;">
</figure>

<br>

<div class="grid grid-cols-[3fr_2fr] gap-10">
<div>

<v-clicks>

* Start at the loss ($\partial L/\partial L = 1$) and walk **backwards**
* Each node needs only **its own** derivative — that is what makes it composable
* A value feeding several nodes: **multiply along paths, add over paths**

</v-clicks>
</div>
<div>

<v-click>

**A weight's gradient = the gradient arriving there × the value it multiplied on the way in.**

</v-click>
</div>
</div>

<span class="refs">Watch: [A. Karpathy, The spelled-out intro to backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) · [3Blue1Brown, Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) · Read: [CS231n](https://cs231n.github.io/optimization-2/)</span>

<!--
Do this one on the whiteboard as well, right to left, saying each multiplication out loud.
If they can do this chain, they can do any network: the rest is bookkeeping.
-->

---
zoom: 0.99
---

# The Same Thing for a Whole Layer

<div class="grid grid-cols-2 gap-8">
<div>

$$
\begin{aligned}
\frac{\partial J}{\partial \mathbf{o}} &= \frac{\partial L}{\partial \mathbf{o}}\\[6pt]
\frac{\partial J}{\partial \mathbf{W}^{(2)}} &= \frac{\partial J}{\partial \mathbf{o}}\,\mathbf{h}^\top + \lambda\mathbf{W}^{(2)}\\[6pt]
\frac{\partial J}{\partial \mathbf{h}} &= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}\\[6pt]
\frac{\partial J}{\partial \mathbf{z}} &= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'(\mathbf{z})\\[6pt]
\frac{\partial J}{\partial \mathbf{W}^{(1)}} &= \frac{\partial J}{\partial \mathbf{z}}\,\mathbf{x}^\top + \lambda\mathbf{W}^{(1)}
\end{aligned}
$$

</div>
<div>

<v-clicks>

* The same three moves as the scalar example, in matrix form
* **Weight gradient** = (gradient arriving) × (input that arrived)$^\top$ — it automatically has the weight matrix's shape, a free sanity check
* **Going back through a layer**<br> = multiply by $\mathbf{W}^\top$
* Frameworks implement these five lines per layer type, and nothing else

</v-clicks>
</div>
</div>

<span class="refs">Read: [d2l.ai 5.3.3](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html#backpropagation) · [Y. LeCun et al., Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)</span>

<!--
The circled-dot is elementwise because phi acts elementwise. The lambda-W terms are weight
decay from Lecture 2, appearing exactly where you would expect them.
-->

---
zoom: 0.95
---

# Training Costs Memory, Not Just Time

<figure>
  <img src="/training_memory.svg" style="width: 830px !important; margin: 0 auto;">
</figure>

<div class="grid grid-cols-2 gap-10">
<div>

* The backward pass needs every forward value, so activations **stay in memory** until it finishes
* This is why **training** [OOMs](https://en.wikipedia.org/wiki/Out_of_memory) and **inference** of the same model does not

</div>
<div>

<v-click>

**When you hit CUDA OOM:** lower the batch size · `torch.no_grad()` for evaluation · gradient checkpointing · mixed precision

</v-click>

<br>

<span class="refs">Read: [d2l.ai 5.3.4](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html#training-neural-networks) · Docs: [torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html)</span>
</div>
</div>


<!--
Left bar: the four things that occupy GPU memory while training. Optimizer state is the
biggest single item here because AdamW keeps two moments per parameter.
Right: activations are linear in depth AND in batch size - that product is what kills you,
not the parameter count. Batch size is the cheapest knob because it is a pure multiplier.
Common bug to mention: appending `loss` instead of `loss.item()` keeps the whole graph alive,
so memory grows every iteration.
Gradient checkpointing: ~30% more compute, large memory saving.
-->

---
zoom: 0.9
---

# Autograd in PyTorch

<div class="grid grid-cols-[3fr_2fr] gap-6">
<div>

```python {all|3-6|8-11|13-14|16-18|all}
import torch

x = torch.tensor([2.0])                       # data: no gradient
y = torch.tensor([1.0])
w = torch.tensor([0.5], requires_grad=True)   # parameters:
v = torch.tensor([3.0], requires_grad=True)   # track them

z = w * x                 # forward: the graph is built as you go
h = torch.relu(z)
o = v * h
loss = (o - y) ** 2

loss.backward()           # one backward pass -> every gradient
print(w.grad, v.grad)     # tensor([24.])  tensor([4.])

with torch.no_grad():     # updating is not part of the graph
    w -= 0.01 * w.grad
    w.grad.zero_()        # gradients ACCUMULATE; clear them
```

</div>
<div>

<v-clicks>

* The same numbers as the figure, now computed for you
* `requires_grad=True` marks the leaves — `nn.Linear` does it for you
* The graph is rebuilt **every forward pass**, so `if`s and loops are fine

</v-clicks>
</div>
</div>

<v-click>

#### Three rules that prevent most autograd bugs
**1.** `zero_grad()` before every `backward()` — gradients add up.  **2.** Evaluate under `torch.no_grad()`.  **3.** Store logs with `.item()`.

</v-click>

<span class="refs">Read: [d2l.ai 2.5](https://d2l.ai/chapter_preliminaries/autograd.html) · [PyTorch autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) · Build one yourself: [A. Karpathy, micrograd](https://github.com/karpathy/micrograd) (≈150 lines)</span>

<!--
Click through the code block. Punchline of the last reveal: the update runs under no_grad
because changing w is not an operation we ever want to differentiate.
-->
