---
layout: center
---

<center>

# Putting It Together

# An MLP that actually trains
</center>

---
zoom: 0.82
---

# An MLP in PyTorch

<div class="grid grid-cols-[3fr_2fr] gap-6">
<div>

```python {all|3-8|10-11|13-19|21-24|all}
import torch; from torch import nn

model = nn.Sequential(              # 784 -> 256 -> 10
    nn.Flatten(),                   # 28x28 image -> 784 vector
    nn.Linear(784, 256),            # hidden layer
    nn.ReLU(),                      # the nonlinearity
    nn.Dropout(0.2),
    nn.Linear(256, 10))             # logits: NO softmax here

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
loss_fn = nn.CrossEntropyLoss()     # applies softmax internally

for epoch in range(num_epochs):
    model.train()                            # dropout ON
    for X, y in train_loader:
        opt.zero_grad()                      # gradients accumulate!
        loss = loss_fn(model(X), y)          # 1. forward  2. loss
        loss.backward()                      # 3. backward
        opt.step()                           # 4. update

    model.eval()                             # dropout OFF
    with torch.no_grad():                    # no graph, no memory
        val = sum(loss_fn(model(X), y).item()
                  for X, y in val_loader) / len(val_loader)
```

</div>
<div>

<v-clicks>

* Lecture 2's `nn.Linear(784, 10)` became **three extra lines**
* `nn.Sequential` calls the layers in order — the graph comes for free
* The four numbered lines also train an LLM
* `train()` / `eval()` switch dropout, **not** gradients

</v-clicks>
</div>
</div>

<span class="refs">Read: [d2l.ai 5.2](https://d2l.ai/chapter_multilayer-perceptrons/mlp-implementation.html) · [S. Raschka, PyTorch in One Hour](https://sebastianraschka.com/teaching/pytorch-1h/) · [PyTorch quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)</span>

---
zoom: 0.95
---

# Choosing Width and Depth

<div class="grid grid-cols-2 gap-10">
<div>

### Starting points that usually work
<v-clicks>

* **Width**: powers of two — 128, 256, 512 (GPU alignment)
* **Depth**: **1–2 hidden layers** for tabular or vector inputs
* Go **wider before deeper**
* Never a 1-unit hidden layer — a rank-1 bottleneck

</v-clicks>
</div>
<div>

### The order to tune in
<v-clicks>

1. **Learning rate** — more than everything below combined *(Lecture 2)*
2. **Width and depth** — capacity
3. **Weight decay and dropout** — once the model can overfit
4. Batch size, activation, optimizer — leave at the defaults

</v-clicks>

<v-click>

#### Copy before you invent
Start from a published architecture. Change one thing at a time.

</v-click>
</div>
</div>

<span class="refs">Read: [Y. Bengio, Practical Recommendations for Gradient-Based Training](https://arxiv.org/abs/1206.5533) · [Google, Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) · [d2l.ai 5.2 Exercises](https://d2l.ai/chapter_multilayer-perceptrons/mlp-implementation.html#exercises)</span>

---
zoom: 0.92
---

# A Debugging Checklist

<div class="grid grid-cols-2 gap-10">
<div>

### Before you train
<v-clicks>

* **Look at your data.** Shapes, a few examples, the label distribution
* **Initial loss $\approx \ln C$** — 2.30 for 10 classes *(Lecture 2)*
* **Overfit a single batch** of ~8 examples to zero loss, regularization off

</v-clicks>
</div>
<div>

### While it trains
<v-clicks>

* Plot **both** curves. Training up → bug; validation up → overfitting
* Gradient norms: `NaN` → exploding, $\approx 0$ → vanishing
* Four classic bugs: no `zero_grad()`, no `eval()`, double softmax, unshuffled data
* **Fix the seed**, change one thing per experiment

</v-clicks>
</div>
</div>

<v-click>

> *"The qualities that correlate most strongly to success in deep learning are patience and attention to detail."* <small>— A. Karpathy, [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)</small>

</v-click>

<span class="refs">Read: [A. Karpathy, A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) · [Google Tuning Playbook](https://github.com/google-research/tuning_playbook)</span>

<!--
The "overfit a single batch" trick is the single most useful thing on this slide: if you cannot,
the bug is in the model, the loss or the data, not in the hyperparameters. Ask them to run it in
HW1 before they report that "the model does not learn".
If the initial loss is not ~ln C, the labels, the loss or the output layer are wrong.
Also: keep a baseline number to beat.
-->

---
zoom: 0.92
---

# Where MLPs Live Today

<div class="grid grid-cols-2 gap-10">
<div>

<v-clicks>

* **Tabular data**: a baseline, but boosted trees usually win *(Lecture 1)*
* **As a head** on a ResNet or a BERT encoder
* **Inside every transformer block**: a two-layer MLP, $d \to 4d \to d$
* Roughly **two thirds of an LLM's parameters**

</v-clicks>
</div>
<div>

<v-click>

#### Not a historical curiosity — the component you will meet most often for the rest of the course.

</v-click>

<v-click>

### What changes later
* **Convolutions** when the input has spatial structure
* **Recurrence and attention** when it is a sequence
* Forward, loss, backward, step — unchanged

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 11.7 The Transformer Architecture](https://d2l.ai/chapter_attention-and-transformers/transformer.html) · [Grinsztajn et al. (2022) on tabular data](https://arxiv.org/abs/2207.08815) · Watch: [A. Karpathy, Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)</span>
