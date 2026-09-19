---
layout: center
---

<center>

# Regularization for Deep Networks

# Early stopping and dropout
</center>

---
zoom: 0.86
---

# Deep Networks Overfit — Watch the Two Curves

<figure>
  <img src="/regularize_curves.svg" style="width: 850px !important; margin: 0 auto;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 2px; text-align: center">
    A 2–96–96–1 ReLU network trained with minibatch SGD on 80 noisy points; validation measured on 1500 fresh points.
  </figcaption>
</figure>

<div class="grid grid-cols-2 gap-10">
<div>

* **Left**: training loss keeps falling; validation bottoms out at **epoch 67**, then climbs 58 %
* The model is memorizing the noise in 80 points *(Lecture 2)*

</div>
<div>

<v-click>

* **Early stopping**: keep the **best checkpoint**, stop after `patience` epochs with no gain
* The cheapest regularizer there is — and it saves 1100 wasted epochs

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 5.5.3 Early Stopping](https://d2l.ai/chapter_multilayer-perceptrons/generalization-deep.html#early-stopping) · Watch: [A. Ng, Other Regularization Methods (early stopping)](https://www.youtube.com/watch?v=BOCLq2gpcGU)</span>

<!--
Stress: the left panel is the picture you will see in TensorBoard. Learning to read it is
worth more than any single technique in this lecture.
-->

---
zoom: 0.86
---

# Dropout

<div class="grid grid-cols-[3fr_2fr] gap-16">
<div>

**Idea:** inject noise *inside* the network — zero each hidden unit with probability $p$, rescale the survivors so nothing changes on average:

$$
h' = \begin{cases} 0 & \text{with probability } p \\[2pt] \dfrac{h}{1-p} & \text{otherwise} \end{cases}
\qquad \Rightarrow \qquad \mathbb{E}[h'] = h
$$

<v-clicks>

* A different random subnetwork every minibatch → the output layer **cannot rely on any particular unit**
* It destroys **co-adaptation**: units that only make sense next to three specific neighbours

</v-clicks>
</div>
<div>
  <figure>
    <img src="/dropout_1.svg" style="width: 280px !important;">
  </figure>

<br>

  <figure>
    <img src="/dropout_2.svg" style="width: 280px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 8px">Image source:
      <a href="https://d2l.ai/chapter_multilayer-perceptrons/dropout.html">d2l.ai Fig. 5.6.1 MLP before and after dropout</a>
    </figcaption>
  </figure>
</div>
</div>

<!--
Equivalent reading worth saying out loud: you train an ensemble of exponentially many thinned
networks and average them at test time. Bishop (1995) proved the ancestor: training with input
noise is equivalent to Tikhonov regularization.
-->

<span class="refs">Paper: [Srivastava, Hinton et al. (2014), JMLR](https://jmlr.org/papers/v15/srivastava14a.html) · Read: [d2l.ai 5.6](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html) · Watch: [A. Ng, Dropout Regularization](https://www.youtube.com/watch?v=D8PJAL-MZv8)</span>

---
zoom: 0.88
---

# Dropout in Practice

<div class="grid grid-cols-2 gap-8">
<div>

```python
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 10))

model.train()   # dropout ON:  zeroed and rescaled
model.eval()    # dropout OFF: the full network
```

<v-clicks>

* **After** the activation of each hidden layer, never on the output
* Typical $p$: **0.5** wide, **0.1–0.3** narrow or near the input
* **Forgetting `model.eval()` is one of the commonest bugs in student code**

</v-clicks>
</div>
<div>

### What the previous figure showed
<v-clicks>

* Final train/validation gap **0.32** → **0.06**
* But the *best* validation loss barely moved: **0.33** vs **0.34**
* Dropout bought **robustness to training too long**, not a better optimum

</v-clicks>

<v-click>

#### Where it is used today
Standard in **transformers**; in modern CNNs largely replaced by batch norm plus augmentation.

</v-click>
</div>
</div>

<span class="refs">Docs: [nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) · The from-scratch version is in the backup slides</span>

---
zoom: 0.74
---

# The Regularization Toolbox

| Tool | What it constrains | In PyTorch | Introduced |
|---|---|---|---|
| **More data / augmentation** | the problem itself | `torchvision.transforms` | — |
| **Early stopping** | number of updates | your training loop | today |
| **Weight decay ($\ell_2$)** | size of the weights | `AdamW(..., weight_decay=1e-2)` | Lecture 2 |
| **Dropout** | co-adaptation of units | `nn.Dropout(p)` | today |
| **A smaller network** | raw capacity | fewer / narrower layers | Lecture 2 |
| **Label smoothing** | over-confidence | `CrossEntropyLoss(label_smoothing=0.1)` | Lecture 2 |
| **Normalization layers** | scale of the activations | `nn.LayerNorm`, `nn.BatchNorm1d` | later |

<v-clicks>

* **Order to try them in:** more data → early stopping → weight decay → dropout → a smaller model. Change **one** at a time
* All are hyperparameters → tuned on validation data, never on the test set *(Lecture 2)*

</v-clicks>

<!--
Worth knowing, not worth a bullet: with deep networks the usual strength of l2 is nowhere near
enough to stop the network interpolating the training set (Zhang et al.), so its benefit may be
about training dynamics rather than capacity control.
-->

<span class="refs">Read: [d2l.ai 5.5](https://d2l.ai/chapter_multilayer-perceptrons/generalization-deep.html) · [Zhang et al., Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530) · Watch: [A. Ng, Regularization](https://www.youtube.com/watch?v=6g0t3Phly2M)</span>
