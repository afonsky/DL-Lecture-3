---
zoom: 0.95
---

# Conclusions: The Model and Backpropagation

<div class="grid grid-cols-2 gap-10">
<div>

### The model
* An affine function of an affine function is affine — **the activation is what makes depth worth having**
* A hidden layer does not classify, it **re-describes** the data. XOR takes two ReLU units
* **ReLU** in hidden layers, **GELU** in transformers, softmax on the output

</div>
<div>

### Backpropagation
* Forward: **keep** every value. Backward: **incoming gradient × local derivative**
* One backward pass gives **every** gradient at ~2× a forward pass
* Weight gradient = (gradient arriving) × (input that arrived)$^\top$
* Activations are stored → **training memory ∝ depth × batch size**

</div>
</div>

---
zoom: 0.95
---

# Conclusions: Making It Train, and Not Overfit

<div class="grid grid-cols-2 gap-10">
<div>

### Making it train
* Gradients through $L$ layers are a product of $L$ terms → they **vanish or explode**
* Identical initial weights never separate: **randomize**, **He** for ReLU, **Xavier** for tanh
* Flat loss → vanishing gradients. `NaN` → exploding gradients

</div>
<div>

### Not overfitting
* Plot both curves on **every** run. **Early stopping** is the cheapest regularizer
* **Dropout**: zero each unit with probability $p$, rescale by $1/(1-p)$, off with `eval()`
* Order: more data → early stopping → weight decay → dropout → smaller network

### The one habit to take away
**Overfit a single batch to zero loss first.** If that fails, no hyperparameter will save you.

</div>
</div>

---
zoom: 0.68
---

# Learn More from the Experts

| Expert | Watch / read | Today's topics |
|---|---|---|
| Andrew Ng | [Deep Learning Specialization, Course 1, Weeks 3–4](https://www.coursera.org/learn/neural-networks-deep-learning) · [Course 2](https://www.coursera.org/learn/deep-neural-network) | hidden layers, activations, init, dropout |
| Yaser Abu-Mostafa | [Learning From Data](https://work.caltech.edu/lectures.html), Lectures 10 and 11 | backpropagation from first principles |
| Yann LeCun | [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) · [NYU Deep Learning](https://atcold.github.io/NYU-DLSP21/) | backprop in practice, initialization |
| Yoshua Bengio | [Deep Learning book, Ch. 6 and 8](https://www.deeplearningbook.org/) · [Practical Recommendations](https://arxiv.org/abs/1206.5533) | MLPs, vanishing gradients, hyperparameters |
| Andrej Karpathy | [The spelled-out intro to backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) · [micrograd](https://github.com/karpathy/micrograd) · [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) | backprop by hand, debugging |
| Sebastian Raschka | [STAT 453](https://sebastianraschka.com/blog/2021/dl-course.html) · [PyTorch in One Hour](https://sebastianraschka.com/teaching/pytorch-1h/) | MLPs and autograd in PyTorch |
| Josh Starmer | [StatQuest video index](https://www.statquest.org/video-index/) — Neural Networks Pts. 1–4, Backpropagation | slow, visual intuition for every step |

<br>

#### Main text: [d2l.ai, Chapter 5 — Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/), sections 5.1–5.6

### Next lecture: convolutional neural networks — what to do when the input has spatial structure
