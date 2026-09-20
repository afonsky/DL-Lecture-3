---
zoom: 0.93
---

# From Lecture 2 to Today

<div class="grid grid-cols-2 gap-20">
<div>

### We already have *(Lectures 1–2)*
* **forward → loss → backward → step**,<br> on minibatches
* Linear and softmax regression = **one-layer** networks
* Cross-entropy, learning rate, weight decay, validation

### We are still missing
* Every model so far draws a **hyperplane**. XOR does not care
* Lecture 1 *named* backpropagation — today we open it up

</div>
<div>

### Plan for today
1. **From linear models to MLPs**<br><small>hidden layers, why a nonlinearity is compulsory</small>
2. **Activation functions**<br><small>ReLU, sigmoid, tanh, GELU</small>
3. **Forward and backward propagation**<br><small>graphs, the chain rule, autograd</small>
4. **Making deep networks trainable**<br><small>vanishing gradients, initialization</small>
5. **Regularization and practice**<br><small>dropout, early stopping, PyTorch</small>

</div>
</div>

<span class="refs">Main text: [d2l.ai, Ch. 5 Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/)</span>

<!--
Say, don't show: Lectures 1-2 gave the whole training machinery; today only the model changes.

Timing plan (80 min, of which 10-15 min is the quiz):
- Logistics + timeline:                     3 min
- History (foundations, pattern recognition): 8 min
- This slide:                               2 min
- 1. From linear models to MLPs:           11 min
- 2. Activation functions:                  6 min
- 3. Forward and backward propagation:     12 min
- 4. Numerical stability and init:          8 min
- 5. Regularization:                        7 min
- 6. Practice (PyTorch, debugging):         6 min
- Conclusions:                              3 min
                                    total  66 min  + quiz

If running late, in this order:
  (a) skip "Where MLPs Live Today",
  (b) skip "The Same Thing for a Whole Layer" (the worked chain already carries it),
  (c) compress the history section to the AI-winter slide alone.
-->
