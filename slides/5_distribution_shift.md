---
layout: chapter-title

# the image source
image: /img_4OsMU3o6S9YdZP2FSXZk.jpg

# a custom class name to the content
class: my-cool-content-on-the-right
---

## Distribution Shift

<span style="color:DimGray; font-size: 11px; position:absolute; right:20px; bottom:20px;">Image credit: Midjourney 6.0<br> prompt: ‘Distribution Shift'
</span>

---

# Training ≠ Testing

* Training minimizes $R_{\mathrm{emp}}(f, X, Y) := \frac{1}{m} \sum\limits_{i=1}^m l(y_i, f(x_i))$

* At test time we want to minimize
  * Expected risk (data drawn from some distribution)
  * Test error, if we have a specific set $\{x_1^\prime, ..., x_m^\prime\}$

* Good performance on training set doesn’t guarantee good test performance, unless we regularize capacity or have independent validation set for calibration

---

# Distribution Shift

#### One can distinguish 3 kinds of distribution shift:

* Covariate shift
  * While the distribution of inputs may change over time, the labeling function, i.e., the conditional distribution does not change

* Label shift
  * Even though the feature distribution remains the same, the label distribution might changes

* Concept shift
  * Definitions of labels can change

---

# Distribution Shift: Covariate Shift

<div>
  <figure>
    <img src="/cat-dog-train.png" style="width: 600px !important;">
  </figure>
</div>
<br>

<div>
  <figure>
    <img src="/cat-dog-test.png" style="width: 600px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Images source:
      <a href="https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html">d2l.ai Top: Fig. 4.7.1, Bottom: Fig. 4.7.1. Train and Test data for distinguishing cats and dogs</a>
    </figcaption>
  </figure>
</div>

---

# Distribution Shift: Label Shift

#### $q(x, y) = q(y) p(x|y)$
<br>

#### Medical diagnosis
* Train on data with few sick patients (in CA)
* Test in South Dakota after Sturgis Biker Rally when $q(\mathrm{C19}) > p(\mathrm{C19})$ while COVID19 symptoms $p(\mathrm{symptoms | C19})$ are still the same

---

# Distribution Shift: Concept Shift

<div>
  <figure>
    <img src="/popvssoda.png" style="width: 600px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 10px">Image source:
      <a href="https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html">d2l.ai Fig. 4.7.3. Concept shift for soft drink names in the United States</a>
    </figcaption>
  </figure>
</div>

---

# Risk and Empirical Risk

* Without regularization our goal is to minimize the loss on the training<br>
$\underset{f}{\mathrm{minimize}} ~\frac{1}{n} \sum\limits_{i=1}^n l(f(\mathbf{x}_i), y_i)$
  * That is **empirical risk**

* The **risk**, instead is the expectation of the loss over the entire population of data drawn from their true distribution $p(\mathbf{x},y)$:
$$E_{p(\mathbf{x}, y)} [l(f(\mathbf{x}), y)] = \int\int l(f(\mathbf{x}), y) p(\mathbf{x}, y) \;d\mathbf{x}dy$$

* Because entire population of data is usually not achievable, minimizing the empirical risk is a practical strategy for ML

---

# Covariate Shift Correction

* Assume that we want to estimate some dependency $P(y \mid \mathbf{x})$ for labeled data $(\mathbf{x}_i, y_i)$
* Unfortunately, the observations $\mathbf{x}_i$ are drawn from:
  * *Source distribution* $q(\mathbf{x})$
  * *Target distribution* $p(\mathbf{x})$

* Fortunately, the conditional distribution does not change: $p(y \mid \mathbf{x}) = q(y \mid \mathbf{x})$


* If the source distribution $q(\mathbf{x})$ is "wrong",
we can correct for that by using the following simple identity in the risk:
$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(y \mid \mathbf{x})p(\mathbf{x}) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(y \mid \mathbf{x})q(\mathbf{x})\frac{p(\mathbf{x})}{q(\mathbf{x})} \;d\mathbf{x}dy
\end{aligned}
$$

---

# Covariate Shift Correction

* In practice, such a correction means reweighting each data example by the ratio of the probability that it would have been drawn from the correct distribution to that from the wrong one:
$$\beta_i := \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}$$

* Plugging in the weight $\beta_i$ for
each data example $(\mathbf{x}_i, y_i)$
we can train our model using
*weighted empirical risk minimization*:

$$\underset{f}{\mathrm{minimize}} ~\frac{1}{n} \sum_{i=1}^n \beta_i l(f(\mathbf{x}_i), y_i)$$

---

# Covariate Shift Correction Algorithm

* Suppose that we have:
  * a training set $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$
  * an unlabeled test set $\{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$

<br>

<div class="bg-orange-100">

### Algorithm
1. Create a binary-classification training set: $\{(\mathbf{x}_1, -1), \ldots, (\mathbf{x}_n, -1), (\mathbf{u}_1, 1), \ldots, (\mathbf{u}_m, 1)\}$.
1. Train a binary classifier using logistic regression to get the function $h$.
1. Weigh training data using $\beta_i = \exp(h(\mathbf{x}_i))$ or better $\beta_i = \min(\exp(h(\mathbf{x}_i)), c)$<br> for some constant $c$.
1. Use weights $\beta_i$ for training on $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$<br> in $\underset{f}{\mathrm{minimize}} ~\frac{1}{n} \sum_{i=1}^n \beta_i l(f(\mathbf{x}_i), y_i)$.
</div>

---

# Label Shift Correction

* Assume that we are dealing with a classification task with $k$ categories

* Assume that the distribution of labels shifts over time:
$q(y) \neq p(y)$,<br> but the class-conditional distribution
stays the same: $q(\mathbf{x} \mid y)=p(\mathbf{x} \mid y)$

* If the source distribution $q(y)$ is "wrong", we can correct for that according to the following identity in the risk as defined previously:
$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) \color{red}{p(\mathbf{x} \mid y)p(y)} \color{#006}{\;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y)} \color{red}{q(\mathbf{x} \mid y)q(y)\frac{p(y)}{q(y)}}\color{#006}{\;d\mathbf{x}dy}
\end{aligned}
$$

* Here, our importance weights will correspond to the
label likelihood ratios:
$$\beta_i := \frac{p(y_i)}{q(y_i)}$$

---

# Label Shift Correction

* Luckily, the labels are often much simpler then the inputs (categories vs. images)

* Therefore, we can get consistent estimates of these weights without ever having to deal with the ambient dimension

* We can estimate the test set label distribution by solving a simple linear system<br>
$\mathbf{C} p(\mathbf{y}) = \mu(\hat{\mathbf{y}})$
  * Where $\mathbf{C}$ is the *confusion matrix*

* If the confusion matrix is invertible, we get a solution $p(\mathbf{y}) = \mathbf{C}^{-1} \mu(\hat{\mathbf{y}})$

* Then, for any training example $i$ with label $y_i$,
we can take the ratio of our estimated $p(y_i)/q(y_i)$
to calculate the weight $\beta_i$,
and plug this into weighted empirical risk minimization

---

# Concept Shift Correction

* Much bigger problem if concept shifts between training and test set

* No real guarantees possible

* More often than not, this kind of problem is "outside the jurisdiction" of ML