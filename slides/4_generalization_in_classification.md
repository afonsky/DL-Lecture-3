---
layout: chapter-title

# the image source
image: /img_MPQJaay1NIgnuJEPnDqb.jpg

# a custom class name to the content
class: my-cool-content-on-the-right
---

## Generalization in Classification

<span style="color:DimGray; font-size: 11px; position:absolute; right:20px; bottom:20px;">Image credit: Midjourney 6.0<br> prompt: ‘Generalization in Classification'
</span>

---

# Generalization in Classification
<!-- <style>
.slidev-layout {
  font-size: 1.2em
}
</style> -->

* Our goal is to learn *general patterns*, as assessed empirically on previously unseen data
* A generalization is guaranteed if we have some required number of training samples **n** if our empirical error will lie within $\epsilon$ of the true error, *for any data generating distribution*

* Suppose that $\mathcal{D} = {(\mathbf{x}^{(i)},y^{(i)})}_{i=1}^n$ is a dataset of examples
* Then $\epsilon_\mathcal{D}(f) = \frac{1}{n}\sum\limits_{i=1}^n \mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)})$ is the *empirical error* of our classifier $f$ on $\mathcal{D}$
* Then the *population error* is:
$$\epsilon(f) =  E_{(\mathbf{x}, y) \sim P} \mathbf{1}(f(\mathbf{x}) \neq y) =
\int\int \mathbf{1}(f(\mathbf{x}) \neq y) p(\mathbf{x}, y) \;d\mathbf{x} dy$$
* We can view $\epsilon_\mathcal{D}(f)$ as a statistical estimator of the population error $\epsilon(f)$

---

# Central Limit Theorem

* Let $X_1, ..., X_n$ be i.i.d. with mean $\mu$ and variance $\sigma^2$

* Let $\bar{X}_n = \frac{1}{n}\sum\limits_{i=1}^n X_i$

* Then $Z_n = \frac{\bar{X}_n - \mu}{\sqrt{\mathbb{V}(\bar{X}_n)}} = \frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \rightarrow Z$
  * Where $Z \sim N(0,1)$

* So, as $\uparrow n$: $~~\epsilon_\mathcal{D}(f) \rightarrow \epsilon(f)$
at a rate of $\mathcal{O}(1/\sqrt{n})$

---

# Central Limit Theorem

* As the RV of interest $\mathbf{1}(f(X) \neq Y)$ can only take values $0$ and $1$ and thus is a Bernoulli RV
  * Here $\mathbf{1}$ means that our classifier made an error, so the parameter of our RV is actually the true error rate $\epsilon(f)$
  * $\mathbb{V}(\mathrm{Bern.}) = p(1-p) = \epsilon(f)(1-\epsilon(f))$
    * As $\epsilon(f) < 1$ then $\mathbb{V}(\mathrm{Bern.})$ is highest when $\epsilon(f) \sim 0.5$
      * So, $~~\epsilon_\mathcal{D}(f) < \sqrt{0.25 / n}$

* In practice, if we want that $1$ SD of our estimate $\epsilon_\mathcal{D}(f)$ of the true error $\epsilon(f)$ be within $\pm 0.01$, then we need at least $\frac{0.25}{0.01^2} = 2500$ samples