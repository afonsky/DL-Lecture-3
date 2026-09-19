---
layout: center
---

<center>

# Foundations

# (1940s–1960s)
</center>

---

# The Birth of Artificial Neurons (1943–1958)

<div class="grid grid-cols-[4fr_2fr] gap-10">
<div>
<figure>
  <img src="/Neuron3.svg" style="width: 300px !important;">
</figure>

* Advances in computability theory inspired computational models of cognition
* Warren McCulloch & Walter Pitts [proposed](https://link.springer.com/article/10.1007/BF02478259) that biological neurons can be described as computational devices (1943)
* In a way, this is another iteration of the problem of describing a general-purpose computation device
</div>
<div>
<br>
<br>
  <figure>
    <img src="/mp-neuron.png" style="width: 200px; position: relative">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>McCulloch-Pitts artificial neuron<br><br><br>Images sources:<br>
      Left:<br> <a href="https://commons.wikimedia.org/wiki/File:Neuron3.svg">https://commons.wikimedia.org/wiki/File:Neuron3.svg</a><br><br>
      Right:<br> <a href="https://pabloinsente.github.io/the-mcculloch-pitts-artificial-neuron-model">https://pabloinsente.github.io/the-mcculloch-pitts-artificial-neuron-model</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Frank Rosenblatt’s Perceptron (1958)

<div class="grid grid-cols-[3fr_2fr]">
<div>

* Developed by [Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) 2 years after his PhD’56<br> at Cornell Aeronautical Laboratory
* Perceptron is a binary classifier<br> with a **hyperplane decision boundary**
* The algorithm minimizes the distance of misclassifications to the hyperplane
* **Goal**: minimize<br> $D(\beta, \beta_0) := -\sum\limits_{i \in \cal{M}} y_i (\beta_0 + \beta^{\prime} x_i)$
	* where $\cal{M}$ is the set of misclassified points
</div>
<div>
  <figure>
    <img src="/Rosenblatt.png" style="width: 380px; position: relative">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: -30px; left: -380px">Image source:
      <a href="https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon">https://news.cornell.edu/stories/2019/09/<br>professors-perceptron-paved-way-ai-60-years-too-soon</a>
    </figcaption>
  </figure>
</div>
</div>
