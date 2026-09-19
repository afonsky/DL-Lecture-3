---
layout: center
---

<center>

# Pattern Recognition Era

# (1960s–1980s)
</center>

---

# Early Pattern Recognition

<br>
<br>

<div class="grid grid-cols-[4fr_1fr] gap-20">
<div>

* Paul V. C. Hough submitted a patent for his<br> ["Method and Means for Recognizing Complex Patterns"](https://patents.google.com/patent/US3069654A/en)<br><br>
* He delivered a general-­audience talk as part of “A Computer Learns to See” a Brookhaven lecture series in February 1962

<br>

#### See also [James E. Dobson, "The Birth of Computer Vision" (2023)](https://www.google.com/books/edition/The_Birth_of_Computer_Vision/LKyTEAAAQBAJ)
</div>
<div>
  <figure>
    <img src="/Paul_VC_Hough.png" style="width: 150px; position: relative">
    <figcaption style="color:#b3b3b3ff; font-size: 11px">Paul V. C. Hough<br>Image source:<br>
      <a href="https://www.gf.org/fellows/paul-v-c-hough/">https://gf.org/fellows/paul-v-c-hough</a>
    </figcaption>
  </figure>
</div>
</div>

<br>

<v-click at="1">

#### Have you ever heard of the [Hough transform](https://en.wikipedia.org/wiki/Hough_transform)?
</v-click>


---
zoom: 0.9
---

# The First AI Winter ❄️


#### Minsky & Papert's book ["Perceptrons: An Introduction to Computational Geometry"](https://en.wikipedia.org/wiki/Perceptrons_(book)) (1969) made pessimistic predictions in the study of perceptrons:
<br>
<div class="grid grid-cols-[4fr_2fr] gap-10">
<div>

1. **Invariance**: perceptrons cannot solve tasks needing invariance to translation, rotation or scaling<br><br>
2. **No advantage**: for forecasting, no functional gain over statistical methods<br><br>
3. **Intractability**: solvable in principle, but needing unrealistic time or memory
</div>
<div>
<br>
<br>
<br>
<v-click>
  <figure>
    <img src="/XOR_problem.png" style="width: 380px; position: relative">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>XOR problem<br>Image source:<br>
      <a href="https://dev.to/jbahire/demystifying-the-xor-problem-1blk">https://dev.to/jbahire/demystifying-the-xor-problem-1blk</a>
    </figcaption>
  </figure>
</v-click>
</div>
</div>

---

# Expert Systems & Rule-Based Approaches

<br>

### Symbolic AI (Expert Systems) filled the vacuum
<br>

* **Core idea**: intelligence = manipulating symbols and logical rules
* **Knowledge**: hand-crafted by experts as explicit "if-then" rules
* **Representation**: interpretable — every rule can be read and audited
* **Strengths**: narrow, well-defined domains (e.g. medical diagnostics)
* **Weaknesses**: brittle outside the rules; doesn't scale (knowledge bottleneck)

<br>

#### See backup slides for a comparison table of expert systems and deep learning.

---
zoom: 0.88
---

# Backpropagation Resurfaces

### Rumelhart, Hinton & Williams publish<br> ["Learning representations by back-propagating errors"](https://www.nature.com/articles/323533a0) in *Nature* (1986)

<br>

<div class="grid grid-cols-[5fr_2fr] gap-8">
<div>

* **Key idea**: the gradient w.r.t. every weight, by the **chain rule**, from output back to input
* This is what lets **hidden layers** actually learn — Minsky's critique, answered
* [Discovered several times before](https://people.idsia.ch/~juergen/who-invented-backpropagation.html) (Linnainmaa 1970, Werbos 1982); the 1986 paper made it practical

<br>

> *"We describe a new learning procedure, back-propagation, for networks of neurone-like units."*

</div>
<div>
<br>
  <figure>
    <img src="/backpropagation_1986.png" style="width: 350px; position: relative">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>
      Multi-layer network with gradient flow.<br>
      Image source:
      <a href="https://www.nature.com/articles/323533a0">Rumelhart, Hinton & Williams,<br>Nature 323, 533–536 (1986)</a>
    </figcaption>
  </figure>
</div>
</div>