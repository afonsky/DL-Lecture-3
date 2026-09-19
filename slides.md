---
theme: seriph
addons:
  - "@twitwi/slidev-addon-ultracharger"
addonsConfig:
  ultracharger:
    inlineSvg:
      markersWorkaround: false
    disable:
      - metaFooter
      - tocFooter
NObackground: >-
  https://images.unsplash.com/photo-1511149755252-35875b273fd6?ixlib=rb-4.0.3&dl=leon-contreras-qpdfU6vehgs-unsplash.jpg&w=1920&q=80&fm=jpg&crop=entropy&cs=tinysrgb
background: /logo/mountain.jpg
highlighter: shiki
routerMode: hash
lineNumbers: false
duration: 80min

css: unocss
title: Deep Learning
subtitle: Multilayer Perceptrons. Backpropagation
date: 21/09/2026
venue: HSE
author: Alexey Boldyrev, Maksim Karpov
ghPrefix: https://github.com/twitwi/slidev-addon-ultracharger/blob/main/
ghSelf: https://github.com/twitwi/slidev-addon-ultracharger-demo/blob/main/
---

# <span style="font-size:28.0pt" v-html="$slidev.configs.title?.replaceAll(' ', '<br/>')"></span>
# <span style="font-size:32.0pt" v-html="$slidev.configs.subtitle?.replaceAll(' ', '<br/>')"></span>
# <span style="font-size:18.0pt" v-html="$slidev.configs.author?.replaceAll(' ', '<br/>')"></span>

<span style="font-size:18.0pt" v-html="$slidev.configs.date?.replaceAll(' ', '<br/>')"></span>

<div class="abs-tl mx-5 my-10">
  <img src="/logo/FCS_logo_full_L.svg" class="h-18">
</div>

<div class="abs-tr mx-5 my-5">
  <img src="/logo/DSBA_logo.png" class="h-28">
</div>

<style>
  :deep(footer) { padding-bottom: 3em !important; }
</style>


---
src: ./slides/0_outline.md
---

---
src: ./slides/1_era_foundations.md
---

---
src: ./slides/2_era_pattern_recognition.md
---

---
src: ./slides/0_roadmap.md
---

---
src: ./slides/3_mlp.md
---

---
src: ./slides/4_activations.md
---

---
src: ./slides/5_backpropagation.md
---

---
src: ./slides/6_stability_init.md
---

---
src: ./slides/7_regularization.md
---

---
src: ./slides/8_practice.md
---

---
src: ./slides/9_conclusions.md
---

---
src: ./slides/0_backup.md
---

---
src: ./slides/0_end.md
---