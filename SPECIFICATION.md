# ML Explorer — Interactive Machine Learning Learning & Visualization Platform
## Software Specification Document

**Version:** 2.1 (Decisions locked — ready for final go-ahead)
**Date:** 2026-05-29
**Deployment target:** Vercel (static build)
**Status:** ✅ Decisions approved (§0) — awaiting green light to start development

---

## 0. Locked Decisions (approved 2026-05-29)

| # | Decision | Choice |
|---|----------|--------|
| 1 | **Tech stack** | Vite + React + TypeScript + Tailwind CSS + TensorFlow.js → static build on Vercel |
| 2 | **CNN classifier (Mode B)** | Bundle a small **pre-trained MNIST CNN** (weights in `/public`, run via TF.js); live digit classification + activation flow |
| 3 | **Scope** | **Focused & polished** — 6 core methods + NLP (see final list below) |
| 4 | **NLP depth** | Illustrative **single-head self-attention heatmap** + in-browser sentiment + word embeddings |

**Final method list (shipped):**
1. Linear & Polynomial Regression + regularization — §6.1
2. Logistic / Softmax Regression — §6.2
3. Support Vector Machine (linear/poly/RBF kernels) — §6.3
4. K-Means / K-Means++ — §6.7
5. Neural Network MLP Playground (advanced) — §6.12
6. ★ CNN Visualizer (Mode A mechanics + Mode B pre-trained classifier) — §6.13
7. NLP — Embeddings & Self-Attention — §6.14

All other methods in §6 (SVM beyond core, KNN, trees/ensembles, Naive Bayes, GMM, DBSCAN, PCA, t-SNE) are **deferred** unless time permits; they remain documented for future expansion. Build order follows §9, with the 7 methods above as the committed deliverable.

---

## 1. Purpose & Vision

**ML Explorer** is a web application that teaches machine learning by letting users *watch algorithms learn and operate in real time*. For every method the user gets a conceptual overview, the governing mathematics, synthetic dataset generation with tunable parameters, and a **live, animated training/inference visualization** running in the browser.

The deep-learning centerpiece is a **CNN Visualizer** that animates a convolution kernel sweeping across an input image, builds feature maps layer-by-layer (conv → activation → pooling), and runs a real classifier on a user-drawn digit while visualizing how activations propagate through every layer.

Core differentiator: **the algorithms are real, inspectable, and animated** — not pre-recorded. Classic ML methods are implemented from scratch; the CNN/heavier deep-learning pieces use a vetted in-browser tensor library (no backend, no API keys).

---

## 2. Design Principles

| Principle | Decision |
|-----------|----------|
| **Deploys statically to Vercel** | Vite build → static `dist/`. Zero server runtime. |
| **No backend / no API keys** | 100% client-side compute (CPU + optional WebGL via TF.js). |
| **Real, animated algorithms** | Classic ML from scratch; CNN/t-SNE/embeddings via TensorFlow.js. |
| **Deep, not shallow** | Every method exposes advanced hyperparameters, multiple variants/optimizers, and quantitative readouts — not toy single-knob demos. |
| **Consistent structure** | Identical 6-section layout per method (see §4). |
| **Graceful degradation** | Heavy models (CNN classifier, t-SNE) load lazily; mechanics demos work even if a model fails to load. |
| **Polished & responsive** | Modern dark theme, smooth 60fps canvas animation, demo-ready on a laptop/projector. |

---

## 3. Technology Stack

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Build / dev | **Vite** | Fast, first-class Vercel support, static output |
| UI framework | **React + TypeScript** | Component/tab structure, typed hyperparameter state |
| Styling | **Tailwind CSS** + CSS variables | Rapid, consistent, polished dark theme |
| Visualization | **HTML5 Canvas 2D** (custom `Plot`/`Heatmap` helpers) | Real-time 60fps scatter, decision regions, feature maps |
| Math rendering | **KaTeX** (`react-katex`) | Beautiful LaTeX formulas |
| Heavy ML (CNN, embeddings, t-SNE) | **TensorFlow.js** (WebGL backend) + a small **pre-trained CNN** (static weights bundled) | Real digit classification & feature maps, fully client-side |
| Classic ML | **From scratch (TypeScript)** | Genuine, inspectable, animatable training |
| Routing | Hash or React Router (static-safe) | Deep-linkable methods on Vercel |

**External assets:** KaTeX (bundled), TensorFlow.js (bundled), one small pre-trained CNN weights file (bundled in `/public`). No runtime external calls required.

---

## 4. Standard Method Layout (6 sections)

Each method is presented through an identical tabbed interface:

1. **Overview** — Problem solved, when to use, intuition, pros/cons, complexity.
2. **Description** — Deeper mechanics, key terms, assumptions, failure modes.
3. **Formulas** — KaTeX-rendered: model equation, objective/loss, training/update rule, with annotations.
4. **Dataset Generation** — Tunable synthetic data generators + description of the generating process.
5. **Visualization & Learning** — The interactive canvas: Train/Step/Pause/Reset, hyperparameter controls, live metrics, animated rendering.
6. **Experiments / Insights** — Guided "try this" prompts (e.g., "raise the RBF γ and watch overfitting"), plus live quantitative readouts (accuracy, loss, inertia, explained variance, etc.).

---

## 5. Global UI / Layout

- **Left sidebar (grouped nav):**
  - *Regression* → Linear/Polynomial Regression (+ Ridge/Lasso)
  - *Classification* → Logistic/Softmax Regression, Support Vector Machine, K-Nearest Neighbors, Decision Tree & Ensembles, Naive Bayes
  - *Unsupervised* → K-Means (++), Gaussian Mixture Models (EM), DBSCAN, PCA, t-SNE
  - *Neural Networks* → MLP Playground (advanced)
  - *Deep Learning* → **CNN Visualizer**, NLP: Embeddings & Self-Attention
- **Top header:** title, tagline, global theme, link to spec/about.
- **Main area:** 6-tab method view (§4).
- **Status bar:** training state, epoch/iteration, FPS.
- **Theme:** Dark, indigo→cyan accent, monospace metrics, smooth transitions.

---

## 6. Method Specification (advanced)

### 6.1 Linear & Polynomial Regression + Regularization *(Regression)*
- **Data:** `y = f(x) + noise`, where f is linear or polynomial (degree 1–9). Params: samples, noise σ, true degree, heteroscedasticity toggle.
- **Models:** OLS via **gradient descent** *and* **closed-form normal equation**; polynomial basis expansion; **Ridge (L2)** and **Lasso (L1)** regularization.
- **Learning:** Animated fit curve; optimizer choice (SGD / Momentum / Adam); live MSE + R²; **bias–variance** demonstration (train vs. validation curve); **regularization path** (coefficients vs. λ).
- **Controls:** polynomial degree, λ (log slider), learning rate, optimizer, samples, noise, train/val split.
- **Visuals:** scatter + fit curve + confidence band, loss curve, coefficient bar chart, regularization path plot.

### 6.2 Logistic & Softmax Regression *(Classification)*
- **Data:** 2-class Gaussian blobs (binary) or 3-class (multinomial). Params: separation, samples, class count (2–4), noise.
- **Model:** binary sigmoid **and** multinomial **softmax**; L2 regularization.
- **Learning:** GD/Adam on (binary cross-entropy | categorical cross-entropy); animated decision boundary + probability surface shading.
- **Controls:** classes, learning rate, λ, optimizer, epochs.
- **Visuals:** probability heatmap, boundaries, per-class confidence, loss + accuracy curves, confusion matrix.

### 6.3 Support Vector Machine *(Classification)*
- **Data:** linearly separable, overlapping, and non-linear (circles/moons) patterns.
- **Model:** soft-margin SVM; kernels: **linear, polynomial, RBF (Gaussian)**; the **kernel trick** explained/visualized.
- **Learning:** margin maximization (SMO-style or subgradient); highlight **support vectors**; show margin width.
- **Controls:** kernel, **C** (soft-margin), **γ** (RBF), polynomial degree.
- **Visuals:** decision boundary + margins, support vectors emphasized, RBF influence heatmap.

### 6.4 K-Nearest Neighbors *(Classification)*
- **Data:** 2–4 labeled clusters; adjustable overlap.
- **Model:** k-NN with uniform / **distance-weighted** voting; Euclidean / Manhattan / Chebyshev metrics.
- **Live:** decision-region map recomputed on parameter change; click a test point → draw lines to its k neighbors + show vote tally.
- **Controls:** k (1–35), weighting, metric, samples, classes.
- **Visuals:** decision regions, Voronoi (k=1), neighbor links, vote breakdown.

### 6.5 Decision Tree & Ensembles *(Classification)*
- **Data:** non-linear 2-class/3-class structure.
- **Models:** CART (Gini/entropy) **single tree**, **Random Forest** (bagging), **Gradient Boosting** (stumps).
- **Learning:** build/visualize axis-aligned regions; show tree structure; **feature importance**; watch regions sharpen as #estimators grows.
- **Controls:** max depth, min samples split, criterion, n_estimators, learning rate (boosting), bootstrap.
- **Visuals:** decision-region map, node tree diagram, importance bars, OOB/val accuracy.

### 6.6 Naive Bayes *(Classification)* — *P1*
- Gaussian NB: visualize per-class conditional densities + resulting boundary; controls for shared/independent covariance.

### 6.7 K-Means / K-Means++ *(Unsupervised)*
- **Data:** k Gaussian blobs (+ anisotropic/unequal-variance toggle).
- **Model:** Lloyd's algorithm; **random vs. k-means++** init.
- **Learning:** animated assign→update with centroid trails; **inertia (WCSS)** per iteration; **elbow method** plot across k.
- **Controls:** chosen k, init, max iters, step/run, regenerate.
- **Visuals:** colored assignments, centroids + trails, Voronoi regions, elbow chart.

### 6.8 Gaussian Mixture Models (EM) *(Unsupervised)*
- **Data:** overlapping anisotropic Gaussians.
- **Model:** GMM via **Expectation-Maximization**; full covariance.
- **Learning:** animated **covariance ellipses** growing/rotating each EM step; **soft** (probabilistic) assignments via color blending; log-likelihood curve.
- **Controls:** components, covariance type (full/diag/spherical), init, step/run.
- **Visuals:** ellipses, soft-assignment coloring, log-likelihood curve.

### 6.9 DBSCAN *(Unsupervised)* — *P1*
- Density clustering; classify **core / border / noise** points; eps + minPts controls; non-globular cluster discovery (moons/rings).

### 6.10 PCA *(Unsupervised — Dimensionality Reduction)*
- **Data:** correlated/high-D synthetic data (project from 3D→2D, or 2D demo).
- **Model:** eigen-decomposition of covariance; principal components.
- **Visuals:** principal axes overlaid, projection animation, **explained-variance** scree plot, reconstruction error vs. #components.

### 6.11 t-SNE *(Unsupervised — Manifold)* — *ambitious*
- **Data:** clustered high-D blobs / digit features.
- **Model:** t-SNE embedding (TF.js or from-scratch) animating from random init → separated clusters.
- **Controls:** perplexity, learning rate, iterations.
- **Visuals:** live 2D embedding animation, KL-divergence readout.

### 6.12 Neural Network Playground — Advanced MLP *(Neural Networks)*
- **Data:** circles, spirals, XOR, two-moons, multi-class (selectable), noise + sample controls.
- **Model:** configurable MLP (1–4 hidden layers, 1–16 neurons each), activations (ReLU/LeakyReLU/Tanh/Sigmoid), **softmax** output for multi-class; from scratch (forward + backprop).
- **Optimizers:** SGD, Momentum, RMSProp, **Adam**. **Regularization:** L2, **dropout**. Mini-batch size.
- **Learning:** live decision-boundary heatmap; **per-neuron activation visualization** (each hidden unit's learned pattern); loss + accuracy curves; epoch counter.
- **Controls:** full architecture editor, optimizer, lr, λ, dropout, batch size, dataset, noise; Train/Pause/Step/Reset.
- **Visuals:** evolving decision heatmap, per-neuron mini-heatmaps, weight magnitudes, loss/acc charts.

### 6.13 ★ CNN Visualizer *(Deep Learning — centerpiece)*
The flagship feature. Two integrated modes:

**Mode A — Convolution Mechanics (animated sweep):**
- **Input:** a built-in image *or* the user's hand-drawn digit (28×28 draw pad) *or* uploaded image.
- **Animated kernel sweep:** the convolution window **slides across the input cell-by-cell**, highlighting the current **receptive field**, showing the element-wise multiply-accumulate, and **writing the resulting pixel into the output feature map** in sync. Adjustable **stride, padding, kernel size, dilation**.
- **Selectable/learned kernels:** edge (Sobel X/Y), blur, sharpen, emboss, or learned filters from the trained net; live output feature map.
- **Layer stack:** chain **Conv → ReLU → MaxPool** and show the feature map shrinking; visualize **multiple filters** producing a stack of feature maps; toggle layers.
- **Controls:** kernel size, stride, padding, dilation, pooling size, filter selector, animation speed, play/pause/step.

**Mode B — Live Classification + Activation Flow:**
- A small **pre-trained CNN** (digit classifier, MNIST-style; weights bundled in `/public`, run via TF.js) classifies the user-drawn digit.
- **Visualizes activations through every layer:** input → conv1 feature maps → pool → conv2 feature maps → dense → **softmax probability bars** for the predicted class.
- Shows the **learned first-layer filters** and lets Mode A use them.
- **Controls:** clear/redraw pad, predict, step-through-layers, brush size.

**Formulas:** 2D discrete convolution, output-size formula `⌊(W−K+2P)/S⌋+1`, ReLU, max-pool, softmax; note that CNNs *learn* the kernels.

### 6.14 NLP — Embeddings & Self-Attention *(Deep Learning)*
- **Tokenization view:** text → tokens → ids, with subword note.
- **Word embeddings:** a small bundled embedding set projected to 2D (PCA/t-SNE); show nearest-neighbor words; vector arithmetic ("king − man + woman").
- **Self-attention visualizer:** for a short input sentence, compute/display a **single-head attention heatmap** (which tokens attend to which); explain Q/K/V intuition. (Weights either bundled-tiny or illustrative-computed.)
- **Sentiment classifier:** bag-of-words/embeddings + logistic regression trained in-browser; **live prediction** on typed text + top influential tokens.
- **Controls:** input text box, model selector (sentiment / attention / embeddings), retrain.
- **Visuals:** attention heatmap, embedding scatter, sentiment gauge, token-weight bars.

---

## 7. Shared Architecture

- **`Plot`** — data↔pixel transforms, axes/grid, scatter, lines, decision-region & probability heatmaps, animation loop.
- **`Heatmap`/`FeatureMap`** — render matrices/feature maps as color grids (for CNN, NN neurons, attention).
- **`DataGen`** — linear/poly, gaussian blobs, circles, spiral, moons, XOR, anisotropic, high-D.
- **`MathUtils` / `linalg`** — vector/matrix ops, eigendecomposition, distances, sigmoid/softmax, kernels (RBF/poly), seeded RNG.
- **`Optimizers`** — SGD, Momentum, RMSProp, Adam (shared by regression/NN).
- **`LossChart`** — live multi-series canvas chart.
- **Method registry** — each method exports `{ id, name, group, Overview, Description, Formulas, DatasetControls, Visualization }` consumed by the tabbed view.
- **Proposed layout:**
  ```
  index.html
  vite.config.ts, tsconfig.json, tailwind.config.js, vercel.json
  public/  (pretrained CNN weights, embeddings, sample images)
  src/
    main.tsx, App.tsx, router.ts
    components/  (Sidebar, MethodView, Tabs, Controls, Charts)
    lib/         (plot, heatmap, datagen, mathutils, optimizers, losschart)
    methods/     (one folder/file per §6 method)
  ```

---

## 8. Vercel Deployment

- **Build:** `vite build` → static `dist/`.
- **Config:** `vercel.json` (SPA rewrite to `/index.html` for client routing), framework preset **Vite**, output `dist`, install `npm install`, build `npm run build`.
- **Assets:** pretrained weights/embeddings served from `/public` (cached, no runtime API).
- **Acceptance:** production URL loads with no console errors; all bundled models load lazily without external calls.

---

## 9. Scope & Prioritization (build order)

**P0 (must):** App shell + nav + 6-tab layout; Linear/Poly Regression; Logistic/Softmax; K-Means(++); **MLP Playground (advanced)**; **CNN Visualizer Mode A (animated convolution sweep + feature maps + pooling)**.
**P1 (should):** SVM (kernels), KNN, Decision Tree/Ensembles, GMM (EM), PCA, **CNN Mode B (live digit classification + activation flow)**, NLP sentiment + attention heatmap.
**P2 (nice):** Naive Bayes, DBSCAN, t-SNE, embedding arithmetic, image upload, editable kernel matrix, regularization-path/elbow extras.

If time is short, drop from P2 upward; every shipped method keeps all 6 sections.

---

## 10. Non-Goals

- No training of large/deep models in-browser (CNN classifier is **pre-trained**; weights bundled).
- No real large datasets (synthetic + small bundled assets only).
- No backend, accounts, persistence (beyond optional last-viewed method in `localStorage`).
- Deep-learning section is **mechanistic visualization + small real models**, not full-scale CNN/transformer training.

---

## 11. Acceptance Criteria

- [ ] Builds with `vite build` and deploys to Vercel; production URL loads, no console errors, no runtime external calls.
- [ ] Sidebar groups all shipped methods; each renders the 6-tab layout (§4).
- [ ] Regression/Logistic/MLP visibly animate during training; loss decreases on screen; optimizers selectable.
- [ ] SVM shows support vectors & kernel boundary; K-Means centroids converge with inertia readout; GMM ellipses animate; PCA shows components + scree.
- [ ] **CNN Mode A:** kernel visibly sweeps the image with receptive-field highlight, writing the output feature map in sync; stride/padding/kernel-size/pooling change behavior; multi-filter feature-map stack renders.
- [ ] **CNN Mode B:** user draws a digit → pre-trained CNN predicts → activations of each layer are visualized → softmax bars shown.
- [ ] NLP: live sentiment prediction on typed text + attention heatmap renders.
- [ ] 60fps-ish animations on a typical laptop; responsive, polished dark UI.

---

## 12. Open Questions for Approver

1. **Stack confirmation:** OK with **Vite + React + TypeScript + Tailwind + TF.js** (build step, deploys cleanly to Vercel)? Or prefer leaner vanilla TS + Vite (lower complexity, but heavier custom code for the CNN classifier)?
2. **CNN classifier source:** Acceptable to **bundle a small pre-trained MNIST-style CNN** (weights in `/public`, run via TF.js)? Alternative: ship Mode A (mechanics) only and make Mode B a stretch goal.
3. **Method set size:** The P0+P1 set is ~11 methods. Confirm that breadth, or should I concentrate effort on fewer methods done to a higher polish (e.g., guarantee CNN + MLP + 4 classics)?
4. **NLP depth:** Is a single-head **self-attention heatmap** (illustrative) enough, or do you want a genuinely pre-trained tiny transformer (heavier)?
5. **Visual style:** Dark + indigo→cyan accent acceptable, or a specific palette/branding?

---

*End of specification. Please review and approve, request changes, or answer §12 before development begins.*
