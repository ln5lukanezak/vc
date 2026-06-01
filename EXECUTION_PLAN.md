# ML Explorer — Execution Plan

**Companion to:** `SPECIFICATION.md` v2.1 (approved 2026-05-29)
**Strategy:** Build a vertical slice first (shell + one full method) to prove the architecture, then ship the CNN headline, then fill in remaining methods. Always keep a deployable, working demo.

---

## Guiding rules

- **Vertical slice first.** Don't build all the shared libs up front — build exactly what method #1 needs, prove the 6-tab pattern works, then generalize.
- **Deploy early.** Get a "hello world" on Vercel in Phase 0 so deployment is never a last-minute surprise.
- **Each method is independently shippable.** If we run out of time, whatever is done is demo-ready.
- **Commit checkpoints** after each phase (git init in Phase 0).
- **Performance budget:** keep canvas redraws in `requestAnimationFrame`, cap training steps/frame, ~60fps target.

---

## Phase 0 — Project scaffold & deploy pipeline  *(foundation)*

**Goal:** A running Vite + React + TS + Tailwind app that deploys to Vercel.

1. `npm create vite@latest` (react-ts) in project root.
2. Install deps: `tailwindcss postcss autoprefixer`, `katex react-katex`, `@tensorflow/tfjs`.
3. Configure Tailwind (`tailwind.config.js`, `index.css` with dark theme tokens: indigo→cyan accent, fonts).
4. Base `App.tsx` shell: header + left sidebar + main content placeholder.
5. Add `vercel.json` (Vite preset, SPA rewrite to `/index.html`, output `dist`).
6. `git init`, first commit. Verify `npm run build` succeeds.
7. **Checkpoint:** deploy to Vercel, confirm the shell loads at the production URL with no console errors.

**Deliverable:** Live URL showing the app shell.

---

## Phase 1 — Core architecture + first full method (Linear Regression)  *(vertical slice)*

**Goal:** Prove the 6-tab method pattern and the shared viz primitives with one complete, animated method.

1. **Shared libs (only what's needed now):**
   - `lib/mathutils.ts` — vectors, RNG (seeded), sigmoid (stub for later).
   - `lib/datagen.ts` — linear/polynomial data generator.
   - `lib/plot.ts` — Canvas `Plot`: data↔pixel transforms, axes/grid, scatter, line.
   - `lib/losschart.ts` — live loss line chart.
   - `lib/optimizers.ts` — SGD, Momentum, Adam (shared later by NN).
2. **Method framework:**
   - `methods/registry.ts` — `MethodDef { id, name, group, Overview, Description, Formulas, Dataset, Visualization }`.
   - `components/MethodView.tsx` + `Tabs.tsx` — render the 6 sections per §4.
   - `components/Sidebar.tsx` — grouped nav from the registry.
   - `components/Controls.tsx` — reusable sliders/selects/buttons.
3. **Linear & Polynomial Regression (§6.1):** overview/description/KaTeX formulas; dataset controls (samples, noise, degree); animated GD fit + live MSE/R² + coefficient bars; optimizer selector; Train/Step/Reset.
4. **Checkpoint:** one method works end-to-end, animates, deploys.

**Deliverable:** Working Linear Regression method in the live app — the template every other method follows.

---

## Phase 2 — ★ CNN Visualizer (the headline)  *(highest value, build while fresh)*

**Goal:** The flagship deep-learning feature, both modes.

1. **Assets:** train a tiny MNIST CNN offline (or obtain compatible weights), export to TF.js format, place in `/public/models/`. Add a 28×28 draw pad component.
2. **`lib/heatmap.ts`** — render matrices/feature maps as color grids.
3. **Mode A — Convolution mechanics:** input image/draw pad; animated kernel **sweep** with receptive-field highlight; live multiply-accumulate → output feature map written in sync; controls for kernel size, stride, padding, dilation, pooling; selectable kernels (Sobel/blur/sharpen/learned); multi-filter feature-map stack; Conv→ReLU→MaxPool chain.
4. **Mode B — Live classification:** load pre-trained CNN via TF.js; classify drawn digit; visualize activations layer-by-layer (conv1 maps → pool → conv2 maps → dense → softmax bars); surface learned first-layer filters into Mode A.
5. **Formulas:** 2D convolution, output-size, ReLU, max-pool, softmax.
6. **Checkpoint:** draw a digit → animated conv sweep + correct prediction + activation flow; deploy.

**Deliverable:** CNN Visualizer live — the demo centerpiece.

---

## Phase 3 — Neural Network MLP Playground (advanced)  *(§6.12)*

**Goal:** TensorFlow-Playground-class interactive trainer (from scratch).

1. `methods/mlp` engine: configurable layers/neurons, activations (ReLU/LeakyReLU/Tanh/Sigmoid), softmax output, forward + backprop.
2. Optimizers (reuse `optimizers.ts`): SGD/Momentum/RMSProp/Adam; L2 + dropout; mini-batch.
3. Datasets: circles/spiral/XOR/moons (extend `datagen.ts`).
4. Viz: live decision-boundary heatmap, per-neuron activation mini-heatmaps, loss/accuracy curves, architecture editor.
5. **Checkpoint:** boundary morphs live on spiral; deploy.

**Deliverable:** Advanced MLP Playground live.

---

## Phase 4 — Classification classics  *(COMPLETE)*

**Goal:** Round out breadth with two more polished methods.

1. **Logistic / Softmax Regression (§6.2):** binary + multinomial; BCE/CE loss; probability heatmap; animated boundary; confusion matrix.
2. **Support Vector Machine (§6.3):** linear/poly/RBF kernels; soft-margin C, γ; support-vector highlighting; margin + kernel influence heatmap.
3. **Checkpoint:** both animate and deploy.

**Deliverable:** Logistic/Softmax and SVM live.

> **NOTE:** K-Means / K-Means++ (§6.7) was built and committed but has been **aborted and
> removed** per project owner decision (2026-06-01). Its code has been deleted; the git history
> retains it for reference.

---

## ~~Phase 5 — NLP: Embeddings & Self-Attention~~  *(ABORTED)*

> **NOTE:** This phase has been **aborted** per project owner decision (2026-06-01).
> No code was ever written for it. NLP (§6.14) is removed from the committed scope.
> Phase 6 (polish/QA/final deploy) is renumbered Phase 5 — see below.

---

## Phase 5 — Polish, QA, final deploy  *(was Phase 6)*

1. Cross-method consistency: tab layout, spacing, dark theme, responsive checks.
2. "Experiments / Insights" prompts (§4.6) per method; tighten copy & formulas.
3. Empty/error states; lazy-load TF.js model; verify no runtime external calls.
4. Run through Acceptance Criteria (§11) as a checklist.
5. Final `vite build` + Vercel production deploy; smoke-test the live URL.

**Deliverable:** Polished, deployed ML Explorer meeting §11.

---

## Dependency / ordering notes

- `plot.ts`, `datagen.ts`, `mathutils.ts`, `optimizers.ts` are built incrementally — each phase extends them as needed.
- `optimizers.ts` is shared by Regression (P1) and MLP (P3) — build the interface in P1.
- `heatmap.ts` (P2) is reused by MLP neuron viz (P3).
- CNN pre-trained weights are the only external asset; secure them early in Phase 2 (fallback: Mode A mechanics ship independently if weights slip).

## Risk register

| Risk | Mitigation |
|------|------------|
| CNN weights/model integration slips | Mode A (mechanics) ships standalone; Mode B is additive, not blocking. |
| Scope overrun | Phases are independently shippable; stop anywhere with a working demo. (Scope reduced to 5 committed methods.) |
| Animation perf jank | `requestAnimationFrame`, cap steps/frame, throttle redraws. |
| Vercel/build surprise | Deploy in Phase 0 and after every phase. |
| TF.js bundle size | Lazy-load TF.js + model only on the CNN route. |

## Build sequence (committed)

`Phase 0 → 1 → 2 → 3 → 4 → 5`
(Headline CNN intentionally early in P2; classics batched in P4; polish/QA in P5; always deployable.)
