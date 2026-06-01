# ML Explorer — Change Proposal: Abort K-Means & NLP

**Status: PROPOSAL — awaiting project owner approval. Nothing has been applied.**
**Prepared:** 2026-06-01
**Companion docs:** `EXECUTION_PLAN.md`, `SPECIFICATION.md` v2.1

---

## 1. Summary

Two committed features are being aborted: **K-Means / K-Means++** (built and committed in
`src/methods/kmeans/`) and **NLP — Embeddings & Self-Attention** (planned as Phase 5, never
started — no code exists). The resulting committed deliverable is **5 shipped methods**, down
from the original 7. All other aspects of the project — architecture, the remaining 5 methods,
deployment target (Vercel), and the deferred CNN Mode B — are unchanged.

---

## 2. Updated Committed Method List

Was 7 methods (§0, `SPECIFICATION.md`). Becomes **5**:

| # | Method | Spec section | Status |
|---|--------|-------------|--------|
| 1 | Linear & Polynomial Regression + Regularization | §6.1 | Shipped |
| 2 | Logistic / Softmax Regression | §6.2 | Shipped |
| 3 | Support Vector Machine (linear/poly/RBF kernels) | §6.3 | Shipped |
| 4 | Neural Network MLP Playground (advanced) | §6.12 | Shipped |
| 5 | CNN Visualizer — Mode A (animated convolution mechanics) | §6.13 | Shipped |

**Removed:** K-Means / K-Means++ (§6.7) and NLP — Embeddings & Self-Attention (§6.14).
**Remains deferred (no change):** CNN Mode B (live classification + activation flow) — deferred
since Phase 2, status unchanged.

---

## 3. Proposed Edits to `EXECUTION_PLAN.md`

### 3.1 Phase 4 — drop K-Means

**Current text (lines 85–94):**
```
## Phase 4 — Classification & clustering classics

**Goal:** Round out breadth with three more polished methods.

1. **Logistic / Softmax Regression (§6.2):** binary + multinomial; BCE/CE loss; probability heatmap; animated boundary; confusion matrix.
2. **Support Vector Machine (§6.3):** linear/poly/RBF kernels; soft-margin C, γ; support-vector highlighting; margin + kernel influence heatmap.
3. **K-Means / K-Means++ (§6.7):** animated assign→update with centroid trails; inertia per iteration; Voronoi; elbow chart.
4. **Checkpoint:** all three animate and deploy.

**Deliverable:** Logistic/Softmax, SVM, K-Means live.
```

**Proposed replacement:**
```
## Phase 4 — Classification classics  *(COMPLETE)*

**Goal:** Round out breadth with two more polished methods.

1. **Logistic / Softmax Regression (§6.2):** binary + multinomial; BCE/CE loss; probability heatmap; animated boundary; confusion matrix.
2. **Support Vector Machine (§6.3):** linear/poly/RBF kernels; soft-margin C, γ; support-vector highlighting; margin + kernel influence heatmap.
3. **Checkpoint:** both animate and deploy.

**Deliverable:** Logistic/Softmax and SVM live.

> **NOTE:** K-Means / K-Means++ (§6.7) was built and committed but has been **aborted and
> removed** per project owner decision (2026-06-01). Its code has been deleted; the git history
> retains it for reference.
```

---

### 3.2 Phase 5 (NLP) — remove the phase entirely

**Current text (lines 98–108):**
```
## Phase 5 — NLP: Embeddings & Self-Attention  *(§6.14)*

**Goal:** The committed NLP feature.

1. Tokenization view (text → tokens → ids).
2. In-browser sentiment classifier (bag-of-words/embeddings + logistic regression) with live prediction + top influential tokens.
3. Bundled small word-embeddings → 2D projection (PCA) + nearest neighbors.
4. **Single-head self-attention heatmap** for a short sentence (Q/K/V intuition).
5. **Checkpoint:** type text → live sentiment + attention heatmap; deploy.

**Deliverable:** NLP method live — all 7 committed methods shipped.
```

**Proposed replacement:**
```
## ~~Phase 5 — NLP: Embeddings & Self-Attention~~  *(ABORTED)*

> **NOTE:** This phase has been **aborted** per project owner decision (2026-06-01).
> No code was ever written for it. NLP (§6.14) is removed from the committed scope.
> Phase 6 (polish/QA/final deploy) is renumbered Phase 5 — see below.
```

---

### 3.3 Phase 6 — renumber to Phase 5

**Current heading (line 112):**
```
## Phase 6 — Polish, QA, final deploy
```

**Proposed replacement:**
```
## Phase 5 — Polish, QA, final deploy  *(was Phase 6)*
```

No other content in that section needs to change. The internal reference to "§11" remains valid.

---

### 3.4 Dependency / ordering notes — update NLP reference

**Current text (line 128):**
```
- `heatmap.ts` (P2) is reused by MLP neuron viz (P3) and attention heatmap (P5).
```

**Proposed replacement:**
```
- `heatmap.ts` (P2) is reused by MLP neuron viz (P3).
```

(The attention heatmap reference is removed because Phase 5 / NLP is aborted.)

---

### 3.5 Risk register — update two rows

**Current row (line 136):**
```
| Scope overrun on 7 methods | Phases are independently shippable; stop anywhere with a working demo. |
```

**Proposed replacement:**
```
| Scope overrun | Phases are independently shippable; stop anywhere with a working demo. (Scope reduced to 5 committed methods.) |
```

**Current row (line 139):**
```
| TF.js bundle size | Lazy-load TF.js + model only on CNN/NLP routes. |
```

**Proposed replacement:**
```
| TF.js bundle size | Lazy-load TF.js + model only on the CNN route. |
```

(NLP no longer exists as a route.)

---

### 3.6 Build sequence (committed) — update

**Current text (lines 141–144):**
```
## Build sequence (committed)

`Phase 0 → 1 → 2 → 3 → 4 → 5 → 6`
(Headline CNN intentionally early in P2; classics batched in P4; always deployable.)
```

**Proposed replacement:**
```
## Build sequence (committed)

`Phase 0 → 1 → 2 → 3 → 4 → 5`
(Headline CNN intentionally early in P2; classics batched in P4; polish/QA in P5; always deployable.)
```

---

## 4. Proposed Edits to `SPECIFICATION.md`

### 4.1 §0 — "Locked Decisions" final method list (lines 20–29)

**Current text:**
```
**Final method list (shipped):**
1. Linear & Polynomial Regression + regularization — §6.1
2. Logistic / Softmax Regression — §6.2
3. Support Vector Machine (linear/poly/RBF kernels) — §6.3
4. K-Means / K-Means++ — §6.7
5. Neural Network MLP Playground (advanced) — §6.12
6. ★ CNN Visualizer (Mode A mechanics + Mode B pre-trained classifier) — §6.13
7. NLP — Embeddings & Self-Attention — §6.14
```

**Proposed replacement:**
```
**Final method list (shipped) — updated 2026-06-01, 5 methods:**
1. Linear & Polynomial Regression + regularization — §6.1
2. Logistic / Softmax Regression — §6.2
3. Support Vector Machine (linear/poly/RBF kernels) — §6.3
4. Neural Network MLP Playground (advanced) — §6.12
5. ★ CNN Visualizer — Mode A (animated convolution mechanics) — §6.13

> **Removed from committed scope (2026-06-01):**
> - ~~K-Means / K-Means++~~ (§6.7) — built and committed, then aborted; code removed.
> - ~~NLP — Embeddings & Self-Attention~~ (§6.14) — never started; phase aborted.
>
> **Remains deferred (no change):** CNN Mode B (live classification + activation flow).
```

Also update Decision #3 in the table (line 17):

**Current:**
```
| 3 | **Scope** | **Focused & polished** — 6 core methods + NLP (see final list below) |
```

**Proposed:**
```
| 3 | **Scope** | **Focused & polished** — 5 committed methods (K-Means and NLP removed 2026-06-01; see final list below) |
```

And remove Decision #4 (NLP depth) since it no longer applies:

**Current:**
```
| 4 | **NLP depth** | Illustrative **single-head self-attention heatmap** + in-browser sentiment + word embeddings |
```

**Proposed:**
```
| 4 | ~~**NLP depth**~~ | ~~Illustrative single-head self-attention heatmap + in-browser sentiment + word embeddings~~ — **ABORTED 2026-06-01** |
```

---

### 4.2 §5 — Global UI / Layout sidebar description (lines 89–95)

**Current text:**
```
- **Left sidebar (grouped nav):**
  - *Regression* → Linear/Polynomial Regression (+ Ridge/Lasso)
  - *Classification* → Logistic/Softmax Regression, Support Vector Machine, K-Nearest Neighbors, Decision Tree & Ensembles, Naive Bayes
  - *Unsupervised* → K-Means (++), Gaussian Mixture Models (EM), DBSCAN, PCA, t-SNE
  - *Neural Networks* → MLP Playground (advanced)
  - *Deep Learning* → **CNN Visualizer**, NLP: Embeddings & Self-Attention
```

**Proposed replacement:**
```
- **Left sidebar (grouped nav) — committed methods only:**
  - *Regression* → Linear/Polynomial Regression (+ Ridge/Lasso)
  - *Classification* → Logistic/Softmax Regression, Support Vector Machine
  - ~~*Unsupervised*~~ → *(group hidden — K-Means removed; no committed methods in this group)*
  - *Neural Networks* → MLP Playground (advanced)
  - *Deep Learning* → **CNN Visualizer** (Mode A)
  - *(All other listed methods remain deferred)*
```

---

### 4.3 §6.7 (K-Means) — mark as removed

Add the following note at the **top** of the §6.7 block (before the bullet list), immediately after the heading on line 142:

```
> **REMOVED — 2026-06-01.** K-Means / K-Means++ has been removed from the committed scope.
> The implementation was built and committed, then aborted by project owner decision.
> The code has been deleted from `src/methods/kmeans/`; git history retains it.
> This section is kept for reference only.
```

---

### 4.4 §6.14 (NLP) — mark as removed

Add the following note at the **top** of the §6.14 block (before the bullet list), immediately after the heading on line 196:

```
> **REMOVED — 2026-06-01.** NLP — Embeddings & Self-Attention has been removed from the
> committed scope. No code was ever written for this feature. This section is kept for
> reference only.
```

---

### 4.5 §9 — Scope & Prioritization (lines 240–244)

**Current text:**
```
**P0 (must):** App shell + nav + 6-tab layout; Linear/Poly Regression; Logistic/Softmax; K-Means(++); **MLP Playground (advanced)**; **CNN Visualizer Mode A (animated convolution sweep + feature maps + pooling)**.
**P1 (should):** SVM (kernels), KNN, Decision Tree/Ensembles, GMM (EM), PCA, **CNN Mode B (live digit classification + activation flow)**, NLP sentiment + attention heatmap.
**P2 (nice):** Naive Bayes, DBSCAN, t-SNE, embedding arithmetic, image upload, editable kernel matrix, regularization-path/elbow extras.
```

**Proposed replacement:**
```
**P0 (must — delivered):** App shell + nav + 6-tab layout; Linear/Poly Regression; Logistic/Softmax; SVM; **MLP Playground (advanced)**; **CNN Visualizer Mode A (animated convolution sweep + feature maps + pooling)**.
**P1 (deferred):** CNN Mode B (live digit classification + activation flow), KNN, Decision Tree/Ensembles, GMM (EM), PCA.
**P2 (nice-to-have):** Naive Bayes, DBSCAN, t-SNE, image upload, editable kernel matrix.

> **Removed from all tiers (2026-06-01):** K-Means / K-Means++ and NLP — Embeddings & Self-Attention.
```

---

### 4.6 §11 — Acceptance Criteria: remove K-Means and NLP lines

**Current lines to remove (lines 262 and 265):**
```
- [ ] SVM shows support vectors & kernel boundary; K-Means centroids converge with inertia readout; GMM ellipses animate; PCA shows components + scree.
```
and
```
- [ ] NLP: live sentiment prediction on typed text + attention heatmap renders.
```

**Proposed replacements:**

Line 262 — remove the K-Means clause:
```
- [ ] SVM shows support vectors & kernel boundary.
```
(GMM and PCA are deferred methods and were never acceptance criteria for the current build; they should simply be removed from the criterion rather than carried forward as false obligations.)

Line 265 — remove the NLP line entirely (no replacement needed; NLP is fully aborted).

---

## 5. Implementation Consequences (to execute ONLY after approval)

This is an ordered checklist of the actual removal work. None of this happens until the project
owner approves this proposal.

- [ ] **Delete `src/methods/kmeans/`** — the entire directory and all 9 files within it:
  `kmeans.ts`, `dataStore.ts`, `Overview.tsx`, `Formulas.tsx`, `Dataset.tsx`,
  `Visualization.tsx`, `Insights.tsx`, `index.ts`, `Description.tsx`.

- [ ] **Remove the K-Means import from `src/main.tsx`** — delete line 10 exactly:
  ```
  import './methods/kmeans/index'
  ```
  The surrounding lines (before: `import './methods/svm/index'`, after:
  `import './methods/cnnVisualizer/index'`) remain untouched.

- [ ] **Remove the K-Means welcome card from `src/App.tsx`** — the third entry in the
  quick-start cards array (lines 47–53 of `App.tsx`) references `id: 'kmeans'` and should
  be deleted (the card for "K-Means" with icon `⬡`). The remaining two cards (Linear
  Regression, CNN Visualizer) stay.

- [ ] **Update the hash-routing comment in `src/App.tsx` (~line 88)** — after the above
  deletions, a cosmetic comment in `App.tsx` will still mention `#/kmeans` as a routing
  example (e.g. `// Hash-based routing: #/linear-regression, #/kmeans, etc.`). Replace
  `#/kmeans` with another existing route (e.g. `#/svm`) or simply remove the token so the
  comment no longer references a deleted route.

- [ ] **Verify no remaining references** — run a project-wide search for `kmeans` (case-
  insensitive). After the above steps (including the routing-comment update), the only
  survivor is the `'Unsupervised'` entry in `GROUP_ORDER` in `src/methods/registry.ts` —
  that string causes no harm and is addressed by the dedicated checklist item below. No
  other `kmeans` references should remain in `src/`.

- [ ] **Remove `'Unsupervised'` from `GROUP_ORDER` in `src/methods/registry.ts` (~line 33)**
  — **Recommended (owner-confirmable at final approval).** No committed method remains in
  the `'Unsupervised'` group; keeping the entry is harmless but cosmetically stale and
  misleads future contributors. Removing it keeps the nav config consistent with what is
  actually shipped. (See §7.1 for rationale; the owner may decline this specific cleanup
  at final approval, but the default plan is to include it in the removal commit.)

- [ ] **Confirm the sidebar no longer shows "Unsupervised"** — with K-Means removed, no
  method registers into the `'Unsupervised'` group. The `getNavGroups()` function in
  `src/methods/registry.ts` only pushes a group when `groupMap.get(label)` returns items,
  so the group disappears automatically. No sidebar code changes are strictly required
  beyond the `GROUP_ORDER` cleanup above (see §7.1).

- [ ] **`npm run build` must pass** after the deletions. Verify there are no lingering
  TypeScript imports or references that would cause a compile error.

- [ ] **Suggested commit message:**
  ```
  Remove K-Means: abort committed feature per project owner decision

  Delete src/methods/kmeans/ (9 files), remove the registration import
  from src/main.tsx, and remove the K-Means welcome card from App.tsx.
  The implementation was fully built but has been cut from scope.
  Git history retains the work for future reference.

  NLP (Phase 5) requires no code removal — it was never implemented.
  ```

- [ ] **NLP — no code removal required.** NLP (§6.14 / former Phase 5) was never started;
  there is no `src/methods/nlp/` directory and no NLP import in `src/main.tsx`. The only
  action is the documentation updates to `EXECUTION_PLAN.md` and `SPECIFICATION.md`
  described in §§3–4 above.

- [ ] **Apply documentation updates** to `EXECUTION_PLAN.md` and `SPECIFICATION.md` as
  described in §§3–4 above, in a separate commit (or the same commit — project owner's
  preference).

---

## 6. What Is NOT Changing

- The 5 shipped methods (Linear/Poly Regression, Logistic/Softmax Regression, SVM, MLP
  Playground, CNN Visualizer Mode A) are untouched.
- The 6-tab method layout, shared libs (`plot.ts`, `datagen.ts`, `mathutils.ts`,
  `optimizers.ts`, `heatmap.ts`, `losschart.ts`), and the method registry framework are
  untouched.
- CNN Mode B (live classification + activation flow) remains **deferred** — neither built
  nor formally aborted; it retains its §6.13 specification.
- Deployment target: **Vercel** (static build via `vite build`).
- Tech stack: Vite + React + TypeScript + Tailwind CSS + TensorFlow.js.
- All other methods listed in §6 of the spec (KNN, Decision Tree, GMM, DBSCAN, PCA, t-SNE,
  etc.) remain in their current deferred state.

---

## 7. Open Considerations / Risks

### 7.1 "Unsupervised" sidebar group becomes empty

After K-Means is removed, **no committed method belongs to the `'Unsupervised'` group**. The
`getNavGroups()` function in `src/methods/registry.ts` already handles this gracefully — it
only adds a group to the nav when at least one method has registered under that label, so the
group disappears from the sidebar automatically without any code change.

However, `GROUP_ORDER` in `registry.ts` (line 33) still includes `'Unsupervised'`. This is
harmless but cosmetically stale. The default plan is to remove it — see the dedicated
checklist item in §5 ("Remove `'Unsupervised'` from `GROUP_ORDER`…"), which is marked
**Recommended (owner-confirmable at final approval)**. The owner may decline this specific
cleanup at final approval, but it is included in the proposed removal commit because no
deferred methods are expected to appear in that group in the near term and it simplifies
future onboarding.

### 7.2 K-Means welcome card and routing comment in App.tsx

`src/App.tsx` contains two `kmeans` references that must be cleaned up manually:

1. **Quick-start card** — a hard-coded `id: 'kmeans'` entry (the `⬡` card). This is a static
   JSX array, not registered through the method registry. It is covered by the §5 checklist
   item "Remove the K-Means welcome card from `src/App.tsx`."
2. **Hash-routing comment** (~line 88) — a cosmetic comment that cites `#/kmeans` as a
   routing example. It is covered by the §5 checklist item "Update the hash-routing comment
   in `src/App.tsx` (~line 88)."

After both items are applied, the "Verify no remaining references" grep step should confirm
that no `kmeans` token survives in `src/`.

### 7.3 Git history preservation

The K-Means implementation (committed in the `5a71940` commit, "Phase 4: K-Means clustering")
will be **permanently accessible in git history** even after deletion. The removal is fully
reversible via `git revert` or `git checkout <sha> -- src/methods/kmeans/`. No work is lost.

### 7.4 No runtime impact from NLP removal

Because NLP was never started, removing it from scope has zero runtime or build impact. The
only artifact is documentation cleanup.

### 7.5 Acceptance criteria update is important

§11 currently lists "K-Means centroids converge with inertia readout" and "NLP: live sentiment
prediction + attention heatmap" as acceptance criteria. These must be removed before the final
QA pass (now Phase 5 instead of Phase 6) to avoid false failures against the checklist.

---

*End of proposal. No files have been modified. Awaiting project owner approval before any
changes are applied.*
