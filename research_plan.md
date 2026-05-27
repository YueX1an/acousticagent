# Research Plan: LLM-Guided Inverse Design with Domain-Agnostic Physical Reasoning

## Target Venues

| Priority | Venue | Type | Deadline | Acceptance |
|----------|-------|------|----------|------------|
| **Primary** | AAAI 2027 | AI Conference | ~Aug 2026 | ~18% |
| **Backup** | Engineering Applications of AI (IF 7.5) | AI Journal | Rolling | ~22% |

---

## 0. Pre-Submission Review (Self-Critique)

### Round 1 & 2 Findings (Addressed)

| Concern | Severity | Resolution |
|---------|:--------:|------------|
| "Multi-Agent" label applied to algorithmic modules | **Critical** | Reframed as **LLM-Guided Optimization** |
| DomainSpec dataclass is software engineering | **High** | Repositioned as **structured prompt interface** |
| Hardcoded Critic violates "AI agent" legitimacy | **Critical** | Replaced with **LLM-based Critic**; heuristic Critic = ablation baseline only |
| Text-to-Math Bridge: NLP → gradient signal | **High** | Formalized penalty function with structured JSON schema (Section 3.6) |
| LLM Hallucination vs Physics Reality | **High** | Hallucination Rate via expert blind review + Failure Cases (Section 4.6) |
| Surrogate-LLM "Double Blind" Problem | **Medium** | LLM as surrogate error detector (Section 3.7, 4.5) |
| Token Cost concerns | Low | Qwen2.5-14B vs GPT-4o-mini cost comparison (Section 4.4) |

### Round 3 Findings (Addressed in This Version)

| Concern | Severity | Resolution |
|---------|:--------:|------------|
| **Gradient Sparsity** — max(0,·)² term has zero gradient when target is met; dominates when target is too aggressive | **High** | Replaced with **Softplus activation** + **dynamic τ scaling** to ensure smooth gradient flow (Section 3.6.1) |
| **Non-Deterministic Weight Oscillation** — LLM may output different W between rounds, causing loss landscape jumps that break Adam momentum | **High** | Added **Feedback Smoothing** (EMA over penalty terms across rounds) in Orchestrator (Section 3.6.2) |
| **GA Fallback Unfairness** — if system reverts to GA mid-loop, comparing against pure GA/optimizer is biased | **Medium** | GA fallback **disabled in main experiments**; reserved as separate Extension Study (Section 4.7) |
| **OOD Error Injection** — random Gaussian noise on weights ≠ real surrogate failures; real errors come from extrapolation OOD | **High** | Changed error injection to **OOD-based**: deliberately remove training data from a target frequency band to force false-positive extrapolation (Section 3.7, 4.5) |
| **Domain-Specific Hallucination Quantification** — LLM likely has more acoustics knowledge than concrete materials knowledge | **Medium** | Hallucination Rate broken down by domain; correlated with LLM training corpus coverage estimate (Section 4.6.1) |

**Resulting narrative:**

> *We propose using an LLM as a scientific critic — it receives a structured description of any inverse design problem (DomainSpec), examines the current design's performance, and provides quantifiable feedback that is mechanically translated into a differentiable penalty function via a formal Text-to-Math bridge. This closed-loop system is validated across three physics domains with the same codebase. We further analyze: (1) when the LLM catches surrogate model errors that would otherwise produce non-physical designs, and (2) when the LLM itself hallucinates incorrect physical advice.*

---

## 1. Problem & Motivation

### 1.1 The Inverse Design Bottleneck

Inverse design — finding parameters X that produce target performance Y — is the central problem in engineering. The standard pipeline is:

```
Define target → Run optimizer → Get design → Done
```

This is **open-loop**. The optimizer has no understanding of physical plausibility. It can produce designs that are numerically optimal but physically invalid (non-causal spectra, impossible geometries). Prior work addresses this with domain-specific validation heuristics — but those don't transfer to new domains.

### 1.2 Recent Competition

| Paper | LLM? | Domain-Agnostic? | Key Limitation |
|-------|:---:|:---:|------|
| Lu et al. (Duke, ACS Photonics 2025) | Yes | No | Pipeline is handcrafted for photonics only |
| AutoMS (Zhao et al., 2025) | Yes | No | Evolutionary search with domain-specific operators |
| SAGA (2025) | Yes | Partial | Requires new scoring function per domain |
| GenAI vs BO (Awan et al., 2025) | LLM as optimizer | No | Single-shot generation, no refinement loop |

None provide a principled mechanism for an LLM to reason about **arbitrary** inverse design domains without per-domain engineering.

### 1.3 Our Thesis

> An LLM, given a structured domain description (DomainSpec), can serve as a domain-agnostic scientific critic — it reasons about physical validity, identifies weaknesses with quantifiable targets, and provides feedback that is mechanically translated into a differentiable loss function. This closed-loop system achieves better designs than open-loop optimization, and the LLM further serves as an error detector for surrogate model predictions.

---

## 2. Novelty & Contributions

| # | Contribution | AAAI Viability |
|---|-------------|:---:|
| **C1** | **LLM-as-Scientific-Critic** — LLM autonomously critiques design outputs from pre-trained physics knowledge, providing structured quantifiable feedback | **Strong** — LLM reasoning + scientific domain; the "critic" role is underexplored |
| **C2** | **Text-to-Math Bridge** — A formal mechanism that translates structured LLM feedback (JSON) into differentiable penalty terms in the Design optimization loss function | **Strong** — addresses the key "how does NLP become gradient signal?" question rigorously |
| **C3** | **DomainSpec as Structured Prompt Interface** — YAML domain specs auto-generate prompts enabling zero-shot cross-domain LLM physical reasoning | **Moderate** — prompt engineering / in-context learning contribution |
| **C4** | **Cross-Domain Validation + Surrogate Error Detection** — Same codebase on 3 domains; LLM Critic catches non-physical surrogate predictions | **Strong** — cross-domain generalization + novel double-blind analysis |

---

## 3. Technical Approach

### 3.1 System Architecture

```
                         ┌──────────────────────────┐
                         │     DOMAIN SPEC (YAML)     │
                         │  input_dim, output_dim     │
                         │  bounds, constraints       │
                         │  targets, domain context   │
                         └────────────┬───────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  Auto-generates structured prompt   │
                    └─────────────────┬─────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│    DESIGNER      │     │    LLM CRITIC        │     │    ORCHESTRATOR     │
│  (Algorithmic)   │◄───►│  (AI Reasoning)      │◄───►│  (Control Logic)    │
│                  │     │                      │     │                     │
│ • Multi-start    │     │ • Structured JSON    │     │ • Manages loop      │
│   Adam optimizer │     │   feedback with      │     │ • Checks convergence│
│ • Text-to-Math   │     │   quantifiable       │     │ • Routes feedback   │
│   bridge converts│     │   targets & weights  │     │ • Tracks history    │
│   JSON to loss   │     │ • Physical validity  │     │                     │
│ • Bounded        │     │   reasoning          │     │                     │
│   projection     │     │ • Causal analysis    │     │                     │
│ • Domain-agnostic│     │ • Surrogate error    │     │                     │
│                  │     │   detection          │     │                     │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘
```

### 3.2 Core Component: LLM Critic

The LLM Critic receives an auto-generated prompt from DomainSpec and returns **structured JSON with quantifiable targets**. This is the primary AI contribution.

**Structured Output Schema (enforced via prompt + JSON retry):**

```json
{
  "verdict": "pass" | "refine",
  "physical_violations": [
    {
      "type": "causality" | "smoothness" | "bounds" | "other",
      "indices": [0, 3],
      "description": "Non-causal absorption at low frequency",
      "penalty_weight": 10.0
    }
  ],
  "weak_regions": [
    {
      "indices": [20, 40],
      "label": "400-800 Hz",
      "current_mean": 0.52,
      "target": 0.70,
      "weight": 1.5,
      "reason": "Anti-resonance from layer-2 cavity depth mismatch"
    }
  ],
  "causal_analysis": "The absorption dip in 400-800 Hz is likely caused by...",
  "exploration_suggestion": "Increase layer-2 cavity depths by 10-20%.",
  "confidence": 0.85
}
```

**Why structured JSON is essential**: Unstructured natural language feedback ("the design is weak at mid frequencies") cannot be translated into a loss function. The `weak_regions[indices]`, `target`, and `weight` fields provide the exact numerical bridge.

**Handling non-differentiable feedback**: If the LLM suggests a topological change (e.g., "add an extra cavity layer"), the system:
1. Flags the suggestion as non-differentiable (cannot be gradient-optimized)
2. Translates it to: expanded parameter bounds + noise injection on the affected parameter group
3. If the suggestion persists across rounds: triggers a fallback to evolutionary search (GA) in that parameter subspace
4. Logs the suggestion for human review

### 3.3 Supporting Component: Designer (Algorithmic Module)

The Designer performs multi-start gradient descent. It reads DomainSpec for problem dimensions and bounds.

### 3.4 Supporting Component: Orchestrator (Control Logic)

Convergence-gated loop: Designer optimizes → LLM Critic reviews → if "refine," feedback passed to next round. Terminates on "pass" or stalled improvement (δ < ε for 2 consecutive rounds).

### 3.5 DomainSpec: The Enabling Interface

Three example domains (abbreviated — full YAML configs in `configs/`):

| Property | Acoustic Metamaterial | Airfoil Aerodynamics | Concrete Materials |
|----------|:---:|:---:|:---:|
| Input dim | 31 | 16 | 8 |
| Output dim | 100 | 2 | 1 |
| Output type | spectrum | vector | scalar |
| Data source | In-house FEM | [CST+CFD Dataset, MIT](https://github.com/kanakaero/Dataset-of-Aerodynamic-and-Geometric-Coefficients-of-Airfoils) | [UCI #165](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) |
| Surrogate | ResNetDNN (31→100) | ResNetDNN (16→2) | ResNetDNN (8→1) |

### 3.6 The Text-to-Math Bridge (Core Algorithmic Contribution)

This section addresses the central technical question: **How does natural language feedback become a gradient signal?**

**Step 1: Structured Feedback Extraction**

The LLM Critic outputs JSON with `weak_regions`, each containing quantifiable fields: `indices`, `target`, `weight`.

**Step 2: Penalty Function Construction**

The Designer's loss function is formally defined as:

$$\mathcal{L}(\theta) = \underbrace{\mathcal{L}_{obj}(\theta)}_{\text{Primary objective}} + \underbrace{\sum_{j \in \mathcal{C}} \gamma_j \cdot \Phi_j(f(\theta))}_{\text{Hard constraint penalties}} + \underbrace{\lambda \cdot \sum_{i \in \mathcal{W}} \text{Softplus}\left(\tau_i - f_i(\theta)\right)}_{\text{LLM-guided weak region penalties}}$$

Where:

| Symbol | Meaning | Source |
|--------|---------|--------|
| $\theta$ | Design parameters | Optimizer state |
| $f(\theta)$ | Surrogate model prediction | Pre-trained ResNetDNN |
| $\mathcal{L}_{obj}$ | Primary objective loss | DomainSpec `optimization_goal` |
| $\mathcal{C}$ | Set of hard constraints | DomainSpec `physical_constraints` |
| $\Phi_j$ | Constraint violation function | Domain-specific |
| $\gamma_j$ | Fixed penalty weight (10.0–100.0) | DomainSpec |
| $\mathcal{W}$ | Set of weak region indices | **LLM Critic output** |
| $\tau_i$ | Target performance per weak region | **LLM Critic output**, dynamically scaled (see 3.6.1) |
| $\lambda$ | Adaptive refinement weight | **LLM Critic output**, smoothed across rounds (see 3.6.2) |

**Key design choice — Softplus activation instead of max(0,·)²:**

The original $\max(0, \tau_i - f_i(\theta))^2$ formulation has a critical flaw: when $f_i(\theta) \geq \tau_i$, the gradient vanishes entirely. If the LLM sets $\tau_i$ too conservatively, the penalty term contributes nothing and the refinement loop wastes rounds. Conversely, if $\tau_i$ is too aggressive, the penalty dominates and the optimizer abandons $\mathcal{L}_{obj}$.

We use $\text{Softplus}(x) = \log(1 + e^x)$ which:
- Has **non-zero gradient everywhere** — prevents gradient sparsity even when target is met
- **Asymptotically approaches max(0, x)** for large $|x|$ — preserves the intended penalty behavior
- Provides **smooth transition** near $x = 0$ — no abrupt landscape jumps

#### 3.6.1 Dynamic τ Scaling

To prevent the LLM from setting unrealistically high targets:

$$\tau_i^{(t)} = \tau_i^{(0)} \cdot \min\left(1.0, \; f_i(\theta^{(t-1)}) + \alpha \cdot (\tau_i^{\text{LLM}} - f_i(\theta^{(t-1)}))\right)$$

Where $\alpha \in (0, 1]$ is a trust factor controlling how aggressively we adopt the LLM's target. In early rounds, $\alpha$ is lower (0.3–0.5) to prevent LLM hallucination from derailing optimization. $\alpha$ increases with each round as the surrogate's predictions stabilize.

#### 3.6.2 Feedback Smoothing (Preventing Loss Landscape Oscillation)

If the LLM changes its mind between rounds and outputs a completely different set of weak regions $\mathcal{W}^{(t)}$, the loss landscape shifts abruptly, breaking Adam's momentum and potentially causing oscillatory divergence.

**Solution — Exponential Moving Average over penalty terms:**

$$\mathcal{W}_{\text{eff}}^{(t)} = \mathcal{W}^{(t)} \cup \{i \in \mathcal{W}^{(t-1)}_{\text{eff}} \mid f_i(\theta^{(t)}) < \tau_i^{(t-1)} \cdot 1.1\}$$

$$\lambda_i^{(t)} = \begin{cases} \lambda_i^{\text{LLM}} & \text{if } i \in \mathcal{W}^{(t)} \\ \beta \cdot \lambda_i^{(t-1)} & \text{if } i \in \mathcal{W}^{(t-1)}_{\text{eff}} \setminus \mathcal{W}^{(t)} \end{cases}$$

Where $\beta \in (0, 1)$ is a decay factor (default 0.5). This means:
- Regions the LLM currently flags as weak → use full LLM-assigned weight
- Regions the LLM stopped flagging but still haven't met target → retain at 50% weight, decaying each round
- Regions that met target → removed from penalty

This prevents the loss landscape from undergoing step-function changes between rounds, preserving Adam's momentum and ensuring smooth optimization trajectories.

**Step 3: Gradient Flow**

$\text{Softplus}(\tau_i - f_i(\theta))$ is fully differentiable with respect to $\theta$ through the surrogate model $f$. The gradient $\nabla_\theta \mathcal{L}$ flows through the surrogate's Jacobian, guiding parameter updates toward regions that satisfy both the primary objective and the LLM's refinement targets.

**Step 4: Fallback for Non-Differentiable Feedback**

| LLM Suggestion Type | Mechanical Translation |
|---------------------|----------------------|
| Parameter adjustment (in-range) | Weighted Softplus penalty (standard path) |
| Parameter adjustment (out-of-bounds) | Expand bounds for affected params + re-project |
| "Explore more diverse designs" | Increase multi-start count + noise magnitude |
| Topology / structural change | **DISABLED in main experiments** — reserved as Extension Study in Appendix |

### 3.7 The Surrogate-LLM "Double Blind" Problem

A scientifically interesting question arises when the surrogate model $f$ produces a false-positive prediction:

- **Real failure mode**: The optimizer pushes $\theta$ into a region where the surrogate has no training data (out-of-distribution, OOD). The surrogate extrapolates erroneously, predicting an impossibly high performance. The optimizer converges to this false optimum.
- **Surrogate signal**: "This design has exceptionally high performance → lock in"
- **LLM signal**: "This output violates physical constraints → reject"

**Research Question**: When these signals conflict, does the LLM Critic catch the OOD false-positive using its pre-trained physics knowledge?

**OOD Error Injection Protocol** (replaces naive Gaussian weight noise):

1. **Training data ablation**: Deliberately remove all training samples from a target output band (e.g., for acoustic metamaterial: remove all designs with absorption > 0.85 in the 500–600 Hz range; for airfoil: remove all designs with Cl > 1.5; for concrete: remove all designs with strength > 60 MPa)
2. **Retrain surrogate**: The model learns to interpolate well in-distribution but has no knowledge of the high-performance region
3. **Run optimization**: The Designer, unaware of the data gap, pushes parameters toward the high-performance OOD region. The surrogate confidently extrapolates (wrongly) → predicts false-positive high performance
4. **LLM Critic evaluation**: Present the OOD false-positive design to the LLM Critic. Can it detect that the predicted performance violates physical limits?

**Metrics**:
- **Correction Rate**: Fraction of OOD false-positives correctly rejected by LLM Critic
- **False Alarm Rate**: Fraction of valid (in-distribution) designs incorrectly flagged
- **Detection Latency**: Number of refinement rounds before LLM catches the error
- **Breakdown by ablation severity**: Compare 10%/30%/50% data removal thresholds

This protocol models real surrogate failure modes — not artificial weight noise — and directly tests whether LLM physics knowledge serves as a safety net against OOD extrapolation errors.

---

## 4. Experiment Plan

### 4.1 Comparison Methods

| Method | Type | Description |
|--------|------|-------------|
| **LLM-Guided (Ours)** | Proposed | Designer + LLM Critic (Qwen2.5-14B) + Orchestrator |
| Heuristic-Guided | Ablation | Designer + Heuristic Critic (rule-based) + Orchestrator |
| Designer-Only | Ablation | Multi-start gradient optimization, single-shot |
| Genetic Algorithm | Baseline | NSGA-II, pop=100, gen=50 |
| Bayesian Optimization | Baseline | GP with Matern kernel, 100 iterations |
| Random Search | Baseline | Uniform sampling, 5000 trials |

### 4.2 Primary Hypotheses

| # | Hypothesis | Success Criterion |
|---|-----------|-------------------|
| **H1** | LLM Critic produces better designs than Heuristic Critic | LLM > Heuristic by ≥5% on ≥2 domains |
| **H2** | LLM Critic converges in fewer refinement rounds | LLM rounds < Heuristic rounds on ≥2 domains |
| **H3** | LLM-Guided closed loop outperforms all open-loop baselines | Ours > GA, BO, Random on ≥2 domains |
| **H4** | Cross-domain generalization holds | All 3 domains complete with zero code changes |
| **H5** | LLM Critic catches OOD surrogate errors | Correction Rate > 60% on OOD errors |
| **H6** | Hallucination Rate correlates with domain corpus coverage | Acoustic < Airfoil < Concrete hallucination rates |

### 4.3 Evaluation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| Normalized Objective | Primary | Domain-specific objective [0,1] |
| Surrogate Calls | Efficiency | Forward passes through surrogate |
| Wall-Clock Time | Efficiency | Total execution time |
| **Token Cost** | **Efficiency** | **Tokens consumed by LLM Critic; compared across Qwen2.5-14B (local, free) and GPT-4o-mini (API)** |
| Refinement Gain | Quality | Improvement from round 1 to final round |
| Physical Validity Score | Quality | Pass/fail on all hard constraints |
| **Hallucination Rate** | **Quality** | **Fraction of LLM suggestions rated "physically incorrect" by domain expert blind review** |
| **Correction Rate** | **Quality** | **Fraction of injected surrogate errors caught by LLM Critic** |
| Convergence Rounds | Efficiency | Refinement rounds until convergence |

### 4.4 Ablation Experiments (Acoustic Domain Only)

| # | Ablation | What We Learn |
|---|----------|---------------|
| A1 | Remove LLM Critic (Designer-Only) | Contribution of refinement loop |
| A2 | Replace LLM with Heuristic Critic | Contribution of LLM reasoning over rules |
| A3 | Remove DomainSpec context from prompt | Contribution of structured domain knowledge |
| A4 | Vary LLM backbone: Qwen2.5-14B vs GPT-4o-mini | Sensitivity to model capability + cost trade-off |
| A5 | Vary refinement rounds (1, 2, 3, 5) | Diminishing returns analysis |
| **A6** | **Inject surrogate errors at known locations** | **Correction Rate + False Alarm Rate (H5)** |

### 4.5 Surrogate Error Detection Experiment (OOD-Based)

A dedicated experiment studying real surrogate failure modes:

1. **Data ablation** (per domain):
   - Acoustic: Remove training samples with avg absorption > 0.85 in 500–600 Hz band
   - Airfoil: Remove training samples with Cl > 1.5
   - Concrete: Remove training samples with strength > 60 MPa
   - Three severity levels: remove 10%, 30%, 50% of target-band samples

2. **Retrain surrogate** on ablated dataset → model has OOD blind spot

3. **Run LLM-Guided optimization** → Designer pushes toward high-performance region → surrogate extrapolates erroneously

4. **Measurement**:
   - **Correction Rate**: % of OOD false-positives rejected by LLM Critic (target: > 60%)
   - **False Alarm Rate**: % of valid in-distribution designs incorrectly flagged
   - **Detection Latency**: how many refinement rounds before LLM catches the OOD error
   - **Severity breakdown**: correction rate vs data ablation severity (10%/30%/50%)
   - **Qualitative analysis**: LLM's stated reasoning for rejection — does it cite specific physical laws or just "looks too good to be true"?

### 4.6 Hallucination Rate & Failure Case Analysis

#### 4.6.1 Domain-Specific Hallucination Quantification

The LLM's physics knowledge is unevenly distributed across domains. We hypothesize:

**H6**: LLM hallucination rate is lower in domains with abundant training corpus (acoustics, fluid dynamics) and higher in specialized domains with sparse coverage (concrete materials science).

**Measurement protocol**:
- For each domain, collect 20 LLM Critic outputs (randomly sampled across refinement rounds and random seeds)
- Domain expert rates each suggestion:
  - ✓ **Correct**: Physically accurate and helpful for optimization
  - ~ **Plausible**: May be correct but lacks specificity or requires verification
  - ✗ **Hallucination**: Physically incorrect or contradicts established principles
- Report: Hallucination Rate (fraction ✗), Plausible Rate (fraction ~), Correct Rate (fraction ✓)
- **Cross-domain comparison**: Acoustic vs Airfoil vs Concrete hallucination rates
- **Correlation analysis**: Hallucination Rate vs estimated LLM training corpus coverage per domain (approximated via web-search frequency of domain-specific terms)

### 4.7 Extension Study: Topological Changes via GA Fallback

GA fallback (when LLM suggests structural changes) is **disabled in main experiments** to ensure fair comparison. If the system invokes GA mid-loop, comparing against pure GA becomes biased — the improvement may come from GA's global search, not from LLM reasoning.

This is reserved as a separate **Extension Study (Appendix)**:
1. Enable GA fallback only when LLM explicitly requests topological change
2. Compare LLM+GA hybrid vs pure GA on computational budget (same total surrogate calls)
3. Measure: does LLM guidance help GA converge faster than random-initialized GA?
4. This demonstrates the **complementary value** of LLM reasoning for guiding global search — not as a main result, but as a promising direction for future work.

### 4.8 Statistical Protocol

- Each (method, domain) pair: 10 independent runs with different random seeds
- Pairwise comparison: Wilcoxon signed-rank test (α=0.05)
- Effect size: Cohen's d
- All results reported with 95% confidence intervals

---

## 5. Paper Outline

**Title**: *LLM-Guided Inverse Design: A Domain-Agnostic Framework with Structured Physical Reasoning*

**1. Introduction (1 page)**
- Inverse design bottleneck, limitations of open-loop optimization
- Existing LLM-agent work is domain-specific
- We propose LLM as domain-agnostic scientific critic with Text-to-Math bridge

**2. Related Work (0.5 page)**
- Inverse Design Methods (surrogate-based, generative, RL)
- LLMs for Scientific Discovery (Lu et al., AutoMS, SAGA, GenAI vs BO)
- Prompt Engineering for Domain Transfer

**3. Method (1.5 pages)**
- 3.1 Problem Formulation
- 3.2 DomainSpec: structured domain description
- 3.3 LLM Critic: structured feedback generation
- 3.4 **Text-to-Math Bridge**: formal loss function + non-differentiable fallback
- 3.5 Designer + Orchestrator
- 3.6 **LLM-Surrogate Interaction**: the double-blind problem

**4. Experiments (2 pages)**
- 4.1 Setup (domains, methods, metrics, protocol)
- 4.2 Main Results: cross-domain comparison
- 4.3 Ablation: LLM Critic vs Heuristic (H1), component contributions
- 4.4 **Surrogate Error Detection**: correction rate analysis
- 4.5 **Hallucination Rate + Failure Cases**: when LLM gets it wrong
- 4.6 Efficiency + Token Cost Analysis

**5. Discussion (0.5 page)**
- When does LLM reasoning help most?
- Limitations: LLM latency, hallucination in specialized domains
- Broader impact: framework extends to any DomainSpec-describable domain

**6. Conclusion (0.25 page)**

---

## 6. Key Figures

1. **Architecture diagram**: DomainSpec → LLM Critic → Text-to-Math Bridge → Designer
2. **Text-to-Math Bridge visualization**: LLM JSON output → loss function construction
3. **Cross-domain bar chart**: 3 domains × 6 methods
4. **Refinement trace**: performance vs rounds for LLM vs Heuristic Critic
5. **Surrogate error detection**: correction rate by domain and error type
6. **Hallucination breakdown**: pie chart or bar chart by domain
7. **Failure case example**: annotated LLM output showing wrong advice + root cause analysis
8. **Token cost comparison**: Qwen2.5-14B vs GPT-4o-mini, cost vs performance

---

## 7. Timeline (3 months to AAAI deadline)

### Phase 1: Core Implementation (Week 1-2)

| # | Task | Est. |
|---|------|------|
| 1.1 | Implement DomainSpec YAML parser + auto-prompt generator | 2h |
| 1.2 | Implement LLM Critic with structured JSON output + retry | 4h |
| 1.3 | Implement **Text-to-Math Bridge** (JSON → penalty function) | 3h |
| 1.4 | Implement non-differentiable feedback fallback logic | 2h |
| 1.5 | Implement Heuristic Critic (baseline only) | 3h |
| 1.6 | Refactor Designer to read DomainSpec | 2h |
| 1.7 | Unit test: full pipeline on acoustic domain | 2h |

### Phase 2: Surrogate Models + New Domains (Week 2)

| # | Task | Est. |
|---|------|------|
| 2.1 | Download Airfoil dataset, train surrogate (16→2), R² ≥ 0.9 | 4h |
| 2.2 | Download Concrete dataset, train surrogate (8→1), R² ≥ 0.9 | 2h |
| 2.3 | Write DomainSpec YAMLs for airfoil + concrete | 1h |
| 2.4 | End-to-end test on all 3 domains | 2h |

### Phase 3: Full Experiments (Week 3-4)

| # | Task | Est. |
|---|------|------|
| 3.1 | Main benchmark: 3 domains × 6 methods × 10 seeds = 180 runs | 8h |
| 3.2 | LLM backbone ablation (Qwen2.5-14B vs GPT-4o-mini) | 3h |
| 3.3 | **Surrogate error injection experiment** (Section 4.5) | 4h |
| 3.4 | **Hallucination rate annotation** (20 Critic outputs × 3 domains) | 3h |
| 3.5 | **Failure case selection + root cause analysis** | 2h |
| 3.6 | Statistical analysis + all figures | 4h |

### Phase 4: Paper Writing (Week 5-8)

| # | Task | Est. |
|---|------|------|
| 4.1 | Method section (especially Sections 3.4 + 3.6) | 3 days |
| 4.2 | Experiments + results | 3 days |
| 4.3 | Introduction + Related Work | 2 days |
| 4.4 | First complete draft | 2 days |
| 4.5 | Internal review V1 + revisions | 3 days |
| 4.6 | Internal review V2 + revisions | 2 days |
| 4.7 | Final polish + AAAI format + submit | 2 days |

---

## 8. Risk Analysis

| Risk | L | I | Mitigation |
|------|:---:|:---:|------|
| LLM produces unparseable JSON → pipeline breaks | M | H | JSON schema enforcement in prompt; retry up to 3× with stricter prompt; fallback to Heuristic Critic |
| **LLM hallucinates physically wrong feedback** | **H** | **H** | **Hallucination Rate metric quantifies this; failure cases documented; expert review validates critical suggestions** |
| LLM Critic slower than Heuristic → efficiency disadvantage | H | M | Report honestly; local Ollama (Qwen2.5-14B) is free; argue that design quality improvement justifies latency |
| **Surrogate model error misleads LLM Critic** | **M** | **H** | **Correction Rate experiment quantifies this; LLM Critic has independent physics knowledge → may catch surrogate errors that Designer misses** |
| Airfoil surrogate R² < 0.9 | M | M | Try deeper model or switch to NREL 2k dataset (250k samples) |
| **LLM fails on scalar domain (Concrete)** | **M** | **M** | **LLM reasoning adds most value for structured outputs (spectra, vectors); for scalar outputs, document honestly as limitation** |
| AAAI acceptance rate < 15% | H | H | Backup: Engineering Applications of AI (IF 7.5) |

---

## 9. Code Structure

```
agent4science/
├── domain_spec.py              # DomainSpec dataclass + YAML → prompt generator
├── text_to_math.py             # NEW — JSON feedback → differentiable loss terms
├── agents/
│   ├── designer.py             # Algorithmic optimizer (reads DomainSpec)
│   ├── llm_critic.py           # LLM Critic (LangChain + Ollama, structured JSON output)
│   └── orchestrator.py         # Convergence-gated refinement loop
├── baselines/
│   ├── heuristic_critic.py     # Rule-based Critic (ablation baseline only)
│   ├── genetic.py              # NSGA-II
│   ├── bayesian_opt.py         # GP-BO
│   └── random_search.py        # Uniform sampling
├── experiments/
│   ├── surrogate_error.py      # NEW — inject surrogate errors, measure correction rate
│   └── hallucination_eval.py   # NEW — expert annotation tools for LLM feedback
├── models/
│   ├── resnet_model.py         # ResNetDNN (variable input_dim, output_dim)
│   └── train_surrogate.py      # Train + evaluate surrogate for any dataset
├── tools/
│   ├── optimize.py             # Gradient-based optimization (reads DomainSpec)
│   ├── evaluate.py             # Forward prediction
│   ├── validate.py             # Physical constraint checker
│   └── visualize.py            # Plotting
├── utils/
│   ├── metrics.py              # Evaluation metrics
│   └── physics.py              # Physical constraint functions
├── configs/
│   ├── acoustic.yaml           # DomainSpec: acoustic metamaterial
│   ├── airfoil.yaml            # DomainSpec: airfoil aerodynamics
│   └── concrete.yaml           # DomainSpec: concrete strength
└── main.py                     # CLI entry point
```

---

## 10. Immediate Next Steps

1. **[ ]** Create `domain_spec.py` — DomainSpec dataclass, YAML parser, prompt generator
2. **[ ]** Create `llm_critic.py` — LangChain + Ollama, structured JSON schema enforcement, retry logic
3. **[ ]** Create `text_to_math.py` — JSON feedback → penalty function construction + non-differentiable fallback
4. **[ ]** Write 3 DomainSpec YAMLs (acoustic, airfoil, concrete)
5. **[ ]** Refactor Designer to read DomainSpec
6. **[ ]** Run acoustic pipeline with LLM Critic end-to-end
7. **[ ]** Download airfoil + concrete datasets, train surrogates
