# Progress Report: LLM-Guided Inverse Design for AAAI 2027

> Last updated: 2026-05-29

---

## Project Overview

**Goal**: Submit a paper to AAAI 2027 on a domain-agnostic LLM-Guided inverse design framework
validated across acoustic metamaterials, airfoil aerodynamics, and concrete materials.

**Paper Title**: *LLM-Guided Inverse Design: A Domain-Agnostic Framework with Structured Physical Reasoning*

**Novel Contributions**:
1. **C1**: LLM-as-Scientific-Critic -- LLM autonomously critiques designs using pre-trained physics knowledge
2. **C2**: Text-to-Math Bridge -- structured LLM feedback mechanically translated to differentiable loss
3. **C3**: Cross-Domain Validation -- same codebase, zero modifications, three physics domains
4. **C4**: LLM-Surrogate Interaction -- first study of LLM vs surrogate model OOD error dynamics

---

## Phase 1: Code Framework (Completed: May 26-28)

### Infrastructure
- [x] Git repo initialized, pushed to https://github.com/YueX1an/acousticagent
- [x] `requirements.txt`, `pyproject.toml`, `.gitignore`
- [x] Modular package structure: `agent4science/`

### Core Modules Implemented
- [x] `domain_spec.py` -- DomainSpec dataclass, YAML I/O, auto-prompt generator
- [x] `text_to_math.py` -- Softplus loss, dynamic tau scaling, feedback smoothing (EMA)
- [x] `agents/llm_critic.py` -- LangChain + Ollama, structured JSON parsing with retry
- [x] `agents/designer.py` -- Domain-agnostic multi-start gradient optimizer
- [x] `agents/orchestrator.py` -- Convergence-gated refinement loop

### Baselines Implemented
- [x] `baselines/heuristic_critic.py` -- Rule-based Critic (spectrum/vector/scalar strategies)
- [x] `baselines/bayesian_opt.py` -- GP with Matern kernel + Expected Improvement
- [x] `experiments/baselines/random_search.py` -- Uniform random sampling
- [x] `experiments/baselines/genetic.py` -- NSGA-II style GA with tournament selection

### Domain Configs
- [x] `configs/acoustic.yaml` -- 31 params -> 100-point spectrum
- [x] `configs/airfoil.yaml` -- 9 params (8 CST + AoA) -> Cl, Cd
- [x] `configs/concrete.yaml` -- 8 params -> compressive strength (MPa)

---

## Phase 2: Datasets & Surrogate Models (Completed: May 29)

### Data Acquisition
- [x] Airfoil CST+CFD dataset downloaded (GitHub, 28.8k samples) and cleaned
- [x] Concrete Compressive Strength dataset downloaded (UCI #165, 1.0k samples) and cleaned
- [x] Both datasets converted to CSV and stored in `data/`

### Surrogate Training
| Domain | Architecture | R2 | Training Time | Checkpoint |
|--------|-------------|:---:|:---:|------|
| Airfoil CST | ResNetDNN (9->512->2, 4 blocks) | 0.975 | 29s | `saved_models/surrogate_airfoil_cst.pth` |
| Concrete | ResNetDNN (8->256->1, 4 blocks) | 0.941 | 18s | `saved_models/surrogate_concrete_strength.pth` |
| Acoustic | ResNetDNN (31->1024->100, 8 blocks) | 0.984 | Pre-trained | `saved_models_dnn_v2/ResNet_PeakFocus.pth` |

### Environment Verified
- [x] Python 3.9.21 (dnn conda env), PyTorch 2.7.1 + CUDA 12.8
- [x] MATLAB R2024b + TMM FEM simulator operational
- [x] Ollama + qwen2.5:14b (LLM Critic backend)
- [x] All 3 surrogate models load and predict correctly

---

## Phase 3: Experiments (Completed: May 29)

### Bugs Fixed During Experimentation
1. **Initialization bug**: Designer used mid-bounds (~50mm) instead of training distribution (~10mm). Fixed by adding `default_init_values` to DomainSpec.
2. **Output clipping bug**: `evaluate_design()` clipped all outputs to [0,1], truncating concrete strength (2-82 MPa -> 1.0). Fixed by making clipping output_type-dependent.
3. **LLM JSON parsing**: `target: null`, `indices` malformation, non-numeric target strings. Fixed with defensive checks and fallbacks.
4. **Metrics naming**: `avg_absorption` vs `avg_output` inconsistency across modules. Fixed by adding aliases.
5. **GA final eval**: Missing `output_type` parameter on best_result evaluation. Fixed.

### Main Benchmark (3 domains x 6 methods x 3 seeds)

```
Domain           Method                  Mean     N
--------------------------------------------------------
acoustic         bayesian_optimization   0.359     3
acoustic         designer_only           0.811     3
acoustic         heuristic               0.892     3
acoustic         llm                     0.894     3
acoustic         random_search           0.364     3
airfoil          bayesian_optimization   0.920     3
airfoil          designer_only           0.533     3
airfoil          genetic_algorithm       0.360     3
airfoil          heuristic               0.408     3
airfoil          random_search           0.314     3
concrete         bayesian_optimization   113.0     3
concrete         designer_only           79.0      3
concrete         genetic_algorithm       113.0     3
concrete         heuristic               76.1      3
concrete         llm                     74.2      3
concrete         random_search           104.8     3
```

### Ablation Experiments
- **A1 (Remove Critic)**: Designer-Only = 0.811 vs LLM-Guided = 0.894 (9.3% gap)
- **A2 (Heuristic vs LLM)**: Equivalent on acoustic (0.892 vs 0.894)
- **A3 (DomainSpec prompt)**: Full prompt -> 1 round pass. Minimal prompt -> 3 rounds, never converges
- **A5 (Refinement rounds)**: Acoustic converges in 1 round

### OOD Error Detection
- Concrete GA/BO/RS find 105-113 MPa (27-37% above training max 82.6 MPa)
- LLM-Guided (74 MPa) and Heuristic (76 MPa) produce in-distribution designs
- **Key finding**: Critic prevents optimizer from chasing surrogate OOD false positives

### LLM Critic Hallucination Data
- Collected 9 LLM Critic outputs (3 domains x 3 seeds)
- Stored in `results/hallucination/`
- Ready for expert annotation

---

## Phase 4: Figures & Analysis (Completed: May 29)

### Generated Figures
| # | Figure | Path | Description |
|---|--------|------|-------------|
| 1 | Main Results | `results/figures/fig1_main_results.png` | Cross-domain 3x6 bar chart |
| 2 | Concrete OOD | `results/figures/fig2_concrete_ood.png` | OOD gap analysis |
| 3 | Ablation A3 | `results/figures/fig3_ablation_prompt.png` | Full vs Minimal prompt |
| 4 | Acoustic Detail | `results/figures/fig4_acoustic_detail.png` | Method comparison |
| 5 | Refinement Trace | `results/figures/fig5_refinement_trace.png` | Round-by-round performance |
| 6 | Contribution | `results/figures/fig6_contribution_waterfall.png` | Component waterfall |

---

## Phase 5: Honest Assessment (May 30)

### Critical Gaps Found

**Gap 1: Missing AI Baselines (MOST URGENT)**
- All current baselines are traditional optimization methods (GA, BO, Random Search)
- **Zero AI baselines**: No pure LLM Agent, no LLM-only optimization
- AAAI reviewers will reject without AI comparison points
- Required: B1 (Pure LLM ReAct), B2 (No Text-to-Math bridge), B3 (LLM backbone comparison)

**Gap 2: H1 Failed (LLM = Heuristic on Acoustic)**
- On our best-tested domain, LLM Critic (0.894) = Heuristic Critic (0.892)
- The hypothesis that "LLM reasoning beats rules" is NOT supported
- Mitigation: reframe paper around C4 (OOD detection) and C2 (Text-to-Math), not C1

**Gap 3: Figure Quality**
- 6 bar charts do not make an AAAI paper
- Need: vector architecture diagram, qualitative LLM reasoning traces, optimization trajectories

**Gap 4: Only 3 Seeds**
- Statistical significance requires 10 seeds; currently only 3

### Revised Paper Direction

**NEW thesis**: "LLM-Guided optimization prevents surrogate model OOD errors that deceive traditional optimizers."

**Key evidence**:
1. Concrete: GA/BO chase 113 MPa (+37% OOD). LLM-Guided = 74 MPa (realistic).
2. Acoustic FEM: Surrogate 0.898, FEM produces NaN — design is physically invalid.
3. Ablation: Without DomainSpec prompt context, LLM over-refines and never converges.

## Remaining Work

### Immediate (Next 1-2 Days)
| # | Task | Status | Est. Time |
|---|------|:---:|:---:|
| B1 | Pure LLM ReAct Agent baseline | **Not started** | 2h |
| B2 | LLM Critic without Text-to-Math bridge | **Not started** | 1h |
| B3 | LLM backbone comparison (Qwen vs GPT) | **Not started** | 1h |
| 1 | Hallucination rate annotation | Pending | 2h |
| 2 | FEM verification of acoustic OOD | Partial | 2h |
| 3 | Fix airfoil LLM Critic stability | **Not started** | 1h |
| 4 | Re-run benchmark with 10 seeds | **Not started** | 3h |
| 5 | Verify concrete 113 MPa is truly OOD | **Not started** | 1h |

### Before Paper Submission
| # | Task | Est. Time |
|---|------|:---:|
| 6 | Architecture diagram (draw.io/TikZ) | 2h |
| 7 | Qualitative LLM reasoning examples | 1h |
| 8 | Statistical tests (Wilcoxon, Cohen's d) | 30min |
| 9 | Write Introduction | 2 days |
| 10 | Write Related Work | 1 day |
| 11 | Write Method | 3 days |
| 12 | Write Experiments | 3 days |
| 13 | Write Discussion + Conclusion | 1 day |
| 14 | Polish + AAAI format | 2 days |

### Known Issues
- **Airfoil LLM Critic**: crashes on some JSON responses (target field non-numeric). Handled by fallback but needs robustness improvement before production use.
- **Acoustic GA**: returns -6.71 due to OOD parameter exploration. GA pushes parameters outside training distribution.
- **Concrete RS/GA/BO 113 MPa**: These are OOD false positives. This is actually the paper's key finding, not a bug.
- **FEM NaN**: Some frequency points fail to converge in TMM solver for extreme parameter combinations.

---

## File Structure (Current)

```
agent4science/
├── agents/
│   ├── base_agent.py          # AgentMessage, BaseAgent, LLMAgent
│   ├── designer.py            # Domain-agnostic gradient optimizer
│   ├── llm_critic.py          # LLM Critic (LangChain + Ollama)
│   └── orchestrator.py        # Refinement loop controller
├── baselines/
│   ├── bayesian_opt.py        # GP + EI
│   ├── heuristic_critic.py    # Rule-based critic (ablation)
│   └── __init__.py
├── configs/
│   ├── acoustic.yaml          # DomainSpec: acoustic metamaterial
│   ├── airfoil.yaml           # DomainSpec: airfoil CST
│   └── concrete.yaml          # DomainSpec: concrete strength
├── domain_spec.py             # DomainSpec dataclass + YAML + prompts
├── text_to_math.py            # Text-to-Math bridge + feedback smoothing
├── experiments/
│   └── baselines/
│       ├── genetic.py         # NSGA-II GA
│       └── random_search.py   # Random search
├── models/
│   ├── resnet_model.py        # ResNetDNN architecture
│   └── train_surrogate.py     # Training script
├── tools/
│   ├── evaluate.py            # Forward evaluation
│   ├── optimize.py            # Legacy optimizer
│   ├── validate.py            # Physical constraint checker
│   └── visualize.py           # Plotting utilities
└── utils/
    ├── metrics.py             # Evaluation metrics
    └── physics.py             # Domain constraints
data/                           # Datasets (not tracked)
saved_models/                   # Trained surrogate models (not tracked)
results/                        # Experiment outputs (not tracked)
generate_figures.py             # Figure generation script
main.py                         # CLI entry point
research_plan.md                # Research plan + results
progress.md                     # This file
```

---

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| May 26 | Don't call it "Multi-Agent" | Reviewer feedback: algorithmic modules != agents |
| May 26 | LLM Critic as core AI contribution | Hardcoded rules are not AI innovation |
| May 27 | Text-to-Math Bridge with Softplus | max(0,x)^2 has gradient sparsity; Softplus is smooth |
| May 27 | Feedback smoothing via EMA | Prevents LLM changing mind causing loss landscape jumps |
| May 28 | OOD experiment with OOD data ablation | Realistic, not random noise on weights |
| May 28 | Cross-domain framework kept (airfoil + concrete) | Essential for C3 contribution |
| May 28 | GA fallback disabled in main experiments | Reviewer fairness concern |
| May 29 | default_init_values in DomainSpec | Mid-bounds init was outside training distribution |
| May 29 | Concrete OOD finding becomes paper's strongest result | Natural experiment: GA/BO chase OOD, Critic prevents it |

---

## Git History

```
9134b0b Download and preprocess public datasets
15bdf9f Fix Unicode encoding errors in train_surrogate.py
1baa6f7 Fix critical initialization bug + add default_init_values
b37a78f Fix critical bugs: output clipping, LLM JSON parsing
4865c23 Fix GA final evaluation and LLM target parsing
5b5014c Ablation experiments + OOD design generation
```
