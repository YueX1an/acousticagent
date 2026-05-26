# Research Plan: Multi-Agent Scientific Inverse Design Framework for Acoustic Metamaterials

## Target Venue: AAAI 2027 (deadline ~August 2026)

---

## 1. Current State Analysis

The existing system is a **single-agent ReAct pipeline**:

| Component | Technology | Role |
|-----------|-----------|------|
| LLM Orchestrator | Ollama + qwen2.5:14b | Parses user intent, invokes tools |
| Neural Surrogate | ResNetDNN (8-block, 1024-dim) | Predicts absorption spectrum from 31 geometric params |
| Inverse Solver | Gradient descent (Adam) | Optimizes input params to maximize absorption |
| Constraint Handler | Heuristic projection | Clamps values, locks thickness |

**What works**: Proof-of-concept that an LLM agent can drive inverse design via tool invocation.  
**What's missing for AAAI**: Scientific novelty, rigorous evaluation, multi-agent coordination, and theoretical grounding.

---

## 2. Research Gap & Novelty

Single-agent inverse design is well-explored. The key open question is:

> **Can a *multi-agent system* with specialized roles, debate, and self-verification outperform monolithic methods in scientific inverse design tasks?**

### Novelty Claims (3 pillars)

1. **Role-Specialized Multi-Agent Architecture**: Instead of one agent doing everything, we propose a society of agents — *Theorist, Designer, Critic, Synthesizer* — each with distinct prompts, tools, and knowledge. This mirrors how real scientific teams operate.

2. **LLM-Guided Constrained Optimization**: The agents don't just invoke a black-box optimizer; they *reason about physical constraints*, propose diverse initialization strategies, interpret spectral features, and negotiate trade-offs (broadband vs. peak, thickness vs. performance).

3. **Self-Verification via Adversarial Debate**: A Critic agent challenges the Designer's output, forcing iterative refinement. This produces designs that are not just numerically optimal but *physically plausible*.

---

## 3. Proposed Multi-Agent Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Meta-Agent)                   │
│  Routes tasks, manages conversation turns, declares convergence│
└──────────────────────────────────────────────────────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────────┐
│   THEORIST    │  │   DESIGNER    │  │     CRITIC       │
│               │  │               │  │                  │
│ • Physical    │  │ • Gradient   │  │ • Spectral       │
│   reasoning   │  │   optimization│  │   analysis       │
│ • Constraint  │  │ • Multi-start │  │ • Feasibility    │
│   formulation │  │   strategies  │  │   check          │
│ • Prior       │  │ • Pareto front│  │ • Comparison w/  │
│   knowledge   │  │   exploration │  │   literature     │
└───────────────┘  └──────────────┘  └──────────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                ┌──────────────────┐
                │   SYNTHESIZER    │
                │ • Report writing │
                │ • Visualization  │
                │ • Design export  │
                └──────────────────┘
```

### Agent Roles in Detail

| Agent | Tools | Output |
|-------|-------|--------|
| **Theorist** | Physical equations, material database | Constraint specification, frequency target selection |
| **Designer** | `optimize_metamaterial_design` (multi-start), `evaluate_design_performance` | Candidate designs with Pareto trade-off curves |
| **Critic** | `validate_physical_constraints`, `check_causality`, `compare_spectrum` | Pass/fail verdict, improvement suggestions |
| **Synthesizer** | Plotting, table generation, LaTeX export | Final report with design blueprint |
| **Orchestrator** | Task routing, turn management | Converged design or request for more iterations |

---

## 4. Research Plan & Timeline (3 months to deadline)

### Phase 1: Framework Engineering (Weeks 1-3)

**Goal**: Build the multi-agent infrastructure.

- [ ] **Task 1.1**: Refactor `agent_perfectabsorbtion.py` into a modular multi-agent package
  - Extract surrogate model inference as a standalone service
  - Create base `Agent` class with role-specific prompt templates
  - Implement agent-to-agent message passing protocol

- [ ] **Task 1.2**: Implement each specialized agent
  - `TheoristAgent`: Prompt with acoustic theory (Helmholtz resonance, transfer matrix method, viscothermal losses)
  - `DesignerAgent`: Multi-start gradient descent, supports different initializations (random, heuristic, theory-guided)
  - `CriticAgent`: Validates absorption spectra against physical constraints (causality, passivity, reciprocity)
  - `SynthesizerAgent`: Generates structured design reports

- [ ] **Task 1.3**: Implement Orchestrator
  - Finite state machine for agent conversation rounds
  - Convergence criterion (performance delta < threshold for K rounds)
  - Budget-aware early stopping

### Phase 2: Core Algorithm Development (Weeks 3-5)

**Goal**: Develop the key algorithmic contributions.

- [ ] **Task 2.1**: Theory-guided initialization
  - Theorist proposes initial parameter guesses based on target frequency band
  - Designer uses these as warm starts (vs. random baseline)
  - Hypothesis: theory-guided init converges faster and to better optima

- [ ] **Task 2.2**: Adversarial refinement loop
  - Designer proposes design → Critic identifies spectral weaknesses → Designer re-optimizes with weighted loss
  - Measure: improvement in worst-band absorption (currently ignored by mean/peak optimization)

- [ ] **Task 2.3**: Multi-objective Pareto exploration
  - Agents negotiate trade-offs: avg_absorption vs. peak_absorption vs. thickness
  - Designer runs multi-objective optimization with different weightings
  - Synthesizer presents Pareto frontier for human selection

- [ ] **Task 2.4**: Uncertainty-aware design
  - Surrogate model epistemic uncertainty quantification (MC Dropout or ensemble)
  - Critic flags designs in high-uncertainty regions
  - Triggers active learning: request new FEM simulation data

### Phase 3: Experiments (Weeks 5-8)

**Goal**: Rigorous empirical evaluation.

- [ ] **Task 3.1**: Baselines
  - Single-agent (current system)
  - Traditional optimization without LLM: Genetic Algorithm (NSGA-II), Bayesian Optimization (Gaussian Process), Random Search
  - Ablation: remove individual agents, measure performance degradation

- [ ] **Task 3.2**: Evaluation Metrics
  - **Performance**: Avg absorption, peak absorption, bandwidth (α > 0.8), lowest frequency with α > 0.5
  - **Efficiency**: Optimization steps to convergence, total wall-clock time, LLM token cost
  - **Robustness**: Performance variance across 50 random seeds, sensitivity to parameter perturbation
  - **Physical Validity**: Causal spectrum check, Kramers-Kronig consistency

- [ ] **Task 3.3**: Task Suite
  - Task 1: Broadband absorber (100mm, 0-2000Hz) — standard benchmark
  - Task 2: Low-frequency absorber (150mm, 0-500Hz) — challenging regime
  - Task 3: Multi-band absorber (120mm, target 500Hz + 1500Hz) — multi-objective
  - Task 4: Ultra-thin absorber (50mm, 0-2000Hz) — extreme constraint

- [ ] **Task 3.4**: Statistical Analysis
  - Wilcoxon signed-rank test for pairwise comparisons
  - Effect size (Cohen's d) for practical significance
  - Confidence intervals on all metrics

### Phase 4: Paper Writing (Weeks 8-11)

**Goal**: Produce AAAI-ready manuscript.

- [ ] **Task 4.1**: Paper outline
  1. **Introduction**: Inverse design problem, limitations of single-agent / traditional optimization
  2. **Related Work**: AI for science, LLM agents, multi-agent systems, metamaterial inverse design
  3. **Method**: Multi-agent architecture, agent roles, optimization algorithms, verification protocol
  4. **Experiments**: Setup, baselines, results, ablation, analysis
  5. **Discussion**: When does multi-agent help? Failure modes. Generalization to other domains.
  6. **Conclusion**

- [ ] **Task 4.2**: Figures
  - Architecture diagram
  - Example conversation trace between agents
  - Pareto frontier plots (multi-agent vs. baselines)
  - Ablation bar charts
  - Case study: design evolution over debate rounds

- [ ] **Task 4.3**: Writing & revisions
  - Week 9: First draft
  - Week 10: Internal review + revisions
  - Week 11: Polish, proofread, format for AAAI style

---

## 5. Key Hypotheses to Test

| # | Hypothesis | Predicted Outcome |
|---|-----------|-------------------|
| H1 | Multi-agent debate produces designs with higher worst-band absorption than single-agent | Critic catches spectral dips that single-agent misses |
| H2 | Theory-guided initialization reduces optimization steps by ≥30% | Theorist's physical priors provide better warm starts |
| H3 | Multi-agent system achieves better Pareto frontiers than NSGA-II | LLM reasoning navigates trade-offs more effectively than evolutionary search |
| H4 | Uncertainty-aware design reduces the number of FEM simulations needed in active learning | Critic avoids querying the surrogate in unreliable regions |

---

## 6. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| LLM API cost too high for large experiments | Medium | Use local Ollama (qwen2.5:14b is free); cache repeated queries; batch evaluations |
| Multi-agent debate doesn't converge | Medium | Hard limit on debate rounds; fallback to best-so-far design |
| Surrogate model accuracy insufficient for critic | High | Retrain with active-learning-augmented dataset; ensemble of 3 models |
| Baselines outperform multi-agent | High | If this happens, the paper becomes a "negative result" — still publishable if well-analyzed, but pivot to explaining *why* LLM agents fail at this task |
| AAAI reviewers unfamiliar with metamaterials | Low | Frame as "AI method applied to scientific domain" not "acoustics paper"; emphasize generalizability |

---

## 7. Code Refactoring Roadmap

Current single-file script → modular package:

```
agent4science/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Abstract agent class
│   ├── theorist_agent.py      # Physics reasoning agent
│   ├── designer_agent.py      # Optimization agent
│   ├── critic_agent.py        # Verification agent
│   ├── synthesizer_agent.py   # Reporting agent
│   └── orchestrator.py        # Multi-agent coordinator
├── models/
│   ├── __init__.py
│   └── resnet_model.py        # Surrogate model (existing, cleaned up)
├── tools/
│   ├── __init__.py
│   ├── optimize.py            # Gradient-based inverse design
│   ├── evaluate.py            # Forward prediction
│   ├── validate.py            # Physical constraint checks
│   └── visualize.py           # Plotting utilities
├── experiments/
│   ├── run_benchmark.py       # Run full benchmark suite
│   ├── baselines/
│   │   ├── genetic.py         # NSGA-II baseline
│   │   ├── bayesian_opt.py    # BO baseline
│   │   └── random_search.py   # Random baseline
│   └── configs/               # YAML experiment configs
├── utils/
│   ├── __init__.py
│   ├── physics.py             # Physical equations, constraints
│   └── metrics.py             # Evaluation metrics
├── main.py                    # Entry point
├── requirements.txt
└── pyproject.toml
```

---

## 8. Immediate Next Steps (This Week)

1. Set up git repo and install dependencies formally (`requirements.txt`)
2. Refactor `agent_perfectabsorbtion.py` into the modular structure above
3. Implement `Orchestrator` with 2-agent prototype (Designer + Critic, no Theorist yet)
4. Run baseline single-agent benchmark on all 4 tasks
5. Schedule weekly meeting to track progress against timeline
