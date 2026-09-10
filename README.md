# AGCCC — Adaptive Graph-Centric Coordination and Control

Reference implementation and experiment harness for **“An Adaptive Runtime
Architecture for Agentic AI: The AGCCC Framework for Graph-Centric Coordination
and Control.”**

AGCCC treats multi-agent coordination as a *governed control problem*. Instead of
fixing the coordination topology before execution, it maintains a time-indexed
coordination graph

```
G_t = (V_t, E_t),   V_t ⊆ A,   E_t ⊆ V_t × V_t
```

and permits `G_{t+1} ≠ G_t` when runtime evidence justifies a governed change:

```
G_{t+1} = Γ(G_t, s_t, Π) = argmax [ U(G′,s_t) − λ·κ(G′,G_t) ]   s.t.  π(G′,s_t)=1  ∀π ∈ Π
```

This repository contains the runnable architecture and every experiment behind
the paper's tables and figures.

---

## Quick start

No API key, no network, no Docker. The full experiment is **420 runs in under a
second**.

```bash
python3 -m venv .venv
./.venv/bin/pip install numpy scipy matplotlib pyyaml pytest

./.venv/bin/python -m pytest -q                 # 51 tests
./.venv/bin/python -m agccc.experiment.report   # regenerates results/
```

A single run:

```bash
./.venv/bin/python -m agccc.experiment.runner --arm A2 --env perturbed --seed 7
```

---

## Headline results

420 seeded runs, `n = 30` per arm per environment, simulated agents with
suite-derived confidence.

| Arm | Completion (%) | Recovery (%) | Compliance (%) | Adapt. |
|---|---|---|---|---|
| **Nominal** ||||
| B1 Static sequential | 0.0 | 0.0 | 0.0 | 0 |
| B2 Supervisor | 43.3 | 43.3 | 100.0 | 0 |
| B3 Static graph | 33.3 | 33.3 | 33.3 | 0 |
| A1 AGCCC, governance off | 93.3 | 93.3 | 100.0 | 2.5 |
| **A2 AGCCC** | **93.3** | **93.3** | **100.0** | 2.5 |
| **Perturbed** ||||
| B2 Supervisor | 33.3 | 33.3 | 100.0 | 0 |
| **A2 AGCCC** | **56.7** | **56.7** | **100.0** | 2.0 |

Nominal completion: Mann–Whitney *U*, `p_Holm = 0.0002`, Cliff's δ = +0.50
(large). AGCCC reaches this using **fewer** agent invocations than the
supervisor (15.0 vs 17.6), so adaptation carries no invocation penalty.

### The governance result

At the declared risk weight the governed and ungoverned arms are
**indistinguishable** — both 93.3% completion, zero violations. We do not claim
governance improves the point estimate, because it does not.

Its value is a guarantee, and a guarantee is evaluated on the worst case. Across
**30 plausible scoring configurations** (varying `λ_R`, `λ_C` and the
release-risk constant; 1,800 runs):

| | A1 ungoverned | **A2 governed** |
|---|---|---|
| Compliance, best config | 100.0% | **100.0%** |
| **Compliance, worst config** | **3.3%** | **100.0%** |
| Mean compliance | 77.6% | **100.0%** |
| Configs admitting a violation | **18 / 30** | **0 / 30** |
| Completion, worst config | 3.3% | **86.7%** |

Wilcoxon signed-rank over matched configurations: *p* = 1.88 × 10⁻⁴,
Cliff's δ = +0.600 (large), significant after Holm correction across all seven
hypotheses.

A soft penalty is a claim about the utility function you wrote; a hard
constraint is a claim about the ones you did not. Filtering removes a prohibited
action for *every* scoring function; penalising it removes the action only for
those weightings where the penalty happens to dominate.

### Reported honestly

- The advantage is **not** concentrated under perturbation, contrary to our
  stated expectation: the gap narrows from 50.0 to 23.4 points.
- The perturbed comparison **does not survive Holm correction at n = 30**
  (`p_Holm = 0.238`). At n = 100 the same effect size gives `p = 0.004`. That is
  a power limitation; we report the pre-registered n = 30 as primary rather than
  raising n until significance appears.

---

## What is real and what is simulated

The **microkernel, situation awareness, governance and graph layers are the real
implementation.** Only the *agent bodies* are simulated — they sit behind the
plugin interface in `agents/base.py`, so a live-model backend swaps in without
touching anything else.

Confidence remains **derived, not self-reported**. The two acceptance suites in
`task/suites.py` are real code evaluated against an explicit artefact:

```
c = 0.5·(F_pass / F_total) + 0.5·(S_pass / S_total)
```

What simulation replaces is the artefact *generator*, not the confidence source.
Model self-reports are recorded for calibration comparison and never enter a
control decision.

**The threat to validity, stated plainly.** The defect-generation distribution
is a modelling choice. Three mitigations are enforced in code:

1. **One draw, shared across arms.** Every stochastic outcome is keyed by
   `(role, attempt, defect)` and derived from the seed, so arm *k* at seed *s*
   faces byte-identically what arm *j* at seed *s* faces. Draws are
   *index-stable*: arms that consume randomness at different rates cannot
   desynchronise.
2. **Sensitivity is a deliverable.** Varying the Coder's security-repair
   probability from 0.05 to 0.45 leaves the supervisor at 43.3% and AGCCC
   between 93.3% and 100.0%.
3. **The λ_R sweep above**, which is what turns the governance claim into a
   measurement.

---

## Architecture

```
agccc/
├── kernel/        event_bus · state_store · agent_registry · executor · audit_logger · runtime
├── awareness/     observer (events → state) · confidence (derived) · state
├── governance/    policies (YAML) · actions · evidence · scoring · decision
├── graph/         model (CoordinationGraph) · ops · diff
├── agents/        base · sim/{planner,coder,tester,reviewer,security}
├── strategies/    static_sequential · supervisor · static_graph · adaptive · ablation
├── task/          artefact · defects · suites
└── experiment/    runner · injection · metrics · stats · emit · report
```

The control path is strictly one-directional:

```
Agent → Event → Situation Awareness → Governance → Policy Filter
      → Decision → Graph Mutation → Kernel
```

### Filter, then rank

```
Generate candidates
  → Policy filter        ← inadmissible actions are DELETED here
     → Utility / cost / risk
        → Rank
           → Select
```

The ranker never receives a rejected candidate. Under rank-then-check a
prohibited action is merely outweighed, and anything outweighed can be outweighed
back by a large enough utility. This ordering is the difference, and it is
asserted by a test: a scorer returning `+inf` for a prohibited action still
cannot cause it to be selected.

### No retry limits

There is no `if retry_count == 3` anywhere in the control path. Behaviour changes
because each action carries a Laplace-smoothed posterior

```
P_success(a) = (successes + 1) / (attempts + 2)
```

which decays `0.500 → 0.333 → 0.250 → 0.200` as an action keeps failing, while
untried alternatives hold their prior. The ranking flips on its own. The
supervisor baseline *does* use a fixed escalation constant — that is its defining
weakness and precisely what AGCCC is compared against.

---

## Experimental arms

| Arm | Coordination | Governance | Adaptation |
|---|---|---|---|
| B1 | fixed pipeline | — | none |
| B2 | supervisor, escalates after k=2 | — | agent selection |
| B3 | static graph, bounded retry edge | — | none |
| A1 | adaptive | **filter disabled** | yes |
| A2 | adaptive | enabled | yes |
| ABL2 | observation only — decides, then discards the decision | — | none |
| ABL4 | governed, no structural repair in the action space | enabled | none |

**B2 and B3 both have the Security specialist in their roster.** Withholding it
would let AGCCC win by privileged access rather than by architecture, making the
comparison definitional instead of empirical. The remaining difference is
evidence-driven escalation and structural re-entry: B2's topology is fixed around
its arbiter, and B3 has no `Security → Tester` edge, so its specialist's work is
never re-verified.

---

## Invariants held by tests, not by convention

`./.venv/bin/python -m pytest -q` → **51 tests**.

- The ranker is never called on an action the policy filter rejected.
- No hard-coded retry limit exists in the control path.
- `kernel/runtime.py` names no agent, policy or retry rule in code.
- `awareness/` cannot import `governance/`, `strategies/` or `graph/`;
  `agents/` cannot import `governance/` or `strategies/`; `task/` can import
  none of them.
- Every arm faces an identical world and injection schedule for a given seed.
- Adaptive arms mutate the graph; non-adaptive arms provably never do.
- A1 and A2 differ **only** in the policy set — identical scoring, identical
  action space.
- Every reported confidence lies on the reachable derived grid, so the scripted
  demonstration value cannot leak into a result.

---

## Generated outputs

`python -m agccc.experiment.report` writes into `results/`:

| File | Contents |
|---|---|
| `table3.tex` | main comparison |
| `table4.tex` | ablation |
| `stats.tex` | Mann–Whitney *U*, Cliff's δ, Holm |
| `e1_trace.tex` | per-round candidate scores |
| `sensitivity.tex` | λ_R and agent-competence sweeps |
| `power.tex` | n = 30 vs n = 100 power note |
| `fig2.pdf`, `fig3.pdf` | plotted measurements |
| `metrics.json`, `runs.json`, `reproducibility.json` | full traces |

The paper `\input{}`s these, so no number is transcribed by hand and a stale
figure is impossible.

---

## Reproducibility

Every run records its arm, environment, seed, scoring-configuration hash, world
parameters, suite version, full execution trace, graph diffs, every candidate set
with scores, and the final artefact. A run whose scoring hash differs from the
experiment's registered hash is **discarded, not reinterpreted** — the runner
raises rather than warns.

Environment used for the committed results: Python 3.14.5, macOS on arm64,
420 runs in 0.92 s.

---

## Status and roadmap

This is the Tier 1 harness: statistically powered, free, and offline. Tier 2 —
validation with live model-backed agents at smaller *n* — is future work, and the
plugin boundary is already in place for it.

## Citation

```bibtex
@article{hussain2026agccc,
  title  = {An Adaptive Runtime Architecture for Agentic AI: The AGCCC Framework
            for Graph-Centric Coordination and Control},
  author = {Hussain, Basharat and Islam, Muhammad},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
