---
title: Recommendation Policy Triage
emoji: "🎯"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: OpenEnv benchmark for trust-aware session recommendation under uncertainty.
---

# Recommendation Policy Triage

Recommendation Policy Triage is an OpenEnv-compatible benchmark for session-level recommendation control under uncertainty.

It evaluates not just what an agent recommends next, but how well it manages user satisfaction, trust, calibration, diversity, and long-horizon session health across a full 20-step session.

The environment frames recommendation as a sequential partially observable control problem (POMDP) rather than a one-step ranking task. At each step, the agent must decide:

what to recommend
when to explore
how much confidence to assign to that decision

while operating under visible system constraints such as cost, risk, latency, and budget, and hidden user dynamics such as intent drift, fatigue accumulation, trust erosion, and latent regime shifts.

In other words, the benchmark is designed to test whether a policy can make recommendations that are not only locally relevant, but also robust, calibrated, and sustainable over time — the core challenge faced by real recommender systems operating over full user sessions.



**Team:** Neonatal  
**Authors:** Dibyajyoti001, ankannayek  
**Hugging Face Space:** [dibyajyoti001-recommender-triage-openenv](https://dibyajyoti001-recommender-triage-openenv.hf.space)

---

## Problem Statement & Why This Benchmark Matters

Most recommendation benchmarks focus only on short-horizon objectives:

- next-click prediction
- next-item prediction
- one-step ranking
- static candidate scoring

These tasks are useful for offline evaluation, but they do **not** capture the actual control problem faced by a live recommender policy operating over a full user session.

In real systems:
- repeating similar content creates fatigue
- historical memory can become stale
- live user intent drifts over time
- feedback is noisy and delayed
- overconfidence damages long-term trust
- operating constraints (cost, risk, latency) matter

A locally strong recommendation can still be globally poor if it leads to session collapse.

**Recommendation Policy Triage** reframes recommendation as a **trust-aware, long-horizon control problem** under partial observability and latent regime shifts. It directly evaluates policies on user experience, trust preservation, and session health - the metrics that actually matter in production recommenders.

---

## Benchmark At A Glance

- **5 tasks** with clear easy -> medium -> hard progression
- **Structured action space** (item + exploration flag + confidence)
- **Partially observable hidden state** (memory, live intent, fatigue, patience, trust, regime, volatility)
- **Analytically generated 6-slot candidate pool** that evolves with hidden state
- **Visible operating constraints** (cost, risk, latency, budgets)
- **Regime-conditioned latent volatility** (stable / drifting / fatigued / distressed + stochastic volatility)
- **Light domain randomization** at reset (each episode simulates a slightly different user)
- **Trajectory-level non-compensatory grading**
- **Counterfactual audit** on fragile turns (Tasks 4 & 5)
- **Live visualization** endpoint (`/visualize`)

---

## What The Environment Evaluates

The benchmark tests whether a policy can:

- exploit stable preference when memory is reliable
- avoid excessive repetition as fatigue accumulates
- detect conflict between historical memory and live intent
- adapt after intent drift
- use exploration deliberately rather than randomly
- calibrate confidence under uncertainty
- act under visible budget, risk, and latency constraints
- preserve trust and session health over the full horizon instead of optimizing only the next step

---

## What The Agent Controls & Observes

**Action**

```python
class Action(BaseModel):
    recommended_item_id: int
    exploration_flag: bool
    confidence_score: float   # 0.0 - 1.0
```

**Observation** (partial signals only - full hidden state is never exposed)

The agent sees task metadata, memory summary, recent interactions, 6 candidate items with visible cost/risk/latency, repetition pressure, feedback volatility, trust signal, engagement signal, and resource budgets.

---

## Hidden User State

Each episode maintains a rich hidden state:

| Variable | Description |
|----------------|-------------|
| `m` | Long-term memory preference (10-topic basis) |
| `z` | Live session intent (10-topic basis) |
| `F_topic` | Fast fatigue accumulator |
| `H_topic` | Slow repetition-pressure accumulator |
| `p` | Engagement / patience reserve |
| `chi` | Memory reliability |
| `trust` | User-system relationship state |
| `regime` | Latent mode: 0=stable, 1=drifting, 2=fatigued, 3=distressed |
| `latent_vol` | Latent volatility scalar [0,1] that modulates drift, noise, fatigue, trust sensitivity |

**New:** `regime` and `latent_vol` create persistent, realistic shifts. Transitions are sticky (hysteresis) and depend on trust, patience, repetition pressure, and alignment.

**New:** Light domain randomization at `reset()` perturbs user-dynamic parameters (fatigue sensitivity, trust gain/loss, volatility base, regime stickiness, etc.) so each episode simulates a **slightly different real user** while keeping task identity intact.

---

## Fixed Topic Basis & Mixed-Topic Items

All computations use a fixed global 10-topic basis:

| Index | Topic |
|-------|-------------|
| 0 | Romance |
| 1 | Comedy |
| 2 | Drama |
| 3 | Thriller |
| 4 | Action |
| 5 | Documentary |
| 6 | Crime |
| 7 | SciFi |
| 8 | News |
| 9 | Lifestyle |

Items are sparse probability distributions over this basis (not one-hot). This enables partial relevance, semantic repetition detection, and smooth fatigue.

---

## Candidate Pool Design

At every turn the environment generates exactly **6 analytically constructed candidates**:

| Slot | Purpose |
|---------------------|---------|
| live_best_fresh | Aligned with current live intent `z` |
| memory_best_fresh | Aligned with long-term memory `m` |
| balanced_bridge | Blend of `z` and `m` |
| fatigue_trap | Tempting but repetition-heavy |
| exploration_option | Diversification candidate |
| conflict_option | Opposing-signal candidate |

The relative attractiveness and damage of each slot is **regime-conditioned** (e.g. fatigue_trap becomes more seductive in fatigued/distressed regimes). The pool is never sampled from a static catalog - it evolves with hidden state.

---

## Task Suite (Detailed)

**task_1 - Stable Preference Exploitation** (Easy)  
Memory is reliable and live intent mostly agrees. Policy should exploit efficiently.

**task_2 - Repetition Fatigue Control** (Medium)  
Strong narrow preference but repetition quickly reduces satisfaction. Must balance relevance and diversity.

**task_3 - Memory vs Live Signal Conflict** (Hard)  
Historical memory and live behavior diverge. Must detect drift and recover.

**task_4 - Echo Chamber Collapse** (Hard)  
Early recommendations look excellent but create slow saturation. Must diversify **before** collapse becomes obvious.

**task_5 - Noisy Signal Calibration** (Hard)  
Feedback is noisy. Must lower confidence under volatility while preserving relevance and trust.

---

## Core Dynamics (All Formulas)

**Relevance**

```math
R_t(i) = \frac{\alpha z_t^\top x_i + (1-\alpha) m^\top x_i + \eta_q q_i}{1 + \eta_q}
```

**Fatigue & Repetition Pressure**

```math
F_{t+1} = \lambda_F F_t + \delta_F x_{a_t}
H_{t+1} = \text{normalize}(\lambda_H H_t + (1-\lambda_H) x_{a_t})
```

**Satisfaction Proxy**

```math
y_t = \sigma(\zeta_1 R_t(a_t) - \zeta_2 C_t(a_t) - \zeta_3 V_t(a_t) + \zeta_4 \cdot \text{trust_term})
```

**Patience / Engagement**

```math
p_{t+1} = \text{clip}(p_t + \beta (2y_t - 1), 0, 1)
```

**Memory Confidence**

```math
\chi_{t+1} = \text{clip}(\chi_t + \alpha_\chi (z_t^\top m - \theta_\chi), 0, 1)
```

**Regime-Conditioned Volatility (New)**
The simulator now maintains a latent regime and volatility scalar that modulate all dynamics (intent drift rate, feedback noise, fatigue growth, trust loss, candidate attractiveness). Transitions are persistent with hysteresis.

**Step Reward**

```math
r_t = w_r R_t - w_f C_t - w_n V_t - w_u U_t - w_\text{conf} P_t
```

(with clipping to [-1, 1]).

---

## Population Heterogeneity (Real-World Utility Boost)

At every `reset()`, light domain randomization is applied to user-dynamic parameters (fatigue sensitivity, trust gain/loss, volatility base, regime stickiness, memory reliability, resource envelopes). This creates a distribution of slightly different users inside each task without changing task identity or public schemas. Policies must therefore generalize across plausible real-user variation - directly improving real-world utility.

---

## Final Grader & Counterfactual Audit

The final score is **trajectory-level** and **non-compensatory** on Tasks 4 & 5. Tasks 4 & 5 also include a regime-aware counterfactual audit on up to 2 fragile turns.

---

## Summary

Recommendation Policy Triage is a compact yet powerful testbed for **trust-aware, long-horizon recommendation control under uncertainty**. It combines:

- sequential POMDP structure
- regime-conditioned latent volatility
- light domain randomization for population realism
- trajectory-level non-compensatory grading
- counterfactual audit on fragile turns

This makes the benchmark both harder and far more realistic than standard ranking benchmarks.

You can now train policies that generalize across different users while preserving session health, trust, and calibration - skills that matter in real recommender systems.

---

## References
- Dudik et al. Doubly Robust Policy Evaluation and Learning
- McInerney et al. Counterfactual Evaluation of Slate Recommendations
- Ng et al. Policy Invariance under Reward Transformations


