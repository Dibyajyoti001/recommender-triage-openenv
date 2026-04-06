---
title: Recommendation Policy Triage
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: OpenEnv benchmark for session-level recommendation under memory uncertainty, repetition fatigue, and intent drift.
---

# Recommendation Policy Triage

Recommendation Policy Triage is an OpenEnv-compatible environment for evaluating **session-level recommendation policy** under **memory uncertainty**, **repetition fatigue**, and **live intent drift**.

It models recommendation as a sequential control problem rather than a static ranking problem. An agent interacts with a simulated user across a 20-step session, receives only partial observations, and must decide what to recommend, when to explore, and how much confidence to assign to each action.

**Team:** Neonatal  
**Authors:** Dibyajyoti001, ankannayek

**Hugging Face Space:** `soon`  
**OpenAPI Docs:** `soon`

---

## Overview

Most recommendation benchmarks focus on one-step prediction: next-item prediction, click prediction, retrieval, or ranking accuracy. Those tasks are useful, but they do not capture the actual control problem faced by a recommender system operating over a full session.

A recommendation that looks good locally can still be poor globally. Repeating similar content can create fatigue. Historical memory can become stale. Live user intent can shift during the session. Exploration can help or hurt depending on context. A policy can also remain overconfident even while user satisfaction is dropping.

Recommendation Policy Triage is built to evaluate that setting directly. The environment maintains a hidden user state over time and scores both the immediate quality of each step and the quality of the full trajectory.

---

## What the Environment Evaluates

The benchmark tests whether an agent can:

- exploit stable preference when memory is reliable
- avoid excessive repetition as fatigue accumulates
- detect conflict between historical memory and live session behavior
- adapt after intent drift
- use exploration deliberately rather than randomly
- calibrate confidence instead of making persistently overconfident decisions

This makes the environment closer to a **recommendation-policy benchmark** than to a ranking-only benchmark.

---

## Task Suite

The environment contains three tasks with a clear easy / medium / hard progression.

### task_1 — Stable Preference Exploitation

Historical memory is reliable and live session intent mostly agrees with it. The correct policy is to exploit stable preference efficiently and avoid unnecessary exploration.

**Grader weights:** satisfaction 0.60, diversity 0.20, adaptation 0.00, memory_use 0.20

### task_2 — Repetition Fatigue Control

The user prefers a narrow topic region, but repeated similar recommendations reduce long-term satisfaction. The correct policy is to balance relevance against diversity and fatigue.

**Grader weights:** satisfaction 0.40, diversity 0.40, adaptation 0.00, memory_use 0.20

### task_3 — Memory vs Live Signal Conflict

Historical memory and live session behavior diverge. The correct policy is to detect drift, reduce reliance on stale memory, and recover quickly.

**Grader weights:** satisfaction 0.30, diversity 0.20, adaptation 0.30, memory_use 0.20

---

## Baseline Results

Measured with a root-level baseline `inference.py` over 20 steps per task.

| task_id | score |
|---|---:|
| task_1 | 0.738 |
| task_2 | 0.583 |
| task_3 | 0.468 |
| average | 0.596 |

These results reflect the intended task ordering: stable exploitation is easiest, repetition control is harder, and memory/live conflict is hardest.

---

## Hidden User State

Each episode maintains six hidden state components. The agent does not observe these directly.

| variable | type | description |
|---|---|---|
| `m` | `List[float]`, len 10 | Long-term memory preference distribution |
| `z` | `List[float]`, len 10 | Live session intent distribution |
| `H_topic` | `List[float]`, len 10 | Slow EMA of recommendation history (repetition pressure) |
| `F_topic` | `List[float]`, len 10 | Fast EMA fatigue accumulator |
| `p` | `float` in [0, 1] | Patience reserve |
| `chi` | `float` in [0, 1] | Memory confidence |

### Fixed Topic Basis

All state vectors, item vectors, and grader computations use a single global 10-topic basis with a fixed index order:


index  0        1       2      3         4        5             6       7      8      9
       Romance  Comedy  Drama  Thriller  Action   Documentary  Crime   SciFi  News   Lifestyle

This ordering is global and never changes within or across episodes.


---

Mixed-Topic Item Representation

Items are represented as sparse probability distributions over the topic basis rather than as one-hot categories.

Example:

[0.65, 0.30, 0.05, 0, 0, 0, 0, 0, 0, 0]

This corresponds to an item that is primarily Romance with Comedy as a secondary component.

This representation allows:

partial relevance rather than exact-category matching

semantic repetition detection across similar items

smoother fatigue accumulation

more realistic conflict between memory and live session signals



---

Candidate Pool

At each turn, the environment generates a structured six-slot candidate pool from the current hidden state.

slot_type	description

live_best_fresh	aligned with current live intent z
memory_best_fresh	aligned with long-term memory m
balanced_bridge	blend of z and m
fatigue_trap	superficially attractive but fatigue-heavy
exploration_option	diversification / uncertainty-handling candidate
conflict_option	opposing signal candidate


The pool is generated analytically from hidden state rather than sampled from a static item catalog, so candidate composition evolves as fatigue builds and intent drifts.


---

Core Dynamics

Relevance

For candidate item i with topic vector x_i ∈ Δ^9 and quality score q_i ∈ [0, 1]:

R_t(i) = ( α · z_t^T x_i + (1−α) · m^T x_i + η_q · q_i ) / (1 + η_q)

This combines live intent alignment, memory alignment, and item quality.

Fatigue and Repetition Pressure

Fatigue and repetition pressure are maintained as separate exponential moving averages:

F_{t+1} = λ_F · F_t + δ_F · x_{a_t}
H_{t+1} = normalize( λ_H · H_t + (1 − λ_H) · x_{a_t} )

Derived quantities:

C_t(i) = F_t^T x_i
ρ_t(i) = H_t^T x_i
V_t(i) = max(0, ρ_t(i) − ν)

Where:

C_t(i) is fatigue cost

ρ_t(i) is repetition pressure

V_t(i) is novelty violation


Fatigue is bounded. Under repeated recommendation of a fixed topic distribution x̄, the accumulator converges to δ_F / (1 − λ_F) · x̄.

Satisfaction Proxy

y_t = σ( ζ₁·R_t(a_t) − ζ₂·C_t(a_t) − ζ₃·V_t(a_t) )

This is the main intermediate signal used by the environment to update patience and evaluate local session quality.

Patience

p_{t+1} = clip( p_t + β·(2y_t − 1), 0, 1 )

If y_t > 0.5, patience grows. If y_t < 0.5, patience erodes.

Memory Confidence

chi_{t+1} = clip( chi_t + α_χ·(z_t^T m − θ_χ), 0, 1 )

When live intent aligns with memory, confidence increases. When they diverge, confidence erodes.

Intent Dynamics

if p_t > τ:    z_{t+1} = normalize( (1−μ) · z_t + μ · φ_t )
else:          z_{t+1} = normalize( (1−κ) · z_t + κ · ψ_t )

φ_t is a continuation target blending current intent and memory.
ψ_t is a conflict target of the form (1−s)·uniform + s·anti_m, where s is task-specific conflict strength.

Step Reward

r_t = w_r·R_t − w_f·C_t − w_n·V_t − w_u·U_t − w_c·P_t

Where:

U_t is an unnecessary-exploration penalty

P_t = max(0, confidence_score − y_t) is an overconfidence penalty


Step reward is clipped to [-1, 1].


---

Final Grader

The terminal score is computed by a trajectory-level grader rather than by summing step rewards:

G(τ) = ω₁·S(τ) + ω₂·D(τ) + ω₃·A(τ) + ω₄·M(τ)

component	description

S(τ)	mean satisfaction proxy over the episode
D(τ)	entropy-based diversity score
A(τ)	adaptation quality
M(τ)	memory-use quality


All components are bounded in [0, 1], and task-specific weights sum to 1.0.

Diversity

Diversity is based on the entropy of the chosen-category distribution:

H(P) = - Σ p_k log p_k

Adaptation

Adaptation is evaluated with exponential decay based on recovery delay:

A(τ) = exp(−λ_A · (t_recovery − t_drift))

Fast recovery scores better than slow recovery.


---

Observation and Action

class Observation(BaseModel):
    task_id: str
    task_name: str
    turn_id: int
    max_turns: int
    memory_summary: MemorySummary
    recent_interactions: List[RecentInteraction]
    candidate_items: List[CandidateItem]
    repetition_counts: List[int]
    repetition_pressure_bucket: str
    memory_confidence_bucket: str
    memory_confidence: float
    session_feedback_signal: float
    done_hint: str

class Action(BaseModel):
    recommended_item_id: int
    exploration_flag: bool
    confidence_score: float

The confidence score is part of the actual decision interface and affects reward through overconfidence penalties.


---

API

GET  /            
GET  /tasks       
POST /reset       
POST /step        
GET  /state       
GET  /grader      
GET  /baseline   

Session isolation is supported through the session_id query parameter.

In the current setup, these variables point to a compatible Hugging Face inference endpoint rather than an OpenAI-hosted endpoint. The implementation still follows the required OpenAI client interface.


---

Environment Variables

variable	description

API_BASE_URL	OpenAI-compatible endpoint
MODEL_NAME	model name used by inference.py
HF_TOKEN	API key
ENV_BASE_URL	environment server URL
SESSION_ID	session identifier



---

Local Setup

git clone https://github.com/Dibyajyoti001/recommender-triage-openenv.git
cd recommender-triage-openenv
python -m venv .venv

Windows PowerShell

.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

Linux / macOS

source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload


---

Docker

docker build -t recommendation-policy-triage .
docker run --rm -p 7860:7860 \
  -e API_BASE_URL=$API_BASE_URL \
  -e MODEL_NAME=$MODEL_NAME \
  -e HF_TOKEN=$HF_TOKEN \
  recommendation-policy-triage


---
```
recommender-triage-openenv/
├── app/
│   ├── main.py           
│   ├── models.py         
│   ├── tasks.py          
│   ├── simulator.py      
│   ├── reward.py        
│   ├── graders.py        
│   ├── candidate_pool.py 
│   └── data.py 
├── tests/
├── inference.py          
├── baselines.py          
├── client.py             
├── openenv.yaml          
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```
---

License

MIT

