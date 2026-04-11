---
title: Recommendation Policy Triage
emoji: "🎯"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: OpenEnv benchmark for trust-aware session recommendation.
---

# Recommendation Policy Triage

Recommendation Policy Triage is an OpenEnv-compatible benchmark for **session-level recommendation control under uncertainty**.
It evaluates not just what an agent recommends next, but how well it manages **user satisfaction, trust, calibration, diversity, and long-horizon session health** over time.

This environment is designed to feel closer to real recommender decision-making than one-step ranking benchmarks:

- user feedback can be noisy
- delayed failure modes exist
- trust can erode and recover
- diversity and relevance can conflict
- visible cost, risk, and latency constraints matter
- the agent only sees a partial view of the true user state

**Team:** Neonatal  
**Authors:** Dibyajyoti001, ankannayek  
**Hugging Face Space:** [dibyajyoti001-recommender-triage-openenv](https://dibyajyoti001-recommender-triage-openenv.hf.space)

## Why This Benchmark Matters

Most recommendation tasks stop at short-horizon optimization:

- maximize next-click probability
- predict the next consumed item
- rank a static candidate set

Production systems have a harder objective:

- serve relevant content without trapping the user in repetition
- recognize when historical memory is no longer trustworthy
- stay calibrated when observed feedback is unstable
- protect long-term trust after mistakes
- operate under safety, latency, and budget pressure

This benchmark reframes recommendation as a **control problem over user experience and trust**, not just a ranking problem.

## Benchmark At A Glance

- **5 tasks** covering stable exploitation, fatigue control, memory conflict, echo-chamber prevention, and noisy calibration
- **Structured action space** with recommendation choice, exploration intent, and confidence declaration
- **Partially observable state** with hidden memory, live intent, fatigue, trust, and patience
- **Analytically generated candidate pool** that evolves with session state
- **Visible operating constraints** through `cost`, `risk`, `latency`, `budget_remaining`, `risk_tolerance`, and `latency_budget`
- **Trajectory-level grading** with non-compensatory gating on collapse-sensitive tasks
- **Critical-turn counterfactual audit** on fragile turns in Tasks 4 and 5

## What The Agent Controls

Each step the policy returns:

- `recommended_item_id`
- `exploration_flag`
- `confidence_score`

That means the agent must decide both:

- what content to serve
- how strongly to stand behind that choice

Confidence is not cosmetic. Overclaiming can hurt trust and final score.

## What The Agent Observes

The observation exposes only coarse, policy-usable signals:

- task metadata
- memory summary over the 10-topic basis
- recent interactions
- candidate items
- repetition counts
- repetition pressure bucket
- memory confidence bucket and scalar
- current feedback signal
- feedback volatility
- trust signal
- engagement signal
- budget remaining
- risk tolerance
- latency budget

The agent does **not** directly observe hidden intent, hidden fatigue state, hidden trust dynamics, or the true internal user model.

## Hidden State

Each episode maintains:

- `m`: long-term memory preference
- `z`: live session intent
- `F_topic`: fast fatigue accumulator
- `H_topic`: slow repetition-pressure accumulator
- `p`: engagement / patience reserve
- `chi`: memory reliability
- `trust`: user-system relationship state

These hidden variables evolve throughout the session and shape the candidate pool, satisfaction, and final grade.

## Fixed Topic Basis

All mixed-topic representations use one shared 10-topic basis:

`Romance, Comedy, Drama, Thriller, Action, Documentary, Crime, SciFi, News, Lifestyle`

Candidates are sparse mixed-topic vectors, not just single-label categories. That lets the environment represent overlap, drift, and soft topic alignment.

## Candidate Pool Design

The environment does not sample from a static item catalog. Instead, each step it constructs six analytical candidate slots from the current hidden state:

- `live_best_fresh`
- `memory_best_fresh`
- `balanced_bridge`
- `fatigue_trap`
- `exploration_option`
- `conflict_option`

This makes the environment state-sensitive across the whole session. As trust, fatigue, drift, and collapse evolve, the candidate pool changes with them.

Every candidate also carries visible resource metadata:

- `cost`
- `risk`
- `latency`

## Task Suite

### `task_1` Stable Preference Exploitation

Historical memory is reliable and live intent mostly agrees with it.
The correct policy exploits stable preference efficiently without over-exploring.

Primary failure mode: unnecessary exploration or wasteful resource use.

### `task_2` Repetition Fatigue Control

The user strongly prefers a narrow topic region, but repeated recommendations create fatigue.
The policy must balance relevance and diversity before short-term wins become long-term decline.

Primary failure mode: greedy repetition.

### `task_3` Memory vs Live Signal Conflict

Historical memory and live session behavior diverge.
The policy must stop over-trusting stale memory and recover relevance after drift.

Primary failure mode: stale-memory lock-in.

### `task_4` Echo Chamber Collapse

The session starts with deceptively strong relevance in one narrow region.
If the policy keeps serving that loop, slow-history pressure saturates, trust erodes, and recovery becomes sticky.

Primary failure mode: reactive behavior that diversifies too late.

### `task_5` Noisy Signal Calibration

Observed feedback is noisy and aligned-looking items can still disappoint.
The policy must lower confidence under volatility while still preserving trust and relevance.

Primary failure mode: persistent overconfidence under uncertainty.

## Core Dynamics

### Relevance

For candidate topic vector `x_i` and quality `q_i`:

`R_t(i) = (alpha * z_t^T x_i + (1 - alpha) * m^T x_i + eta_q * q_i) / (1 + eta_q)`

### Fatigue And Repetition

The environment tracks both short-term and slow-burn repetition effects:

- `F_{t+1} = lambda_F * F_t + delta_F * x_{a_t}`
- `H_{t+1} = normalize(lambda_H * H_t + (1 - lambda_H) * x_{a_t})`

From these it derives:

- fatigue cost `C_t(i) = F_t^T x_i`
- repetition pressure `rho_t(i) = H_t^T x_i`
- novelty violation `V_t(i) = max(0, rho_t(i) - nu)`

### Trust-Aware Satisfaction

Satisfaction is not just relevance minus penalties. It is modulated by relationship state:

`y_t = sigmoid(zeta1 * R_t - zeta2 * C_t - zeta3 * V_t + zeta4 * trust_term)`

The same item can feel worse to a user who no longer trusts the system.

### Feedback Noise And Calibration

The observed `session_feedback_signal` is not always equal to the hidden utility.

In Task 5:

- observed feedback includes Gaussian noise
- the environment tracks rolling feedback volatility
- the ideal confidence target is derived from volatility with an exponential decay mapping

This makes confidence calibration an actual control problem instead of a cosmetic output field.

### Resource And System-Health Overlay

The benchmark adds a resource-aware overlay without replacing the original recommendation physics.
It derives:

- `resource_pressure`
- `risk_exposure`
- `diversity_pressure`

from visible item properties and existing dynamics.

This is important: the environment stays mathematically coherent because these are **derived overlays**, not a rewrite of the base relevance/fatigue model.

### Collapse-Sensitive Regimes

Tasks 4 and 5 include collapse-sensitive behavior:

- trust collapse can cap future achievable satisfaction
- risk collapse can raise future noise and volatility floor
- diversity collapse can alter future candidate geometry

These collapse states are triggered by **sustained unhealthy patterns**, not by one unlucky step.

## Reward And Grading

### Step Reward

The step reward shapes local behavior:

`r_t = w_r * R_t - w_f * C_t - w_n * V_t - w_u * U_t - w_conf * P_t`

Where:

- `U_t` penalizes unnecessary exploration
- `P_t = max(0, confidence_score - y_t)` penalizes overclaiming

The reward is clipped to `[-1, 1]`.

### Final Grader

The final benchmark score is trajectory-level, not just the average of step rewards.
Available grading components include:

- `satisfaction`
- `diversity`
- `adaptation`
- `memory_use`
- `trust`
- `calibration`
- `risk_safety`
- `resource_efficiency`

Tasks 1-3 use weighted trajectory grading.
Tasks 4-5 use a **non-compensatory gated grader** with collapse-sensitive penalties so that catastrophic behavior cannot be averaged away by doing well on one other axis.

### Counterfactual Decision Audit

Tasks 4 and 5 add a light **counterfactual decision audit** on a small number of fragile turns.
This is not a new task and it does not rewrite the simulator reward. Instead, it asks a narrower question:

> Was the chosen action actually a robust local decision when a safer plausible alternative existed?

For up to two fragile turns per episode, the environment:

- snapshots the hidden state before the action
- stores the visible candidate set
- records the chosen action
- replays a short 4-step branch from that snapshot

The audit compares:

- the policy's chosen action
- a safer plausible alternative from the same candidate set
- a greedier alternative from the same candidate set

To keep the comparison fair and low-variance, replay branches use **matched per-step random seeds**.
That means branch differences are driven much more by the action choice itself than by luck in noise or dud outcomes.

The branch value mixes:

- short-horizon return
- terminal trust after replay
- mean risk safety over the replay
- whether replay triggered collapse

The resulting audit score is blended into Tasks 4 and 5 with a small weight.
This gives the benchmark a second lens: not only "what happened overall," but also "was the local decision robust at the moment it mattered?"

## Why The Benchmark Is Useful

This environment is useful for:

- LLM-as-policy evaluation
- RL training under partial observability
- comparing exploration strategies
- testing calibration under noisy feedback
- stress-testing long-horizon recommender policies
- studying reward shaping versus trajectory-level grading

It is especially useful when you care about **how a policy behaves over a session**, not just whether it can rank the next item.

## API

### Core Environment Endpoints

- `GET /`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /grader`
- `GET /baseline`

### Runtime / Validation Endpoints

- `GET /health`
- `GET /metadata`
- `GET /schema`
- `POST /mcp`

Session isolation uses the `session_id` query parameter.

## Quickstart

### Local Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run The Server

Using uvicorn directly:

```powershell
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

Using the packaged entry point from `pyproject.toml`:

```powershell
uv run server
```

### Run The Inference Script

Required environment variables:

- `HF_TOKEN`

Optional environment variables with defaults:

- `API_BASE_URL`
- `MODEL_NAME`
- `ENV_BASE_URL`
- `BENCHMARK`
- `SESSION_ID`

Then run:

```powershell
python inference.py
```

## Docker

Build and run locally:

```bash
docker build -t recommendation-policy-triage .
docker run --rm -p 7860:7860 recommendation-policy-triage
```

The manifest and container both serve:

`server.app:app`

## Validation

Run tests:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Validate the repo:

```powershell
.venv\Scripts\openenv.exe validate
```

Validate a running server:

```powershell
.venv\Scripts\openenv.exe validate --url http://127.0.0.1:7860
```

## Repo Layout

```text
openenv_hackathon/
  app/
    main.py
    models.py
    tasks.py
    simulator.py
    reward.py
    graders.py
    candidate_pool.py
    data.py
  server/
    app.py
  tests/
  inference.py
  baselines.py
  client.py
  openenv.yaml
  Dockerfile
```

## Summary

Recommendation Policy Triage is not a toy recommender benchmark.
It is a compact testbed for **user experience and trust management under uncertainty**, with:

- sequential control instead of one-shot ranking
- hidden state instead of full observability
- calibration and trust as first-class objectives
- delayed collapse dynamics
- visible operating constraints
- trajectory-level grading that rewards robust long-horizon behavior

That combination is what makes the benchmark both harder and more realistic.

## References

The counterfactual decision audit in this benchmark is an environment-specific mechanism designed for sequential recommendation control. It is conceptually inspired by prior work on offline / counterfactual policy evaluation and careful reward-shaping practice:

- M. Dudik, J. Langford, and L. Li. [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601).
- J. McInerney, B. Brost, P. Chandar, R. Mehrotra, and B. Carterette. [Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions](https://arxiv.org/abs/2007.12986).
- A. Y. Ng, D. Harada, and S. Russell. [Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping](https://wordpress.andrewng.org/index.php/publication/policy-invariance-under-reward-transformations-theory-and-application-to-reward-shaping/).
