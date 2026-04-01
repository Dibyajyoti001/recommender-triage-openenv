# Recommendation Policy Triage Environment

This repository contains a complete simulation environment for the
**Recommendation Policy Triage** problem.  The environment follows
the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) API and is
designed as a testbed for sequential recommendation agents under
dynamic user behaviour.  It was built for the OpenEnv hackathon and
implements the mechanics, reward shaping and grading logic described
in the accompanying design document.

## High‑Level Overview

The simulation models a single user session with four hidden
properties:

* **Long‑term memory** (`m`) – a preference distribution over content
  categories derived from the user's history.
* **Session intent** (`z`) – what the user wants in this particular
  session; it can drift over time.
* **Fatigue** (`F_cat`, `F_item`) – how bored the user is with a
  category or specific item; it increases with repetition and decays
  with time.
* **Patience / novelty tolerance** (`p`) – how tolerant the user is of
  irrelevant or novel recommendations.  Low patience triggers intent
  drift and encourages exploration.

At every turn the environment generates a **candidate pool** of six
items with varying categories, qualities and freshness.  The agent
chooses one item to recommend and optionally signals whether it is
exploring.  After each action the hidden state is updated: relevance
is computed using a convex combination of the live session intent and
long‑term memory, fatigue is updated, novelty violations are detected
and patience is adjusted.  The environment returns a shaped step
reward and a new observation.  An episode lasts for up to 20 steps; a
final grader then computes satisfaction, diversity, adaptation and
memory‑use scores to produce a final grade.

## File Structure

```
openenv_hackathon/
│
├── app/
│   ├── main.py            # FastAPI server exposing reset/step/state endpoints
│   ├── models.py          # Pydantic data models for observations and actions
│   ├── simulator.py       # Core environment logic (hidden state updates)
│   ├── candidate_pool.py  # Deterministic candidate generation (6 slots)
│   ├── reward.py          # Shaped step reward computation
│   ├── graders.py         # Final episode grading (satisfaction, diversity, etc.)
│   └── tasks.py           # Parameter sets for the three tasks
│
├── inference.py          # Baseline heuristic agent
├── openenv.yaml          # Schema and task definitions for OpenEnv
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container specification for HF Spaces
└── README.md             # This file
```

## Running Locally

Install dependencies and start the server:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 7860
```

The server exposes three endpoints:

* `POST /reset` – initialise a new environment and return the first
  observation.  Provide an optional `task_id` to choose between the
  three tasks.
* `POST /step` – submit an action (chosen item, exploration flag,
  confidence) and receive the next observation, reward and done flag.
* `GET /state` – return a JSON serialisation of the hidden state for
  debugging (this is not used in official grading).

## Baseline Agent

`inference.py` contains a simple baseline policy that interacts with
the live environment via HTTP.  It illustrates how to parse
observations, select an item heuristically and send actions back to
the server.  The baseline agent follows these very naïve rules:

1. Pick the candidate with the highest advertised quality unless the
   category has been recommended in the last two steps.
2. If the user’s patience drops below a threshold, explore by
   selecting a novel risky distractor.
3. Use the reported memory summary to resolve conflicts between
   long‑term memory and live intent.

This baseline is not intended to be competitive; it simply
demonstrates how to consume the API.  Participants are encouraged to
replace it with more sophisticated logic or reinforcement‑learning
agents.

## Deployment to Hugging Face Spaces

The provided `Dockerfile` builds a minimal image using Python 3.10.  To
deploy on HF Spaces:

1. Create a new Docker Space and upload this repository.
2. Set any necessary environment variables (e.g. `OPENAI_API_KEY` if
   your inference policy uses OpenAI models).
3. Build and deploy; the server will run on port 7860 as expected by
   OpenEnv.

## Notes and Limitations

* The parameter values used for fatigue decay, patience updates and
  reward shaping are approximate translations from the design document.
  They can be tuned in `tasks.py`.
* The final grading functions implement satisfaction, diversity,
  adaptation and memory‑use metrics; however adaptation and
  memory‑use heuristics are simplified for clarity.
* This environment is intended as a starting point for research and
  hackathon experimentation.  Contributors can extend or refine any
  component – particularly the baseline agent – to explore more
  sophisticated recommendation strategies.