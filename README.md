# ENPIRE-Sim

A **simulation-only reproduction and extension** of NVIDIA's
[ENPIRE](https://research.nvidia.com/labs/gear/enpire/) (*Agentic Robot Policy
Self-Improvement in the Real World*, Xiao et al., 2026), built as a master's
thesis research codebase.

ENPIRE is a harness in which **coding agents** (Claude Code, Codex, Kimi Code)
autonomously improve real-robot manipulation policies through a closed loop of
four modules — **E**nvironment, **P**olicy-**I**mprovement, **R**ollout, and
**E**volution. This project rebuilds that loop **entirely in simulation** (no
robot hardware, no GPU required) and adds a **novel contribution**: it replaces
ENPIRE's greedy, Git-emergent hypothesis selection with a *principled,
bandit-based Evolution module* (UCB and Thompson sampling), so the
exploration/exploitation strategy can be studied and ablated rigorously.

> **Status:** research scaffold. It runs out of the box with zero API keys and
> only `numpy`, using a built-in Push-T simulation and a deterministic
> `MockAgent`. Real coding-agent backends (Claude, OpenAI, local Ollama models)
> and the high-fidelity `gym-pusht` environment are optional add-ons.

---

## Why this exists

The original ENPIRE is a *systems* paper requiring 8 bimanual robot stations
and frontier-model API budgets. That's not reproducible in a typical lab. This
codebase distills the same closed-loop autoresearch idea into something a
single student can run on a laptop, while making one clean scientific
contribution:

**Research question:** *Does treating hypothesis selection as a multi-armed
bandit (UCB / Thompson sampling) accelerate policy improvement compared to the
greedy hill-climbing that ENPIRE uses by default?*

---

## How the modules map to ENPIRE

| ENPIRE module | This repo | What it does |
|---|---|---|
| **EN** — Environment | `enpire/environment/` | Self-resetting, self-verifying Push-T task (Gym-style). Reset + reward are free in sim. |
| **PI** — Policy Improvement | `enpire/agents/` + `enpire/policy/` | Coding agents propose **hypotheses** (heuristic-policy parameter settings); each compiles to a runnable policy. |
| **R** — Rollout | `enpire/rollout/` | Evaluates a hypothesis over many randomized episodes → a noisy Bernoulli success rate. |
| **E** — Evolution | `enpire/evolution/` | **The novel contribution.** Bandit selectors decide which hypothesis to invest the next rollout budget in: `greedy` (baseline), `ucb`, `thompson`, `random`. |
| Fleet / metrics | `enpire/metrics.py`, `enpire/orchestrator.py` | Multi-agent loop + MRU / MTU efficiency metrics and time-to-success. |

The crucial design choice: **generation (agents) is decoupled from selection
(the Evolution module)**, so every selector is compared on the *same* stream of
agent-proposed hypotheses — exactly the fair ablation ENPIRE calls for.

---

## Quickstart (zero setup beyond pip)

```bash
git clone https://github.com/goldenglorys/enpire-sim.git
cd enpire-sim
pip install -r requirements.txt        # numpy, matplotlib, pyyaml, pytest

# Run the test suite (should be all green)
pytest -q

# Run one autoresearch experiment — fully offline, no API keys
python scripts/run_experiment.py --agent mock --n-agents 4 --selector ucb \
    --max-steps 80 --out results/ucb_mock.json

# Run the headline ablation (greedy vs ucb vs thompson vs random)
python scripts/run_ablation.py --selectors greedy ucb thompson random \
    --n-agents 4 --seeds 8 --max-steps 100 --out results/ablation.json

# Plot the success-rate-vs-time curves
python scripts/plot_results.py results/ablation.json --out results/curves.png
```

Everything above runs with the built-in numpy Push-T simulation and the
API-free `MockAgent` (a deterministic local search), so it is free,
reproducible, and CI-friendly.

---

## Using real coding agents

The `MockAgent` exists so you can develop and run the bandit ablation without
cost. To reproduce ENPIRE's *frontier coding agents*, swap in a real backend:

### Claude (frontier)
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python scripts/run_experiment.py --agent claude --n-agents 1 --selector greedy \
    --max-steps 40
```
Defaults to `claude-opus-4-8` with structured outputs; override with `--model`.

### OpenAI (frontier)
```bash
pip install openai
export OPENAI_API_KEY=sk-...
python scripts/run_experiment.py --agent openai --model gpt-4o --n-agents 1
```

### Local / open-source (the democratization study)
```bash
# Install Ollama (https://ollama.com), then:
ollama pull qwen2.5-coder
python scripts/run_experiment.py --agent local --model qwen2.5-coder --n-agents 1
```
No API key, no cost — runs on your own GPU/CPU. This backs the
*frontier-vs-open-source* comparison described below.

---

## High-fidelity environment (optional)

The built-in Push-T uses a **simplified quasi-static** physics model so the repo
has no heavy dependencies. For publication-grade results, switch to the official
`gym-pusht`:

```bash
pip install gym-pusht gymnasium
python scripts/run_experiment.py --backend gym_pusht --agent mock
```

Both backends share the exact same interface (`enpire/environment/base.py`), so
nothing else changes.

---

## RoboCasa — the simulation ENPIRE uses for its sim results

ENPIRE reports simulation results on **RoboCasa365**, a large MuJoCo
kitchen-manipulation suite. ENPIRE-Sim ships a RoboCasa adapter
(`enpire/environment/robocasa.py`) behind the **same `BaseEnv` interface**, so
the identical orchestrator, agents, and bandit-based Evolution module drive
policy improvement on RoboCasa just as on Push-T.

### Setup (heavy — MuJoCo + assets)

```bash
pip install robosuite robocasa mujoco torch
python -m robocasa.scripts.download_kitchen_assets   # multi-GB asset download
```

### Run

```python
from enpire.environment import make_env
env = make_env("robocasa", task="OpenDrawer", robots="PandaOmron")
# then drive it with the same Orchestrator as Push-T
```

Available tasks are listed in `enpire/environment/robocasa.ROBOCASA_TASKS`
(e.g. `PnPCounterToCabinet`, `OpenDrawer`, `TurnOnSinkFaucet`, `CoffeeSetupMug`).

### Important: RoboCasa needs a *learned* policy

A 5-parameter heuristic can drive Push-T, but a 7-DoF arm in a kitchen cannot be
controlled that way. RoboCasa therefore uses the **behavior-cloning regime**
(`enpire/policy/behavior_cloning.py`), where the agent's *hypothesis* is a set of
**BC training hyperparameters** (`BC_PARAM_SPACE`: learning rate, hidden size,
layers, epochs, BC-loss weight, …) — the RoboCasa analogue of the Push-T
parameter genome, and exactly the kind of recipe ENPIRE's agents tuned.

What's wired today vs. the remaining integration step:

| Component | Status |
|---|---|
| RoboCasa env adapter (`reset`/`step`/`is_success`, obs flattening) | ✅ implemented |
| `make_env("robocasa", ...)` routing | ✅ implemented |
| BC policy (torch MLP, `.fit(demos)`) + `BC_PARAM_SPACE` | ✅ implemented |
| Bandit Evolution module over BC hyperparameters | ✅ works unchanged |
| **Demo source + BC training/eval loop wired to the orchestrator** | ⬜ next step |

To finish the RoboCasa path you provide a **demonstration source** for the task
(scripted policy, teleop, or RoboCasa's demo datasets) and have
`evaluate_hypothesis` train a `BehaviorCloningPolicy` on those demos before
rollout. The bandit selectors and metrics need no changes — that's the whole
point of the decoupled design.

> ⚠️ RoboCasa cannot run in every environment (it needs MuJoCo + GPU-friendly
> assets). Develop the loop on the offline Push-T path first, then switch the
> `make_env` call once RoboCasa is installed locally.

---

## Repository layout

```
enpire-sim/
├── enpire/
│   ├── environment/      # EN: Push-T env (builtin + gym-pusht adapter)
│   ├── policy/           # PI: Hypothesis representation + heuristic policy
│   ├── agents/           # PI: coding agents (mock, claude, openai, local)
│   ├── rollout/          # R:  evaluate a hypothesis → success rate
│   ├── evolution/        # E:  bandit selectors (greedy/ucb/thompson/random)  <- novel
│   ├── metrics.py        # MRU, MTU, time-to-success
│   ├── orchestrator.py   # the closed-loop autoresearch driver
│   └── utils/            # logging (console + JSONL idea-tree)
├── scripts/
│   ├── run_experiment.py # one run
│   ├── run_ablation.py   # selectors x fleet sizes x seeds   <- core thesis experiment
│   └── plot_results.py   # success-vs-time figure
├── configs/              # default.yaml, ablation.yaml
├── tests/                # pytest suite (env, selectors, orchestrator)
└── results/              # experiment outputs (gitignored)
```

---

## The novel contribution, in detail

ENPIRE's Evolution module is *greedy*: agents coordinate through Git and
cherry-pick whatever currently has the best average success rate. That makes no
explicit exploration/exploitation trade-off. ENPIRE-Sim reframes hypothesis
selection as a **multi-armed bandit**:

- each **hypothesis** is an arm,
- pulling an arm spends a rollout budget to (re-)evaluate it,
- the reward is the noisy, per-episode Bernoulli success.

Implemented selectors (`enpire/evolution/`):

| Selector | Strategy | Role |
|---|---|---|
| `greedy` | epsilon-greedy on empirical mean | **ENPIRE baseline** |
| `ucb` | UCB1: `mean + c·√(ln N / nᵢ)` | principled optimism |
| `thompson` | Beta-posterior sampling | Bayesian comparator |
| `random` | uniform | lower-bound control |

The headline experiment (`run_ablation.py`) compares them on *time-to-target
success rate*, *final success rate*, and ENPIRE's efficiency metrics — **Mean
Robot Utilization (MRU)** and **Mean Token Utilization (MTU)**.

### A natural second study (no extra code)

Because the agent backend is a single CLI flag, you can also run the
*frontier-vs-open-source* comparison: run identical experiments with `--agent
claude`, `--agent openai`, and `--agent local` and compare success, tokens, and
cost. This is a strong, practical thesis chapter on the cost of physical
autoresearch.

---

## Suggested thesis roadmap

1. **Reproduce** the loop in sim with `MockAgent` on built-in Push-T (done — this repo).
2. **Run the ablation** (greedy vs UCB vs Thompson) across fleet sizes and seeds; report MRU/MTU and time-to-success.
3. **Swap in a frontier agent** (Claude/OpenAI) and one **open-source** agent (Ollama) for the democratization study.
4. **Scale up fidelity** with `gym-pusht`, then extend to a second task (e.g. a RoboCasa task) behind the same `BaseEnv` interface.
5. **Write up.** Target ICRA / CoRL / an ICLR robot-learning workshop — the paper is < 1 month old, so building on it is timely.

---

## Reproducibility notes

- All randomness is seeded (`--seed`); the `MockAgent` path is fully
  deterministic, so ablation numbers are exactly reproducible.
- Every run can emit a JSONL **idea tree** (`--log results/run.jsonl`): one line
  per proposed hypothesis and per rollout, with parent links — the simulation
  analogue of ENPIRE's Git experiment history.

## Citing the original work

This is an independent reproduction. Please cite the original paper:

> Xiao, W., Xie, J., Zhang, T., Lin, H., et al. *ENPIRE: Agentic Robot Policy
> Self-Improvement in the Real World.* NVIDIA GEAR Lab / CMU / UC Berkeley, 2026.
> https://research.nvidia.com/labs/gear/enpire/

## License

MIT — see [LICENSE](LICENSE).
