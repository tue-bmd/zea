# Agent — cognitive ultrasound

`zea/agent/` is what makes `zea` a *cognitive* ultrasound toolbox. It provides building blocks for
**active perception**: closing the action–perception loop so the scanner adapts what it acquires
based on what it currently believes about the tissue (`zea/agent/__init__.py`).

## The loop

Active perception iterates (`zea/agent/__init__.py`):

1. **Perceive** the tissue state from acquired measurements (often via a generative model that
   produces `particles` — samples of the belief state).
2. **Select transmit actions** based on those beliefs.
3. **Acquire** new data and loop back.

The partial/frame-level access of the [data format](data.md) and the [models](models.md) that
produce belief states are the other two legs of this loop.

## Action-selection strategies

`zea/agent/selection.py` implements the "select" step. Strategies are **stateless** — they hold no
internal state between calls — and currently target *focused transmit* actions
(`zea/agent/selection.py` module docstring; general transmit schemes are a documented
work-in-progress in `zea/agent/__init__.py`).

The class hierarchy builds on masking (`zea/agent/selection.py`):

- `MaskActionModel` — base class; `apply(action, observation)` returns `observation * action`.
- `LinesActionModel(MaskActionModel)` — base for strategies that select scan **lines**, parametrized
  by `n_actions`, `n_possible_actions`, `img_width`, `img_height`.

Available strategies (registered in `action_selection_registry`, listed in `zea/agent/__init__.py`):

| Strategy | Idea |
| --- | --- |
| `GreedyEntropy` | Select lines that maximize entropy reduction. |
| `UniformRandomLines` | Randomly sample scan lines uniformly. |
| `EquispacedLines` | Equispaced lines sweeping across the image. |
| `CovarianceSamplingLines` | Model line-to-line correlation to pick the highest-entropy masks. |
| `TaskBasedLines` | Select lines to maximize information gain for a downstream task. |

## Supporting modules

- `zea/agent/masks.py` — mask construction utilities used by the line strategies.
- `zea/agent/gumbel.py` — Gumbel(-softmax) sampling utilities (differentiable discrete selection);
  see `tests/test_gumbel.py`.
- Selection uses `zea.backend.autograd.AutoGrad` for backend-agnostic gradients
  (`zea/agent/selection.py` imports), so entropy/information objectives differentiate on any backend.

## Where to look / what to watch

- Strategies must stay stateless and backend-agnostic (`keras.ops` + `AutoGrad`).
- Register new strategies with `action_selection_registry` so they are selectable by name.
- Relevant tests: `tests/test_agent.py`, `tests/test_gumbel.py`, `tests/test_fnumber_mask.py`.
- Example notebook referenced by the code: `docs/source/notebooks/agent/agent_example`.
