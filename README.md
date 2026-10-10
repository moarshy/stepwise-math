# Stepwise Math

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment for RL training on multi-step math problems that need to be reasoned through one step at a time with a calculator. It was built for [OpenEnv Arena](https://openenvarena-arena.hf.space/), which targets the *Math & formal reasoning* domain.

Each episode draws a **fresh random problem** from a procedural generator, so the model can't memorise answers and has to learn the method. Every answer key is exact (Python `Fraction`) and checked independently by brute-force tests.

## An episode

```
ENV   : A bag has 8 red, 5 blue and 7 green marbles. Two marbles are drawn at random
        without replacement. What is the probability they are the same colour? ...
MODEL : {"command": "calc", "text": "(8*7 + 5*4 + 7*6) / (20*19)"}
ENV   : (8*7 + 5*4 + 7*6) / (20*19) = 59/190
MODEL : {"command": "submit", "text": "59/190"}
ENV   : Correct.                       reward 1.0, done
```

| | |
|---|---|
| **Actions** | `{"command": "calc", "text": "<expression>"}` runs an exact calculator (`+ - * / // % **` and brackets, exponents up to 64). `{"command": "submit", "text": "<answer>"}` gives the final answer. |
| **Observation** | `message` (the problem, a calculator result or the grade) and `moves_left` |
| **Reward** | 1 if the submitted value equals the answer exactly, otherwise 0. Equivalent forms count (`14/24` = `7/12`, `0.375` = `3/8`); rounded decimals like `0.5833` don't |
| **Limit** | 20 moves. Running out scores 0. |

The calculator's exponent limit is deliberate. For "last 3 digits of 13^7146" the model has to reduce the power by reasoning instead of brute-forcing it.

## Tasks

The table shows 7 families × 2 levels. **v1 submits the 8 marked tasks**: the ones that Qwen3.8-27B (thinking off) solved some of the time but not always, which is where GRPO gets a learning signal.

| Task | Example | Qwen3.8-27B solved | v1 |
|---|---|---|---|
| counting-medium | Committees of k from n where two people refuse to serve together | 1/5, 4/5 | ✅ |
| counting-hard | Arrangements of the letters in PGCAAGG that begin with C | 0/5, 2/5 | ✅ |
| modular-medium | Remainder of 12^504 divided by 11 | 3/5 | ✅ |
| modular-hard | Last 3 digits of 11^3471 | 0/5, 1/5 | ✅ |
| sequences-medium | Sum of the terms of an arithmetic sequence between 100 and 900 | 2/5 | ✅ |
| sequences-hard | a(n) = 3a(n-1) + 2a(n-2): find a(11) | 1/5, 2/5 | ✅ |
| probability-medium | Two dice: P(sum ≥ k) | 3/5, 3/5 | ✅ |
| probability-hard | Three marbles drawn: P(all different colours) | 3/5 | ✅ |
| divisibility-medium | Count 1..n divisible by a or b | 5/5 (too easy) | |
| divisibility-hard | Count 1..n divisible by a or b but not c | 0/5 | |
| algebra-medium | y = mx + k and ax + by = c: find x + y | 0/5 | |
| algebra-hard | General 2×2 system: find x + y | 0/5 | |

Solve rates come from `scripts/calibrate.py`, which plays the episodes through OpenRouter the same way the arena does (system message = action schema, observations as JSON, one JSON action per reply). Where two numbers are given, they are separate runs. Five tries per task is noisy, so treat these numbers as rough bands. Logs and transcripts are in `calibration/`.

### What calibration showed
- With thinking off, the model reasons **only through its calculator calls**. It often packs a whole solution into one invented formula.
- Typical failures are planning slips, not arithmetic. Examples: subtracting the wrong complement in counting; forgetting that fixing the first letter removes it from the arrangement; using a × b instead of lcm(a, b).
- Families where a single calculator call does everything (plain fractions, simple word problems) were 5/5 and were dropped.

## Run it

```sh
uv sync
uv run --project . server          # http://localhost:8000
python -m pytest tests             # brute-force checks of every answer key
```

Docker (the arena needs `linux/amd64`):

```sh
export DOCKER_DEFAULT_PLATFORM=linux/amd64
openenv build -t ghcr.io/moarshy/stepwise-math:v1
docker run --rm -p 8000:8000 ghcr.io/moarshy/stepwise-math:v1
openenv validate --url http://localhost:8000
python scripts/check_container.py http://127.0.0.1:8000   # arena-style admission check
```

## Layout

```
models.py                         action / observation schema
server/generators.py              the 12 problem generators + V1_TASKS
server/math_arena_environment.py  reset / step, exact calculator, grading
scripts/calibrate.py              difficulty check against Qwen3.8-27B
scripts/check_container.py        arena-style admission check
tests/                            independent brute-force verification
```

Built with OpenEnv pinned at `86a180ed` (the arena's revision).
