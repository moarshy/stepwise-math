"""
Difficulty check: let Qwen3.8-27B (thinking off) play our tasks the way the arena does,
and report how often it solves each one.

Mimics the arena loop described in its AGENTS.md:
  system message = our action schema, first user turn = reset observation (JSON),
  each reply must be exactly one JSON action (or {"finish": true}); anything else ends
  the episode with reward 0.

Usage:
  python scripts/calibrate.py --episodes 1 --tasks probability-hard
  python scripts/calibrate.py --episodes 20 --workers 8
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import InferenceClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import MathArenaAction, MathArenaObservation  # noqa: E402
from server.generators import GENERATORS, fmt  # noqa: E402
from server.math_arena_environment import MathArenaEnvironment  # noqa: E402

BACKENDS = {
    # Hugging Face Inference Providers (needs HF credits)
    "hf": dict(model="Qwen/Qwen3.8-27B", extra={"chat_template_kwargs": {"enable_thinking": False}}),
    # OpenRouter (key from OPENROUTER_API_KEY)
    "openrouter": dict(model="qwen/qwen3.8-27b", extra={"reasoning": {"enabled": False}}),
}
SCHEMA = {
    "action": MathArenaAction.model_json_schema(),
    "observation": MathArenaObservation.model_json_schema(),
}
SYSTEM = (
    "You are acting in an environment. Each of your replies must be exactly one JSON object and nothing else: "
    "either an action matching this JSON schema, or {\"finish\": true} to end the episode.\n\n"
    f"Action schema:\n{json.dumps(SCHEMA['action'])}"
)


def obs_json(obs) -> str:
    return json.dumps({k: v for k, v in obs.model_dump().items() if k != "metadata"})


def parse_reply(text: str):
    """Exactly one JSON object, nothing else; returns dict or None."""
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def play(client, backend, task_id, seed, max_tokens):
    env = MathArenaEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": obs_json(obs)}]
    usage = {"prompt": 0, "completion": 0}
    outcome, reward = "out_of_moves", 0.0
    while True:
        for attempt in range(4):
            try:
                resp = client.chat_completion(
                    messages=messages, model=backend["model"], max_tokens=max_tokens, temperature=1.0,
                    extra_body=backend["extra"],
                )
                break
            except Exception as e:  # transient provider errors
                if attempt == 3:
                    return dict(task=task_id, seed=seed, outcome=f"api_error: {e}", reward=0.0,
                                answer=fmt(env._answer), usage=usage, messages=messages)
                time.sleep(2 ** attempt)
        reply = resp.choices[0].message.content or ""
        if resp.usage:
            usage["prompt"] += resp.usage.prompt_tokens
            usage["completion"] += resp.usage.completion_tokens
        messages.append({"role": "assistant", "content": reply})

        action = parse_reply(reply)
        if action is None:
            outcome = "invalid_reply"
            break
        if action == {"finish": True}:
            outcome = "finished_without_submit"
            break
        try:
            obs = env.step(MathArenaAction(**action))
        except Exception:
            outcome = "invalid_action"
            break
        messages.append({"role": "user", "content": obs_json(obs)})
        if obs.done:
            reward = obs.reward
            outcome = "correct" if reward == 1.0 else ("wrong" if action.get("command") == "submit" else "out_of_moves")
            break
    return dict(task=task_id, seed=seed, outcome=outcome, reward=reward, answer=fmt(env._answer),
                usage=usage, messages=messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--tasks", nargs="*", default=list(GENERATORS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--backend", choices=list(BACKENDS), default="openrouter")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default=str(ROOT / "calibration" / "run.jsonl"))
    args = ap.parse_args()

    backend = BACKENDS[args.backend]
    if args.backend == "openrouter":
        client = InferenceClient(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    else:
        client = InferenceClient(provider="auto")
    jobs = [(t, 1000 + i) for t in args.tasks for i in range(args.episodes)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(args.workers) as pool, out.open("w") as f:
        for r in pool.map(lambda j: play(client, backend, j[0], j[1], args.max_tokens), jobs):
            results.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{r['task']:20s} seed={r['seed']} {r['outcome']:24s} answer={r['answer']}", flush=True)

    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)
    print(f"\n{'task':20s} {'solved':>8s}  outcomes")
    for t in args.tasks:
        rs = by_task[t]
        solved = sum(r["reward"] == 1.0 for r in rs)
        counts = defaultdict(int)
        for r in rs:
            counts[r["outcome"]] += 1
        print(f"{t:20s} {solved:>3d}/{len(rs):<4d}  {dict(counts)}")
    tok_in = sum(r["usage"]["prompt"] for r in results)
    tok_out = sum(r["usage"]["completion"] for r in results)
    print(f"\ntokens: {tok_in:,} in, {tok_out:,} out; transcripts in {out}")


if __name__ == "__main__":
    main()
