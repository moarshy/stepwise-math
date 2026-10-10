"""
Mimic the arena's admission check against a running container: play the example
actions on every v1 task and require done=True with a reward in [0, 1].

Usage: python scripts/check_container.py http://127.0.0.1:8000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from math_arena.client import MathArenaEnv  # noqa: E402
from math_arena.models import MathArenaAction  # noqa: E402
from math_arena.server.generators import V1_TASKS  # noqa: E402

EXAMPLE_ACTIONS = [{"command": "calc", "text": "2+2"}, {"command": "submit", "text": "0"}]

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
ok = True
for task_id in V1_TASKS:
    with MathArenaEnv(base_url=url).sync() as env:
        r = env.reset(task_id=task_id)
        first = r.observation.message.split("\n")[0][:70]
        for a in EXAMPLE_ACTIONS:
            r = env.step(MathArenaAction(**a))
            if r.done:
                break
        passed = r.done and r.reward is not None and 0.0 <= r.reward <= 1.0
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {task_id:20s} done={r.done} reward={r.reward}  | {first}…")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
