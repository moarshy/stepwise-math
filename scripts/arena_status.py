"""Poll our arena submission (and its run, once started). Usage: python scripts/arena_status.py stepwise-math-v1 [--watch]"""
import json
import sys
import time

import httpx
from huggingface_hub import get_token

BASE = "https://openenvarena-arena.hf.space/api/openenv"
sub_id = sys.argv[1]
watch = "--watch" in sys.argv
H = {"Authorization": f"Bearer {get_token()}"}


def get(path):
    for _ in range(5):
        try:
            r = httpx.get(BASE + path, headers=H, timeout=60)
            if r.status_code not in (502, 503, 504):
                return r.json()
        except httpx.HTTPError:
            pass
        time.sleep(5)
    return {}


while True:
    s = get(f"/submissions/{sub_id}")
    checks = {c["name"]: c["status"] for c in s.get("report", {}).get("checks", [])}
    run = s.get("run") or {}
    line = f"{time.strftime('%H:%M:%S')} state={s.get('state')} episode_check={checks.get('isolated_openenv_episode')}"
    if run:
        line += f" run={run.get('run_id')} run_state={run.get('state')} step={run.get('policy_version')}"
    print(line, flush=True)
    if s.get("state") == "rejected":
        print(json.dumps({k: s.get(k) for k in ("error_code", "error_origin")}, indent=1))
        print(json.dumps(s.get("report", {}).get("error_details"), indent=1))
        break
    if not watch or (s.get("state") == "validated" and run.get("run_id")) or run.get("state") in ("completed", "failed", "cancelled"):
        break
    time.sleep(60)
