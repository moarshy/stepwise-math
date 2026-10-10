"""Re-derive every generator's answer a second, independent way (brute force where possible)."""

import itertools
import math
import random
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import generators as g  # noqa: E402
from server.math_arena_environment import MathArenaEnvironment, calculate, parse_answer  # noqa: E402
from models import MathArenaAction  # noqa: E402

N = 300
sys.set_int_max_str_digits(0)  # brute-force check prints 40,000-digit powers


def nums(q):
    return [int(x) for x in re.findall(r"-?\d+", q)]


def test_all_generators_run_and_are_seeded():
    for name, gen in g.GENERATORS.items():
        for seed in range(N):
            q, a = gen(random.Random(seed))
            assert isinstance(a, Fraction) and q
            assert gen(random.Random(seed)) == (q, a), name  # same seed, same problem


def test_counting_medium_brute_force():
    for seed in range(40):
        q, a = g.counting_medium(random.Random(seed))
        n, k = nums(q)[:2]
        # members 0 and 1 are Ana and Ben
        assert a == sum(1 for c in itertools.combinations(range(n), k) if not {0, 1} <= set(c))


def test_counting_hard_brute_force():
    for seed in range(40):
        q, a = g.counting_hard(random.Random(seed))
        word, first = re.search(r"letters in (\w+) begin with the letter (\w)", q).groups()
        if len(word) > 9:
            continue
        perms = {p for p in itertools.permutations(word) if p[0] == first}
        assert a == len(perms), (q, a, len(perms))


def test_probability_brute_force():
    for seed in range(N):
        q, a = g.probability_medium(random.Random(seed))
        k = nums(q)[-1]
        assert a == Fraction(sum(i + j >= k for i in range(1, 7) for j in range(1, 7)), 36)
    for seed in range(60):
        q, a = g.probability_hard(random.Random(seed))
        r, b, gr = nums(q)[:3]
        bag = ["r"] * r + ["b"] * b + ["g"] * gr
        triples = list(itertools.combinations(range(len(bag)), 3))
        hits = sum(len({bag[i], bag[j], bag[k]}) == 3 for i, j, k in triples)
        assert a == Fraction(hits, len(triples))


def test_divisibility():
    for seed in range(60):
        q, a = g.divisibility_medium(random.Random(seed))
        n, x, y = nums(q)[1:4]
        assert a == sum(1 for k in range(1, n + 1) if k % x == 0 or k % y == 0)
    for seed in range(60):
        q, a = g.divisibility_hard(random.Random(seed))
        n, x, y, z = nums(q)[1:5]
        # inclusion-exclusion, independent of the generator's brute force
        L = math.lcm
        inc = n // x + n // y - n // L(x, y)
        exc = n // L(x, z) + n // L(y, z) - n // L(x, y, z)
        assert a == inc - exc


def test_algebra_by_search():
    for gen in (g.algebra_medium, g.algebra_hard):
        for seed in range(N):
            q, a = gen(random.Random(seed))
            if gen is g.algebra_medium:
                m, k, ca, cb, rhs = map(int, re.search(
                    r"y = (-?\d+)x \+ (-?\d+) and (-?\d+)x \+ (-?\d+)y = (-?\d+)", q.replace(" - ", " + -")
                ).groups())
                checks = lambda x, y: y == m * x + k and ca * x + cb * y == rhs  # noqa: E731
            else:
                ca, cb, r1, cc, cd, r2 = nums(q.replace(" - ", " + -"))
                checks = lambda x, y: ca * x + cb * y == r1 and cc * x + cd * y == r2  # noqa: E731
            sols = [(x, y) for x in range(-12, 13) for y in range(-200, 201) if checks(x, y)]
            assert len(sols) == 1 and a == sum(sols[0]), q


def test_modular_and_sequences():
    for seed in range(N):
        q, a = g.modular_hard(random.Random(seed))
        base, n = nums(q)[1:3]
        digits = int(re.search(r"last (\d) digits", q).group(1))
        assert a == int(str(base ** n)[-digits:])
    for seed in range(N):
        q, a = g.modular_medium(random.Random(seed))
        base, n, m = nums(q)
        assert int(str(base ** n)) % m == a and n > 64
    for seed in range(N):
        q, a = g.sequences_medium(random.Random(seed))
        i, ti, j, tj, lo, hi = nums(q)
        d = (tj - ti) // (j - i)
        first = ti - (i - 1) * d
        assert a == sum(t for t in range(first, hi + 1, d) if t >= lo)


def test_calculator_and_parsing():
    assert calculate("1/3 + 1/6") == "1/3 + 1/6 = 1/2"
    assert calculate("2**10 % 7") == "2**10 % 7 = 2"
    assert "exponent" in calculate("7**5000")
    assert "not" in calculate("__import__('os')") or "only" in calculate("__import__('os')")
    assert "division by zero" in calculate("1/0")
    assert parse_answer(" 7/12 ") == Fraction(7, 12)
    assert parse_answer("14/24") == Fraction(7, 12)
    assert parse_answer("0.375") == Fraction(3, 8)
    assert parse_answer("seven") is None


def test_episode_right_wrong_and_timeout():
    env = MathArenaEnvironment()
    for task in g.GENERATORS:
        obs = env.reset(seed=1, task_id=task)
        assert not obs.done and obs.moves_left == 20
        right = g.fmt(env._answer)
        obs = env.step(MathArenaAction(command="submit", text=right))
        assert obs.done and obs.reward == 1.0, task
        env.reset(seed=1, task_id=task)
        obs = env.step(MathArenaAction(command="submit", text="999999999"))
        assert obs.done and obs.reward == 0.0
    env.reset(seed=1, task_id="counting-medium")
    for _ in range(19):
        obs = env.step(MathArenaAction(command="calc", text="1+1"))
        assert not obs.done
    obs = env.step(MathArenaAction(command="calc", text="1+1"))
    assert obs.done and obs.reward == 0.0
