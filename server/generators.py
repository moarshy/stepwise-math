# SPDX-License-Identifier: BSD-3-Clause

"""
Procedural math problem generators.

Each generator takes a random.Random and returns (question, answer), where the
answer is an exact Fraction. A fresh random draw on every reset means the model
never sees the same instance twice, so it has to learn the method, not the answer.
"""

import math
import random
from fractions import Fraction
from typing import Callable, Dict, Tuple

Problem = Tuple[str, Fraction]


def fmt(x: Fraction) -> str:
    """Canonical answer string: "42" or "7/12"."""
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


# --------------------------------------------------------------- divisibility

def divisibility_medium(rng: random.Random) -> Problem:
    """Inclusion-exclusion with two divisors that may share factors."""
    a, b = rng.sample(range(3, 16), 2)
    n = rng.randint(1000, 9999)
    q = f"How many integers from 1 to {n} inclusive are divisible by {a} or by {b} (or both)?"
    return q, Fraction(n // a + n // b - n // math.lcm(a, b))


def divisibility_hard(rng: random.Random) -> Problem:
    """Two divisors in, one divisor out."""
    a, b, c = rng.sample(range(2, 13), 3)
    n = rng.randint(1000, 9999)
    count = sum(1 for k in range(1, n + 1) if (k % a == 0 or k % b == 0) and k % c)
    q = (f"How many integers from 1 to {n} inclusive are divisible by {a} or by {b}, "
         f"but not divisible by {c}?")
    return q, Fraction(count)


# --------------------------------------------------------------------- counting

def counting_medium(rng: random.Random) -> Problem:
    n, k = rng.randint(8, 16), rng.randint(3, 6)
    q = (f"A club has {n} members. In how many ways can a committee of {k} be chosen "
         f"if two particular members, Ana and Ben, refuse to serve together?")
    return q, Fraction(math.comb(n, k) - math.comb(n - 2, k - 2))


def counting_hard(rng: random.Random) -> Problem:
    letters = rng.sample("ABCDEFGHKMPRST", rng.randint(3, 4))
    counts = [rng.randint(1, 3) for _ in letters]
    lead = rng.randrange(len(letters))
    word = "".join(l * c for l, c in zip(letters, counts))
    word = "".join(rng.sample(word, len(word)))
    # arrangements that start with letters[lead]: drop one copy of it, permute the rest
    rest = counts.copy()
    rest[lead] -= 1
    total = math.factorial(sum(rest))
    for c in rest:
        total //= math.factorial(c)
    q = (f"How many distinct arrangements of all the letters in {word} "
         f"begin with the letter {letters[lead]}?")
    return q, Fraction(total)


# -------------------------------------------------------------------- modular

def modular_medium(rng: random.Random) -> Problem:
    """Exponent is beyond the calculator's limit, so the power must be reduced first."""
    a, m = rng.randint(2, 30), rng.choice([7, 9, 11, 13, 17, 19, 23, 25, 27, 29, 31, 37])
    if a % m == 0:
        a += 1
    n = rng.randint(100, 999)
    q = f"What is the remainder when {a}^{n} is divided by {m}?"
    return q, Fraction(pow(a, n, m))


def modular_hard(rng: random.Random) -> Problem:
    a = rng.choice([3, 7, 11, 13, 17, 19, 23])
    n = rng.randint(1000, 9999)
    m = rng.choice([100, 1000])
    q = f"What are the last {len(str(m)) - 1} digits of {a}^{n}? Give them as an integer (drop leading zeros)."
    return q, Fraction(pow(a, n, m))


# ------------------------------------------------------------------ sequences

def sequences_medium(rng: random.Random) -> Problem:
    """Sum of the terms that fall inside a range; first find which terms those are."""
    a1, d = rng.randint(-20, 30), rng.randint(3, 17)
    i, j = sorted(rng.sample(range(2, 12), 2))
    lo = rng.randint(50, 300)
    hi = lo + rng.randint(200, 900)
    terms = [a1 + k * d for k in range(0, (hi - a1) // d + 2)]
    q = (f"In an increasing arithmetic sequence, term number {i} is {a1 + (i - 1) * d} "
         f"and term number {j} is {a1 + (j - 1) * d}. What is the sum of all terms of the sequence "
         f"that are between {lo} and {hi} inclusive?")
    return q, Fraction(sum(t for t in terms if lo <= t <= hi))


def sequences_hard(rng: random.Random) -> Problem:
    p, r = rng.randint(1, 3), rng.choice([-2, -1, 1, 2])
    x1, x2 = rng.randint(1, 5), rng.randint(1, 7)
    n = rng.randint(10, 14)
    seq = [x1, x2]
    while len(seq) < n:
        seq.append(p * seq[-1] + r * seq[-2])
    rs = f"+ {r}" if r > 0 else f"- {-r}"
    q = (f"A sequence starts with a(1) = {x1} and a(2) = {x2}, and for n >= 3, "
         f"a(n) = {p}*a(n-1) {rs}*a(n-2). What is a({n})?")
    return q, Fraction(seq[-1])


# -------------------------------------------------------------------- algebra

def algebra_medium(rng: random.Random) -> Problem:
    """One equation already solved for y, so substitution works directly."""
    while True:
        x = rng.randint(-12, 12)
        m, k = rng.choice([i for i in range(-6, 7) if i]), rng.randint(-15, 15)
        a, b = (rng.choice([i for i in range(-9, 10) if i]) for _ in range(2))
        if a + b * m:
            break
    y = m * x + k
    ks = f"+ {k}" if k >= 0 else f"- {-k}"
    q = (f"Solve the system: y = {m}x {ks} and {a}x + {b}y = {a * x + b * y}. "
         f"What is x + y?").replace("+ -", "- ")
    return q, Fraction(x + y)


def algebra_hard(rng: random.Random) -> Problem:
    """General two-equation system; needs elimination."""
    while True:
        x, y = rng.randint(-12, 12), rng.randint(-12, 12)
        a, b, c, d = (rng.choice([i for i in range(-9, 10) if i]) for _ in range(4))
        if a * d - b * c:
            break
    q = (f"Solve the system: {a}x + {b}y = {a * x + b * y} and {c}x + {d}y = {c * x + d * y}. "
         f"What is x + y?").replace("+ -", "- ")
    return q, Fraction(x + y)


# ---------------------------------------------------------------- probability

def probability_medium(rng: random.Random) -> Problem:
    k = rng.randint(4, 10)
    hits = sum(1 for i in range(1, 7) for j in range(1, 7) if i + j >= k)
    q = f"Two fair six-sided dice are rolled. What is the probability that the sum is at least {k}? Give a reduced fraction."
    return q, Fraction(hits, 36)


def probability_hard(rng: random.Random) -> Problem:
    """Three draws, all different colours."""
    r, b, g = rng.randint(2, 8), rng.randint(2, 8), rng.randint(2, 8)
    q = (f"A bag has {r} red, {b} blue and {g} green marbles. Three marbles are drawn at random "
         f"without replacement. What is the probability that all three are different colours? "
         f"Give a reduced fraction.")
    return q, Fraction(r * b * g, math.comb(r + b + g, 3))


GENERATORS: Dict[str, Callable[[random.Random], Problem]] = {
    "divisibility-medium": divisibility_medium,
    "divisibility-hard": divisibility_hard,
    "counting-medium": counting_medium,
    "counting-hard": counting_hard,
    "modular-medium": modular_medium,
    "modular-hard": modular_hard,
    "sequences-medium": sequences_medium,
    "sequences-hard": sequences_hard,
    "algebra-medium": algebra_medium,
    "algebra-hard": algebra_hard,
    "probability-medium": probability_medium,
    "probability-hard": probability_hard,
}


# Tasks submitted in v1: the ones Qwen3.8-27B (thinking off) solved some but not all of the time.
V1_TASKS = [
    "counting-medium", "counting-hard",
    "modular-medium", "modular-hard",
    "sequences-medium", "sequences-hard",
    "probability-medium", "probability-hard",
]
