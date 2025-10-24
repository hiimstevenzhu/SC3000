#!/usr/bin/env python3
"""
Generate DPO dataset of positve and negative pairs.

Writes a JSON list of objects with keys: "positive", "negative".
- Positives: "The answer is {y} because {short_reason}."
- Negatives: mixed: "I don't know", wrong answer, bad reasoning, poor formatting.

Usage examples:
  python gen_dpo_dataset.py --n 20000 --stage1_share 0.35 --out dpo/pos_neg_pairs.json
"""

import json
import random
import argparse
from typing import Tuple, Dict, List
from random import choice

RNG = random.Random(42)

IDK_POOL = [
    "Sorry, I don't know!",
    "I am not sure about that.",
    "I don't have enough information.",
    "Can't answer this question.",
    "I'm unable to compute that.",
    "I don't know.",
]


def rand_int(a: int, b: int) -> int:
    return RNG.randint(a, b)


def coin(p: float) -> bool:
    return RNG.random() < p

########################################################
# ---------- Problem generators ----------
""" Each returns a tuple (prompt, answer, reason) for each problem type.
    reason is a brief explanation of how the answer was obtained, e.g. "2+3 equals 5" or "x = 10/2 = 5" etc.
    This is used specifically in the positive explanation generation.
    The examples include basic arithematic, a paratheses mix and some linear equations that will solve for x in various rearrangements.
    The last function will generate a simple word problem involving 
"""

# Addition problem generator
def gen_add() -> Tuple[str, int, str]:
    digit_range = choice([(10, 99), (100, 999)])
    
    x, y = rand_int(*digit_range), rand_int(*digit_range)
    ans = x + y #did this to fix addition range as the pairs were not balanced
    
    prompt = f"{x}+{y}=?"
    reason = f"{x}+{y} equals {ans}"
    return prompt, ans, reason

# Subtraction problem generator
def gen_sub() -> Tuple[str, int, str]:
    x, y = rand_int(1, 999), rand_int(1, 999)
    ans = x - y
    prompt = f"{x}-{y}=?"
    reason = f"{x}-{y} equals {ans}"
    return prompt, ans, reason

# Multiplication problem generator
def gen_mul() -> Tuple[str, int, str]:
    x, y = rand_int(1, 99), rand_int(1, 99)
    ans = x * y
    prompt = f"{x}*{y}=?"
    reason = f"{x}*{y} equals {ans}"
    return prompt, ans, reason

# Division problem generator
def gen_div() -> Tuple[str, int, str]:
    y = rand_int(1, 99)
    ans = rand_int(1, 99)
    x = y * ans
    prompt = f"{x}/{y}=?"
    reason = f"{x}/{y} equals {ans}"
    return prompt, ans, reason

# Paren mix problem generator
def gen_paren_mix() -> Tuple[str, int, str]:
    # (a + b) * c - d
    a, b, c, d = (rand_int(1, 20) for _ in range(4))
    ans = (a + b) * c - d
    prompt = f"({a}+{b})*{c}-{d}=?"
    reason = f"({a}+{b})*{c}-{d} equals {ans}"
    return prompt, ans, reason

# Linear equation generators
def gen_eq_ax_eq_b() -> Tuple[str, int, str]:
    # a*x = b  (a != 0)
    a = rand_int(1, 12)
    x = rand_int(-50, 50)
    b = a * x
    prompt = f"{a}*x={b}, x=?"
    ans = x
    reason = f"x = {b}/{a} = {x}"
    return prompt, ans, reason

# Linear equation: a*x + b = c
def gen_eq_ax_plus_b_eq_c() -> Tuple[str, int, str]:
    # a*x + b = c
    a = rand_int(1, 12)
    x = rand_int(-30, 30)
    b = rand_int(-50, 50)
    c = a * x + b
    prompt = f"{a}*x+{b}={c}, x=?"
    ans = x
    reason = f"x = ({c}-{b})/{a} = {x}"
    return prompt, ans, reason

# Linear equation: c - x = b
def gen_eq_c_minus_x_eq_b():
    # c - x = b  -> x = c - b
    b = rand_int(-50, 50)
    x = rand_int(-50, 50)
    c = b + x
    prompt = f"{c}-x={b}, x=?"
    ans = c - b  # equals x
    reason = f"{c}-{x}={b}, so x={c}-{b}={ans}"
    return prompt, ans, reason

# Linear equation: x + a = b
def gen_eq_x_plus_a_eq_b():
    # x + a = b -> x = b - a
    a = rand_int(-30, 30)
    x = rand_int(-50, 50)
    b = x + a
    prompt = f"x+{a}={b}, x=?"
    ans = b - a
    reason = f"{ans}+{a}={b}, so x={b}-{a}={ans}"
    return prompt, ans, reason

# Linear equation: x - a = b
def gen_eq_x_minus_a_eq_b():
    # x - a = b -> x = b + a
    a = rand_int(-30, 30)
    x = rand_int(-50, 50)
    b = x - a
    prompt = f"x-{a}={b}, x=?"
    ans = b + a
    reason = f"{ans}-{a}={b}, so x={b}+{a}={ans}"
    return prompt, ans, reason

# Linear equation: a / x = b
def gen_eq_a_over_x_eq_b():
    b = rand_int(1, 12)
    x = rand_int(1, 50)
    a = b * x
    prompt = f"{a}/x={b}, x=?"
    ans = x
    # Reason: keep simple, avoid // and avoid embedding wrong patterns
    reason = f"x = {a}/{b} = {x}"
    return prompt, ans, reason

# Linear equation: x * a = b
def gen_eq_x_times_a_eq_b() -> Tuple[str, int, str]:
    # x * a = b  -> x = b / a  (integer)
    a = rand_int(1, 12)
    x = rand_int(-50, 50)
    b = a * x
    prompt = f"x*{a}={b}, x=?"
    ans = x
    reason = f"x = {b}/{a} = {x}"
    return prompt, ans, reason

# Linear equation: x / a = b
def gen_eq_x_over_a_eq_b() -> Tuple[str, int, str]:
    # x / a = b  -> x = a * b
    a = rand_int(1, 12)
    b = rand_int(-30, 30)
    x = a * b
    prompt = f"x/{a}={b}, x=?"
    ans = x
    reason = f"x = {a}*{b} = {x}"
    return prompt, ans, reason

# Linear equation: a + x = b
def gen_eq_a_plus_x_eq_b() -> Tuple[str, int, str]:
    # a + x = b  -> x = b - a
    a = rand_int(-30, 30)
    x = rand_int(-50, 50)
    b = a + x
    prompt = f"{a}+x={b}, x=?"
    ans = b - a
    reason = f"x = {b}-{a} = {ans}"
    return prompt, ans, reason

# Linear equation: a - x = b
def gen_eq_a_minus_x_eq_b() -> Tuple[str, int, str]:
    # a - x = b  -> x = a - b
    a = rand_int(-30, 30)
    x = rand_int(-50, 50)
    b = a - x
    prompt = f"{a}-x={b}, x=?"
    ans = a - b
    reason = f"x = {a}-{b} = {ans}"
    return prompt, ans, reason

# Word problem generator
def gen_word_sum() -> Tuple[str, int, str]:
    a, b = rand_int(1, 50), rand_int(1, 50)
    ans = a + b
    prompt = f"If Tom has {a} apples and buys {b} more, how many apples does he have in total?"
    reason = f"{a}+{b} equals {ans}"
    return prompt, ans, reason


GENS = [
    ("add",        gen_add,                 0.11),
    ("sub",        gen_sub,                 0.11),
    ("mul",        gen_mul,                 0.11),
    ("div",        gen_div,                 0.11),

    ("eq1",        gen_eq_ax_eq_b,          0.06),  # a*x=b
    ("eq1b",       gen_eq_x_times_a_eq_b,   0.06),  # x*a=b

    ("eq2",        gen_eq_ax_plus_b_eq_c,   0.06),  # a*x+b=c

    ("a_over_x",   gen_eq_a_over_x_eq_b,    0.06),  # a/x=b
    ("x_over_a",   gen_eq_x_over_a_eq_b,    0.06),  # x/a=b

    ("c_minus_x",  gen_eq_c_minus_x_eq_b,   0.06),  # c - x = b
    ("x_plus_a",   gen_eq_x_plus_a_eq_b,    0.05),  # x + a = b
    ("a_plus_x",   gen_eq_a_plus_x_eq_b,    0.05),  # a + x = b
    ("x_minus_a",  gen_eq_x_minus_a_eq_b,   0.05),  # x - a = b
    ("a_minus_x",  gen_eq_a_minus_x_eq_b,   0.05),  # a - x = b
]


# Sample a problem according to weighted probabilities(for each GEN type)
# probs sum to 1.0
def sample_problem() -> Tuple[str, int, str]:
    r = RNG.random()
    acc = 0.0
    for _, fn, p in GENS:
        acc += p
        if r <= acc:
            return fn()
    return gen_add()

########################################################
# ---------- Negative constructors ----------
"""
Each returns a negative explanation string given the true answer and prompt.
We created helpers that dont just return "Sorry, I don't know", but vary the negative responses
(e.g. wrong answer, messy explanation, contradicting reasoning, poor formatting).
Doing this to increase the diversity of negative samples and help the model learn better.
Theoretically should help improve performance, generalization and ability to learn.
"""

# Each returns a negative explanation string given the true answer and prompt.
def wrong_answer_nearby(ans: int) -> int:
    # pick a wrong answer close to true one
    delta = RNG.choice([-3, -2, -1, 1, 2, 3, 4, 5, -4, -5])
    wrong = ans + delta
    if wrong == ans:
        wrong += 1
    return wrong

# Messy explanation but correct answer
def messy_explanation(ans: int, prompt: str) -> str:
    # keep correct answer but explanation is messy/contradictory
    choices = [
        f"The answer is {ans} because math is hard lol.",
        f"Answer: {ans}. Because {prompt} obviously equals {ans}, duh.",
        f"{ans}.",  # no explanation
    ]
    return RNG.choice(choices)

# Contradicting reasoning: correct answer but wrong reason, or vice-versa
def contradicting_reasoning(ans: int, prompt: str) -> str:
    wrong = wrong_answer_nearby(ans)
    # give correct reasoning but wrong final, or vice-versa
    if coin(0.5):
        return f"The answer is {wrong} because {prompt} evaluates to {ans}."
    else:
        return f"The answer is {ans} because {prompt} evaluates to {wrong}."

# Poor formatting & wrong result
def poor_format_wrong_result(ans: int, prompt: str) -> str:
    wrong = wrong_answer_nearby(ans)
    # ugly formatting & vague reason
    return f"Ans={wrong}. {prompt} idk probly."

# Random "I don't know" response
def idk_response() -> str:
    return RNG.choice(IDK_POOL)

########################################################
#---------- Dataset construction ----------
"""
Build dataset of positive/negative pairs.
Stage 1: "I don't know" negatives to teach it deferral and to attempt.
Stage 2: introduces the other negative types with custom weights.
Each loop samples a problem, formats the positive explanation, then attaches the negative variant. 
It stores the final strings as { "negative": "...", "positive": "..." } and shuffles the result so the stages mix together.

The weights for Stage 2 negative types are:
    - "wrong": 0.5
        - Most common failure mode, so the model learns to favour numerically correct results.
    - "messy": 0.15
        - Encourages clearer reasoning and better formatting. Not too high weight to avoid confusion.
    - "contradict": 0.2
        - makes this second-most frequent since it’s a subtle error the model should learn to reject decisively.
    - "poorfmt": 0.15
        - users still see poorly formatted output, but stays a minority class so it isnt dominated by formatting quirks.
"""

# Positive explanation
def make_positive(ans: int, reason: str) -> str:
    return f"The answer is {ans} because {reason}."

# Negative explanation
def make_negative(ans: int, prompt: str, kind: str) -> str:
    if kind == "idk":
        return idk_response()
    elif kind == "wrong":
        return f"The answer is {wrong_answer_nearby(ans)} because I think so."
    elif kind == "messy":
        return messy_explanation(ans, prompt)
    elif kind == "contradict":
        return contradicting_reasoning(ans, prompt)
    elif kind == "poorfmt":
        return poor_format_wrong_result(ans, prompt)
    else:
        return idk_response()


NEG_KIND_STAGE1 = ["idk"]  # Stage 1 = teach it to attempt
NEG_KIND_STAGE2 = [
    "wrong",
    "messy",
    "contradict",
    "poorfmt",
]  # Stage 2 = correctness & format


def build_dataset(n_total: int, stage1_share: float) -> List[Dict[str, str]]:
    n_stage1 = int(n_total * stage1_share)
    n_stage2 = n_total - n_stage1
    out: List[Dict[str, str]] = []

    # Stage 1: “I don’t know” style negatives
    for _ in range(n_stage1):
        prompt, ans, reason = sample_problem()
        pos = make_positive(ans, reason)
        neg = make_negative(ans, prompt, RNG.choice(NEG_KIND_STAGE1))
        out.append({"negative": f"{prompt} {neg}", "positive": f"{prompt} {pos}"})

    # Stage 2: incorrect math & messy formatting negatives

    for _ in range(n_stage2):
        prompt, ans, reason = sample_problem()
        pos = make_positive(ans, reason)
        kind = RNG.choices(NEG_KIND_STAGE2, weights=[0.5, 0.15, 0.2, 0.15], k=1)[0]
        neg = make_negative(ans, prompt, kind)
        out.append({"negative": f"{prompt} {neg}", "positive": f"{prompt} {pos}"})

    # small shuffle to mix stages
    RNG.shuffle(out)
    return out

########################################################
#---------- Main function ----------

# Generate dataset of chosen number of data points and write to JSON
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400000, help="total number of pairs")
    ap.add_argument(
        "--stage1_share",
        type=float,
        default=0.35,
        help="fraction of Stage-1 'idk' negatives",
    )
    ap.add_argument(
        "--out", type=str, default="dpo/pos_neg_pairs.json", help="output JSON path"
    )
    args = ap.parse_args()

    data = build_dataset(args.n, args.stage1_share)

    print(f"[debug] total: {len(data)}")

    # quick validation: ensure negatives are not trivially equal to positives
    bad = sum(1 for ex in data if ex["negative"] == ex["positive"])
    if bad:
        print(f"[warn] {bad} pairs had identical pos/neg (should be 0).")

    # ensure JSON array (list of dicts with strings)
    for i, ex in enumerate(data[:5]):
        assert isinstance(ex, dict) and "positive" in ex and "negative" in ex, i
        assert isinstance(ex["positive"], str) and isinstance(ex["negative"], str), i

    # write
    out_path = args.out
    # make sure directory exists
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} pairs to {out_path}")


if __name__ == "__main__":
    main()
