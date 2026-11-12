"""Scores the assistant on the fourth held-out set (bench/heldout_v4.py).

    python -m bench.assistant_v4 --tag before     # recorded before the verification pass existed
    python -m bench.assistant_v4 --tag after
"""
from __future__ import annotations

import argparse

from bench.assistant import text_to_cad, tool_calling
from bench.common import save
from bench.heldout_v4 import COMMANDS_V4, PARTS_V4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-cad", action="store_true")
    a = ap.parse_args()
    rep = {"tool_calling": tool_calling(COMMANDS_V4, "v4")}
    if not a.no_cad:
        rep["text_to_cad"] = text_to_cad(PARTS_V4)
    print(save(f"assistant_v4_{a.tag}", rep))
    for k, v in rep["tool_calling"].items():
        print(k, {x: v[x] for x in ("exact_match", "exact_match_ci95", "call_f1", "median_latency_s")})
    t = rep.get("text_to_cad")
    if t:
        print("cad", {x: t[x] for x in ("valid_solid_rate", "dimension_match_rate", "dimension_match_ci95", "median_seconds")})


if __name__ == "__main__":
    main()
