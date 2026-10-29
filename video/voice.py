"""Narration for the DeFeat videos with Kokoro (open-weight, Apache-2.0), one WAV per line.

    video/.tts/bin/python video/voice.py            # writes video/public/vo/<video>_<id>.wav + vo.json
Needs video/tts_models/kokoro-v1.0.onnx and voices-v1.0.bin from
https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
"""
from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "public", "vo")


def trim(x, sr, thr=0.008, pad=0.03):
    """Cut leading/trailing silence, keep a short natural tail."""
    idx = np.where(np.abs(x) > thr)[0]
    if not len(idx):
        return x
    a, b = max(0, idx[0] - int(pad * sr)), min(len(x), idx[-1] + int(pad * 2 * sr))
    return x[a:b]


def main():
    plan = json.load(open(os.path.join(HERE, "plan.json")))
    k = Kokoro(os.path.join(HERE, "tts_models", "kokoro-v1.0.onnx"), os.path.join(HERE, "tts_models", "voices-v1.0.bin"))
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for video in ("intro", "guide"):
        for ln in plan[video]["lines"]:
            x, sr = k.create(ln["text"], voice=ln.get("voice", plan["voice"]), speed=ln.get("speed", plan["speed"]), lang="en-us")
            x = trim(np.asarray(x, dtype=np.float32), sr)
            path = os.path.join(OUT, f"{video}_{ln['id']}.wav")
            sf.write(path, x, sr)
            meta[f"{video}_{ln['id']}"] = {"sr": sr, "seconds": round(len(x) / sr, 3)}
            print(f"{video}_{ln['id']}: {len(x) / sr:.2f}s")
    json.dump(meta, open(os.path.join(OUT, "vo.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
