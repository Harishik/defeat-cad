"""Original 96 BPM scores for the DeFeat videos, synthesised with numpy (royalty-free by construction).

96 BPM -> one beat = 0.625 s, one bar = 2.5 s = 75 frames at 30 fps, so every cut lands on a beat.
    python video/music.py          # writes video/public/intro.wav and video/public/guide.wav
"""
from __future__ import annotations

import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, lfilter

SR = 44100
BPM = 96
BEAT = 60 / BPM
BAR = 4 * BEAT
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")

# A minor: Am9 - Fmaj7 - Cmaj7(add9) - Em7, one chord per bar
def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)

CHORDS = [
    [57, 60, 64, 67, 71],   # Am9   A C E G B
    [53, 57, 60, 64, 67],   # Fmaj7 F A C E (+G)
    [48, 55, 60, 64, 74],   # Cadd9 C G C E D
    [52, 55, 59, 62, 66],   # Em7   E G B D F#
]
ROOTS = [33, 29, 36, 28]   # bass roots (A1, F1, C2, E1)
rng = np.random.default_rng(7)


def t_of(seconds):
    return np.arange(int(seconds * SR)) / SR


def env_adsr(n, a, d, s, r, sr=SR):
    a, d, r = int(a * sr), int(d * sr), int(r * sr)
    e = np.full(n, s, dtype=np.float64)
    a = min(a, n); e[:a] = np.linspace(0, 1, a, endpoint=False)
    d = min(d, n - a); e[a:a + d] = np.linspace(1, s, d, endpoint=False)
    if r and n > r:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def lowpass(x, fc, order=2):
    b, a = butter(order, min(fc / (SR / 2), 0.99), "low")
    return lfilter(b, a, x)


def highpass(x, fc, order=2):
    b, a = butter(order, fc / (SR / 2), "high")
    return lfilter(b, a, x)


def bandpass(x, lo, hi):
    b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], "band")
    return lfilter(b, a, x)


def saw(f, t, phase=0.0):
    return 2 * ((f * t + phase) % 1.0) - 1


# ---------------------------------------------------------------- instruments
def pad_bar(chord, seconds, bright=0.5):
    t = t_of(seconds)
    out = np.zeros_like(t)
    for n in chord:
        for det in (-0.09, 0.0, 0.11):
            out += saw(midi(n + det), t, rng.random())
    out /= len(chord) * 3
    out = lowpass(out, 500 + 2200 * bright)
    return out * env_adsr(len(t), 0.6, 0.4, 0.85, 0.5)


def pluck(note, seconds=0.5):
    t = t_of(seconds)
    f = midi(note)
    x = 0.6 * np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t) + 0.12 * saw(f, t)
    return lowpass(x, 3800) * np.exp(-t * 9)


def kick():
    t = t_of(0.45)
    f = 45 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    click = np.exp(-t * 300) * 0.4
    return (np.sin(ph) * np.exp(-t * 7.5) + click) * 0.95


def clap():
    t = t_of(0.35)
    n = rng.standard_normal(len(t))
    bursts = sum(np.exp(-np.clip(t - k * 0.011, 0, None) * 220) * (t >= k * 0.011) for k in range(3))
    tail = np.exp(-t * 18)
    return bandpass(n, 900, 5200) * (0.6 * bursts + 0.5 * tail) * 0.55


def hat(open_=False):
    t = t_of(0.28 if open_ else 0.06)
    n = highpass(rng.standard_normal(len(t)), 7500)
    return n * np.exp(-t * (14 if open_ else 70)) * (0.22 if open_ else 0.16)


def sub(note, seconds):
    t = t_of(seconds)
    f = midi(note)
    x = np.sin(2 * np.pi * f * t) + 0.18 * np.sin(2 * np.pi * 2 * f * t)
    return lowpass(x, 240) * env_adsr(len(t), 0.01, 0.1, 0.9, 0.05) * 0.55


def riser(seconds):
    t = t_of(seconds)
    n = rng.standard_normal(len(t))
    out = np.zeros_like(n)
    steps = 24
    seg = len(n) // steps
    for k in range(steps):  # sweeping band-pass
        fc = 400 * (12 ** (k / steps))
        sl = slice(k * seg, (k + 1) * seg if k < steps - 1 else len(n))
        out[sl] = bandpass(n[sl], fc * 0.7, min(fc * 1.4, SR / 2 - 100))
    return out * (t / seconds) ** 2 * 0.35


def impact():
    t = t_of(2.4)
    f = 32 + 60 * np.exp(-t * 6)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    air = lowpass(rng.standard_normal(len(t)), 2500) * np.exp(-t * 3) * 0.25
    return (boom + air) * 0.8


# ---------------------------------------------------------------- mix helpers
class Track:
    def __init__(self, seconds):
        self.n = int(seconds * SR) + SR * 3
        self.bus = {k: np.zeros(self.n) for k in ("pad", "bass", "drums", "arp", "fx")}

    def add(self, bus, x, at):
        i = int(at * SR)
        j = min(self.n, i + len(x))
        if j > i:
            self.bus[bus][i:j] += x[: j - i]


def reverb_ir(seconds=2.6, decay=3.2):
    t = t_of(seconds)
    ir = rng.standard_normal(len(t)) * np.exp(-t * decay)
    return lowpass(ir, 6000) * 0.12


def render(arr, bars, name, title_hits=()):
    """arr: dict section -> set of layers per bar index."""
    total = bars * BAR
    tr = Track(total)
    for b in range(bars):
        L = arr(b)
        t0 = b * BAR
        ch = CHORDS[b % 4]
        if "pad" in L:
            br = L.get("pad", 0.5)
            tr.add("pad", pad_bar(ch, BAR + 0.6, bright=br) * (0.16 + 0.42 * br), t0)
        if "bass" in L:
            for k in range(4):
                tr.add("bass", sub(ROOTS[b % 4], BEAT * 0.9), t0 + k * BEAT)
        if "kick" in L:
            for k in (0, 1, 2, 3) if L.get("kick") == "four" else (0, 2):
                tr.add("drums", kick(), t0 + k * BEAT)
        if "clap" in L:
            for k in (1, 3):
                tr.add("drums", clap(), t0 + k * BEAT)
        if "hats" in L:
            for k in range(8):
                tr.add("drums", hat(open_=(k % 2 == 1 and L.get("hats") == "open")), t0 + k * BEAT / 2)
        if "arp" in L:
            seq = [ch[0] + 12, ch[2] + 12, ch[1] + 12, ch[3] + 12, ch[4] + 12, ch[3] + 12, ch[2] + 12, ch[1] + 12]
            for k in range(16):
                tr.add("arp", pluck(seq[k % 8], 0.5) * (0.24 if k % 4 == 0 else 0.17), t0 + k * BEAT / 4)
        if "riser" in L:
            tr.add("fx", riser(BAR * L["riser"]), t0 + BAR - BAR * L["riser"] + BAR * (L["riser"] - 1))
        if "impact" in L:
            tr.add("fx", impact(), t0)
    # sidechain: duck pad/bass under every kick
    side = np.ones(tr.n)
    kick_env = np.abs(tr.bus["drums"])
    kick_env = lowpass(kick_env, 8)
    kick_env /= kick_env.max() + 1e-9
    side = 1 - 0.55 * np.clip(kick_env * 2.2, 0, 1)
    for k in ("pad", "bass"):
        tr.bus[k] *= side
    # delay on arp (dotted eighth), reverb on pad/arp/fx
    d = int(BEAT * 0.75 * SR)
    arp = tr.bus["arp"].copy()
    for i, g in enumerate((0.45, 0.25, 0.12), 1):
        arp[d * i:] += tr.bus["arp"][: tr.n - d * i] * g
    ir = reverb_ir()
    wet = fftconvolve(tr.bus["pad"] * 0.6 + arp * 0.5 + tr.bus["fx"] * 0.5, ir)[: tr.n]
    mix = tr.bus["pad"] + tr.bus["bass"] + tr.bus["drums"] + arp * 0.9 + tr.bus["fx"] + wet
    # stereo: slight Haas widening on the wet + arp
    wide = np.zeros(tr.n)
    sh = int(0.012 * SR)
    wide[sh:] = (wet + arp * 0.4)[:-sh]
    left = mix
    right = mix - (wet + arp * 0.4) + wide
    st = np.stack([left, right], 1)
    st = st[: int(total * SR) + int(1.5 * SR)]
    # fade out over the last bar, master glue, normalise to -1 dBFS
    fade = int(BAR * SR)
    st[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.5
    st = np.tanh(st * 1.25) / np.tanh(1.25)
    st /= np.abs(st).max() / 0.89
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    wavfile.write(path, SR, (st * 32767).astype(np.int16))
    print(f"{name}: {bars} bars, {total:.1f} s, written to {path}")


def intro_arr(b):
    # 38 bars = 95.0 s
    if b < 4:  return {"pad": 0.25, **({"impact": 1} if b == 0 else {})}
    if b < 8:  return {"pad": 0.4, "arp": 1, **({"riser": 1} if b == 7 else {})}
    if b < 16: return {"pad": 0.55, "arp": 1, "bass": 1, "kick": "half", "hats": "closed", **({"impact": 1} if b == 8 else {}),
                       **({"riser": 2} if b == 15 else {})}
    if b < 18: return {"pad": 0.3, "arp": 1, **({"riser": 1} if b == 17 else {})}
    if b < 30: return {"pad": 0.8, "arp": 1, "bass": 1, "kick": "four", "clap": 1, "hats": "open", **({"impact": 1} if b == 18 else {}),
                       **({"riser": 1} if b == 29 else {})}
    if b < 34: return {"pad": 0.35, "arp": 1, **({"impact": 1} if b == 30 else {})}
    return {"pad": 0.6, "bass": 1, "kick": "half", "hats": "closed", **({"impact": 1} if b == 34 else {})}


def guide_arr(b):
    # 64 bars = 160.0 s, calmer and steadier for a walkthrough
    if b < 4:  return {"pad": 0.25, **({"impact": 1} if b == 0 else {})}
    if b < 16: return {"pad": 0.45, "bass": 1, "kick": "half", "hats": "closed"}
    if b < 32: return {"pad": 0.55, "bass": 1, "kick": "half", "clap": 1, "hats": "closed", "arp": 1,
                       **({"riser": 1} if b == 31 else {})}
    if b < 36: return {"pad": 0.3, "arp": 1, **({"impact": 1} if b == 32 else {})}
    if b < 56: return {"pad": 0.7, "bass": 1, "kick": "four", "clap": 1, "hats": "open", "arp": 1,
                       **({"impact": 1} if b == 36 else {}), **({"riser": 1} if b == 55 else {})}
    if b < 60: return {"pad": 0.35, "arp": 1}
    return {"pad": 0.5, "bass": 1, "kick": "half", "hats": "closed", **({"impact": 1} if b == 60 else {})}


if __name__ == "__main__":
    render(intro_arr, 38, "intro.wav")
    render(guide_arr, 64, "guide.wav")
