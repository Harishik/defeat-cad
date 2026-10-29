"""Original 96 BPM scores, sound design and the final narrated mix for the DeFeat videos.

Everything is synthesised with numpy (royalty-free by construction). The narration (video/voice.py) is placed
on the bar grid from plan.json, the music ducks under it, and the timing is exported to src/cues.ts so every
cut, caption and sound effect lands on the same frame.

    python video/score.py        # writes public/intro.wav, public/guide.wav and src/cues.ts
"""
from __future__ import annotations

import json
import os
import re
import subprocess

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, filtfilt, lfilter, resample_poly

SR = 44100
BPM = 96
BEAT = 60 / BPM
BAR = 4 * BEAT
FPS = 30
HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, "public")
rng = np.random.default_rng(7)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def t_of(seconds):
    return np.arange(int(seconds * SR)) / SR


def lowpass(x, fc, order=2):
    b, a = butter(order, min(fc / (SR / 2), 0.99), "low")
    return lfilter(b, a, x, axis=0)


def highpass(x, fc, order=2):
    b, a = butter(order, fc / (SR / 2), "high")
    return lfilter(b, a, x, axis=0)


def bandpass(x, lo, hi, order=2):
    b, a = butter(order, [lo / (SR / 2), min(hi / (SR / 2), 0.99)], "band")
    return lfilter(b, a, x, axis=0)


def env(n, a, r, hold=None):
    """Attack / sustain / release envelope over n samples."""
    a, r = max(1, int(a * SR)), max(1, int(r * SR))
    e = np.ones(n)
    e[:min(a, n)] = np.linspace(0, 1, min(a, n))
    if n > r:
        e[-r:] *= np.linspace(1, 0, r) ** 1.5
    return e


def pan(x, p):
    """Equal-power pan, p in [-1, 1] (scalar or per-sample array)."""
    th = (np.asarray(p) + 1) * np.pi / 4
    return np.stack([x * np.cos(th), x * np.sin(th)], 1)


def svf(x, fc, q=0.7):
    """Time-varying state-variable band-pass (fc per sample). Used for whooshes and sweeps."""
    fc = np.broadcast_to(np.asarray(fc, dtype=float), x.shape)
    f = 2 * np.sin(np.pi * np.clip(fc, 20, SR / 6) / SR)
    lo = bp = 0.0
    out = np.empty_like(x)
    damp = 1 / q
    for i in range(len(x)):
        hi = x[i] - lo - damp * bp
        bp += f[i] * hi
        lo += f[i] * bp
        out[i] = bp
    return out


# ---------------------------------------------------------------- instruments
CHORDS = {
    "Am9": [57, 60, 64, 67, 71], "Fmaj7": [53, 57, 60, 64, 67], "Cadd9": [48, 55, 60, 64, 74], "Em7": [52, 55, 59, 62, 66],
    "Dm9": [50, 53, 57, 60, 64], "G6": [55, 59, 62, 64, 67], "Esus": [52, 57, 59, 64, 71],
}
ROOT = {"Am9": 33, "Fmaj7": 29, "Cadd9": 36, "Em7": 28, "Dm9": 38, "G6": 31, "Esus": 28}
PROG = ["Am9", "Fmaj7", "Cadd9", "Em7"]
LIFT = ["Fmaj7", "G6", "Am9", "Esus"]


def pad(chord, seconds, bright=0.5):
    """Wide analog pad: detuned saws per side, slow filter drift."""
    t = t_of(seconds)
    sides = []
    for side in (0, 1):
        x = np.zeros_like(t)
        for n in chord:
            for det in ((-0.08, 0.05) if side == 0 else (0.09, -0.04)):
                f = midi(n + det)
                x += 2 * ((f * t + rng.random()) % 1.0) - 1
        x /= len(chord) * 2
        sides.append(lowpass(x, 420 + 2400 * bright))
    st = np.stack(sides, 1)
    return st * env(len(t), 1.2, 1.4)[:, None]


def strings(chord, seconds, level=1.0):
    """Higher, slower ensemble swell for lifts."""
    t = t_of(seconds)
    x = np.zeros_like(t)
    for n in chord[1:]:
        for det in (-0.12, 0.0, 0.13):
            f = midi(n + 12 + det) * (1 + 0.003 * np.sin(2 * np.pi * 5.2 * t + rng.random() * 6))
            x += 2 * ((np.cumsum(f) / SR) % 1.0) - 1
    x = bandpass(x / (len(chord) * 3), 300, 5200)
    e = np.sin(np.pi * np.clip(t / seconds, 0, 1)) ** 1.4
    return pan(x * e * level, 0.0) * 0.9


def felt(note, seconds=3.0, vel=1.0):
    """Felt piano: inharmonic partials, soft hammer, quick top-end decay."""
    t = t_of(seconds)
    f = midi(note)
    x = np.zeros_like(t)
    for k in range(1, 9):
        fk = f * k * np.sqrt(1 + 0.0004 * k * k)
        x += (0.9 / k ** 1.3) * np.sin(2 * np.pi * fk * t) * np.exp(-t * (1.1 + 0.9 * k))
    hammer = lowpass(rng.standard_normal(len(t)), 1800) * np.exp(-t * 90) * 0.15
    x = lowpass(x + hammer, 2600) * env(len(t), 0.004, 0.4) * vel
    return pan(x, np.clip((note - 64) / 30, -0.5, 0.5))


def pluck(note, seconds=0.6, vel=1.0):
    t = t_of(seconds)
    f = midi(note)
    x = 0.6 * np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t) + 0.1 * (2 * ((f * t) % 1) - 1)
    return lowpass(x, 3600) * np.exp(-t * 8) * vel


def sub(note, seconds):
    t = t_of(seconds)
    f = midi(note)
    x = np.sin(2 * np.pi * f * t) + 0.2 * np.sin(2 * np.pi * 2 * f * t)
    return pan(lowpass(x, 220) * env(len(t), 0.01, 0.06) * 0.55, 0)


def pulse_bass(note, seconds):
    """Filtered saw eighths for drive."""
    t = t_of(seconds)
    f = midi(note + 12)
    x = 2 * ((f * t) % 1) - 1
    x = lowpass(x, 650) * np.exp(-t * 7)
    return pan(x * 0.32, 0)


def kick(v=1.0):
    t = t_of(0.5)
    f = 42 + 120 * np.exp(-t * 30)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) + np.exp(-t * 400) * 0.3
    return pan(x * 0.9 * v, 0)


def rim(v=1.0):
    t = t_of(0.2)
    x = bandpass(rng.standard_normal(len(t)), 1400, 6000) * np.exp(-t * 45) + 0.4 * np.sin(2 * np.pi * 1700 * t) * np.exp(-t * 80)
    return pan(x * 0.32 * v, 0.1)


def clap(v=1.0):
    t = t_of(0.35)
    n = rng.standard_normal(len(t))
    bursts = sum(np.exp(-np.clip(t - k * 0.011, 0, None) * 220) * (t >= k * 0.011) for k in range(3))
    return pan(bandpass(n, 900, 5200) * (0.6 * bursts + 0.5 * np.exp(-t * 18)) * 0.42 * v, -0.05)


def shaker(v=1.0, p=0.3):
    t = t_of(0.09)
    x = highpass(rng.standard_normal(len(t)), 6500) * np.sin(np.pi * t / 0.09) ** 2
    return pan(x * 0.13 * v, p)


# ---------------------------------------------------------------- sound design
def whoosh(seconds=1.1, lo=250, hi=3800, direction=1, level=1.0):
    t = t_of(seconds)
    u = t / seconds
    shape = np.sin(np.pi * u) ** 2
    fc = lo + (hi - lo) * np.sin(np.pi * u) ** 1.5
    x = svf(rng.standard_normal(len(t)), fc, 1.4) * shape
    return pan(x * 0.55 * level, direction * (u * 1.6 - 0.8))


def swell(seconds=2.4, level=1.0):
    """Reverse-cymbal style swell into a hit."""
    t = t_of(seconds)
    x = highpass(rng.standard_normal(len(t)), 3000) * (t / seconds) ** 3
    x += svf(rng.standard_normal(len(t)), 300 + 5000 * (t / seconds) ** 2, 2.0) * (t / seconds) ** 2 * 0.6
    return np.stack([x, np.roll(x, 300)], 1) * 0.3 * level


def boom(level=1.0, seconds=3.2):
    """Cinematic low hit with an airy tail."""
    t = t_of(seconds)
    f = 30 + 70 * np.exp(-t * 7)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.7)
    x += lowpass(rng.standard_normal(len(t)), 900) * np.exp(-t * 3.5) * 0.35
    x += bandpass(rng.standard_normal(len(t)), 2000, 9000) * np.exp(-t * 9) * 0.12
    return pan(np.tanh(x * 1.4) * 0.85 * level, 0)


def braam(chord, level=1.0, seconds=4.0):
    """Brassy low swell for the title reveal."""
    t = t_of(seconds)
    x = np.zeros_like(t)
    for n in (chord[0] - 12, chord[0], chord[2] - 12):
        for det in (-0.1, 0.1):
            f = midi(n + det)
            x += 2 * ((f * t + rng.random()) % 1) - 1
    cut = 200 + 1600 * np.exp(-t * 1.4)
    x = np.tanh(svf(x / 6, cut, 0.9) * 3) * np.exp(-t * 0.9) * env(len(t), 0.02, 1.2)
    return np.stack([x, np.roll(x, 400)], 1) * 0.55 * level


def shimmer(seconds=3.0, root=81, level=1.0):
    t = t_of(seconds)
    x = sum(np.sin(2 * np.pi * midi(root + iv) * t + rng.random() * 6) * (0.5 + 0.5 * np.sin(2 * np.pi * (0.7 + k * 0.3) * t))
            for k, iv in enumerate((0, 7, 12, 16, 19)))
    x *= np.sin(np.pi * t / seconds) ** 2 * 0.06 * level
    return np.stack([x, np.roll(x, 900)], 1)


def click(level=1.0):
    t = t_of(0.06)
    x = np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 160) * 0.6 + highpass(rng.standard_normal(len(t)), 4000) * np.exp(-t * 300) * 0.4
    return pan(x * 0.45 * level, 0.15)


def tick(level=1.0, pitch=3200):
    t = t_of(0.04)
    x = np.sin(2 * np.pi * pitch * t) * np.exp(-t * 220)
    return pan(x * 0.22 * level, 0.25)


def chime(note=88, level=1.0):
    t = t_of(2.5)
    x = sum(np.sin(2 * np.pi * midi(note) * r * t) * np.exp(-t * d) * a for r, d, a in ((1, 2.2, 0.6), (2.76, 4.0, 0.25), (5.4, 7.0, 0.1)))
    return pan(x * 0.18 * level, 0.2)


def glitch(seconds=0.5, level=1.0):
    t = t_of(seconds)
    x = np.sign(np.sin(2 * np.pi * rng.uniform(80, 600) * t)) * (rng.random(len(t)) > 0.6)
    gate = (np.floor(t * 40) % 3 == 0).astype(float)
    return pan(bandpass(x * gate, 600, 7000) * np.exp(-t * 4) * 0.14 * level, rng.uniform(-0.6, 0.6))


def typing(seconds, level=1.0):
    out = np.zeros((int(seconds * SR) + SR // 10, 2))
    tt = 0.0
    while tt < seconds:
        i = int(tt * SR)
        k = tick(0.5 * level, rng.uniform(2200, 3600))
        out[i:i + len(k)] += k
        tt += rng.uniform(0.045, 0.11)
    return out


# ---------------------------------------------------------------- mix
class Mix:
    def __init__(self, seconds):
        self.n = int(seconds * SR) + SR * 4
        self.bus = {k: np.zeros((self.n, 2)) for k in ("pad", "keys", "bass", "drums", "fx", "vo")}

    def add(self, bus, x, at):
        if x.ndim == 1:
            x = pan(x, 0)
        i = int(round(at * SR))
        if i < 0:
            x, i = x[-i:], 0
        j = min(self.n, i + len(x))
        if j > i:
            self.bus[bus][i:j] += x[: j - i]


def reverb(x, seconds=3.0, decay=2.6, pre=0.02, tone=5200):
    t = t_of(seconds)
    out = []
    for s in (0, 1):
        ir = rng.standard_normal(len(t)) * np.exp(-t * decay)
        ir = lowpass(ir, tone)
        ir = np.concatenate([np.zeros(int(pre * SR)), ir]) * 0.09
        out.append(fftconvolve(x[:, s], ir)[: len(x)])
    return np.stack(out, 1)


def follower(x, attack=0.02, release=0.35):
    """Smoothed amplitude envelope (mono)."""
    a = np.abs(x).mean(1) if x.ndim == 2 else np.abs(x)
    a = lowpass(a, 12, 1)
    up = lfilter([1 - np.exp(-1 / (attack * 200))], [1, -np.exp(-1 / (attack * 200))], a[::220])
    # peak hold with release on the decimated signal
    rel = np.exp(-1 / (release * 200))
    e = np.zeros_like(up)
    for i in range(len(up)):
        e[i] = max(up[i], e[i - 1] * rel if i else 0)
    return np.interp(np.arange(len(a)), np.arange(len(e)) * 220, e)


def compress(x, thr=0.12, ratio=3.0):
    e = follower(x, 0.005, 0.12) + 1e-6
    g = np.where(e > thr, (thr + (e - thr) / ratio) / e, 1.0)
    return x * g[:, None]


def voice(path):
    """Clean, dry narration: rumble filter and gentle levelling only. No reverb and no parallel EQ copies,
    which smear the phase and make a voice sound hollow, as if it were in a bowl."""
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.dtype.kind == "i" or np.abs(x).max() > 2:
        x /= 32768.0
    if x.ndim > 1:
        x = x.mean(1)
    x = resample_poly(x, SR, sr) if sr != SR else x
    b, a = butter(2, 70 / (SR / 2), "high")
    x = filtfilt(b, a, x)
    st = pan(x, 0)
    st = compress(st, 0.14, 2.0)
    return st / (np.abs(st).max() + 1e-9) * 0.8


def master(m, total, name, duck=0.6):
    vo = m.bus["vo"]
    e = follower(vo, 0.03, 0.3)
    e = np.clip(e / (e.max() + 1e-9) * 3.0, 0, 1)
    e = np.concatenate([e[int(0.12 * SR):], np.zeros(int(0.12 * SR))])   # look-ahead: duck just before each line
    side_vo = 1 - duck * e
    # kick sidechain on pads/bass
    ke = follower(m.bus["drums"], 0.002, 0.18)
    ke = np.clip(ke / (ke.max() + 1e-9) * 2.0, 0, 1)
    side_k = 1 - 0.4 * ke
    for k in ("pad", "bass"):
        m.bus[k] *= (side_vo * side_k)[:, None]
    for k in ("keys", "drums"):
        m.bus[k] *= side_vo[:, None]
    m.bus["fx"] *= (1 - 0.35 * e)[:, None]
    music = m.bus["pad"] + m.bus["keys"] * 0.9 + m.bus["bass"] + m.bus["drums"] + m.bus["fx"]
    wet = reverb(m.bus["pad"] * 0.5 + m.bus["keys"] * 0.7 + m.bus["fx"] * 0.6, 3.4, 2.2)
    music = music + wet
    music = np.tanh(music * 1.1) / np.tanh(1.1)
    # set the music level so the narration sits ~11 dB above it while it speaks
    act = e > 0.3
    r = lambda a: np.sqrt((a[act] ** 2).mean()) + 1e-9
    g = r(m.bus["vo"]) / r(music) / 10 ** (11 / 20)
    out = music * g + m.bus["vo"]
    print(f"  music gain {g:.2f}; voice over music: {20 * np.log10(r(m.bus['vo']) / r(music * g)):+.1f} dB")
    out = out[: int(total * SR) + int(0.5 * SR)]
    fade = int(BAR * 0.8 * SR)
    out[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.6
    out /= np.abs(out).max() / 0.9
    raw = os.path.join(HERE, "out", f"{name}_raw.wav")
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    wavfile.write(raw, SR, (out * 32767).astype(np.int16))
    dst = os.path.join(PUB, f"{name}.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-af", "loudnorm=I=-15:TP=-1.2:LRA=9", "-ar", str(SR), dst], check=True)
    print(f"{name}.wav: {total:.1f} s, loudness-normalised to -15 LUFS")


# ---------------------------------------------------------------- placement
def place_lines(video, lines, meta):
    """Narration on the bar grid; never overlapping, with a breath between lines."""
    out, prev_end = {}, 0.0
    for ln in lines:
        start = max(ln["bar"] * BAR, prev_end + 0.35)
        if start > ln["bar"] * BAR + 0.05:
            print(f"  ! {video}.{ln['id']} pushed {start - ln['bar'] * BAR:.2f}s later than planned")
        dur = meta[f"{video}_{ln['id']}"]["seconds"]
        # sentence-level subtitle timing, proportional to characters
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ln["text"]) if s.strip()]
        total_c = sum(len(s) for s in sents)
        t, subs = start, []
        for s in sents:
            d = dur * len(s) / total_c
            subs.append({"text": s, "s": round(t * FPS), "e": round((t + d) * FPS)})
            t += d
        out[ln["id"]] = {"s": round(start * FPS), "e": round((start + dur) * FPS), "start": start, "dur": dur,
                         "text": ln["text"], "subs": subs, "chapter": ln.get("chapter")}
        prev_end = start + dur
    return out, prev_end


def lay_music(m, bars, arr):
    for b in range(bars):
        L = arr(b)
        if not L:
            continue
        t0 = b * BAR
        name = L.get("chord") or PROG[b % 4]
        ch = CHORDS[name]
        if "pad" in L:
            m.add("pad", pad(ch, BAR + 1.4, L["pad"]) * (0.14 + 0.36 * L["pad"]), t0)
        if "strings" in L:
            m.add("pad", strings(ch, BAR * 2 + 0.5, L["strings"]) * 0.5, t0)
        if "sub" in L:
            m.add("bass", sub(ROOT[name], BAR * 0.98), t0)
        if "pulse" in L:
            for k in range(8):
                m.add("bass", pulse_bass(ROOT[name], BEAT / 2 * 0.9) * L["pulse"], t0 + k * BEAT / 2)
        if "piano" in L:
            motif = {"sparse": [(0, 0, 0.9), (2.5, 2, 0.6)], "melody": [(0, 4, 0.8), (1, 2, 0.55), (1.5, 3, 0.5), (2, 1, 0.65), (3, 2, 0.5)]}[L["piano"]]
            for beat_at, idx, v in motif:
                m.add("keys", felt(ch[idx] + 12, 3.0, v * 0.55), t0 + beat_at * BEAT)
            m.add("keys", felt(ch[0], 4.0, 0.35), t0)
        if "arp" in L:
            seq = [ch[0] + 12, ch[2] + 12, ch[1] + 12, ch[3] + 12, ch[4] + 12, ch[3] + 12, ch[2] + 12, ch[1] + 12]
            for k in range(16):
                p = pluck(seq[k % 8], 0.5, (0.22 if k % 4 == 0 else 0.15) * L["arp"])
                m.add("keys", pan(p, 0.35 * np.sin(k * 0.8)), t0 + k * BEAT / 4)
        if "kick" in L:
            for k in ((0, 1, 2, 3) if L["kick"] == "four" else (0, 2.5) if L["kick"] == "broken" else (0,)):
                m.add("drums", kick(0.9), t0 + k * BEAT)
        if "rim" in L:
            for k in (1, 3):
                m.add("drums", rim(0.8), t0 + k * BEAT)
        if "clap" in L:
            for k in (1, 3):
                m.add("drums", clap(), t0 + k * BEAT)
        if "shaker" in L:
            for k in range(8):
                m.add("drums", shaker(0.6 + 0.4 * (k % 2), 0.3 if k % 2 else -0.2), t0 + k * BEAT / 2 + (0.02 if k % 2 else 0))


def sfx_events(m, events, cue):
    """events: (time, kind, kwargs); time is seconds, or 'line_id' / 'line_id+0.5' relative to a narration start."""
    def when(x):
        if isinstance(x, (int, float)):
            return float(x)
        mm = re.match(r"([a-z0-9_]+)([+-][0-9.]+)?(\$)?", x)
        c = cue[mm.group(1)]
        base = c["start"] + (c["dur"] if mm.group(3) else 0)
        return base + float(mm.group(2) or 0)
    for at, kind, kw in events:
        t = when(at)
        if kind == "whoosh":
            m.add("fx", whoosh(**kw), t - kw.get("seconds", 1.1) * 0.5)
        elif kind == "swell":
            s = kw.get("seconds", 2.4)
            m.add("fx", swell(**kw), t - s)
        elif kind == "boom":
            m.add("fx", boom(**kw), t)
        elif kind == "braam":
            m.add("fx", braam(CHORDS["Am9"], **kw), t)
        elif kind == "shimmer":
            m.add("fx", shimmer(**kw), t)
        elif kind == "click":
            m.add("fx", click(**kw), t)
        elif kind == "tick":
            m.add("fx", tick(**kw), t)
        elif kind == "chime":
            m.add("fx", chime(**kw), t)
        elif kind == "glitch":
            m.add("fx", glitch(**kw), t)
        elif kind == "typing":
            m.add("fx", typing(**kw), t)
        elif kind == "ticks":   # counter ticks accelerating then slowing
            n, dur = kw["n"], kw["seconds"]
            for k in range(n):
                u = (k / n) ** 1.6
                m.add("fx", tick(0.7, 2600 + 900 * u), t + dur * u)


# ---------------------------------------------------------------- arrangements
def intro_arr(b):
    # 38 bars = 95.0 s, cut like an ad: dark open -> rhythmic details -> tension -> stop -> hit -> build -> drive -> breath -> finale
    if b < 4:   return {"pad": 0.16, "piano": "sparse", **({"sub": 1} if b >= 2 else {})}
    if b < 8:   return {"pad": 0.3, "sub": 1, "pulse": 0.6, "rim": 1, "shaker": 1, "piano": "sparse"}
    if b < 12:  return {"pad": 0.45, "sub": 1, "pulse": 1.0, "kick": "broken", "shaker": 1, "strings": 0.7 if b in (8, 10) else 0,
                        "chord": ["Am9", "Fmaj7", "Dm9", "Esus"][b - 8]}
    if b == 12: return {"pad": 0.2, "chord": "Esus"}                       # the drop-out before the reveal
    if b < 17:  return {"pad": 0.6, "sub": 1, "strings": 0.9 if b in (13, 15) else 0, "piano": "melody"}
    if b < 21:  return {"pad": 0.6, "sub": 1, "arp": 0.8, "kick": "broken", "rim": 1, "shaker": 1}
    if b < 29:  return {"pad": 0.75, "sub": 1, "pulse": 1.0, "arp": 1, "kick": "four", "clap": 1, "shaker": 1,
                        **({"strings": 1} if b in (21, 25) else {})}
    if b < 33:  return {"pad": 0.35, "piano": "melody", "chord": LIFT[b - 29]}
    if b < 35:  return {"pad": 0.7, "sub": 1, "arp": 0.9, "kick": "four", "clap": 1, "shaker": 1, "strings": 1 if b == 33 else 0}
    if b < 38:  return {"pad": 0.4, "piano": "sparse", "chord": "Am9", **({"sub": 1} if b == 35 else {})}
    return {}


def intro_sfx(c):
    ev = [
        (0.2, "shimmer", {"seconds": 6, "level": 1.1}),
        (0.4, "whoosh", {"seconds": 3.0, "lo": 120, "hi": 1200, "level": 0.5}),
        (4.2 * BAR, "whoosh", {"seconds": 1.0, "level": 0.8}),
        ("w1", "boom", {"level": 0.35}), ("w2", "boom", {"level": 0.35}), ("w3", "boom", {"level": 0.4}),
        ("w1", "click", {}), ("w2", "click", {}), ("w3", "click", {}),
        (8.2 * BAR, "whoosh", {"seconds": 1.2, "direction": -1}),
        ("noise+0.2", "glitch", {"seconds": 0.5}), ("noise+1.4", "glitch", {"seconds": 0.4}), ("noise+2.6", "glitch", {"seconds": 0.7}),
        (9.5 * BAR, "ticks", {"n": 30, "seconds": 5.0}),
        (13 * BAR, "swell", {"seconds": 4.5, "level": 1.3}),
        (13 * BAR, "braam", {"level": 1.2}), (13 * BAR, "boom", {"level": 1.1}), (13 * BAR, "shimmer", {"seconds": 5, "root": 76}),
        (14.4 * BAR, "whoosh", {"seconds": 1.6, "lo": 200, "hi": 2600, "level": 0.6}),
        (16.5 * BAR, "whoosh", {"seconds": 1.0}), ("s2-0.3", "whoosh", {"seconds": 0.9, "direction": -1}), ("s3-0.3", "whoosh", {"seconds": 0.9}),
        ("s3+0.6", "shimmer", {"seconds": 2.5, "root": 88, "level": 0.8}),
        (21 * BAR, "swell", {"seconds": 2.0, "level": 0.8}), (21 * BAR, "boom", {"level": 0.8}),
        ("fast+0.1", "ticks", {"n": 14, "seconds": 0.9}),
        (29 * BAR, "swell", {"seconds": 2.2, "level": 0.7}), (29 * BAR, "boom", {"level": 0.6}),
        ("o1", "chime", {"note": 84, "level": 0.6}), ("o2", "chime", {"note": 88, "level": 0.6}), ("o3", "chime", {"note": 91, "level": 0.6}),
        (33 * BAR, "boom", {"level": 0.7}),
        ("end-2.6", "swell", {"seconds": 2.6, "level": 0.9}), ("end", "boom", {"level": 0.9}), ("end", "shimmer", {"seconds": 6, "root": 81}),
    ]
    for k in range(22, 29):          # a whoosh on every cut of the montage
        ev.append((k * BAR, "whoosh", {"seconds": 0.8, "level": 0.55, "direction": 1 if k % 2 else -1}))
    return ev


def guide_arr(b, bars, chapters):
    if b >= bars - 3:
        return {"pad": 0.35, "piano": "sparse", "chord": "Am9"} if b < bars - 1 else {"pad": 0.2, "chord": "Am9"}
    if b < 4:
        return {"pad": 0.22, "piano": "sparse", **({"sub": 1} if b >= 2 else {})}
    ch = sum(1 for c in chapters if c <= b)
    if b in chapters:
        return {"pad": 0.3, "piano": "melody", "sub": 1}
    base = {"pad": 0.45 + 0.04 * (ch % 3), "sub": 1, "kick": "broken", "rim": 1, "shaker": 1}
    if ch in (3, 4, 6, 8):
        base["arp"] = 0.7
    if ch in (5, 6):
        base["pulse"] = 0.6
    return base


def guide_sfx(c, chapter_times):
    ev = [(0.2, "shimmer", {"seconds": 5}), (0.6, "whoosh", {"seconds": 2.2, "lo": 150, "hi": 1500, "level": 0.6})]
    for t in chapter_times:
        ev += [(t - 0.15, "whoosh", {"seconds": 1.1, "level": 0.8}), (t + 0.1, "chime", {"note": 86, "level": 0.7})]
    ev += [
        ("install+0.4", "typing", {"seconds": 1.6}), ("install+4.6", "typing", {"seconds": 0.5}), ("run+2.4", "typing", {"seconds": 0.5}),
        ("open+1.4", "click", {}), ("samples+1.0", "click", {}),
        ("isolate+0.5", "click", {}), ("choose+1.6", "click", {}), ("h+0.8", "click", {}),
        ("simplify+0.5", "click", {"level": 1.2}), ("verify+3.0", "chime", {"note": 84, "level": 0.6}),
        ("guard+0.2", "boom", {"level": 0.35}), ("compare+0.6", "whoosh", {"seconds": 1.6, "lo": 300, "hi": 2400, "level": 0.6}),
        ("export+2.0", "click", {}), ("ask+1.0", "typing", {"seconds": 1.8}), ("cmdk+0.9", "click", {}),
        ("batch+0.3", "typing", {"seconds": 1.4}), ("bench+0.8", "ticks", {"n": 20, "seconds": 2.0}),
        ("end-2.4", "swell", {"seconds": 2.4, "level": 0.8}), ("end", "boom", {"level": 0.7}), ("end", "shimmer", {"seconds": 5}),
    ]
    return ev


def main():
    plan = json.load(open(os.path.join(HERE, "plan.json")))
    meta = json.load(open(os.path.join(PUB, "vo", "vo.json")))
    cues_ts = {}

    # intro: fixed 38 bars
    bars = plan["intro"]["bars"]
    cue, last = place_lines("intro", plan["intro"]["lines"], meta)
    assert last < bars * BAR - 1.5, "intro narration runs past the end"
    m = Mix(bars * BAR)
    lay_music(m, bars, intro_arr)
    sfx_events(m, intro_sfx(cue), cue)
    for k, c in cue.items():
        m.add("vo", voice(os.path.join(PUB, "vo", f"intro_{k}.wav")), c["start"])
    master(m, bars * BAR, "intro", duck=0.82)
    cues_ts["intro"] = {"frames": round(bars * BAR * FPS), "lines": cue}

    # guide: length follows the narration
    cue, last = place_lines("guide", plan["guide"]["lines"], meta)
    bars = int(np.ceil((last + 4.0) / BAR))
    chapters = sorted({int(np.floor(c["start"] / BAR)) for c in cue.values() if c["chapter"] is not None})
    chapter_times = [c["start"] for c in cue.values() if c["chapter"] is not None]
    m = Mix(bars * BAR)
    lay_music(m, bars, lambda b: guide_arr(b, bars, chapters))
    sfx_events(m, guide_sfx(cue, chapter_times), cue)
    for k, c in cue.items():
        m.add("vo", voice(os.path.join(PUB, "vo", f"guide_{k}.wav")), c["start"])
    master(m, bars * BAR, "guide", duck=0.78)
    cues_ts["guide"] = {"frames": round(bars * BAR * FPS), "lines": cue}

    for v in cues_ts.values():
        for c in v["lines"].values():
            c.pop("start"); c.pop("dur")
    with open(os.path.join(HERE, "src", "cues.ts"), "w") as fh:
        fh.write("// generated by video/score.py from plan.json and the narration timing; do not edit\n")
        fh.write("export type Sub = { text: string; s: number; e: number }\n")
        fh.write("export type Cue = { s: number; e: number; text: string; subs: Sub[]; chapter: number | null }\n")
        fh.write(f"export const CUES: Record<'intro' | 'guide', {{ frames: number; lines: Record<string, Cue> }}> = {json.dumps(cues_ts, indent=1)}\n")
    print("src/cues.ts written:", {k: v["frames"] for k, v in cues_ts.items()})


if __name__ == "__main__":
    main()
