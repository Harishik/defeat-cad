"""Keep / remove decisions. Thresholds are multiples of the target mesh size h,
because a feature only hurts the mesh when it is small relative to the elements."""
from __future__ import annotations

from dataclasses import dataclass, field

PRESETS = {
    "static": {
        "label": "Linear static (stress)",
        "blurb": "Removes cosmetic detail; keeps ribs, gussets and load-path geometry.",
        "rules": {
            "emboss": {"remove": "always"},
            "chamfer": {"remove_below": 1.5},
            "fillet": {"remove_below": 1.5},
            "hole": {"remove_below": 2.0},
            "slot": {"remove_below": 1.0},
            "pocket": {"remove_below": 1.0},
            "step": {"remove_below": 1.0},
            "boss": {"remove_below": 1.5},
            "rib": {"remove": "never"},
            "protrusion": {"remove": "never"},
        },
    },
    "modal": {
        "label": "Modal (vibration)",
        "blurb": "Keeps stiffness and mass-carrying geometry; strips small detail.",
        "rules": {
            "emboss": {"remove": "always"},
            "chamfer": {"remove_below": 2.0},
            "fillet": {"remove_below": 2.0},
            "hole": {"remove_below": 2.5},
            "slot": {"remove_below": 1.5},
            "pocket": {"remove_below": 1.5},
            "step": {"remove_below": 1.5},
            "boss": {"remove_below": 1.0},
            "rib": {"remove": "never"},
            "protrusion": {"remove": "never"},
        },
    },
    "thermal": {
        "label": "Thermal",
        "blurb": "Keeps ribs and fins (they carry heat); removes blends and small holes.",
        "rules": {
            "emboss": {"remove": "always"},
            "chamfer": {"remove_below": 2.0},
            "fillet": {"remove_below": 2.0},
            "hole": {"remove_below": 2.0},
            "slot": {"remove_below": 1.0},
            "pocket": {"remove_below": 1.0},
            "step": {"remove_below": 1.0},
            "boss": {"remove_below": 1.0},
            "rib": {"remove": "never"},
            "protrusion": {"remove": "never"},
        },
    },
    "cfd": {
        "label": "CFD (external flow)",
        "blurb": "Aggressive cleanup for a light surface mesh; thin ribs go too.",
        "rules": {
            "emboss": {"remove": "always"},
            "chamfer": {"remove_below": 3.0},
            "fillet": {"remove_below": 3.0},
            "hole": {"remove_below": 3.0},
            "slot": {"remove_below": 2.0},
            "pocket": {"remove_below": 2.0},
            "step": {"remove_below": 2.0},
            "boss": {"remove_below": 2.0},
            "rib": {"remove_below": 1.0},
            "protrusion": {"remove_below": 1.0},
        },
    },
}

SIZE_LABEL = {"hole": "Ø", "slot": "width", "pocket": "width", "step": "width", "boss": "Ø", "rib": "thickness",
              "fillet": "R", "chamfer": "width", "emboss": "height", "protrusion": "height"}


def default_h(char_len: float) -> float:
    """Default target element size: ~1/40 of the part diagonal, rounded to a nice number."""
    raw = char_len / 40.0
    for nice in (0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20):
        if raw <= nice:
            return float(nice)
    return round(raw)


@dataclass
class Decision:
    feature_id: int
    action: str          # "remove" | "keep"
    reason: str
    locked: bool = False  # set by the user / chat; presets don't override

    def to_dict(self):
        return {"feature_id": self.feature_id, "action": self.action, "reason": self.reason, "locked": self.locked}


@dataclass
class Policy:
    preset: str = "static"
    h: float = 2.0
    overrides: dict = field(default_factory=dict)        # class -> rule dict (from chat)
    feature_overrides: dict = field(default_factory=dict)  # feature id -> "keep"/"remove"
    protected_faces: set = field(default_factory=set)

    def rule_for(self, kind):
        r = dict(PRESETS[self.preset]["rules"].get(kind, {"remove": "never"}))
        if kind in self.overrides:
            r = dict(self.overrides[kind])
        return r


def _fmt(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def decide(features, policy: Policy):
    out = {}
    for ft in features:
        if ft.id in policy.feature_overrides:
            a = policy.feature_overrides[ft.id]
            out[ft.id] = Decision(ft.id, a, f"Set by you: {a}", locked=True)
            continue
        if policy.protected_faces & set(ft.faces):
            out[ft.id] = Decision(ft.id, "keep", "Touches a protected (load / boundary) face")
            continue
        rule = policy.rule_for(ft.kind)
        lbl = SIZE_LABEL.get(ft.kind, "size")
        size_txt = f"{lbl}{_fmt(ft.size)}" if lbl in ("Ø", "R") else f"{lbl} {_fmt(ft.size)}"
        if rule.get("remove") == "always":
            out[ft.id] = Decision(ft.id, "remove", f"{ft.kind.capitalize()} is cosmetic for {PRESETS[policy.preset]['label'].lower()}")
        elif rule.get("remove") == "never":
            out[ft.id] = Decision(ft.id, "keep", f"{ft.kind.capitalize()}s are kept for {PRESETS[policy.preset]['label'].lower()}")
        else:
            k = rule.get("remove_below", 0)
            lim = k * policy.h
            if "max_size" in rule:
                lim = rule["max_size"]
            if ft.size < lim:
                out[ft.id] = Decision(ft.id, "remove", f"{size_txt} < {_fmt(lim)} ({_fmt(k)}×h, h={_fmt(policy.h)}) — too small to matter for the mesh"
                                      if "max_size" not in rule else f"{size_txt} < {_fmt(lim)} (your limit)")
            else:
                out[ft.id] = Decision(ft.id, "keep", f"{size_txt} ≥ {_fmt(lim)} — large enough to affect results")
    return out
