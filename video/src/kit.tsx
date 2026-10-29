import React, { useEffect, useMemo, useRef } from 'react'
import {
  AbsoluteFill, Easing, Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig,
} from 'remotion'
import { loadFont as loadGeist } from '@remotion/google-fonts/Geist'
import { loadFont as loadMono } from '@remotion/google-fonts/GeistMono'
import { loadFont as loadPixel } from '@remotion/google-fonts/GeistPixel'
import type { Cue } from './cues'

export const SANS = loadGeist('normal', { weights: ['300', '400', '500', '600'], subsets: ['latin'] }).fontFamily
export const MONO = loadMono('normal', { weights: ['400'], subsets: ['latin'] }).fontFamily
export const PIXEL = loadPixel().fontFamily

// 96 BPM at 30 fps: a beat is 18.75 frames, a bar is exactly 75 frames
export const FPS = 30
export const BEAT = 18.75
export const BAR = 75
export const bar = (n: number) => Math.round(n * BAR)
export const beat = (n: number) => Math.round(n * BEAT)

export const C = {
  bg: '#0e1013', bg2: '#15181c', card: '#1b1f23', stone: '#212529',
  ink: '#e9ecef', ink2: '#a2a8b0', ink3: '#646b74', line: 'rgba(233,236,239,0.09)',
  orange: '#fb5607', royal: '#191970', royalHi: '#5a60ff', sel: '#6c72ff', green: '#3ddc97',
}

export const clamp = { extrapolateLeft: 'clamp' as const, extrapolateRight: 'clamp' as const }
export const ease = Easing.bezier(0.2, 0.8, 0.2, 1)
export const inOut = Easing.bezier(0.65, 0, 0.35, 1)

/** interpolate with clamping and the house ease */
export const lerp = (f: number, i: number[], o: number[], e = ease) => interpolate(f, i, o, { ...clamp, easing: e })

/** 0..1 critically damped spring starting at `start` frames */
export function useSpring(start: number, dur = 18, bounce = false) {
  const f = useCurrentFrame()
  const { fps } = useVideoConfig()
  return spring({ frame: f - start, fps, durationInFrames: dur, config: bounce ? { damping: 14, stiffness: 140 } : { damping: 200 } })
}

// ---------------------------------------------------------------- atmosphere
export const Backdrop: React.FC<{ aura?: number; grid?: number; tint?: 'royal' | 'orange' | 'mix'; drift?: number }> = ({ aura = 1, grid = 1, tint = 'mix', drift = 1 }) => {
  const f = useCurrentFrame()
  const dx = Math.sin(f / 140) * 70 * drift, dy = Math.cos(f / 170) * 45 * drift
  const a = tint === 'orange' ? 'rgba(251,86,7,0.26)' : 'rgba(36,40,170,0.5)'
  const b = tint === 'royal' ? 'rgba(36,40,170,0.32)' : 'rgba(251,86,7,0.22)'
  return (
    <AbsoluteFill style={{ background: C.bg, overflow: 'hidden' }}>
      <AbsoluteFill style={{
        opacity: aura, transform: `translate(${dx}px, ${dy}px) scale(1.15)`,
        background: `radial-gradient(40% 50% at 72% 28%, ${a}, transparent 70%), radial-gradient(34% 40% at 28% 78%, ${b}, transparent 70%), radial-gradient(26% 30% at 88% 82%, ${a}, transparent 72%)`,
      }} />
      <AbsoluteFill style={{
        opacity: 0.45 * grid,
        backgroundImage: 'radial-gradient(rgba(160,170,255,0.10) 1.2px, transparent 1.4px)', backgroundSize: '28px 28px',
        backgroundPosition: `${(f * 0.3) % 28}px ${(f * 0.15) % 28}px`,
        maskImage: 'radial-gradient(80% 70% at 50% 45%, #000 25%, transparent 85%)',
        WebkitMaskImage: 'radial-gradient(80% 70% at 50% 45%, #000 25%, transparent 85%)',
      }} />
    </AbsoluteFill>
  )
}

export const Grain: React.FC<{ opacity?: number }> = ({ opacity = 0.06 }) => {
  const f = useCurrentFrame()
  return (
    <AbsoluteFill style={{
      opacity, pointerEvents: 'none', mixBlendMode: 'overlay', backgroundPosition: `${(f * 37) % 160}px ${(f * 53) % 160}px`,
      backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
    }} />
  )
}

export const Vignette: React.FC<{ strength?: number }> = ({ strength = 0.75 }) => (
  <AbsoluteFill style={{ pointerEvents: 'none', background: `radial-gradient(120% 95% at 50% 48%, transparent 55%, rgba(0,0,0,${strength}) 100%)` }} />
)

/** Anamorphic light leak that blooms across a cut (centred on frame `at`). */
export const LightLeak: React.FC<{ at: number; dur?: number; hue?: 'orange' | 'royal' | 'white'; x?: number }> = ({ at, dur = 26, hue = 'orange', x = 50 }) => {
  const f = useCurrentFrame()
  if (f < at - dur || f > at + dur) return null
  const k = 1 - Math.abs(f - at) / dur
  const e = k * k * (3 - 2 * k)
  const col = hue === 'orange' ? '251,86,7' : hue === 'royal' ? '90,96,255' : '255,240,225'
  const sweep = interpolate(f, [at - dur, at + dur], [x - 30, x + 30])
  return (
    <AbsoluteFill style={{ pointerEvents: 'none', mixBlendMode: 'screen', opacity: e }}>
      <AbsoluteFill style={{ background: `radial-gradient(60% 18% at ${sweep}% 50%, rgba(${col},0.85), transparent 70%)`, filter: 'blur(30px)' }} />
      <AbsoluteFill style={{ background: `radial-gradient(35% 60% at ${sweep + 8}% 40%, rgba(${col},0.35), transparent 70%)` }} />
      <AbsoluteFill style={{ background: `linear-gradient(90deg, transparent ${sweep - 14}%, rgba(255,255,255,0.18) ${sweep}%, transparent ${sweep + 14}%)` }} />
    </AbsoluteFill>
  )
}

/** Fade from/to black (or a colour) over a frame range. */
export const Dip: React.FC<{ from: number; to: number; color?: string; peak?: number }> = ({ from, to, color = '#000', peak = 1 }) => {
  const f = useCurrentFrame()
  if (f < from || f > to) return null
  const m = (from + to) / 2
  const o = f < m ? interpolate(f, [from, m], [0, peak]) : interpolate(f, [m, to], [peak, 0])
  return <AbsoluteFill style={{ background: color, opacity: o, pointerEvents: 'none' }} />
}

// ---------------------------------------------------------------- typography
/** Words rise out of a mask, one after another (stagger in frames). */
export const Rise: React.FC<{ text: string; at: number; size?: number; weight?: number; color?: string; stagger?: number;
  font?: string; tracking?: number; out?: number; align?: 'left' | 'center'; lh?: number; highlight?: string[]; glow?: boolean }> =
  ({ text, at, size = 96, weight = 600, color = C.ink, stagger = 3, font = SANS, tracking = -0.04, out, align = 'left', lh = 1.02, highlight = [], glow }) => {
    const f = useCurrentFrame()
    const words = text.split(' ')
    const outP = out != null ? lerp(f, [out, out + 12], [0, 1]) : 0
    return (
      <div style={{ fontFamily: font, fontSize: size, fontWeight: weight, color, letterSpacing: `${tracking}em`, lineHeight: lh,
        display: 'flex', flexWrap: 'wrap', justifyContent: align === 'center' ? 'center' : 'flex-start', gap: `0 ${size * 0.26}px`,
        textShadow: glow ? `0 0 ${size * 0.5}px rgba(251,86,7,0.25)` : undefined }}>
        {words.map((w, i) => {
          const p = lerp(f, [at + i * stagger, at + i * stagger + 16], [0, 1])
          const hl = highlight.includes(w.replace(/[.,]/g, ''))
          return (
            <span key={i} style={{ overflow: 'hidden', display: 'inline-block', paddingBottom: size * 0.14, marginBottom: -size * 0.14 }}>
              <span style={{ display: 'inline-block', transform: `translateY(${(1 - p) * 105 + outP * -105}%)`, filter: `blur(${(1 - p) * 8 + outP * 6}px)`,
                opacity: 1 - outP * 0.6, color: hl ? C.orange : undefined }}>{w}</span>
            </span>
          )
        })}
      </div>
    )
  }

/**
 * Narration-synced kinetic type: each word lands when the voice says it (timing interpolated across the cue by characters).
 * `text` may differ from the spoken line, as long as it is a subset spoken in order (words are matched by position in the line).
 */
export const Spoken: React.FC<{ cue: Cue; text?: string; size?: number; weight?: number; color?: string; align?: 'left' | 'center';
  highlight?: string[]; out?: number; font?: string; tracking?: number; lh?: number; lead?: number; dim?: boolean }> =
  ({ cue, text, size = 84, weight = 600, color = C.ink, align = 'left', highlight = [], out, font = SANS, tracking = -0.04, lh = 1.04, lead = 2, dim }) => {
    const f = useCurrentFrame()
    const line = cue.text
    const show = (text ?? line).split(' ')
    const L = line.length
    let cursor = 0
    const outP = out != null ? lerp(f, [out, out + 14], [0, 1]) : 0
    return (
      <div style={{ fontFamily: font, fontSize: size, fontWeight: weight, color, letterSpacing: `${tracking}em`, lineHeight: lh,
        display: 'flex', flexWrap: 'wrap', justifyContent: align === 'center' ? 'center' : 'flex-start', gap: `0 ${size * 0.25}px` }}>
        {show.map((w, i) => {
          const bare = w.replace(/[.,!?]/g, '').toLowerCase()
          const idx = line.toLowerCase().indexOf(bare, cursor)
          const pos = idx >= 0 ? idx : cursor
          cursor = idx >= 0 ? idx + bare.length : cursor
          const at = cue.s + (cue.e - cue.s) * (pos / L) - lead
          const p = lerp(f, [at, at + 10], [0, 1])
          const hl = highlight.includes(w.replace(/[.,!?]/g, ''))
          return (
            <span key={i} style={{ overflow: 'hidden', display: 'inline-block', paddingBottom: size * 0.14, marginBottom: -size * 0.14 }}>
              <span style={{ display: 'inline-block', transform: `translateY(${(1 - p) * 70 + outP * -105}%)`, filter: `blur(${(1 - p) * 10 + outP * 6}px)`,
                opacity: (dim ? 0.25 + 0.75 * p : p) * (1 - outP * 0.6), color: hl ? C.orange : undefined }}>{w}</span>
            </span>
          )
        })}
      </div>
    )
  }

export const Fade: React.FC<{ at: number; out?: number; children: React.ReactNode; y?: number; style?: React.CSSProperties; dur?: number }> =
  ({ at, out, children, y = 16, style, dur = 16 }) => {
    const f = useCurrentFrame()
    const p = lerp(f, [at, at + dur], [0, 1])
    const o = out != null ? interpolate(f, [out, out + 12], [1, 0], clamp) : 1
    return <div style={{ opacity: p * o, transform: `translateY(${(1 - p) * y}px)`, filter: `blur(${(1 - p) * 6 + (1 - o) * 4}px)`, ...style }}>{children}</div>
  }

export const Pill: React.FC<{ children: React.ReactNode; dot?: string; style?: React.CSSProperties }> = ({ children, dot = C.orange, style }) => (
  <div style={{ display: 'inline-flex', alignItems: 'center', gap: 12, padding: '10px 20px', borderRadius: 999, border: `1px solid ${C.line}`,
    background: 'rgba(33,37,41,0.55)', fontFamily: SANS, fontSize: 24, color: C.ink2, backdropFilter: 'blur(12px)', ...style }}>
    <span style={{ width: 10, height: 10, borderRadius: 5, background: dot, boxShadow: `0 0 14px ${dot}` }} />{children}
  </div>
)

/** Sentence subtitle synced to the narration (walkthrough). */
export const Subtitles: React.FC<{ cues: Record<string, Cue> }> = ({ cues }) => {
  const f = useCurrentFrame()
  for (const c of Object.values(cues)) {
    for (const s of c.subs) {
      if (f >= s.s - 4 && f <= s.e + 8) {
        const o = Math.min(lerp(f, [s.s - 4, s.s + 6], [0, 1]), interpolate(f, [s.e, s.e + 8], [1, 0], clamp))
        return (
          <div style={{ position: 'absolute', left: 0, right: 0, bottom: 46, display: 'flex', justifyContent: 'center', pointerEvents: 'none' }}>
            <div style={{ maxWidth: 1400, textAlign: 'center', fontFamily: SANS, fontSize: 34, fontWeight: 500, color: C.ink, letterSpacing: '-0.01em',
              padding: '12px 26px', borderRadius: 14, background: 'rgba(10,12,14,0.62)', backdropFilter: 'blur(16px)', opacity: o,
              transform: `translateY(${(1 - o) * 8}px)`, boxShadow: '0 10px 40px rgba(0,0,0,0.35)' }}>{s.text}</div>
          </div>
        )
      }
    }
  }
  return null
}

// ---------------------------------------------------------------- dot cloud (continuity element)
type Cloud = { p: number[]; r: number[] }
export const DotCloud: React.FC<{ data: Cloud | null; size?: number; x?: number; y?: number; scale?: number;
  assemble?: number; dissolve?: number; highlight?: number; spin?: number; tilt?: number; dot?: number; base?: string; accent?: string;
  opacity?: number; glow?: number; jitter?: number; mesh?: number; t?: number }> =
  ({ data, size = 900, x = 0, y = 0, scale = 1, assemble = 1, dissolve = 0, highlight = 1, spin = 0, tilt = 0.5, dot = 1.6,
    base = '#d6dbee', accent = C.orange, opacity = 1, glow = 1, jitter = 0, mesh = 0, t = 0 }) => {
    const ref = useRef<HTMLCanvasElement>(null)
    const seeds = useMemo(() => {
      if (!data) return null
      const n = data.r.length
      const s = new Float32Array(n * 4)
      let st = 1234567
      const rnd = () => ((st = (st * 1103515245 + 12345) % 2147483648) / 2147483648)
      for (let i = 0; i < n; i++) {
        const th = rnd() * Math.PI * 2, ph = Math.acos(2 * rnd() - 1), R = 2.2 + rnd() * 2.5
        s[4 * i] = R * Math.sin(ph) * Math.cos(th); s[4 * i + 1] = R * Math.sin(ph) * Math.sin(th); s[4 * i + 2] = R * Math.cos(ph)
        s[4 * i + 3] = rnd()
      }
      return s
    }, [data])
    // nearest-neighbour pairs on a subsample -> a "mesh" of short edges
    const edges = useMemo(() => {
      if (!data) return null
      const n = data.r.length
      const step = Math.max(1, Math.floor(n / 2600))
      const idx: number[] = []
      for (let i = 0; i < n; i += step) idx.push(i)
      const out: number[] = []
      for (const i of idx) {
        let b1 = -1, b2 = -1, d1 = 1e9, d2 = 1e9
        for (const j of idx) {
          if (j === i) continue
          const dx = data.p[3 * i] - data.p[3 * j], dy = data.p[3 * i + 1] - data.p[3 * j + 1], dz = data.p[3 * i + 2] - data.p[3 * j + 2]
          const d = dx * dx + dy * dy + dz * dz
          if (d < d1) { d2 = d1; b2 = b1; d1 = d; b1 = j } else if (d < d2) { d2 = d; b2 = j }
        }
        if (b1 >= 0) out.push(i, b1)
        if (b2 >= 0) out.push(i, b2)
      }
      return out
    }, [data])
    useEffect(() => {
      const cv = ref.current
      if (!cv || !data || !seeds) return
      const ctx = cv.getContext('2d')!
      ctx.clearRect(0, 0, size, size)
      const n = data.r.length
      const cy = Math.cos(spin), sy = Math.sin(spin), cx = Math.cos(tilt), sx = Math.sin(tilt)
      const P = new Float32Array(n * 4)   // screen x, y, depth, alpha
      const items: number[] = []
      for (let i = 0; i < n; i++) {
        const sd = seeds[4 * i + 3]
        const a = Math.min(1, Math.max(0, (assemble - sd * 0.35) / 0.65))
        const e = a * a * (3 - 2 * a)
        let px = data.p[3 * i], py = data.p[3 * i + 2], pz = -data.p[3 * i + 1]   // Z-up -> Y-up
        px = seeds[4 * i] * (1 - e) + px * e; py = seeds[4 * i + 1] * (1 - e) + py * e; pz = seeds[4 * i + 2] * (1 - e) + pz * e
        if (jitter > 0) {
          const j = jitter * 0.04
          px += Math.sin(t * 0.9 + sd * 90) * j; py += Math.cos(t * 1.1 + sd * 70) * j; pz += Math.sin(t * 0.7 + sd * 50) * j
        }
        const rm = data.r[i]
        let alpha = 1
        if (rm && dissolve > 0) {
          const d = Math.min(1, Math.max(0, (dissolve - sd * 0.5) / 0.5))
          const L = Math.hypot(px, py, pz) || 1
          px += (px / L) * d * 1.1; py += (py / L) * d * 1.1 + d * 0.35; pz += (pz / L) * d * 1.1
          alpha = 1 - d
        }
        const X = px * cy + pz * sy, Z1 = -px * sy + pz * cy
        const Y = py * cx - Z1 * sx, Z = py * sx + Z1 * cx
        const persp = 4.2 / (4.2 + Z)
        P[4 * i] = size / 2 + X * size * 0.36 * scale * persp
        P[4 * i + 1] = size / 2 - Y * size * 0.36 * scale * persp
        P[4 * i + 2] = Z
        P[4 * i + 3] = alpha * (0.45 + 0.55 * persp)
        items.push(i)
      }
      items.sort((p, q) => P[4 * q + 2] - P[4 * p + 2])
      if (mesh > 0 && edges) {
        ctx.lineWidth = 0.8
        for (let k = 0; k < edges.length; k += 2) {
          const i = edges[k], j = edges[k + 1]
          const al = Math.min(P[4 * i + 3], P[4 * j + 3]) * mesh * 0.55
          if (al < 0.02) continue
          ctx.globalAlpha = al
          ctx.strokeStyle = data.r[i] && highlight > 0.5 ? accent : '#9aa3c7'
          ctx.beginPath(); ctx.moveTo(P[4 * i], P[4 * i + 1]); ctx.lineTo(P[4 * j], P[4 * j + 1]); ctx.stroke()
        }
      }
      if (glow > 0) {
        ctx.globalCompositeOperation = 'lighter'
        for (const i of items) {
          const h = data.r[i] ? highlight : 0
          if (h < 0.5) continue
          const al = P[4 * i + 3]
          if (al <= 0.02) continue
          ctx.globalAlpha = Math.min(1, al) * 0.07 * glow
          ctx.fillStyle = accent
          ctx.beginPath(); ctx.arc(P[4 * i], P[4 * i + 1], dot * 5, 0, Math.PI * 2); ctx.fill()
        }
        ctx.globalCompositeOperation = 'source-over'
      }
      for (const i of items) {
        const al = P[4 * i + 3]
        if (al <= 0.01) continue
        const h = data.r[i] ? highlight : 0
        const fog = Math.max(0, Math.min(1, (P[4 * i + 2] + 1.2) / 2.4))
        ctx.globalAlpha = Math.min(1, al)
        ctx.fillStyle = h > 0.5 ? accent : fog > 0.6 ? '#8f98c8' : base
        const r = dot * (1 + h * 0.5) * (1.15 - 0.3 * fog)
        ctx.beginPath(); ctx.arc(P[4 * i], P[4 * i + 1], r, 0, Math.PI * 2); ctx.fill()
      }
      ctx.globalAlpha = 1
    })
    return <canvas ref={ref} width={size} height={size}
      style={{ position: 'absolute', left: `calc(50% - ${size / 2}px + ${x}px)`, top: `calc(50% - ${size / 2}px + ${y}px)`, opacity }} />
  }

// ---------------------------------------------------------------- app screens, shot in 3D
type Pose = { rx?: number; ry?: number; rz?: number; x?: number; y?: number; z?: number; s?: number }
const P0: Required<Pose> = { rx: 0, ry: 0, rz: 0, x: 0, y: 0, z: 0, s: 1 }

/**
 * A screenshot shot like a product film: rises out of depth with a focus pull, drifts between two camera poses,
 * gets one light sweep, and pushes back into the dark on exit. `in`/`out` are frames relative to the Sequence.
 */
export const Shot: React.FC<{ src: string; dur: number; from?: Pose; to?: Pose; zoom?: [number, number]; focus?: [number, number];
  width?: number; sheen?: number; children?: React.ReactNode; exit?: boolean; enter?: boolean; radius?: number }> =
  ({ src, dur, from = {}, to = {}, zoom = [1, 1.06], focus = [0.5, 0.5], width = 1560, sheen = 18, children, exit = true, enter = true, radius = 24 }) => {
    const f = useCurrentFrame()
    const a = { ...P0, ...from }, b = { ...P0, ...to }
    const k = lerp(f, [0, dur], [0, 1], inOut)
    const pose = Object.fromEntries(Object.keys(P0).map((key) => [key, (a as any)[key] + ((b as any)[key] - (a as any)[key]) * k])) as Required<Pose>
    const pin = enter ? lerp(f, [0, 24], [0, 1]) : 1
    const pout = exit ? lerp(f, [dur - 14, dur], [0, 1], Easing.in(Easing.cubic)) : 0
    const z = lerp(f, [0, dur], zoom, inOut)
    const h = width * 1080 / 1920
    const sw = lerp(f, [sheen, sheen + 40], [-40, 140], inOut)
    return (
      <div style={{ position: 'absolute', inset: 0, perspective: 2200, perspectiveOrigin: '50% 45%' }}>
        <div style={{ position: 'absolute', left: (1920 - width) / 2, top: (1080 - h) / 2, width, height: h, transformStyle: 'preserve-3d',
          transform: `translate3d(${pose.x}px, ${pose.y + (1 - pin) * 80}px, ${pose.z - (1 - pin) * 260 - pout * 220}px) rotateX(${pose.rx + (1 - pin) * 14}deg) rotateY(${pose.ry}deg) rotateZ(${pose.rz}deg) scale(${pose.s})`,
          opacity: pin * (1 - pout), filter: `blur(${(1 - pin) * 14 + pout * 10}px)` }}>
          <div style={{ position: 'absolute', inset: 0, borderRadius: radius, overflow: 'hidden', background: C.bg,
            boxShadow: '0 60px 160px rgba(0,0,0,0.7), 0 0 0 1px rgba(233,236,239,0.10), inset 0 1px 0 rgba(255,255,255,0.08)' }}>
            <div style={{ width: '100%', height: '100%', transform: `scale(${z})`, transformOrigin: `${focus[0] * 100}% ${focus[1] * 100}%` }}>
              <Img src={staticFile(`shots/${src}.png`)} style={{ width: '100%', height: '100%', display: 'block' }} />
              {children}
            </div>
            <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', mixBlendMode: 'screen',
              background: `linear-gradient(105deg, transparent ${sw - 12}%, rgba(255,255,255,0.10) ${sw - 3}%, rgba(255,255,255,0.18) ${sw}%, rgba(255,255,255,0.06) ${sw + 4}%, transparent ${sw + 14}%)` }} />
          </div>
          {/* soft floor reflection */}
          <div style={{ position: 'absolute', left: 0, right: 0, top: '100%', height: '40%', marginTop: 10, borderRadius: radius, overflow: 'hidden', opacity: 0.12,
            transform: 'scaleY(-1)', maskImage: 'linear-gradient(to top, #000, transparent 60%)', WebkitMaskImage: 'linear-gradient(to top, #000, transparent 60%)' }}>
            <Img src={staticFile(`shots/${src}.png`)} style={{ position: 'absolute', bottom: 0, width: '100%', display: 'block' }} />
          </div>
        </div>
      </div>
    )
  }

/** Ring + label pointing at a box (in 1920x1080 screenshot coordinates) inside a Shot. */
export const Spot: React.FC<{ b?: { x: number; y: number; width: number; height: number }; at: number; label?: string; pad?: number;
  side?: 'top' | 'bottom' | 'left' | 'right'; out?: number }> =
  ({ b, at, label, pad = 8, side = 'top', out }) => {
    const f = useCurrentFrame()
    if (!b) return null
    const p = lerp(f, [at, at + 14], [0, 1])
    const o = out != null ? interpolate(f, [out, out + 10], [1, 0], clamp) : 1
    const pulse = 0.5 + 0.5 * Math.sin((f - at) / 7)
    const pct = (v: number, t: number) => `${(v / t) * 100}%`
    return (
      <div style={{ opacity: o }}>
        <div style={{ position: 'absolute', left: pct(b.x - pad, 1920), top: pct(b.y - pad, 1080), width: pct(b.width + 2 * pad, 1920),
          height: pct(b.height + 2 * pad, 1080), borderRadius: 16, border: `3px solid ${C.orange}`, opacity: p, transform: `scale(${1.08 - 0.08 * p})`,
          boxShadow: `0 0 0 ${6 + pulse * 6}px rgba(251,86,7,${0.16 - pulse * 0.08}), 0 0 50px rgba(251,86,7,0.35)` }} />
        {label && (
          <div style={{ position: 'absolute', opacity: p, fontFamily: SANS, fontWeight: 600, fontSize: 26, color: '#1c0b02', background: C.orange,
            padding: '8px 16px', borderRadius: 999, whiteSpace: 'nowrap', boxShadow: '0 10px 30px rgba(251,86,7,0.35)',
            left: side === 'right' ? pct(b.x + b.width + pad + 14, 1920) : side === 'left' ? undefined : pct(b.x, 1920),
            right: side === 'left' ? pct(1920 - b.x + pad + 14, 1920) : undefined,
            top: side === 'top' ? pct(b.y - pad - 58, 1080) : side === 'bottom' ? pct(b.y + b.height + pad + 12, 1080) : pct(b.y + b.height / 2 - 22, 1080),
            transform: `translateY(${(1 - p) * 10}px)` }}>{label}</div>
        )}
      </div>
    )
  }

/** Animated cursor gliding between points (screenshot coords), with a click ripple at each stop. */
export const Cursor: React.FC<{ path: { t: number; x: number; y: number; click?: boolean }[] }> = ({ path }) => {
  const f = useCurrentFrame()
  let x = path[0].x, y = path[0].y
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i], b = path[i + 1]
    if (f >= a.t && f <= b.t) {
      const k = Easing.bezier(0.5, 0, 0.1, 1)((f - a.t) / Math.max(1, b.t - a.t))
      x = a.x + (b.x - a.x) * k; y = a.y + (b.y - a.y) * k
    } else if (f > b.t) { x = b.x; y = b.y }
  }
  const pressed = path.some((p) => p.click && f >= p.t && f < p.t + 5)
  const clicks = path.filter((p) => p.click && f >= p.t && f < p.t + 20)
  const appear = lerp(f, [path[0].t, path[0].t + 8], [0, 1])
  return (
    <>
      {clicks.map((c, i) => {
        const k = (f - c.t) / 20
        return <div key={i} style={{ position: 'absolute', left: `${(c.x / 1920) * 100}%`, top: `${(c.y / 1080) * 100}%`, width: 80 * k, height: 80 * k,
          marginLeft: -40 * k, marginTop: -40 * k, borderRadius: '50%', border: `3px solid ${C.orange}`, opacity: 1 - k }} />
      })}
      <svg width="36" height="36" viewBox="0 0 24 24" style={{ position: 'absolute', left: `${(x / 1920) * 100}%`, top: `${(y / 1080) * 100}%`, opacity: appear,
        transform: `scale(${pressed ? 0.85 : 1})`, transformOrigin: '4px 3px', filter: 'drop-shadow(0 6px 10px rgba(0,0,0,0.5))' }}>
        <path d="M4 2.5 20 12l-7.2 1.7L9.4 21 4 2.5Z" fill="#fff" stroke="#121417" strokeWidth="1.2" strokeLinejoin="round" />
      </svg>
    </>
  )
}

/** Before/after slider over two screenshots, inside a 3D shot. */
export const Wipe: React.FC<{ a: string; b: string; dur: number; width?: number; from?: Pose; to?: Pose }> = ({ a, b, dur, width = 1560, from = {}, to = {} }) => {
  const f = useCurrentFrame()
  const t = interpolate(f, [12, dur * 0.42, dur * 0.68, dur - 6], [0.94, 0.28, 0.64, 0.5], { ...clamp, easing: inOut })
  return (
    <Shot src={a} dur={dur} width={width} from={from} to={to} zoom={[1, 1.02]} sheen={dur - 50}>
      <div style={{ position: 'absolute', inset: 0, clipPath: `inset(0 0 0 ${t * 100}%)` }}>
        <Img src={staticFile(`shots/${b}.png`)} style={{ position: 'absolute', width: '100%', height: '100%' }} />
      </div>
      <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${t * 100}%`, width: 3, background: C.orange, boxShadow: `0 0 30px ${C.orange}, 0 0 80px rgba(251,86,7,0.5)` }} />
      <div style={{ position: 'absolute', top: '50%', left: `${t * 100}%`, width: 46, height: 46, marginLeft: -23, marginTop: -23, borderRadius: 23, background: C.orange,
        boxShadow: '0 8px 30px rgba(251,86,7,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#1c0b02', fontSize: 22, fontWeight: 700, fontFamily: SANS }}>⇆</div>
      <div style={{ position: 'absolute', top: 40, left: `calc(${t * 100}% - 170px)`, fontFamily: MONO, fontSize: 22, color: C.ink, background: 'rgba(18,20,23,0.8)', padding: '6px 12px', borderRadius: 8 }}>original</div>
      <div style={{ position: 'absolute', top: 40, left: `calc(${t * 100}% + 18px)`, fontFamily: MONO, fontSize: 22, color: '#1c0b02', background: C.orange, padding: '6px 12px', borderRadius: 8 }}>simplified</div>
    </Shot>
  )
}

export const Counter: React.FC<{ to: number; at: number; dur?: number; decimals?: number; prefix?: string; suffix?: string; style?: React.CSSProperties; from?: number }> =
  ({ to, at, dur = 30, decimals = 0, prefix = '', suffix = '', style, from = 0 }) => {
    const f = useCurrentFrame()
    const k = interpolate(f, [at, at + dur], [0, 1], { ...clamp, easing: Easing.out(Easing.cubic) })
    const v = from + (to - from) * k
    const s = decimals ? v.toFixed(decimals) : Math.round(v).toLocaleString('en-US')
    return <span style={{ fontVariantNumeric: 'tabular-nums', ...style }}>{prefix}{s}{suffix}</span>
  }

export const Brand: React.FC<{ at?: number; size?: number; glow?: number }> = ({ at = 0, size = 34, glow = 0 }) => {
  const p = useSpring(at, 24)
  const on = ['11111', '10001', '10001', '10001', '11110']
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: size * 0.4, opacity: p, transform: `translateY(${(1 - p) * 12}px) scale(${0.96 + 0.04 * p})`,
      filter: glow ? `drop-shadow(0 0 ${30 * glow}px rgba(251,86,7,${0.35 * glow}))` : undefined }}>
      <svg viewBox="0 0 25 25" width={size} height={size}>
        {on.flatMap((row, yy) => [...row].map((c, xx) => c === '1' && <circle key={`${xx}${yy}`} cx={xx * 5 + 2.5} cy={yy * 5 + 2.5} r="1.8" fill={C.ink} />))}
        <circle cx={22.5} cy={22.5} r="1.8" fill={C.orange} />
      </svg>
      <span style={{ fontFamily: SANS, fontWeight: 600, fontSize: size * 0.95, color: C.ink, letterSpacing: '-0.035em' }}>DeFeat</span>
    </div>
  )
}
