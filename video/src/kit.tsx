import React, { useEffect, useMemo, useRef } from 'react'
import {
  AbsoluteFill, Easing, Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig,
} from 'remotion'
import { loadFont as loadGeist } from '@remotion/google-fonts/Geist'
import { loadFont as loadMono } from '@remotion/google-fonts/GeistMono'
import { loadFont as loadPixel } from '@remotion/google-fonts/GeistPixel'

export const SANS = loadGeist('normal', { weights: ['400', '500', '600'], subsets: ['latin'] }).fontFamily
export const MONO = loadMono('normal', { weights: ['400'], subsets: ['latin'] }).fontFamily
export const PIXEL = loadPixel().fontFamily

// 96 BPM at 30 fps: a beat is 18.75 frames, a bar is exactly 75 frames
export const FPS = 30
export const BEAT = 18.75
export const BAR = 75
export const bar = (n: number) => Math.round(n * BAR)
export const beat = (n: number) => Math.round(n * BEAT)

export const C = {
  bg: '#121417', bg2: '#191c20', card: '#1b1f23', stone: '#212529',
  ink: '#e9ecef', ink2: '#a2a8b0', ink3: '#646b74', line: 'rgba(233,236,239,0.09)',
  orange: '#fb5607', royal: '#191970', royalHi: '#5a60ff', sel: '#6c72ff', green: '#3ddc97',
}

export const clamp = { extrapolateLeft: 'clamp' as const, extrapolateRight: 'clamp' as const }
export const ease = Easing.bezier(0.2, 0.8, 0.2, 1)

/** 0..1 critically damped spring starting at `start` frames */
export function useSpring(start: number, dur = 18, bounce = false) {
  const f = useCurrentFrame()
  const { fps } = useVideoConfig()
  return spring({ frame: f - start, fps, durationInFrames: dur, config: bounce ? { damping: 14, stiffness: 140 } : { damping: 200 } })
}

// ---------------------------------------------------------------- backgrounds
export const Backdrop: React.FC<{ aura?: number; grid?: number; tint?: 'royal' | 'orange' | 'mix' }> = ({ aura = 1, grid = 1, tint = 'mix' }) => {
  const f = useCurrentFrame()
  const dx = Math.sin(f / 140) * 60, dy = Math.cos(f / 170) * 40
  const a = tint === 'orange' ? 'rgba(251,86,7,0.30)' : 'rgba(36,40,170,0.55)'
  const b = tint === 'royal' ? 'rgba(36,40,170,0.35)' : 'rgba(251,86,7,0.28)'
  return (
    <AbsoluteFill style={{ background: C.bg, overflow: 'hidden' }}>
      <AbsoluteFill style={{
        opacity: aura, transform: `translate(${dx}px, ${dy}px)`,
        background: `radial-gradient(40% 50% at 70% 30%, ${a}, transparent 70%), radial-gradient(34% 40% at 30% 75%, ${b}, transparent 70%), radial-gradient(26% 30% at 85% 80%, ${a}, transparent 72%)`,
      }} />
      <AbsoluteFill style={{
        opacity: 0.55 * grid,
        backgroundImage: 'radial-gradient(rgba(160,170,255,0.10) 1.2px, transparent 1.4px)', backgroundSize: '28px 28px',
        maskImage: 'radial-gradient(80% 70% at 50% 45%, #000 30%, transparent 85%)',
        WebkitMaskImage: 'radial-gradient(80% 70% at 50% 45%, #000 30%, transparent 85%)',
      }} />
      <Grain />
    </AbsoluteFill>
  )
}

export const Grain: React.FC<{ opacity?: number }> = ({ opacity = 0.07 }) => {
  const f = useCurrentFrame()
  return (
    <AbsoluteFill style={{
      opacity, pointerEvents: 'none', backgroundPosition: `${(f * 37) % 160}px ${(f * 53) % 160}px`,
      backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
    }} />
  )
}

// ---------------------------------------------------------------- typography
/** Words rise out of a mask, one after another (stagger in frames). */
export const Rise: React.FC<{ text: string; at: number; size?: number; weight?: number; color?: string; stagger?: number;
  font?: string; tracking?: number; out?: number; align?: 'left' | 'center'; lh?: number; highlight?: string[] }> =
  ({ text, at, size = 96, weight = 600, color = C.ink, stagger = 3, font = SANS, tracking = -0.04, out, align = 'left', lh = 1.02, highlight = [] }) => {
    const f = useCurrentFrame()
    const words = text.split(' ')
    const outP = out != null ? interpolate(f, [out, out + 10], [0, 1], { ...clamp, easing: ease }) : 0
    return (
      <div style={{ fontFamily: font, fontSize: size, fontWeight: weight, color, letterSpacing: `${tracking}em`, lineHeight: lh,
        display: 'flex', flexWrap: 'wrap', justifyContent: align === 'center' ? 'center' : 'flex-start', gap: `0 ${size * 0.26}px` }}>
        {words.map((w, i) => {
          const p = interpolate(f, [at + i * stagger, at + i * stagger + 14], [0, 1], { ...clamp, easing: ease })
          const hl = highlight.includes(w.replace(/[.,]/g, ''))
          return (
            <span key={i} style={{ overflow: 'hidden', display: 'inline-block', paddingBottom: size * 0.12, marginBottom: -size * 0.12 }}>
              <span style={{ display: 'inline-block', transform: `translateY(${(1 - p) * 105 + outP * -105}%)`, filter: `blur(${(1 - p) * 6}px)`,
                color: hl ? C.orange : undefined }}>{w}</span>
            </span>
          )
        })}
      </div>
    )
  }

export const Fade: React.FC<{ at: number; out?: number; children: React.ReactNode; y?: number; style?: React.CSSProperties }> =
  ({ at, out, children, y = 16, style }) => {
    const f = useCurrentFrame()
    const p = interpolate(f, [at, at + 14], [0, 1], { ...clamp, easing: ease })
    const o = out != null ? interpolate(f, [out, out + 10], [1, 0], clamp) : 1
    return <div style={{ opacity: p * o, transform: `translateY(${(1 - p) * y}px)`, filter: `blur(${(1 - p) * 4}px)`, ...style }}>{children}</div>
  }

export const Pill: React.FC<{ children: React.ReactNode; dot?: string; style?: React.CSSProperties }> = ({ children, dot = C.orange, style }) => (
  <div style={{ display: 'inline-flex', alignItems: 'center', gap: 12, padding: '10px 20px', borderRadius: 999, border: `1px solid ${C.line}`,
    background: 'rgba(33,37,41,0.6)', fontFamily: SANS, fontSize: 24, color: C.ink2, backdropFilter: 'blur(12px)', ...style }}>
    <span style={{ width: 10, height: 10, borderRadius: 5, background: dot, boxShadow: `0 0 14px ${dot}` }} />{children}
  </div>
)

// ---------------------------------------------------------------- dot cloud (continuity element)
type Cloud = { p: number[]; r: number[] }
export const DotCloud: React.FC<{ data: Cloud | null; size?: number; x?: number; y?: number; scale?: number;
  assemble?: number; dissolve?: number; highlight?: number; spin?: number; tilt?: number; dot?: number; base?: string; accent?: string; opacity?: number }> =
  ({ data, size = 900, x = 0, y = 0, scale = 1, assemble = 1, dissolve = 0, highlight = 1, spin = 0, tilt = 0.5, dot = 1.6, base = '#d6dbee', accent = C.orange, opacity = 1 }) => {
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
    useEffect(() => {
      const cv = ref.current
      if (!cv || !data || !seeds) return
      const ctx = cv.getContext('2d')!
      ctx.clearRect(0, 0, size, size)
      const n = data.r.length
      const cy = Math.cos(spin), sy = Math.sin(spin), cx = Math.cos(tilt), sx = Math.sin(tilt)
      const items: [number, number, number, number, number][] = []
      for (let i = 0; i < n; i++) {
        const sd = seeds[4 * i + 3]
        // assemble: scattered sphere -> part (staggered by seed)
        const a = Math.min(1, Math.max(0, (assemble - sd * 0.35) / 0.65))
        const e = a * a * (3 - 2 * a)
        let px = data.p[3 * i], py = data.p[3 * i + 2], pz = -data.p[3 * i + 1]   // Z-up -> Y-up
        px = seeds[4 * i] * (1 - e) + px * e; py = seeds[4 * i + 1] * (1 - e) + py * e; pz = seeds[4 * i + 2] * (1 - e) + pz * e
        const rm = data.r[i]
        let alpha = 1
        if (rm && dissolve > 0) {
          const d = Math.min(1, Math.max(0, (dissolve - sd * 0.5) / 0.5))
          const L = Math.hypot(px, py, pz) || 1
          px += (px / L) * d * 0.9; py += (py / L) * d * 0.9 + d * 0.3; pz += (pz / L) * d * 0.9
          alpha = 1 - d
        }
        // rotate: yaw then pitch
        const X = px * cy + pz * sy, Z1 = -px * sy + pz * cy
        const Y = py * cx - Z1 * sx, Z = py * sx + Z1 * cx
        const persp = 4.2 / (4.2 + Z)
        const sxp = size / 2 + X * size * 0.36 * scale * persp
        const syp = size / 2 - Y * size * 0.36 * scale * persp
        items.push([Z, sxp, syp, rm ? highlight : 0, alpha * (0.55 + 0.45 * persp)])
      }
      items.sort((p, q) => q[0] - p[0])
      for (const [, px, py, h, al] of items) {
        if (al <= 0.01) continue
        ctx.globalAlpha = Math.min(1, al)
        ctx.fillStyle = h > 0.5 ? accent : base
        const r = dot * (1 + h * 0.6)
        ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill()
      }
      ctx.globalAlpha = 1
    })
    return <canvas ref={ref} width={size} height={size}
      style={{ position: 'absolute', left: `calc(50% - ${size / 2}px + ${x}px)`, top: `calc(50% - ${size / 2}px + ${y}px)`, opacity }} />
  }

// ---------------------------------------------------------------- app screens
export const Screen: React.FC<{ src: string; at: number; zoom?: [number, number]; focus?: [number, number]; radius?: number;
  width?: number; tiltIn?: boolean; children?: React.ReactNode; dur?: number }> =
  ({ src, at, zoom = [1, 1.06], focus = [0.5, 0.5], radius = 22, width = 1500, tiltIn = true, children, dur = 150 }) => {
    const f = useCurrentFrame()
    const p = interpolate(f, [at, at + 22], [0, 1], { ...clamp, easing: ease })
    const z = interpolate(f, [at, at + dur], zoom, { ...clamp, easing: Easing.inOut(Easing.cubic) })
    const rx = tiltIn ? (1 - p) * 18 : 0
    const h = width * 1080 / 1920
    return (
      <div style={{ position: 'absolute', left: (1920 - width) / 2, top: (1080 - h) / 2, width, height: h, perspective: 2400 }}>
        <div style={{ width: '100%', height: '100%', borderRadius: radius, overflow: 'hidden', opacity: p,
          transform: `rotateX(${rx}deg) translateY(${(1 - p) * 60}px) scale(${0.94 + 0.06 * p})`,
          boxShadow: '0 40px 120px rgba(0,0,0,0.6), 0 0 0 1px rgba(233,236,239,0.08)', background: C.bg }}>
          <div style={{ width: '100%', height: '100%', transform: `scale(${z})`, transformOrigin: `${focus[0] * 100}% ${focus[1] * 100}%` }}>
            <Img src={staticFile(`shots/${src}.png`)} style={{ width: '100%', height: '100%', display: 'block' }} />
            {children}
          </div>
        </div>
      </div>
    )
  }

/** Ring + label pointing at a box (in 1920x1080 screenshot coordinates) inside a Screen. */
export const Spot: React.FC<{ b?: { x: number; y: number; width: number; height: number }; at: number; label?: string; pad?: number; side?: 'top' | 'bottom' | 'left' | 'right' }> =
  ({ b, at, label, pad = 8, side = 'top' }) => {
    const f = useCurrentFrame()
    if (!b) return null
    const p = interpolate(f, [at, at + 12], [0, 1], { ...clamp, easing: ease })
    const pulse = 0.5 + 0.5 * Math.sin((f - at) / 6)
    const pct = (v: number, t: number) => `${(v / t) * 100}%`
    return (
      <>
        <div style={{ position: 'absolute', left: pct(b.x - pad, 1920), top: pct(b.y - pad, 1080), width: pct(b.width + 2 * pad, 1920),
          height: pct(b.height + 2 * pad, 1080), borderRadius: 16, border: `3px solid ${C.orange}`, opacity: p,
          boxShadow: `0 0 0 ${6 + pulse * 6}px rgba(251,86,7,${0.18 - pulse * 0.08}), 0 0 40px rgba(251,86,7,0.35)` }} />
        {label && (
          <div style={{ position: 'absolute', opacity: p, fontFamily: SANS, fontWeight: 600, fontSize: 26, color: '#1c0b02', background: C.orange,
            padding: '8px 16px', borderRadius: 999, whiteSpace: 'nowrap',
            left: side === 'right' ? pct(b.x + b.width + pad + 14, 1920) : side === 'left' ? undefined : pct(b.x, 1920),
            right: side === 'left' ? pct(1920 - b.x + pad + 14, 1920) : undefined,
            top: side === 'top' ? pct(b.y - pad - 58, 1080) : side === 'bottom' ? pct(b.y + b.height + pad + 12, 1080) : pct(b.y + b.height / 2 - 22, 1080),
            transform: `translateY(${(1 - p) * 10}px)` }}>{label}</div>
        )}
      </>
    )
  }

/** Animated cursor gliding between points (screenshot coords), with a click ripple at each stop. */
export const Cursor: React.FC<{ path: { t: number; x: number; y: number; click?: boolean }[] }> = ({ path }) => {
  const f = useCurrentFrame()
  let x = path[0].x, y = path[0].y
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i], b = path[i + 1]
    if (f >= a.t && f <= b.t) {
      const k = Easing.inOut(Easing.cubic)((f - a.t) / Math.max(1, b.t - a.t))
      x = a.x + (b.x - a.x) * k; y = a.y + (b.y - a.y) * k
    } else if (f > b.t) { x = b.x; y = b.y }
  }
  const clicks = path.filter((p) => p.click && f >= p.t && f < p.t + 18)
  return (
    <>
      {clicks.map((c, i) => {
        const k = (f - c.t) / 18
        return <div key={i} style={{ position: 'absolute', left: `${(c.x / 1920) * 100}%`, top: `${(c.y / 1080) * 100}%`, width: 70 * k, height: 70 * k,
          marginLeft: -35 * k, marginTop: -35 * k, borderRadius: '50%', border: `3px solid ${C.orange}`, opacity: 1 - k }} />
      })}
      <svg width="34" height="34" viewBox="0 0 24 24" style={{ position: 'absolute', left: `${(x / 1920) * 100}%`, top: `${(y / 1080) * 100}%`,
        filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.5))' }}>
        <path d="M4 2.5 20 12l-7.2 1.7L9.4 21 4 2.5Z" fill="#fff" stroke="#121417" strokeWidth="1.2" strokeLinejoin="round" />
      </svg>
    </>
  )
}

export const Wipe: React.FC<{ a: string; b: string; at: number; dur: number; width?: number }> = ({ a, b, at, dur, width = 1500 }) => {
  const f = useCurrentFrame()
  const t = interpolate(f, [at + 10, at + dur * 0.45, at + dur * 0.7, at + dur], [0.92, 0.3, 0.62, 0.5], { ...clamp, easing: Easing.inOut(Easing.cubic) })
  const h = width * 1080 / 1920
  const appear = interpolate(f, [at, at + 18], [0, 1], { ...clamp, easing: ease })
  return (
    <div style={{ position: 'absolute', left: (1920 - width) / 2, top: (1080 - h) / 2, width, height: h, borderRadius: 22, overflow: 'hidden', opacity: appear,
      boxShadow: '0 40px 120px rgba(0,0,0,0.6), 0 0 0 1px rgba(233,236,239,0.08)', transform: `scale(${0.95 + 0.05 * appear})` }}>
      <Img src={staticFile(`shots/${a}.png`)} style={{ position: 'absolute', width: '100%', height: '100%' }} />
      <div style={{ position: 'absolute', inset: 0, clipPath: `inset(0 0 0 ${t * 100}%)` }}>
        <Img src={staticFile(`shots/${b}.png`)} style={{ position: 'absolute', width: '100%', height: '100%' }} />
      </div>
      <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${t * 100}%`, width: 3, background: C.orange, boxShadow: `0 0 24px ${C.orange}` }} />
      <div style={{ position: 'absolute', top: 40, left: `calc(${t * 100}% - 170px)`, fontFamily: MONO, fontSize: 22, color: C.ink, background: 'rgba(18,20,23,0.8)', padding: '6px 12px', borderRadius: 8 }}>original</div>
      <div style={{ position: 'absolute', top: 40, left: `calc(${t * 100}% + 18px)`, fontFamily: MONO, fontSize: 22, color: '#1c0b02', background: C.orange, padding: '6px 12px', borderRadius: 8 }}>simplified</div>
    </div>
  )
}

export const Counter: React.FC<{ to: number; at: number; dur?: number; decimals?: number; prefix?: string; suffix?: string; style?: React.CSSProperties }> =
  ({ to, at, dur = 30, decimals = 0, prefix = '', suffix = '', style }) => {
    const f = useCurrentFrame()
    const k = interpolate(f, [at, at + dur], [0, 1], { ...clamp, easing: Easing.out(Easing.cubic) })
    return <span style={{ fontVariantNumeric: 'tabular-nums', ...style }}>{prefix}{(to * k).toFixed(decimals)}{suffix}</span>
  }

/** Beat-synced accent tick: a small dot that flashes on every beat (subtle rhythm cue). */
export const BeatTick: React.FC<{ from?: number; to?: number }> = ({ from = 0, to = 1e9 }) => {
  const f = useCurrentFrame()
  if (f < from || f > to) return null
  const ph = ((f - from) % BEAT) / BEAT
  const o = Math.max(0, 1 - ph * 3)
  return <div style={{ position: 'absolute', right: 60, bottom: 56, width: 12, height: 12, borderRadius: 6, background: C.orange, opacity: 0.25 + 0.75 * o, boxShadow: `0 0 ${18 * o}px ${C.orange}` }} />
}

export const Brand: React.FC<{ at?: number; size?: number }> = ({ at = 0, size = 34 }) => {
  const p = useSpring(at, 20)
  const on = ['11111', '10001', '10001', '10001', '11110']
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: size * 0.4, opacity: p, transform: `translateY(${(1 - p) * 10}px)` }}>
      <svg viewBox="0 0 25 25" width={size} height={size}>
        {on.flatMap((row, yy) => [...row].map((c, xx) => c === '1' && <circle key={`${xx}${yy}`} cx={xx * 5 + 2.5} cy={yy * 5 + 2.5} r="1.8" fill={C.ink} />))}
        <circle cx={22.5} cy={22.5} r="1.8" fill={C.orange} />
      </svg>
      <span style={{ fontFamily: SANS, fontWeight: 600, fontSize: size * 0.95, color: C.ink, letterSpacing: '-0.03em' }}>DeFeat</span>
    </div>
  )
}
