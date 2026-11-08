import React from 'react'
import { AbsoluteFill, Audio, Easing, interpolate, Sequence, staticFile, useCurrentFrame } from 'remotion'
import { Backdrop, bar, Brand, C, clamp, Counter, Dip, Fade, Grain, inOut, lerp, LightLeak, MONO, PIXEL, SANS, Shot, Spoken, Spot, Vignette, Wipe } from './kit'
import { NUM, useBoxes } from './data'
import { CUES } from './cues'
import { Cam, Item, Stage3D, usePreloadParts } from './Part3D'

// The launch film: 38 bars at 96 BPM = 95.0 s. Real parts, rendered in 3D; narration and cuts from cues.ts.
export const INTRO_FRAMES = CUES.intro.frames
const L = CUES.intro.lines
const sec = (s: number) => Math.round(s * 30)

/** A frame range; children get local frame + duration. */
const Seg: React.FC<{ a: number; b: number; children: (t: number, dur: number) => React.ReactNode }> = ({ a, b, children }) => (
  <Sequence from={a} durationInFrames={Math.max(1, b - a)}><Local dur={b - a}>{children}</Local></Sequence>
)
const Local: React.FC<{ dur: number; children: (t: number, dur: number) => React.ReactNode }> = ({ dur, children }) => <>{children(useCurrentFrame(), dur)}</>

/** Huge single-word super, lands with a blur-in and a slight scale settle. */
const Super: React.FC<{ text: string; at: number; out?: number; size?: number; color?: string; x?: number; y?: number; align?: 'left' | 'center' }> =
  ({ text, at, out, size = 210, color = C.ink, x = 150, y = 0, align = 'left' }) => {
    const f = useCurrentFrame()
    const p = lerp(f, [at, at + 12], [0, 1])
    const o = out != null ? interpolate(f, [out, out + 8], [1, 0], clamp) : 1
    const scrim = align === 'left'
      ? 'linear-gradient(90deg, rgba(8,10,13,0.88) 0%, rgba(8,10,13,0.6) 32%, transparent 60%)'
      : 'radial-gradient(55% 40% at 50% 50%, rgba(8,10,13,0.75), transparent 75%)'
    return (
      <>
      <div style={{ position: 'absolute', inset: 0, background: scrim, opacity: p * o, pointerEvents: 'none' }} />
      <div style={{ position: 'absolute', left: align === 'left' ? x : 0, right: align === 'center' ? 0 : undefined, top: 0, bottom: 0, display: 'flex',
        alignItems: 'center', justifyContent: align === 'center' ? 'center' : 'flex-start', transform: `translateY(${y}px)`, pointerEvents: 'none' }}>
        <div style={{ fontFamily: SANS, fontSize: size, fontWeight: 600, letterSpacing: '-0.055em', lineHeight: 0.9, color, opacity: p * o,
          transform: `scale(${1.08 - 0.08 * p})`, filter: `blur(${(1 - p) * 16 + (1 - o) * 10}px)`, transformOrigin: align === 'left' ? 'left center' : 'center',
          textShadow: '0 20px 80px rgba(0,0,0,0.5)' }}>{text}</div>
      </div>
      </>
    )
  }

/** Dark gradient from the bottom edge so captions and the logo read over bright metal. */
const BottomScrim: React.FC<{ h?: number; o?: number }> = ({ h = 55, o = 0.9 }) => (
  <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', background: `linear-gradient(to top, rgba(8,10,13,${o}) 0%, rgba(8,10,13,${o * 0.6}) ${h * 0.45}%, transparent ${h}%)` }} />
)

/** Big stat card for the montage. */
const Stat: React.FC<{ at: number; value: React.ReactNode; label: string; sub?: string; dur: number }> = ({ at, value, label, sub, dur }) => {
  const f = useCurrentFrame()
  const p = lerp(f, [at, at + 14], [0, 1])
  const o = interpolate(f, [dur - 10, dur], [1, 0], clamp)
  return (
    <AbsoluteFill style={{ justifyContent: 'center', alignItems: 'center', opacity: p * o }}>
      <div style={{ fontFamily: PIXEL, fontSize: 300, lineHeight: 0.9, color: C.ink, transform: `scale(${1.1 - 0.1 * p})`, filter: `blur(${(1 - p) * 12}px)`,
        textShadow: '0 0 80px rgba(251,86,7,0.25)' }}>{value}</div>
      <div style={{ fontFamily: SANS, fontSize: 52, fontWeight: 500, color: C.ink, letterSpacing: '-0.02em', marginTop: 26 }}>{label}</div>
      {sub && <div style={{ fontFamily: MONO, fontSize: 24, color: C.ink3, marginTop: 12, letterSpacing: '0.06em' }}>{sub}</div>}
    </AbsoluteFill>
  )
}

/** What the single 3D stage shows at frame f (null = empty stage). Mirrors the 2D scene timing below. */
function stageAt(f: number): { items: Item[]; cam: Cam; opacity?: number } | null {
  const cut1 = L.w1.s, cut2 = L.w2.s, cut3 = L.w3.s, cut4 = L.real.s - 8
  const seg = (a: number, b: number) => (f >= a && f < b ? { t: f - a, dur: b - a } : null)
  let x
  if ((x = seg(0, cut1 - 30))) {
    const { t, dur } = x
    return { items: [{ name: 'mold_plate' }], cam: { az: lerp(t, [0, dur], [1.15, 0.55], inOut), el: lerp(t, [0, dur], [0.18, 0.5], inOut),
      dist: lerp(t, [0, dur], [1.1, 3.0], Easing.bezier(0.5, 0, 0.2, 1)), target: [lerp(t, [0, dur], [0.35, 0], inOut), 0, lerp(t, [0, dur], [0.15, 0], inOut)], fov: 34 } }
  }
  if ((x = seg(cut1 - 30, cut1))) { const { t, dur } = x; return { items: [{ name: 'gearbox_cover' }], cam: { az: lerp(t, [0, dur], [-0.4, -0.1]), el: 0.5, dist: lerp(t, [0, dur], [2.0, 1.6]), target: [0.1, 0.05, 0] } } }
  if ((x = seg(cut1, cut2))) { const { t, dur } = x; return { items: [{ name: 'l_bracket', highlight: 1 }], cam: { az: lerp(t, [0, dur], [0.9, 1.05]), el: 0.3, dist: 1.3, target: [0.35, -0.5, 0.3] } } }
  if ((x = seg(cut2, cut3))) { const { t, dur } = x; return { items: [{ name: 'robot_link', highlight: 1 }], cam: { az: lerp(t, [0, dur], [0.3, 0.45]), el: 0.55, dist: 1.2, target: [0.55, 0, 0] } } }
  if ((x = seg(cut3, cut4))) { const { t, dur } = x; return { items: [{ name: 'flanged_hub', highlight: 1 }], cam: { az: lerp(t, [0, dur], [0.2, 0.4]), el: 1.05, dist: lerp(t, [0, dur], [2.0, 1.8]), target: [0.3, -0.2, 0] } } }
  if ((x = seg(cut4, bar(8.2)))) {
    const { t } = x
    return { items: (['mold_plate', 'gearbox_cover', 'heat_sink'] as const).map((n, i) => ({ name: n, pos: [(i - 1) * 2.35, 0, 0] as [number, number, number],
      spin: 0.7 + t * 0.003 + i * 0.4, highlight: lerp(t, [8 + i * 6, 20 + i * 6], [0, 1]) })), cam: { az: 0, el: 0.42, dist: 8.4, target: [0, -0.1, 0] } }
  }
  if ((x = seg(bar(8.2), bar(12.5)))) {
    const { t, dur } = x
    return { items: [{ name: 'printer_chassis', highlight: 1, wire: lerp(t, [6, 40], [0, 1]), jitter: lerp(t, [20, dur], [0, 3]), t: t / 3 }],
      cam: { az: lerp(t, [0, dur], [0.4, 1.4], inOut), el: lerp(t, [0, dur], [0.7, 0.45]), dist: lerp(t, [0, dur], [3.4, 2.2], inOut) },
      opacity: Math.floor(t / 2) % 9 === 0 ? 0.6 : 1 }
  }
  if ((x = seg(bar(13), bar(16.5)))) {
    const { t, dur } = x
    return { items: [{ name: 'mold_plate', highlight: lerp(t, [0, 8], [0, 1]), peel: lerp(t, [sec(1.0), sec(5.6)], [0, 1], Easing.bezier(0.4, 0, 0.3, 1)),
      simplified: lerp(t, [sec(5.4), sec(6.2)], [0, 1]) }],
      cam: { az: 0.55 + t * 0.0022, el: 0.48, dist: lerp(t, [0, dur], [2.6, 3.2], inOut), target: [0, lerp(t, [sec(3.0), sec(4.2)], [0, -0.5], inOut), 0] },
      opacity: interpolate(t, [dur - 14, dur], [1, 0], clamp) }
  }
  if ((x = seg(bar(16.5), L.s2.s - 6))) { const { t, dur } = x; return { items: [{ name: 'gearbox_cover', highlight: lerp(t, [sec(0.9), sec(1.6)], [0, 1]) }], cam: { az: lerp(t, [0, dur], [-0.6, -0.2], inOut), el: 0.55, dist: 2.7 } } }
  if ((x = seg(L.s2.s - 6, L.s3.s - 6))) { const { t, dur } = x; return { items: [{ name: 'robot_link', highlight: lerp(t, [0, 10, dur - 20, dur], [0.2, 1, 1, 0.3]) }], cam: { az: lerp(t, [0, dur], [0.2, 0.7], inOut), el: 0.6, dist: 2.6 } } }
  if ((x = seg(L.s3.s - 6, bar(21.2)))) {
    const { t, dur } = x
    return { items: [{ name: 'l_bracket', highlight: 1, peel: lerp(t, [sec(0.5), sec(3.0)], [0, 1], Easing.bezier(0.4, 0, 0.3, 1)), simplified: lerp(t, [sec(2.8), sec(3.4)], [0, 1]) }],
      cam: { az: lerp(t, [0, dur], [0.5, 1.1], inOut), el: 0.42, dist: 2.9 } }
  }
  if ((x = seg(bar(21.2), bar(22)))) { const { t, dur } = x; return { items: [{ name: 'heat_sink', highlight: 1, peel: lerp(t, [4, dur], [0, 1]), simplified: lerp(t, [dur - 8, dur], [0, 1]) }], cam: { az: 0.9 - t * 0.006, el: 0.5, dist: 2.4 } } }
  if ((x = seg(bar(22), bar(23)))) return { items: [{ name: 'flanged_hub', simplified: 1 }], cam: { az: x.t * 0.02, el: 0.6, dist: 3.6 }, opacity: 0.35 }
  if ((x = seg(bar(24), bar(25)))) return { items: [{ name: 'printer_chassis', wire: 1, simplified: 1 }], cam: { az: 0.5 + x.t * 0.01, el: 0.6, dist: 3.2 }, opacity: 0.4 }
  if ((x = seg(bar(26), bar(27)))) return { items: [{ name: 'robot_link', simplified: 1 }], cam: { az: 1.2 - x.t * 0.01, el: 0.5, dist: 2.8 }, opacity: 0.35 }
  if ((x = seg(bar(28), bar(29)))) return { items: [{ name: 'mold_plate', simplified: 1 }], cam: { az: 0.4 + x.t * 0.012, el: 0.55, dist: 3.2 }, opacity: 0.35 }
  if ((x = seg(bar(29), bar(33)))) {
    const { t } = x
    return { items: (['gearbox_cover', 'flanged_hub', 'heat_sink', 'robot_link'] as const).map((n, i) => ({ name: n, simplified: 1,
      pos: [(i - 1.5) * 2.2, lerp(t, [6 + i * 8, 30 + i * 8], [-1.2, 0]), 0] as [number, number, number], spin: 0.6 + t * 0.006 + i * 0.7 })),
      cam: { az: 0, el: 0.32, dist: 10.5, target: [0, 0.55, 0] }, opacity: interpolate(t, [0, 10, bar(4) - 12, bar(4)], [0, 1, 1, 0.3], clamp) }
  }
  if ((x = seg(bar(33), INTRO_FRAMES))) {
    const { t, dur } = x
    const endT = L.end.s - bar(33)
    return { items: [{ name: 'mold_plate', simplified: 1 }], cam: { az: -0.4 + t * 0.004, el: lerp(t, [0, dur], [0.25, 0.55], inOut),
      dist: lerp(t, [0, dur], [2.4, 3.8], inOut), target: [0, lerp(t, [endT - 30, endT + 20], [0, -0.55], inOut), 0] } }
  }
  return null
}

export const Intro: React.FC = () => {
  const f = useCurrentFrame()
  const B = useBoxes()
  const ready = usePreloadParts(['mold_plate', 'gearbox_cover', 'flanged_hub', 'heat_sink', 'robot_link', 'l_bracket', 'printer_chassis'])
  const cut1 = L.w1.s, cut2 = L.w2.s, cut3 = L.w3.s, cut4 = L.real.s - 8
  const blackout = [bar(12.5), bar(13)]

  if (!ready) return <AbsoluteFill style={{ background: C.bg }} />
  const st = stageAt(f)
  return (
    <AbsoluteFill style={{ background: C.bg, fontFamily: SANS }}>
      <Audio src={staticFile('intro.wav')} />
      <Backdrop aura={interpolate(f, [0, bar(3), blackout[0], blackout[0] + 1, bar(13), bar(13) + 20], [0, 0.7, 0.7, 0, 0, 0.9], clamp)} />
      <Stage3D items={st?.items ?? []} cam={st?.cam ?? {}} opacity={st?.opacity ?? 1} />

      {/* ---------- 1. cold open: macro glide across a mold cavity plate */}
      <Seg a={0} b={cut1 - 30}>{(t, dur) => (
        <>
          <BottomScrim h={40} />
          <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 110 }}>
            <Spoken cue={L.detail} size={58} weight={500} align="center" highlight={['detail']} out={dur - 18} />
          </AbsoluteFill>
        </>
      )}</Seg>

      {/* ---------- 2. the details, one per beat */}
      <Seg a={cut1} b={cut2}>{(t, dur) => (
        <>
          <Super text="Fillets." at={0} />
        </>
      )}</Seg>
      <Seg a={cut2} b={cut3}>{(t, dur) => (
        <>
          <Super text="Chamfers." at={0} />
        </>
      )}</Seg>
      <Seg a={cut3} b={cut4}>{(t, dur) => (
        <>
          <Super text="Holes." at={0} />
        </>
      )}</Seg>
      <Seg a={cut4} b={bar(8.2)}>{(t, dur) => (
        <>
          <BottomScrim h={40} />
          <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 110 }}>
            <Spoken cue={{ ...L.real, s: L.real.s - cut4, e: L.real.e - cut4 }} size={60} weight={500} align="center" highlight={['real.']} out={dur - 16} />
          </AbsoluteFill>
        </>
      )}</Seg>

      {/* ---------- 3. noise */}
      <Seg a={bar(8.2)} b={blackout[0]}>{(t, dur) => {
        return (
          <>
            <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to bottom, rgba(8,10,13,0.75), transparent 30%, transparent 70%, rgba(8,10,13,0.8))' }} />
            <Super text="Noise." at={L.noise.s - bar(8.2) + sec(2.3)} align="center" size={260} color={C.orange} />
            <AbsoluteFill style={{ justifyContent: 'flex-start', alignItems: 'center', paddingTop: 120 }}>
              <Spoken cue={{ ...L.noise, s: L.noise.s - bar(8.2), e: L.noise.e - bar(8.2) }} text="But to a simulation," size={52} weight={500} align="center" />
            </AbsoluteFill>
          </>
        )
      }}</Seg>

      {/* ---------- 4. the reveal: every removable feature lifts away */}
      <Seg a={bar(13)} b={bar(16.5)}>{(t, dur) => {
        const peel = lerp(t, [sec(1.0), sec(5.6)], [0, 1], Easing.bezier(0.4, 0, 0.3, 1))
        const simp = lerp(t, [sec(5.4), sec(6.2)], [0, 1])
        return (
          <>
            <div style={{ position: 'absolute', inset: 0, opacity: lerp(t, [sec(3.0), sec(3.8)], [0, 1]) }}><BottomScrim h={60} o={0.95} /></div>
            <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 150, opacity: interpolate(t, [dur - 14, dur], [1, 0], clamp) }}>
              <Brand at={sec(3.2)} size={150} glow={lerp(t, [sec(3.2), sec(4.0), sec(7)], [0, 1, 0.5])} />
            </AbsoluteFill>
          </>
        )
      }}</Seg>

      {/* ---------- 5. sees / keeps / removes */}
      <Seg a={bar(16.5)} b={L.s2.s - 6}>{(t, dur) => (
        <>
          <Super text="Finds." at={L.s1.s - bar(16.5) + 2} size={190} y={-300} />
        </>
      )}</Seg>
      <Seg a={L.s2.s - 6} b={L.s3.s - 6}>{(t, dur) => (
        <>
          <Super text="Keeps." at={6} size={190} y={-300} />
        </>
      )}</Seg>
      <Seg a={L.s3.s - 6} b={bar(21.2)}>{(t, dur) => (
        <>
          <Super text="Removes." at={6} size={190} y={-300} color={C.orange} />
        </>
      )}</Seg>

      {/* ---------- 6. the montage, one cut per bar */}
      <Seg a={bar(22)} b={bar(23)}>{(t, dur) => (
        <>
          <Stat at={4} dur={dur} value={<><Counter to={1.0} decimals={1} at={4} dur={20} /> s</>} label="for a typical part" sub="MEDIAN PER PART · 500 REAL PARTS" />
        </>
      )}</Seg>
      <Seg a={bar(23)} b={bar(24)}>{(t, dur) => (
        <Shot src="features" dur={dur} width={1500} from={{ ry: -22, rx: 8, z: -200 }} to={{ ry: -10, rx: 4, z: 40 }} zoom={[1.05, 1.18]} focus={[0.55, 0.45]} sheen={10} />
      )}</Seg>
      <Seg a={bar(24)} b={bar(25)}>{(t, dur) => (
        <>
          <Stat at={4} dur={dur} value={<><span style={{ color: C.orange }}>−</span><Counter to={NUM.tets} at={4} dur={22} />%</>} label="fewer mesh elements" sub="SAME MESH SIZE · SAMPLE BRACKET" />
        </>
      )}</Seg>
      <Seg a={bar(25)} b={bar(26)}>{(t, dur) => (
        <Shot src="results" dur={dur} width={1500} from={{ ry: 20, rx: 6, z: -180 }} to={{ ry: 8, rx: 3, z: 20 }} zoom={[1.1, 1.3]} focus={[0.02, 0.38]} sheen={8}>
          <Spot b={B['results.fea']} at={10} pad={6} />
        </Shot>
      )}</Seg>
      <Seg a={bar(26)} b={bar(27)}>{(t, dur) => (
        <>
          <Stat at={4} dur={dur} value={<><span style={{ color: C.orange }}>&lt;</span><Counter to={3} at={4} dur={18} />%</>} label="change in stress response" sub="FOUR LOAD CASES · SAMPLE BRACKET" />
        </>
      )}</Seg>
      <Seg a={bar(27)} b={bar(28)}>{(t, dur) => (
        <Wipe a="view_original" b="view_simplified" dur={dur} width={1500} from={{ ry: 16, rx: 6, z: -160 }} to={{ ry: 4, rx: 3, z: 20 }} />
      )}</Seg>
      <Seg a={bar(28)} b={bar(29)}>{(t, dur) => (
        <>
          <Stat at={4} dur={dur} value={<><Counter to={NUM.valid} at={4} dur={22} />%</>} label="valid solids" sub={`${NUM.parts} RANDOM REAL PARTS`} />
        </>
      )}</Seg>

      {/* ---------- 7. open, private, free: the family of simplified parts */}
      <Seg a={bar(29)} b={bar(33)}>{(t, dur) => (
        <>
          <div style={{ position: 'absolute', left: 0, right: 0, top: 120, display: 'flex', justifyContent: 'center', gap: 60, fontFamily: SANS, fontSize: 88,
            fontWeight: 600, letterSpacing: '-0.045em', color: C.ink }}>
            {([['Open source.', L.o1], ['Private.', L.o2], ['Free.', L.o3]] as const).map(([w, c], i) => {
              const p = lerp(t, [c.s - bar(29) - 2, c.s - bar(29) + 10], [0, 1])
              return <span key={w} style={{ opacity: p, filter: `blur(${(1 - p) * 10}px)`, transform: `translateY(${(1 - p) * 20}px)`, color: i === 2 ? C.orange : C.ink }}>{w}</span>
            })}
          </div>
          <Fade at={L.o3.s - bar(29) + 20} style={{ position: 'absolute', left: 0, right: 0, bottom: 90, textAlign: 'center' }}>
            <div style={{ fontFamily: MONO, fontSize: 24, color: C.ink3, letterSpacing: '0.12em' }}>RUNS ON YOUR MACHINE · WINDOWS · MACOS · LINUX · FREECAD</div>
          </Fade>
          <AbsoluteFill style={{ background: '#000', opacity: interpolate(t, [dur - 12, dur], [0, 0.7], clamp) }} />
        </>
      )}</Seg>

      {/* ---------- 8. finale */}
      <Seg a={bar(33)} b={INTRO_FRAMES}>{(t, dur) => {
        const endT = L.end.s - bar(33)
        return (
          <>
            <div style={{ position: 'absolute', inset: 0, opacity: lerp(t, [endT - 20, endT + 10], [0, 1]) }}><BottomScrim h={62} o={0.95} /></div>
            <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 170 }}>
              <Brand at={endT - 4} size={120} glow={lerp(t, [endT, endT + 30, endT + 90], [0, 1, 0.5])} />
              <div style={{ height: 28 }} />
              <Spoken cue={{ ...L.end, s: endT, e: L.end.e - bar(33) }} text="Simulate what matters." size={46} weight={400} color={C.ink2} align="center" highlight={['matters.']} />
            </AbsoluteFill>
          </>
        )
      }}</Seg>

      {/* transitions + finishing */}
      <LightLeak at={cut1} dur={10} hue="white" />
      <LightLeak at={cut4} dur={14} hue="royal" />
      <LightLeak at={bar(13)} dur={34} hue="white" />
      <LightLeak at={bar(16.5)} dur={16} />
      <LightLeak at={bar(21.2)} dur={20} hue="royal" />
      <LightLeak at={bar(29)} dur={26} />
      <LightLeak at={L.end.s} dur={30} hue="white" />
      <Dip from={bar(16.5) - 10} to={bar(16.5) + 10} peak={0.7} />
      <AbsoluteFill style={{ background: '#000', opacity: f >= blackout[0] && f < blackout[1] ? 1 : 0 }} />
      <Vignette strength={0.8} />
      <AbsoluteFill style={{ background: '#000', opacity: interpolate(f, [0, 40, INTRO_FRAMES - 45, INTRO_FRAMES - 4], [1, 0, 0, 1], clamp) }} />
      <Grain opacity={0.06} />
    </AbsoluteFill>
  )
}
