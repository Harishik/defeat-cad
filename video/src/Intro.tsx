import React from 'react'
import { AbsoluteFill, Audio, Easing, interpolate, Sequence, staticFile, useCurrentFrame } from 'remotion'
import { Backdrop, bar, Brand, C, clamp, Counter, Cursor, Dip, DotCloud, Fade, Grain, inOut, lerp, LightLeak, MONO, Pill, PIXEL,
  Rise, SANS, Shot, Spoken, Spot, Vignette, Wipe } from './kit'
import { NUM, useBoxes, useCloud } from './data'
import { CUES } from './cues'

// 38 bars at 96 BPM = 95.0 s; every scene is cut to the narration in cues.ts
export const INTRO_FRAMES = CUES.intro.frames
const L = CUES.intro.lines
const sec = (s: number) => Math.round(s * 30)

export const Intro: React.FC = () => {
  const f = useCurrentFrame()
  const cloud = useCloud('l_bracket')
  const B = useBoxes()

  // ---- the dot-matrix bracket carries the first 15 bars
  const assemble = lerp(f, [sec(0.4), bar(3.4)], [0, 1], Easing.bezier(0.3, 0, 0.1, 1))
  const cloudX = lerp(f, [L.list.s - 20, L.list.s + 10, L.noise.s - 30, L.noise.s], [0, 420, 420, 0], inOut)
  const cloudY = lerp(f, [0, bar(3.6), bar(4.4), L.noise.s - 20, L.noise.s + 20], [-30, -30, -80, -80, -10], inOut)
  const cloudScale = interpolate(f, [0, bar(4.6), L.list.s, L.noise.s, L.noise.e, L.knows.s, bar(13), bar(13) + 30],
    [0.78, 0.86, 1.0, 0.98, 1.0, 0.98, 1.12, 0.72], { ...clamp, easing: inOut })
  const highlight = f >= L.list.s - 6 && f < bar(13) ? 1 : 0
  const jitter = lerp(f, [L.noise.s, L.noise.s + 40, L.noise.e, L.knows.s], [0, 1.6, 1.6, 0])
  const mesh = lerp(f, [L.noise.s - 10, L.noise.s + 20, L.knows.s - 10, L.knows.s + 20], [0, 1, 1, 0])
  const dissolve = lerp(f, [L.knows.s + 10, L.knows.e + 30], [0, 1], inOut)
  const cloudOpacity = interpolate(f, [bar(13), bar(13) + 20, bar(14.6), bar(15)], [1, 0.2, 0.2, 0], clamp)
  const spin = -0.75 + f * 0.0042 + lerp(f, [L.noise.s, L.noise.e], [0, 0.6])
  const tilt = 0.42 + Math.sin(f / 90) * 0.04

  // ---- end card bracket
  const endAssemble = lerp(f, [L.end.s - 60, L.end.s + 40], [0, 1], Easing.bezier(0.3, 0, 0.1, 1))

  return (
    <AbsoluteFill style={{ background: C.bg, fontFamily: SANS }}>
      <Audio src={staticFile('intro.wav')} />
      <Backdrop aura={interpolate(f, [0, bar(2), bar(25.2), bar(25.6), bar(27), bar(27.2)], [0, 0.9, 0.9, 0.35, 0.35, 1], clamp)} />

      {/* light line that opens the film */}
      <Sequence from={0} durationInFrames={sec(3.5)}>
        <OpeningLine />
      </Sequence>

      {f < bar(15) && (
        <DotCloud data={cloud} size={1080} x={cloudX} y={cloudY} scale={cloudScale} assemble={assemble} dissolve={dissolve} highlight={highlight}
          spin={spin} tilt={tilt} dot={1.9} opacity={cloudOpacity} jitter={jitter} mesh={mesh} t={f / 6} glow={1} />
      )}

      {/* 1. cold open */}
      <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 120 }}>
        {f >= L.made.s - 10 && f < L.sim.s - 4 && <Spoken cue={L.made} size={54} weight={500} align="center" out={L.sim.s - 18} />}
        {f >= L.sim.s - 10 && f < L.list.s && <Spoken cue={L.sim} size={54} weight={500} align="center" highlight={['simulated']} out={L.list.s - 16} />}
      </AbsoluteFill>

      {/* 2. the details */}
      {f >= L.list.s - 6 && f < L.noise.s && (
        <AbsoluteFill style={{ justifyContent: 'center', paddingLeft: 170 }}>
          <div style={{ width: 640 }}>
            <Spoken cue={L.list} size={128} lh={0.98} out={L.noise.s - 16} lead={4} highlight={['logo']} />
          </div>
        </AbsoluteFill>
      )}

      {/* 3. noise */}
      {f >= L.noise.s - 6 && f < L.knows.s && (
        <AbsoluteFill>
          <AbsoluteFill style={{ justifyContent: 'center', alignItems: 'center' }}>
            <div style={{ marginTop: -40 }}>
              <Spoken cue={L.noise} text="To a solver, each one is noise." size={104} align="center" highlight={['noise']} out={L.knows.s - 18} />
            </div>
          </AbsoluteFill>
          <NoiseStats />
        </AbsoluteFill>
      )}

      {/* 4. the difference */}
      {f >= L.knows.s - 6 && f < bar(13) && (
        <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 110 }}>
          <Spoken cue={L.knows} size={58} weight={500} align="center" highlight={['difference']} out={bar(13) - 14} />
        </AbsoluteFill>
      )}

      {/* 5. title */}
      <Sequence from={bar(13)} durationInFrames={bar(2)}>
        <Title />
      </Sequence>

      {/* 6. recognition */}
      <Sequence from={bar(15)} durationInFrames={L.analysis.s - 16 - bar(15)}>
        <Shot src="features" dur={L.analysis.s - 16 - bar(15)} from={{ ry: -24, rx: 10, x: 180, z: -260, y: 70 }} to={{ ry: -9, rx: 5, x: 40, z: -60, y: 70 }}
          zoom={[1.0, 1.22]} focus={[0.9, 0.32]} width={1440} sheen={30}>
          <Spot b={B['features.frow']} at={L.reads.s - bar(15) + sec(4.4)} pad={4} />
          <Cursor path={[{ t: sec(2.6), x: 1000, y: 700 }, { t: L.reads.s - bar(15) + sec(4.6), x: (B['features.frow']?.x ?? 1500) + 160, y: (B['features.frow']?.y ?? 200) + 30, click: true },
            { t: L.reads.s - bar(15) + sec(8), x: (B['features.frow']?.x ?? 1500) + 120, y: (B['features.frow']?.y ?? 200) + 220 }]} />
        </Shot>
        <div style={{ position: 'absolute', left: 150, top: 76 }}>
          <Spoken cue={{ ...L.reads, s: L.reads.s - bar(15), e: L.reads.e - bar(15) }} text="recognizes every feature." size={62} highlight={['every']} />
        </div>
        <div style={{ position: 'absolute', right: 150, top: 86, display: 'flex', gap: 14 }}>
          <Fade at={L.reads.s - bar(15) + sec(5.6)}><Pill dot={C.ink2}>Rules</Pill></Fade>
          <Fade at={L.reads.s - bar(15) + sec(6.3)}><Pill>Neural network · real-world CAD</Pill></Fade>
        </div>
      </Sequence>

      {/* 7. analysis */}
      <Sequence from={L.analysis.s - 16} durationInFrames={L.removes.s - 10 - (L.analysis.s - 16)}>
        <Analysis dur={L.removes.s - 10 - (L.analysis.s - 16)} off={L.analysis.s - 16} />
      </Sequence>

      {/* 8. removal */}
      <Sequence from={L.removes.s - 10} durationInFrames={bar(25.3) - (L.removes.s - 10)}>
        <Wipe a="view_original" b="view_simplified" dur={bar(25.3) - (L.removes.s - 10)} width={1400} from={{ ry: 14, rx: 6, z: -120, y: 60 }} to={{ ry: -6, rx: 3, z: 0, y: 60 }} />
        <div style={{ position: 'absolute', left: 0, right: 0, top: 64, display: 'flex', justifyContent: 'center' }}>
          <Spoken cue={{ ...L.removes, s: L.removes.s - (L.removes.s - 10), e: L.removes.e - (L.removes.s - 10) }} text="Cleanly. Without ever breaking the solid."
            size={56} weight={500} align="center" highlight={['Cleanly']} />
        </div>
      </Sequence>

      {/* 9. trust */}
      {f >= L.trust.s - 10 && f < bar(27.1) && (
        <AbsoluteFill style={{ justifyContent: 'center', alignItems: 'center', padding: '0 220px' }}>
          <Spoken cue={L.trust} size={104} align="center" highlight={['earns']} out={bar(27.1) - 16} />
        </AbsoluteFill>
      )}

      {/* 10. proof */}
      <Sequence from={bar(27.1)} durationInFrames={L.numbers.s - 12 - bar(27.1)}>
        <Proof dur={L.numbers.s - 12 - bar(27.1)} off={bar(27.1)} B={B} />
      </Sequence>

      {/* 11. numbers */}
      <Sequence from={L.numbers.s - 12} durationInFrames={L.open.s - 8 - (L.numbers.s - 12)}>
        <Numbers off={L.numbers.s - 12} dur={L.open.s - 8 - (L.numbers.s - 12)} />
      </Sequence>

      {/* 12. open */}
      {f >= L.open.s - 8 && f < L.end.s - 20 && (
        <AbsoluteFill style={{ justifyContent: 'center', alignItems: 'center' }}>
          <Spoken cue={L.open} text="Open source. Private by design." size={96} align="center" highlight={['Private']} out={L.end.s - 34} />
          <div style={{ height: 30 }} />
          <Spoken cue={L.open} text="It runs entirely on your machine." size={48} weight={400} color={C.ink2} align="center" out={L.end.s - 34} />
          <div style={{ height: 40 }} />
          <Fade at={L.open.s + sec(3.2)} out={L.end.s - 34}>
            <div style={{ fontFamily: MONO, fontSize: 26, color: C.ink3, letterSpacing: '0.08em' }}>WINDOWS · LINUX · MACOS · FREECAD</div>
          </Fade>
        </AbsoluteFill>
      )}

      {/* 13. end card */}
      {f >= L.end.s - 60 && (
        <AbsoluteFill>
          <DotCloud data={cloud} size={820} y={-120} assemble={endAssemble} dissolve={1} highlight={0} spin={-0.7 + f * 0.003} tilt={0.42} dot={1.6} glow={0} />
          <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 150 }}>
            <Brand at={L.end.s - 4} size={84} glow={lerp(f, [L.end.s, L.end.s + 30, L.end.s + 90], [0, 1, 0.4])} />
            <div style={{ height: 26 }} />
            <Spoken cue={L.end} text="Simulation-ready, by design." size={40} weight={400} color={C.ink2} align="center" />
          </AbsoluteFill>
        </AbsoluteFill>
      )}

      {/* transitions + finishing */}
      <LightLeak at={bar(13)} dur={30} hue="white" />
      <LightLeak at={bar(15)} dur={22} />
      <LightLeak at={L.removes.s - 10} dur={20} hue="royal" x={30} />
      <LightLeak at={bar(27.1)} dur={24} x={70} />
      <LightLeak at={L.numbers.s - 12} dur={20} hue="royal" />
      <LightLeak at={L.end.s} dur={30} hue="white" />
      <Dip from={L.analysis.s - 26} to={L.analysis.s - 6} peak={0.6} />
      <Vignette />
      <AbsoluteFill style={{ background: '#000', opacity: interpolate(f, [INTRO_FRAMES - 40, INTRO_FRAMES - 4], [0, 1], clamp) }} />
      <Grain opacity={0.07} />
    </AbsoluteFill>
  )
}

const OpeningLine: React.FC = () => {
  const f = useCurrentFrame()
  const w = lerp(f, [4, 50], [0, 1], Easing.bezier(0.7, 0, 0.2, 1))
  const o = interpolate(f, [60, 100], [1, 0], clamp)
  return (
    <AbsoluteFill style={{ justifyContent: 'center', alignItems: 'center', opacity: o }}>
      <div style={{ width: 1400 * w, height: 2, background: `linear-gradient(90deg, transparent, ${C.orange}, #fff, ${C.orange}, transparent)`,
        boxShadow: `0 0 30px ${C.orange}, 0 0 90px rgba(251,86,7,0.6)` }} />
    </AbsoluteFill>
  )
}

const NoiseStats: React.FC = () => {
  const f = useCurrentFrame()
  const a = L.noise.s + sec(2.6), b = L.noise.s + sec(4.4)
  const out = L.knows.s - 18
  const flick = (k: number) => (Math.floor(f / 2 + k) % 7 === 0 ? 0.55 : 1)
  return (
    <>
      <Fade at={a} out={out} style={{ position: 'absolute', left: 170, bottom: 130, opacity: flick(1) }}>
        <div style={{ fontFamily: MONO, fontSize: 24, color: C.ink3, letterSpacing: '0.1em' }}>MESH ELEMENTS</div>
        <Counter to={1284000} from={96000} at={a} dur={sec(3.2)} style={{ fontFamily: PIXEL, fontSize: 92, color: C.ink }} />
      </Fade>
      <Fade at={b} out={out} style={{ position: 'absolute', right: 170, bottom: 130, textAlign: 'right', opacity: flick(3) }}>
        <div style={{ fontFamily: MONO, fontSize: 24, color: C.ink3, letterSpacing: '0.1em' }}>SOLVE TIME</div>
        <Counter to={6.4} from={0.3} decimals={1} suffix=" h" at={b} dur={sec(2.2)} style={{ fontFamily: PIXEL, fontSize: 92, color: C.orange }} />
      </Fade>
    </>
  )
}

const Title: React.FC = () => {
  const f = useCurrentFrame()
  const s = lerp(f, [0, 150], [1.06, 1.0], Easing.out(Easing.cubic))
  const out = interpolate(f, [bar(2) - 16, bar(2)], [1, 0], clamp)
  return (
    <AbsoluteFill style={{ justifyContent: 'center', alignItems: 'center', transform: `scale(${s})`, opacity: out }}>
      <Brand at={4} size={170} glow={lerp(f, [4, 30, 120], [0, 1, 0.5])} />
      <div style={{ height: 34 }} />
      <Rise text="Simulation-ready CAD, automatically." at={40} size={40} weight={400} color={C.ink2} align="center" stagger={2} />
    </AbsoluteFill>
  )
}

const Analysis: React.FC<{ dur: number; off: number }> = ({ dur, off }) => {
  const f = useCurrentFrame()
  const B = useBoxes()
  const cue = { ...L.analysis, s: L.analysis.s - off, e: L.analysis.e - off }
  const words = ['Static', 'Modal', 'Thermal', 'Flow']
  // each analysis word lands as it is spoken; positions estimated from the characters of the line
  const at = (w: string) => cue.s + (cue.e - cue.s) * (L.analysis.text.toLowerCase().indexOf(w.toLowerCase()) / L.analysis.text.length) - 2
  return (
    <AbsoluteFill>
      <Shot src="features" dur={dur} from={{ ry: 18, rx: 6, x: 360, z: -300, y: 40 }} to={{ ry: 10, rx: 4, x: 300, z: -180, y: 40 }}
        zoom={[1.45, 1.6]} focus={[0.62, 0.02]} width={1300} sheen={40}>
        <Spot b={B['features.presets']} at={at('Static')} pad={4} />
      </Shot>
      <div style={{ position: 'absolute', left: 150, top: 0, bottom: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 6 }}>
        <Fade at={cue.s - 6}><div style={{ fontFamily: MONO, fontSize: 24, color: C.ink3, letterSpacing: '0.12em', marginBottom: 18 }}>TELL IT THE ANALYSIS</div></Fade>
        {words.map((w, i) => {
          const p = lerp(f, [at(w), at(w) + 12], [0, 1])
          const next = i < 3 ? lerp(f, [at(words[i + 1]), at(words[i + 1]) + 10], [0, 1]) : 0
          return (
            <div key={w} style={{ fontSize: 120, fontWeight: 600, letterSpacing: '-0.045em', lineHeight: 1.0,
              color: i === 3 ? C.orange : C.ink, opacity: p * (1 - 0.65 * next),
              transform: `translateX(${(1 - p) * -30}px)`, filter: `blur(${(1 - p) * 8}px)` }}>{w}.</div>
          )
        })}
        <div style={{ height: 24 }} />
        <Spoken cue={cue} text="It decides what matters." size={46} weight={500} color={C.ink2} highlight={['matters']} />
      </div>
    </AbsoluteFill>
  )
}

const Proof: React.FC<{ dur: number; off: number; B: Record<string, any> }> = ({ dur, off, B }) => {
  const f = useCurrentFrame()
  const p0 = L.proof.s - off
  const guardAt = p0 + sec(3.8)
  const g = lerp(f, [guardAt, guardAt + 20], [0, 1])
  const restored = lerp(f, [guardAt + sec(1.6), guardAt + sec(2.4)], [0, 1], inOut)
  return (
    <AbsoluteFill>
      <Shot src="results" dur={dur} from={{ ry: 16, rx: 6, x: -160, z: -220, y: 40 }} to={{ ry: 5, rx: 3, x: -220, z: -60, y: 40 }}
        zoom={[1.0, 1.22]} focus={[0.02, 0.32]} width={1360} sheen={24}>
        <Spot b={B['results.stats']} at={p0 + sec(1.0)} pad={6} out={guardAt + 10} />
        <Spot b={B['results.fea']} at={p0 + sec(2.4)} pad={6} />
      </Shot>
      <div style={{ position: 'absolute', left: 150, top: 76 }}>
        <Spoken cue={{ ...L.proof, s: p0, e: L.proof.e - off }} text="Meshed and solved, side by side." size={56} weight={500} />
      </div>
      {/* stress-guard card */}
      <div style={{ position: 'absolute', right: 130, bottom: 140, width: 560, padding: '28px 32px', borderRadius: 24, background: 'rgba(20,23,27,0.82)',
        border: `1px solid ${C.line}`, backdropFilter: 'blur(20px)', boxShadow: '0 40px 120px rgba(0,0,0,0.6)', opacity: g,
        transform: `perspective(1600px) rotateY(${(1 - g) * -20}deg) translateY(${(1 - g) * 40}px)` }}>
        <Pill style={{ fontSize: 20, padding: '6px 14px' }}>Stress guard</Pill>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginTop: 22 }}>
          <div style={{ fontFamily: MONO, fontSize: 22, color: C.ink3 }}>Δ deflection</div>
          <div style={{ fontFamily: PIXEL, fontSize: 64, color: restored > 0.5 ? C.green : C.orange }}>
            {restored > 0.5 ? `${(7.4 - restored * 5.3).toFixed(1)}%` : '7.4%'}
          </div>
        </div>
        <div style={{ height: 8, borderRadius: 4, background: C.line, marginTop: 14, overflow: 'hidden' }}>
          <div style={{ width: `${(74 - restored * 53)}%`, height: '100%', background: restored > 0.5 ? C.green : C.orange, borderRadius: 4 }} />
        </div>
        <div style={{ marginTop: 18, fontSize: 26, color: C.ink, opacity: restored }}>3 key features restored · <span style={{ color: C.green }}>automatically</span></div>
      </div>
    </AbsoluteFill>
  )
}

const Numbers: React.FC<{ off: number; dur: number }> = ({ off, dur }) => {
  const f = useCurrentFrame()
  const a = L.numbers.s - off, b = a + sec(2.4)
  const out = dur - 16
  const col = (at: number, big: React.ReactNode, label: string, sub: string) => (
    <Fade at={at - 4} out={out} y={30}>
      <div style={{ width: 720 }}>
        <div style={{ fontFamily: PIXEL, fontSize: 230, lineHeight: 0.95, color: C.ink }}>{big}</div>
        <div style={{ height: 1, background: C.line, margin: '26px 0 20px', transform: `scaleX(${lerp(f, [at, at + 30], [0, 1])})`, transformOrigin: 'left' }} />
        <div style={{ fontSize: 40, fontWeight: 500, color: C.ink, letterSpacing: '-0.02em' }}>{label}</div>
        <div style={{ fontSize: 26, color: C.ink3, marginTop: 8 }}>{sub}</div>
      </div>
    </Fade>
  )
  return (
    <AbsoluteFill style={{ justifyContent: 'center', alignItems: 'center' }}>
      <div style={{ display: 'flex', gap: 120 }}>
        {col(a, <><span style={{ color: C.orange }}>−</span><Counter to={NUM.tets} at={a} dur={sec(1.4)} />%</>, 'fewer mesh elements', 'sample bracket, same mesh size')}
        {col(b, <><span style={{ color: C.orange }}>&lt;</span><Counter to={3} at={b} dur={sec(1.2)} />%</>, 'change in deflection', 'static stress check, both models')}
      </div>
      <Fade at={b + sec(1.6)} out={out} style={{ marginTop: 70 }}>
        <div style={{ fontFamily: MONO, fontSize: 24, color: C.ink3, letterSpacing: '0.06em' }}>
          {NUM.valid}% VALID SOLIDS · {NUM.parts} REAL PARTS · FILLET F1 {NUM.filletF1.toFixed(2)}
        </div>
      </Fade>
    </AbsoluteFill>
  )
}
