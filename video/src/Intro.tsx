import React from 'react'
import { AbsoluteFill, Audio, interpolate, Sequence, staticFile, useCurrentFrame } from 'remotion'
import { Backdrop, BAR, bar, beat, BeatTick, Brand, C, clamp, Counter, Cursor, DotCloud, ease, Fade, Grain, MONO, Pill, PIXEL,
  Rise, SANS, Screen, Spot, Wipe } from './kit'
import { NUM, useBoxes, useCloud } from './data'

// 38 bars at 96 BPM = 95.0 s = 2850 frames
export const INTRO_FRAMES = bar(38)

const STEPS = [
  { n: '01', t: 'Read the exact geometry', s: 'B-Rep face graph · convex, concave, smooth edges' },
  { n: '02', t: 'Recognize every feature', s: 'Rules + a GNN trained on real CAD' },
  { n: '03', t: 'Decide per analysis', s: 'Static · Modal · Thermal · CFD · mesh size h' },
  { n: '04', t: 'Remove safely', s: 'OpenCASCADE in a guarded worker · never an invalid solid' },
  { n: '05', t: 'Prove it', s: 'gmsh on both models · static stress check' },
  { n: '06', t: 'Hand back everything', s: 'Simplified STEP · report · Python · C++' },
]

export const Intro: React.FC = () => {
  const f = useCurrentFrame()
  const cloud = useCloud('l_bracket')
  const B = useBoxes()

  // ---- continuity element: the dot-matrix bracket across the first 18 bars
  const assemble = interpolate(f, [bar(0.3), bar(2.6)], [0, 1], { ...clamp, easing: ease })
  const cloudX = interpolate(f, [bar(3.6), bar(4.4), bar(15.6), bar(16.2)], [0, 360, 360, 0], { ...clamp, easing: ease })
  const cloudScale = interpolate(f, [bar(0), bar(3.4), bar(4.2), bar(16), bar(17.6)], [0.82, 0.82, 1.0, 1.0, 2.6], { ...clamp, easing: ease })
  const cloudY = interpolate(f, [bar(0), bar(3.4), bar(4.2)], [-120, -120, 20], { ...clamp, easing: ease })
  const highlight = f >= bar(4) && f < bar(16) ? 1 : 0
  const dissolve = interpolate(f, [bar(11), bar(12)], [0, 1], { ...clamp, easing: ease })
  const cloudOpacity = interpolate(f, [bar(16.8), bar(17.8)], [1, 0], clamp)
  const spin = -0.7 + f * 0.004
  const endCloud = interpolate(f, [bar(34), bar(36)], [0, 1], { ...clamp, easing: ease })

  return (
    <AbsoluteFill style={{ background: C.bg, fontFamily: SANS }}>
      <Audio src={staticFile('intro.wav')} endAt={INTRO_FRAMES} />
      <Backdrop aura={interpolate(f, [0, bar(1.5)], [0, 1], clamp)} />

      {f < bar(18) && (
        <DotCloud data={cloud} size={1000} x={cloudX} y={cloudY} scale={cloudScale} assemble={assemble} dissolve={dissolve}
          highlight={highlight} spin={spin} tilt={0.42} dot={1.9} opacity={cloudOpacity} />
      )}

      {/* ---- 1. title (bars 0-3) */}
      <Sequence from={bar(1.6)} durationInFrames={bar(2.4)}>
        <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 120 }}>
          <Brand at={0} size={58} />
          <div style={{ height: 22 }} />
          <Rise text="Simulation-ready CAD, in a sentence." at={beat(2)} size={46} weight={500} color={C.ink2} align="center" out={bar(2.1)} />
        </AbsoluteFill>
      </Sequence>

      {/* ---- 2. problem (bars 4-7) */}
      <Sequence from={bar(4)} durationInFrames={bar(4)}>
        <AbsoluteFill style={{ padding: '0 0 0 140px', justifyContent: 'center' }}>
          <div style={{ width: 760 }}>
            <Fade at={0}><Pill>The problem</Pill></Fade>
            <div style={{ height: 34 }} />
            <Rise text="Real parts carry detail your simulation doesn’t need." at={beat(1)} size={78} out={bar(3.8)} highlight={['detail']} />
            <div style={{ height: 40 }} />
            <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
              {[['7', 'holes'], ['8', 'chamfers'], ['6', 'fillets'], ['1', 'logo'], ['2', 'gussets']].map(([n, l], i) => (
                <Fade key={l} at={bar(1) + beat(i)} out={bar(3.8)}>
                  <div style={{ fontFamily: MONO, fontSize: 26, color: C.ink, padding: '10px 18px', borderRadius: 12, border: `1px solid ${C.line}`, background: 'rgba(27,31,35,0.7)' }}>
                    <span style={{ color: C.orange }}>{n}</span> {l}
                  </div>
                </Fade>
              ))}
            </div>
            <Fade at={bar(2.5)} out={bar(3.8)}><div style={{ marginTop: 34, fontSize: 30, color: C.ink3 }}>Each one multiplies mesh elements and solve time.</div></Fade>
          </div>
        </AbsoluteFill>
      </Sequence>

      {/* ---- 3. how it works (bars 8-15) */}
      {STEPS.map((s, i) => (
        <Sequence key={s.n} from={bar(8 + i)} durationInFrames={bar(1)}>
          <AbsoluteFill style={{ padding: '0 0 0 140px', justifyContent: 'center' }}>
            <div style={{ width: 760 }}>
              <div style={{ fontFamily: PIXEL, fontSize: 150, color: C.orange, lineHeight: 1 }}>
                <Rise text={s.n} at={0} size={150} font={PIXEL} color={C.orange} tracking={0} out={bar(0.86)} />
              </div>
              <div style={{ height: 24 }} />
              <Rise text={s.t} at={4} size={70} out={bar(0.86)} />
              <div style={{ height: 22 }} />
              <Fade at={10} out={bar(0.86)}><div style={{ fontFamily: MONO, fontSize: 26, color: C.ink2 }}>{s.s}</div></Fade>
            </div>
          </AbsoluteFill>
        </Sequence>
      ))}
      <Sequence from={bar(12)} durationInFrames={bar(2)}>
        <AbsoluteFill style={{ left: 1080, top: 760, width: 700, height: 200, display: 'flex', flexDirection: 'row', gap: 18 }}>
          {[['−55%', 'mesh elements'], ['84 → 31', 'faces'], ['−3.0%', 'deflection']].map(([v, l], i) => (
            <Fade key={l} at={beat(i)} out={bar(1.85)}>
              <div style={{ padding: '18px 22px', borderRadius: 18, background: 'rgba(27,31,35,0.75)', border: `1px solid ${C.line}` }}>
                <div style={{ fontSize: 44, fontWeight: 600, color: i === 0 ? C.orange : C.ink, letterSpacing: '-0.03em' }}>{v}</div>
                <div style={{ fontSize: 20, color: C.ink3 }}>{l}</div>
              </div>
            </Fade>
          ))}
        </AbsoluteFill>
      </Sequence>
      <Sequence from={bar(14)} durationInFrames={bar(2)}>
        <AbsoluteFill style={{ padding: '0 0 0 140px', justifyContent: 'center' }}>
          <div style={{ width: 780 }}>
            <Rise text="23 features gone. Same part, ready for the solver." at={0} size={74} out={bar(1.85)} highlight={['gone']} />
          </div>
        </AbsoluteFill>
      </Sequence>

      {/* ---- 4. breakdown (bars 16-17) */}
      <Sequence from={bar(16.4)} durationInFrames={bar(1.6)}>
        <AbsoluteFill style={{ justifyContent: 'center', alignItems: 'center' }}>
          <Rise text="Here’s how it feels." at={beat(1)} size={110} align="center" out={bar(1.45)} />
        </AbsoluteFill>
      </Sequence>

      {/* ---- 5. product montage (bars 18-29), one screen per 2 bars */}
      <Sequence from={bar(18)} durationInFrames={bar(2)}>
        <Screen src="welcome" at={0} zoom={[1.0, 1.08]} focus={[0.7, 0.45]} dur={bar(2)} />
        <Caption at={beat(2)} n="01" text="Drop a STEP file, or describe a part" out={bar(1.85)} />
      </Sequence>
      <Sequence from={bar(20)} durationInFrames={bar(2)}>
        <Screen src="features" at={0} zoom={[1.0, 1.1]} focus={[0.85, 0.4]} dur={bar(2)}>
          <Spot b={B['features.side']} at={beat(3)} pad={4} />
        </Screen>
        <Caption at={beat(2)} n="02" text="Every feature, with a reason" out={bar(1.85)} />
      </Sequence>
      <Sequence from={bar(22)} durationInFrames={bar(2)}>
        <Screen src="isolate" at={0} zoom={[1.0, 1.08]} focus={[0.3, 0.7]} dur={bar(2)}>
          <Cursor path={[{ t: 0, x: 900, y: 600 }, { t: beat(2), x: (B['isolate.chip']?.x ?? 300) + 60, y: (B['isolate.chip']?.y ?? 1040) + 16, click: true }, { t: bar(2), x: 760, y: 820 }]} />
        </Screen>
        <Caption at={beat(2)} n="03" text="Isolate any feature class" out={bar(1.85)} />
      </Sequence>
      <Sequence from={bar(24)} durationInFrames={bar(2)}>
        <Screen src="cmdk" at={0} zoom={[1.0, 1.12]} focus={[0.5, 0.25]} dur={bar(2)} />
        <Caption at={beat(2)} n="04" text="⌘K for everything" out={bar(1.85)} />
      </Sequence>
      <Sequence from={bar(26)} durationInFrames={bar(2)}>
        <Screen src="results" at={0} zoom={[1.0, 1.16]} focus={[0.02, 0.35]} dur={bar(2)}>
          <Spot b={B['results.stats']} at={beat(3)} pad={6} />
          <Spot b={B['results.fea']} at={bar(1)} pad={6} />
        </Screen>
        <Caption at={beat(2)} n="05" text="Mesh and stress, proven on both models" out={bar(1.85)} />
      </Sequence>
      <Sequence from={bar(28)} durationInFrames={bar(2)}>
        <Wipe a="view_original" b="view_simplified" at={0} dur={bar(2)} />
        <Caption at={beat(2)} n="06" text="Before and after, side by side" out={bar(1.85)} />
      </Sequence>

      {/* ---- 6. numbers (bars 30-33) */}
      <Sequence from={bar(30)} durationInFrames={bar(4)}>
        <AbsoluteFill style={{ padding: '120px 140px', justifyContent: 'center' }}>
          <Fade at={0} out={bar(3.85)}><Pill>Measured on held-out, real CAD</Pill></Fade>
          <div style={{ height: 50 }} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '46px 80px' }}>
            {[
              { v: NUM.filletF1, d: 2, s: '', l: 'fillet F1 on 1,500 real designer parts' },
              { v: NUM.completed, d: 1, s: '%', l: `of ${NUM.parts} random real parts completed` },
              { v: NUM.valid, d: 0, s: '%', l: 'valid solids out, never a broken model' },
              { v: NUM.tets, d: 0, s: '%', p: '−', l: 'mesh elements on the sample bracket' },
            ].map((x, i) => (
              <Fade key={x.l} at={beat(2 + i * 2)} out={bar(3.85)}>
                <div style={{ borderTop: `1px solid ${C.line}`, paddingTop: 22 }}>
                  <Counter to={x.v} at={beat(2 + i * 2)} decimals={x.d} prefix={x.p || ''} suffix={x.s}
                    style={{ fontFamily: PIXEL, fontSize: 120, color: i === 0 ? C.orange : C.ink, lineHeight: 1 }} />
                  <div style={{ fontSize: 28, color: C.ink2, marginTop: 10 }}>{x.l}</div>
                </div>
              </Fade>
            ))}
          </div>
        </AbsoluteFill>
      </Sequence>

      {/* ---- 7. outro (bars 34-37) */}
      <Sequence from={bar(34)} durationInFrames={bar(4)}>
        <AbsoluteFill>
          <DotCloud data={cloud} size={760} y={-80} assemble={endCloud} dissolve={1} highlight={0} spin={-0.6 + f * 0.003} tilt={0.42} dot={1.6} />
          <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 150 }}>
            <Brand at={bar(1)} size={64} />
            <div style={{ height: 26 }} />
            <Rise text="Open source. Runs offline. Windows · Linux · macOS." at={bar(1.4)} size={38} weight={500} color={C.ink2} align="center" />
          </AbsoluteFill>
        </AbsoluteFill>
      </Sequence>

      <BeatTick from={bar(18)} to={bar(30)} />
      <AbsoluteFill style={{ background: '#000', opacity: interpolate(f, [bar(37.3), bar(38)], [0, 1], clamp) }} />
      <Grain opacity={0.05} />
    </AbsoluteFill>
  )
}

const Caption: React.FC<{ at: number; n: string; text: string; out?: number }> = ({ at, n, text, out }) => {
  const f = useCurrentFrame()
  return (
    <div style={{ position: 'absolute', left: 160, bottom: 34, display: 'flex', alignItems: 'baseline', gap: 18 }}>
      <Fade at={at} out={out}><span style={{ fontFamily: PIXEL, fontSize: 40, color: C.orange }}>{n}</span></Fade>
      <Rise text={text} at={at + 2} size={38} weight={500} out={out} stagger={2} />
    </div>
  )
}
