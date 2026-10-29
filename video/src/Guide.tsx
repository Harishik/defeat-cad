import React from 'react'
import { AbsoluteFill, Audio, interpolate, Sequence, staticFile, useCurrentFrame } from 'remotion'
import { Backdrop, bar, beat, Brand, C, clamp, Cursor, DotCloud, ease, Fade, Grain, MONO, Pill, PIXEL, Rise, SANS, Screen, Spot, Wipe } from './kit'
import { useBoxes, useCloud } from './data'

// 64 bars at 96 BPM = 160.0 s = 4800 frames
export const GUIDE_FRAMES = bar(64)

const CHAPTERS = [
  { n: '01', t: 'Install', from: 4, len: 6 },
  { n: '02', t: 'Open a part', from: 10, len: 6 },
  { n: '03', t: 'Read the feature map', from: 16, len: 8 },
  { n: '04', t: 'Choose the analysis', from: 24, len: 4 },
  { n: '05', t: 'Simplify & verify', from: 28, len: 8 },
  { n: '06', t: 'Compare & export', from: 36, len: 8 },
  { n: '07', t: 'Ask in plain English', from: 44, len: 8 },
  { n: '08', t: 'Batch runs & proof', from: 52, len: 8 },
]

export const Guide: React.FC = () => {
  const f = useCurrentFrame()
  const B = useBoxes()
  const cloud = useCloud('printer_chassis')
  const cur = CHAPTERS.findIndex((c) => f >= bar(c.from) && f < bar(c.from + c.len))
  return (
    <AbsoluteFill style={{ background: C.bg, fontFamily: SANS }}>
      <Audio src={staticFile('guide.wav')} endAt={GUIDE_FRAMES} />
      <Backdrop aura={0.8} tint="mix" />

      {/* ---- title (bars 0-3) */}
      <Sequence from={0} durationInFrames={bar(4)}>
        <DotCloud data={cloud} size={900} x={420} y={10} assemble={interpolate(f, [bar(0.2), bar(2.2)], [0, 1], { ...clamp, easing: ease })}
          highlight={1} spin={-0.6 + f * 0.004} tilt={0.75} dot={1.5} opacity={interpolate(f, [bar(3.5), bar(4)], [1, 0], clamp)} />
        <AbsoluteFill style={{ padding: '0 0 0 140px', justifyContent: 'center' }}>
          <Brand at={beat(1)} size={48} />
          <div style={{ height: 30 }} />
          <Rise text="The walkthrough" at={beat(3)} size={110} out={bar(3.7)} />
          <div style={{ height: 30 }} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px 40px', width: 760 }}>
            {CHAPTERS.map((c, i) => (
              <Fade key={c.n} at={bar(1.2) + beat(i * 0.5)} out={bar(3.7)}>
                <div style={{ fontSize: 28, color: C.ink2 }}><span style={{ fontFamily: PIXEL, color: C.orange, marginRight: 14 }}>{c.n}</span>{c.t}</div>
              </Fade>
            ))}
          </div>
        </AbsoluteFill>
      </Sequence>

      {/* ---- 01 install */}
      <Chapter i={0} />
      <Sequence from={bar(5)} durationInFrames={bar(5)}>
        <Terminal at={0} title="macOS / Linux" lines={[
          { c: '$ git clone https://github.com/Harishik/defeat-cad && cd defeat-cad', t: 0 },
          { c: '$ ./setup.sh', t: beat(4) },
          { o: 'Installing OpenCASCADE, gmsh, scikit-fem, PyTorch Geometric …', t: beat(5) },
          { o: '✓ Environment ready · optional: ollama pull qwen3.5:4b', t: beat(8), ok: true },
          { c: '$ ./run.sh', t: beat(10) },
          { o: 'DeFeat running at http://127.0.0.1:8000', t: beat(11), ok: true },
        ]} />
        <Fade at={bar(3)} out={bar(4.85)} style={{ position: 'absolute', right: 140, bottom: 120 }}>
          <div style={{ fontFamily: MONO, fontSize: 26, color: C.ink2, padding: '18px 22px', borderRadius: 16, border: `1px solid ${C.line}`, background: 'rgba(27,31,35,0.8)' }}>
            Windows: <span style={{ color: C.ink }}>setup.ps1</span> then <span style={{ color: C.ink }}>run.bat</span>
          </div>
        </Fade>
      </Sequence>

      {/* ---- 02 open a part */}
      <Chapter i={1} />
      <Sequence from={bar(11)} durationInFrames={bar(2.5)}>
        <Screen src="welcome" at={0} zoom={[1, 1.12]} focus={[0.3, 0.6]} dur={bar(2.5)}>
          <Spot b={B['welcome.command']} at={beat(3)} label="Drop a STEP file or describe a part" side="top" />
          <Cursor path={[{ t: 0, x: 1300, y: 300 }, { t: beat(5), x: (B['welcome.command']?.x ?? 500) + 120, y: (B['welcome.command']?.y ?? 700) + 40, click: true }]} />
        </Screen>
      </Sequence>
      <Sequence from={bar(13.5)} durationInFrames={bar(2.5)}>
        <Screen src="welcome_cards" at={0} zoom={[1, 1.06]} focus={[0.5, 0.6]} dur={bar(2.5)} />
        <Note at={beat(2)} out={bar(2.35)} text="Or start from a sample part. Each card previews what will be removed." />
      </Sequence>

      {/* ---- 03 feature map */}
      <Chapter i={2} />
      <Sequence from={bar(17)} durationInFrames={bar(2.5)}>
        <Screen src="features" at={0} zoom={[1, 1.35]} focus={[0.95, 0.25]} dur={bar(2.5)}>
          <Spot b={B['features.frow']} at={beat(4)} label="Plain-English reason + GNN second opinion" side="left" pad={4} />
        </Screen>
      </Sequence>
      <Sequence from={bar(19.5)} durationInFrames={bar(2.25)}>
        <Screen src="isolate_holes" at={0} zoom={[1, 1.1]} focus={[0.35, 0.6]} dur={bar(2.25)}>
          <Spot b={B['features.chips']} at={beat(2)} label="Click a chip to isolate a class" side="top" pad={4} />
        </Screen>
      </Sequence>
      <Sequence from={bar(21.75)} durationInFrames={bar(2.25)}>
        <Screen src="isolate" at={0} zoom={[1.05, 1.18]} focus={[0.4, 0.62]} dur={bar(2.25)} />
        <Note at={beat(2)} out={bar(2.1)} text="Bright = will be removed · faded = kept. Text, holes, fillets, chamfers, ribs, bosses, slots, steps." />
      </Sequence>

      {/* ---- 04 analysis */}
      <Chapter i={3} />
      <Sequence from={bar(25)} durationInFrames={bar(3)}>
        <Screen src="features" at={0} zoom={[1, 1.5]} focus={[0.62, 0.02]} dur={bar(3)}>
          <Spot b={B['features.presets']} at={beat(2)} label="Static · Modal · Thermal · CFD" side="bottom" pad={4} />
          <Spot b={B['features.h']} at={bar(1.4)} label="Target mesh size h" side="bottom" pad={4} />
        </Screen>
        <Note at={bar(1)} out={bar(2.85)} text="Features smaller than a multiple of h are removed. Ribs stay for stiffness." />
      </Sequence>

      {/* ---- 05 simplify & verify */}
      <Chapter i={4} />
      <Sequence from={bar(29)} durationInFrames={bar(2)}>
        <Screen src="loader" at={0} zoom={[1, 1.25]} focus={[0.5, 0.5]} dur={bar(2)} />
        <Note at={beat(2)} out={bar(1.85)} text="Guarded OpenCASCADE removal, then both models are meshed and solved." />
      </Sequence>
      <Sequence from={bar(31)} durationInFrames={bar(3)}>
        <Screen src="results" at={0} zoom={[1, 1.3]} focus={[0.02, 0.3]} dur={bar(3)}>
          <Spot b={B['results.stats']} at={beat(3)} pad={6} />
          <Spot b={B['results.fea']} at={bar(1.5)} pad={6} />
        </Screen>
      </Sequence>
      <Sequence from={bar(34)} durationInFrames={bar(2)}>
        <AbsoluteFill style={{ padding: '0 160px', justifyContent: 'center' }}>
          <Pill>New · stress guard</Pill>
          <div style={{ height: 30 }} />
          <Rise text="If simplifying moves deflection by more than 5%, DeFeat restores the key features automatically." at={beat(1)} size={62} out={bar(1.85)} highlight={['5%,', 'automatically']} />
        </AbsoluteFill>
      </Sequence>

      {/* ---- 06 compare & export */}
      <Chapter i={5} />
      <Sequence from={bar(37)} durationInFrames={bar(3)}>
        <Wipe a="view_original" b="view_simplified" at={0} dur={bar(3)} />
      </Sequence>
      <Sequence from={bar(40)} durationInFrames={bar(2)}>
        <Screen src="results" at={0} zoom={[1, 1.4]} focus={[0.02, 0.72]} dur={bar(2)}>
          <Spot b={B['results.dl']} at={beat(2)} pad={6} />
          <Spot b={B['results.code']} at={bar(1)} pad={6} />
        </Screen>
        <Note at={beat(3)} out={bar(1.85)} text="STEP, report, and Python / C++ that reproduce the result from the original file." />
      </Sequence>
      <Sequence from={bar(42)} durationInFrames={bar(2)}>
        <Screen src="report" at={0} zoom={[1, 1.12]} focus={[0.5, 0.2]} dur={bar(2)} />
        <Note at={beat(2)} out={bar(1.85)} text="A printable report with every decision and the stress check." />
      </Sequence>

      {/* ---- 07 assistant */}
      <Chapter i={6} />
      <Sequence from={bar(45)} durationInFrames={bar(2.5)}>
        <Screen src="chat_typed" at={0} zoom={[1, 1.25]} focus={[0.92, 0.85]} dur={bar(2.5)}>
          <Spot b={B['chat.input']} at={beat(3)} label="Plain English, local model" side="top" pad={4} />
        </Screen>
      </Sequence>
      <Sequence from={bar(47.5)} durationInFrames={bar(2.5)}>
        <Screen src="chat_reply" at={0} zoom={[1.55, 1.75]} focus={[0.97, 0.3]} dur={bar(2.5)} />
        <Note at={beat(2)} out={bar(2.35)} text="Every request becomes exact tool calls you can see. Nothing leaves the machine." />
      </Sequence>
      <Sequence from={bar(50)} durationInFrames={bar(2)}>
        <Screen src="cmdk" at={0} zoom={[1, 1.18]} focus={[0.5, 0.22]} dur={bar(2)} />
        <Note at={beat(2)} out={bar(1.85)} text="Or press ⌘K: presets, views, downloads, or ask anything." />
      </Sequence>

      {/* ---- 08 batch & proof */}
      <Chapter i={7} />
      <Sequence from={bar(53)} durationInFrames={bar(3)}>
        <Terminal at={0} title="Batch mode" lines={[
          { c: '$ python -m defeat.cli parts/*.step --preset static --out out/', t: 0 },
          { o: 'l_bracket.step        23/27 removed  faces 84→31   tets −55%  Δdeflection −3.0%', t: beat(3) },
          { o: 'printer_chassis.step  21/31 removed  faces 261→105 tets −42%  Δdeflection −0.4%', t: beat(5) },
          { o: '✓ STEP · report · Python · C++ written for every part', t: beat(7), ok: true },
        ]} />
      </Sequence>
      <Sequence from={bar(56)} durationInFrames={bar(4)}>
        <Bars />
      </Sequence>

      {/* ---- outro */}
      <Sequence from={bar(60)} durationInFrames={bar(4)}>
        <DotCloud data={cloud} size={760} y={-90} assemble={interpolate(f, [bar(60), bar(61.6)], [0, 1], { ...clamp, easing: ease })}
          dissolve={1} highlight={0} spin={-0.6 + f * 0.003} tilt={0.75} dot={1.4} />
        <AbsoluteFill style={{ justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 150 }}>
          <Brand at={bar(1)} size={60} />
          <div style={{ height: 24 }} />
          <Rise text="STEP in. Simulation-ready STEP out. Proven." at={bar(1.4)} size={40} weight={500} color={C.ink2} align="center" />
        </AbsoluteFill>
      </Sequence>

      {/* chapter rail */}
      {cur >= 0 && f >= bar(5) && (
        <div style={{ position: 'absolute', left: 160, right: 160, top: 34, display: 'flex', gap: 10 }}>
          {CHAPTERS.map((c, i) => {
            const p = i < cur ? 1 : i > cur ? 0 : interpolate(f, [bar(c.from), bar(c.from + c.len)], [0, 1], clamp)
            return (
              <div key={c.n} style={{ flex: c.len, height: 4, borderRadius: 2, background: C.line, overflow: 'hidden' }}>
                <div style={{ width: `${p * 100}%`, height: '100%', background: i === cur ? C.orange : C.ink3 }} />
              </div>
            )
          })}
        </div>
      )}
      <AbsoluteFill style={{ background: '#000', opacity: interpolate(f, [bar(63.3), bar(64)], [0, 1], clamp) }} />
      <Grain opacity={0.05} />
    </AbsoluteFill>
  )
}

/** One-bar chapter card: pixel number + title, then it shrinks into the corner label. */
const Chapter: React.FC<{ i: number }> = ({ i }) => {
  const c = CHAPTERS[i]
  const f = useCurrentFrame()
  const start = bar(c.from)
  const end = bar(c.from + c.len)
  if (f < start || f >= end) return null
  const local = f - start
  const k = interpolate(local, [bar(0.78), bar(1)], [0, 1], { ...clamp, easing: ease })
  return (
    <>
      {local < bar(1) && (
        <AbsoluteFill style={{ justifyContent: 'center', padding: '0 160px', opacity: 1 - k }}>
          <Rise text={c.n} at={start} size={200} font={PIXEL} color={C.orange} tracking={0} />
          <Rise text={c.t} at={start + 4} size={96} />
        </AbsoluteFill>
      )}
      {local >= bar(0.9) && (
        <div style={{ position: 'absolute', left: 160, top: 56, display: 'flex', gap: 16, alignItems: 'baseline', opacity: interpolate(local, [bar(0.9), bar(1.1)], [0, 1], clamp) }}>
          <span style={{ fontFamily: PIXEL, fontSize: 30, color: C.orange }}>{c.n}</span>
          <span style={{ fontSize: 28, color: C.ink, fontWeight: 500 }}>{c.t}</span>
        </div>
      )}
    </>
  )
}

const Note: React.FC<{ at: number; out?: number; text: string }> = ({ at, out, text }) => (
  <div style={{ position: 'absolute', left: 160, right: 160, bottom: 30 }}>
    <Fade at={at} out={out}>
      <div style={{ display: 'inline-block', fontSize: 32, color: C.ink, padding: '14px 22px', borderRadius: 16, background: 'rgba(18,20,23,0.82)',
        border: `1px solid ${C.line}`, backdropFilter: 'blur(14px)' }}>{text}</div>
    </Fade>
  </div>
)

type Line = { c?: string; o?: string; t: number; ok?: boolean }
const Terminal: React.FC<{ at: number; title: string; lines: Line[] }> = ({ at, title, lines }) => {
  const f = useCurrentFrame()
  const p = interpolate(f, [at, at + 18], [0, 1], { ...clamp, easing: ease })
  return (
    <div style={{ position: 'absolute', left: 160, right: 160, top: 210, borderRadius: 22, overflow: 'hidden', opacity: p,
      transform: `translateY(${(1 - p) * 40}px)`, background: 'rgba(25,28,32,0.92)', border: `1px solid ${C.line}`, boxShadow: '0 40px 120px rgba(0,0,0,0.55)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '16px 22px', borderBottom: `1px solid ${C.line}` }}>
        {['#ff5f57', '#febc2e', '#28c840'].map((c) => <span key={c} style={{ width: 14, height: 14, borderRadius: 7, background: c }} />)}
        <span style={{ marginLeft: 16, fontFamily: MONO, fontSize: 22, color: C.ink3 }}>{title}</span>
      </div>
      <div style={{ padding: '26px 30px', fontFamily: MONO, fontSize: 28, lineHeight: 1.7, minHeight: 340 }}>
        {lines.map((l, i) => {
          if (f < at + l.t) return null
          if (l.c) {
            const n = Math.floor(interpolate(f, [at + l.t, at + l.t + Math.max(10, l.c.length * 0.6)], [0, l.c.length], clamp))
            const caret = n < l.c.length || Math.floor(f / 15) % 2 === 0
            return <div key={i} style={{ color: C.ink }}>{l.c.slice(0, n)}{caret && i === lines.filter((x) => f >= at + x.t).length - 1 ? <span style={{ color: C.orange }}>▋</span> : null}</div>
          }
          return <div key={i} style={{ color: l.ok ? C.green : C.ink2, whiteSpace: 'pre' }}>{l.o}</div>
        })}
      </div>
    </div>
  )
}

const Bars: React.FC = () => {
  const f = useCurrentFrame()
  const rows = [
    { l: 'Fillets · F1 on 1,500 real parts', a: 0.67, b: 0.84 },
    { l: 'Chamfers · F1', a: 0.56, b: 0.71 },
    { l: 'Feature vs body · F1 on 2,000 MFCAD parts', a: 0.95, b: 0.95 },
    { l: 'Valid solids out · 500 real parts', a: 1.0, b: 1.0 },
  ]
  return (
    <AbsoluteFill style={{ padding: '170px 160px', justifyContent: 'center' }}>
      <Fade at={0} out={bar(3.85)}><Pill>Held-out benchmarks</Pill></Fade>
      <div style={{ height: 44 }} />
      {rows.map((r, i) => {
        const p = interpolate(f, [beat(2 + i * 2), beat(2 + i * 2) + 24], [0, 1], { ...clamp, easing: ease })
        const o = interpolate(f, [bar(3.85), bar(4)], [1, 0], clamp)
        return (
          <div key={r.l} style={{ marginBottom: 34, opacity: Math.min(p * 2, 1) * o }}>
            <div style={{ fontSize: 30, color: C.ink, marginBottom: 12 }}>{r.l}</div>
            <div style={{ position: 'relative', height: 30 }}>
              <div style={{ position: 'absolute', height: 12, top: 0, width: `${r.a * 70 * p}%`, background: '#3a4048', borderRadius: 6 }} />
              <div style={{ position: 'absolute', height: 14, top: 15, width: `${r.b * 70 * p}%`, background: C.orange, borderRadius: 7, boxShadow: `0 0 20px rgba(251,86,7,0.4)` }} />
              <span style={{ position: 'absolute', left: `calc(${r.b * 70 * p}% + 18px)`, top: 0, fontFamily: PIXEL, fontSize: 34, color: C.orange }}>{(r.b * p).toFixed(2)}</span>
            </div>
          </div>
        )
      })}
      <Fade at={beat(10)} out={bar(3.85)}><div style={{ fontFamily: MONO, fontSize: 22, color: C.ink3 }}>grey = rules only · orange = DeFeat (rules + GNN)</div></Fade>
    </AbsoluteFill>
  )
}
