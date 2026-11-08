import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion, LayoutGroup } from 'framer-motion'
import Viewer from './Viewer'
import DotPart from './DotPart'
import { api, CLASS_META } from './api'

const spring = { type: 'spring', bounce: 0, duration: 0.4 }
const springSoft = { type: 'spring', bounce: 0.15, duration: 0.45 }

function useDark() {
  const q = typeof window !== 'undefined' ? window.matchMedia('(prefers-color-scheme: dark)') : null
  const [d, setD] = useState(q?.matches ?? false)
  useEffect(() => {
    if (!q) return
    const f = (e) => setD(e.matches)
    q.addEventListener('change', f)
    return () => q.removeEventListener('change', f)
  }, [])
  return d
}

// ------------------------------------------------------------------ icons (inline, 1.6 stroke)
const I = {
  logo: (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none"><path d="M12 2.8 20 7.4v9.2L12 21.2 4 16.6V7.4L12 2.8Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round"/><path d="M4.5 7.6 12 12l7.5-4.4M12 12v8.8" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round"/></svg>
  ),
  upload: <svg viewBox="0 0 24 24" width="20" height="20" fill="none"><path d="M12 15V4m0 0-4 4m4-4 4 4M5 15v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>,
  send: <svg viewBox="0 0 24 24" width="18" height="18" fill="none"><path d="M12 19V5m0 0-6 6m6-6 6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>,
  play: <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M7 5.5v13a1 1 0 0 0 1.5.86l10.6-6.5a1 1 0 0 0 0-1.72L8.5 4.64A1 1 0 0 0 7 5.5Z"/></svg>,
  check: <svg viewBox="0 0 24 24" width="14" height="14" fill="none"><path d="m5 12.5 4.5 4.5L19 7.5" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/></svg>,
  back: <svg viewBox="0 0 24 24" width="18" height="18" fill="none"><path d="m14.5 6-6 6 6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>,
  dl: <svg viewBox="0 0 24 24" width="16" height="16" fill="none"><path d="M12 4v11m0 0 4-4m-4 4-4-4M5 19h14" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/></svg>,
  sparkle: <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M12 2.5c.5 4.6 2.4 6.9 7 7.5-4.6.6-6.5 2.9-7 7.5-.5-4.6-2.4-6.9-7-7.5 4.6-.6 6.5-2.9 7-7.5Zm6.5 12c.25 2 1.1 3 3 3.25-1.9.25-2.75 1.25-3 3.25-.25-2-1.1-3-3-3.25 1.9-.25 2.75-1.25 3-3.25Z"/></svg>,
  close: <svg viewBox="0 0 24 24" width="16" height="16" fill="none"><path d="M6 6l12 12M18 6 6 18" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>,
}

// ------------------------------------------------------------------ helpers
const fmt = (n, d = 1) => (n == null ? '—' : Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 }))

function Pressable({ className = '', children, ...p }) {
  return (
    <motion.button whileTap={{ scale: 0.96 }} transition={{ type: 'spring', bounce: 0, duration: 0.18 }} className={className} {...p}>
      {children}
    </motion.button>
  )
}

// ------------------------------------------------------------------ welcome
const STATS = [
  { v: '0.988', l: 'fillet F1 on 6,454 real designer parts' },
  { v: '500', l: 'fresh real-world parts run end to end' },
  { v: '100%', l: 'valid solids out on those 500 parts' },
  { v: '−50%', l: 'mesh elements on the sample bracket' },
]

function usePreview(name) {
  const [d, setD] = useState(null)
  useEffect(() => { if (name) fetch(`/api/preview/${encodeURIComponent(name)}`).then((r) => r.json()).then(setD).catch(() => {}) }, [name])
  return d
}

function SampleCard({ s, i, onOpen, busy, dark }) {
  const data = usePreview(s.name)
  return (
    <motion.button className="card" onClick={() => onOpen(s.name)} disabled={busy}
      initial={{ y: 24, opacity: 0 }} whileInView={{ y: 0, opacity: 1 }} viewport={{ once: true }}
      transition={{ ...spring, delay: i * 0.06 }} whileTap={{ scale: 0.985 }}>
      <div className="card-media">
        {data && <DotPart data={data} count={9000} size={1.35} interactive={false} distance={1.05}
          accent="#fb5607" base={dark ? '#cdd3e8' : '#212529'} />}
        <span className="badge">{data ? `${data.remove} of ${data.features} features removable` : 'Loading…'}</span>
      </div>
      <div className="card-foot">
        <div>
          <div className="card-title">{s.title}</div>
          <div className="card-meta mono">{s.name} · {data ? `${data.faces} faces` : '—'} · {s.kb} KB</div>
        </div>
        <span className="arrow">↗</span>
      </div>
    </motion.button>
  )
}

function Welcome({ onOpen, onUpload, onCreate, busy, llm, dark }) {
  const [samples, setSamples] = useState([])
  const [drag, setDrag] = useState(false)
  const [prompt, setPrompt] = useState('')
  const file = useRef()
  const promptRef = useRef()
  const hero = usePreview('l_bracket.step')
  useEffect(() => { api.samples().then(setSamples).catch(() => {}) }, [])
  useEffect(() => {
    const k = (e) => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); promptRef.current?.focus() } }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [])
  const rise = (d) => ({ initial: { y: 22, opacity: 0, filter: 'blur(6px)' }, animate: { y: 0, opacity: 1, filter: 'blur(0px)' }, transition: { ...spring, duration: 0.6, delay: d } })
  return (
    <motion.div className="welcome" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, scale: 0.985 }} transition={spring}
      onDragOver={(e) => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; f && onUpload(f) }}>
      <div className="aura" aria-hidden />
      <div className="grid-bg" aria-hidden />
      <nav className="wnav">
        <div className="brand"><DotMark /><span>DeFeat</span></div>
        <button className="kbar" onClick={() => promptRef.current?.focus()}>
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none"><circle cx="11" cy="11" r="6.5" stroke="currentColor" strokeWidth="1.7"/><path d="m16 16 4 4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"/></svg>
          <span>Describe a part or a task</span><kbd>⌘K</kbd>
        </button>
        <div className={`status ${llm ? 'on' : ''}`}><span className="led" />{llm ? 'Local AI online' : 'Keyword mode'}</div>
      </nav>

      <section className="whero">
        <div className="whero-copy">
          <motion.div className="pill-badge" {...rise(0.05)}><span className="spark" />Open source · runs fully offline</motion.div>
          <motion.h1 {...rise(0.1)}><span className="nowrap">Simulation‑ready</span> <span className="nowrap">CAD, in a</span> sentence.</motion.h1>
          <motion.p className="lede" {...rise(0.16)}>
            Drop in a mold part. DeFeat finds its <b>holes, fillets, ribs, bosses and logos</b>, decides what matters for <b>your analysis</b>,
            and hands back a clean STEP, <b>checked</b> by meshing and a stress check on both models.
          </motion.p>

          <motion.div className={`command ${drag ? 'drag' : ''}`} {...rise(0.22)}>
            <button className="drop-row" onClick={() => file.current.click()}>
              <span className="drop-ico">{I.upload}</span>
              <span className="drop-txt"><b>Drop a STEP file</b> or click to choose<span className="mono dim"> · STEP · IGES · BREP</span></span>
              <span className="drop-go">↗</span>
            </button>
            <input ref={file} type="file" accept=".step,.stp,.iges,.igs,.brep" hidden onChange={(e) => e.target.files[0] && onUpload(e.target.files[0])} />
            <form className="ask-row" onSubmit={(e) => { e.preventDefault(); prompt.trim() && onCreate(prompt.trim()) }}>
              <span className="ask-ico">{I.sparkle}</span>
              <input ref={promptRef} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="…or describe a part: “a 60 mm flange with six M6 bolt holes”" />
              <Pressable className="send" type="submit" disabled={!prompt.trim()}>{I.send}</Pressable>
            </form>
          </motion.div>
          <motion.div className="fine mono" {...rise(0.28)}>Nothing leaves this machine · OpenCASCADE · gmsh · PyTorch · Ollama</motion.div>
        </div>

        <motion.div className="whero-art" initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1 }} transition={{ ...spring, duration: 1, delay: 0.15 }}>
          {hero && <DotPart data={hero} count={16000} size={1.6} accent="#fb5607" base={dark ? '#d6dbee' : '#212529'} />}
          <div className="art-legend mono">
            <span><i className="lg acc" />removed for simulation</span>
            <span><i className="lg base" />kept</span>
          </div>
        </motion.div>
      </section>

      <motion.section className="stats-row" {...rise(0.34)}>
        {STATS.map((s) => (
          <div className="bigstat" key={s.v}><div className="bigstat-v">{s.v}</div><div className="bigstat-l">{s.l}</div></div>
        ))}
      </motion.section>

      <section className="cards">
        <div className="section-head"><span className="mono dim">01</span><h2>Try a sample part</h2><span className="mono dim">{samples.length} parts</span></div>
        <div className="card-grid">
          {samples.map((s, i) => <SampleCard key={s.name} s={s} i={i} onOpen={onOpen} busy={busy} dark={dark} />)}
        </div>
      </section>
      <AnimatePresence>{busy && <Loader steps={busy.steps} title={busy.title} />}</AnimatePresence>
    </motion.div>
  )
}

function DotMark() {
  // 5x5 dot-matrix "D" with one accent dot: the removed feature
  const on = ['11111', '10001', '10001', '10001', '11110']
  return (
    <svg viewBox="0 0 25 25" width="22" height="22" aria-hidden>
      {on.flatMap((row, y) => [...row].map((c, x) => c === '1' &&
        <circle key={`${x}${y}`} cx={x * 5 + 2.5} cy={y * 5 + 2.5} r="1.7" fill={x === 4 && y === 4 ? 'var(--accent)' : 'currentColor'} />))}
      <circle cx={22.5} cy={22.5} r="1.7" fill="var(--accent)" />
    </svg>
  )
}

// ------------------------------------------------------------------ staged loader
function DotLoader() {
  return (
    <div className="dotloader" aria-hidden>
      {Array.from({ length: 49 }, (_, i) => {
        const x = i % 7, y = Math.floor(i / 7)
        const d = Math.hypot(x - 3, y - 3)
        return <span key={i} style={{ animationDelay: `${d * 0.11}s`, opacity: d > 3.4 ? 0 : 1 }} />
      })}
    </div>
  )
}

function Loader({ title, steps = [], progress }) {
  return (
    <motion.div className="loader-scrim" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
      <motion.div className="loader glass" initial={{ scale: 0.94, opacity: 0, filter: 'blur(8px)' }} animate={{ scale: 1, opacity: 1, filter: 'blur(0px)' }}
        exit={{ scale: 0.97, opacity: 0, filter: 'blur(6px)' }} transition={spring}>
        <DotLoader />
        <div className="loader-title">{title}</div>
        {progress != null && (
          <div className="bar"><motion.div className="bar-fill" animate={{ width: `${Math.max(4, progress * 100)}%` }} transition={spring} /></div>
        )}
        <div className="steps">
          <AnimatePresence initial={false}>
            {steps.slice(-5).map((s, i, arr) => (
              <motion.div key={s + i + arr.length} className={`step ${i === arr.length - 1 ? 'live' : 'done'}`} layout
                initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={spring}>
                <span className="tick">{i === arr.length - 1 ? <span className="pulse" /> : I.check}</span>{s}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </motion.div>
    </motion.div>
  )
}

// ------------------------------------------------------------------ feature panel
function FeaturePanel({ sum, selected, setSelected, hovered, setHovered, onToggle, onClass }) {
  const groups = useMemo(() => {
    const g = {}
    for (const f of sum.features) (g[f.kind] ||= []).push(f)
    return Object.entries(g).sort((a, b) => Object.keys(CLASS_META).indexOf(a[0]) - Object.keys(CLASS_META).indexOf(b[0]))
  }, [sum])
  const listRef = useRef()
  useEffect(() => {
    if (selected == null) return
    const box = listRef.current, row = box?.querySelector(`[data-f="${selected}"]`)
    if (!box || !row) return
    const top = row.offsetTop - box.offsetTop, bottom = top + row.offsetHeight
    if (top < box.scrollTop + 40) box.scrollTo({ top: top - 48, behavior: 'smooth' })
    else if (bottom > box.scrollTop + box.clientHeight) box.scrollTo({ top: bottom - box.clientHeight + 12, behavior: 'smooth' })
  }, [selected])
  return (
    <div className="features" ref={listRef}>
      {groups.map(([kind, fs]) => {
        const meta = CLASS_META[kind] || CLASS_META.protrusion
        const nRm = fs.filter((f) => f.decision.action === 'remove').length
        return (
          <div className="fgroup" key={kind}>
            <div className="fgroup-head">
              <span className="swatch" style={{ background: meta.color }} />
              <span className="fgroup-title">{meta.label}</span>
              <span className="count">{nRm}/{fs.length} removed</span>
              <div className="mini-seg">
                <button onClick={() => onClass(kind, 'remove')} className={nRm === fs.length ? 'on' : ''}>All</button>
                <button onClick={() => onClass(kind, 'auto')} className={nRm > 0 && nRm < fs.length ? 'on' : ''}>Auto</button>
                <button onClick={() => onClass(kind, 'keep')} className={nRm === 0 ? 'on' : ''}>None</button>
              </div>
            </div>
            {fs.map((f) => {
              const rm = f.decision.action === 'remove'
              return (
                <motion.div layout key={f.id} data-f={f.id} className={`frow ${selected === f.id ? 'sel' : ''} ${hovered === f.id ? 'hov' : ''}`}
                  onMouseEnter={() => setHovered(f.id)} onMouseLeave={() => setHovered(null)} onClick={() => setSelected(selected === f.id ? null : f.id)}>
                  <div className="frow-main">
                    <div className="frow-title">{f.note}<span className="fid">#{f.id}</span></div>
                    <div className="frow-why">{f.decision.reason}</div>
                    {f.net && (
                      <div className={`ai ${f.net.agree ? 'yes' : 'no'}`} title={Object.entries(f.net.probs).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(' · ')}>
                        {I.sparkle}{f.net.agree ? `Network agrees · ${(f.net.agree_p * 100).toFixed(0)}%` : `Network says ${f.net.label} · ${(f.net.p * 100).toFixed(0)}% — review`}
                      </div>
                    )}
                  </div>
                  <button className={`toggle ${rm ? 'rm' : 'kp'}`} onClick={(e) => { e.stopPropagation(); onToggle(f.id, rm ? 'keep' : 'remove') }}
                    aria-label={rm ? 'Remove' : 'Keep'}>
                    <motion.span className="knob" layout transition={spring} />
                    <span className="tlabel">{rm ? 'Remove' : 'Keep'}</span>
                  </button>
                </motion.div>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}

// ------------------------------------------------------------------ chat
const SUGGESTIONS = [
  'Prep this for a modal analysis and run it',
  'Remove the logo and fillets under 3 mm, keep all holes',
  'Give me the Python code',
  'Build a CFD flow domain around it',
]

function ChatPanel({ sid, messages, setMessages, onResult, llm, queued, clearQueued }) {
  const [text, setText] = useState('')
  const [thinking, setThinking] = useState(false)
  const end = useRef()
  useEffect(() => {
    // scroll only the chat list; scrollIntoView would also scroll the page and shift the workspace
    const box = end.current?.parentElement
    if (box) box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' })
  }, [messages, thinking])
  const send = async (msg) => {
    const m = (msg ?? text).trim()
    if (!m || thinking) return
    setText('')
    setMessages((x) => [...x, { role: 'user', content: m }])
    setThinking(true)
    try {
      const r = await api.chat(sid, m)
      setMessages((x) => [...x, { role: 'assistant', content: r.reply, calls: r.calls, via: r.via, seconds: r.seconds }])
      onResult(r)
    } catch (e) {
      setMessages((x) => [...x, { role: 'assistant', content: `Something went wrong: ${e.message}` }])
    } finally {
      setThinking(false)
    }
  }
  useEffect(() => { if (queued) { clearQueued?.(); send(queued) } }, [queued])
  return (
    <div className="chat">
      <div className="chat-scroll">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-title">Tell DeFeat what you need</div>
            <div className="chat-empty-sub">Plain English works. I'll only change the model through the same exact tools as the buttons — you see every change before it runs.</div>
            <div className="sugs">
              {SUGGESTIONS.map((s) => <Pressable key={s} className="sug" onClick={() => send(s)}>{s}</Pressable>)}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <motion.div key={i} className={`msg ${m.role}`} initial={{ opacity: 0, y: 10, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={springSoft}>
            <div className="bubble">{m.content}</div>
            {m.calls?.length > 0 && (
              <div className="calls">
                {m.calls.map((c, k) => (
                  <span className="call" key={k}><code>{c.name}</code>{Object.keys(c.arguments || {}).length ? ` ${Object.entries(c.arguments).map(([a, b]) => `${a}=${Array.isArray(b) ? b.join(',') : b}`).join(' ')}` : ''}</span>
                ))}
              </div>
            )}
            {m.via && <div className="via">{m.via} · {m.seconds}s</div>}
          </motion.div>
        ))}
        {thinking && (
          <motion.div className="msg assistant" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <div className="bubble typing"><span /><span /><span /></div>
          </motion.div>
        )}
        <div ref={end} />
      </div>
      <form className="chat-input" onSubmit={(e) => { e.preventDefault(); send() }}>
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder={llm ? 'Ask or instruct…' : 'Instruct (keyword mode)…'} />
        <Pressable className="send" type="submit" disabled={!text.trim() || thinking}>{I.send}</Pressable>
      </form>
    </div>
  )
}

// ------------------------------------------------------------------ results sheet
const CHECKS = {
  static: { title: 'Static stress check', noun: 'load cases', sub: 'two bending directions, tension, torsion' },
  modal: { title: 'Natural frequency check', noun: 'modes', sub: 'first three natural frequencies' },
  thermal: { title: 'Heat path check', noun: 'heat paths', sub: 'conduction along each axis' },
  cfd: { title: 'Flow area check', noun: 'areas', sub: 'frontal area along each axis, wetted area' },
}
const CASE_LABEL = (k) => ({ bend_thin: 'Bending, thin way', bend_wide: 'Bending, wide way', tension: 'Tension', torsion: 'Torsion', your_load: 'Your load', your_heat_path: 'Your heat path',
  mode_1: 'Mode 1', mode_2: 'Mode 2', mode_3: 'Mode 3', conduction_x: 'Conduction, X', conduction_y: 'Conduction, Y', conduction_z: 'Conduction, Z',
  frontal_x: 'Frontal area, X', frontal_y: 'Frontal area, Y', frontal_z: 'Frontal area, Z', wetted: 'Wetted area' }[k] || k)

function BoundaryCard({ boundary, mode, setMode, onChange, onClose }) {
  const b = { fixed: [], loaded: [], force: [0, 0, -100], ...boundary }
  const setF = (i, v) => { const force = [...b.force]; force[i] = +v || 0; onChange({ ...b, force }) }
  return (
    <motion.div className="bcard glass" initial={{ y: 12, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 12, opacity: 0 }} transition={spring}>
      <div className="bcard-head"><b>Supports &amp; loads</b><button className="icon-btn" onClick={onClose} aria-label="Close">{I.close}</button></div>
      <div className="seg small">
        <button className={mode === 'fixed' ? 'on' : ''} onClick={() => setMode('fixed')}><span className="dot-fixed" />Support · {b.fixed.length}</button>
        <button className={mode === 'loaded' ? 'on' : ''} onClick={() => setMode('loaded')}><span className="dot-load" />Load · {b.loaded.length}</button>
      </div>
      <p className="bcard-hint">Click faces on the part. Features touching them are kept, and the check uses these instead of the automatic ones.</p>
      {mode === 'loaded' && (
        <div className="bcard-force">
          {['Fx', 'Fy', 'Fz'].map((l, i) => <label key={l}>{l}<input type="number" value={b.force[i]} onChange={(e) => setF(i, e.target.value)} /><span>N</span></label>)}
        </div>
      )}
      {(b.fixed.length > 0 || b.loaded.length > 0) && <button className="link" onClick={() => onChange({ fixed: [], loaded: [], force: b.force })}>Clear and use automatic</button>}
    </motion.div>
  )
}

function Stat({ label, value, sub, accent }) {
  return (
    <div className="stat">
      <div className="stat-v" style={accent ? { color: accent } : null}>{value}</div>
      <div className="stat-l">{label}</div>
      {sub && <div className="stat-s">{sub}</div>}
    </div>
  )
}

function Results({ sid, res, onClose }) {
  const [tab, setTab] = useState('python')
  const [verify, setVerify] = useState(null)
  const g = res.metrics.geometry, m = res.metrics.mesh, fea = res.metrics.fea
  const failed = Object.keys(res.metrics.failed || {}).length
  const worst = fea?.delta?.cases_delta_pct ? Math.max(...Object.values(fea.delta.cases_delta_pct).map(Math.abs)) : null
  const guard = res.metrics.guard
  const analysis = res.summary?.policy?.preset || 'static'
  const check = CHECKS[analysis] || CHECKS.static
  const nCases = Object.keys(fea?.delta?.cases_delta_pct || {}).length
  const guardOutside = guard?.triggered && (!guard.accepted || (guard.within_tol === false))
  const unchanged = res.metrics.removed.length === 0
  const solveSpeedup = fea?.original && fea?.defeatured ? fea.original.seconds / Math.max(fea.defeatured.seconds, 0.01) : null
  const signed = (v, digits = 0) => `${v > 0 ? '−' : v < 0 ? '+' : ''}${fmt(Math.abs(v), digits)}%`
  return (
    <motion.div className="results glass" initial={{ y: 40, opacity: 0, scale: 0.98 }} animate={{ y: 0, opacity: 1, scale: 1 }} exit={{ y: 40, opacity: 0 }} transition={spring}>
      <div className="results-head">
        <div>
          <div className="results-title">{!unchanged ? 'Ready for meshing' : guard?.undone ? 'Kept as designed' : 'Nothing to simplify'}</div>
          <div className="results-sub">{unchanged && guard?.undone ? 'The stress guard undid every removal, so this is your original part' : `${res.metrics.removed.length} features removed`}{failed ? ` · ${failed} kept (kernel could not remove)` : ''}{unchanged ? '' : ` · ${res.metrics.strategy}`} · {fmt(res.metrics.seconds, 1)} s</div>
        </div>
        <Pressable className="icon-btn" onClick={onClose}>{I.close}</Pressable>
      </div>
      <div className="stats">
        {m?.tet_reduction_pct != null && <Stat label="Mesh elements" value={signed(m.tet_reduction_pct)} sub={`${fmt(m.original.tets, 0)} → ${fmt(m.defeatured.tets, 0)} tets`} accent="var(--accent)" />}
        <Stat label="Faces" value={`${g.faces[0]} → ${g.faces[1]}`} sub={signed(100 * (1 - g.faces[1] / g.faces[0]))} />
        <Stat label="Volume change" value={`${g.volume_delta_pct >= 0 ? '+' : ''}${fmt(g.volume_delta_pct, 2)}%`} sub={g.bbox_changed ? 'bounding box changed' : 'bounding box unchanged'} />
        <Stat label="Max deviation" value={`${fmt(g.deviation.max, 2)} mm`} sub={`mean ${fmt(g.deviation.mean, 2)} mm`} />
        {m?.defeatured?.ok && <Stat label="Mesh quality (min)" value={fmt(m.defeatured.q_min, 2)} sub={`was ${fmt(m.original.q_min, 2)} · SICN`} />}
        {!unchanged && m?.speedup >= 1.1 && <Stat label="Meshing time" value={`${fmt(m.speedup, 1)}× faster`} sub={`${fmt(m.original.seconds, 2)} s → ${fmt(m.defeatured.seconds, 2)} s`} />}
      </div>
      {fea?.delta && (
        <div className="fea">
          <div className="fea-head">
            <div className="fea-title">{check.title}</div>
            <div className="fea-sub">{fea.load_case}</div>
          </div>
          {res.metrics.guard_message && <div className={`fea-guard${guardOutside ? ' warn' : ''}`}><b>{analysis === 'static' ? 'Stress guard.' : 'Guard.'}</b> {res.metrics.guard_message}</div>}
          <div className="stats">
            {worst != null && <Stat label={nCases > 1 ? `Worst of ${nCases} ${check.noun}` : 'Your case'} value={`${fmt(worst, 2)}%`} sub={nCases > 1 ? check.sub : 'with your supports and loads'} accent={worst <= 5 ? 'var(--green)' : '#ff9f0a'} />}
            {analysis !== 'static' && Object.entries(fea.delta.cases_delta_pct || {}).map(([k, v]) => <Stat key={k} label={CASE_LABEL(k)} value={`${v >= 0 ? '+' : ''}${fmt(v, 2)}%`} sub={`${fmt(fea.original.cases[k], k.startsWith('mode') ? 0 : 3)} → ${fmt(fea.defeatured.cases[k], k.startsWith('mode') ? 0 : 3)}${k.startsWith('mode') ? ' Hz' : ''}`} />)}
            {analysis === 'static' && <Stat label="Tip deflection" value={`${fea.delta.tip_disp_delta_pct >= 0 ? '+' : ''}${fmt(fea.delta.tip_disp_delta_pct, 2)}%`} sub={`${fmt(fea.original.tip_disp, 4)} → ${fmt(fea.defeatured.tip_disp, 4)} mm`} accent={Math.abs(fea.delta.tip_disp_delta_pct) < 5 ? 'var(--green)' : '#ff9f0a'} />}
            {analysis === 'static' && <Stat label="Peak stress (p99)" value={`${fea.delta.vm_p99_delta_pct >= 0 ? '+' : ''}${fmt(fea.delta.vm_p99_delta_pct, 2)}%`} sub={`${fmt(fea.original.vm_p99, 2)} → ${fmt(fea.defeatured.vm_p99, 2)} MPa`} accent={Math.abs(fea.delta.vm_p99_delta_pct) < 5 ? 'var(--green)' : '#ff9f0a'} />}
            {!unchanged && solveSpeedup >= 1.1 && <Stat label="Solve time" value={`${fmt(solveSpeedup, 1)}× faster`} sub={`${fmt(fea.original.dofs, 0)} → ${fmt(fea.defeatured.dofs, 0)} DOFs`} />}
          </div>
        </div>
      )}
      <div className="downloads">
        <a className="dl primary" href={api.download(sid, 'step')}>{I.dl}Simplified STEP</a>
        <a className="dl" href={api.download(sid, 'report')} target="_blank" rel="noreferrer">{I.dl}Report</a>
        <a className="dl" href={api.download(sid, 'py')}>{I.dl}Python</a>
        <a className="dl" href={api.download(sid, 'cpp')}>{I.dl}C++</a>
      </div>
      <div className="code-card">
        <div className="code-head">
          <div className="seg small">
            {['python', 'cpp'].map((t) => <button key={t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>{t === 'python' ? 'Python' : 'C++'}</button>)}
          </div>
          <Pressable className="verify" onClick={async () => { setVerify('running'); try { setVerify(await api.verify(sid)) } catch (e) { setVerify({ ok: false }) } }}>
            {verify === 'running' ? 'Re-running script…' : verify?.ok ? <>{I.check} Reproduces result exactly ({verify.faces[0]} faces)</> : verify ? 'Verification failed' : 'Verify script reproduces this result'}
          </Pressable>
        </div>
        <pre className="code"><code>{res.code[tab]}</code></pre>
      </div>
    </motion.div>
  )
}

// ------------------------------------------------------------------ workspace
// ------------------------------------------------------------------ command palette
function CommandPalette({ open, onClose, commands, onAsk }) {
  const [q, setQ] = useState('')
  const [i, setI] = useState(0)
  const inp = useRef()
  useEffect(() => { if (open) { setQ(''); setI(0); setTimeout(() => inp.current?.focus(), 30) } }, [open])
  const list = useMemo(() => {
    const t = q.trim().toLowerCase()
    const hits = commands.filter((c) => !t || (c.label + ' ' + (c.group || '') + ' ' + (c.keys || '')).toLowerCase().includes(t))
    return t ? [...hits, { id: 'ask', group: 'Assistant', label: `Ask the assistant: “${q.trim()}”`, run: () => onAsk(q.trim()), icon: I.sparkle }] : hits
  }, [q, commands, onAsk])
  useEffect(() => { setI(0) }, [q])
  const go = (c) => { onClose(); c && setTimeout(() => c.run(), 60) }
  return (
    <AnimatePresence>
      {open && (
        <motion.div className="cmdk-scrim" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }} onMouseDown={onClose}>
          <motion.div className="cmdk glass" onMouseDown={(e) => e.stopPropagation()}
            initial={{ y: -12, scale: 0.97, opacity: 0, filter: 'blur(6px)' }} animate={{ y: 0, scale: 1, opacity: 1, filter: 'blur(0px)' }}
            exit={{ y: -8, scale: 0.98, opacity: 0, filter: 'blur(4px)' }} transition={spring}>
            <div className="cmdk-input">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none"><circle cx="11" cy="11" r="6.5" stroke="currentColor" strokeWidth="1.7"/><path d="m16 16 4 4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"/></svg>
              <input ref={inp} value={q} onChange={(e) => setQ(e.target.value)} placeholder="Type a command, or ask anything…"
                onKeyDown={(e) => {
                  if (e.key === 'ArrowDown') { e.preventDefault(); setI((v) => Math.min(list.length - 1, v + 1)) }
                  else if (e.key === 'ArrowUp') { e.preventDefault(); setI((v) => Math.max(0, v - 1)) }
                  else if (e.key === 'Enter') { e.preventDefault(); go(list[i]) }
                  else if (e.key === 'Escape') onClose()
                }} />
              <kbd>esc</kbd>
            </div>
            <div className="cmdk-list">
              {list.length === 0 && <div className="cmdk-empty">No matches</div>}
              {list.map((c, k) => (
                <div key={c.id} className={`cmdk-item ${k === i ? 'on' : ''}`} onMouseEnter={() => setI(k)} onClick={() => go(c)}>
                  {k === i && <motion.span layoutId="cmdk-hl" className="cmdk-hl" transition={{ type: 'spring', bounce: 0, duration: 0.25 }} />}
                  <span className="cmdk-ico">{c.icon || '›'}</span>
                  <span className="cmdk-label">{c.label}</span>
                  <span className="cmdk-group mono">{c.group}</span>
                  {c.keys && <kbd>{c.keys}</kbd>}
                </div>
              ))}
            </div>
            <div className="cmdk-foot mono"><span>↑↓ navigate</span><span>↵ run</span><span>DeFeat</span></div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function Workspace({ initial, onExit, dark, llm }) {
  const [sum, setSum] = useState(initial)
  const [orig, setOrig] = useState(null)
  const [resMesh, setResMesh] = useState(null)
  const [res, setRes] = useState(null)
  const [showRes, setShowRes] = useState(false)
  const [mode, setMode] = useState('original')
  const [wipe, setWipe] = useState(0.5)
  const wipeApi = useRef(null)
  const [tab, setTab] = useState('features')
  const [selected, setSelected] = useState(null)
  const [hovered, setHovered] = useState(null)
  const [run, setRun] = useState(null)
  const [messages, setMessages] = useState([])
  const [isolate, setIsolate] = useState(null)
  const [palette, setPalette] = useState(false)
  const [queued, setQueued] = useState(null)
  const [faceMode, setFaceMode] = useState(null)
  const sid = sum.id

  useEffect(() => {
    const k = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setPalette((v) => !v) }
      else if (e.key === 'Escape') setIsolate(null)
    }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [])

  useEffect(() => { api.mesh(sid, 'original').then(setOrig) }, [sid])

  // Policy changes go to the server one at a time and in order, so a quick second click can't overtake the first.
  const chain = useRef(Promise.resolve())
  const latestB = useRef(null)   // supports & loads as the user last set them, ahead of the server's reply
  const applyPolicy = useCallback((body) => {
    if (body.boundary) {
      latestB.current = body.boundary
      setSum((s) => ({ ...s, policy: { ...s.policy, boundary: body.boundary } }))   // show the pick at once
    }
    const job = chain.current.then(() => api.policy(sid, body)).then((r) => {
      if (body.boundary && latestB.current === body.boundary) latestB.current = null
      // a pick made since this request left stays on screen until its own reply arrives
      setSum(latestB.current ? { ...r, policy: { ...r.policy, boundary: latestB.current } } : r)
    })
    chain.current = job.catch(() => {})
    return job
  }, [sid])

  const poll = useCallback(async () => {
    setRun({ steps: ['Starting'], progress: 0.02 })
    for (;;) {
      await new Promise((r) => setTimeout(r, 350))
      const st = await api.status(sid)
      const s = st.status
      setRun({ steps: s.steps.map((x) => x.msg), progress: s.progress })
      if (!s.busy) {
        if (s.stage === 'done') {
          const [r, m] = await Promise.all([api.result(sid), api.mesh(sid, 'result')])
          setRes(r); setResMesh(m); setSum(r.summary)
          setRun(null); setShowRes(true); setMode('compare'); setWipe(0.5)
        } else {
          setRun(null)
          alert(s.error || 'Run failed')
        }
        return
      }
    }
  }, [sid])

  const runNow = async () => { await api.run(sid); poll() }

  const onChatResult = (r) => {
    if (r.summary) setSum(r.summary)
    for (const a of r.actions || []) {
      if (a.type === 'run') poll()
      if (a.type === 'focus' && a.features?.length) setSelected(a.features[0])
      if (a.type === 'download' && a.format === 'fluid') window.open(`/api/session/${sid}/download_fluid`)
      if (a.type === 'download' && res && a.format !== 'fluid') { setShowRes(true) }
      if (a.type === 'create_job') watchJob(a.job)
    }
  }
  const watchJob = async (jid) => {
    for (;;) {
      await new Promise((r) => setTimeout(r, 800))
      const j = await api.job(jid)
      if (j.done) {
        if (j.session) {
          setMessages((x) => [...x, { role: 'assistant', content: 'Part created. Opening it now — the code that built it is in this message.\n\n' + (j.code || '') }])
          const s = await api.session(j.session)
          onExit(s)
        } else setMessages((x) => [...x, { role: 'assistant', content: `I couldn't build a valid solid from that (${j.error}). Try adding dimensions.` }])
        return
      }
    }
  }

  const boundary = sum.policy.boundary || {}
  const toggleFace = (fid) => {
    const b = { fixed: [], loaded: [], force: [0, 0, -100], ...(latestB.current || boundary) }
    const other = faceMode === 'fixed' ? 'loaded' : 'fixed'
    const mine = b[faceMode].includes(fid) ? b[faceMode].filter((x) => x !== fid) : [...b[faceMode], fid]
    applyPolicy({ boundary: { ...b, [faceMode]: mine, [other]: b[other].filter((x) => x !== fid) } })
  }
  const rmCount = sum.features.filter((f) => f.decision.action === 'remove').length
  const preset = sum.policy.preset
  const dl = (kind) => () => window.open(api.download(sid, kind))
  const commands = [
    { id: 'run', group: 'Action', label: `Simplify — remove ${rmCount} features`, run: runNow, icon: I.play, keys: '↵' },
    ...Object.entries(sum.presets).map(([k, p]) => ({ id: `p-${k}`, group: 'Preset', label: `Prepare for ${p.label}`, run: () => applyPolicy({ preset: k }) })),
    ...(resMesh ? [['original', 'Show original'], ['compare', 'Compare before / after'], ['result', 'Show simplified']].map(([k, l]) => ({ id: `v-${k}`, group: 'View', label: l, run: () => setMode(k) })) : []),
    ...Object.keys(sum.feature_counts).map((k) => ({ id: `i-${k}`, group: 'Isolate', label: `Isolate ${(CLASS_META[k] || {}).label || k}`, run: () => setIsolate(k) })),
    { id: 'chat', group: 'Assistant', label: 'Open the assistant', run: () => setTab('chat'), icon: I.sparkle },
    ...(res ? [
      { id: 'd-step', group: 'Download', label: 'Download simplified STEP', run: dl('step'), icon: I.dl },
      { id: 'd-rep', group: 'Download', label: 'Open the report', run: dl('report'), icon: I.dl },
      { id: 'd-py', group: 'Download', label: 'Download Python script', run: dl('py'), icon: I.dl },
      { id: 'd-cpp', group: 'Download', label: 'Download C++ program', run: dl('cpp'), icon: I.dl },
      { id: 'res', group: 'View', label: 'Show results', run: () => setShowRes(true) },
    ] : []),
    { id: 'home', group: 'Navigate', label: 'Back to start', run: () => onExit(null), icon: I.back },
  ]
  const viewFeatures = isolate ? sum.features.filter((f) => f.kind === isolate) : sum.features
  return (
    <motion.div className="workspace" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
      <div className="aura" aria-hidden />
      <div className="ws-grid" aria-hidden />
      <Viewer original={orig} result={resMesh} features={viewFeatures} mode={mode} wipe={wipe} dark={dark} wipeApi={wipeApi}
        boundary={boundary} faceMode={faceMode} onFace={toggleFace}
        selected={selected} hovered={hovered} onPick={(id) => { setSelected(id); if (id != null) setTab('features') }} onHover={setHovered} />
      {mode === 'compare' && resMesh && (
        <>
          <div className="wipe-line"><span>Original</span><span>Simplified</span>
            {!faceMode && (
              <button className="wipe-grip" role="slider" aria-label="Compare position" aria-valuemin={2} aria-valuemax={98} aria-valuenow={Math.round(wipe * 100)}
                onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId) }}
                onPointerMove={(e) => {
                  if (!e.currentTarget.hasPointerCapture(e.pointerId)) return
                  const w = wipeApi.current?.xToWipe(e.clientX)
                  if (w != null) setWipe(w)
                }}
                onKeyDown={(e) => {
                  const d = e.key === 'ArrowLeft' ? -0.02 : e.key === 'ArrowRight' ? 0.02 : 0
                  if (d) { e.preventDefault(); setWipe((w) => Math.min(0.98, Math.max(0.02, w + d))) }
                }}>
                <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden><path d="M9 6l-6 6 6 6M15 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </button>
            )}
          </div>
        </>
      )}

      {/* top bar */}
      <motion.div className="topbar glass" initial={{ y: -20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={spring}>
        <Pressable className="icon-btn" onClick={() => onExit(null)} title="Back">{I.back}</Pressable>
        <div className="part">
          <div className="part-name">{sum.name}</div>
          <div className="part-sub">{sum.faces} faces · {sum.size.map((v) => fmt(v, 0)).join(' × ')} mm · {sum.features.length} features</div>
        </div>
        <div className="seg">
          <LayoutGroup id="preset">
            {Object.entries(sum.presets).map(([k, p]) => (
              <button key={k} className={preset === k ? 'on' : ''} onClick={() => applyPolicy({ preset: k })} title={p.blurb}>
                {preset === k && <motion.span layoutId="seg-pill" className="pill" transition={spring} />}
                <span>{p.label.split(' ')[0] === 'Linear' ? 'Static' : p.label.split(' (')[0]}</span>
              </button>
            ))}
          </LayoutGroup>
        </div>
        <div className="hctl" title="Target mesh element size">
          <span className="hlabel">h</span>
          <input type="range" min="0.5" max={Math.max(10, Math.ceil(sum.char_len / 12))} step="0.5" value={sum.policy.h}
            onChange={(e) => setSum({ ...sum, policy: { ...sum.policy, h: +e.target.value } })}
            onPointerUp={(e) => applyPolicy({ h: +e.target.value })} onKeyUp={(e) => applyPolicy({ h: +e.target.value })} />
          <span className="hval">{fmt(sum.policy.h, 1)} mm</span>
        </div>
        <button className="kbtn" onClick={() => setPalette(true)} title="Command palette"><kbd>⌘K</kbd></button>
        <Pressable className="run" onClick={runNow} disabled={!!run || rmCount === 0}>
          {I.play}<span>Simplify {rmCount}</span>
        </Pressable>
      </motion.div>

      {/* view switch */}
      {resMesh && (
        <motion.div className="viewswitch glass" initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={spring}>
          <LayoutGroup id="view">
            {[['original', 'Original'], ['compare', 'Compare'], ['result', 'Simplified']].map(([k, l]) => (
              <button key={k} className={mode === k ? 'on' : ''} onClick={() => setMode(k)}>
                {mode === k && <motion.span layoutId="view-pill" className="pill" transition={spring} />}
                <span>{l}</span>
              </button>
            ))}
          </LayoutGroup>
          {res && !showRes && <button className="link" onClick={() => setShowRes(true)}>Results</button>}
        </motion.div>
      )}

      {/* legend, with the supports card stacked above it so the two never overlap */}
      <div className="legend-stack">
      <AnimatePresence>
        {faceMode && <BoundaryCard boundary={boundary} mode={faceMode} setMode={setFaceMode} onChange={(b) => applyPolicy({ boundary: b })} onClose={() => setFaceMode(null)} />}
      </AnimatePresence>
      <div className="legend">
        {Object.entries(sum.feature_counts).map(([k, n]) => (
          <motion.button className={`chip ${isolate === k ? 'on' : ''} ${isolate && isolate !== k ? 'off' : ''}`} key={k} whileTap={{ scale: 0.95 }}
            onClick={() => setIsolate(isolate === k ? null : k)} title={isolate === k ? 'Show all' : 'Isolate this class'}>
            <span className="swatch" style={{ background: (CLASS_META[k] || CLASS_META.protrusion).color }} />{(CLASS_META[k] || {}).label || k} <b className="mono">{n}</b>
          </motion.button>
        ))}
        <motion.button className={`chip ${faceMode ? 'on' : ''}`} whileTap={{ scale: 0.95 }} onClick={() => { setFaceMode(faceMode ? null : 'fixed'); if (!faceMode) setMode('original') }}>
          <span className="swatch" style={{ background: '#0A84FF' }} />Supports &amp; loads{(boundary.fixed?.length || 0) > 0 && <b className="mono">{boundary.fixed.length + (boundary.loaded?.length || 0)}</b>}
        </motion.button>
        <span className="chip muted">{isolate ? 'Esc to show all' : 'Bright = removed · faded = kept · click to isolate'}</span>
      </div>
      </div>

      {/* side panel */}
      <motion.div className="side glass" initial={{ x: 30, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ ...spring, delay: 0.05 }}>
        <div className="side-tabs">
          <LayoutGroup id="tabs">
            {[['features', `Features`], ['chat', 'Assistant']].map(([k, l]) => (
              <button key={k} className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>
                {tab === k && <motion.span layoutId="tab-pill" className="pill" transition={spring} />}
                <span>{l}{k === 'features' ? ` · ${sum.features.length}` : ''}</span>
              </button>
            ))}
          </LayoutGroup>
        </div>
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={tab} className="side-body" initial={{ opacity: 0, x: tab === 'chat' ? 16 : -16 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: tab === 'chat' ? 16 : -16 }} transition={spring}>
            {tab === 'features' ? (
              <FeaturePanel sum={sum} selected={selected} setSelected={setSelected} hovered={hovered} setHovered={setHovered}
                onToggle={(id, a) => applyPolicy({ feature_overrides: { [id]: a } })}
                onClass={(kind, a) => {
                  const fo = Object.fromEntries(sum.features.filter((f) => f.kind === kind).map((f) => [f.id, 'auto']))
                  const ov = { ...sum.policy.overrides }
                  if (a === 'auto') delete ov[kind]; else ov[kind] = { remove: a === 'remove' ? 'always' : 'never' }
                  applyPolicy({ overrides: ov, merge: false, feature_overrides: fo })
                }} />
            ) : (
              <ChatPanel sid={sid} messages={messages} setMessages={setMessages} onResult={onChatResult} llm={llm}
                queued={queued} clearQueued={() => setQueued(null)} />
            )}
          </motion.div>
        </AnimatePresence>
      </motion.div>

      <CommandPalette open={palette} onClose={() => setPalette(false)} commands={commands}
        onAsk={(t) => { setTab('chat'); setQueued(t) }} />
      <AnimatePresence>{run && <Loader title="Simplifying" steps={run.steps} progress={run.progress} />}</AnimatePresence>
      <AnimatePresence>{showRes && res && <Results sid={sid} res={res} onClose={() => setShowRes(false)} />}</AnimatePresence>
    </motion.div>
  )
}

// ------------------------------------------------------------------ app
export default function App() {
  const dark = useDark()
  const [session, setSession] = useState(null)
  const [busy, setBusy] = useState(null)
  const [llm, setLlm] = useState(false)
  const [err, setErr] = useState(null)
  useEffect(() => { api.health().then((h) => setLlm(h.llm)).catch(() => {}) }, [])
  useEffect(() => { document.documentElement.dataset.theme = dark ? 'dark' : 'light' }, [dark])

  const open = async (fn, title) => {
    setErr(null)
    const steps = ['Reading geometry', 'Building face-adjacency graph', 'Recognizing features', 'Deciding what matters']
    let i = 1
    setBusy({ title, steps: steps.slice(0, 1) })
    const t = setInterval(() => { i = Math.min(steps.length, i + 1); setBusy({ title, steps: steps.slice(0, i) }) }, 650)
    try { setSession(await fn()) } catch (e) { setErr(e.message) } finally { clearInterval(t); setBusy(null) }
  }
  const create = async (prompt) => {
    setErr(null)
    setBusy({ title: 'Designing your part', steps: ['Writing CadQuery code with the local model'] })
    try {
      const r = await api.chat(null, prompt.match(/\b(make|create|design|build|model)\b/i) ? prompt : `Create a part: ${prompt}`)
      const job = r.actions?.find((a) => a.type === 'create_job')?.job
      if (!job) { setBusy(null); setErr(r.reply); return }
      for (;;) {
        await new Promise((res) => setTimeout(res, 800))
        const j = await api.job(job)
        setBusy({ title: 'Designing your part', steps: ['Writing CadQuery code with the local model', ...j.steps] })
        if (j.done) {
          if (j.session) setSession(await api.session(j.session))
          else setErr(`Couldn't build a valid solid: ${j.error}`)
          break
        }
      }
    } catch (e) { setErr(e.message) } finally { setBusy(null) }
  }

  return (
    <div className="app">
      <AnimatePresence mode="wait">
        {session ? (
          <Workspace key={session.id} initial={session} dark={dark} llm={llm} onExit={(s) => setSession(s)} />
        ) : (
          <Welcome key="welcome" busy={busy} llm={llm} dark={dark}
            onOpen={(n) => open(() => api.openSample(n), 'Opening sample')}
            onUpload={(f) => open(() => api.upload(f), `Reading ${f.name}`)}
            onCreate={create} />
        )}
      </AnimatePresence>
      <AnimatePresence>
        {err && (
          <motion.div className="toast glass" initial={{ y: 30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 30, opacity: 0 }} transition={spring} onClick={() => setErr(null)}>
            {err}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
