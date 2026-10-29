import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { decodeMesh } from './api'

const VERT = /* glsl */ `
  attribute float aRemovable;
  attribute float aSeed;
  attribute vec3 aDrift;
  uniform float uTime;
  uniform float uDissolve;   // 0 = features shown, 1 = features gone
  uniform float uSize;
  uniform float uPixelRatio;
  varying float vRemovable;
  varying float vAlpha;
  void main() {
    float d = smoothstep(aSeed * 0.6, aSeed * 0.6 + 0.4, uDissolve) * aRemovable;
    vec3 p = position + aDrift * d * 1.0;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float pulse = 0.85 + 0.15 * sin(uTime * 2.2 + aSeed * 12.0);
    gl_PointSize = uSize * uPixelRatio * (1.0 + 0.7 * aRemovable * pulse) * (8.0 / -mv.z);
    vRemovable = aRemovable;
    vAlpha = (1.0 - d) * clamp(1.25 - (-mv.z - 6.0) * 0.09, 0.35, 1.0);
  }
`

const FRAG = /* glsl */ `
  uniform vec3 uBase;
  uniform vec3 uAccent;
  varying float vRemovable;
  varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = length(c);
    if (r > 0.5) discard;
    float edge = smoothstep(0.5, 0.36, r);
    vec3 col = mix(uBase, uAccent, vRemovable);
    gl_FragColor = vec4(col, edge * vAlpha * mix(0.82, 1.0, vRemovable));
  }
`

function edgeSegments(positions, indices) {
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  g.setIndex(new THREE.BufferAttribute(indices, 1))
  const e = new THREE.EdgesGeometry(g, 28)
  return e.attributes.position.array
}

function samplePoints(mesh, removable, count, edgeShare = 0.32) {
  const { positions, indices, faceIds } = decodeMesh(mesh)
  const segs = edgeSegments(positions, indices)
  const nEdge = Math.floor(count * edgeShare)
  const nSurf = count - nEdge
  const rem = new Set(removable)
  const nT = indices.length / 3
  const areas = new Float32Array(nT)
  const raw = new Float32Array(nT)
  let total = 0, remA = 0, baseA = 0
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3()
  for (let t = 0; t < nT; t++) {
    a.fromArray(positions, indices[3 * t] * 3)
    b.fromArray(positions, indices[3 * t + 1] * 3)
    c.fromArray(positions, indices[3 * t + 2] * 3)
    raw[t] = b.clone().sub(a).cross(c.clone().sub(a)).length() / 2
    if (rem.has(faceIds[t])) remA += raw[t]; else baseA += raw[t]
  }
  // tiny features (holes, text, thin chamfers) would get almost no dots by area alone:
  // give removable faces at least ~22% of the surface dots so they always read
  const share = 0.22
  const boost = remA > 0 ? Math.max(5, (share * baseA) / ((1 - share) * remA)) : 1
  for (let t = 0; t < nT; t++) {
    total += rem.has(faceIds[t]) ? raw[t] * boost : raw[t]
    areas[t] = total
  }
  const pos = new Float32Array(count * 3)
  const flag = new Float32Array(count)
  const seed = new Float32Array(count)
  const drift = new Float32Array(count * 3)
  // edge dots: evenly spaced along sharp edges (the "technical drawing" look)
  const nS = segs.length / 6
  const cum = new Float32Array(nS)
  let L = 0
  for (let k = 0; k < nS; k++) {
    L += Math.hypot(segs[6 * k + 3] - segs[6 * k], segs[6 * k + 4] - segs[6 * k + 1], segs[6 * k + 5] - segs[6 * k + 2])
    cum[k] = L
  }
  for (let i = 0; i < nEdge && nS; i++) {
    const r = (i + 0.5) / nEdge * L
    let lo = 0, hi = nS - 1
    while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < r) lo = m + 1; else hi = m }
    const seg0 = lo ? cum[lo - 1] : 0
    const f = (r - seg0) / Math.max(cum[lo] - seg0, 1e-9)
    for (let d = 0; d < 3; d++) pos[3 * i + d] = segs[6 * lo + d] + f * (segs[6 * lo + 3 + d] - segs[6 * lo + d])
    flag[i] = 0
    seed[i] = Math.random()
  }
  for (let i = nEdge; i < count; i++) {
    const r = Math.random() * total
    let lo = 0, hi = nT - 1
    while (lo < hi) { const m = (lo + hi) >> 1; if (areas[m] < r) lo = m + 1; else hi = m }
    a.fromArray(positions, indices[3 * lo] * 3)
    b.fromArray(positions, indices[3 * lo + 1] * 3)
    c.fromArray(positions, indices[3 * lo + 2] * 3)
    let u = Math.random(), v = Math.random()
    if (u + v > 1) { u = 1 - u; v = 1 - v }
    const p = a.clone().addScaledVector(b.clone().sub(a), u).addScaledVector(c.clone().sub(a), v)
    pos.set([p.x, p.y, p.z], 3 * i)
    flag[i] = rem.has(faceIds[lo]) ? 1 : 0
    seed[i] = Math.random()
  }
  return { pos, flag, seed, drift }
}

export default function DotPart({ data, count = 9000, size = 2.2, accent = '#d4ff3f', base = '#e8e8ea',
  dissolve = true, interactive = true, className = '', distance = 1.0 }) {
  const host = useRef(null)
  useEffect(() => {
    if (!data) return
    const el = host.current
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100)
    camera.position.set(0, 0, 10 * distance)

    const { pos, flag, seed, drift } = samplePoints(data.mesh, data.removable, count)
    // normalise into a ~4.2 unit box, Z-up CAD -> Y-up view
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.computeBoundingBox()
    const bb = g.boundingBox, ctr = bb.getCenter(new THREE.Vector3()), sz = bb.getSize(new THREE.Vector3())
    const k = 4.2 / Math.max(sz.x, sz.y, sz.z)
    for (let i = 0; i < count; i++) {
      pos[3 * i] = (pos[3 * i] - ctr.x) * k
      pos[3 * i + 1] = (pos[3 * i + 1] - ctr.y) * k
      pos[3 * i + 2] = (pos[3 * i + 2] - ctr.z) * k
      // removable dots drift outward from the centre when they dissolve
      const dx = pos[3 * i], dy = pos[3 * i + 1], dz = pos[3 * i + 2]
      const L = Math.hypot(dx, dy, dz) || 1
      drift[3 * i] = (dx / L) * 0.9 + (Math.random() - 0.5) * 0.6
      drift[3 * i + 1] = (dy / L) * 0.9 + (Math.random() - 0.5) * 0.6
      drift[3 * i + 2] = (dz / L) * 0.9 + 0.8 + Math.random() * 0.6
    }
    g.attributes.position.needsUpdate = true
    g.setAttribute('aRemovable', new THREE.BufferAttribute(flag, 1))
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1))
    g.setAttribute('aDrift', new THREE.BufferAttribute(drift, 3))
    const mat = new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false,
      uniforms: {
        uTime: { value: 0 }, uDissolve: { value: 0 }, uSize: { value: size },
        uPixelRatio: { value: renderer.getPixelRatio() },
        uBase: { value: new THREE.Color(base) }, uAccent: { value: new THREE.Color(accent) },
      },
    })
    const pts = new THREE.Points(g, mat)
    const group = new THREE.Group()
    group.add(pts)
    group.rotation.x = -Math.PI / 2   // Z-up
    const pivot = new THREE.Group()
    pivot.add(group)
    const flat = 1 - sz.z / Math.max(sz.x, sz.y, sz.z)
    const tilt = 0.38 + 0.42 * flat
    pivot.rotation.set(tilt, -0.65, 0)
    scene.add(pivot)

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(el)

    // pointer parallax (spring-damped)
    const target = { x: 0, y: 0 }, cur = { x: 0, y: 0, vx: 0, vy: 0 }
    const onMove = (e) => {
      const r = el.getBoundingClientRect()
      target.x = ((e.clientX - r.left) / r.width - 0.5) * 0.6
      target.y = ((e.clientY - r.top) / r.height - 0.5) * 0.35
    }
    if (interactive) window.addEventListener('pointermove', onMove)

    let raf, t0 = performance.now(), last = t0
    const tick = (now) => {
      raf = requestAnimationFrame(tick)
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const t = (now - t0) / 1000
      mat.uniforms.uTime.value = t
      if (!reduce) {
        // critically damped spring toward the pointer target
        const kS = 40, dS = 2 * Math.sqrt(kS)
        cur.vx += (kS * (target.x - cur.x) - dS * cur.vx) * dt
        cur.vy += (kS * (target.y - cur.y) - dS * cur.vy) * dt
        cur.x += cur.vx * dt
        cur.y += cur.vy * dt
        pivot.rotation.y = -0.65 + t * 0.12 + cur.x
        pivot.rotation.x = tilt + cur.y
        if (dissolve) {
          // 7 s loop: show features -> dissolve -> simplified -> re-form
          const c = t % 7
          const e = (x) => x * x * (3 - 2 * x)
          mat.uniforms.uDissolve.value = c < 2.4 ? 0 : c < 3.6 ? e((c - 2.4) / 1.2) : c < 5.6 ? 1 : 1 - e((c - 5.6) / 1.4)
        }
      }
      renderer.render(scene, camera)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      window.removeEventListener('pointermove', onMove)
      g.dispose(); mat.dispose(); renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [data, count, size, accent, base, dissolve, interactive, distance])
  return <div ref={host} className={`dotpart ${className}`} />
}
