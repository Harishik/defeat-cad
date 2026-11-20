import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { SimpleControls } from './navigate'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { decodeMesh, CLASS_META } from './api'

const BASE = new THREE.Color('#dfe3e8')
const BASE_DARK = new THREE.Color('#7e8594')

function buildGeometry(mesh) {
  const { positions, indices, faceIds } = decodeMesh(mesh)
  // un-index so each triangle can be coloured by its B-Rep face
  const g0 = new THREE.BufferGeometry()
  g0.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  g0.setIndex(new THREE.BufferAttribute(indices, 1))
  const g = g0.toNonIndexed()
  g.computeVertexNormals()
  const triFace = faceIds
  const colors = new Float32Array(g.attributes.position.count * 3)
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  g.userData.triFace = triFace
  g.computeBoundingBox()
  g.computeBoundingSphere()
  return g
}

function paint(g, faceColor, dark) {
  const col = g.attributes.color.array
  const tf = g.userData.triFace
  const base = dark ? BASE_DARK : BASE
  for (let t = 0; t < tf.length; t++) {
    const c = faceColor.get(tf[t]) || base
    for (let k = 0; k < 3; k++) {
      const o = (t * 3 + k) * 3
      col[o] = c.r; col[o + 1] = c.g; col[o + 2] = c.b
    }
  }
  g.attributes.color.needsUpdate = true
}

export default function Viewer({ original, result, features, mode, wipe, selected, hovered, onPick, onHover, dark, boundary, faceMode, onFace, wipeApi }) {
  const host = useRef(null)
  const S = useRef({})

  // The wipe plane slides along the camera's right axis, so its depth never changes and its screen x is
  // linear in the wipe value: two projections are enough to turn a pointer x back into a wipe value.
  useEffect(() => {
    if (!wipeApi) return
    wipeApi.current = {
      xToWipe(clientX) {
        const s = S.current
        if (!s.center || !s.camera || !host.current) return null
        const r = host.current.getBoundingClientRect()
        const right = new THREE.Vector3().setFromMatrixColumn(s.camera.matrixWorld, 0).normalize()
        const sx = (w) => r.left + ((s.center.clone().addScaledVector(right, (w - 0.5) * 2 * s.radius).project(s.camera).x + 1) / 2) * r.width
        const x0 = sx(0), x1 = sx(1)
        if (Math.abs(x1 - x0) < 1) return null
        return Math.min(0.98, Math.max(0.02, (clientX - x0) / (x1 - x0)))
      },
    }
  }, [wipeApi])

  // ---------------------------------------------------------------- scene setup
  useEffect(() => {
    const el = host.current
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.localClippingEnabled = true
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.0
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const pmrem = new THREE.PMREMGenerator(renderer)
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture
    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 10000)
    const controls = new SimpleControls(camera, renderer.domElement)
    controls.addEventListener('start', () => S.current.onInteract?.())
    const key = new THREE.DirectionalLight(0xffffff, 1.4)
    key.position.set(2, -3, 4)
    scene.add(key, new THREE.HemisphereLight(0xffffff, 0x9aa3ad, 0.6))
    const group = new THREE.Group()
    scene.add(group)
    const clipA = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)
    const clipB = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0)
    const ray = new THREE.Raycaster()
    S.current = { renderer, scene, camera, controls, group, clipA, clipB, ray, meshes: {}, anim: [] }

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight
      renderer.setSize(w, h)
      camera.aspect = w / h
      // centre the part in the space the feature panel leaves free, not behind the panel
      const side = el.parentElement?.querySelector('.side')
      const r = side?.getBoundingClientRect(), me = el.getBoundingClientRect()
      const s = r && r.left > me.left + me.width / 2 ? Math.max(0, (me.right - r.left) / 2) : 0
      if (s > 0) camera.setViewOffset(w + 2 * s, h, 2 * s, 0, w, h)
      else camera.clearViewOffset()
      camera.updateProjectionMatrix()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    const side = el.parentElement?.querySelector('.side')
    if (side) ro.observe(side)
    const late = setTimeout(resize, 400)   // the panel can mount just after the viewer
    let raf
    const tick = (t) => {
      raf = requestAnimationFrame(tick)
      const s = S.current
      s.anim = s.anim.filter((a) => a(t))
      // camera-aligned wipe plane through the model centre
      if (s.center) {
        const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).normalize()
        const off = s.center.clone().addScaledVector(right, (s.wipe - 0.5) * 2 * s.radius)
        clipA.setFromNormalAndCoplanarPoint(right.clone().negate(), off)
        clipB.setFromNormalAndCoplanarPoint(right, off)
        if (s.wipeLine) {
          const p = off.clone().project(camera)
          s.wipeLine.style.left = `${((p.x + 1) / 2) * 100}%`
        }
      }
      controls.update()
      if (s.radius) {
        // keep the clipping planes around the part wherever the camera is, so zooming in never slices it
        const d = camera.position.distanceTo(controls.target)
        const near = Math.max(s.radius * 1e-4, Math.min(d * 0.02, s.radius * 0.05))
        const far = d + s.radius * 40
        if (Math.abs(near - camera.near) > near * 0.05 || Math.abs(far - camera.far) > far * 0.05) {
          camera.near = near; camera.far = far; camera.updateProjectionMatrix()
        }
      }
      renderer.render(scene, camera)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(late)
      ro.disconnect()
      renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [])

  // ---------------------------------------------------------------- meshes
  const makeMesh = (geom, name) => {
    const mat = new THREE.MeshPhysicalMaterial({
      vertexColors: true, roughness: 0.42, metalness: 0.05, clearcoat: 0.35, clearcoatRoughness: 0.4,
      transparent: true, opacity: 1, side: THREE.DoubleSide,
    })
    const m = new THREE.Mesh(geom, mat)
    m.name = name
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geom, 28),
      new THREE.LineBasicMaterial({ color: 0x3a3a3c, transparent: true, opacity: 0.22 })
    )
    m.add(edges)
    return m
  }

  useEffect(() => {
    const s = S.current
    if (!original) return
    const g = buildGeometry(original)
    if (s.meshes.original) s.group.remove(s.meshes.original)
    const m = makeMesh(g, 'original')
    s.meshes.original = m
    s.group.add(m)
    // frame the part (Z-up CAD convention)
    const sph = g.boundingSphere
    s.center = sph.center.clone()
    s.radius = sph.radius
    s.camera.up.set(0, 0, 1)
    const dir = new THREE.Vector3(1.1, -1.6, 0.95).normalize()
    const dist = sph.radius / Math.sin((s.camera.fov * Math.PI) / 360) * 1.05
    const from = s.camera.position.clone()
    const to = sph.center.clone().addScaledVector(dir, dist)
    s.controls.target.copy(sph.center)
    s.controls.minDistance = sph.radius * 0.04   // close enough for the smallest feature, never inside the part
    s.controls.maxDistance = dist * 6
    s.camera.near = dist / 100
    s.camera.far = dist * 20
    s.camera.updateProjectionMatrix()
    // ease in from slightly further away (critically damped feel)
    const start = performance.now()
    const startPos = from.lengthSq() ? from : to.clone().multiplyScalar(1.6)
    s.anim.push((t) => {
      const k = Math.min(1, (t - start) / 900)
      const e = 1 - Math.pow(1 - k, 4)
      s.camera.position.lerpVectors(startPos, to, e)
      return k < 1
    })
    // fade-in
    m.material.opacity = 0
    s.anim.push((t) => {
      const k = Math.min(1, (t - start) / 600)
      m.material.opacity = k
      return k < 1
    })
  }, [original])

  useEffect(() => {
    const s = S.current
    if (s.meshes.result) {
      s.group.remove(s.meshes.result)
      s.meshes.result = null
    }
    if (!result) return
    const g = buildGeometry(result)
    paint(g, new Map(), dark)
    const m = makeMesh(g, 'result')
    s.meshes.result = m
    s.group.add(m)
  }, [result, dark])

  // ---------------------------------------------------------------- colours
  useEffect(() => {
    const s = S.current
    const m = s.meshes.original
    if (!m || !features) return
    const fc = new Map()
    s.faceToFeature = new Map()
    for (const f of features) {
      const meta = CLASS_META[f.kind] || CLASS_META.protrusion
      const c = new THREE.Color(meta.color)
      const removing = f.decision.action === 'remove'
      const isSel = selected === f.id, isHov = hovered === f.id
      if (!removing) c.lerp(dark ? BASE_DARK : BASE, 0.72)
      if (isHov || isSel) c.offsetHSL(0, 0.05, isSel ? 0.12 : 0.06)
      for (const fid of f.faces) {
        fc.set(fid, c)
        s.faceToFeature.set(fid, f.id)
      }
    }
    for (const fid of boundary?.fixed || []) fc.set(fid, new THREE.Color('#0A84FF'))
    for (const fid of boundary?.loaded || []) fc.set(fid, new THREE.Color('#FF375F'))
    paint(m.geometry, fc, dark)
  }, [features, selected, hovered, original, dark, boundary])

  // ---------------------------------------------------------------- view mode
  useEffect(() => {
    const s = S.current
    s.wipe = wipe
    const o = s.meshes.original, r = s.meshes.result
    if (!o) return
    const setClip = (mesh, planes) => {
      if (!mesh) return
      mesh.material.clippingPlanes = planes
      mesh.children[0].material.clippingPlanes = planes
      mesh.material.needsUpdate = true
    }
    // a newer view mode cancels any cross-dissolve still running, so it can't fade a model out afterwards
    const fade = (s.fade = (s.fade || 0) + 1)
    if (mode === 'compare' && r) {
      // both halves at full opacity: an earlier switch to one model may have faded the other to zero
      for (const m of [o, r]) { m.visible = true; m.material.opacity = 1 }
      setClip(o, [s.clipA]); setClip(r, [s.clipB])
    } else {
      setClip(o, null); setClip(r, null)
      const showResult = mode === 'result' && r
      // cross-dissolve between the two states
      const from = showResult ? o : r, to = showResult ? r : o
      if (to) {
        to.visible = true
        const t0 = performance.now()
        s.anim.push((t) => {
          if (s.fade !== fade) return false
          const k = Math.min(1, (t - t0) / 650)
          const e = 1 - Math.pow(1 - k, 3)
          to.material.opacity = e
          if (from && from !== to) {
            from.material.opacity = 1 - e
            if (k >= 1) from.visible = false
          }
          return k < 1
        })
      }
    }
  }, [mode, wipe, result, original])

  useEffect(() => {
    S.current.wipeLine = host.current?.parentElement?.querySelector('.wipe-line')
  })

  // ---------------------------------------------------------------- navigation
  // Glide the camera to look at `target` from direction `dir` at distance `dist` (critically damped feel).
  const glide = (target, dir, dist, ms = 520) => {
    const s = S.current
    if (!s.camera) return
    const p0 = s.camera.position.clone(), t0 = s.controls.target.clone()
    const p1 = target.clone().addScaledVector(dir.clone().normalize(), dist)
    const start = performance.now()
    s.anim.push((t) => {
      const k = Math.min(1, (t - start) / ms)
      const e = 1 - Math.pow(1 - k, 4)
      s.controls.target.lerpVectors(t0, target, e)
      s.camera.position.lerpVectors(p0, p1, e)
      return k < 1
    })
  }
  const fitDist = (r) => r / Math.sin((S.current.camera.fov * Math.PI) / 360) * 1.08
  // direction to look from, and which way is up on screen (Z-up CAD parts; the top view has +Y up)
  const VIEWS = { iso: [[1.1, -1.6, 0.95], [0, 0, 1]], front: [[0, -1, 0], [0, 0, 1]], top: [[0, 0, 1], [0, 1, 0]], side: [[1, 0, 0], [0, 0, 1]] }
  const view = (name) => {
    const s = S.current
    if (!s.center) return
    if (name === 'fit') { glide(s.center.clone(), s.camera.position.clone().sub(s.controls.target), fitDist(s.radius)); return }
    const [d, u] = VIEWS[name]
    const up0 = s.camera.up.clone(), up1 = new THREE.Vector3(...u)
    const start = performance.now()
    s.anim.push((t) => {   // turn the screen's up direction along with the glide, so the view lands upright
      const k = Math.min(1, (t - start) / 520)
      s.camera.up.copy(up0).lerp(up1, 1 - Math.pow(1 - k, 4)).normalize()
      return k < 1
    })
    glide(s.center.clone(), new THREE.Vector3(...d), fitDist(s.radius))
  }
  // double-click a point: it becomes the centre of rotation, and the view glides to it
  const focus = (e) => {
    const s = S.current
    const r = host.current.getBoundingClientRect()
    const p = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
    s.ray.setFromCamera(p, s.camera)
    const meshes = [s.meshes.original, s.meshes.result].filter((m) => m && m.visible)
    const hit = s.ray.intersectObjects(meshes, false)[0]
    if (!hit) { view('fit'); return }
    const off = s.camera.position.clone().sub(s.controls.target)
    glide(hit.point, off, Math.max(off.length() * 0.6, s.radius * 0.25))
  }
  useEffect(() => {
    const k = (e) => {
      if (e.target.closest?.('input, textarea, [contenteditable]') || e.metaKey || e.ctrlKey || e.altKey) return
      const m = { f: 'fit', 0: 'iso', 1: 'front', 2: 'top', 3: 'side' }[e.key.toLowerCase()]
      if (m) { e.preventDefault(); view(m) }
    }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [])
  const [hint, setHint] = useState(true)
  useEffect(() => { S.current.onInteract = () => setHint(false) }, [])

  // ---------------------------------------------------------------- picking
  const pick = (e, raw = false) => {
    const s = S.current
    const m = s.meshes.original
    if (!m || !m.visible) return null
    const r = host.current.getBoundingClientRect()
    const p = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
    s.ray.setFromCamera(p, s.camera)
    const hit = s.ray.intersectObject(m, false)[0]
    if (!hit) return null
    const fid = m.geometry.userData.triFace[hit.faceIndex]
    return raw ? fid : s.faceToFeature?.get(fid) ?? null
  }
  const down = useRef(null)
  const tools = [['fit', 'Fit to view (F)', 'M4 9V5h4M20 9V5h-4M4 15v4h4M20 15v4h-4'], ['iso', '3D view (0)', 'M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5'],
    ['front', 'Front (1)', 'M5 5h14v14H5z'], ['top', 'Top (2)', 'M4 9l8-5 8 5-8 5zM4 9v6l8 5 8-5V9'], ['side', 'Side (3)', 'M8 4l10 4v12L8 16z']]
  return (
    <>
    <div className="viewtools glass" role="toolbar" aria-label="View">
      {tools.map(([k, label, d]) => (
        <button key={k} onClick={() => { view(k); setHint(false) }} title={label} aria-label={label}>
          <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden><path d={d} fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" strokeLinecap="round" /></svg>
        </button>
      ))}
    </div>
    {hint && (
      <div className="navhint glass" aria-hidden>
        Drag to turn the part · Shift-drag to move it · Scroll or pinch to zoom · Double-click to centre
      </div>
    )}
    <div
      ref={host}
      className="viewer"
      onDoubleClick={(e) => { if (!faceMode) { focus(e); setHint(false) } }}
      onPointerDown={(e) => (down.current = [e.clientX, e.clientY])}
      onPointerUp={(e) => {
        const d = down.current
        if (d && Math.hypot(e.clientX - d[0], e.clientY - d[1]) < 4) {
          if (faceMode) { const f = pick(e, true); if (f != null) onFace?.(f) } else onPick?.(pick(e), { x: e.clientX, y: e.clientY })
        }
      }}
      onPointerMove={(e) => { if (e.buttons === 0 && !faceMode) onHover?.(pick(e)) }}
    />
    </>
  )
}
