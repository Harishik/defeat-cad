import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
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
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.09
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
      camera.updateProjectionMatrix()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(el)
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
      renderer.render(scene, camera)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
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
  return (
    <div
      ref={host}
      className="viewer"
      onPointerDown={(e) => (down.current = [e.clientX, e.clientY])}
      onPointerUp={(e) => {
        const d = down.current
        if (d && Math.hypot(e.clientX - d[0], e.clientY - d[1]) < 4) {
          if (faceMode) { const f = pick(e, true); if (f != null) onFace?.(f) } else onPick?.(pick(e))
        }
      }}
      onPointerMove={(e) => { if (e.buttons === 0 && !faceMode) onHover?.(pick(e)) }}
    />
  )
}
