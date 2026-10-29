import React, { useEffect, useMemo, useState } from 'react'
import { ThreeCanvas } from '@remotion/three'
import { continueRender, delayRender, staticFile } from 'remotion'
import * as THREE from 'three'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { useThree } from '@react-three/fiber'

type Mesh = { p: number[]; i: number[]; k?: number[]; kinds?: string[] }
export type PartData = { original: Mesh; simplified: Mesh; faces: [number, number | null]; removed: number }

const cache: Record<string, PartData> = {}

/**
 * Load every part once, at the root of the composition, before the first frame is released.
 * Parts then read synchronously from the cache, so a scene can never wait on (or miss) its data.
 */
export function usePreloadParts(names: string[]): boolean {
  const [ready, setReady] = useState(() => names.every((n) => cache[n]))
  const [h] = useState(() => (ready ? null : delayRender('load parts')))
  useEffect(() => {
    if (ready) return
    Promise.all(names.map((n) => cache[n] ? null : fetch(staticFile(`part_${n}.json`)).then((r) => r.json()).then((j) => { cache[n] = j })))
      .then(() => setReady(true))
  }, [])   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (ready && h) continueRender(h) }, [ready, h])
  return ready
}
export const usePart = (name: string): PartData | null => cache[name] ?? null

/** Z-up CAD coordinates -> Y-up three.js */
const yUp = (p: number[]) => {
  const out = new Float32Array(p.length)
  for (let i = 0; i < p.length; i += 3) { out[i] = p[i]; out[i + 1] = p[i + 2]; out[i + 2] = -p[i + 1] }
  return out
}

function build(d: PartData) {
  const P = yUp(d.original.p), I = d.original.i, K = d.original.k ?? []
  const body: number[] = [], feat: number[] = [], dirs: number[] = [], seeds: number[] = [], cents: number[] = []
  let st = 7
  const rnd = () => ((st = (st * 16807) % 2147483647) / 2147483647)
  const v = (j: number) => [P[3 * j], P[3 * j + 1], P[3 * j + 2]]
  for (let t = 0; t < I.length / 3; t++) {
    const [a, b, c] = [I[3 * t], I[3 * t + 1], I[3 * t + 2]]
    if (!K[t]) { body.push(a, b, c); continue }
    const A = v(a), B = v(b), Cc = v(c)
    const ctr = [(A[0] + B[0] + Cc[0]) / 3, (A[1] + B[1] + Cc[1]) / 3, (A[2] + B[2] + Cc[2]) / 3]
    const n = new THREE.Vector3().crossVectors(new THREE.Vector3(B[0] - A[0], B[1] - A[1], B[2] - A[2]), new THREE.Vector3(Cc[0] - A[0], Cc[1] - A[1], Cc[2] - A[2])).normalize()
    const s = rnd()
    for (const X of [A, B, Cc]) { feat.push(...X); dirs.push(n.x, n.y, n.z); seeds.push(s); cents.push(...ctr) }
  }
  const bodyG = new THREE.BufferGeometry()
  bodyG.setAttribute('position', new THREE.BufferAttribute(P, 3)); bodyG.setIndex(body); bodyG.computeVertexNormals()
  const featG = new THREE.BufferGeometry()
  const base = new Float32Array(feat)
  featG.setAttribute('position', new THREE.BufferAttribute(base.slice(), 3)); featG.computeVertexNormals()
  const simpG = new THREE.BufferGeometry()
  simpG.setAttribute('position', new THREE.BufferAttribute(yUp(d.simplified.p), 3)); simpG.setIndex(d.simplified.i); simpG.computeVertexNormals()
  // dense "finite-element" wireframe: every triangle subdivided twice
  const wire: number[] = [], wireF: number[] = []
  const sub = (A: number[], B: number[], Cc: number[], lvl: number, out: number[]) => {
    if (lvl === 0) { out.push(...A, ...B, ...B, ...Cc, ...Cc, ...A); return }
    const m = (x: number[], y: number[]) => [(x[0] + y[0]) / 2, (x[1] + y[1]) / 2, (x[2] + y[2]) / 2]
    const ab = m(A, B), bc = m(B, Cc), ca = m(Cc, A)
    sub(A, ab, ca, lvl - 1, out); sub(ab, B, bc, lvl - 1, out); sub(ca, bc, Cc, lvl - 1, out); sub(ab, bc, ca, lvl - 1, out)
  }
  for (let t = 0; t < I.length / 3; t++) sub(v(I[3 * t]), v(I[3 * t + 1]), v(I[3 * t + 2]), 2, K[t] ? wireF : wire)
  const wireG = new THREE.BufferGeometry(); wireG.setAttribute('position', new THREE.Float32BufferAttribute(wire, 3))
  const wireFG = new THREE.BufferGeometry(); wireFG.setAttribute('position', new THREE.Float32BufferAttribute(wireF, 3))
  return { bodyG, featG, simpG, wireG, wireFG, base, dirs: new Float32Array(dirs), seeds: new Float32Array(seeds), cents: new Float32Array(cents) }
}

const Env: React.FC = () => {
  const { gl, scene } = useThree()
  // set synchronously so the very first frame is already lit by the studio environment
  useMemo(() => {
    const pm = new THREE.PMREMGenerator(gl)
    scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture
    pm.dispose()
  }, [gl, scene])
  return null
}

export type Cam = { az?: number; el?: number; dist?: number; target?: [number, number, number]; fov?: number; roll?: number }
export type Item = {
  name: string; pos?: [number, number, number]; spin?: number
  highlight?: number; peel?: number; simplified?: number; wire?: number; jitter?: number; t?: number; tint?: string
}

// geometry is built once per part and reused for the whole film (a part appears at most once per frame)
const geo: Record<string, ReturnType<typeof build>> = {}
const geom = (name: string, d: PartData) => (geo[name] ??= build(d))

const Model: React.FC<{ g: ReturnType<typeof build> } & Item> = ({ g, pos = [0, 0, 0], highlight = 0, peel = 0, simplified = 0, wire = 0, jitter = 0, t = 0, spin = 0, tint = '#8e95a1' }) => {
  // the feature triangles lift off along their normals, drift upward, and shrink to nothing
  const pa = g.featG.getAttribute('position') as THREE.BufferAttribute
  const arr = pa.array as Float32Array
  for (let i = 0; i < arr.length / 3; i++) {
    const s = g.seeds[i]
    const k = Math.min(1, Math.max(0, (peel - s * 0.45) / 0.55))
    const e = k * k * (3 - 2 * k)
    const sc = 1 - e
    const cx = g.cents[3 * i], cy = g.cents[3 * i + 1], cz = g.cents[3 * i + 2]
    const j = jitter * 0.012 * Math.sin(t * 1.7 + s * 50)
    arr[3 * i] = cx + (g.base[3 * i] - cx) * sc + g.dirs[3 * i] * (e * 0.4 + j)
    arr[3 * i + 1] = cy + (g.base[3 * i + 1] - cy) * sc + g.dirs[3 * i + 1] * (e * 0.4 + j) + e * e * 0.25
    arr[3 * i + 2] = cz + (g.base[3 * i + 2] - cz) * sc + g.dirs[3 * i + 2] * (e * 0.4 + j)
  }
  pa.needsUpdate = true
  const metal = { metalness: 0.8, roughness: 0.32, clearcoat: 0.5, clearcoatRoughness: 0.2, envMapIntensity: 0.55 }
  return (
    <group position={pos} rotation={[0, spin, 0]}>
      {simplified < 1 && (
        <>
          <mesh geometry={g.bodyG}>
            <meshPhysicalMaterial color={tint} {...metal} transparent={simplified > 0} opacity={1 - simplified} />
          </mesh>
          <mesh geometry={g.featG}>
            <meshPhysicalMaterial color={highlight > 0.01 ? '#ffb48a' : tint} emissive="#fb5607" emissiveIntensity={highlight * 1.4}
              metalness={0.6} roughness={0.35} envMapIntensity={0.55} side={THREE.DoubleSide} transparent={simplified > 0} opacity={1 - simplified} />
          </mesh>
        </>
      )}
      {simplified > 0 && (
        <mesh geometry={g.simpG}>
          <meshPhysicalMaterial color={tint} {...metal} transparent={simplified < 1} opacity={simplified} />
        </mesh>
      )}
      {wire > 0 && (
        <>
          <lineSegments geometry={g.wireG}><lineBasicMaterial color="#8f9cff" transparent opacity={0.28 * wire} /></lineSegments>
          <lineSegments geometry={g.wireFG}><lineBasicMaterial color="#fb5607" transparent opacity={0.7 * wire} /></lineSegments>
        </>
      )}
    </group>
  )
}

const Rig: React.FC<Cam> = ({ az = 0.6, el = 0.35, dist = 3.2, target = [0, 0, 0], fov = 30, roll = 0 }) => {
  const { camera } = useThree()
  const cam = camera as THREE.PerspectiveCamera
  cam.fov = fov
  cam.position.set(target[0] + dist * Math.cos(el) * Math.sin(az), target[1] + dist * Math.sin(el), target[2] + dist * Math.cos(el) * Math.cos(az))
  cam.up.set(Math.sin(roll), Math.cos(roll), 0)
  cam.lookAt(target[0], target[1], target[2])
  cam.updateProjectionMatrix()
  return null
}

/**
 * The film's single 3D stage: one WebGL canvas that stays mounted for the whole composition (browsers cap live
 * WebGL contexts, so scenes never create their own). Each frame just says which parts are on stage and where the camera is.
 */
export const Stage3D: React.FC<{ items: Item[]; cam: Cam; opacity?: number }> = ({ items, cam, opacity = 1 }) => (
  <div style={{ position: 'absolute', inset: 0, opacity }}>
    <ThreeCanvas width={1920} height={1080} gl={{ antialias: true, alpha: true, toneMapping: THREE.ACESFilmicToneMapping }}
      camera={{ fov: 30, near: 0.05, far: 100 }}>
      <Env />
      <ambientLight intensity={0.15} />
      <directionalLight position={[3, 5, 4]} intensity={1.6} color="#ffffff" />
      <directionalLight position={[-4, 2, -3]} intensity={1.4} color="#fb7a35" />
      <directionalLight position={[-3, -2, 4]} intensity={0.5} color="#6a74ff" />
      <Rig {...cam} />
      {items.map((it) => {
        const d = usePart(it.name)
        return d ? <Model key={it.name} g={geom(it.name, d)} {...it} /> : null
      })}
    </ThreeCanvas>
  </div>
)
