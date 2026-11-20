import * as THREE from 'three'

// Camera controls where the part simply follows the hand:
//   drag            the part turns the way you drag, in any direction, with no stops at the poles
//   shift / right   the part moves with the pointer (also middle-drag, or two fingers on a touch screen)
//   scroll / pinch  zoom, keeping the point under the cursor where it is
// Rotation is in screen space (about the camera's own up and right axes), so the same drag always does
// the same thing on screen, whichever way the part faces. A flick keeps turning briefly and settles.
export class SimpleControls {
  constructor(camera, dom) {
    this.camera = camera
    this.dom = dom
    this.target = new THREE.Vector3()
    this.minDistance = 0
    this.maxDistance = Infinity
    this.rotateSpeed = 1.0          // half a turn per drag across the view's height
    this.listeners = {}
    this.mode = null                // 'rotate' | 'pan' while a pointer is down
    this.last = null
    this.vel = new THREE.Vector2()  // rotation velocity in px per frame, for the flick
    this.touches = new Map()
    this.pinch = null

    dom.style.touchAction = 'none'
    this._down = this._down.bind(this)
    this._move = this._move.bind(this)
    this._up = this._up.bind(this)
    this._wheel = this._wheel.bind(this)
    this._menu = (e) => e.preventDefault()
    dom.addEventListener('pointerdown', this._down)
    dom.addEventListener('pointermove', this._move)
    dom.addEventListener('pointerup', this._up)
    dom.addEventListener('pointercancel', this._up)
    dom.addEventListener('wheel', this._wheel, { passive: false })
    dom.addEventListener('contextmenu', this._menu)
  }

  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn) }
  _emit(type) { for (const fn of this.listeners[type] || []) fn() }

  _axes() {
    const q = this.camera.quaternion
    return { right: new THREE.Vector3(1, 0, 0).applyQuaternion(q), up: new THREE.Vector3(0, 1, 0).applyQuaternion(q) }
  }

  rotate(dx, dy) {
    const h = this.dom.clientHeight || 1
    const { right, up } = this._axes()
    // grab-and-turn: dragging right turns the part right, dragging down tips its top toward you
    const q = new THREE.Quaternion().setFromAxisAngle(up, (-dx / h) * Math.PI * this.rotateSpeed)
    q.multiply(new THREE.Quaternion().setFromAxisAngle(right, (-dy / h) * Math.PI * this.rotateSpeed))
    const off = this.camera.position.clone().sub(this.target).applyQuaternion(q)
    this.camera.up.applyQuaternion(q).normalize()
    this.camera.position.copy(this.target).add(off)
    this.camera.lookAt(this.target)
  }

  pan(dx, dy) {
    const h = this.dom.clientHeight || 1
    const dist = this.camera.position.distanceTo(this.target)
    const perPx = (2 * dist * Math.tan((this.camera.fov * Math.PI) / 360)) / h
    const { right, up } = this._axes()
    const move = right.multiplyScalar(-dx * perPx).add(up.multiplyScalar(dy * perPx))
    this.target.add(move)
    this.camera.position.add(move)
  }

  // zoom by `scale` (< 1 is closer), keeping the point under (clientX, clientY) fixed on screen
  dolly(scale, clientX, clientY) {
    const cam = this.camera
    const dist = cam.position.distanceTo(this.target)
    const next = THREE.MathUtils.clamp(dist * scale, this.minDistance, this.maxDistance)
    if (Math.abs(next - dist) < 1e-9) return
    const f = next / dist
    let p = this.target.clone()
    if (clientX != null) {
      const r = this.dom.getBoundingClientRect()
      const ndc = new THREE.Vector3(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1, 0.5)
      const dir = ndc.unproject(cam).sub(cam.position).normalize()
      const fwd = this.target.clone().sub(cam.position).normalize()
      const t = dist / Math.max(dir.dot(fwd), 1e-3)   // where the cursor's ray meets the plane through the target
      p = cam.position.clone().addScaledVector(dir, t)
    }
    cam.position.sub(p).multiplyScalar(f).add(p)
    this.target.sub(p).multiplyScalar(f).add(p)
  }

  _down(e) {
    this._emit('start')
    this.vel.set(0, 0)
    this.dom.setPointerCapture?.(e.pointerId)
    if (e.pointerType === 'touch') {
      this.touches.set(e.pointerId, [e.clientX, e.clientY])
      this.pinch = null
      this.mode = this.touches.size === 1 ? 'rotate' : 'touch2'
    } else {
      this.mode = e.button === 2 || e.button === 1 || e.shiftKey || e.metaKey || e.ctrlKey ? 'pan' : 'rotate'
    }
    this.last = [e.clientX, e.clientY]
  }

  _move(e) {
    if (!this.mode) return
    if (e.pointerType === 'touch') {
      if (!this.touches.has(e.pointerId)) return
      this.touches.set(e.pointerId, [e.clientX, e.clientY])
      if (this.touches.size >= 2) {
        const [a, b] = [...this.touches.values()]
        const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
        const span = Math.hypot(a[0] - b[0], a[1] - b[1])
        if (this.pinch) {
          this.pan(mid[0] - this.pinch.mid[0], mid[1] - this.pinch.mid[1])
          if (span > 0 && this.pinch.span > 0) this.dolly(this.pinch.span / span, mid[0], mid[1])
        }
        this.pinch = { mid, span }
        return
      }
    }
    const dx = e.clientX - this.last[0], dy = e.clientY - this.last[1]
    this.last = [e.clientX, e.clientY]
    if (this.mode === 'rotate') {
      this.rotate(dx, dy)
      this.vel.set(dx, dy)
      this.lastMove = performance.now()
    } else if (this.mode === 'pan') {
      this.pan(dx, dy)
    }
  }

  _up(e) {
    if (e.pointerType === 'touch') {
      this.touches.delete(e.pointerId)
      this.pinch = null
      if (this.touches.size) { this.mode = this.touches.size === 1 ? 'rotate' : 'touch2'; const [p] = this.touches.values(); this.last = p; return }
    }
    // only a release while still moving keeps turning; a pause before letting go stops dead
    if (this.mode !== 'rotate' || performance.now() - (this.lastMove || 0) > 60) this.vel.set(0, 0)
    this.mode = null
  }

  _wheel(e) {
    e.preventDefault()
    this._emit('start')
    this.vel.set(0, 0)
    const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * 400 : e.deltaY
    // a trackpad pinch arrives as a wheel with ctrlKey and small steps: give it more weight
    // one mouse-wheel notch zooms about 10%; each event is capped so a fast spin can't fly into the part
    const f = Math.exp(dy * (e.ctrlKey ? 0.008 : 0.001))
    this.dolly(THREE.MathUtils.clamp(f, 0.8, 1.25), e.clientX, e.clientY)
  }

  // called every frame: lets a flick settle, then keeps the camera looking at the target
  update() {
    if (!this.mode && this.vel.lengthSq() > 0.01) {
      this.rotate(this.vel.x, this.vel.y)
      this.vel.multiplyScalar(0.88)
    }
    this.camera.lookAt(this.target)
  }

  dispose() {
    const d = this.dom
    d.removeEventListener('pointerdown', this._down)
    d.removeEventListener('pointermove', this._move)
    d.removeEventListener('pointerup', this._up)
    d.removeEventListener('pointercancel', this._up)
    d.removeEventListener('wheel', this._wheel)
    d.removeEventListener('contextmenu', this._menu)
  }
}
