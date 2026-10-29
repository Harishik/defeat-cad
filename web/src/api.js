const j = async (r) => {
  if (!r.ok) {
    let msg = r.statusText
    try { msg = (await r.json()).detail || msg } catch {}
    throw new Error(msg)
  }
  return r.json()
}

export const api = {
  health: () => fetch('/api/health').then(j),
  samples: () => fetch('/api/samples').then(j),
  openSample: (name) => fetch(`/api/sample/${encodeURIComponent(name)}`, { method: 'POST' }).then(j),
  upload: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch('/api/upload', { method: 'POST', body: fd }).then(j)
  },
  session: (id) => fetch(`/api/session/${id}`).then(j),
  mesh: (id, which) => fetch(`/api/session/${id}/mesh/${which}`).then(j),
  policy: (id, body) =>
    fetch(`/api/session/${id}/policy`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(j),
  run: (id) => fetch(`/api/session/${id}/run`, { method: 'POST' }).then(j),
  status: (id) => fetch(`/api/session/${id}/status`).then(j),
  result: (id) => fetch(`/api/session/${id}/result`).then(j),
  verify: (id) => fetch(`/api/session/${id}/verify`, { method: 'POST' }).then(j),
  chat: (session, message) =>
    fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session, message }) }).then(j),
  job: (id) => fetch(`/api/job/${id}`).then(j),
  download: (id, kind) => `/api/session/${id}/download/${kind}`,
}

export function decodeMesh(m) {
  const b = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer
  return {
    positions: new Float32Array(b(m.positions)),
    indices: new Uint32Array(b(m.indices)),
    faceIds: new Uint32Array(b(m.faceIds)),
  }
}

export const CLASS_META = {
  hole: { label: 'Holes', color: '#0A84FF' },
  slot: { label: 'Slots', color: '#5E5CE6' },
  pocket: { label: 'Pockets & cut-outs', color: '#BF5AF2' },
  step: { label: 'Steps & shoulders', color: '#AC8E68' },
  boss: { label: 'Bosses', color: '#FF9F0A' },
  rib: { label: 'Ribs & gussets', color: '#30D158' },
  emboss: { label: 'Text & logos', color: '#FF375F' },
  fillet: { label: 'Fillets & rounds', color: '#40C8E0' },
  chamfer: { label: 'Chamfers', color: '#FFD60A' },
  protrusion: { label: 'Other protrusions', color: '#98989D' },
}
