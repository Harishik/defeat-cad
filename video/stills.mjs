// Render a handful of still frames for review:  node stills.mjs Intro 60 400 900 ...
import { bundle } from '@remotion/bundler'
import { renderStill, selectComposition } from '@remotion/renderer'
import path from 'node:path'
import fs from 'node:fs'

const [id, ...frames] = process.argv.slice(2)
const out = path.resolve('out/stills')
fs.mkdirSync(out, { recursive: true })
const serveUrl = await bundle({ entryPoint: path.resolve('src/index.ts') })
const composition = await selectComposition({ serveUrl, id })
for (const f of frames.map(Number)) {
  const file = path.join(out, `${id}_${String(f).padStart(5, '0')}.png`)
  await renderStill({ composition, serveUrl, output: file, frame: f, scale: 0.5 })
  console.log(file)
}
