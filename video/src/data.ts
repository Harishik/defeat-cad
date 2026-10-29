import { useEffect, useState } from 'react'
import { continueRender, delayRender, staticFile } from 'remotion'

export type Box = { x: number; y: number; width: number; height: number }

function useJson<T>(file: string): T | null {
  const [d, setD] = useState<T | null>(null)
  const [h] = useState(() => delayRender(`load ${file}`))
  useEffect(() => {
    fetch(staticFile(file)).then((r) => r.json()).then((j) => { setD(j); continueRender(h) }).catch(() => continueRender(h))
  }, [file, h])
  return d
}

export const useCloud = (name: string) => useJson<{ p: number[]; r: number[]; faces: number; features: number; remove: number }>(`points_${name}.json`)
export const useBoxes = () => useJson<Record<string, Box>>('boxes.json') || {}

// headline numbers (from bench/results, held-out data)
export const NUM = {
  filletF1: 0.84, chamferF1: 0.71, mfcadF1: 0.95,
  parts: 500, completed: 99.8, valid: 100, removed: 91.1,
  tets: 55, defl: -3.0, faces: [84, 31] as [number, number],
}
