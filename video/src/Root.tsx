import React from 'react'
import { Composition } from 'remotion'
import { Intro, INTRO_FRAMES } from './Intro'
import { Guide, GUIDE_FRAMES } from './Guide'

export const Root: React.FC = () => (
  <>
    <Composition id="Intro" component={Intro} durationInFrames={INTRO_FRAMES} fps={30} width={1920} height={1080} />
    <Composition id="Guide" component={Guide} durationInFrames={GUIDE_FRAMES} fps={30} width={1920} height={1080} />
  </>
)
