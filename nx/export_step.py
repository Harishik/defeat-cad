"""NX journal: export every open or listed part to STEP so DeFeat can read it.

NX's .prt format is proprietary and has no open-source reader, so this runs inside your own NX.

In NX:  File > Execute > NX Open, pick this file. It exports the work part.
Batch:  run_journal export_step.py -args part1.prt part2.prt   (writes part1.step, part2.step)

Then open the .step files in DeFeat.
"""
import os
import sys

import NXOpen


def export(session, part, out):
    creator = session.DexManager.CreateStepCreator()
    try:
        creator.ExportAs = NXOpen.StepCreator.ExportAsOption.Ap214
        creator.ExportFrom = NXOpen.StepCreator.ExportFromOption.DisplayPart
        creator.InputFile = part.FullPath
        creator.OutputFile = out
        creator.ObjectTypes.Solids = True
        creator.FileSaveFlag = False
        creator.Commit()
    finally:
        creator.Destroy()


def main():
    session = NXOpen.Session.GetSession()
    paths = sys.argv[1:]
    if not paths:
        part = session.Parts.Work
        if part is None:
            raise RuntimeError("Open a part first, or pass .prt files with -args")
        export(session, part, os.path.splitext(part.FullPath)[0] + ".step")
        return
    for path in paths:
        part, status = session.Parts.OpenBaseDisplay(path)
        status.Dispose()
        export(session, part, os.path.splitext(path)[0] + ".step")
        part.Close(NXOpen.BasePart.CloseWholeTree.TrueValue, NXOpen.BasePart.CloseModified.CloseModified, None)


if __name__ == "__main__":
    main()
