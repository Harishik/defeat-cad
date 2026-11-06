# DeFeat for FreeCAD

A FreeCAD macro that simplifies the selected part for simulation using DeFeat. The original object is hidden but never modified; the result is added to the document as `<name>_defeatured`.

## Use
1. Start DeFeat: `./run.sh` (macOS/Linux) or `run.bat` (Windows). It serves on `http://127.0.0.1:8000`.
2. In FreeCAD, open **Macro → Macros…**, note the user macro folder, and copy `DeFeat.FCMacro` into it.
3. Select a solid, run **DeFeat**, and pick the analysis (static, modal, thermal or CFD).
4. The Report view lists each step and a summary: features removed, faces, volume, mesh elements, deflection change, and whether the stress guard kept anything.

Set `DEFEAT_URL` in the environment if DeFeat runs on another port or machine.

## Without FreeCAD
The macro's core is plain Python, so it also works as a small command-line client:
```bash
python freecad/DeFeat.FCMacro part.step part_defeatured.step modal
```

## Tested
- **Headless FreeCAD 1.1.4 (`freecadcmd`) on the bracket sample, this release:** 25 features recognized, 16 removed, 84 → 36 faces, one valid solid, the original left at 84 faces, volume +0.08%, mesh elements −26%, deflection −0.89%.
- **Full FreeCAD 1.1.4 GUI on the bracket sample, previous release (modal preset):** the preset dialog lists static, modal, thermal and CFD; the result is added as `<name>_defeatured`; the original is hidden and left unchanged; the summary dialog reports the result. The dialogs were answered by a startup script, not clicked by hand, and this hasn't been repeated on this release.
