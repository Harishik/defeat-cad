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
- **Headless FreeCAD 1.1.4 (`freecadcmd`) on the bracket sample:** 27 features recognized, 23 removed, 84 → 31 faces, a valid solid, mesh elements −55%.
- **Not yet tested by hand:** the interactive preset dialog in the full FreeCAD GUI.
