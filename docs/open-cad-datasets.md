# Open CAD datasets: what you may use, share and build on

A practical list of CAD part collections for feature recognition, defeaturing and simulation-prep work, sorted by **what their licenses allow**. The point of this list is the license column: many well-known CAD datasets are for research only, which matters as soon as a model or a parts library leaves the lab.

> Not legal advice. Licenses change and individual parts inside a collection can carry their own terms. Read the license at the source before you redistribute anything or use it commercially. Checked November 2026.

## Free to use and share

These allow redistribution (with the conditions shown) and are not limited to research.

| Dataset | What's in it | Format | Size | License | Conditions | Real or synthetic |
|---|---|---|---|---|---|---|
| [NIST MBE PMI test models](https://www.nist.gov/el/systems-integration-division-73400/mbe-pmi-validation-and-conformance-testing/download-free-cad) | Engineering test parts from the US National Institute of Standards and Technology, modelled in several CAD systems with full product manufacturing information (tolerances, datums) | STEP (AP203/AP242) and native NX, Creo, SolidWorks | Dozens of parts | Usable **without restriction** | NIST asks for acknowledgement and does not allow its logo in promotion | Real engineering parts (designed as test cases) |
| [FreeCAD parts library](https://github.com/FreeCAD/FreeCAD-library) | Community-made mechanical parts: fasteners, profiles, mechanical and electrical components, mechanisms | STEP and FreeCAD (.FCStd) | Hundreds of parts, about 1 GB | **CC-BY 3.0** | Attribute each part's author (traceable through the repository history). Spot-check individual parts: an [open issue](https://github.com/FreeCAD/FreeCAD-library/issues/295) discusses moving to CC-BY 4.0 | Real |
| [KiCad 3D models](https://gitlab.com/kicad/libraries/kicad-packages3D) | Electronic components, including many **injection-molded** connector housings, sockets and enclosure parts | STEP and WRL | Thousands of parts, several GB | **CC-BY-SA 4.0** with the [KiCad libraries exception](https://spdx.org/licenses/KiCad-libraries-exception.html) | Using a model in a design doesn't make the design CC-BY-SA, but **redistributing the models as a collection must keep CC-BY-SA** (share-alike) | Real |
| [MFCAD](https://github.com/hducg/MFCAD) | Parts with machining features (slots, steps, pockets, passages) and a label on every face | STEP | 15,488 parts | **MIT** | Keep the copyright notice | Synthetic |
| [CADGenBench inputs](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-data) | Real mechanical fixtures with mating interfaces: locating jigs, bolt patterns, slots | STEP | 81 parts | **ODC-BY** | Attribution | Real |

## Promising, check before relying on them

| Dataset | What's in it | Why to check |
|---|---|---|
| [InfoBay 3D CAD catalogue](https://huggingface.co/datasets/InfoBayAI/3d-cad-models-dataset) | About 1.2 million CAD assets in several formats including STEP, with injection molds among the samples | States CC BY 4.0, but access is gated and the original sources of the models aren't listed. Confirm provenance before treating the license as settled |
| [MoldCAD / AIMold](https://arxiv.org/abs/2608.00800) | Single-body parts paired with professionally designed injection-mold assemblies, filtered for moldability | The most relevant collection for mold work, but the license and data source aren't stated in the abstract. Check the paper's data-availability section |
| [HybridCAD++](https://data.mendeley.com/datasets/m966nfzrhc/1) | 161,441 parts with labeled faces for hybrid additive-subtractive features | License not confirmed (Mendeley Data often uses CC BY 4.0). Synthetic |
| [Purdue mechanical parts database](https://assemblymag.com/articles/96049-can-purdue-database-help-researchers-apply-machine-learning-to-3d-printed-parts-in-machines) | More than 58,000 annotated mechanical parts | License and file formats not confirmed; standard parts rather than molded ones |

## Research use only, or unclear: don't redistribute

| Dataset | What's in it | License |
|---|---|---|
| [Fusion 360 Gallery](https://github.com/AutodeskAILab/Fusion360GalleryDataset) | Tens of thousands of real designer parts with face labels and modelling histories | **Non-commercial research only.** You may not redistribute it in full, and any portion or modification you share must stay limited to non-commercial research. Models trained on it should be treated with the same caution for commercial use |
| [ABC dataset](https://deep-geometry.github.io/abc-dataset) | About one million models from public Onshape documents | Unclear: the models are owned by their creators under Onshape's terms of use; some archive records say MIT, which conflicts with that |
| GrabCAD, TraceParts and similar portals | Large libraries of real parts | Their terms forbid redistributing the models |

## How to pick

- **Showing parts publicly** (a demo, a parts library on a website): use the first table. NIST and the FreeCAD library are the easiest, with attribution for FreeCAD. KiCad works if the collection you publish keeps CC-BY-SA.
- **Testing a tool honestly on parts it has never seen:** NIST, FreeCAD, KiCad and CADGenBench are all independent of the big research datasets. Robustness measures (does the run finish, is the output a valid solid, how long does it take) need no labels.
- **Training a model that will be used commercially:** prefer the first table. Few permissive datasets have per-face feature labels; MFCAD does, but it is synthetic. Expect to label real parts yourself or to train on a labeller you can check.
- **Mold parts specifically:** KiCad's connector housings are the largest freely shareable set of real injection-molded parts found here. MoldCAD may become the best source once its license is clear.
