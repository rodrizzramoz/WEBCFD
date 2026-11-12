# Flowbench

3D flow simulation (CFD) that runs entirely in the browser. Load an STL, set the flow, press Run. Nothing is uploaded: the solver runs on your own GPU through WebGPU.

## What it does

- **Models:** STEP, IGES and BREP CAD files (read with OpenCascade compiled to WebAssembly, loaded only when needed), plus STL, OBJ and 3MF meshes, by file picker or drag and drop. CAD files have a surface detail setting, and on the structure page their real CAD faces are what you click.
- **Mesh preview:** both pages can show the exact cell model the solver will use before you run anything, with cell edges, a cell count, and a warning when the part is too coarse.
- **Grid detail:** presets plus a custom value. The flow solver splits its storage over several GPU buffers, so the ceiling is about 100 million cells on a GPU with a 4 GB buffer limit (roughly 172 bytes per cell in total). Above 6 million cells the 3D view is drawn from a coarser averaged copy while the solve and the readouts stay at full detail.
- **Interactive tools on the flow page:** hover probe on the slice, Shift + click smoke wand, live speed and fluid changes, click to pick inlet and outlet faces, yaw angle, replay timeline.
- **Units and orientation:** unit choice (mm, cm, m, inch) and 90 degree rotations. Four built-in examples.
- **Flow around a part (external):** the part sits in a virtual wind tunnel with flow along +X. Reports drag, lift and side force in newtons, plus Cd and Cl on the frontal area.
- **Flow through a part (internal):** pick an inlet face and an outlet face of the bounding box. Reports pressure drop, flow rate, peak speed and mass balance. The STL can be the solid part (flow goes through its holes) or the fluid volume itself.
- **Custom ports:** for parts with several openings, such as an air mover fed by a compressed-air nozzle. Every opening is found and listed; set each one to air fed in (with a speed), open to atmosphere, or wall. Reports air drawn in and the amplification ratio, with a table of flow and pressure per opening.
- **Wall shape:** walls can follow the true triangle surface (interpolated bounce-back using the exact distance from each cell to the surface) or use plain cube steps, which is faster and fine for a quick look.
- **OpenFOAM:** the current setup can be exported as a ready-to-run OpenFOAM case (snappyHexMesh mesh snapped to the real surface, simpleFoam), or solved by OpenFOAM directly from the page through the small local runner in `server/`.
- **Real units:** air, water, hydraulic oil or a custom fluid; speeds in m/s; results in N, Pa and L/min.
- **Live 3D results while it solves:** slice plane (speed or pressure), streamlines, surface pressure, optional time averaging, and a history chart.
- **Export:** screenshot (PNG), result history (CSV), setup (JSON, reloadable).

## Structure page (fea.html)

A second page does stress analysis and shape optimisation on the same STL files.

- **Supports and loads:** click faces on the model to mark them fixed or loaded (flat faces, or smooth curved surfaces such as bores). Forces are entered in newtons.
- **Mesh type:** tetrahedra that follow the real surface (default), or cubes. The tetrahedral mesher needs no external library: it cuts a regular grid of tetrahedra against the exact surface, places nodes on the surface and on sharp edges, and gives a conforming mesh with the true volume. Cubes are faster and can use the GPU on fine grids, but round the part to whole cells.
- **Stress analysis:** linear elastic finite elements, solved matrix-free with a preconditioned conjugate gradient method in a Web Worker. Tetrahedra use edge-based strain smoothing so that bending is not too stiff. Shows deflection, von Mises stress, safety factor and mass.
- **Shape optimisation:** topology optimisation (SIMP material model, sensitivity filter, optimality criteria update). It removes material in 10 percent steps, re-analyses each step, and plots the trade-off between mass and stiffness. The lightest design that still meets your safety factor is highlighted, and any design can be exported as a smoothed STL.

Checks (`node dev/fea-tet-check.mjs` and `node dev/fea-check.mjs`): on a 100 x 20 x 20 mm cantilever with an end load, where the grid does not line up with the part, the tetrahedral mesh has the right volume to 0.01 percent and a tip deflection 2.0 to 2.6 percent under beam theory at 24 to 72 cells along the beam. The cube mesh on the same grids is 8.5 percent over on volume and 16 to 19 percent too stiff. Sphere and cylinder volumes are within 0.3 percent at 32 cells; a plate with a through hole meshes conformingly with no inverted elements.

Limits: small deflections, one isotropic material, no buckling, fatigue, contact or bolt preload. Tetrahedral meshes run on the CPU: analysis is quick up to several hundred thousand elements, optimisation is practical to about 100,000. Sharp edges that do not fall on the grid are bevelled by a fraction of a cell, and walls thinner than about one cell can be lost. The cube mesh has a GPU solver above 20,000 elements, checked only on a software-emulated GPU.

See `docs/ACCURACY.md` for a researched review of how accurate the flow solver is and what would improve it.

## OpenFOAM

OpenFOAM cannot run inside a browser tab, so there are two ways to use it with the same setup:

1. **Export:** the "OpenFOAM case" button saves a zip. Unzip it and run `sh Allrun`. The README inside explains where the results are.
2. **Local runner:** run `python3 server/flowbench_openfoam_server.py` on the computer that has OpenFOAM. It prints an access code. Paste the code into the flow page and press "Solve with OpenFOAM". The page sends the case, the runner meshes and solves it, and the flow rates, pressures and pressure force come back into a table under the readouts.

The runner listens on 127.0.0.1 only, needs the access code on every request, accepts only a fixed list of OpenFOAM dictionary files, rejects any file that could run code, and runs a fixed list of commands. It uses only the Python standard library.

Both were checked against OpenFOAM v1912 (the Ubuntu package). In that particular package every function object fails at start-up, so the export's built-in reporting block (force coefficients, patch flow rates) could not be exercised there; the runner does not use it and computes the numbers from the written fields instead. On Windows, OpenFOAM needs WSL or Docker.

By default the exported case is steady and incompressible. Tick "Treat air as compressible" for parts fed with compressed air: fed openings set by supply pressure become reservoirs at that pressure, air is a perfect gas, and the case is solved with rhoPimpleFoam in local time stepping mode, with the supply pressure brought up gradually so the solver stays stable. Checked on the air mover demo at 2 bar gauge: the nozzle chokes and passes 19.7 g/s against 20.1 g/s from nozzle theory (25,000 cells), and the page shows mass flow per opening and the amplification by mass.

## Run it

```
npm install
npm run dev      # local dev server
npm run build    # static site in dist/
```

`dist/` is a plain static site with relative paths. Upload it to any static host (GitHub Pages, Cloudflare Pages, Netlify, an nginx folder). The page must be served over HTTPS or from localhost, because browsers only enable WebGPU there.

Needs a browser with WebGPU: current Chrome, Edge, Firefox or Safari on a machine with a real graphics card.

## How it works

| File | Job |
| --- | --- |
| `src/io.js` | Model import for both pages: STL, OBJ, 3MF, and STEP, IGES, BREP through occt-import-js. |
| `src/domain.js` | Voxelises the STL (signed ray crossings along X) and builds the cell grid, including finding the inlet and outlet openings and flood-filling the fluid region for internal flow. |
| `src/solver.js` | Lattice Boltzmann solver as a WebGPU compute shader: D3Q19 lattice, regularised BGK collision, Smagorinsky sub-grid turbulence model, halfway bounce-back walls, wall forces by momentum exchange. |
| `src/results.js` | Slice images, streamlines, surface colours and internal flow metrics from the solver output. |
| `src/viewer.js` | three.js scene. |
| `src/walls.js` | Exact distance from near-wall cells to the triangle surface, for the true-surface wall option. |
| `src/openfoam.js`, `src/foamrun.js` | OpenFOAM case writer and the client for the local runner. |
| `server/flowbench_openfoam_server.py` | Optional local service that runs OpenFOAM for the page. |
| `src/main.js` | UI, unit conversion, run loop, readouts, export. |
| `src/fea/tetmesh.js`, `src/fea/tetcore.js` | Body-fitted tetrahedral mesher, and the tetrahedral solver and optimiser. |
| `src/fea/core.js`, `src/fea/gpu.js` | Cube-mesh solver and optimiser, and its WebGPU solver. |
| `src/fea/worker.js` | Runs the solver off the main thread. |
| `src/fea/mesh.js` | Face picking regions, mapping faces to voxels, result meshes, marching cubes, STL export. |
| `src/fea/main.js`, `src/fea/view.js` | Structure page UI and 3D view. |
| `dev/bench.html` | Solver-only test page used for headless benchmarking. |

## Accuracy: read this before trusting a number

This is a fast, interactive solver, not a replacement for a validated CFD package.

- The grid is uniform and walls are stair-stepped. There are no boundary layer cells and no wall functions, so skin friction and separation points on smooth curved bodies are approximate.
- Real-world Reynolds numbers are far beyond what the grid resolves. The turbulence model keeps the solution stable, but the result is an under-resolved large eddy simulation.
- Coarse grids overpredict drag and pressure drop. Always raise the grid detail until the number you care about stops changing, and compare designs against each other at the same settings.
- The method is weakly compressible. The app keeps the fastest flow well below the lattice speed of sound, which is why internal flows with strong contractions take more steps.
- STLs must be watertight with consistent normals. Walls thinner than one cell disappear.

Checks run during development are tabulated at the end of `docs/ACCURACY.md`, including cross-checks against OpenFOAM. In short: pipe friction is within 2.5 percent, sharp-edged bodies are within about 15 percent, smooth curved bodies overpredict drag (less so with the true-surface wall option), and jet-driven entrainment is badly underpredicted on coarse grids, so use the OpenFOAM route for air movers and ejectors.

## Licence

No licence chosen yet.
