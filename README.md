# Flowbench

3D flow simulation (CFD) that runs entirely in the browser. Load an STL, set the flow, press Run. Nothing is uploaded: the solver runs on your own GPU through WebGPU.

## What it does

- **Models:** STEP, IGES and BREP CAD files (read with OpenCascade compiled to WebAssembly, loaded only when needed), plus STL, OBJ and 3MF meshes, by file picker or drag and drop. CAD files have a surface detail setting, and on the structure page their real CAD faces are what you click.
- **Mesh preview:** both pages can show the exact cell model the solver will use before you run anything, with cell edges, a cell count, and a warning when the part is too coarse.
- **Grid detail:** presets plus a custom value, up to whatever your GPU memory allows on the flow page.
- **Interactive tools on the flow page:** hover probe on the slice, Shift + click smoke wand, live speed and fluid changes, click to pick inlet and outlet faces, yaw angle, replay timeline.
- **Units and orientation:** unit choice (mm, cm, m, inch) and 90 degree rotations. Four built-in examples.
- **Flow around a part (external):** the part sits in a virtual wind tunnel with flow along +X. Reports drag, lift and side force in newtons, plus Cd and Cl on the frontal area.
- **Flow through a part (internal):** pick an inlet face and an outlet face of the bounding box. Reports pressure drop, flow rate, peak speed and mass balance. The STL can be the solid part (flow goes through its holes) or the fluid volume itself.
- **Real units:** air, water, hydraulic oil or a custom fluid; speeds in m/s; results in N, Pa and L/min.
- **Live 3D results while it solves:** slice plane (speed or pressure), streamlines, surface pressure, optional time averaging, and a history chart.
- **Export:** screenshot (PNG), result history (CSV), setup (JSON, reloadable).

## Structure page (fea.html)

A second page does stress analysis and shape optimisation on the same STL files.

- **Supports and loads:** click faces on the model to mark them fixed or loaded (flat faces, or smooth curved surfaces such as bores). Forces are entered in newtons.
- **Stress analysis:** linear elastic finite elements on a voxel grid (8-node brick per cell), solved matrix-free with a preconditioned conjugate gradient method in a Web Worker. Shows deflection, von Mises stress, safety factor and mass.
- **Shape optimisation:** topology optimisation (SIMP material model, sensitivity filter, optimality criteria update). It removes material in 10 percent steps, re-analyses each step, and plots the trade-off between mass and stiffness. The lightest design that still meets your safety factor is highlighted, and any design can be exported as a smoothed STL.

Checks: a 100 x 20 x 20 mm cantilever with an end load matches beam theory to within 2 to 5 percent for the geometry the grid actually represents (`src/fea/core.js` tested in Node). In the app the part is rounded to whole cells, so at 48 cells along the beam the section comes out 20.8 mm instead of 20 mm and the result is about 15 percent too stiff. Finer grids shrink that error.

Limits: small deflections, one isotropic material, no buckling, fatigue, contact or bolt preload. It runs on the CPU, so grids above roughly 50,000 elements are slow.

See `docs/ACCURACY.md` for a researched review of how accurate the flow solver is and what would improve it.

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
| `src/main.js` | UI, unit conversion, run loop, readouts, export. |
| `src/fea/core.js` | Finite element solver and topology optimiser (no browser dependencies). |
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

Checks run during development (solver-only page `dev/bench.html`, headless Chrome):

| Case | Grid | Result | Reference |
| --- | --- | --- | --- |
| Laminar oil flow in a straight pipe, 16 cells across | 80 x 18 x 18 | pressure drop about 12 percent above theory, mass balance 100.0 percent | Hagen-Poiseuille plus entry loss |
| Cube, face on, 12 cells across | 80 x 50 x 50 | Cd about 1.15 to 1.2 | about 1.05 |
| Sphere, 22 cells across, tight domain | 96 x 63 x 63 | Cd about 0.93 | about 0.47 |
| Venturi, 11 cells across the throat | 128 x 34 x 34 | pressure drop several times too high | a real venturi recovers most of its pressure |

So sharp-edged shapes and simple ducts come out reasonably well, while smooth curved bodies and narrow throats need far finer grids than these before the numbers mean much.

## Licence

No licence chosen yet.
