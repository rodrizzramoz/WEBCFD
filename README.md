# Flowbench

3D flow simulation (CFD) that runs entirely in the browser. Load an STL, set the flow, press Run. Nothing is uploaded: the solver runs on your own GPU through WebGPU.

## What it does

- **Models:** binary or ASCII STL, by file picker or drag and drop. Unit choice (mm, cm, m, inch) and 90 degree rotations. Four built-in examples.
- **Flow around a part (external):** the part sits in a virtual wind tunnel with flow along +X. Reports drag, lift and side force in newtons, plus Cd and Cl on the frontal area.
- **Flow through a part (internal):** pick an inlet face and an outlet face of the bounding box. Reports pressure drop, flow rate, peak speed and mass balance. The STL can be the solid part (flow goes through its holes) or the fluid volume itself.
- **Real units:** air, water, hydraulic oil or a custom fluid; speeds in m/s; results in N, Pa and L/min.
- **Live 3D results while it solves:** slice plane (speed or pressure), streamlines, surface pressure, optional time averaging, and a history chart.
- **Export:** screenshot (PNG), result history (CSV), setup (JSON, reloadable).

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
| `src/domain.js` | Voxelises the STL (signed ray crossings along X) and builds the cell grid, including finding the inlet and outlet openings and flood-filling the fluid region for internal flow. |
| `src/solver.js` | Lattice Boltzmann solver as a WebGPU compute shader: D3Q19 lattice, regularised BGK collision, Smagorinsky sub-grid turbulence model, halfway bounce-back walls, wall forces by momentum exchange. |
| `src/results.js` | Slice images, streamlines, surface colours and internal flow metrics from the solver output. |
| `src/viewer.js` | three.js scene. |
| `src/main.js` | UI, unit conversion, run loop, readouts, export. |
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
