# Flowbench roadmap

Where the suite is and what comes next. Written 11 November 2026.

## What exists

| Area | State |
| --- | --- |
| Geometry import | STEP, IGES, BREP (OpenCascade in WebAssembly), STL, OBJ, 3MF. CAD faces are kept for picking on the structure page. |
| Meshing | Uniform cell grid from a signed ray-cast voxeliser. Mesh preview on both pages. Custom grid detail. |
| Flow (CFD) | WebGPU lattice Boltzmann, external and internal flow, SI readouts, slice, streamlines, surface pressure, probe, smoke wand, timeline, exports. |
| Structure (FEA) | Linear elastic brick elements, matrix-free solver, face picking for supports and loads, topology optimisation with a weight against stiffness chart, smoothed STL export. |
| Evidence | docs/ACCURACY.md (literature review of the flow solver), dev/bench.html and dev/fea-check.mjs (repeatable checks). |

## Next, in order

1. **Flow accuracy, cheap fixes.** Report total pressure loss as well as static pressure drop for internal flow. Show blockage. Longer outlet region and automatic inlet and outlet extensions. These come straight from docs/ACCURACY.md section 5.
2. **Flow accuracy, curved walls.** Interpolated bounce-back using the wall distance along each lattice link, computed by the voxeliser from the triangles. This is the largest gain for the effort and removes most of the stair-step error.
3. **Validation page.** One button that runs the benchmark cases (laminar pipe, cube, sphere, Ahmed body, venturi) on the user's GPU at several grid sizes and tabulates the result against the reference values, so accuracy is measured on real hardware instead of the software GPU used in development.
4. **Wall model.** Wall function plus near-wall damping of the turbulence model, for high Reynolds number external flow.
5. **Larger grids.** Split the population storage across several GPU buffers to pass the per-buffer limit, and halve memory with a single-buffer streaming scheme. Local grid refinement around the part after that.
6. **Structure solver.** GPU conjugate gradient for fine grids (in progress), then a multigrid preconditioner. Stress-constrained optimisation, several load cases, bolt and bearing loads, minimum member size, print-direction overhang limits.
7. **Coupling.** Take surface pressure from a flow run as the load on the structure page, so a part can be optimised for the loads the flow actually puts on it. Then true multi-objective search: drag, mass and stiffness on one chart.
8. **Suite features.** Projects that save model, setup and results together; a report export; shared units and materials library; a landing page that links the tools.

## Known limits today

- Flow numbers are trustworthy for sharp-edged shapes and simple ducts at moderate grid sizes, and overpredict drag and pressure drop on smooth curved bodies and narrow throats. See docs/ACCURACY.md.
- The structure solver rounds geometry to whole cells, so thin walls need fine grids.
- Everything was developed against a software-emulated GPU. Speed and the largest grid settings have not been measured on real hardware.
