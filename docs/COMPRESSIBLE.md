# Compressible gas solver (src/compressible.js)

This is the solver for air that moves fast enough to be squeezed: nozzles fed at several bar,
air movers, ejectors, anything where the jet reaches the speed of sound. The existing flow
solver (src/solver.js, lattice Boltzmann) treats the fluid as incompressible and is only valid
below about Mach 0.3. This one solves the full compressible equations, so it can represent
choked nozzles, supersonic jets and shock waves.

It is a self-contained module. The flow page uses it through src/gasrun.js when Openings is set to Custom and Air model to "Compressible gas".

All figures in this document were measured with dev/compressible-check.html, run in headless
Chrome on a software GPU (SwiftShader, 2 CPU cores). Nothing here has been run on a real
graphics card yet.

## What it solves

Conservation of mass, momentum and energy for an ideal gas (the compressible Navier-Stokes
equations), with:

- laminar viscosity and heat conduction (Prandtl number 0.72),
- a Smagorinsky eddy viscosity (constant 0.16, turbulent Prandtl number 0.9), meant to stand
  in for the eddies the grid cannot resolve so that jets and shear layers mix. Test 6 shows
  that on a coarse grid it does not mix a jet nearly enough.

Each cell stores density, momentum (three components) and total energy. These are the
"conserved" variables: whatever leaves one cell through a face enters the neighbour through
the same face, so mass and energy cannot leak. A sixth number per cell holds the eddy viscosity.

## Method

- Grid: the uniform cube grid from buildDomain, one value set per cell (cell-centred finite volume).
- Fluxes between cells: the HLLC approximate Riemann solver, with the simple Davis bounds on the
  wave speeds. A Riemann solver treats each face as a tiny shock tube problem between the
  states on its two sides, which is what lets the method capture shocks without wiggles.
- Second order in space: the states on the two sides of a face are extrapolated from the cell
  centres using limited slopes of density, velocity and pressure (MUSCL reconstruction with the
  monotonised central limiter). The limiter switches the slope off at peaks and next to jumps.
- Time stepping: two-stage strong-stability-preserving Runge-Kutta (Heun's method), explicit.
- Time step: one global step for the whole grid, set by the fastest signal anywhere:
  `dt = 2 x cfl x dx / max(|u| + |v| + |w| + 3c + viscous term)`, with cfl = 0.4 by default and c
  the local speed of sound. The maximum is found on the GPU and only a few kilobytes are
  read back, every step for the first 16 steps and every 8 steps after that (every step again
  for a while if the signal speed jumps by more than 25 percent).
- Viscous terms: the stress and heat flow through a face come from the difference between the
  two cells on either side of it (the Laplacian form). The cross-derivative parts of the full
  stress tensor are left out. That is exact for constant viscosity at low speed and a small
  error otherwise; it is far smaller than the errors from the coarse grid.
- Positivity: density and pressure must stay above zero. The limited reconstruction cannot
  produce a negative value at a face. If a cell still loses more than 70 percent of its density
  or pressure in one stage it is marked as troubled, and every face of a troubled cell uses
  first-order fluxes in the next stage (both neighbours see the mark, so this stays
  conservative). As a last resort a value that would go below 0.001 of ambient is clamped.
  Both events are counted and returned by read() as `firstOrderFallbacks` and `clamps`.
  A non-zero clamp count means mass or energy was not conserved exactly at that cell.
  No test so far has triggered the clamp, so that last branch is untested.
- Numbers: 32-bit floats. Inside the shader everything is non-dimensional: pressure in units of
  ambient pressure, density in units of ambient density, lengths in cells. Ambient air at rest
  is then density 1, pressure 1, speed of sound 1.18. Working directly in pascals would waste
  most of the 7 digits a 32-bit float has on the 101325.

## Boundary conditions

Cell types are the ones buildDomain writes into grid.flags.

- Solid wall (type 1): no flow through the face. The wall pressure comes from the Riemann
  problem between the cell and its mirror image. Friction is the laminar no-slip stress over
  the half cell between the cell centre and the wall. There is no heat flow into walls.
  Walls are the staircase of cube faces; the wall-distance table from the "exact walls"
  option is ignored.
- Feed port (ports[n].type = "feed"): a reservoir at a given total (stagnation) pressure and
  total temperature. The inflow speed follows from the reservoir conditions and the one
  pressure wave that travels from the inside of the domain out to the boundary, so the port
  delivers whatever flow the geometry downstream allows. The inflow is normal to the face and
  at most sonic. If the pressure inside pushes back harder than the reservoir, the port acts as a wall.
- Open port (ports[n].type = "open"): surroundings at a given static pressure and temperature.
  Air leaving slower than sound feels the outside pressure. Air leaving faster than sound is
  simply let out. Air being drawn in comes from still surroundings (total pressure = the given
  pressure), which is what an entrainment inlet needs.
  The outside pressure is applied softly to air that is leaving: each stage the face pressure
  moves a small fraction of the way to it (gas.openRelax = 0.25, divided by the longest side of
  the grid in cells). In steady flow the pressure at the opening still ends up exactly at the
  given value, but pressure waves pass out instead of bouncing back. With a rigidly fixed
  pressure a duct that is open at both ends rings like an organ pipe for a very long time
  (see test 6). Set openRelax to Infinity for the rigid version.
- Far field (type 2, external flow): a fixed pressure, temperature and velocity
  (gas.farfield). The flux through the face is the Riemann solution between the cell inside
  and the far field state, which lets waves out with little reflection.

Port cells hold the state "just outside" the port. A small extra GPU pass refreshes them from
the fluid cell inside before every stage.

## Interface

```js
import { CompressibleSolver } from "./compressible.js";

const s = await CompressibleSolver.create();      // device and pipelines; s.maxCells, s.lost as in Solver
await s.setup(grid, gas, ports);                  // grid from buildDomain
await s.step(k);                                  // k time steps; updates s.steps and s.time (seconds)
const r = await s.read();
s.dispose();
```

gas: `{gamma: 1.4, R: 287.05, mu, ambientPressure, ambientTemperature, cfl: 0.4}` plus the optional
`farfield: {pressure, temperature, velocity: [x, y, z]}`, `smagorinsky` (0.16; 0 switches it off),
`prandtl`, `turbulentPrandtl`, `openRelax` (0.25, see open ports above), `rampSteps` (raise the feed pressure smoothly over this many steps;
0 = full pressure at once) and `dtInterval` (8).

ports, indexed by the port number stored in the flags:
`{type: "feed", totalPressure, totalTemperature}` or `{type: "open", pressure, temperature}`,
pressures absolute in Pa, temperatures in K. A port with no entry opens to ambient.

read() returns:

- `fields`: Float32Array, 8 floats per cell in SI units: density, velocity x, y, z, pressure
  (absolute), temperature, Mach number, 0.
- `ports[n]`: `massFlow` in kg/s (positive into the domain), `momentumFlow` in N (mass flow x
  speed plus pressure above ambient x area, summed over the port), `meanPressure` in Pa
  (absolute, in the fluid cells just inside the port), `area` in m2.
- `force`: [x, y, z] in N on the solid walls, from pressure above ambient. Friction is not included.
- `maxMach`, `time`, `steps`, `dt`, `mass` (kg), `energy` (J), `minDensity`, `minPressure`,
  `firstOrderFallbacks`, `clamps`, `bad` (cells with invalid numbers; the run has failed if not zero).
- `stride` and `dims`: on grids above 6 million cells `fields` is a block-averaged copy
  (stride cells per side per block, dims = size of the copy), like Solver does. Port flows,
  forces and totals are always summed over the real cells on the GPU, so they do not lose accuracy.

Extras: `totals()` gives everything except `fields` without copying the flow back (cheap, use
it for convergence monitoring). `setState(fields)` overwrites the flow from an array in the
`fields` layout. `updatePorts(ports)` changes port settings during a run. `s.diverged` is set
and step() stops if the flow has blown up.

## Test results

Run on 12 November 2026. "Cells" means all cells of the grid including solid ones.

### 1. Sod shock tube

A tube with high pressure on the left and low on the right, released at time zero; the exact
answer is known. Duct of 2 x 2 fluid cells with solid walls, compared at time 0.2 with the
exact solution. L1 error = mean absolute error along the tube, in units of the left state.

| Cells along tube | Density | Velocity | Pressure |
|---|---|---|---|
| 100 | 0.00500 | 0.01069 | 0.00361 |
| 200 | 0.00264 | 0.00537 | 0.00180 |
| 400 | 0.00143 | 0.00255 | 0.00087 |

The errors halve each time the cell count doubles, which is the expected behaviour for a
problem with jumps in it. The tube was run along x, y and z: all three give the same errors
to four figures, the four cells of the cross-section agree to 6 parts in a million and
nothing moves sideways (below 3 parts in a million of the reference speed). No fallbacks or clamps.

Harder tubes, to push the safeguards (none diverged, no clamps):

- Two streams pulling apart at Mach 2.7 each (Toro's "123" problem; density falls to 2 % of
  the start): density error 0.0034 at 200 cells and 0.0018 at 400. The velocity error stays
  near 0.015 at both, a known weakness of this class of method where the gas is very thin.
- The same pulled apart twice as fast, which leaves a vacuum: survives, density falls to
  0.6 % of the start, and the first-order fallback was used 24 times. This is the only test
  that triggered the fallback. There is no error figure because the exact solver on the check
  page does not handle a vacuum.
- Pressure ratio 100 across the diaphragm: errors 0.004 (density), 0.017 (velocity), fine.
- Pressure ratio 100,000 (a shock at about Mach 200, far beyond anything compressed air does):
  stable, but the four cells of the cross-section drift apart (7 % at 200 cells, 85 % at 400).
  This is the "carbuncle" instability that HLLC is known for at very strong shocks lined up
  with the grid. See Limits.

### 2. Conservation

Closed box of 24 x 24 x 24 cells with a solid block inside, started with a lumpy density, a
pressure peak 2.5 times ambient and a swirl (Mach 0.7), viscosity and eddy viscosity on.
Over 300 steps total mass changed by at most 3 parts in a billion and total energy by at most
8 parts in a billion (summed on the CPU in double precision). That is round-off level for
32-bit floats. No fallbacks or clamps.

### 3. Converging nozzle

Round nozzle, 7 mm inlet narrowing to a 4 mm throat (8 cells across the throat, 52 cells of
throat area), fed from a reservoir at 20 C and blowing into open surroundings. Grid 32 x 20 x 20.
Mass flow compared with the ideal isentropic nozzle of src/airjet.js for the same throat area
(the area of the 52 cells).

| Gauge pressure | Mass flow | Ideal | Ratio | Mean Mach at throat | Steps to settle |
|---|---|---|---|---|---|
| 0.5 bar | 3.80 g/s | 4.43 g/s | 85.7 % | 0.67 (jet 0.78, ideal 0.78) | 2000 |
| 2 bar | 8.24 g/s | 9.25 g/s | 89.1 % | 1.01 | 1400 |
| 5 bar | 16.44 g/s | 18.45 g/s | 89.1 % | 1.01 | 1600 |

- The 2 and 5 bar cases are choked: Mach 1 at the throat, and the mass flow scales with the
  absolute reservoir pressure (16.44 / 8.24 = 1.995, pressure ratio 1.996).
- The staircase nozzle shows a discharge coefficient of 0.89 choked and 0.86 unchoked. The
  loss is real for this shape: each step of the staircase is a small sharp-edged contraction
  and the flow necks in behind it.
- To separate the solver from the staircase, the same case was run with a straight 4 mm pipe
  (no steps) feeding the jet at 2 bar. That gives 99.8 % of the ideal mass flow (99.85 % with
  viscosity switched off) and Mach 1.01 at the exit. So the boundary conditions and the
  choking are right, and the missing 11 % in the table is the staircase.
- Mass flow out through the open boundary matches the feed to within 0.3 % once settled.

### 4. Uniform flow

Box of 48 x 14 x 14 cells with far field cells all round and an outlet on X max, as
buildDomain makes for external flow.

- Undisturbed, at Mach 0.3 and Mach 2: after 200 steps the pressure and velocity are bit for
  bit what they were at the start (the deviation printed, 8e-8 and 1e-7, is just the
  conversion to SI units and is identical at step 0).
- With a 10 percent pressure pulse placed in the stream: at Mach 0.3 the largest pressure
  deviation falls from 9.4 % to 0.19 % after 100 steps, 0.007 % after 200 and 0.0008 % after
  600. At Mach 2 it falls to 0.03 % after 200 steps and to round-off (8e-7) by 300. The pulse
  leaves through the boundaries and nothing grows.

### 5. Mach 2 oblique shock

Two versions of the same flow, both inviscid, both stable for 3 flow-through times with no
fallbacks or clamps.

(a) Wall along the grid. A Mach 2 stream arrives tilted 15 degrees towards a flat plate, so the
wall has no staircase. Grid 120 x 52 x 5.

| | Theory | Measured |
|---|---|---|
| Shock angle to the plate | 30.34 degrees | 30.34 degrees |
| Pressure ratio across the shock | 2.195 | 2.195 (2.169 to 2.223 over 884 cells) |
| Mach number behind the shock | 1.446 | 1.444 |

The shock is captured over about 2 cells.

(b) Staircase wedge. A 15 degree wedge built from cubes in a Mach 2 stream along x. Grid 140 x 62 x 5.

| | Theory | Measured |
|---|---|---|
| Shock angle | 45.3 degrees | 52.6 degrees (near the tip) |
| Pressure ratio | 2.195 | about 2.33 |
| Mach number behind the shock | 1.446 | about 1.36 |

It runs stably (Mach number range 0.04 to 2.0) but the answer is wrong by a clear margin. The
shock stands 3.4 cells ahead of the wedge tip and is steeper and stronger than it should be,
because every step of the staircase is a small forward-facing step with its own bow shock.
Refining the grid is not expected to cure this (the steps get smaller but there are more of
them); that was not tested.
**For supersonic flow along a wall that is at an angle to the grid, shock angles and surface
pressures are not reliable until the walls are represented properly (cut cells or an
immersed boundary).** Walls that lie along the grid are fine, as (a) shows.

### 6. Air mover demo ("jetpump")

The demo geometry from src/demos.js (30 mm bore tube, 140 mm long, with a 6 mm bore nozzle
tube on its axis), built with buildDomain exactly as the page does (port mode "ports"), feed
port at 2 bar gauge, both other ports open to atmosphere. Grid detail 64: 64 x 18 x 18 cells of
2.19 mm, so the nozzle bore is under 3 cells across (4 cells of area).

It runs stably: 8000 steps, 10.3 ms of simulated time, no fallbacks, no clamps, no invalid
cells. Flows are averages over each 400-step window.

| Port | Mass flow | Note |
|---|---|---|
| 0 feed (nozzle) | +13.60 g/s | 99.9 % of the ideal choked flow for the 4-cell port area (19.1 mm2); the real 6 mm bore (28.3 mm2) would pass 20.1 g/s |
| 1 open (suction, X min) | +5.64 g/s | still rising slowly at the end |
| 2 open (outlet, X max) | -19.26 g/s | |

- Air drawn in / air fed = **0.41** at 10.3 ms.
- Settling: the feed flow is steady to 0.1 % after 2000 steps (2.6 ms). The suction flow is
  not: 0.31 at 4.6 ms, 0.37 at 7.2 ms, 0.41 at 10.3 ms, still gaining 1 to 3 % per 0.5 ms.
  Inflow minus outflow balances to 0.5 % of the feed by the end.
- Jet thrust at the feed port 5.4 N; Mach number on the axis 1.29 just after the nozzle,
  falling only to 0.84 at the outlet.

**How believable is the 0.41? Not at all, as a prediction of the real device.** The reasons,
in order of importance:

1. The jet does not mix. It leaves the tube still at Mach 0.84 on the axis, and the outlet
   stream still carries 4.8 N of momentum flow against the 5.4 N the jet started with, in a
   narrow fast core instead of shared out over the surrounding air. By the usual rules of
   thumb for a free turbulent jet, 17 nozzle diameters downstream it has slowed to well under
   half its speed and carries several times its starting mass flow. A rough hand momentum
   balance for this tube with a fully mixed outlet gives a ratio of about 5.
2. The amount of mixing is set almost entirely by the eddy viscosity constant, not by
   physics the grid resolves. The same case with the Smagorinsky constant raised from 0.16
   to 0.5 gives a ratio of 2.9 at 8.3 ms and still rising (axis Mach number 0.40 at the
   outlet). A result that moves by a factor of 7 with one tuning constant is not a prediction.
3. Under 3 cells across the nozzle cannot represent the jet's shock cells or its shear
   layer, and the port area is 32 % smaller than the real bore.
4. With a rigidly fixed outlet pressure (openRelax = Infinity) the tube rings: single
   readings of the ratio swung between 0.3 and 1.6 for at least 5 ms. The soft opening
   removes the ringing but it shows how sensitive this flow is to the boundary treatment.

What can be trusted from this run: it is stable, mass is conserved, the choked feed flow is
right for the area the grid gives the nozzle, and the suction side does draw air in the
right direction. What is needed before the entrainment figure means anything: at least 8 to
10 cells across the nozzle (grid detail of about 190 to 230 for this demo, half a million to
a million cells, so that the jet's own eddies can be resolved; whether they then are has
not been tested), or a proper turbulence model for the jet. Either way it then has to be
checked against a measured ejector. This is the main piece of work
still to do for the owner's kind of problem.

Resolution check (not run to the end). The same case at grid detail 96 (96 x 27 x 27 cells
of 1.46 mm, about 4 cells across the nozzle) was run for 6240 steps, 5.3 ms, and then stopped
because it is slow on the software GPU. It was stable with no fallbacks or clamps. Feed flow
19.64 g/s (the real bore would pass 20.1). Air drawn in / air fed was only 0.08 at 5.3 ms,
against about 0.32 at the same time on the coarser grid. So a finer grid gives four times
less suction, not a similar figure: the mixing in these runs comes from the grid and the
eddy viscosity (both of which shrink with the cell size), which confirms that the
entrainment figure is not converged and not predictive at these resolutions.

### 7. Speed and memory

- Speed here: 237,000 cell updates per second on a 48 x 48 x 48 box at best (one update = one
  full time step of one cell = two flux stages), and about 120,000 while another job was
  using one of the two cores. The existing lattice Boltzmann
  solver manages about a million here, so a step costs roughly four times as much. The
  software GPU works through every cell at the same cost, solid or not; a real GPU skips solid cells early.
  Speed on a real graphics card has not been measured.
- Memory: 76 bytes per cell (two copies of the state at 24 bytes, 4 bytes of flags, 24 bytes
  for the read-back copy), against 188 for the existing solver. Above 6 million cells the
  read-back copy is block-averaged and the cost falls to 53 bytes per cell. On top of that
  there is a diagnostics buffer of 176 bytes per (y, z) column of the grid, which is small.
- Largest grid: `maxCells` = largest buffer / 24 bytes (44.7 million cells on the software GPU here).
- Reading the full field back took 53 ms for 110,592 cells.

## Limits

- **Staircase walls.** See test 5 (b) and test 3. Angled or curved walls lose about 10 % of
  nozzle mass flow at 8 cells across the throat and give wrong shock angles in supersonic
  flow. Walls along the grid axes are accurate.
- **No wall model.** Wall friction is the laminar stress over half a cell, which is far below
  real turbulent friction when the cell is much thicker than the viscous sublayer (it nearly
  always is). Pressure losses in long pipes and boundary layer growth are therefore
  under-predicted, and flow separation points are not to be trusted. The eddy viscosity is
  deliberately not applied at walls: an early version did, and a short straight pipe lost
  22 % of its mass flow to friction that was not real.
- **The eddy viscosity is the simplest kind.** Fixed constant, no damping, no account of
  compressibility. Jet spreading and entrainment depend on it and on the grid.
- **Explicit time step.** The step is tied to the cell size and the fastest signal:
  roughly `dt = dx / 1300 m/s` for air at rest and about `dx / 1700 m/s` with a supersonic jet
  on the grid. A 1 mm cell gives about 0.6 to 0.8 microseconds per step. Halving the cell size doubles the
  number of steps for the same simulated time and multiplies the cell count by 8, so the
  work goes up 16 times.
- **Steps needed.** Pressure waves have to cross the domain several times before the flow
  settles. The nozzle cases above (32 cells long) settled in 1400 to 2000 steps, about 45 to 65
  steps per cell of domain length. Devices where slow entrained air has to travel the length
  of a duct need more: in the air mover demo (64 cells long) the feed flow was steady after 2000 steps but the
  suction flow was still creeping up after 8000 steps (10 ms of simulated time). Expect tens
  of thousands of steps for an ejector at useful resolution.
- **Very strong shocks.** HLLC lets very strong grid-aligned shocks go unstable sideways
  (carbuncle). It showed at a shock Mach number of about 200 and not at pressure ratio 100 or
  in the Mach 2.8 jet of the 5 bar nozzle. Where the onset lies in between was not tested. The
  usual cure is to blend in a more diffusive flux at strong shocks; not done.
- **32-bit floats.** Pressure differences below about 1 part in a million of ambient (0.1 Pa)
  are lost in round-off. That is fine for compressed-air work, but this is not the tool for
  low-speed flows: use the existing solver below about Mach 0.2.
- **One gas.** Air and air only: no mixing of different gases, no moisture, no condensation
  in cold supersonic jets, constant gamma.
- **Start-up.** With rampSteps = 0 the feed is opened instantly, like bursting a diaphragm.
  That is robust here but sends a strong shock through the device first.

## Wiring it into the page

1. Create it next to the existing solver: `compressible = await CompressibleSolver.create()`.
   It makes its own GPU device.
2. Use it when a feed port is set by pressure (the page already has `p.pressure` in Pa gauge
   and uses `nozzle()` from airjet.js for the incompressible stand-in). Build `ports` in the
   order of `grid.ports`: feed ports become
   `{type: "feed", totalPressure: 101325 + p.pressure, totalTemperature: 293.15}`, open ports
   `{type: "open", pressure: 101325, temperature: 293.15}`. buildDomain still demands a feed
   `speed` above zero; any positive number will do, the compressible solver ignores it.
3. Check `cells <= compressible.maxCells` and show `CompressibleSolver.memoryBytes(cells)`.
4. Run loop: `await compressible.step(k)` works like Solver.step. Simulated time is
   `compressible.time` (the step size changes during a run, so do not multiply steps by a fixed dt).
   Stop and tell the user if `compressible.diverged` becomes true.
5. Results: `read()` returns 8 floats per cell in SI units, not the 4 lattice-unit floats of
   Solver.read(). The helpers in src/results.js (fieldStats, sliceImage, streamlines, probe)
   index 4 floats per cell, so either give them a stride argument or build a 4-float array
   (density, velocity) from `fields` for them. New things worth drawing: Mach number
   (offset 6), pressure (offset 4) and temperature (offset 5). On very large grids draw with
   `viewGrid(grid, r.stride)` exactly as now.
6. Port read-outs come straight from `r.ports[n].massFlow` (kg/s); no need for portMetrics.
   Useful to show: mass flow of each port, free-air flow (mass flow / 1.204 kg/m3), the ratio
   of air drawn in to air fed, and `r.maxMach`.
7. Between full read-backs call `totals()` to judge convergence. Average the port mass flows
   over a few hundred steps before showing them or comparing them: single readings wobble
   while pressure waves are still travelling through the device.
8. Tell the user when the run is at a resolution where the numbers are rough: fewer than
   about 8 cells across the smallest nozzle.

## Running the checks

```
npx vite build -c dev/vite.compressible.config.js
cd /home/claude/c-dist && python3 -m http.server 8799 &
python3 /home/claude/ctest/run.py "sod(0, 100)"
```

Functions on the check page: `sod(axis, cells)`, `box()`, `nozzleRun(gaugeBar, options)`,
`uniform(mach, pulse, steps, every)`, `plate()`, `wedge()`, `jetpump(res, gaugeBar, maxSteps, every)`,
`speed()`, `viewCheck()`.

`viewCheck()` compares the block-averaged copy used for very large grids against a full
read-back: they agree to 3 parts in 10 million, and the GPU-summed mass and wall force are identical.
