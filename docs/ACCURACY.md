# Flowbench accuracy report

Date: 11 November 2026. Scope: how far the current solver is from reference data, why, and what to change first.

How to read the source tags: "[S3]" points to the numbered list at the end. Every source in that list was opened during this review unless it is in the "not opened" list. Where a number comes from memory or from my own arithmetic it is marked "unverified" or "my estimate". Some PDFs were read through a text extraction tool that garbled equations and tables; those cases are flagged too.

## 1. Summary

1. The solver is behaving the way this class of method is known to behave. FluidX3D, which uses the same building blocks (D3Q19, halfway bounce-back, Smagorinsky, no wall model), warns in its own documentation that body forces in highly turbulent flow can be too large "by up to a factor 2" and that a wall function is still needed [S13]. Our sphere result (0.93 against 0.4 to 0.5) is that warning reproduced.
2. The laminar pipe error (12 percent) is almost certainly the staircase wall. Simple bounce-back on a curved wall is first order accurate; Bouzidi type interpolated bounce-back is second order and costs 20 to 60 percent more boundary compute [S10]. This is the best value change available.
3. The turbulent cases (sphere, Ahmed body, venturi) are limited by resolution and the missing wall model, not by the staircase alone. Published lattice Boltzmann results that match experiment use 60 to 640 cells across the body, local grid refinement and usually a wall model [S7, S8, S15, S16]. We ran 5.5 to 22 cells.
4. The venturi number (37 inlet dynamic pressures) is not only a solver error. It is larger than even a complete failure of the diffuser could produce at the nominal throat speed, which points to an effective throat that is too small plus a jet that leaves the short outlet without recovering. The reported static pressure drop then includes kinetic energy that has not been lost yet. See section 4.5.
5. Blockage and the short domain matter for the sphere (my estimate: up to about 20 percent of the reading), but the side boundaries are not solid walls, so the classical tunnel correction is only an upper bound.
6. Mach number is a minor error for external flow (lattice speed 0.07) but not for the venturi, where the measured peak speed implies a local Mach number of roughly 0.2 to 0.37.

Top three changes: (a) domain, outlet and reporting fixes, which are nearly free; (b) interpolated bounce-back with a per-link wall distance from the voxeliser; (c) a wall function with near-wall damping of the Smagorinsky model. Details in section 5.

## 2. What the code does (as read from the source)

- `src/solver.js`: D3Q19, pull streaming with two full copies of the populations (2 x 76 bytes), second order equilibrium, projection type regularised collision with one relaxation time, Smagorinsky eddy viscosity from the non-equilibrium stress (Cs = 0.12, no damping near walls), halfway bounce-back, momentum exchange force summed on the GPU in fixed point.
- Boundary cells hold an equilibrium state. Far field: fixed velocity and density 1. Inlet: fixed velocity, density copied from the neighbour. Outlet: density 1, velocity copied from the neighbour and clamped at 0.25.
- `src/domain.js`: one ray per (y, z) column along X, signed crossings, cell centre inside means solid. External domain: 1.5 reference lengths upstream, 4 downstream, 1.5 each side, where the reference length is the larger of the cross size and 0.4 times the length. The inlet face and the four side faces are far field; the downstream face is an outlet.
- `latticeFor` in `src/main.js`: lattice inlet speed is the smaller of 0.07 and 0.11 divided by the contraction factor; lattice viscosity is capped at 0.3; external runs start impulsively with the full flow, internal runs ramp up.
- Internal pressure drop is the mean static pressure next to the inlet minus the mean static pressure next to the outlet (`internalMetrics` in `src/results.js`). It is not a total pressure loss.

## 3. Reference values

| Case | Reference value | Source and status |
|---|---|---|
| Smooth sphere, Re 10^4 to about 2 x 10^5 | Cd = 0.47 | White, Fluid Mechanics, Table 7.3, as reproduced in [S2]. Textbook not opened directly. |
| Smooth sphere, same range | Cd about 0.39 at Re 10^4, 0.41 at 6.6 x 10^4, 0.42 at 10^5 | Morrison correlation [S1]. The constants were partly garbled in extraction and the values are my arithmetic, so treat as approximate. A fair target band is 0.40 to 0.50. |
| Sphere drag crisis | Cd falls from about 0.5 to about 0.2 near Re 3 x 10^5 | [S3] (encyclopaedia summary). Morrison's curve [S1] bottoms out near 0.09 at Re 4 to 5 x 10^5 (my arithmetic). A simulation paper describes the drop as roughly to one fifth [S8]. Achenbach's 1972 paper was not opened. |
| Sphere, blockage | Drag coefficient and critical Reynolds number both rise with blockage | Achenbach 1974 abstract [S4]. Numbers are behind a paywall, not verified. |
| Cube, face on, Re above 10^4 | Cd = 1.07 (edge on: 0.81) | White Table 7.3 via [S2]. The often quoted 1.05 (Hoerner) is unverified. |
| Ahmed body, 25 degree slant, with ground | Cd = 0.2875 at 60 m/s, Re 4.29 x 10^6 on length | Read from Ahmed, Ramm and Faltin 1984 Fig. 4 by a vendor validation report [S7]. The SAE paper itself was not opened. The usual quote is 0.285. |
| Ahmed body, 35 degree slant | about 0.26 | Unverified, from memory. I could not open a source for it. Meile et al. 2011 (0.299 at 25 degrees, 0.279 at 35 degrees) is also unverified. |
| Ahmed body, set-up | Lienhart and Becker: 40 m/s, Re 2.78 x 10^6, blockage under 4 percent, pressure and velocity data | [S5]. Ground clearance is 0.174 body heights, which is 50 mm, and the tests had a floor and stilts [S6]. Our demo has neither. |
| Classical venturi, geometry | Convergent 21 degrees included; divergent 7 to 15 degrees included, 7 to 8 recommended for low loss | ISO 5167-4:2022 clauses 5.2.3 and 5.2.5, read from the draft Indian adoption [S17]. |
| Classical venturi, discharge coefficient | 0.995 (machined, pipe Re 2 x 10^5 to 10^6) | Same, clause 5.5 [S17]. |
| Classical venturi, permanent loss | Generally 5 to 20 percent of the differential pressure; falls with larger diameter ratio and higher Re, rises with divergent angle and roughness | Same, clause 5.9 [S17]. |
| Our venturi, expected values | Ideal differential = (1/beta^4 - 1) = 15 inlet dynamic pressures for beta = 0.5. Loss therefore about 0.75 to 3 inlet dynamic pressures | Bernoulli plus the ISO range; my arithmetic. Caveat: our demo has a 28 degree convergent, a 12.7 degree divergent, a 20 mm bore and Re 20,000, all outside the range where the ISO figures are certified, so expect the upper end or a little above. |
| Conical diffuser loss (Idelchik) | Not found | I could not open Idelchik or an equivalent table. Use the ISO figure above. |
| Sudden expansion (worst case for a failed diffuser) | Loss = (V1 - V2)^2 / 2 per unit mass, the Borda-Carnot result | Unverified, standard textbook formula from memory. |
| Laminar pipe | Darcy friction factor = 64 / Re, valid below Re 2300 | [S18]. Equivalent to pressure drop = 32 x viscosity x length x mean speed / diameter^2. |
| Laminar entry length | L/D = [0.619^1.6 + (0.0567 Re)^1.6]^(1/1.6); about 0.62 D as Re tends to zero. Simple rule 0.06 Re D. | Durst et al. 2005 as quoted in [S19]; simple rule from [S18]. At Re 8 this gives about 0.83 D (my arithmetic). |
| Laminar entry loss | Extra loss of about 1.2 to 1.3 dynamic pressures for a round pipe at high Re, larger at very low Re | Unverified, from memory of Shah and London. The only value I could confirm is 1.43 for a square duct [S20]. For our pipe (fully developed drop about 40 dynamic pressures) this term is about 3 percent, so the uncertainty hardly matters. |
| Blockage, closed tunnel, bluff body | Dynamic pressure rises by eps x Cd x S / C, with eps about 2.5 for three-dimensional bodies (2.77 for a square plate), so Cd_corrected = Cd / (1 + 2.5 Cd S / C) | Maskell 1963 [S21]; the constant is confirmed in AGARD-AG-336 [S22], which also notes that it tends to over-correct at large area ratios. The final rearranged formula was derived from the paper's relations because the printed equation was garbled. Maskell assumes solid walls and a fixed separation line, so it is doubtful for spheres. |

## 4. What published lattice Boltzmann work achieves

### 4.1 Wall treatment on curved walls

- Simple (staircase) bounce-back converges at first order; Bouzidi (BFL) and Guo-Zheng-Shi converge at second order. Tested in round tubes 12 to 96 cells across at Re up to 300. In pulsating flow at 48 cells across, the error measure was 0.0085 to 0.034 for simple bounce-back against 0.0013 to 0.013 for BFL. BFL cost 1.3 to 1.6 times a simple boundary update and was judged the best all-round choice. Collision operator and velocity set made little difference there [S10].
- For a fixed sphere at Re 20 to 150, interpolated bounce-back reaches about 1 percent drag error at 15 cells per diameter; immersed boundary methods need about 25 [S9].
- For a sphere in Stokes flow at 16 cells per diameter with a two-relaxation-time collision (magic parameter 3/16), plain bounce-back overestimates the force by under 5 percent, and the interpolated schemes (central linear interpolation, multi-reflection) are clearly better. With that collision choice the force does not change with viscosity [S11].
- With a single relaxation time the effective wall position of halfway bounce-back moves with tau. The error in plane Poiseuille flow is Uc x [4 tau (4 tau - 5) + 3] / [3 (n - 1)^2] and is acceptable for tau between 0.5 and 1.25 [S12]. Our cap of 0.3 on lattice viscosity allows tau up to 1.4, just outside that range.
- Interpolated bounce-back does not conserve mass exactly except when the wall is exactly halfway [S14].
- Bottom line for our question: at low Reynolds number, staircase bounce-back on a sphere of 10 to 30 cells is wrong by a few percent to perhaps 10 percent, and interpolation brings that to about 1 percent. I found no paper that isolates the staircase error on a sphere at Re near 10^5 with 10 to 30 cells, because nobody publishes that combination; the resolution is far too low for the physics.

### 4.2 Turbulent bluff bodies

- Sphere, cumulant collision, interpolated bounce-back, no turbulence model, five refinement levels, 256 finest cells per diameter: Cd at Re 1000 within the reference band (0.48 to 0.52 for 64 to 256 cells). Plain BGK was unstable above Re 8000. The authors did not trust their own results at Re 10^5 without a finer grid [S15].
- Sphere drag crisis: captured with a tuned cumulant method at 410, 512 and 640 finest cells per diameter (41 to 133 million nodes, box 11 diameters wide, so blockage under 1 percent). The 410 cell grid still got the critical Reynolds number too high and the drag drop too small; the untuned cumulant method missed the crisis entirely [S8].
- Ahmed body, 25 degrees, commercial lattice Boltzmann solver with k-omega SST and wall modelling: Cd error 32.5 percent at 5 mm surface cells (21 million cells, about 58 cells per body height), 7.7 percent at 2 mm (50 million, about 144 cells), 4.7 percent at 0.8 mm (183 million, about 360 cells) [S7]. Cells per height are my conversion from the 288 mm body height.
- Ahmed body with PowerFLOW: 0.72 mm voxels, about 400 cells across the body height, nine refinement levels, wall law with pressure gradient effects [S16].
- FluidX3D: same wall treatment as ours; states that even large resolution does not capture the turbulent boundary layer and that forces can be up to twice too large [S13].

### 4.3 Wall functions

- Wall-function bounce with Spalding's law, D3Q19, multiple relaxation times, Smagorinsky Cs = 0.1 with van Driest damping, turbulent channel at friction Reynolds number 2003: velocity profile error fell from 0.43 (plain bounce-back) to 0.08 with 40 cells across the channel, and from 0.24 to 0.07 with 80. Plain bounce-back gave a log-law intercept of -0.1 to -4.1 against a reference of 5.4, which means a badly overpredicted wall shear [S23].
- Cumulant method with a partial slip wall function fed through interpolated bounce-back using the wall distance q: reasonable velocity profiles with 10 to 40 cells per channel half height up to friction Reynolds number 16,000; applied to a car at first-cell y+ of 250 to 510 [S24].

### 4.4 Collision operators

- In a stability study on D2Q9, the projection regularised model (the family ours belongs to) was stable only up to Mach 0.11 to 0.14 in a double shear layer at Re 3 x 10^4, against 0.49 to 0.52 for the recursive regularised model. Recursive regularised and regularised cumulant or central-Hermite models were the best candidates at low viscosity [S25]. Our external runs sit at Mach 0.12 and rely on the Smagorinsky term for the remaining margin.
- Cumulant collisions extend the stable Reynolds range by more than a factor of ten over BGK on the same grid [S15] and, when tuned, change the predicted physics (drag crisis) [S8]. They need D3Q27.

## 5. Diagnosis of each measured result

Confidence: high means I would be surprised to be wrong; medium means the leading cause is likely but the split is uncertain.

### 5.1 Laminar pipe, 16 cells, 12 percent high

- Leading cause: staircase wall. Pressure drop at fixed flow scales with diameter^-4, so 12 percent corresponds to an effective bore about 3 percent (half a cell) too small (my arithmetic). First order behaviour of simple bounce-back in tubes is documented [S10]. Confidence: high.
- Minor: tau dependent wall position. With tau about 0.9 the formula in [S12] gives about 0.3 percent of centreline speed (my arithmetic). Entry loss uncertainty is about 1 percent of the total.
- Not a cause: Mach number, domain, mass conservation (balance is 100.0 percent).
- Confirming tests: run 16, 32 and 64 cells across; the error should roughly halve each time. Run a square duct aligned with the grid, where the wall really is halfway; it should be within 1 to 2 percent at 16 cells. Measure the drop between two stations more than 1 D from the inlet to remove the entry term.

### 5.2 Cube, 12 cells, Cd 1.15 to 1.2 against 1.07

- Separation is fixed by the sharp edges and the faces are grid aligned, so the method is in its comfort zone. The 8 to 12 percent excess is most likely resolution: the shear layers are about one cell thick, which is 8 percent of the body side, so the body looks slightly larger than it is. Confidence: medium.
- Blockage (7 percent): Maskell's formula would take 1.175 down to about 0.98 (my arithmetic), which over-corrects, as expected because our sides are fixed-velocity, fixed-pressure cells and not solid walls. The true blockage effect is smaller and its size is unknown until tested.
- Confirming tests: double the domain cross-section at fixed cell size (isolates blockage), then double the resolution at fixed domain (isolates resolution).

### 5.3 Sphere, 22 cells, Cd 0.93 against 0.4 to 0.5

- Leading cause: unresolved boundary layer with no wall model on a staircase surface. At Re 66,000 the laminar boundary layer before separation is of order 0.01 diameters thick, about a fifth of one cell (my estimate), so the solver cannot place the separation line; the staircase and the one-cell-thick numerical shear layer produce a wake as wide as a blunt body. The factor of two matches the FluidX3D warning [S13]. Confidence: high that this is the largest part.
- Second: blockage and domain. With about 10 percent blockage the solid-wall upper bound from Maskell is about 23 percent (my arithmetic), and blockage is known to raise sphere drag [S4]. The inlet is only about one diameter ahead of the sphere and the fixed-pressure outlet cuts through the wake about 2.5 diameters behind it. Confidence: medium that this is worth 10 to 20 percent.
- Minor: Mach 0.12 (errors of order Mach squared, about 1 to 2 percent), and run time if the average was taken before the wake settled.
- Confirming tests: (1) same sphere at Re 100, where the reference is about 1.0 to 1.1 (my arithmetic from [S1]; unverified textbook value 1.09) and the boundary layer is resolved; if this is within 5 to 10 percent, the high Re error is physics, not a bug. (2) Standard domain against a domain twice as wide and long. (3) Halve the lattice speed.

### 5.4 Ahmed body, 5.5 cells tall, Cd 1.5 against 0.29

- Leading cause: resolution, by a wide margin. The 100 mm nose radius is under two cells, so the front is a brick and the flow separates there; the 25 degree slant is two or three stair steps. A wall-modelled commercial solver is still 32 percent off at about 58 cells per height [S7]. Confidence: high.
- Second: at 5.5 x 7.5 cells the voxel frontal area can differ from the true area by 10 to 20 percent and a one-cell shear layer adds a third to the effective height (my estimates).
- Third: the missing ground and stilts change the reference itself, so 0.29 is not strictly the right target for our geometry.
- Confirming test: run 16, 32 and 64 cells per height. Expect a steep fall towards roughly 0.4 to 0.6 and then a plateau that only a wall model moves.

### 5.5 Venturi, 11 cells in the throat, 37 inlet dynamic pressures

- The expected loss is 0.75 to 3. If the diffuser failed completely at the nominal throat speed (4 times inlet), the sudden expansion formula gives (4 - 1)^2 = 9. A reading of 37 needs an effective jet speed near 7 times inlet, since (7 - 1)^2 = 36 (my arithmetic, unverified formula). The measured peak of 10 times inlet supports this: the effective throat is only about 55 to 60 percent of its nominal area.
- Likely contributors, in order: (1) a peaked, laminar-like throat profile, because Smagorinsky without wall damping adds its largest eddy viscosity in the wall cells and 11 cells cannot hold a turbulent profile; (2) staircase steps in the 14 degree convergent cone that trip a separation and a vena contracta; (3) a voxelised throat smaller than nominal; (4) a diffuser that cannot recover pressure at all because the jet never reattaches, which is exactly the regime a wall model exists for; (5) only 20 mm of outlet pipe, so the jet leaves through the fixed-pressure outlet and the static drop includes kinetic energy that has merely not been recovered; (6) compressibility, since a peak lattice speed of 0.1 to 0.2 is Mach 0.2 to 0.37, giving density changes of 2 to 7 percent. Confidence: high that the jet does not recover; medium on the ranking.
- Confirming tests: count throat fluid cells and compare with the true area; report total pressure loss (static plus dynamic, mass-flow weighted) as well as static drop; add 10 diameters of outlet pipe; check the throat static pressure against Bernoulli (15 inlet dynamic pressures below inlet); rerun at Re about 200 where the flow is resolved; halve the lattice speed.

## 6. Ranked solver changes

Ranked by benefit per unit of effort for a WebGPU compute shader.

### 6.1 Domain, outlet and reporting fixes (effort: hours)

- Fixes: blockage, wake cut-off, the misleading static pressure drop.
- What to do: show the blockage ratio and warn above about 3 percent; make the default external domain larger (published sphere work uses a box 11 diameters wide [S8, S15]); put the outlet at least 8 to 10 body sizes downstream; for internal flow add automatic straight extensions at inlet and outlet and report total pressure loss; average forces over a stated number of flow-through times and show the running mean.
- Expected gain: my estimate is 10 to 20 percent on the sphere and cube readings; decisive for interpreting the venturi.
- Risks: more cells for the same body resolution. A box 9 x 9 x 15 body sizes costs about 12 times the cells of the current default (my arithmetic).

### 6.2 Interpolated bounce-back (effort: days)

- Fixes: first order geometry error on every curved or inclined wall; makes the result vary smoothly when the model is moved or the grid changed.
- Expected gain: second order convergence and roughly 3 to 6 times lower error in tubes at 48 cells [S10]; about 1 percent drag error on a sphere at 15 cells at low Re [S9]. The pipe case should go from 12 percent to 1 to 3 percent (my estimate). It will not fix the sphere at Re 66,000.
- Data per cell: only cells next to a wall need it. Store 18 values of q (the fraction of the link from the fluid node to the wall), quantised to 8 bits, in a compact array indexed from the flags buffer. Cost is 18 to 24 bytes per boundary cell and nothing for bulk cells.
- Voxeliser: it currently casts rays along X only. Cast along all 9 link axes (3 axial, 6 diagonal), or for each boundary link intersect the segment with nearby triangles using a uniform triangle bin grid. Fall back to q = 0.5 when no hit is found.
- Shader: the pull scheme already has what is needed in the source buffer. For a link with outgoing direction o and incoming direction i at cell x: if q is below 0.5, f_i(x) = 2q f_o(x) + (1 - 2q) f_o(x - c_o); otherwise f_i(x) = f_o(x) / (2q) + (2q - 1) / (2q) f_i(x), all taken after collision. This is the linear Bouzidi rule written from memory; the original paper was not opened, so check it against the paper or the Palabos or OpenLB source before coding. If the second cell is not fluid, use halfway bounce-back. The force per link becomes (f_o + f_i) c_o.
- Risks: small mass drift, since only q = 0.5 conserves mass [S14]; less robust than plain bounce-back at very low viscosity; narrow gaps need the fallback. Keep tau inside 0.5 to 1.25 [S12] or move to two relaxation times with the magic parameter 3/16 so the wall position stops depending on viscosity [S11].

### 6.3 Wall function plus near-wall damping of Smagorinsky (effort: one to two weeks)

- Fixes: wall shear and near-wall velocity at high Reynolds number; the over-thick numerical boundary layer; part of the venturi and Ahmed error.
- Expected gain: channel-flow profile error cut from 0.43 to 0.08 at 40 cells [S23]. With wall modelling an Ahmed body reaches about 30 percent at 58 cells per height and under 10 percent at 144 [S7].
- Data per boundary cell: unit wall normal (3 signed bytes) and wall distance, both available from the same ray hits as q.
- Shader: take the tangential speed at the first or second fluid cell, obtain the friction velocity from a wall law (Spalding with a few Newton steps as in [S23], or an explicit power law to avoid iteration), then either modify the wall-facing populations to carry that shear [S23] or impose a partial slip velocity through the interpolated bounce-back term [S24]. Multiply the Smagorinsky length by a van Driest factor [S23].
- Risks: equilibrium wall laws assume attached flow, so separation on smooth bodies (sphere, 25 degree slant, diffusers) stays unreliable. Noisy normals on staircase geometry. Do this after 6.2, because it needs the same geometric data.

### 6.4 More cells from the same memory (effort: days to a week)

- Fixes: resolution, which is the dominant error in every turbulent case.
- What to do: in-place streaming (one copy of the populations) and optionally 16-bit storage. FluidX3D reports 93 bytes per cell with 32-bit storage and 55 with 16-bit, against our 190 [S13].
- Expected gain: 2 to 3.4 times more cells, which is 1.26 to 1.5 times finer in each direction. Also raise the per-buffer ceiling by splitting the populations over several buffers.
- Risks: in-place schemes are fiddly and race-prone in a compute shader; 16-bit storage needs careful scaling.

### 6.5 Better collision model (effort: one week or more)

- Recursive regularisation is a moderate change to the existing collision and has much wider stability than the projection form [S25]. A cumulant model is the best documented for under-resolved turbulence [S8, S15] but needs D3Q27 (42 percent more memory) and a long shader.
- Expected gain: robustness and less numerical damping; no source promises a specific drag improvement at our resolutions. Do not expect it to fix the sphere.

### 6.6 Nested grid refinement (effort: weeks)

- Every high Reynolds result in section 4.2 used it [S8, S15, S16]. It is the real route to 100 or more cells across a body in a large domain, and the hardest item here. Not recommended until 6.1 to 6.3 are done.

### 6.7 What this class of solver will not do, even after these changes

- Predict the drag crisis or any transition-sensitive flow. That took 410 to 640 cells per diameter and a tuned cumulant model [S8].
- Give absolute drag of streamlined or smoothly curved bodies to engineering accuracy. Expect 10 to 30 percent at best on a single desktop GPU without refinement, and worse where the separation point is free.
- Predict diffuser stall, venturi recovery at high Reynolds number, or skin-friction dominated losses in long turbulent pipes without a wall function.
- Handle Mach numbers above about 0.3, cavitation, heat transfer or walls thinner than a cell.
- It can rank design variants at fixed settings, give sharp-edged bluff body drag to about 10 percent, and give laminar internal flow to a few percent once 6.2 is in.

### 6.8 Grid sizes on a 24 GB GPU

- At 190 bytes per cell, 24 GB holds 126 million cells; keeping 10 percent free gives about 113 million, for example 484^3 or 768 x 384 x 384 (my arithmetic).
- Current default domain (6.5 x 4 x 4 body sizes): about 100 cells across the body. A domain with about 1 percent blockage (15 x 9 x 9): about 45 cells across.
- With in-place streaming at about 93 bytes per cell [S13] these become roughly 130 and 57 cells; FluidX3D quotes 646^3 for 24 GB at that footprint [S13].
- Browser limit: the code sizes the grid from the adapter's largest buffer divided by 76 bytes. The WebGPU defaults are only 256 MB per buffer and 128 MB per storage binding; adapters may report more, and the values vary by browser and GPU [S26]. I could not verify what Chrome grants on a 24 GB card. If the limit is 2 GB per binding (unverified), the present layout stops near 28 million cells (about 304^3) whatever the memory, so check `solver.maxCells` on the target machine and split the populations across buffers if needed.

## 7. Validation plan

Run each case at two or three resolutions and record the trend, not only the final value. Use blockage under 2 percent for the external cases.

| # | Case | Target | Tolerance to aim for | What it tests |
|---|---|---|---|---|
| 1 | Laminar round pipe, Re 10 to 100, drop measured between two stations beyond the entry length | 32 x viscosity x length x speed / diameter^2 [S18] | Now: error should halve with each doubling. After 6.2: within 3 percent at 16 cells, 1 percent at 32 | Wall position, viscosity scaling |
| 2 | Sphere at Re 100 | Cd about 1.0 to 1.1 (my arithmetic from [S1]; treat as approximate) | Within 10 percent at 20 cells now; within 3 percent after 6.2 [S9] | Curved wall and force evaluation with physics fully resolved |
| 3 | Cube face on, Re above 10^4 | Cd = 1.07 [S2] | Within 10 percent at 24 cells | Sharp-edge separation, blockage handling |
| 4 | Sphere at Re 10^4 to 6.6 x 10^4 | Cd 0.40 to 0.50 [S1, S2] | Honest target: within 25 percent at 64 cells after 6.2 and 6.3. Expect a failure today | Smooth-body separation, wall model |
| 5 | Ahmed body, 25 degrees, with a floor at 50 mm clearance, 40 or 60 m/s | Cd = 0.2875 [S7] | Within 30 percent at about 60 cells per height, which is what a commercial wall-modelled solver achieved [S7] | Whole chain on a realistic shape |
| 6 | Venturi, beta 0.5, with 10 diameters of outlet pipe | Throat static pressure 15 inlet dynamic pressures below inlet (Bernoulli, about 1 percent more with C = 0.995 [S17]); total pressure loss 0.75 to 3 [S17] | Throat drop within 10 percent at 24 or more throat cells; loss under 6 at 40 or more throat cells | Contraction, Mach control, diffuser recovery |

Three procedural checks to run alongside: domain independence (double the box, change under 3 percent), Mach independence (halve the lattice speed, change under 3 percent), and time convergence (running mean over the last third of the run changes by under 2 percent).

## 8. What could not be verified

- Ahmed et al. 1984 and Lienhart and Becker 2003 were not opened; the 25 degree value comes through a vendor report [S7], and the 35 degree value is from memory.
- Achenbach 1972 and the numbers in Achenbach 1974 (paywalled); Hoerner's cube value of 1.05; Idelchik's diffuser tables; the round-pipe entry loss coefficient; the Borda-Carnot formula (textbook, not opened).
- Bouzidi et al. 2001, Mei, Luo and Shyy 1999 and Ginzburg and d'Humieres 2003 were not opened. Their behaviour is taken from papers that test them [S9, S10, S11, S14]. The Bouzidi formula in 6.2 is from memory.
- Morrison's constants [S1] and Maskell's final equation [S21] were partly garbled in extraction.
- No source was found for staircase bounce-back on a sphere at Re near 10^5 with 10 to 30 cells, or for a Palabos, OpenLB or waLBerla validation report on these exact cases.
- Chrome's actual buffer limits on a 24 GB GPU.

## 9. Sources

Opened:

- [S1] F. A. Morrison, Data Correlation for Drag Coefficient for Sphere, Michigan Technological University, 2013. https://galileoandeinstein.phys.virginia.edu/more_stuff/Applets/Projectile/DataCorrelationForSphereDrag2013.pdf
- [S2] Volupe, Drag coefficients (reproduces White, Fluid Mechanics, 3rd ed., Table 7.3). https://volupe.com/support/c_d-and-c_l-calculation/
- [S3] Wikipedia, Drag crisis. https://en.wikipedia.org/wiki/Drag_crisis
- [S4] E. Achenbach, The effects of surface roughness and tunnel blockage on the flow past spheres, J. Fluid Mech. 1974 (abstract only). https://www.cambridge.org/core/services/aop-cambridge-core/content/view/B9276280823D702358961E435CA661C8/S0022112074001285a.pdf/div-class-title-the-effects-of-surface-roughness-and-tunnel-blockage-on-the-flow-past-spheres-div.pdf
- [S5] ERCOFTAC Knowledge Base, Ahmed body, Test Data AC1-05. https://kbwiki.ercoftac.org/w/index.php/Test_Data_AC1-05
- [S6] S. Krajnovic and L. Davidson, Flow around a simplified car, part 1: large eddy simulation, J. Fluids Eng. https://www.cfd-sweden.se/lada/postscript_files/sinisa-JFE-1.pdf
- [S7] SimScale, Validation case: aerodynamics of the Ahmed body with the lattice Boltzmann method (Pacefish). https://www.simscale.com/docs/validation-cases/aerodynamics-ahmed-body-lbm/
- [S8] M. Geier, A. Pasquali, M. Schonherr, Parametrization of the cumulant lattice Boltzmann method for fourth order accurate diffusion, part II, J. Comput. Phys. 2017. https://www.tu-braunschweig.de/fileadmin/Redaktionsgruppen/Forschung/SFB880/Veroeffentlichungen/Veroeffentlichung_A5_2017_Kutscher_Param_cumul_Journ_Compu_rev_2.pdf
- [S9] C. Peng, O. Ayala, L.-P. Wang, A comparative study of immersed boundary method and interpolated bounce-back scheme, part I, laminar flows. https://arxiv.org/abs/1906.05445
- [S10] R. W. Nash et al., Choice of boundary condition for lattice-Boltzmann simulation of moderate Reynolds number flow in complex domains. https://arxiv.org/abs/1211.0205
- [S11] C. Rettinger and U. Rude, A comparative study of fluid-particle coupling methods for fully resolved lattice Boltzmann simulations (waLBerla group). https://ar5iv.labs.arxiv.org/html/1702.04910
- [S12] X. He, Q. Zou, L.-S. Luo, M. Dembo, Analytic solutions of simple flows and analysis of nonslip boundary conditions for the lattice Boltzmann BGK model, J. Stat. Phys. 1997. https://scispace.com/pdf/analytic-solutions-of-simple-flows-and-analysis-of-nonslip-50b8fapfyx.pdf
- [S13] FluidX3D, README and documentation. https://github.com/ProjectPhysX/FluidX3D and https://github.com/ProjectPhysX/FluidX3D/blob/master/DOCUMENTATION.md
- [S14] Modified mass-conservative curved boundary scheme for lattice Boltzmann simulations. https://arxiv.org/abs/1908.09235
- [S15] M. Geier, M. Schonherr, A. Pasquali, M. Krafczyk, The cumulant lattice Boltzmann equation in three dimensions: theory and validation, Comput. Math. Appl. 2015. https://www.tu-braunschweig.de/fileadmin/Redaktionsgruppen/Forschung/SFB880/Veroeffentlichungen/Veroeffentlichung_A5_2015_Kutscher_cumul_latti_Compu_Mathe_rev_1.pdf
- [S16] Hoessler et al., LES and lattice Boltzmann (PowerFLOW) study of the 25 degree Ahmed body, IUTAM 2007 proceedings. https://eng-web77-v02.ocio.monash.edu/intranet/proceedings/iutam2007/pages/pdf/S6_Po_Hoessler.pdf
- [S17] ISO 5167-4:2022, Venturi tubes, text as circulated in the draft Indian adoption. https://services.bis.gov.in/tmp/F1WRD9122064_25072024_1.pdf
- [S18] C. Wassgren, Pipe Flows (course notes, Purdue University). https://engineering.purdue.edu/~wassgren/teaching/ME30900/NotesAndReading/PipeFlows_Losses_Reading.pdf
- [S19] R. J. Poole and R. P. Chhabra, Development length requirements for fully developed laminar pipe flow of yield stress fluids (quotes the Durst et al. 2005 correlation). https://scispace.com/pdf/development-length-requirements-for-fully-developed-laminar-22hpnuwidb.pdf
- [S20] Numerical investigation of the effect of an oblique flow entry on the pressure losses in square channels (quotes Shah and London). https://arxiv.org/abs/2502.07678
- [S21] E. C. Maskell, A theory of the blockage effects on bluff bodies and stalled wings in a closed wind tunnel, ARC R and M 3400, 1963. https://scispace.com/pdf/a-theory-of-the-blockage-effects-on-bluff-bodies-and-stalled-2rgdeqjly6.pdf
- [S22] AGARD-AG-336, Wind tunnel wall corrections, chapter 6. https://www.sto.nato.int/publications/AGARD/AGARD-AG-336/08chap06.pdf
- [S23] A wall function approach in lattice Boltzmann method: algorithm and validation using turbulent channel flow. https://arxiv.org/abs/2009.04352
- [S24] A. Pasquali, M. Geier, M. Krafczyk, Near-wall treatment for the simulation of turbulent flow by the cumulant lattice Boltzmann method. https://www.tu-braunschweig.de/fileadmin/Redaktionsgruppen/Forschung/SFB880/Veroeffentlichungen/Veroeffentlichung_A5_2017_Kutscher_Near-_treat_Compu_Mathe_rev_1.pdf
- [S25] C. Coreixas et al., Impact of collision models on the physical properties and the stability of lattice Boltzmann methods. https://arxiv.org/abs/2002.05265
- [S26] MDN, GPUSupportedLimits. https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits

Also opened, used for context only:

- Mei, Shyy, Yu, Luo, Lattice Boltzmann method for 3-D flows with curved boundary, NASA/CR-2002-211657. https://ntrs.nasa.gov/api/citations/20020063595/downloads/20020063595.pdf
- Lattice Boltzmann methodology for unconfined flows (cylinder at Re 100: Cd 1.33 to 1.4, Strouhal number 0.16 to 0.164 in the literature it cites). https://arxiv.org/abs/2605.02185
- Flow force calculation in lattice Boltzmann method. https://arxiv.org/abs/2210.10377

Not opened (cited by name only): Ahmed, Ramm and Faltin, SAE 840300 (1984); Lienhart and Becker, SAE 2003-01-0656; Meile et al. 2011; Achenbach 1972; Bouzidi, Firdaouss and Lallemand 2001; Mei, Luo and Shyy 1999; Ginzburg and d'Humieres 2003; Idelchik, Handbook of Hydraulic Resistance; Shah and London 1978; Hoerner, Fluid-Dynamic Drag; White, Fluid Mechanics; Durst et al. 2005.

## Addendum: cross-checks run on 11 November 2026

These were run after the review above, once OpenFOAM v1912 was available on the development machine. "Flowbench" is the in-browser lattice Boltzmann solver, in its two wall modes. "OpenFOAM" is the case this app exports (snappyHexMesh, simpleFoam, steady, k-omega SST where turbulent), solved on deliberately coarse meshes of similar cost. All Flowbench runs used the software-emulated GPU, so the grids are small.

| Case | Flowbench, cube steps | Flowbench, true surface | OpenFOAM export | Reference |
| --- | --- | --- | --- | --- |
| Laminar pipe, developed pressure gradient, 16 cells across | 0.976 of theory | 1.023 of theory | not measured | 1.000 (Hagen-Poiseuille) |
| Laminar pipe, inlet to outlet static drop in inlet dynamic pressures, Re 8 | 45.6 | 47.4 | 43.5 (7,552 cells) | about 46, with an uncertain entry-loss term |
| Sphere drag coefficient, 10 cells across, Re 66,000 | 1.05 | 0.77 | not run | about 0.47 |
| Cube drag coefficient, face on | 1.15 to 1.2 (12 cells across) | not run | 1.49, pressure part only (10 cells across, 5,584 cells) | about 1.05 |
| Air mover demo, air drawn in per air fed | not run | about 0.6 to 0.85, fluctuating (jet 4 cells across, 24,000 steps) | 2.8 (6,221 cells), 3.2 (24,839 cells) | 2 to 3 from a momentum balance |

What this shows:

- **Pipe friction is fine in both wall modes** (within 2.5 percent). The earlier "12 percent high" figure was mostly my own underestimate of the laminar entry loss at very low Reynolds number, not a solver error.
- **The true-surface wall option matters for curved bodies.** On the sphere it removed about half of the excess drag at the same grid. The rest is resolution and the missing wall model, as the review above explains.
- **OpenFOAM is not automatically better at the same cost.** On the cube, a coarse steady RANS run was further from the reference than the browser solver.
- **Jet-driven entrainment is where the browser solver is weakest at coarse grids.** With the nozzle only 4 cells across, the jet does not mix with the surrounding air quickly enough, and the predicted amplification is roughly a quarter of what OpenFOAM and a simple momentum balance give. This has not been tested at fine grids, where it should improve. Until it has been, use the OpenFOAM export or runner for air movers, ejectors and similar parts, and treat the browser result as a picture of the flow pattern.

Not verified: any of these on a real GPU or at production grid sizes; the reporting block of the exported case (force coefficients and patch flow rates through OpenFOAM function objects), because function objects do not start in the Ubuntu v1912 package. The numbers above were computed from the written fields instead.

### Compressible check, 12 November 2026

The compressible OpenFOAM export (rhoPimpleFoam, local time stepping, k-omega SST) was run on the air mover demo with the 6 mm nozzle fed at 2 bar gauge.

| Mesh | Mass flow through the nozzle | Nozzle theory (choked, ideal) | Air drawn in per air fed, by mass |
| --- | --- | --- | --- |
| 6,221 cells | 21.7 g/s | 20.1 g/s | 2.2 |
| 24,839 cells | 19.7 g/s | 20.1 g/s | 2.5 |

The nozzle mass flow is within 2 percent of theory on the finer mesh, which is the part of the answer that has an exact reference. The entrainment figure has no exact reference and still moves with the mesh. Two settings were needed for stability and are built into the export: the supply pressure is ramped over the first 500 iterations, and the local time step is capped at ten cell crossing times of a sound wave.
