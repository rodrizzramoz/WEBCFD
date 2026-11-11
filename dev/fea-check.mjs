// Check of the FEA core against cantilever beam theory, plus an optimiser smoke test.
// Run with: node dev/fea-check.mjs
import { Model, Optimiser } from "../src/fea/core.js";
// Cantilever: L x b x d, fixed at x = 0, tip load P in -z.
function beam(nx, ny, nz, L) {
  const h = L / nx, NX = nx + 1, NY = ny + 1, NZ = nz + 1;
  const solid = new Uint8Array(nx * ny * nz).fill(1);
  const fixed = new Uint8Array(NX * NY * NZ), force = new Float64Array(3 * NX * NY * NZ);
  const P = 100, tip = [];
  for (let k = 0; k < NZ; k++) for (let j = 0; j < NY; j++) { fixed[0 + NX * (j + NY * k)] = 1; tip.push(nx + NX * (j + NY * k)); }
  for (const n of tip) force[3 * n + 2] = -P / tip.length;
  return { m: new Model({ nx, ny, nz, h, solid, fixed, force, E: 2e9, nu: 0.3 }), P, h, tip };
}
for (const [nx, ny, nz] of [[20, 4, 4], [40, 8, 8], [60, 12, 12]]) {
  const L = 0.1, { m, P, h, tip } = beam(nx, ny, nz, L);
  const t = Date.now(), info = m.solve(1e-6, 20000);
  const b = ny * h, d = nz * h, I = b * d ** 3 / 12, E = 2e9, G = E / 2.6;
  const theory = P * L ** 3 / (3 * E * I) + P * L / (5 / 6 * G * b * d);
  let tipz = 0; for (const n of tip) tipz += m.u[3 * n + 2]; tipz /= tip.length;
  const vm = new Float64Array(m.ne); m.vonMises(vm);
  const sigTheory = P * L * (d / 2) / I;
  console.log(nx, ny, nz, "elems", m.ne, "iters", info.iterations, "ms", Date.now() - t, "tip", (-tipz * 1e3).toFixed(4), "mm theory", (theory * 1e3).toFixed(4), "ratio", (-tipz / theory).toFixed(3), "vm max MPa", (Math.max(...vm) / 1e6).toFixed(2), "root bending theory", (sigTheory / 1e6).toFixed(2), "compliance", m.compliance().toExponential(3), "P*tip", (P * -tipz).toExponential(3));
}
// quick optimisation smoke test
{
  const { m } = beam(40, 4, 16, 0.1);
  const o = new Optimiser(m);
  const t = Date.now();
  for (let i = 0; i < 25; i++) { const r = o.update(0.5); if (i % 6 === 0 || i === 24) console.log("opt", i, "c", r.compliance.toExponential(3), "change", r.change.toFixed(3), "cg", r.iterations, "vol", o.volume().toFixed(3)); }
  console.log("opt ms", Date.now() - t);
  // print mid-plane density
  const { nx, ny, nz } = m; let out = "";
  for (let k = nz - 1; k >= 0; k--) { for (let i = 0; i < nx; i++) { const x = o.x[m.cellToElem[i + nx * (1 + ny * k)]]; out += x > 0.7 ? "#" : x > 0.3 ? "+" : "."; } out += "\n"; }
  console.log(out);
}
