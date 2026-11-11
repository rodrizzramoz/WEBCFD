// Checks of the tetrahedral mesher and solver. Run with: node dev/fea-tet-check.mjs
//   1. Cantilever against beam theory: tetrahedra with smoothed strain (what the
//      page uses), plain linear tetrahedra, and cubes, side by side.
//   2. Volume of a sphere and a cylinder against the exact values.
//   3. A plate with a through hole: hole kept, mesh conforming, no inverted elements.
//   4. Optimiser smoke test on the tetrahedral mesh.
import * as THREE from "three";
import { buildTetMesh, gridFor, facesOnTriangles, checkMesh, boundaryFace } from "../src/fea/tetmesh.js";
import { TetModel, TetOptimiser } from "../src/fea/tetcore.js";
import { Model } from "../src/fea/core.js";
import { voxelize } from "../src/domain.js";

const quick = process.argv.includes("--quick");
function soup(g, scale = 0.001) {
  if (g.index) g = g.toNonIndexed();
  return Float64Array.from(g.getAttribute("position").array, (v) => v * scale);
}
function trisWhere(tris, test) {
  const out = [];
  for (let t = 0; t < tris.length / 9; t++) {
    const o = 9 * t, u = [tris[o + 3] - tris[o], tris[o + 4] - tris[o + 1], tris[o + 5] - tris[o + 2]], v = [tris[o + 6] - tris[o], tris[o + 7] - tris[o + 1], tris[o + 8] - tris[o + 2]];
    const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]], l = Math.hypot(...n) || 1;
    const c = [(tris[o] + tris[o + 3] + tris[o + 6]) / 3, (tris[o + 1] + tris[o + 4] + tris[o + 7]) / 3, (tris[o + 2] + tris[o + 5] + tris[o + 8]) / 3];
    if (test(n.map((x) => x / l), c)) out.push(t);
  }
  return out;
}
// Supports and loads from picked triangles, the same way the page does it.
function tetSetup(mesh, tris, conds, E, nu, smooth) {
  const nn = mesh.nodes.length / 3, fixed = new Uint8Array(nn), force = new Float64Array(3 * nn), marked = new Uint8Array(nn);
  const loaded = [];
  for (const c of conds) {
    const sel = facesOnTriangles(mesh, tris, c.tris);
    if (!sel.nodes.size) throw new Error("condition found no mesh faces");
    for (const [n, a] of sel.nodes) {
      marked[n] = 1;
      if (c.kind === "fixed") fixed[n] = 1;
      else { for (let k = 0; k < 3; k++) force[3 * n + k] += c.force[k] * a / sel.area; loaded.push(n); }
    }
    c.area = sel.area;
  }
  const passive = new Uint8Array(mesh.vol.length);
  for (let q = 0; q < mesh.tets.length; q++) if (marked[mesh.tets[q]]) passive[q >> 2] = 1;
  return { model: new TetModel({ nodes: mesh.nodes, tets: mesh.tets, fixed, force, passive, E, nu, smooth }), loaded };
}
const pct = (v) => (v >= 0 ? "+" : "") + (100 * v).toFixed(2) + " %";

// ---------- 1. cantilever ----------
{
  const L = 0.1, b = 0.02, d = 0.02, P = 100, E = 3.5e9, nu = 0.36, G = E / (2 * (1 + nu)), I = b * d ** 3 / 12;
  const theory = P * L ** 3 / (3 * E * I) + P * L / (5 / 6 * G * b * d);
  console.log("CANTILEVER 100 x 20 x 20 mm, 100 N, E 3.5 GPa, nu 0.36. Beam theory with shear: " + (theory * 1e3).toFixed(4) + " mm, volume 40000 mm3");
  const tris = soup(new THREE.BoxGeometry(100, 20, 20).translate(50, 0, 0));
  const conds = () => [
    { kind: "fixed", tris: trisWhere(tris, (n) => n[0] < -0.9), force: [0, 0, 0] },
    { kind: "load", tris: trisWhere(tris, (n) => n[0] > 0.9), force: [0, 0, -P] },
  ];
  for (const res of quick ? [24, 48] : [24, 48, 72]) {
    const grid = gridFor(tris, res);
    let t = Date.now();
    const mesh = buildTetMesh(tris, grid), tMesh = Date.now() - t;
    console.log("  res " + res + ": " + mesh.vol.length + " tetrahedra, " + mesh.nodes.length / 3 + " nodes, quality min " + mesh.quality.min.toFixed(3) + " mean " + mesh.quality.mean.toFixed(3) +
      ", volume " + (mesh.volume * 1e9).toFixed(1) + " mm3 (" + pct(mesh.volume / (L * b * d) - 1) + "), meshed in " + tMesh + " ms");
    for (const smooth of [true, false]) {
      const { model, loaded } = tetSetup(mesh, tris, conds(), E, nu, smooth);
      t = Date.now();
      const info = model.solve(1e-6, 40000);
      let tip = 0; for (const n of loaded) tip += model.u[3 * n + 2]; tip /= -loaded.length;
      const vm = new Float64Array(model.ne); model.vonMises(vm);
      console.log("    tetrahedra, " + (smooth ? "smoothed strain" : "plain linear   ") + ": tip " + (tip * 1e3).toFixed(4) + " mm (" + pct(tip / theory - 1) + "), " + info.iterations + " passes, solve " + (Date.now() - t) +
        " ms, largest von Mises " + (Math.max(...vm) / 1e6).toFixed(2) + " MPa (root bending stress by theory " + (P * L * d / 2 / I / 1e6).toFixed(2) + ")");
    }
    // the cube mesh at the same grid
    const { nx, ny, nz, h, origin } = grid, NX = nx + 1, NY = ny + 1;
    const solid = voxelize(tris, origin, h, nx, ny, nz), nn = NX * NY * (nz + 1);
    const fixed = new Uint8Array(nn), force = new Float64Array(3 * nn), tipNodes = [];
    for (let k = 0; k <= nz; k++) for (let j = 0; j <= ny; j++) { fixed[NX * (j + NY * k)] = 1; tipNodes.push(nx + NX * (j + NY * k)); }
    for (const n of tipNodes) force[3 * n + 2] = -P / tipNodes.length;
    const cube = new Model({ nx, ny, nz, h, solid, fixed, force, E, nu });
    t = Date.now();
    const ci = cube.solve(1e-6, 40000);
    let ctip = 0; for (const n of tipNodes) ctip += cube.u[3 * n + 2]; ctip /= -tipNodes.length;
    console.log("    cubes: " + cube.ne + " elements, section " + (ny * h * 1e3).toFixed(2) + " x " + (nz * h * 1e3).toFixed(2) + " mm, volume " + (cube.ne * h ** 3 * 1e9).toFixed(1) +
      " mm3 (" + pct(cube.ne * h ** 3 / (L * b * d) - 1) + "), tip " + (ctip * 1e3).toFixed(4) + " mm (" + pct(ctip / theory - 1) + "), " + ci.iterations + " passes, solve " + (Date.now() - t) + " ms");
  }
  // What the answer should converge to: cubes on a grid that fits the 20 mm section exactly.
  // (A clamped root is a little stiffer than beam theory assumes, so this ends just under it.)
  for (const res of quick ? [25] : [25, 50]) {
    const nx = res, ny = res / 5, nz = res / 5, h = L / nx, NX = nx + 1, NY = ny + 1, nn = NX * NY * (nz + 1);
    const fixed = new Uint8Array(nn), force = new Float64Array(3 * nn), tipNodes = [];
    for (let k = 0; k <= nz; k++) for (let j = 0; j <= ny; j++) { fixed[NX * (j + NY * k)] = 1; tipNodes.push(nx + NX * (j + NY * k)); }
    for (const n of tipNodes) force[3 * n + 2] = -P / tipNodes.length;
    const cube = new Model({ nx, ny, nz, h, solid: new Uint8Array(nx * ny * nz).fill(1), fixed, force, E, nu });
    cube.solve(1e-6, 40000);
    let ctip = 0; for (const n of tipNodes) ctip += cube.u[3 * n + 2]; ctip /= -tipNodes.length;
    console.log("  reference, cubes on a fitted grid " + nx + " x " + ny + " x " + nz + ": tip " + (ctip * 1e3).toFixed(4) + " mm (" + pct(ctip / theory - 1) + ")");
  }
}

// ---------- 2. sphere and cylinder volume ----------
{
  console.log("VOLUME of curved parts (surface triangulated finely, so the triangle volume is the reference)");
  const triVol = (t) => { let v = 0; for (let o = 0; o < t.length; o += 9) v += t[o] * (t[o + 4] * t[o + 8] - t[o + 5] * t[o + 7]) - t[o + 1] * (t[o + 3] * t[o + 8] - t[o + 5] * t[o + 6]) + t[o + 2] * (t[o + 3] * t[o + 7] - t[o + 4] * t[o + 6]); return Math.abs(v) / 6; };
  const parts = [
    ["sphere radius 25 mm", soup(new THREE.SphereGeometry(25, 96, 64)), 4 / 3 * Math.PI * 25 ** 3],
    ["cylinder radius 15 mm, length 60 mm", soup(new THREE.CylinderGeometry(15, 15, 60, 128, 1)), Math.PI * 15 ** 2 * 60],
  ];
  for (const [label, tris, exact] of parts) for (const res of [16, 32]) {
    const grid = gridFor(tris, res), mesh = buildTetMesh(tris, grid), chk = checkMesh(mesh);
    const vox = voxelize(tris, grid.origin, grid.h, grid.nx, grid.ny, grid.nz);
    let cubes = 0; for (const s of vox) cubes += s;
    console.log("  " + label + ", res " + res + ": tets " + mesh.vol.length + ", volume " + (mesh.volume * 1e9).toFixed(0) + " mm3 against exact " + exact.toFixed(0) + " (" + pct(mesh.volume * 1e9 / exact - 1) +
      "; triangles themselves " + pct(triVol(tris) * 1e9 / exact - 1) + "; cube mesh " + pct(cubes * grid.h ** 3 * 1e9 / exact - 1) + "), quality min " + mesh.quality.min.toFixed(3) + " mean " + mesh.quality.mean.toFixed(3) +
      ", conforming " + (chk.facesSharedByMore === 0 && chk.openEdges === 0 && chk.boundaryMatches) + ", inverted " + chk.inverted);
  }
}

// ---------- 3. plate with a through hole ----------
{
  console.log("PLATE 60 x 40 x 10 mm with a 16 mm through hole");
  const s = new THREE.Shape();
  s.moveTo(0, 0); s.lineTo(60, 0); s.lineTo(60, 40); s.lineTo(0, 40); s.closePath();
  const hole = new THREE.Path(); hole.absarc(30, 20, 8, 0, 2 * Math.PI, true); s.holes.push(hole);
  const tris = soup(new THREE.ExtrudeGeometry(s, { depth: 10, bevelEnabled: false, curveSegments: 48 }));
  const exact = 60 * 40 * 10 - Math.PI * 64 * 10;
  for (const res of [24, 48]) {
    const grid = gridFor(tris, res), mesh = buildTetMesh(tris, grid), chk = checkMesh(mesh);
    // hole kept: no element centre closer to the hole axis than the radius, and the boundary has genus 1
    let closest = Infinity;
    for (let e = 0; e < mesh.vol.length; e++) {
      let x = 0, y = 0; for (let n = 0; n < 4; n++) { x += mesh.nodes[3 * mesh.tets[4 * e + n]] / 4; y += mesh.nodes[3 * mesh.tets[4 * e + n] + 1] / 4; }
      closest = Math.min(closest, Math.hypot(x - 0.03, y - 0.02));
    }
    let rmin = Infinity, rmax = 0;
    const bn = new Set(), be = new Set();
    for (let bq = 0; bq < mesh.bfaces.length; bq++) {
      const fn = Array.from(boundaryFace(mesh, bq).nodes);
      for (let v = 0; v < 3; v++) { bn.add(fn[v]); const p = fn[v], q = fn[(v + 1) % 3]; be.add(p < q ? p + "," + q : q + "," + p); }
    }
    for (const n of bn) { const r = Math.hypot(mesh.nodes[3 * n] - 0.03, mesh.nodes[3 * n + 1] - 0.02); if (r < 0.0085) { rmin = Math.min(rmin, r); rmax = Math.max(rmax, r); } }
    const euler = bn.size - be.size + mesh.bfaces.length;
    console.log("  res " + res + ": " + chk.elements + " elements, " + chk.nodes + " nodes, volume " + (mesh.volume * 1e9).toFixed(1) + " mm3 against " + exact.toFixed(1) + " (" + pct(mesh.volume * 1e9 / exact - 1) + ")");
    console.log("    interior faces (shared by two) " + chk.interiorFaces + ", boundary faces (one) " + chk.boundaryFaces + ", faces shared by more than two " + chk.facesSharedByMore +
      ", open boundary edges " + chk.openEdges + ", inverted elements " + chk.inverted + ", volume from boundary surface " + (chk.volumeFromSurface * 1e9).toFixed(1) + " mm3");
    console.log("    hole: nearest element centre to the axis " + (closest * 1e3).toFixed(2) + " mm, wall nodes at radius " + (rmin * 1e3).toFixed(2) + " to " + (rmax * 1e3).toFixed(2) + " mm (true 8), Euler number of the boundary " + euler +
      " (0 means one through hole), quality min " + mesh.quality.min.toFixed(3) + " mean " + mesh.quality.mean.toFixed(3));
  }
}

// ---------- 4. optimiser smoke test ----------
{
  console.log("OPTIMISER on a 100 x 10 x 40 mm cantilever, 50 % material");
  const tris = soup(new THREE.BoxGeometry(100, 10, 40).translate(50, 0, 0));
  const mesh = buildTetMesh(tris, gridFor(tris, 40));
  const { model } = tetSetup(mesh, tris, [
    { kind: "fixed", tris: trisWhere(tris, (n) => n[0] < -0.9), force: [0, 0, 0] },
    { kind: "load", tris: trisWhere(tris, (n) => n[0] > 0.9), force: [0, 0, -100] },
  ], 2e9, 0.3);
  const o = new TetOptimiser(model), t = Date.now();
  for (let i = 0; i < 25; i++) { const r = o.update(0.5); if (i % 6 === 0) console.log("  pass " + i + " compliance " + r.compliance.toExponential(3) + " change " + r.change.toFixed(3) + " solver passes " + r.iterations + " volume fraction " + o.volume().toFixed(3)); }
  console.log("  " + mesh.vol.length + " elements, " + (Date.now() - t) + " ms. Density on the mid plane (side view):");
  const nx = 50, nz = 20, sum = new Float64Array(nx * nz), cnt = new Float64Array(nx * nz);
  for (let e = 0; e < model.ne; e++) {
    const c = o.centre;
    if (Math.abs(c[3 * e + 1]) > 0.0025) continue;
    const i = Math.min(nx - 1, Math.floor(c[3 * e] / 0.1 * nx)), k = Math.min(nz - 1, Math.floor((c[3 * e + 2] + 0.02) / 0.04 * nz));
    sum[i + nx * k] += o.x[e] * model.vol[e]; cnt[i + nx * k] += model.vol[e];
  }
  let out = "";
  for (let k = nz - 1; k >= 0; k--) { out += "  "; for (let i = 0; i < nx; i++) { const x = cnt[i + nx * k] ? sum[i + nx * k] / cnt[i + nx * k] : 0; out += x > 0.7 ? "#" : x > 0.3 ? "+" : "."; } out += "\n"; }
  console.log(out);
}
