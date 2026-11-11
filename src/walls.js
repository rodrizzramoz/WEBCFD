// True-surface wall data for the flow solver.
//
// The solver grid is made of cubes, but a wall does not have to sit on a cube
// face. For every fluid cell next to the part, this works out how far along
// each of the 18 lattice links the real triangle surface lies (a fraction
// between 0 and 1). The solver then reflects flow off the wall at that true
// position (interpolated bounce-back, after Bouzidi, Firdaouss and Lallemand)
// instead of at the half-way point between cell centres.
import { FLUID, SOLID } from "./domain.js";

// Lattice directions 1 to 18, in the same order as the shader.
const C = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
  [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
  [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]];

// Squared distance from point p to triangle a, b, c (Ericson, Real-Time Collision Detection).
function distSq(px, py, pz, t, o) {
  const ax = t[o], ay = t[o + 1], az = t[o + 2];
  const abx = t[o + 3] - ax, aby = t[o + 4] - ay, abz = t[o + 5] - az;
  const acx = t[o + 6] - ax, acy = t[o + 7] - ay, acz = t[o + 8] - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
  let qx, qy, qz;
  if (d1 <= 0 && d2 <= 0) { qx = ax; qy = ay; qz = az; }
  else {
    const bpx = apx - abx, bpy = apy - aby, bpz = apz - abz;
    const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
    if (d3 >= 0 && d4 <= d3) { qx = ax + abx; qy = ay + aby; qz = az + abz; }
    else {
      const vc = d1 * d4 - d3 * d2;
      if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); qx = ax + v * abx; qy = ay + v * aby; qz = az + v * abz; }
      else {
        const cpx = apx - acx, cpy = apy - acy, cpz = apz - acz;
        const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
        if (d6 >= 0 && d5 <= d6) { qx = ax + acx; qy = ay + acy; qz = az + acz; }
        else {
          const vb = d5 * d2 - d1 * d6;
          if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); qx = ax + w * acx; qy = ay + w * acy; qz = az + w * acz; }
          else {
            const va = d3 * d6 - d5 * d4;
            if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
              const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
              qx = ax + abx + w * (acx - abx); qy = ay + aby + w * (acy - aby); qz = az + abz + w * (acz - abz);
            } else {
              const den = 1 / (va + vb + vc), v = vb * den, w = vc * den;
              qx = ax + abx * v + acx * w; qy = ay + aby * v + acy * w; qz = az + abz * v + acz * w;
            }
          }
        }
      }
    }
  }
  const dx = px - qx, dy = py - qy, dz = pz - qz;
  return dx * dx + dy * dy + dz * dz;
}

// Adds wall data to a grid built by buildDomain. geomSolid(c) says whether
// cell c is inside the part's material according to the triangles themselves.
// Stores a table index in bits 6 and up of each near-wall fluid cell's flag,
// and returns the table: 5 words per entry, one byte per link (0 = use the half-way rule).
export function addWallDistances(grid, tris, geomSolid) {
  const { nx, ny, nz, n, dx, origin, flags } = grid;
  const off = C.map((c) => c[0] + nx * (c[1] + ny * c[2]));
  const need = new Uint8Array(n), cells = [];
  for (let k = 1; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
    const c = i + nx * (j + ny * k);
    if ((flags[c] & 7) !== FLUID) continue;
    let near = false;
    for (let q = 0; q < 18; q++) {
      const s = c - off[q];
      if ((flags[s] & 7) === SOLID && geomSolid(s)) { need[s] = 1; near = true; }
    }
    if (near) { need[c] = 1; cells.push(c); }
  }
  // unsigned distance to the surface, only where needed
  const phi = new Float32Array(n).fill(Infinity);
  const R = 2;
  for (let t = 0; t < tris.length; t += 9) {
    const lo = [0, 0, 0], hi = [0, 0, 0], dims = [nx, ny, nz];
    for (let a = 0; a < 3; a++) {
      const mn = Math.min(tris[t + a], tris[t + 3 + a], tris[t + 6 + a]), mx = Math.max(tris[t + a], tris[t + 3 + a], tris[t + 6 + a]);
      lo[a] = Math.max(0, Math.floor((mn - origin[a]) / dx) - R);
      hi[a] = Math.min(dims[a] - 1, Math.floor((mx - origin[a]) / dx) + R);
    }
    for (let k = lo[2]; k <= hi[2]; k++) for (let j = lo[1]; j <= hi[1]; j++) {
      const base = nx * (j + ny * k);
      const py = origin[1] + (j + 0.5) * dx, pz = origin[2] + (k + 0.5) * dx;
      for (let i = lo[0]; i <= hi[0]; i++) {
        if (!need[base + i]) continue;
        const d = distSq(origin[0] + (i + 0.5) * dx, py, pz, tris, t);
        if (d < phi[base + i]) phi[base + i] = d;
      }
    }
  }
  const table = new Uint32Array(Math.max(1, cells.length) * 5);
  let used = 0;
  for (let e = 0; e < cells.length; e++) {
    const c = cells[e], pf = Math.sqrt(phi[c]);
    let any = false;
    for (let q = 0; q < 18; q++) {
      const s = c - off[q];
      if ((flags[s] & 7) !== SOLID || !geomSolid(s)) continue;
      const ps = Math.sqrt(phi[s]);
      if (!isFinite(pf) || !isFinite(ps) || pf + ps <= 0) continue;
      const frac = Math.min(1, Math.max(0.03, pf / (pf + ps)));
      table[5 * e + (q >> 2)] |= Math.max(1, Math.round(frac * 255)) << (8 * (q & 3));
      any = true;
    }
    if (any) used++;
    flags[c] = (flags[c] & 63) | ((e + 1) << 6);
  }
  grid.wallQ = table;
  grid.wallCells = used;
  return table;
}
