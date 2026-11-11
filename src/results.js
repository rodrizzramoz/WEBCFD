// Turns raw solver output (density and velocity per cell, lattice units)
// into things to look at: slice images, streamlines, surface colours.
import { SOLID, FLUID } from "./domain.js";

const STOPS = [[35, 23, 90], [48, 96, 200], [32, 164, 200], [70, 208, 150], [170, 226, 80], [246, 214, 60], [250, 140, 40], [214, 48, 36]];
export const LUT = new Uint8Array(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = (i / 255) * (STOPS.length - 1), a = Math.min(STOPS.length - 2, Math.floor(t)), f = t - a;
  for (let c = 0; c < 3; c++) LUT[3 * i + c] = Math.round(STOPS[a][c] + (STOPS[a + 1][c] - STOPS[a][c]) * f);
}
export const LUT_CSS = "linear-gradient(to right, " + STOPS.map((c) => "rgb(" + c.join(",") + ")").join(", ") + ")";

const WHITE = -1; // colour marker for smoke wand lines
function lutIndex(v, lo, hi) {
  const t = (v - lo) / (hi - lo);
  return t <= 0 ? 0 : t >= 1 ? 255 : (t * 255) | 0;
}

// Value of the chosen field at cell c, in lattice units.
function value(m, c, field) {
  const o = 4 * c;
  if (field === "pressure") return (m[o] - 1) / 3;
  return Math.sqrt(m[o + 1] * m[o + 1] + m[o + 2] * m[o + 2] + m[o + 3] * m[o + 3]);
}

// Range of speed and pressure over the fluid, plus a blow-up check.
export function fieldStats(grid, m) {
  const fl = grid.flags, n = grid.n;
  let vmax = 0, pmin = Infinity, pmax = -Infinity, bad = false;
  const stride = n > 2e6 ? 3 : 1;
  for (let c = 0; c < n; c += stride) {
    if ((fl[c] & 7) !== FLUID) continue;
    const o = 4 * c, r = m[o];
    if (!(r > 0.4 && r < 2.5)) { bad = true; break; }
    const v2 = m[o + 1] * m[o + 1] + m[o + 2] * m[o + 2] + m[o + 3] * m[o + 3];
    if (v2 > vmax) vmax = v2;
    const p = (r - 1) / 3;
    if (p < pmin) pmin = p;
    if (p > pmax) pmax = p;
  }
  return { vmax: Math.sqrt(vmax), pmin, pmax, bad };
}

// RGBA image of one grid plane. axis 0/1/2 = X/Y/Z, index = cell layer.
export function sliceImage(grid, m, axis, index, field, lo, hi) {
  const { nx, ny, nz, flags } = grid;
  const w = axis === 0 ? ny : nx, h = axis === 2 ? ny : nz;
  const out = new Uint8Array(w * h * 4);
  for (let v = 0; v < h; v++) {
    for (let u = 0; u < w; u++) {
      const c = axis === 0 ? index + nx * (u + ny * v) : axis === 1 ? u + nx * (index + ny * v) : u + nx * (v + ny * index);
      const o = 4 * (u + w * v);
      if ((flags[c] & 7) === SOLID) { out[o + 3] = 0; continue; }
      const l = 3 * lutIndex(value(m, c, field), lo, hi);
      out[o] = LUT[l]; out[o + 1] = LUT[l + 1]; out[o + 2] = LUT[l + 2]; out[o + 3] = 235;
    }
  }
  return { data: out, w, h };
}

// Speed and gauge pressure (lattice units) at the cell containing a world point.
// Returns null outside the grid or inside a wall. "fix" optionally pins one axis to a cell layer.
export function probe(grid, m, point, fixAxis = -1, fixIndex = 0) {
  const { origin, dx, nx, ny, nz, flags } = grid, dims = [nx, ny, nz], ijk = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    ijk[a] = a === fixAxis ? fixIndex : Math.floor((point[a] - origin[a]) / dx);
    if (ijk[a] < 0 || ijk[a] >= dims[a]) return null;
  }
  const c = ijk[0] + nx * (ijk[1] + ny * ijk[2]);
  if ((flags[c] & 7) === SOLID) return null;
  return { cell: c, ijk, speed: value(m, c, "speed"), pressure: value(m, c, "pressure") };
}

// A small ring of streamline seeds (grid coordinates) around a world point: one in the
// middle and the rest on a circle about 2 cells across, square to the local flow.
export function seedCluster(grid, m, point, count = 12) {
  const { origin, dx } = grid;
  const x = (point[0] - origin[0]) / dx, y = (point[1] - origin[1]) / dx, z = (point[2] - origin[2]) / dx;
  const u = [0, 0, 0];
  let d = grid.inVec;
  if (m && sampleVelocity(grid, m, x, y, z, u)) {
    const sp = Math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    if (sp > 1e-6) d = [u[0] / sp, u[1] / sp, u[2] / sp];
  }
  // two unit vectors square to d
  const h = Math.abs(d[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  let e1 = [d[1] * h[2] - d[2] * h[1], d[2] * h[0] - d[0] * h[2], d[0] * h[1] - d[1] * h[0]];
  const l1 = Math.hypot(e1[0], e1[1], e1[2]);
  e1 = e1.map((v) => v / l1);
  const e2 = [d[1] * e1[2] - d[2] * e1[1], d[2] * e1[0] - d[0] * e1[2], d[0] * e1[1] - d[1] * e1[0]];
  const out = [x, y, z], R = 1.2, ring = count - 1;
  for (let s = 0; s < ring; s++) {
    const a = 2 * Math.PI * s / ring, ca = R * Math.cos(a), sa = R * Math.sin(a);
    out.push(x + ca * e1[0] + sa * e2[0], y + ca * e1[1] + sa * e2[1], z + ca * e1[2] + sa * e2[2]);
  }
  return out;
}

function sampleVelocity(grid, m, x, y, z, out) {
  const { nx, ny, nz } = grid;
  const gx = x - 0.5, gy = y - 0.5, gz = z - 0.5;
  const i = Math.floor(gx), j = Math.floor(gy), k = Math.floor(gz);
  if (i < 0 || j < 0 || k < 0 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1) return false;
  const fx = gx - i, fy = gy - j, fz = gz - k;
  let ux = 0, uy = 0, uz = 0;
  for (let d = 0; d < 8; d++) {
    const di = d & 1, dj = (d >> 1) & 1, dk = d >> 2;
    const w = (di ? fx : 1 - fx) * (dj ? fy : 1 - fy) * (dk ? fz : 1 - fz);
    const o = 4 * (i + di + nx * (j + dj + ny * (k + dk)));
    ux += w * m[o + 1]; uy += w * m[o + 2]; uz += w * m[o + 3];
  }
  out[0] = ux; out[1] = uy; out[2] = uz;
  return true;
}

// Seed points (grid coordinates) for streamlines.
export function makeSeeds(grid) {
  const seeds = [];
  if (grid.mode === "external") {
    const { bb, origin, dx } = grid;
    const cy = (bb.min[1] + bb.max[1]) / 2, cz = (bb.min[2] + bb.max[2]) / 2;
    const hy = bb.size[1] * 0.75 + dx, hz = bb.size[2] * 0.75 + dx;
    const NY = 13, NZ = 11;
    for (let b = 0; b < NZ; b++) {
      for (let a = 0; a < NY; a++) {
        const y = cy + hy * (2 * a / (NY - 1) - 1), z = cz + hz * (2 * b / (NZ - 1) - 1);
        seeds.push(2.5, (y - origin[1]) / dx, (z - origin[2]) / dx);
      }
    }
  } else {
    const nbr = grid.inletNbr, { nx, ny } = grid;
    const want = Math.min(140, nbr.length), stepN = nbr.length / want;
    for (let s = 0; s < want; s++) {
      const c = nbr[Math.floor(s * stepN + stepN / 2)];
      seeds.push((c % nx) + 0.5, (((c / nx) | 0) % ny) + 0.5, ((c / (nx * ny)) | 0) + 0.5);
    }
  }
  return seeds;
}

// Integrates streamlines through the velocity field. Returns line-segment
// vertex positions in metres and a colour per vertex. Seeds from index "plainFrom"
// onwards (smoke wand seeds) are drawn near white so they stand out from the rest.
export function streamlines(grid, m, seeds, lo, hi, plainFrom = Infinity) {
  const { origin, dx, nx, flags } = grid;
  const maxSteps = Math.min(1500, Math.ceil((grid.nx + grid.ny + grid.nz) * 1.6));
  const pos = [], col = [];
  const u = [0, 0, 0], u2 = [0, 0, 0];
  const H = 0.8;
  for (let s = 0; s < seeds.length; s += 3) {
    let x = seeds[s], y = seeds[s + 1], z = seeds[s + 2];
    let px = 0, py = 0, pz = 0, pc = 0, have = false;
    for (let it = 0; it < maxSteps; it++) {
      if (!sampleVelocity(grid, m, x, y, z, u)) break;
      const sp = Math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
      if (sp < 1e-6) break;
      const c = (x | 0) + nx * ((y | 0) + grid.ny * (z | 0));
      if ((flags[c] & 7) === SOLID) break;
      const li = s >= plainFrom ? WHITE : 3 * lutIndex(sp, lo, hi);
      const wx = origin[0] + x * dx, wy = origin[1] + y * dx, wz = origin[2] + z * dx;
      if (have) {
        pos.push(px, py, pz, wx, wy, wz);
        if (pc === WHITE) col.push(0.96, 0.97, 1, 0.96, 0.97, 1);
        else col.push(LUT[pc] / 255, LUT[pc + 1] / 255, LUT[pc + 2] / 255, LUT[li] / 255, LUT[li + 1] / 255, LUT[li + 2] / 255);
      }
      px = wx; py = wy; pz = wz; pc = li; have = true;
      // midpoint step
      const mx = x + 0.5 * H * u[0] / sp, my = y + 0.5 * H * u[1] / sp, mz = z + 0.5 * H * u[2] / sp;
      if (!sampleVelocity(grid, m, mx, my, mz, u2)) break;
      const s2 = Math.sqrt(u2[0] * u2[0] + u2[1] * u2[1] + u2[2] * u2[2]);
      if (s2 < 1e-6) break;
      x += H * u2[0] / s2; y += H * u2[1] / s2; z += H * u2[2] / s2;
    }
  }
  return { positions: new Float32Array(pos), colors: new Float32Array(col) };
}

// Pressure colour for each mesh vertex, sampled one cell off the surface.
export function surfaceColors(grid, m, positions, normals, lo, hi) {
  const { origin, dx, nx, ny, nz, flags } = grid;
  const out = new Float32Array(positions.length);
  const at = (x, y, z) => {
    const i = Math.floor((x - origin[0]) / dx), j = Math.floor((y - origin[1]) / dx), k = Math.floor((z - origin[2]) / dx);
    if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return -1;
    const c = i + nx * (j + ny * k);
    return (flags[c] & 7) === SOLID ? -1 : c;
  };
  for (let v = 0; v < positions.length; v += 3) {
    const x = positions[v], y = positions[v + 1], z = positions[v + 2];
    const a = normals[v], b = normals[v + 1], c = normals[v + 2];
    let cell = -1;
    for (const d of [1.0, -1.0, 1.9, -1.9]) {
      cell = at(x + a * d * dx, y + b * d * dx, z + c * d * dx);
      if (cell >= 0) break;
    }
    if (cell < 0) { out[v] = 0.55; out[v + 1] = 0.58; out[v + 2] = 0.62; continue; }
    const li = 3 * lutIndex((m[4 * cell] - 1) / 3, lo, hi);
    out[v] = LUT[li] / 255; out[v + 1] = LUT[li + 1] / 255; out[v + 2] = LUT[li + 2] / 255;
  }
  return out;
}

// Inlet-to-outlet pressure difference and the flow actually passing the
// inlet and the outlet, in lattice units. "m" values are mass flow (density weighted).
export function internalMetrics(grid, m) {
  let pin = 0, pout = 0, qin = 0, qout = 0, min = 0, mout = 0, tin = 0, tout = 0;
  const no = grid.outNormal, ni = grid.inVec;
  for (const c of grid.inletNbr) {
    const o = 4 * c, f = m[o + 1] * ni[0] + m[o + 2] * ni[1] + m[o + 3] * ni[2];
    const p = (m[o] - 1) / 3, ke = 0.5 * m[o] * (m[o + 1] * m[o + 1] + m[o + 2] * m[o + 2] + m[o + 3] * m[o + 3]);
    pin += p; qin += f; min += f * m[o]; tin += f * (p + ke);
  }
  for (const c of grid.outletNbr) {
    const o = 4 * c, f = -(m[o + 1] * no[0] + m[o + 2] * no[1] + m[o + 3] * no[2]);
    const p = (m[o] - 1) / 3, ke = 0.5 * m[o] * (m[o + 1] * m[o + 1] + m[o + 2] * m[o + 2] + m[o + 3] * m[o + 3]);
    pout += p; qout += f; mout += f * m[o]; tout += f * (p + ke);
  }
  // Total pressure (static plus kinetic), averaged by flow, is what is actually lost.
  // Static drop alone also counts kinetic energy that is still in the jet at the outlet.
  const loss = qin > 1e-12 && qout > 1e-12 ? tin / qin - tout / qout : 0;
  return { dp: pin / grid.inletNbr.length - pout / grid.outletNbr.length, loss, qin, qout, balance: min !== 0 ? mout / min : 0 };
}
