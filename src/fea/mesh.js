// Geometry helpers for the structure page: surface picking regions, mapping
// picked faces onto voxels, result meshes, marching cubes and STL export.
import { edgeTable, triTable } from "three/addons/objects/MarchingCubes.js";
import { LUT } from "../results.js";
import { TET_FACE } from "./tetmesh.js";

// Triangle adjacency across shared edges, found by welding identical vertices.
export function buildAdjacency(pos) {
  const nt = pos.length / 9, ids = new Int32Array(nt * 3), map = new Map();
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < pos.length; i++) { if (pos[i] < min) min = pos[i]; if (pos[i] > max) max = pos[i]; }
  const q = 1e5 / Math.max(max - min, 1e-12);
  for (let v = 0; v < nt * 3; v++) {
    const key = Math.round((pos[3 * v] - min) * q) + "," + Math.round((pos[3 * v + 1] - min) * q) + "," + Math.round((pos[3 * v + 2] - min) * q);
    let id = map.get(key);
    if (id === undefined) { id = map.size; map.set(key, id); }
    ids[v] = id;
  }
  const edges = new Map(), nbr = new Int32Array(nt * 3).fill(-1);
  for (let t = 0; t < nt; t++) {
    for (let e = 0; e < 3; e++) {
      const a = ids[3 * t + e], b = ids[3 * t + (e + 1) % 3];
      const key = a < b ? a * 4294967296 + b : b * 4294967296 + a;
      const other = edges.get(key);
      if (other === undefined) edges.set(key, t * 3 + e);
      else { nbr[t * 3 + e] = (other / 3) | 0; nbr[other] = t; }
    }
  }
  const normals = new Float32Array(nt * 3);
  for (let t = 0; t < nt; t++) {
    const o = 9 * t;
    const ux = pos[o + 3] - pos[o], uy = pos[o + 4] - pos[o + 1], uz = pos[o + 5] - pos[o + 2];
    const vx = pos[o + 6] - pos[o], vy = pos[o + 7] - pos[o + 1], vz = pos[o + 8] - pos[o + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    normals[3 * t] = nx / l; normals[3 * t + 1] = ny / l; normals[3 * t + 2] = nz / l;
  }
  return { nbr, normals };
}

// Grows a face region from a seed triangle. "flat" keeps to one plane;
// "curved" follows a smoothly curving surface such as a bore or a fillet.
export function growRegion(adj, seed, mode) {
  const { nbr, normals } = adj, seen = new Set([seed]), stack = [seed];
  const stepCos = Math.cos((mode === "curved" ? 22 : 4) * Math.PI / 180);
  const seedCos = mode === "curved" ? -2 : Math.cos(4 * Math.PI / 180);
  const sx = normals[3 * seed], sy = normals[3 * seed + 1], sz = normals[3 * seed + 2];
  while (stack.length) {
    const t = stack.pop(), tx = normals[3 * t], ty = normals[3 * t + 1], tz = normals[3 * t + 2];
    for (let e = 0; e < 3; e++) {
      const o = nbr[3 * t + e];
      if (o < 0 || seen.has(o)) continue;
      const ox = normals[3 * o], oy = normals[3 * o + 1], oz = normals[3 * o + 2];
      if (ox * tx + oy * ty + oz * tz < stepCos) continue;
      if (ox * sx + oy * sy + oz * sz < seedCos) continue;
      seen.add(o); stack.push(o);
    }
  }
  return Int32Array.from(seen);
}

// Solid cells lying just under a set of surface triangles. Returns a map from
// cell index to the cell face that points out of the surface (axis * 2 + 1 if positive).
export function cellsUnderTriangles(grid, solid, pos, adj, tris) {
  const { nx, ny, nz, h, origin } = grid, out = new Map();
  const isSolid = (i, j, k) => i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz && solid[i + nx * (j + ny * k)] === 1;
  for (const t of tris) {
    const o = 9 * t, n = [adj.normals[3 * t], adj.normals[3 * t + 1], adj.normals[3 * t + 2]];
    const ax = Math.abs(n[0]) >= Math.abs(n[1]) && Math.abs(n[0]) >= Math.abs(n[2]) ? 0 : Math.abs(n[1]) >= Math.abs(n[2]) ? 1 : 2;
    const face = ax * 2 + (n[ax] > 0 ? 1 : 0);
    const a = [pos[o], pos[o + 1], pos[o + 2]], b = [pos[o + 3], pos[o + 4], pos[o + 5]], c = [pos[o + 6], pos[o + 7], pos[o + 8]];
    const edge = Math.max(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]), Math.hypot(c[0] - a[0], c[1] - a[1], c[2] - a[2]), Math.hypot(c[0] - b[0], c[1] - b[1], c[2] - b[2]));
    const div = Math.min(400, Math.max(1, Math.ceil(edge / (0.45 * h))));
    for (let p = 0; p <= div; p++) {
      for (let q = 0; q <= div - p; q++) {
        const w1 = (p + 0.02) / (div + 0.06), w2 = (q + 0.02) / (div + 0.06), w0 = 1 - w1 - w2;
        const x = w0 * a[0] + w1 * b[0] + w2 * c[0], y = w0 * a[1] + w1 * b[1] + w2 * c[1], z = w0 * a[2] + w1 * b[2] + w2 * c[2];
        const i = Math.floor((x - n[0] * 0.5 * h - origin[0]) / h), j = Math.floor((y - n[1] * 0.5 * h - origin[1]) / h), k = Math.floor((z - n[2] * 0.5 * h - origin[2]) / h);
        if (isSolid(i, j, k)) { out.set(i + nx * (j + ny * k), face); continue; }
        let best = -1, bd = 2.3 * h * h;
        for (let dk = -1; dk <= 1; dk++) for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
          if (!isSolid(i + di, j + dj, k + dk)) continue;
          const cx = origin[0] + (i + di + 0.5) * h - x, cy = origin[1] + (j + dj + 0.5) * h - y, cz = origin[2] + (k + dk + 0.5) * h - z;
          const d = cx * cx + cy * cy + cz * cz;
          if (d < bd) { bd = d; best = i + di + nx * (j + dj + ny * (k + dk)); }
        }
        if (best >= 0 && !out.has(best)) out.set(best, face);
      }
    }
  }
  return out;
}

// The four nodes of a cell on one of its faces (face = axis * 2 + 1 if on the positive side).
export function cellNodes(grid, c, face) {
  const { nx, ny } = grid, NX = nx + 1, NY = ny + 1;
  const i = c % nx, j = ((c / nx) | 0) % ny, k = (c / (nx * ny)) | 0, out = [];
  const axis = face >> 1, side = face & 1;
  for (let d = 0; d < 8; d++) {
    const off = [d & 1, (d >> 1) & 1, d >> 2];
    if (off[axis] === side) out.push(i + off[0] + NX * (j + off[1] + NY * (k + off[2])));
  }
  return out;
}

const FACES = [ // neighbour offset, then the four corner offsets (anticlockwise seen from outside)
  [[-1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
  [[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
  [[0, -1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
  [[0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
  [[0, 0, -1], [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
  [[0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
];

// Outer skin of the voxel model, optionally deformed and coloured by a per-element value.
// cells: grid cell index of each element; show: per element 0/1; u: nodal displacements or null.
// rgb (optional): Float32Array with 3 numbers per element; a negative red means no override.
export function voxelSkin(grid, cells, show, u, deform, values, lo, hi, rgb) {
  const { nx, ny, nz, h, origin } = grid, NX = nx + 1, NY = ny + 1;
  const elemOf = new Int32Array(nx * ny * nz).fill(-1);
  for (let e = 0; e < cells.length; e++) if (show[e]) elemOf[cells[e]] = e;
  const pos = [], col = [];
  const corner = (i, j, k) => {
    const n = i + NX * (j + NY * k);
    const dx = u ? u[3 * n] * deform : 0, dy = u ? u[3 * n + 1] * deform : 0, dz = u ? u[3 * n + 2] * deform : 0;
    pos.push(origin[0] + i * h + dx, origin[1] + j * h + dy, origin[2] + k * h + dz);
  };
  for (let e = 0; e < cells.length; e++) {
    if (!show[e]) continue;
    const c = cells[e], i = c % nx, j = ((c / nx) | 0) % ny, k = (c / (nx * ny)) | 0;
    let r = 0.72, g = 0.76, b = 0.8;
    if (values) {
      const t = (values[e] - lo) / (hi - lo || 1), l = 3 * (t <= 0 ? 0 : t >= 1 ? 255 : (t * 255) | 0);
      r = LUT[l] / 255; g = LUT[l + 1] / 255; b = LUT[l + 2] / 255;
    }
    if (rgb && rgb[3 * e] >= 0) { r = rgb[3 * e]; g = rgb[3 * e + 1]; b = rgb[3 * e + 2]; }
    for (const f of FACES) {
      const a = i + f[0][0], bb = j + f[0][1], cc = k + f[0][2];
      if (a >= 0 && bb >= 0 && cc >= 0 && a < nx && bb < ny && cc < nz && elemOf[a + nx * (bb + ny * cc)] >= 0) continue;
      for (const q of [1, 2, 3, 1, 3, 4]) { corner(i + f[q][0], j + f[q][1], k + f[q][2]); col.push(r, g, b); }
    }
  }
  return { positions: new Float32Array(pos), colors: new Float32Array(col) };
}

// Outer skin of a tetrahedral mesh: every element face with no shown element
// behind it. show: per element 0/1, or null for all. u: nodal displacements or
// null. values: one per element, or one per node when nodal is true.
// marks (optional): Map from element * 4 + face to an [r, g, b] colour.
export function tetSkin(mesh, show, u, deform, values, lo, hi, marks, nodal) {
  const { nodes, tets, nbr } = mesh, ne = tets.length / 4;
  let count = 0;
  for (let id = 0; id < 4 * ne; id++) if ((!show || show[id >> 2]) && (nbr[id] < 0 || (show && !show[nbr[id]]))) count++;
  const pos = new Float32Array(9 * count), col = new Float32Array(9 * count);
  const shade = (v, out, o) => {
    const t = (v - lo) / (hi - lo || 1), l = 3 * (t <= 0 ? 0 : t >= 1 ? 255 : (t * 255) | 0);
    out[o] = LUT[l] / 255; out[o + 1] = LUT[l + 1] / 255; out[o + 2] = LUT[l + 2] / 255;
  };
  let o = 0;
  for (let id = 0; id < 4 * ne; id++) {
    const e = id >> 2;
    if ((show && !show[e]) || (nbr[id] >= 0 && (!show || show[nbr[id]]))) continue;
    const f = TET_FACE[id & 3], mark = marks && marks.get(id);
    for (let v = 0; v < 3; v++, o += 3) {
      const n = tets[4 * e + f[v]];
      pos[o] = nodes[3 * n] + (u ? u[3 * n] * deform : 0);
      pos[o + 1] = nodes[3 * n + 1] + (u ? u[3 * n + 1] * deform : 0);
      pos[o + 2] = nodes[3 * n + 2] + (u ? u[3 * n + 2] * deform : 0);
      if (mark) { col[o] = mark[0]; col[o + 1] = mark[1]; col[o + 2] = mark[2]; }
      else if (values) shade(values[nodal ? n : e], col, o);
      else { col[o] = 0.72; col[o + 1] = 0.76; col[o + 2] = 0.8; }
    }
  }
  return { positions: pos, colors: col, triangles: true };
}

// Fraction of every grid cell filled by a design on the tetrahedral mesh
// (x = density per element, or null for the solid part). Feeds smoothSurface.
export function tetDensity(mesh, x) {
  const { nx, ny, nz, h } = mesh.grid, dens = new Float32Array(nx * ny * nz), h3 = h * h * h;
  for (let e = 0; e < mesh.vol.length; e++) dens[mesh.cell[e]] += (x ? x[e] : 1) * mesh.vol[e] / h3;
  for (let c = 0; c < dens.length; c++) if (dens[c] > 1) dens[c] = 1;
  return dens;
}

// Smooth surface through a density field (one value per grid cell) at level 0.5.
const MC_CORNER = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const MC_EDGE = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
export function smoothSurface(grid, density) {
  const { nx, ny, nz, h, origin } = grid;
  // sample grid = cell centres, padded by two layers of empty space
  const P = 2, sx = nx + 2 * P, sy = ny + 2 * P, sz = nz + 2 * P;
  let f = new Float32Array(sx * sy * sz);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) f[i + P + sx * (j + P + sy * (k + P))] = density[i + nx * (j + ny * k)];
  // light blur to round off the voxel steps
  const g = new Float32Array(f.length);
  for (let k = 1; k < sz - 1; k++) for (let j = 1; j < sy - 1; j++) for (let i = 1; i < sx - 1; i++) {
    const c = i + sx * (j + sy * k);
    g[c] = (2 * f[c] + f[c - 1] + f[c + 1] + f[c - sx] + f[c + sx] + f[c - sx * sy] + f[c + sx * sy]) / 8;
  }
  f = g;
  const pos = [], val = new Float32Array(8), vert = new Float32Array(36);
  for (let k = 0; k < sz - 1; k++) for (let j = 0; j < sy - 1; j++) for (let i = 0; i < sx - 1; i++) {
    let idx = 0;
    for (let n = 0; n < 8; n++) {
      val[n] = f[i + MC_CORNER[n][0] + sx * (j + MC_CORNER[n][1] + sy * (k + MC_CORNER[n][2]))];
      if (val[n] < 0.5) idx |= 1 << n;
    }
    const bits = edgeTable[idx];
    if (!bits) continue;
    for (let e = 0; e < 12; e++) {
      if (!(bits & (1 << e))) continue;
      const a = MC_EDGE[e][0], b = MC_EDGE[e][1], t = (0.5 - val[a]) / (val[b] - val[a] || 1e-9);
      for (let d = 0; d < 3; d++) vert[3 * e + d] = MC_CORNER[a][d] + t * (MC_CORNER[b][d] - MC_CORNER[a][d]);
    }
    for (let q = idx * 16; triTable[q] !== -1; q += 3) {
      for (const e of [triTable[q], triTable[q + 1], triTable[q + 2]]) {
        pos.push(origin[0] + (i - P + 0.5 + vert[3 * e]) * h, origin[1] + (j - P + 0.5 + vert[3 * e + 1]) * h, origin[2] + (k - P + 0.5 + vert[3 * e + 2]) * h);
      }
    }
  }
  const out = new Float32Array(pos);
  // make the triangles face outward (positive enclosed volume)
  let vol = 0;
  for (let o = 0; o < out.length; o += 9) {
    vol += out[o] * (out[o + 4] * out[o + 8] - out[o + 5] * out[o + 7]) - out[o + 1] * (out[o + 3] * out[o + 8] - out[o + 5] * out[o + 6]) + out[o + 2] * (out[o + 3] * out[o + 7] - out[o + 4] * out[o + 6]);
  }
  if (vol < 0) for (let o = 0; o < out.length; o += 9) for (let d = 0; d < 3; d++) { const tmp = out[o + 3 + d]; out[o + 3 + d] = out[o + 6 + d]; out[o + 6 + d] = tmp; }
  return { positions: out, volume: Math.abs(vol) / 6 };
}

// Binary STL from triangle positions (metres), written in the chosen file unit.
export function toSTL(positions, unitsPerMetre) {
  const nt = positions.length / 9, buf = new ArrayBuffer(84 + 50 * nt), dv = new DataView(buf);
  dv.setUint32(80, nt, true);
  for (let t = 0; t < nt; t++) {
    const o = 9 * t, base = 84 + 50 * t;
    const ux = positions[o + 3] - positions[o], uy = positions[o + 4] - positions[o + 1], uz = positions[o + 5] - positions[o + 2];
    const vx = positions[o + 6] - positions[o], vy = positions[o + 7] - positions[o + 1], vz = positions[o + 8] - positions[o + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    dv.setFloat32(base, nx / l, true); dv.setFloat32(base + 4, ny / l, true); dv.setFloat32(base + 8, nz / l, true);
    for (let v = 0; v < 9; v++) dv.setFloat32(base + 12 + 4 * v, positions[o + v] * unitsPerMetre, true);
  }
  return buf;
}
