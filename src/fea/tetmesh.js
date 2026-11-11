// Body-fitted tetrahedral mesher for the structure page.
//
// Method (no external library):
//   1. A regular grid of nodes is laid over the part. Each node gets a signed
//      distance to the triangle surface: exact point-to-triangle distance in a
//      narrow band, sign from a ray-cast winding test at the node itself.
//   2. Nodes closer to the surface than a fraction of the cell size are moved
//      onto the surface (their distance becomes zero). This stops the cut in
//      step 4 from making sliver elements.
//   3. Every grid cube is split into 6 tetrahedra (Kuhn subdivision, which
//      matches across neighbouring cubes).
//   4. Tetrahedra fully inside are kept. Tetrahedra crossing the surface are
//      cut where their edges cross the surface (marching tetrahedra style),
//      and the inside part is split into 1, 2 or 3 tetrahedra. Cut nodes are
//      shared through a table keyed by the grid edge, and quadrilateral faces
//      are always split from their lowest numbered corner, so neighbouring
//      elements always agree and the mesh is conforming.
//   5. Cut nodes sit exactly where their grid edge crosses the true triangles
//      (found by bisection), and grid edges that run along a face of the part
//      get a cut node where that face ends at a sharp edge.
//
// Limits: sharp edges and corners of the part that do not fall on grid nodes
// or grid edges come out slightly bevelled (by a fraction of one cell), and
// walls thinner than about one cell can be lost.
//
// Units are whatever the triangles use (the page works in metres).

// Closest point on triangle a, b, c to point p (Ericson, Real-Time Collision
// Detection; same method as src/walls.js). Returns the squared distance and
// leaves the closest point in CP.
export const CP = new Float64Array(3);
export function closestOnTriangle(px, py, pz, t, o) {
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
  CP[0] = qx; CP[1] = qy; CP[2] = qz;
  const dx = px - qx, dy = py - qy, dz = pz - qz;
  return dx * dx + dy * dy + dz * dz;
}

// Winding test along x rays, one ray per (y, z) row of sample points. The
// same rules as voxelize in src/domain.js, but the samples sit where the
// caller says (grid nodes here) and not at cell centres.
// Returns an inside flag for every node of the grid.
const JY = 0.00137, JZ = 0.00071; // tiny offset so rays never run exactly along a mesh edge
function nodeSigns(tris, origin, h, NX, NY, NZ) {
  const inside = new Uint8Array(NX * NY * NZ), cols = new Array(NY * NZ);
  const oy = origin[1], oz = origin[2];
  for (let t = 0; t < tris.length; t += 9) {
    const ax = tris[t], ay = tris[t + 1], az = tris[t + 2];
    const bx = tris[t + 3], by = tris[t + 4], bz = tris[t + 5];
    const cx = tris[t + 6], cy = tris[t + 7], cz = tris[t + 8];
    const d = (by - ay) * (cz - az) - (cy - ay) * (bz - az);
    if (Math.abs(d) < 1e-30) continue;
    const sign = d < 0 ? 1 : -1;
    const j0 = Math.max(0, Math.ceil((Math.min(ay, by, cy) - oy) / h - JY));
    const j1 = Math.min(NY - 1, Math.floor((Math.max(ay, by, cy) - oy) / h - JY));
    const k0 = Math.max(0, Math.ceil((Math.min(az, bz, cz) - oz) / h - JZ));
    const k1 = Math.min(NZ - 1, Math.floor((Math.max(az, bz, cz) - oz) / h - JZ));
    for (let k = k0; k <= k1; k++) {
      const pz = oz + (k + JZ) * h;
      for (let j = j0; j <= j1; j++) {
        const py = oy + (j + JY) * h;
        const w1 = ((py - ay) * (cz - az) - (cy - ay) * (pz - az)) / d;
        const w2 = ((by - ay) * (pz - az) - (py - ay) * (bz - az)) / d;
        if (w1 < 0 || w2 < 0 || w1 + w2 > 1) continue;
        const c = j + k * NY;
        (cols[c] || (cols[c] = [])).push(ax + w1 * (bx - ax) + w2 * (cx - ax), sign);
      }
    }
  }
  for (let c = 0; c < cols.length; c++) {
    const hits = cols[c];
    if (!hits) continue;
    const n = hits.length / 2;
    const order = Array.from({ length: n }, (_, i) => i).sort((p, q) => hits[2 * p] - hits[2 * q]);
    // A closed surface is entered as often as it is left. If not, the mesh has
    // a gap on this ray, so stop filling after the last crossing.
    let total = 0;
    for (let i = 0; i < n; i++) total += hits[2 * i + 1];
    let p = 0, wind = 0;
    const base = c * NX;
    for (let i = 0; i < NX; i++) {
      const x = origin[0] + i * h;
      while (p < n && hits[2 * order[p]] < x) { wind += hits[2 * order[p] + 1]; p++; }
      if (p === n && total !== 0) break;
      if (wind !== 0) inside[base + i] = 1;
    }
  }
  return inside;
}

// Six times the signed volume of tetrahedron a, b, c, d (node numbers into P).
function vol6(P, a, b, c, d) {
  const ax = P[3 * a], ay = P[3 * a + 1], az = P[3 * a + 2];
  const ux = P[3 * b] - ax, uy = P[3 * b + 1] - ay, uz = P[3 * b + 2] - az;
  const vx = P[3 * c] - ax, vy = P[3 * c + 1] - ay, vz = P[3 * c + 2] - az;
  const wx = P[3 * d] - ax, wy = P[3 * d + 1] - ay, wz = P[3 * d + 2] - az;
  return ux * (vy * wz - vz * wy) - uy * (vx * wz - vz * wx) + uz * (vx * wy - vy * wx);
}

// Shape quality of a tetrahedron: 1 for a regular one, 0 for a flat one.
export function tetQuality(P, a, b, c, d) {
  const v = vol6(P, a, b, c, d) / 6;
  if (v <= 0) return 0;
  const n = [a, b, c, d];
  let s = 0;
  for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
    const x = P[3 * n[i]] - P[3 * n[j]], y = P[3 * n[i] + 1] - P[3 * n[j] + 1], z = P[3 * n[i] + 2] - P[3 * n[j] + 2];
    s += x * x + y * y + z * z;
  }
  return 12 * Math.pow(3 * v, 2 / 3) / s;
}

// The 6 Kuhn tetrahedra of a cube, as corner numbers (bit 0 = x, 1 = y, 2 = z),
// all positively oriented.
const KUHN = (() => {
  const out = [];
  for (const p of [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]) {
    const c = [0, 1 << p[0], (1 << p[0]) | (1 << p[1]), 7];
    const even = (p[0] === 0 && p[1] === 1) || (p[0] === 1 && p[1] === 2) || (p[0] === 2 && p[1] === 0);
    out.push(even ? c : [c[0], c[1], c[3], c[2]]);
  }
  return out;
})();

// Faces of a positively oriented tetrahedron, face f opposite node f, wound so
// the normal points out of the element.
export const TET_FACE = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

// Rotation table for splitting a prism so the lowest numbered node comes first
// (Dompierre, Labbe, Vallet and Camarero, 1999).
const PRISM_ROT = [[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3], [2, 0, 1, 5, 3, 4], [3, 5, 4, 0, 2, 1], [4, 3, 5, 1, 0, 2], [5, 4, 3, 2, 1, 0]];

// tris: 9 numbers per triangle. grid: {nx, ny, nz, h, origin} in the same
// units, covering the part (cells, as for the cube mesh).
// options: snap = how close to the surface, as a fraction of the cell size, a
// node must be to get moved onto it (default 0.25).
export function buildTetMesh(tris, grid, options = {}) {
  const { nx, ny, nz, h, origin } = grid;
  const NX = nx + 1, NY = ny + 1, NZ = nz + 1, nn = NX * NY * NZ;
  const snap = (options.snap ?? 0.25) * h;

  // ---- distance from every grid node to the surface, in a narrow band, and
  // for each node the list of triangles within one cell of it
  const inside = nodeSigns(tris, origin, h, NX, NY, NZ);
  const d2 = new Float64Array(nn).fill(Infinity), near = new Int32Array(nn).fill(-1);
  const R = 2, dims = [NX, NY, NZ], lo = [0, 0, 0], hi = [0, 0, 0];
  const reach2 = 1.05 * 1.05 * h * h, pairNode = [], pairTri = [], listStart = new Int32Array(nn + 1);
  let signedVol = 0;
  for (let t = 0; t < tris.length; t += 9) {
    signedVol += tris[t] * (tris[t + 4] * tris[t + 8] - tris[t + 5] * tris[t + 7]) - tris[t + 1] * (tris[t + 3] * tris[t + 8] - tris[t + 5] * tris[t + 6]) + tris[t + 2] * (tris[t + 3] * tris[t + 7] - tris[t + 4] * tris[t + 6]);
    for (let a = 0; a < 3; a++) {
      const mn = Math.min(tris[t + a], tris[t + 3 + a], tris[t + 6 + a]), mx = Math.max(tris[t + a], tris[t + 3 + a], tris[t + 6 + a]);
      lo[a] = Math.max(0, Math.floor((mn - origin[a]) / h) - R);
      hi[a] = Math.min(dims[a] - 1, Math.ceil((mx - origin[a]) / h) + R);
    }
    for (let k = lo[2]; k <= hi[2]; k++) for (let j = lo[1]; j <= hi[1]; j++) {
      const base = NX * (j + NY * k), py = origin[1] + j * h, pz = origin[2] + k * h;
      for (let i = lo[0]; i <= hi[0]; i++) {
        const d = closestOnTriangle(origin[0] + i * h, py, pz, tris, t);
        if (d < d2[base + i]) { d2[base + i] = d; near[base + i] = t; }
        if (d < reach2) { pairNode.push(base + i); pairTri.push(t); listStart[base + i + 1]++; }
      }
    }
  }
  for (let n = 0; n < nn; n++) listStart[n + 1] += listStart[n];
  const listTri = new Int32Array(pairTri.length);
  {
    const fill = listStart.slice(0, nn);
    for (let q = 0; q < pairNode.length; q++) listTri[fill[pairNode[q]]++] = pairTri[q];
  }
  // Signed distance (negative inside) at a point close to grid node n, from the
  // triangles listed for that node. The sign comes from the side of the nearest
  // triangle the point is on; among equally near triangles (the point is nearest
  // to an edge) the one seen most face-on decides.
  const flip = signedVol < 0 ? -1 : 1;
  const signedDistance = (px, py, pz, n) => {
    let best = Infinity, side = 0, lim = Infinity;
    for (let q = listStart[n]; q < listStart[n + 1]; q++) {
      const t = listTri[q], d = closestOnTriangle(px, py, pz, tris, t);
      if (d > lim) continue;
      const ux = tris[t + 3] - tris[t], uy = tris[t + 4] - tris[t + 1], uz = tris[t + 5] - tris[t + 2];
      const vx = tris[t + 6] - tris[t], vy = tris[t + 7] - tris[t + 1], vz = tris[t + 8] - tris[t + 2];
      const ax = uy * vz - uz * vy, ay = uz * vx - ux * vz, az = ux * vy - uy * vx, al = Math.hypot(ax, ay, az);
      if (!(al > 0)) continue;
      const s = ((px - CP[0]) * ax + (py - CP[1]) * ay + (pz - CP[2]) * az) / al;
      if (d < best - 1e-9 * h * h - 1e-6 * best) { best = d; side = s; }
      else if (Math.abs(s) > Math.abs(side)) { side = s; if (d < best) best = d; }
      lim = best + 1e-9 * h * h + 1e-6 * best;
    }
    if (best === Infinity) return NaN;
    return (side * flip < 0 ? -1 : 1) * Math.sqrt(best);
  };

  // node positions (grid nodes first, cut nodes are appended later) and distances
  const P = [];
  const phi = new Float64Array(nn);
  let snapped = 0;
  for (let k = 0, n = 0; k < NZ; k++) for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++, n++) {
    const x = origin[0] + i * h, y = origin[1] + j * h, z = origin[2] + k * h;
    const d = Math.sqrt(d2[n]);
    if (d < snap) {
      closestOnTriangle(x, y, z, tris, near[n]);
      P.push(CP[0], CP[1], CP[2]); phi[n] = 0; snapped++;
    } else {
      P.push(x, y, z);
      phi[n] = (inside[n] ? -1 : 1) * (isFinite(d) ? d : 3 * h);
    }
  }

  // ---- cut nodes: where a grid edge from a node inside or on the surface (a)
  // to a node outside (b) leaves the part. Shared through a table keyed by the edge.
  //   a inside: the true surface crossing, found by bisection on the signed
  //     distance, starting from the straight-line estimate.
  //   a on the surface: normally the part ends at a itself. But next to a sharp
  //     edge of the part the surface (or the inside) carries on along the grid
  //     edge for a while; then a cut node is placed where it really ends. Without
  //     this, sharp edges would be cut back by up to a whole cell.
  const cuts = new Map(), eps = 0.002 * h;
  let extended = 0;
  const cut = (a, b) => {
    const key = a < b ? a * nn + b : b * nn + a;
    let id = cuts.get(key);
    if (id !== undefined) return id;
    const ax = P[3 * a], ay = P[3 * a + 1], az = P[3 * a + 2], ex = P[3 * b] - ax, ey = P[3 * b + 1] - ay, ez = P[3 * b + 2] - az;
    const f = (t) => signedDistance(ax + t * ex, ay + t * ey, az + t * ez, t < 0.5 ? a : b) - eps;
    let t = -1;
    if (phi[a] < 0) {
      t = phi[a] / (phi[a] - phi[b]);
      let v = f(t);
      if (v === v && Math.abs(v + eps) > 1e-4 * h) {
        let t0 = 0, t1 = 1;
        for (let it = 0; it < 14 && v === v; it++) {
          if (v > 0) t1 = t; else t0 = t;
          t = 0.5 * (t0 + t1); v = f(t);
        }
        if (v !== v) t = phi[a] / (phi[a] - phi[b]);
      }
      t = Math.min(0.97, Math.max(0.03, t));
    } else {
      for (const s of [0.75, 0.5, 0.25]) {
        if (!(f(s) <= 0)) continue;
        let t0 = s, t1 = s + 0.25;
        if (s === 0.75 && !(f(1) > 0)) { t0 = t1 = 0.97; }
        for (let it = 0; it < 10; it++) { const m = 0.5 * (t0 + t1); if (f(m) <= 0) t0 = m; else t1 = m; }
        t = Math.min(0.97, t0); extended++;
        break;
      }
    }
    if (t < 0) id = a;
    else { id = P.length / 3; P.push(ax + t * ex, ay + t * ey, az + t * ez); }
    cuts.set(key, id);
    return id;
  };

  // ---- elements
  const T = [], TC = [];
  const tiny = 1e-7 * h * h * h;
  let cell = 0, dropped = 0;
  const add = (a, b, c, d) => {
    if (a === b || a === c || a === d || b === c || b === d || c === d) return;
    const v = vol6(P, a, b, c, d);
    if (Math.abs(v) < tiny) { dropped++; return; }
    if (v > 0) T.push(a, b, c, d); else T.push(a, b, d, c);
    TC.push(cell);
  };
  // Splits a prism (triangle v0 v1 v2, triangle v3 v4 v5, edges v0-v3, v1-v4,
  // v2-v5) into three tetrahedra. Each four-sided face is split from its lowest
  // numbered corner, so the element on the other side of the face agrees.
  // Corners may coincide (a collapsed edge); the flat pieces are skipped.
  const prism = (v) => {
    let m = 0;
    for (let i = 1; i < 6; i++) if (v[i] < v[m]) m = i;
    const r = PRISM_ROT[m], a = v[r[0]], b = v[r[1]], c = v[r[2]], d = v[r[3]], e = v[r[4]], f = v[r[5]];
    if (Math.min(b, f) < Math.min(c, e)) { add(a, b, c, f); add(a, b, f, e); add(a, e, f, d); }
    else { add(a, b, c, e); add(a, e, c, f); add(a, e, f, d); }
  };
  const corner = new Int32Array(8), A = [], O = [];
  const minParent = 0.02 * h * h * h;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    cell = i + nx * (j + ny * k);
    let some = false;
    for (let c = 0; c < 8; c++) {
      const n = (i + (c & 1)) + NX * ((j + ((c >> 1) & 1)) + NY * (k + (c >> 2)));
      corner[c] = n;
      if (phi[n] <= 0) some = true;
    }
    if (!some) continue;
    for (const q of KUHN) {
      const a = corner[q[0]], b = corner[q[1]], c = corner[q[2]], d = corner[q[3]];
      A.length = 0; O.length = 0;
      let anyIn = false;
      for (const n of [a, b, c, d]) { if (phi[n] > 0) O.push(n); else { A.push(n); if (phi[n] < 0) anyIn = true; } }
      if (!A.length) continue;
      if (vol6(P, a, b, c, d) < minParent) { dropped++; continue; } // squashed by the node moves
      if (!O.length) {
        if (!anyIn) {
          // all four nodes on the surface: a wall one element thick, or a pocket of empty space.
          // Decide by looking just inside the element next to its first node.
          const px = 0.8 * P[3 * a] + 0.2 * (P[3 * b] + P[3 * c] + P[3 * d]) / 3, py = 0.8 * P[3 * a + 1] + 0.2 * (P[3 * b + 1] + P[3 * c + 1] + P[3 * d + 1]) / 3, pz = 0.8 * P[3 * a + 2] + 0.2 * (P[3 * b + 2] + P[3 * c + 2] + P[3 * d + 2]) / 3;
          if (!(signedDistance(px, py, pz, a) < 0)) continue;
        }
        T.push(a, b, c, d); TC.push(cell);
      } else if (O.length === 3) add(A[0], cut(A[0], O[0]), cut(A[0], O[1]), cut(A[0], O[2]));
      else if (O.length === 2) prism([A[0], cut(A[0], O[0]), cut(A[0], O[1]), A[1], cut(A[1], O[0]), cut(A[1], O[1])]);
      else prism([A[0], A[1], A[2], cut(A[0], O[0]), cut(A[1], O[0]), cut(A[2], O[0])]);
    }
  }
  if (!T.length) return null;
  const mesh = finishMesh(Float64Array.from(P), Int32Array.from(T), Int32Array.from(TC), grid);
  mesh.snapped = snapped; mesh.extended = extended; mesh.dropped += dropped;
  return mesh;
}

// Packs nodes and elements into a mesh object, dropping unused nodes, and works
// out volumes, element quality, neighbours and the boundary faces.
// Returns {
//   nodes: Float64Array (3 per node), tets: Int32Array (4 per element),
//   cell: Int32Array (grid cell each element came from), grid,
//   vol: Float64Array per element, volume: total,
//   nbr: Int32Array (4 per element: the element across face f, or -1 on the boundary),
//   bfaces: Int32Array (element * 4 + face for every boundary face),
//   quality: {min, mean}, inverted: number of elements with volume <= 0 }
export function finishMesh(P, T, TC, grid) {
  const used = new Int32Array(P.length / 3).fill(-1);
  let nn = 0;
  for (let q = 0; q < T.length; q++) if (used[T[q]] < 0) used[T[q]] = nn++;
  const nodes = new Float64Array(3 * nn), ne = T.length / 4, tets = new Int32Array(T.length);
  for (let n = 0; n < used.length; n++) if (used[n] >= 0) { nodes[3 * used[n]] = P[3 * n]; nodes[3 * used[n] + 1] = P[3 * n + 1]; nodes[3 * used[n] + 2] = P[3 * n + 2]; }
  for (let q = 0; q < T.length; q++) tets[q] = used[T[q]];
  const vol = new Float64Array(ne);
  let volume = 0, qmin = Infinity, qsum = 0, inverted = 0;
  for (let e = 0; e < ne; e++) {
    const a = tets[4 * e], b = tets[4 * e + 1], c = tets[4 * e + 2], d = tets[4 * e + 3];
    vol[e] = vol6(nodes, a, b, c, d) / 6; volume += vol[e];
    if (!(vol[e] > 0)) inverted++;
    const q = tetQuality(nodes, a, b, c, d);
    qsum += q; if (q < qmin) qmin = q;
  }
  // neighbours: bucket every face under its lowest node, then match the other two
  const nf = 4 * ne, start = new Int32Array(nn + 1), fa = new Int32Array(nf), fb = new Int32Array(nf), fmin = new Int32Array(nf);
  for (let e = 0; e < ne; e++) for (let f = 0; f < 4; f++) {
    let p = tets[4 * e + TET_FACE[f][0]], q = tets[4 * e + TET_FACE[f][1]], r = tets[4 * e + TET_FACE[f][2]], s;
    if (p > q) { s = p; p = q; q = s; }
    if (q > r) { s = q; q = r; r = s; }
    if (p > q) { s = p; p = q; q = s; }
    const id = 4 * e + f;
    fmin[id] = p; fa[id] = q; fb[id] = r; start[p + 1]++;
  }
  for (let n = 0; n < nn; n++) start[n + 1] += start[n];
  const fill = start.slice(0, nn), order = new Int32Array(nf);
  for (let id = 0; id < nf; id++) order[fill[fmin[id]]++] = id;
  const nbr = new Int32Array(nf).fill(-1);
  let shared3 = 0;
  for (let n = 0; n < nn; n++) {
    for (let p = start[n]; p < start[n + 1]; p++) {
      const id = order[p];
      if (nbr[id] >= 0) continue;
      for (let q = p + 1; q < start[n + 1]; q++) {
        const od = order[q];
        if (fa[od] !== fa[id] || fb[od] !== fb[id]) continue;
        if (nbr[od] >= 0 || nbr[id] >= 0) { shared3++; continue; }
        nbr[id] = od >> 2; nbr[od] = id >> 2;
      }
    }
  }
  const bf = [];
  for (let id = 0; id < nf; id++) if (nbr[id] < 0) bf.push(id);
  return {
    nodes, tets, cell: TC, grid, vol, volume, nbr, bfaces: Int32Array.from(bf),
    quality: { min: qmin, mean: qsum / ne }, inverted, dropped: 0, overShared: shared3,
  };
}

// Keeps only the elements flagged in keep (Uint8Array per element).
export function filterMesh(mesh, keep) {
  const T = [], TC = [];
  for (let e = 0; e < keep.length; e++) if (keep[e]) { T.push(mesh.tets[4 * e], mesh.tets[4 * e + 1], mesh.tets[4 * e + 2], mesh.tets[4 * e + 3]); TC.push(mesh.cell[e]); }
  const out = finishMesh(mesh.nodes, Int32Array.from(T), Int32Array.from(TC), mesh.grid);
  out.snapped = mesh.snapped; out.extended = mesh.extended; out.dropped = mesh.dropped;
  return out;
}

// Elements joined face to face to any of the seed elements (Uint8Array per element).
export function connectedTo(mesh, seeds) {
  const ne = mesh.vol.length, reach = new Uint8Array(ne), queue = [];
  for (const e of seeds) if (!reach[e]) { reach[e] = 1; queue.push(e); }
  for (let q = 0; q < queue.length; q++) {
    const e = queue[q];
    for (let f = 0; f < 4; f++) { const o = mesh.nbr[4 * e + f]; if (o >= 0 && !reach[o]) { reach[o] = 1; queue.push(o); } }
  }
  return reach;
}

// Corner nodes and area vector (pointing out of the part, length = area) of boundary face number b.
const FN = new Int32Array(3), FA = new Float64Array(3);
export function boundaryFace(mesh, b) {
  const id = mesh.bfaces[b], e = id >> 2, f = TET_FACE[id & 3], N = mesh.nodes;
  const p = mesh.tets[4 * e + f[0]], q = mesh.tets[4 * e + f[1]], r = mesh.tets[4 * e + f[2]];
  FN[0] = p; FN[1] = q; FN[2] = r;
  const ux = N[3 * q] - N[3 * p], uy = N[3 * q + 1] - N[3 * p + 1], uz = N[3 * q + 2] - N[3 * p + 2];
  const vx = N[3 * r] - N[3 * p], vy = N[3 * r + 1] - N[3 * p + 1], vz = N[3 * r + 2] - N[3 * p + 2];
  FA[0] = 0.5 * (uy * vz - uz * vy); FA[1] = 0.5 * (uz * vx - ux * vz); FA[2] = 0.5 * (ux * vy - uy * vx);
  return { nodes: FN, area: FA };
}

// Boundary faces of the mesh that lie on a set of surface triangles of the
// original model: the face centre must be within tol of one of the triangles
// and face roughly the same way. tris: 9 numbers per triangle; picked: the
// triangle numbers. Returns {
//   faces: boundary face numbers (indices into mesh.bfaces),
//   nodes: Map from node number to the face area attached to it (a third of each face),
//   area: total area found }
export function facesOnTriangles(mesh, tris, picked, tol) {
  const h = mesh.grid.h, o = mesh.grid.origin;
  tol = tol ?? 0.35 * h;
  const bins = new Map(), W = 2097152;
  const key = (i, j, k) => (i + 4096) + W * ((j + 4096) + W * (k + 4096));
  for (const t of picked) {
    const lo = [0, 0, 0], hi = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.floor((Math.min(tris[9 * t + a], tris[9 * t + 3 + a], tris[9 * t + 6 + a]) - tol - o[a]) / h);
      hi[a] = Math.floor((Math.max(tris[9 * t + a], tris[9 * t + 3 + a], tris[9 * t + 6 + a]) + tol - o[a]) / h);
    }
    for (let k = lo[2]; k <= hi[2]; k++) for (let j = lo[1]; j <= hi[1]; j++) for (let i = lo[0]; i <= hi[0]; i++) {
      // skip bins the triangle does not come near (keeps long slanted triangles cheap)
      if (closestOnTriangle(o[0] + (i + 0.5) * h, o[1] + (j + 0.5) * h, o[2] + (k + 0.5) * h, tris, 9 * t) > (0.87 * h + tol) ** 2) continue;
      const kk = key(i, j, k);
      let l = bins.get(kk);
      if (!l) bins.set(kk, l = []);
      l.push(t);
    }
  }
  const faces = [], nodes = new Map(), N = mesh.nodes;
  let total = 0;
  for (let b = 0; b < mesh.bfaces.length; b++) {
    const { nodes: fn, area } = boundaryFace(mesh, b);
    const cx = (N[3 * fn[0]] + N[3 * fn[1]] + N[3 * fn[2]]) / 3, cy = (N[3 * fn[0] + 1] + N[3 * fn[1] + 1] + N[3 * fn[2] + 1]) / 3, cz = (N[3 * fn[0] + 2] + N[3 * fn[1] + 2] + N[3 * fn[2] + 2]) / 3;
    const l = bins.get(key(Math.floor((cx - o[0]) / h), Math.floor((cy - o[1]) / h), Math.floor((cz - o[2]) / h)));
    if (!l) continue;
    const A = Math.hypot(area[0], area[1], area[2]);
    if (!(A > 0)) continue;
    let hit = false;
    for (const t of l) {
      const q = 9 * t;
      if (closestOnTriangle(cx, cy, cz, tris, q) > tol * tol) continue;
      const ux = tris[q + 3] - tris[q], uy = tris[q + 4] - tris[q + 1], uz = tris[q + 5] - tris[q + 2];
      const vx = tris[q + 6] - tris[q], vy = tris[q + 7] - tris[q + 1], vz = tris[q + 8] - tris[q + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx, nl = Math.hypot(nx, ny, nz);
      if (nl > 0 && (nx * area[0] + ny * area[1] + nz * area[2]) / (nl * A) > 0.4) { hit = true; break; }
    }
    if (!hit) continue;
    faces.push(b); total += A;
    for (let v = 0; v < 3; v++) nodes.set(fn[v], (nodes.get(fn[v]) || 0) + A / 3);
  }
  return { faces: Int32Array.from(faces), nodes, area: total };
}

// Checks that the mesh is conforming and has no inverted elements. Slow; for tests.
// Every interior face must be shared by exactly two elements and every other
// face by one, and the boundary must be a closed surface.
export function checkMesh(mesh) {
  const ne = mesh.vol.length, faces = new Map(), edges = new Map();
  for (let e = 0; e < ne; e++) for (let f = 0; f < 4; f++) {
    const k = [mesh.tets[4 * e + TET_FACE[f][0]], mesh.tets[4 * e + TET_FACE[f][1]], mesh.tets[4 * e + TET_FACE[f][2]]].sort((a, b) => a - b).join(",");
    faces.set(k, (faces.get(k) || 0) + 1);
  }
  let once = 0, twice = 0, more = 0;
  for (const c of faces.values()) { if (c === 1) once++; else if (c === 2) twice++; else more++; }
  // closed boundary: every boundary edge is used by exactly two boundary faces, once each way
  let surfVol = 0;
  for (let b = 0; b < mesh.bfaces.length; b++) {
    const { nodes: fn, area } = boundaryFace(mesh, b);
    for (let v = 0; v < 3; v++) {
      const p = fn[v], q = fn[(v + 1) % 3], k = p < q ? p + "," + q : q + "," + p;
      edges.set(k, (edges.get(k) || 0) + (p < q ? 1 : -1) + 1000);
    }
    surfVol += (mesh.nodes[3 * fn[0]] * area[0] + mesh.nodes[3 * fn[0] + 1] * area[1] + mesh.nodes[3 * fn[0] + 2] * area[2]) / 3;
  }
  let openEdges = 0;
  for (const c of edges.values()) if (c % 1000 !== 0 || Math.round(c / 1000) % 2 !== 0) openEdges++;
  return {
    elements: ne, nodes: mesh.nodes.length / 3, interiorFaces: twice, boundaryFaces: once, facesSharedByMore: more,
    boundaryMatches: once === mesh.bfaces.length, openEdges, inverted: mesh.inverted,
    volume: mesh.volume, volumeFromSurface: surfVol,
  };
}

// Grid that covers a triangle soup with res cells along its longest side
// (the same grid the cube mesh uses).
export function gridFor(tris, res) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < tris.length; i += 3) for (let a = 0; a < 3; a++) { min[a] = Math.min(min[a], tris[i + a]); max[a] = Math.max(max[a], tris[i + a]); }
  const size = max.map((v, a) => v - min[a]), h = Math.max(...size) / res;
  const dims = size.map((s) => Math.max(1, Math.ceil(s / h - 1e-6)));
  return { nx: dims[0], ny: dims[1], nz: dims[2], h, origin: min };
}
