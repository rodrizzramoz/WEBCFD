// Turns a triangle mesh into the cell grid the solver runs on.
//
// Cell types (low 3 bits of each flag):
//   0 fluid, 1 solid wall, 2 far field (fixed velocity, fixed pressure),
//   3 velocity inlet (fixed velocity, floating pressure), 4 pressure outlet.
// Bits 3-5 hold the face index for boundary cells. Faces are numbered
//   0 X min, 1 X max, 2 Y min, 3 Y max, 4 Z min, 5 Z max
// and the inward normal of face f is NORMALS[f].

export const FLUID = 0, SOLID = 1, FARFIELD = 2, INLET = 3, OUTLET = 4;
export const NORMALS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
export const FACE_NAMES = ["X min", "X max", "Y min", "Y max", "Z min", "Z max"];

// Winding-number voxeliser. Casts one ray along +X through every (y, z) cell
// column and counts signed surface crossings, so overlapping closed shells
// and inverted normals are both handled.
export function voxelize(tris, origin, dx, nx, ny, nz) {
  const solid = new Uint8Array(nx * ny * nz);
  let openColumns = 0, hitColumns = 0;
  const cols = new Array(ny * nz);
  const JY = 0.50137, JZ = 0.50071; // tiny offset so rays never run exactly along a mesh edge
  const oy = origin[1], oz = origin[2];
  for (let t = 0; t < tris.length; t += 9) {
    const ax = tris[t], ay = tris[t + 1], az = tris[t + 2];
    const bx = tris[t + 3], by = tris[t + 4], bz = tris[t + 5];
    const cx = tris[t + 6], cy = tris[t + 7], cz = tris[t + 8];
    const d = (by - ay) * (cz - az) - (cy - ay) * (bz - az);
    if (Math.abs(d) < 1e-30) continue;
    const sign = d < 0 ? 1 : -1;
    const j0 = Math.max(0, Math.ceil((Math.min(ay, by, cy) - oy) / dx - JY));
    const j1 = Math.min(ny - 1, Math.floor((Math.max(ay, by, cy) - oy) / dx - JY));
    const k0 = Math.max(0, Math.ceil((Math.min(az, bz, cz) - oz) / dx - JZ));
    const k1 = Math.min(nz - 1, Math.floor((Math.max(az, bz, cz) - oz) / dx - JZ));
    for (let k = k0; k <= k1; k++) {
      const pz = oz + (k + JZ) * dx;
      for (let j = j0; j <= j1; j++) {
        const py = oy + (j + JY) * dx;
        const w1 = ((py - ay) * (cz - az) - (cy - ay) * (pz - az)) / d;
        const w2 = ((by - ay) * (pz - az) - (py - ay) * (bz - az)) / d;
        if (w1 < 0 || w2 < 0 || w1 + w2 > 1) continue;
        const x = ax + w1 * (bx - ax) + w2 * (cx - ax);
        const c = j + k * ny;
        (cols[c] || (cols[c] = [])).push(x, sign);
      }
    }
  }
  for (let c = 0; c < cols.length; c++) {
    const hits = cols[c];
    if (!hits) continue;
    const n = hits.length / 2;
    const order = Array.from({ length: n }, (_, i) => i).sort((p, q) => hits[2 * p] - hits[2 * q]);
    // A closed surface is entered as often as it is left. If the total is not zero the
    // mesh has a gap on this ray, so stop filling after the last crossing instead of
    // drawing a solid streak all the way to the end of the grid.
    let total = 0;
    for (let i = 0; i < n; i++) total += hits[2 * i + 1];
    hitColumns++;
    if (total !== 0) openColumns++;
    let p = 0, wind = 0;
    const base = c * nx;
    for (let i = 0; i < nx; i++) {
      const xc = origin[0] + (i + 0.5) * dx;
      while (p < n && hits[2 * order[p]] < xc) { wind += hits[2 * order[p] + 1]; p++; }
      if (p === n && total !== 0) break;
      if (wind !== 0) solid[base + i] = 1;
    }
  }
  // Extra information for callers that want to warn about a mesh that is not watertight.
  solid.openColumns = openColumns;
  solid.hitColumns = hitColumns;
  return solid;
}

function bounds(tris) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < tris.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = tris[i + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
  }
  return { min, max, size: [max[0] - min[0], max[1] - min[1], max[2] - min[2]] };
}

// Domain box for the current setup, without voxelising. Used for the live preview.
export function domainBox(tris, opt) {
  const bb = bounds(tris);
  if (opt.mode === "external") {
    const s = bb.size, cross = Math.max(s[1], s[2]);
    const ref = Math.max(cross, 0.4 * s[0]) * opt.domainScale;
    const up = 1.5 * ref, down = 4 * ref, lat = 1.5 * ref;
    const min = [bb.min[0] - up, bb.min[1] - lat, bb.min[2] - lat];
    const max = [bb.max[0] + down, bb.max[1] + lat, bb.max[2] + lat];
    const dx = (max[0] - min[0]) / opt.res;
    const dims = [opt.res, Math.ceil((max[1] - min[1]) / dx), Math.ceil((max[2] - min[2]) / dx)];
    return { bb, origin: min, dx, dims };
  }
  const dx = Math.max(bb.size[0], bb.size[1], bb.size[2]) / opt.res;
  const origin = [0, 0, 0], dims = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const onMin = opt.inletFace === 2 * a || opt.outletFace === 2 * a;
    const onMax = opt.inletFace === 2 * a + 1 || opt.outletFace === 2 * a + 1;
    const e = bb.size[a] / dx;
    if (onMin && onMax) { origin[a] = bb.min[a]; dims[a] = Math.max(3, Math.floor(e + 0.5)); }
    else if (onMin) { origin[a] = bb.min[a]; dims[a] = Math.ceil(e) + 1; }
    else if (onMax) { dims[a] = Math.ceil(e) + 1; origin[a] = bb.max[a] - dims[a] * dx; }
    else { dims[a] = Math.ceil(e) + 2; origin[a] = bb.min[a] - dx; }
  }
  return { bb, origin, dx, dims };
}

// Cells of one boundary layer, as [index, u, v] triples where (u, v) are the two in-plane coordinates.
function layerCells(face, dims) {
  const a = face >> 1, b = (a + 1) % 3, c = (a + 2) % 3;
  const pos = [0, 0, 0];
  pos[a] = face & 1 ? dims[a] - 1 : 0;
  const out = [];
  for (let v = 0; v < dims[c]; v++) {
    for (let u = 0; u < dims[b]; u++) {
      pos[b] = u; pos[c] = v;
      out.push(pos[0] + dims[0] * (pos[1] + dims[1] * pos[2]), u, v);
    }
  }
  return { cells: out, nu: dims[b], nv: dims[c] };
}

// Open cells on a face layer. For a solid part, openings that reach the rim of
// the face are the outside of the part, not a port, so they are dropped.
function findOpenings(face, dims, isOpen, dropExterior) {
  const { cells, nu, nv } = layerCells(face, dims);
  const n = nu * nv;
  const open = new Uint8Array(n);
  let outsideOnly = false;
  for (let m = 0; m < n; m++) open[m] = isOpen(cells[3 * m]) ? 1 : 0;
  if (dropExterior) {
    const ext = new Uint8Array(n), stack = [];
    for (let m = 0; m < n; m++) {
      const u = m % nu, v = (m / nu) | 0;
      if ((u <= 1 || v <= 1 || u >= nu - 2 || v >= nv - 2) && open[m]) { ext[m] = 1; stack.push(m); }
    }
    while (stack.length) {
      const m = stack.pop(), u = m % nu, v = (m / nu) | 0;
      const nb = [u > 0 ? m - 1 : -1, u < nu - 1 ? m + 1 : -1, v > 0 ? m - nu : -1, v < nv - 1 ? m + nu : -1];
      for (const q of nb) if (q >= 0 && open[q] && !ext[q]) { ext[q] = 1; stack.push(q); }
    }
    let inner = 0;
    for (let m = 0; m < n; m++) if (open[m] && !ext[m]) inner++;
    if (inner > 0) { for (let m = 0; m < n; m++) if (ext[m]) open[m] = 0; }
    else outsideOnly = true;
  }
  const list = [];
  for (let m = 0; m < n; m++) if (open[m]) list.push(cells[3 * m]);
  list.outsideOnly = outsideOnly && list.length > 0; // no enclosed port here, only the space around the part
  return list;
}

export function buildDomain(tris, opt) {
  const box = domainBox(tris, opt);
  const [nx, ny, nz] = box.dims, dx = box.dx, n = nx * ny * nz;
  const raw = voxelize(tris, box.origin, dx, nx, ny, nz);
  const flags = new Uint32Array(n);
  const grid = { nx, ny, nz, n, dx, origin: box.origin, flags, bb: box.bb, mode: opt.mode };
  grid.openColumns = raw.openColumns; grid.hitColumns = raw.hitColumns;
  const stride = [1, nx, nx * ny];

  if (opt.mode === "external") {
    let solidCount = 0;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const c = i + nx * (j + ny * k);
          let f = raw[c] ? SOLID : FLUID;
          if (i === nx - 1) f = OUTLET | (1 << 3);
          else if (i === 0 || j === 0 || k === 0 || j === ny - 1 || k === nz - 1) f = FARFIELD;
          if (f === SOLID) solidCount++;
          flags[c] = f;
        }
      }
    }
    let frontal = 0;
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        const base = nx * (j + ny * k);
        for (let i = 1; i < nx - 1; i++) if (raw[base + i]) { frontal++; break; }
      }
    }
    if (!solidCount) {
      throw new Error(raw.openColumns > 0
        ? "This STL has gaps in its surface (it is not watertight), so no solid shape could be built from it. Repair the mesh and load it again."
        : "The model is smaller than one cell at this grid detail, or it is a flat surface with no thickness. Raise the grid detail, check the STL units, or use a closed solid.");
    }
    grid.frontalArea = frontal * dx * dx;
    grid.refLength = box.bb.size[0];
    grid.inVec = [1, 0, 0];
    grid.speedUp = 1.5;
    return grid;
  }

  // Internal flow.
  const fluidIsInside = opt.stlIsFluid;
  const isOpen = (c) => (raw[c] === 1) === fluidIsInside;
  const interior = (c) => {
    const i = c % nx, j = ((c / nx) | 0) % ny, k = (c / (nx * ny)) | 0;
    return i > 0 && j > 0 && k > 0 && i < nx - 1 && j < ny - 1 && k < nz - 1;
  };
  const step = (face) => { const nrm = NORMALS[face]; return nrm[0] * stride[0] + nrm[1] * stride[1] + nrm[2] * stride[2]; };
  const inOpen = findOpenings(opt.inletFace, box.dims, isOpen, !fluidIsInside);
  const outOpen = findOpenings(opt.outletFace, box.dims, isOpen, !fluidIsInside);
  if (!inOpen.length) throw new Error("No opening found on the inlet face (" + FACE_NAMES[opt.inletFace] + "). Pick another face or rotate the model.");
  if (!outOpen.length) throw new Error("No opening found on the outlet face (" + FACE_NAMES[opt.outletFace] + "). Pick another face or rotate the model.");

  const reached = new Uint8Array(n);
  const queue = [];
  const sIn = step(opt.inletFace), sOut = step(opt.outletFace);
  for (const c of inOpen) {
    const q = c + sIn;
    if (interior(q) && isOpen(q) && !reached[q]) { reached[q] = 1; queue.push(q); }
  }
  for (let h = 0; h < queue.length; h++) {
    const c = queue[h];
    for (let a = 0; a < 3; a++) {
      for (const q of [c - stride[a], c + stride[a]]) {
        if (!reached[q] && interior(q) && isOpen(q)) { reached[q] = 1; queue.push(q); }
      }
    }
  }
  flags.fill(SOLID);
  for (const c of queue) flags[c] = FLUID;
  const inletNbr = [], outletNbr = [];
  for (const c of inOpen) if (reached[c + sIn]) { flags[c] = INLET | (opt.inletFace << 3); inletNbr.push(c + sIn); }
  for (const c of outOpen) if (reached[c + sOut]) { flags[c] = OUTLET | (opt.outletFace << 3); outletNbr.push(c + sOut); }
  if (!outletNbr.length) throw new Error("There is no open path from the inlet face to the outlet face. Check the faces, or switch what the STL represents (solid part or fluid volume).");

  // Narrowest cross-section along the inlet axis, to keep the fastest flow inside the solver's speed limit.
  const a = opt.inletFace >> 1, slices = new Uint32Array(box.dims[a]);
  for (const c of queue) {
    const p = a === 0 ? c % nx : a === 1 ? ((c / nx) | 0) % ny : (c / (nx * ny)) | 0;
    slices[p]++;
  }
  let minSlice = Infinity;
  for (let s = 1; s < slices.length - 1; s++) if (slices[s] > 0 && slices[s] < minSlice) minSlice = slices[s];
  grid.speedUp = Math.min(8, Math.max(1.5, 1.3 * inletNbr.length / Math.max(1, Math.min(minSlice, outletNbr.length))));
  grid.inletArea = inletNbr.length * dx * dx;
  grid.outletArea = outletNbr.length * dx * dx;
  grid.refLength = Math.sqrt(4 * grid.inletArea / Math.PI);
  grid.inletNbr = Uint32Array.from(inletNbr);
  grid.outletNbr = Uint32Array.from(outletNbr);
  grid.inVec = NORMALS[opt.inletFace];
  grid.outNormal = NORMALS[opt.outletFace];
  grid.fluidCells = queue.length;
  grid.portsOutside = [inOpen.outsideOnly ? opt.inletFace : -1, outOpen.outsideOnly ? opt.outletFace : -1].filter((f) => f >= 0);
  return grid;
}
