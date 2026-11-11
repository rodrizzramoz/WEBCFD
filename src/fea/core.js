// Linear elastic finite element analysis and topology optimisation on a
// voxel grid. Every solid voxel is one 8-node brick element. The stiffness
// matrix is never assembled: the solver multiplies element by element
// (matrix-free) inside a preconditioned conjugate gradient loop.
//
// Units: metres, newtons, pascals.

// Local node order of a brick: bottom face anticlockwise, then top face.
const CORNER = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];

function elasticity(nu) {
  // Isotropic stress-strain matrix for E = 1 (engineering shear strains).
  const D = new Float64Array(36);
  const a = 1 / ((1 + nu) * (1 - 2 * nu));
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) D[i * 6 + j] = a * (i === j ? 1 - nu : nu);
  for (let i = 3; i < 6; i++) D[i * 6 + i] = a * (1 - 2 * nu) / 2;
  return D;
}

// Strain-displacement matrix (6 x 24) of a unit cube at natural point (r, s, t) in [-1, 1].
function bMatrix(r, s, t) {
  const B = new Float64Array(6 * 24);
  for (let n = 0; n < 8; n++) {
    const sx = CORNER[n][0] ? 1 : -1, sy = CORNER[n][1] ? 1 : -1, sz = CORNER[n][2] ? 1 : -1;
    // derivatives of the shape function with respect to x, y, z on a cube of side 1
    const dx = 0.25 * sx * (1 + sy * s) * (1 + sz * t);
    const dy = 0.25 * sy * (1 + sx * r) * (1 + sz * t);
    const dz = 0.25 * sz * (1 + sx * r) * (1 + sy * s);
    const c = 3 * n;
    B[0 * 24 + c] = dx; B[1 * 24 + c + 1] = dy; B[2 * 24 + c + 2] = dz;
    B[3 * 24 + c] = dy; B[3 * 24 + c + 1] = dx;           // xy
    B[4 * 24 + c + 1] = dz; B[4 * 24 + c + 2] = dy;       // yz
    B[5 * 24 + c] = dz; B[5 * 24 + c + 2] = dx;           // zx
  }
  return B;
}

// Element stiffness (24 x 24) of a unit cube with E = 1. Real stiffness = E * h * KE.
export function elementStiffness(nu) {
  const D = elasticity(nu), KE = new Float64Array(576), g = 1 / Math.sqrt(3);
  const DB = new Float64Array(6 * 24);
  for (const r of [-g, g]) for (const s of [-g, g]) for (const t of [-g, g]) {
    const B = bMatrix(r, s, t);
    DB.fill(0);
    for (let i = 0; i < 6; i++) for (let k = 0; k < 6; k++) {
      const d = D[i * 6 + k];
      if (d !== 0) for (let j = 0; j < 24; j++) DB[i * 24 + j] += d * B[k * 24 + j];
    }
    for (let i = 0; i < 24; i++) for (let j = 0; j < 24; j++) {
      let v = 0;
      for (let k = 0; k < 6; k++) v += B[k * 24 + i] * DB[k * 24 + j];
      KE[i * 24 + j] += v * 0.125; // Gauss weight 1, Jacobian determinant 1/8
    }
  }
  return KE;
}

export class Model {
  // grid: {nx, ny, nz, h}; solid: Uint8Array per cell (1 = material);
  // fixed: Uint8Array per node; force: Float64Array per degree of freedom;
  // passive: Uint8Array per cell (1 = must stay solid during optimisation)
  constructor({ nx, ny, nz, h, solid, fixed, force, passive, E, nu }) {
    Object.assign(this, { nx, ny, nz, h, E, nu });
    const NX = nx + 1, NY = ny + 1;
    const cells = [];
    for (let c = 0; c < solid.length; c++) if (solid[c]) cells.push(c);
    const ne = cells.length;
    this.ne = ne;
    this.cells = Int32Array.from(cells);
    this.cellToElem = new Int32Array(solid.length).fill(-1);
    this.conn = new Int32Array(ne * 8);
    for (let e = 0; e < ne; e++) {
      const c = cells[e];
      this.cellToElem[c] = e;
      const i = c % nx, j = ((c / nx) | 0) % ny, k = (c / (nx * ny)) | 0;
      for (let n = 0; n < 8; n++) this.conn[e * 8 + n] = (i + CORNER[n][0]) + NX * ((j + CORNER[n][1]) + NY * (k + CORNER[n][2]));
    }
    const nn = NX * NY * (nz + 1);
    this.ndof = 3 * nn;
    // A degree of freedom is free if its node touches material and is not fixed.
    this.free = new Float64Array(this.ndof);
    for (let q = 0; q < this.conn.length; q++) {
      const n = this.conn[q];
      if (!fixed[n]) { this.free[3 * n] = 1; this.free[3 * n + 1] = 1; this.free[3 * n + 2] = 1; }
    }
    this.f = new Float64Array(this.ndof);
    for (let d = 0; d < this.ndof; d++) this.f[d] = force[d] * this.free[d];
    this.passive = new Uint8Array(ne);
    if (passive) for (let e = 0; e < ne; e++) this.passive[e] = passive[cells[e]];
    this.KE = elementStiffness(nu);
    this.B0 = bMatrix(0, 0, 0);
    this.D = elasticity(nu);
    this.u = new Float64Array(this.ndof);
    this.scale = new Float64Array(ne).fill(1); // element stiffness multiplier (E_e / E)
    this._r = new Float64Array(this.ndof); this._z = new Float64Array(this.ndof);
    this._p = new Float64Array(this.ndof); this._q = new Float64Array(this.ndof);
    this._diag = new Float64Array(this.ndof);
    this._ue = new Float64Array(24);
  }

  // out = K * x (in units of E * h)
  multiply(x, out) {
    out.fill(0);
    const { conn, KE, scale, ne } = this, ue = this._ue;
    for (let e = 0; e < ne; e++) {
      const s = scale[e], b = e * 8;
      for (let n = 0; n < 8; n++) {
        const d = 3 * conn[b + n];
        ue[3 * n] = x[d]; ue[3 * n + 1] = x[d + 1]; ue[3 * n + 2] = x[d + 2];
      }
      for (let n = 0; n < 8; n++) {
        const d = 3 * conn[b + n];
        for (let a = 0; a < 3; a++) {
          const row = (3 * n + a) * 24;
          let v = 0;
          for (let j = 0; j < 24; j++) v += KE[row + j] * ue[j];
          out[d + a] += s * v;
        }
      }
    }
    const free = this.free;
    for (let d = 0; d < out.length; d++) out[d] *= free[d];
  }

  // Solves K u = f, starting from the current u. Returns iterations and final relative residual.
  solve(tol, maxIter, onProgress) {
    const { u, f, free, ndof, conn, KE, scale, ne } = this;
    const r = this._r, z = this._z, p = this._p, q = this._q, diag = this._diag;
    const k = this.E * this.h;
    diag.fill(0);
    for (let e = 0; e < ne; e++) {
      for (let n = 0; n < 8; n++) {
        const d = 3 * conn[e * 8 + n];
        for (let a = 0; a < 3; a++) diag[d + a] += scale[e] * KE[(3 * n + a) * 25];
      }
    }
    let fnorm = 0;
    for (let d = 0; d < ndof; d++) fnorm += f[d] * f[d];
    fnorm = Math.sqrt(fnorm);
    if (fnorm === 0) { u.fill(0); return { iterations: 0, residual: 0 }; }
    // work in scaled units: (K / k) u = f / k
    this.multiply(u, q);
    let rz = 0;
    for (let d = 0; d < ndof; d++) {
      r[d] = f[d] / k - q[d];
      z[d] = diag[d] > 0 ? r[d] / diag[d] : 0;
      p[d] = z[d]; rz += r[d] * z[d];
    }
    const target = tol * fnorm / k;
    let it = 0, rn = Infinity;
    for (; it < maxIter; it++) {
      rn = 0;
      for (let d = 0; d < ndof; d++) rn += r[d] * r[d];
      rn = Math.sqrt(rn);
      if (rn <= target) break;
      if (onProgress && it % 25 === 0) onProgress(it, rn * k / fnorm);
      this.multiply(p, q);
      let pq = 0;
      for (let d = 0; d < ndof; d++) pq += p[d] * q[d];
      const alpha = rz / pq;
      let rzNew = 0;
      for (let d = 0; d < ndof; d++) {
        u[d] += alpha * p[d]; r[d] -= alpha * q[d];
        z[d] = diag[d] > 0 ? r[d] / diag[d] : 0;
        rzNew += r[d] * z[d];
      }
      const beta = rzNew / rz; rz = rzNew;
      for (let d = 0; d < ndof; d++) p[d] = z[d] + beta * p[d];
    }
    for (let d = 0; d < ndof; d++) u[d] *= free[d];
    return { iterations: it, residual: rn * k / fnorm };
  }

  // Strain energy per element for unit stiffness multiplier: u_e . KE u_e * E * h
  elementEnergy(out) {
    const { conn, KE, u, ne } = this, ue = this._ue, k = this.E * this.h;
    for (let e = 0; e < ne; e++) {
      for (let n = 0; n < 8; n++) {
        const d = 3 * conn[e * 8 + n];
        ue[3 * n] = u[d]; ue[3 * n + 1] = u[d + 1]; ue[3 * n + 2] = u[d + 2];
      }
      let s = 0;
      for (let i = 0; i < 24; i++) {
        let v = 0;
        for (let j = 0; j < 24; j++) v += KE[i * 24 + j] * ue[j];
        s += ue[i] * v;
      }
      out[e] = s * k;
    }
  }

  compliance() {
    let c = 0;
    for (let d = 0; d < this.ndof; d++) c += this.f[d] * this.u[d];
    return c; // joules; equals twice the stored strain energy
  }

  // Von Mises stress at each element centre, for full-strength material (Pa).
  vonMises(out) {
    const { conn, u, ne, B0, D, h, E } = this, ue = this._ue;
    const eps = new Float64Array(6), sig = new Float64Array(6);
    for (let e = 0; e < ne; e++) {
      for (let n = 0; n < 8; n++) {
        const d = 3 * conn[e * 8 + n];
        ue[3 * n] = u[d]; ue[3 * n + 1] = u[d + 1]; ue[3 * n + 2] = u[d + 2];
      }
      for (let i = 0; i < 6; i++) {
        let v = 0;
        for (let j = 0; j < 24; j++) v += B0[i * 24 + j] * ue[j];
        eps[i] = v / h;
      }
      for (let i = 0; i < 6; i++) {
        let v = 0;
        for (let j = 0; j < 6; j++) v += D[i * 6 + j] * eps[j];
        sig[i] = v * E;
      }
      const [sx, sy, sz, txy, tyz, tzx] = sig;
      out[e] = Math.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy * txy + tyz * tyz + tzx * tzx));
    }
  }

  maxDisplacement() {
    let m = 0;
    const u = this.u;
    for (let d = 0; d < u.length; d += 3) {
      const v = u[d] * u[d] + u[d + 1] * u[d + 1] + u[d + 2] * u[d + 2];
      if (v > m) m = v;
    }
    return Math.sqrt(m);
  }
}

// Topology optimiser: SIMP material model, sensitivity filter, optimality criteria update.
export class Optimiser {
  constructor(model, { penal = 3, rmin = 1.6, emin = 1e-3 } = {}) {
    this.m = model; this.penal = penal; this.rmin = rmin; this.emin = emin;
    const ne = model.ne;
    this.x = new Float64Array(ne).fill(1);
    this.dc = new Float64Array(ne); this.dcf = new Float64Array(ne); this.energy = new Float64Array(ne);
    // neighbour list for the filter
    const { nx, ny, nz, cells, cellToElem } = model, R = Math.ceil(rmin) - 1 || 1;
    const nb = [], w = [], start = new Int32Array(ne + 1);
    for (let e = 0; e < ne; e++) {
      const c = cells[e], i = c % nx, j = ((c / nx) | 0) % ny, k = (c / (nx * ny)) | 0;
      for (let dk = -R; dk <= R; dk++) for (let dj = -R; dj <= R; dj++) for (let di = -R; di <= R; di++) {
        const a = i + di, b = j + dj, cc = k + dk;
        if (a < 0 || b < 0 || cc < 0 || a >= nx || b >= ny || cc >= nz) continue;
        const wt = rmin - Math.sqrt(di * di + dj * dj + dk * dk);
        if (wt <= 0) continue;
        const o = cellToElem[a + nx * (b + ny * cc)];
        if (o >= 0) { nb.push(o); w.push(wt); }
      }
      start[e + 1] = nb.length;
    }
    this.nb = Int32Array.from(nb); this.w = Float64Array.from(w); this.start = start;
  }

  applyDensities() {
    const { x, m, penal, emin } = this;
    for (let e = 0; e < m.ne; e++) m.scale[e] = emin + Math.pow(x[e], penal) * (1 - emin);
  }

  // One design update toward the target volume fraction. Returns the largest density change.
  update(volFrac, solveTol = 1e-3, solveIter = 600) {
    const { x, m, penal, emin, dc, dcf, nb, w, start } = this, ne = m.ne;
    this.applyDensities();
    const info = m.solve(solveTol, solveIter);
    m.elementEnergy(this.energy);
    for (let e = 0; e < ne; e++) dc[e] = -penal * Math.pow(x[e], penal - 1) * (1 - emin) * this.energy[e];
    for (let e = 0; e < ne; e++) {
      let s = 0, ws = 0;
      for (let q = start[e]; q < start[e + 1]; q++) { const o = nb[q]; s += w[q] * x[o] * dc[o]; ws += w[q]; }
      dcf[e] = s / (Math.max(1e-3, x[e]) * ws);
    }
    // bisection on the Lagrange multiplier so the volume matches the target
    const move = 0.2, xnew = new Float64Array(ne);
    let lo = 0, hi = 1, change = 0;
    let dmax = 0;
    for (let e = 0; e < ne; e++) if (-dcf[e] > dmax) dmax = -dcf[e];
    hi = dmax * 10 + 1e-30; lo = hi * 1e-12;
    for (let it = 0; it < 80 && (hi - lo) / (hi + lo) > 1e-4; it++) {
      const mid = 0.5 * (lo + hi);
      let vol = 0;
      for (let e = 0; e < ne; e++) {
        let v = m.passive[e] ? 1 : x[e] * Math.sqrt(Math.max(0, -dcf[e]) / mid);
        if (!m.passive[e]) v = Math.max(1e-3, Math.max(x[e] - move, Math.min(1, Math.min(x[e] + move, v))));
        xnew[e] = v; vol += v;
      }
      if (vol > volFrac * ne) lo = mid; else hi = mid;
    }
    for (let e = 0; e < ne; e++) { change = Math.max(change, Math.abs(xnew[e] - x[e])); x[e] = xnew[e]; }
    return { change, compliance: m.compliance(), iterations: info.iterations };
  }

  volume() { let v = 0; for (let e = 0; e < this.m.ne; e++) v += this.x[e]; return v / this.m.ne; }
}
