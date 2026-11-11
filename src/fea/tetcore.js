// Linear elastic finite element analysis and topology optimisation on a mesh
// of 4-node (linear) tetrahedra. Same role as Model and Optimiser in core.js,
// which do the same job on a grid of cubes.
//
// The stiffness matrix is never assembled. A linear tetrahedron has constant
// strain, so the element only needs its four shape-function gradients and its
// volume; K u is formed element by element inside a preconditioned conjugate
// gradient loop, in 64-bit floats on the CPU.
//
// Plain linear tetrahedra are much too stiff in bending on a coarse mesh. So by
// default the strain is smoothed: for every mesh edge, the strain is averaged
// (weighted by volume) over the elements around that edge, and each element
// then uses the mean of its six edge values. This is the edge-based smoothed
// finite element method (ES-FEM-T4; Liu, Nguyen-Thoi and Lam, 2009). It uses
// the same nodes, gradients and volumes, costs about three times as much per
// solver pass, and removes most of the bending error. Pass smooth: false for
// plain linear tetrahedra.
//
// Units: metres, newtons, pascals.

const EDGE = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

export class TetModel {
  // nodes: Float64Array (3 per node); tets: Int32Array (4 per element, positive volume);
  // fixed: Uint8Array per node; force: Float64Array per degree of freedom;
  // passive: Uint8Array per element (1 = must stay solid during optimisation);
  // smooth: false switches the strain smoothing off; coarse: false switches the
  // coarse-level part of the preconditioner off
  constructor({ nodes, tets, fixed, force, passive, E, nu, smooth, coarse, coarseLimit }) {
    const ne = tets.length / 4, nn = nodes.length / 3;
    Object.assign(this, { nodes, conn: tets, ne, nn, E, nu, ndof: 3 * nn });
    this.lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    this.mu = E / (2 * (1 + nu));
    this.G = new Float64Array(12 * ne); // gradient of each node's shape function
    this.vol = new Float64Array(ne);
    this.volume = 0;
    const G = this.G;
    for (let e = 0; e < ne; e++) {
      const a = 3 * tets[4 * e], b = 3 * tets[4 * e + 1], c = 3 * tets[4 * e + 2], d = 3 * tets[4 * e + 3];
      const x1 = nodes[b] - nodes[a], y1 = nodes[b + 1] - nodes[a + 1], z1 = nodes[b + 2] - nodes[a + 2];
      const x2 = nodes[c] - nodes[a], y2 = nodes[c + 1] - nodes[a + 1], z2 = nodes[c + 2] - nodes[a + 2];
      const x3 = nodes[d] - nodes[a], y3 = nodes[d + 1] - nodes[a + 1], z3 = nodes[d + 2] - nodes[a + 2];
      const det = x1 * (y2 * z3 - z2 * y3) - y1 * (x2 * z3 - z2 * x3) + z1 * (x2 * y3 - y2 * x3);
      const o = 12 * e, inv = 1 / det;
      // rows of the inverse of the edge matrix are the gradients of nodes 1, 2 and 3
      G[o + 3] = (y2 * z3 - z2 * y3) * inv; G[o + 4] = (z2 * x3 - x2 * z3) * inv; G[o + 5] = (x2 * y3 - y2 * x3) * inv;
      G[o + 6] = (y3 * z1 - z3 * y1) * inv; G[o + 7] = (z3 * x1 - x3 * z1) * inv; G[o + 8] = (x3 * y1 - y3 * x1) * inv;
      G[o + 9] = (y1 * z2 - z1 * y2) * inv; G[o + 10] = (z1 * x2 - x1 * z2) * inv; G[o + 11] = (x1 * y2 - y1 * x2) * inv;
      for (let k = 0; k < 3; k++) G[o + k] = -(G[o + 3 + k] + G[o + 6 + k] + G[o + 9 + k]);
      this.vol[e] = det / 6; this.volume += det / 6;
    }
    this.free = new Float64Array(this.ndof);
    for (let q = 0; q < tets.length; q++) {
      const n = tets[q];
      if (!fixed[n]) { this.free[3 * n] = 1; this.free[3 * n + 1] = 1; this.free[3 * n + 2] = 1; }
    }
    this.f = new Float64Array(this.ndof);
    for (let d = 0; d < this.ndof; d++) this.f[d] = force[d] * this.free[d];
    this.passive = passive ? Uint8Array.from(passive) : new Uint8Array(ne);
    this.u = new Float64Array(this.ndof);
    this.scale = new Float64Array(ne).fill(1); // element stiffness multiplier (E_e / E)
    this._r = new Float64Array(this.ndof); this._z = new Float64Array(this.ndof);
    this._p = new Float64Array(this.ndof); this._q = new Float64Array(this.ndof);
    this._diag = new Float64Array(this.ndof);
    this.smooth = smooth !== false;
    if (coarse !== false) this._buildBlocks(coarseLimit);
    if (this.smooth) {
      // number the mesh edges; edge[6 e + k] is edge k of element e
      const map = new Map(), edge = this.edge = new Int32Array(6 * ne);
      for (let e = 0; e < ne; e++) for (let k = 0; k < 6; k++) {
        const a = tets[4 * e + EDGE[k][0]], b = tets[4 * e + EDGE[k][1]], key = a < b ? a * nn + b : b * nn + a;
        let id = map.get(key);
        if (id === undefined) { id = map.size; map.set(key, id); }
        edge[6 * e + k] = id;
      }
      this.nedge = map.size;
      this.W = new Float64Array(this.nedge); // volume of the elements around each edge
      for (let e = 0; e < ne; e++) for (let k = 0; k < 6; k++) this.W[edge[6 * e + k]] += this.vol[e];
      this._S = new Float64Array(6 * this.nedge); this._fac = new Float64Array(this.nedge);
      this._prepare();
    }
  }

  // ---- Coarse level of the preconditioner.
  // The diagonal (Jacobi) preconditioner alone needs many passes, because it
  // only corrects the error node by node. So the nodes are also grouped into
  // blocks a few elements wide, and each block is allowed to move as a rigid
  // body (3 shifts and 3 rotations). The stiffness matrix seen through those
  // few motions is small enough to factorise directly, and solving it in every
  // pass removes the long-range part of the error at once. (Two-level additive
  // preconditioner with a rigid-body coarse space.)
  // limit = most coarse unknowns allowed. The coarse solve costs time in every
  // pass, so small meshes get fewer blocks; the default balances the two.
  _buildBlocks(limit) {
    if (!(limit > 0)) limit = Math.min(1500, Math.max(300, Math.round(3 * Math.sqrt(this.ne))));
    const { nodes, nn, free } = this, min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let n = 0; n < nn; n++) for (let a = 0; a < 3; a++) { const v = nodes[3 * n + a]; if (v < min[a]) min[a] = v; if (v > max[a]) max[a] = v; }
    const he = Math.cbrt(6 * this.volume / this.ne);
    let block, count, H;
    for (let b = 3; ; b++) {
      H = b * he;
      const ids = new Map();
      block = new Int32Array(nn).fill(-1);
      for (let n = 0; n < nn; n++) {
        if (!free[3 * n]) continue;
        const key = Math.floor((nodes[3 * n] - min[0]) / H) + 4096 * (Math.floor((nodes[3 * n + 1] - min[1]) / H) + 4096 * Math.floor((nodes[3 * n + 2] - min[2]) / H));
        let id = ids.get(key);
        if (id === undefined) { id = ids.size; ids.set(key, id); }
        block[n] = id;
      }
      count = ids.size;
      if (6 * count <= limit) break;
    }
    if (!count) return;
    // arm of every node from the centre of its block, in units of the block size
    const centre = new Float64Array(3 * count), num = new Float64Array(count), arm = new Float64Array(3 * nn);
    for (let n = 0; n < nn; n++) { const b = block[n]; if (b < 0) continue; num[b]++; for (let a = 0; a < 3; a++) centre[3 * b + a] += nodes[3 * n + a]; }
    for (let n = 0; n < nn; n++) { const b = block[n]; if (b < 0) continue; for (let a = 0; a < 3; a++) arm[3 * n + a] = (nodes[3 * n + a] - centre[3 * b + a] / num[b]) / H; }
    this._block = block; this._arm = arm; this._nc = 6 * count;
    this._A = new Float64Array(this._nc * this._nc); this._rc = new Float64Array(this._nc);
  }

  // Builds and factorises the coarse matrix for the current stiffness multipliers.
  _factorCoarse() {
    const N = this._nc, A = this._A, { conn, G, vol, scale, ne, lambda, mu } = this, block = this._block, arm = this._arm;
    A.fill(0);
    const R = new Float64Array(4 * 18), M = new Float64Array(18), K = new Float64Array(9);
    for (let e = 0; e < ne; e++) {
      const o = 12 * e, s = scale[e] * vol[e];
      for (let n = 0; n < 4; n++) {
        // R = [identity, C] with C w = w x arm
        const nd = conn[4 * e + n], dx = arm[3 * nd], dy = arm[3 * nd + 1], dz = arm[3 * nd + 2], r = 18 * n;
        R[r] = 1; R[r + 1] = 0; R[r + 2] = 0; R[r + 3] = 0; R[r + 4] = dz; R[r + 5] = -dy;
        R[r + 6] = 0; R[r + 7] = 1; R[r + 8] = 0; R[r + 9] = -dz; R[r + 10] = 0; R[r + 11] = dx;
        R[r + 12] = 0; R[r + 13] = 0; R[r + 14] = 1; R[r + 15] = dy; R[r + 16] = -dx; R[r + 17] = 0;
      }
      for (let n = 0; n < 4; n++) {
        const bn = block[conn[4 * e + n]];
        if (bn < 0) continue;
        const gn0 = G[o + 3 * n], gn1 = G[o + 3 * n + 1], gn2 = G[o + 3 * n + 2];
        for (let m = n; m < 4; m++) {
          const bm = block[conn[4 * e + m]];
          if (bm < 0) continue;
          const gm0 = G[o + 3 * m], gm1 = G[o + 3 * m + 1], gm2 = G[o + 3 * m + 2], dot = mu * (gn0 * gm0 + gn1 * gm1 + gn2 * gm2);
          K[0] = s * (lambda * gn0 * gm0 + mu * gn0 * gm0 + dot); K[1] = s * (lambda * gn0 * gm1 + mu * gn1 * gm0); K[2] = s * (lambda * gn0 * gm2 + mu * gn2 * gm0);
          K[3] = s * (lambda * gn1 * gm0 + mu * gn0 * gm1); K[4] = s * (lambda * gn1 * gm1 + mu * gn1 * gm1 + dot); K[5] = s * (lambda * gn1 * gm2 + mu * gn2 * gm1);
          K[6] = s * (lambda * gn2 * gm0 + mu * gn0 * gm2); K[7] = s * (lambda * gn2 * gm1 + mu * gn1 * gm2); K[8] = s * (lambda * gn2 * gm2 + mu * gn2 * gm2 + dot);
          const rn = 18 * n, rm = 18 * m;
          for (let i = 0; i < 3; i++) for (let j = 0; j < 6; j++) M[6 * i + j] = K[3 * i] * R[rm + j] + K[3 * i + 1] * R[rm + 6 + j] + K[3 * i + 2] * R[rm + 12 + j];
          for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
            const v = R[rn + i] * M[j] + R[rn + 6 + i] * M[6 + j] + R[rn + 12 + i] * M[12 + j];
            A[(6 * bn + i) * N + 6 * bm + j] += v;
            if (m !== n) A[(6 * bm + j) * N + 6 * bn + i] += v;
          }
        }
      }
    }
    // Cholesky factor in the lower triangle. A block with very few nodes cannot
    // make all six motions, so a small shift keeps the factorisation going.
    let big = 0;
    for (let i = 0; i < N; i++) if (A[i * N + i] > big) big = A[i * N + i];
    const shift = 1e-9 * big;
    for (let j = 0; j < N; j++) {
      let d = A[j * N + j] + shift;
      for (let k = 0; k < j; k++) d -= A[j * N + k] * A[j * N + k];
      d = Math.sqrt(d > shift * 1e-3 ? d : shift);
      A[j * N + j] = d;
      for (let i = j + 1; i < N; i++) {
        let v = A[i * N + j];
        const ri = i * N, rj = j * N;
        for (let k = 0; k < j; k++) v -= A[ri + k] * A[rj + k];
        A[ri + j] = v / d;
      }
    }
  }

  // z = r / diagonal + coarse correction
  _precondition(r, z) {
    const { ndof, free } = this, diag = this._diag;
    for (let d = 0; d < ndof; d++) z[d] = diag[d] > 0 && free[d] ? r[d] / diag[d] : 0;
    if (!this._nc) return;
    const N = this._nc, A = this._A, c = this._rc, block = this._block, arm = this._arm, nn = this.nn;
    c.fill(0);
    for (let n = 0; n < nn; n++) {
      const b = block[n];
      if (b < 0) continue;
      const q = 6 * b, x = r[3 * n], y = r[3 * n + 1], w = r[3 * n + 2], dx = arm[3 * n], dy = arm[3 * n + 1], dz = arm[3 * n + 2];
      c[q] += x; c[q + 1] += y; c[q + 2] += w;
      c[q + 3] += dy * w - dz * y; c[q + 4] += dz * x - dx * w; c[q + 5] += dx * y - dy * x;
    }
    for (let i = 0; i < N; i++) { let v = c[i]; const ri = i * N; for (let k = 0; k < i; k++) v -= A[ri + k] * c[k]; c[i] = v / A[ri + i]; }
    for (let i = N - 1; i >= 0; i--) { let v = c[i]; for (let k = i + 1; k < N; k++) v -= A[k * N + i] * c[k]; c[i] = v / A[i * N + i]; }
    for (let n = 0; n < nn; n++) {
      const b = block[n];
      if (b < 0) continue;
      const q = 6 * b, dx = arm[3 * n], dy = arm[3 * n + 1], dz = arm[3 * n + 2], wx = c[q + 3], wy = c[q + 4], wz = c[q + 5];
      z[3 * n] += c[q] + wy * dz - wz * dy; z[3 * n + 1] += c[q + 1] + wz * dx - wx * dz; z[3 * n + 2] += c[q + 2] + wx * dy - wy * dx;
    }
  }

  // Per-edge factor for multiply. Call after the stiffness multipliers change.
  _prepare() {
    if (!this.smooth) return;
    const { edge, vol, scale, ne, W, nedge } = this, fac = this._fac;
    fac.fill(0);
    for (let e = 0; e < ne; e++) { const v = vol[e] * scale[e], q = 6 * e; fac[edge[q]] += v; fac[edge[q + 1]] += v; fac[edge[q + 2]] += v; fac[edge[q + 3]] += v; fac[edge[q + 4]] += v; fac[edge[q + 5]] += v; }
    for (let k = 0; k < nedge; k++) fac[k] /= 6 * W[k] * W[k];
  }

  // Fills S with volume times strain summed over the elements around each edge
  // (xx, yy, zz, xy, yz, zx with engineering shear), for displacement x.
  _edgeStrain(x) {
    const { conn, G, vol, ne, edge } = this, S = this._S;
    S.fill(0);
    for (let e = 0; e < ne; e++) {
      const o = 12 * e, q = 4 * e, w = 6 * e;
      const a = 3 * conn[q], b = 3 * conn[q + 1], c = 3 * conn[q + 2], d = 3 * conn[q + 3];
      const ax = x[a], ay = x[a + 1], az = x[a + 2], bx = x[b], by = x[b + 1], bz = x[b + 2];
      const cx = x[c], cy = x[c + 1], cz = x[c + 2], dx = x[d], dy = x[d + 1], dz = x[d + 2];
      const g0 = G[o], g1 = G[o + 1], g2 = G[o + 2], g3 = G[o + 3], g4 = G[o + 4], g5 = G[o + 5];
      const g6 = G[o + 6], g7 = G[o + 7], g8 = G[o + 8], g9 = G[o + 9], g10 = G[o + 10], g11 = G[o + 11];
      const v = vol[e];
      const exx = v * (ax * g0 + bx * g3 + cx * g6 + dx * g9), eyy = v * (ay * g1 + by * g4 + cy * g7 + dy * g10), ezz = v * (az * g2 + bz * g5 + cz * g8 + dz * g11);
      const gxy = v * (ax * g1 + bx * g4 + cx * g7 + dx * g10 + ay * g0 + by * g3 + cy * g6 + dy * g9);
      const gyz = v * (ay * g2 + by * g5 + cy * g8 + dy * g11 + az * g1 + bz * g4 + cz * g7 + dz * g10);
      const gzx = v * (az * g0 + bz * g3 + cz * g6 + dz * g9 + ax * g2 + bx * g5 + cx * g8 + dx * g11);
      for (let k = 0; k < 6; k++) {
        const i = 6 * edge[w + k];
        S[i] += exx; S[i + 1] += eyy; S[i + 2] += ezz; S[i + 3] += gxy; S[i + 4] += gyz; S[i + 5] += gzx;
      }
    }
  }

  // out = K * x with smoothed strains
  _multiplySmooth(x, out) {
    out.fill(0);
    const { conn, G, vol, ne, lambda, mu, edge, nedge } = this, S = this._S, fac = this._fac;
    this._edgeStrain(x);
    for (let k = 0, i = 0; k < nedge; k++, i += 6) { const f = fac[k]; S[i] *= f; S[i + 1] *= f; S[i + 2] *= f; S[i + 3] *= f; S[i + 4] *= f; S[i + 5] *= f; }
    for (let e = 0; e < ne; e++) {
      const o = 12 * e, q = 4 * e, w = 6 * e;
      let exx = 0, eyy = 0, ezz = 0, gxy = 0, gyz = 0, gzx = 0;
      for (let k = 0; k < 6; k++) {
        const i = 6 * edge[w + k];
        exx += S[i]; eyy += S[i + 1]; ezz += S[i + 2]; gxy += S[i + 3]; gyz += S[i + 4]; gzx += S[i + 5];
      }
      const v = vol[e], lt = lambda * (exx + eyy + ezz) * v, m = mu * v;
      const sxx = lt + 2 * m * exx, syy = lt + 2 * m * eyy, szz = lt + 2 * m * ezz, sxy = m * gxy, syz = m * gyz, szx = m * gzx;
      const a = 3 * conn[q], b = 3 * conn[q + 1], c = 3 * conn[q + 2], d = 3 * conn[q + 3];
      const g0 = G[o], g1 = G[o + 1], g2 = G[o + 2], g3 = G[o + 3], g4 = G[o + 4], g5 = G[o + 5];
      const g6 = G[o + 6], g7 = G[o + 7], g8 = G[o + 8], g9 = G[o + 9], g10 = G[o + 10], g11 = G[o + 11];
      out[a] += sxx * g0 + sxy * g1 + szx * g2; out[a + 1] += sxy * g0 + syy * g1 + syz * g2; out[a + 2] += szx * g0 + syz * g1 + szz * g2;
      out[b] += sxx * g3 + sxy * g4 + szx * g5; out[b + 1] += sxy * g3 + syy * g4 + syz * g5; out[b + 2] += szx * g3 + syz * g4 + szz * g5;
      out[c] += sxx * g6 + sxy * g7 + szx * g8; out[c + 1] += sxy * g6 + syy * g7 + syz * g8; out[c + 2] += szx * g6 + syz * g7 + szz * g8;
      out[d] += sxx * g9 + sxy * g10 + szx * g11; out[d + 1] += sxy * g9 + syy * g10 + syz * g11; out[d + 2] += szx * g9 + syz * g10 + szz * g11;
    }
    const free = this.free;
    for (let i = 0; i < out.length; i++) out[i] *= free[i];
  }

  // out = K * x
  multiply(x, out) {
    if (this.smooth) { this._multiplySmooth(x, out); return; }
    out.fill(0);
    const { conn, G, vol, scale, ne, lambda, mu } = this;
    for (let e = 0; e < ne; e++) {
      const o = 12 * e, q = 4 * e;
      const a = 3 * conn[q], b = 3 * conn[q + 1], c = 3 * conn[q + 2], d = 3 * conn[q + 3];
      const ax = x[a], ay = x[a + 1], az = x[a + 2], bx = x[b], by = x[b + 1], bz = x[b + 2];
      const cx = x[c], cy = x[c + 1], cz = x[c + 2], dx = x[d], dy = x[d + 1], dz = x[d + 2];
      const g0 = G[o], g1 = G[o + 1], g2 = G[o + 2], g3 = G[o + 3], g4 = G[o + 4], g5 = G[o + 5];
      const g6 = G[o + 6], g7 = G[o + 7], g8 = G[o + 8], g9 = G[o + 9], g10 = G[o + 10], g11 = G[o + 11];
      // displacement gradient
      const hxx = ax * g0 + bx * g3 + cx * g6 + dx * g9, hxy = ax * g1 + bx * g4 + cx * g7 + dx * g10, hxz = ax * g2 + bx * g5 + cx * g8 + dx * g11;
      const hyx = ay * g0 + by * g3 + cy * g6 + dy * g9, hyy = ay * g1 + by * g4 + cy * g7 + dy * g10, hyz = ay * g2 + by * g5 + cy * g8 + dy * g11;
      const hzx = az * g0 + bz * g3 + cz * g6 + dz * g9, hzy = az * g1 + bz * g4 + cz * g7 + dz * g10, hzz = az * g2 + bz * g5 + cz * g8 + dz * g11;
      const s = scale[e] * vol[e], lt = lambda * (hxx + hyy + hzz) * s, m = mu * s;
      const sxx = lt + 2 * m * hxx, syy = lt + 2 * m * hyy, szz = lt + 2 * m * hzz;
      const sxy = m * (hxy + hyx), syz = m * (hyz + hzy), szx = m * (hzx + hxz);
      out[a] += sxx * g0 + sxy * g1 + szx * g2; out[a + 1] += sxy * g0 + syy * g1 + syz * g2; out[a + 2] += szx * g0 + syz * g1 + szz * g2;
      out[b] += sxx * g3 + sxy * g4 + szx * g5; out[b + 1] += sxy * g3 + syy * g4 + syz * g5; out[b + 2] += szx * g3 + syz * g4 + szz * g5;
      out[c] += sxx * g6 + sxy * g7 + szx * g8; out[c + 1] += sxy * g6 + syy * g7 + syz * g8; out[c + 2] += szx * g6 + syz * g7 + szz * g8;
      out[d] += sxx * g9 + sxy * g10 + szx * g11; out[d + 1] += sxy * g9 + syy * g10 + syz * g11; out[d + 2] += szx * g9 + syz * g10 + szz * g11;
    }
    const free = this.free;
    for (let i = 0; i < out.length; i++) out[i] *= free[i];
  }

  // Solves K u = f, starting from the current u. Returns iterations and final relative residual.
  solve(tol, maxIter, onProgress) {
    const { u, f, free, ndof, conn, G, vol, scale, ne, lambda, mu } = this;
    const r = this._r, z = this._z, p = this._p, q = this._q, diag = this._diag;
    this._prepare();
    // preconditioner: the diagonal of the plain (not smoothed) stiffness matrix
    diag.fill(0);
    for (let e = 0; e < ne; e++) {
      const s = scale[e] * vol[e];
      for (let n = 0; n < 4; n++) {
        const o = 12 * e + 3 * n, d = 3 * conn[4 * e + n];
        const gg = G[o] * G[o] + G[o + 1] * G[o + 1] + G[o + 2] * G[o + 2];
        for (let a = 0; a < 3; a++) diag[d + a] += s * ((lambda + mu) * G[o + a] * G[o + a] + mu * gg);
      }
    }
    let fnorm = 0;
    for (let d = 0; d < ndof; d++) fnorm += f[d] * f[d];
    fnorm = Math.sqrt(fnorm);
    if (fnorm === 0) { u.fill(0); return { iterations: 0, residual: 0 }; }
    if (this._nc) this._factorCoarse();
    this.multiply(u, q);
    let rz = 0;
    for (let d = 0; d < ndof; d++) r[d] = f[d] - q[d];
    this._precondition(r, z);
    for (let d = 0; d < ndof; d++) { p[d] = z[d]; rz += r[d] * z[d]; }
    const target = tol * fnorm;
    let it = 0, rn = Infinity;
    for (; it < maxIter; it++) {
      rn = 0;
      for (let d = 0; d < ndof; d++) rn += r[d] * r[d];
      rn = Math.sqrt(rn);
      if (rn <= target) break;
      if (onProgress && it % 25 === 0) onProgress(it, rn / fnorm);
      this.multiply(p, q);
      let pq = 0;
      for (let d = 0; d < ndof; d++) pq += p[d] * q[d];
      const alpha = rz / pq;
      let rzNew = 0;
      for (let d = 0; d < ndof; d++) { u[d] += alpha * p[d]; r[d] -= alpha * q[d]; }
      this._precondition(r, z);
      for (let d = 0; d < ndof; d++) rzNew += r[d] * z[d];
      const beta = rzNew / rz; rz = rzNew;
      for (let d = 0; d < ndof; d++) p[d] = z[d] + beta * p[d];
    }
    for (let d = 0; d < ndof; d++) u[d] *= free[d];
    return { iterations: it, residual: rn / fnorm };
  }

  // Strain of element e written to s as xx, yy, zz, xy, yz, zx (engineering shear).
  _strain(e, u, s) {
    const { conn, G } = this, o = 12 * e;
    s.fill(0);
    for (let n = 0; n < 4; n++) {
      const d = 3 * conn[4 * e + n], gx = G[o + 3 * n], gy = G[o + 3 * n + 1], gz = G[o + 3 * n + 2];
      s[0] += u[d] * gx; s[1] += u[d + 1] * gy; s[2] += u[d + 2] * gz;
      s[3] += u[d] * gy + u[d + 1] * gx; s[4] += u[d + 1] * gz + u[d + 2] * gy; s[5] += u[d + 2] * gx + u[d] * gz;
    }
  }

  // Turns a strain into the stress of full-strength material, in place.
  // Returns stress times strain (twice the strain energy density).
  _stress(s) {
    const { lambda, mu } = this, lt = lambda * (s[0] + s[1] + s[2]);
    const exx = s[0], eyy = s[1], ezz = s[2], gxy = s[3], gyz = s[4], gzx = s[5];
    s[0] = lt + 2 * mu * exx; s[1] = lt + 2 * mu * eyy; s[2] = lt + 2 * mu * ezz; s[3] = mu * gxy; s[4] = mu * gyz; s[5] = mu * gzx;
    return s[0] * exx + s[1] * eyy + s[2] * ezz + s[3] * gxy + s[4] * gyz + s[5] * gzx;
  }

  // Smoothed strain of every edge for displacement u, left in this._S.
  _smoothedEdges(u) {
    this._edgeStrain(u);
    const S = this._S, W = this.W;
    for (let k = 0, i = 0; k < this.nedge; k++, i += 6) { const f = 1 / W[k]; S[i] *= f; S[i + 1] *= f; S[i + 2] *= f; S[i + 3] *= f; S[i + 4] *= f; S[i + 5] *= f; }
  }

  // Strain energy per element for unit stiffness multiplier (joules times two),
  // so that the compliance is the sum of multiplier times this.
  elementEnergy(out, u = this.u) {
    const s = new Float64Array(6), { ne, vol, edge } = this;
    if (!this.smooth) { for (let e = 0; e < ne; e++) { this._strain(e, u, s); out[e] = this._stress(s) * vol[e]; } return; }
    this._smoothedEdges(u);
    const S = this._S, w = new Float64Array(this.nedge);
    for (let k = 0; k < this.nedge; k++) { for (let i = 0; i < 6; i++) s[i] = S[6 * k + i]; w[k] = this._stress(s); }
    for (let e = 0; e < ne; e++) {
      const q = 6 * e;
      out[e] = vol[e] / 6 * (w[edge[q]] + w[edge[q + 1]] + w[edge[q + 2]] + w[edge[q + 3]] + w[edge[q + 4]] + w[edge[q + 5]]);
    }
  }

  compliance(u = this.u) {
    let c = 0;
    for (let d = 0; d < this.ndof; d++) c += this.f[d] * u[d];
    return c; // joules; equals twice the stored strain energy
  }

  // Von Mises stress of each element, for full-strength material (Pa).
  // With smoothing on, the element strain is the mean of its six edge strains.
  vonMises(out, u = this.u) {
    const s = new Float64Array(6), { ne, edge } = this, S = this._S;
    if (this.smooth) this._smoothedEdges(u);
    for (let e = 0; e < ne; e++) {
      if (this.smooth) {
        s.fill(0);
        for (let k = 0; k < 6; k++) { const i = 6 * edge[6 * e + k]; for (let c = 0; c < 6; c++) s[c] += S[i + c] / 6; }
      } else this._strain(e, u, s);
      this._stress(s);
      out[e] = Math.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2) + 3 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]));
    }
  }

  maxDisplacement(u = this.u) {
    let m = 0;
    for (let d = 0; d < u.length; d += 3) {
      const v = u[d] * u[d] + u[d + 1] * u[d + 1] + u[d + 2] * u[d + 2];
      if (v > m) m = v;
    }
    return Math.sqrt(m);
  }
}

// Topology optimiser on the tetrahedral mesh: SIMP material model, sensitivity
// filter and optimality criteria update. Elements differ in size, so volume
// fractions, the filter and the update are all weighted by element volume.
// radius: filter radius in metres, measured between element centres.
export class TetOptimiser {
  constructor(model, { penal = 3, radius, emin = 1e-3 } = {}) {
    this.m = model; this.penal = penal; this.emin = emin;
    const ne = model.ne, { nodes, conn } = model;
    this.x = new Float64Array(ne).fill(1);
    this.dc = new Float64Array(ne); this.dcf = new Float64Array(ne); this.energy = new Float64Array(ne);
    // element centres, sorted into a hash grid with cells one radius wide
    const c = this.centre = new Float64Array(3 * ne), min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let e = 0; e < ne; e++) for (let a = 0; a < 3; a++) {
      const v = (nodes[3 * conn[4 * e] + a] + nodes[3 * conn[4 * e + 1] + a] + nodes[3 * conn[4 * e + 2] + a] + nodes[3 * conn[4 * e + 3] + a]) / 4;
      c[3 * e + a] = v; if (v < min[a]) min[a] = v; if (v > max[a]) max[a] = v;
    }
    if (!(radius > 0)) radius = 1.5 * Math.cbrt(6 * model.volume / ne); // about one and a half grid cells
    this.radius = radius;
    const dims = this.dims = min.map((v, a) => Math.max(1, Math.floor((max[a] - v) / radius) + 1));
    const cellOf = this.cellOf = new Int32Array(ne), start = this.start = new Int32Array(dims[0] * dims[1] * dims[2] + 1);
    for (let e = 0; e < ne; e++) {
      const i = Math.min(dims[0] - 1, Math.floor((c[3 * e] - min[0]) / radius)), j = Math.min(dims[1] - 1, Math.floor((c[3 * e + 1] - min[1]) / radius)), k = Math.min(dims[2] - 1, Math.floor((c[3 * e + 2] - min[2]) / radius));
      cellOf[e] = i + dims[0] * (j + dims[1] * k); start[cellOf[e] + 1]++;
    }
    for (let q = 0; q < start.length - 1; q++) start[q + 1] += start[q];
    const fill = start.slice(0, start.length - 1);
    this.list = new Int32Array(ne);
    for (let e = 0; e < ne; e++) this.list[fill[cellOf[e]]++] = e;
    // Up to a moderate size the neighbours and weights are stored once (about
    // 90 per element); larger meshes search the hash grid on every update to save memory.
    if (ne <= 150000) {
      const nb = [], w = [], first = new Int32Array(ne + 1);
      for (let e = 0; e < ne; e++) { this._near(e, (o, wt) => { nb.push(o); w.push(wt); }); first[e + 1] = nb.length; }
      this.nb = Int32Array.from(nb); this.w = Float32Array.from(w); this.first = first;
    }
  }

  // Calls visit(element, weight) for every element whose centre is within the filter radius of element e.
  _near(e, visit) {
    const { centre, radius, dims, start, list, cellOf } = this, r2 = radius * radius, sx = dims[0], sxy = dims[0] * dims[1];
    const cx = centre[3 * e], cy = centre[3 * e + 1], cz = centre[3 * e + 2];
    const cell = cellOf[e], ci = cell % sx, cj = ((cell / sx) | 0) % dims[1], ck = (cell / sxy) | 0;
    for (let k = Math.max(0, ck - 1); k <= Math.min(dims[2] - 1, ck + 1); k++) for (let j = Math.max(0, cj - 1); j <= Math.min(dims[1] - 1, cj + 1); j++) {
      const row = sx * j + sxy * k, c0 = row + Math.max(0, ci - 1), c1 = row + Math.min(sx - 1, ci + 1);
      for (let q = start[c0]; q < start[c1 + 1]; q++) {
        const o = list[q], ax = centre[3 * o] - cx, ay = centre[3 * o + 1] - cy, az = centre[3 * o + 2] - cz;
        const d2 = ax * ax + ay * ay + az * az;
        if (d2 < r2) visit(o, radius - Math.sqrt(d2));
      }
    }
  }

  applyDensities() {
    const { x, m, penal, emin } = this;
    for (let e = 0; e < m.ne; e++) m.scale[e] = emin + Math.pow(x[e], penal) * (1 - emin);
  }

  // One design update toward the target volume fraction. Returns the largest density change.
  update(volFrac, solveTol = 1e-3, solveIter = 600) {
    this.applyDensities();
    return this.finishUpdate(volFrac, this.m.solve(solveTol, solveIter));
  }

  // Sensitivities, filter and the optimality criteria step, after a solve of the model.
  finishUpdate(volFrac, info) {
    const { x, m, penal, emin, dc, dcf } = this, ne = m.ne, vol = m.vol;
    m.elementEnergy(this.energy);
    // dc = change of compliance with density; xdc = density times dc, the quantity that is filtered
    for (let e = 0; e < ne; e++) dc[e] = -penal * Math.pow(x[e], penal - 1) * (1 - emin) * this.energy[e] * x[e];
    let sum = 0, wsum = 0;
    const add = (o, w) => { sum += w * dc[o]; wsum += w * vol[o]; };
    for (let e = 0; e < ne; e++) {
      sum = 0; wsum = 0;
      if (this.nb) { const nb = this.nb, w = this.w; for (let q = this.first[e]; q < this.first[e + 1]; q++) { const o = nb[q]; sum += w[q] * dc[o]; wsum += w[q] * vol[o]; } }
      else this._near(e, add);
      dcf[e] = sum / (Math.max(1e-3, x[e]) * wsum); // per unit volume
    }
    // bisection on the Lagrange multiplier so the volume matches the target
    const move = 0.2, xnew = new Float64Array(ne), targetVol = volFrac * m.volume;
    let dmax = 0;
    for (let e = 0; e < ne; e++) if (-dcf[e] > dmax) dmax = -dcf[e];
    let hi = dmax * 10 + 1e-300, lo = hi * 1e-12, change = 0;
    for (let it = 0; it < 80 && (hi - lo) / (hi + lo) > 1e-4; it++) {
      const mid = 0.5 * (lo + hi);
      let v = 0;
      for (let e = 0; e < ne; e++) {
        let t = 1;
        if (!m.passive[e]) {
          t = x[e] * Math.sqrt(Math.max(0, -dcf[e]) / mid);
          t = Math.max(1e-3, Math.max(x[e] - move, Math.min(1, Math.min(x[e] + move, t))));
        }
        xnew[e] = t; v += t * vol[e];
      }
      if (v > targetVol) lo = mid; else hi = mid;
    }
    for (let e = 0; e < ne; e++) { change = Math.max(change, Math.abs(xnew[e] - x[e])); x[e] = xnew[e]; }
    return { change, compliance: m.compliance(), iterations: info.iterations };
  }

  // Fraction of the full volume that the current design uses.
  volume() { let v = 0; for (let e = 0; e < this.m.ne; e++) v += this.x[e] * this.m.vol[e]; return v / this.m.volume; }
}
