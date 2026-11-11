// WebGPU solver for the structural model in core.js: solves K u = f on the
// graphics card so that fine grids (hundreds of thousands to a few million
// brick elements) stay practical.
//
// How it works
// 1. Matrix-free product, one shader invocation per grid node. A node looks at
//    its (up to) 8 neighbouring cells; for every cell with a non-zero stiffness
//    multiplier it adds scale * KE(rows of this node, all 24 columns) * u_cell.
//    The 27 surrounding node values are read once. Cell data lives on the full
//    grid (multiplier 0 = empty cell) so all indexing is implicit.
// 2. Jacobi preconditioned conjugate gradients in 32-bit floats. The dot
//    products are reduced on the GPU (chunks of 128, level by level, down to a
//    small scalar buffer) and alpha and beta are computed there too, so a whole
//    batch of iterations runs without talking to the CPU. One 32 byte readback
//    per batch gives the residual for the stopping test and for progress.
// 3. Mixed precision refinement. 32-bit conjugate gradients drift: the
//    residual carried by the recurrence keeps falling while the true residual
//    stalls (around 1e-2 of the starting value on a slender cantilever). So
//    the GPU never solves for u directly. The CPU computes the true residual
//    r = f - K u in 64-bit (Model.residual), the GPU solves K d = r
//    approximately from d = 0, and the CPU adds d to u. This repeats until the
//    true residual meets the tolerance or stops improving. A warm start is
//    simply a non-zero u on entry. The price is one CPU product per pass and a
//    restart of the iteration, so a tight tolerance costs two to four passes.
//
// Memory on the GPU
//   per grid node: 100 bytes = five vec4f fields (u, r, p, q, inverse diagonal,
//                  16 bytes each), a 4 byte fixed flag and a 16 byte staging
//                  copy used to read the displacement back
//   per grid cell: 4 bytes (stiffness multiplier)
//   plus about 0.2 bytes per node for the partial sums of the reductions.
// The CPU side keeps one extra Float32Array of 16 bytes per node for uploads.
// The largest single buffer is 16 bytes per node; it must fit within the
// device limits maxBufferSize and maxStorageBufferBindingSize (checked below).

const CHUNK = 128;        // values summed by one invocation of a reduction pass
const FINAL_MAX = 1024;   // partial sums left for the last, single invocation pass

// Corner code (x + 2 y + 4 z) to local node number of core.js and back
// (the permutation is its own inverse).
const PERM = "const PERM = array<u32, 8>(0u, 1u, 3u, 2u, 4u, 5u, 7u, 6u);";

const HEAD = /* wgsl */ `
struct Grid { nx: u32, ny: u32, nz: u32, pad: u32 }
@group(0) @binding(0) var<uniform> g: Grid;
fn nodeIndex(id: vec3u) -> u32 { return id.x + (g.nx + 1u) * (id.y + (g.ny + 1u) * id.z); }
fn outside(id: vec3u) -> bool { return id.x > g.nx || id.y > g.ny || id.z > g.nz; }
`;

// q = K x, and q.w = x . q for the dot product that follows.
const WGSL_MATVEC = HEAD + PERM + /* wgsl */ `
@group(0) @binding(1) var<storage, read> ke: array<vec4f>;      // 64 blocks of 3 columns
@group(0) @binding(2) var<storage, read> scale: array<f32>;     // per cell
@group(0) @binding(3) var<storage, read> dinv: array<vec4f>;    // w = 1 on free nodes
@group(0) @binding(4) var<storage, read> x: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> q: array<vec4f>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (outside(id)) { return; }
  let n = nodeIndex(id);
  if (dinv[n].w == 0.0) { q[n] = vec4f(0.0); return; }
  let top = vec3i(i32(g.nx), i32(g.ny), i32(g.nz));
  let me = vec3i(id);
  var xs: array<vec3f, 27>;
  for (var c = 0; c < 27; c++) {
    let o = clamp(me + vec3i(c % 3, (c / 3) % 3, c / 9) - 1, vec3i(0), top);
    xs[c] = x[nodeIndex(vec3u(o))].xyz;
  }
  var acc = vec3f(0.0);
  for (var c = 0u; c < 8u; c++) {
    let off = vec3i(i32(c & 1u), i32((c >> 1u) & 1u), i32(c >> 2u));
    let cell = me + off - 1;
    if (any(cell < vec3i(0)) || any(cell >= top)) { continue; }
    let s = scale[u32(cell.x) + g.nx * (u32(cell.y) + g.ny * u32(cell.z))];
    if (s == 0.0) { continue; }
    // this node is the corner opposite to off in that cell
    let row = PERM[7u - c] * 8u;
    var v = vec3f(0.0);
    for (var m = 0u; m < 8u; m++) {
      let xv = xs[u32(off.x) + (m & 1u) + 3u * (u32(off.y) + ((m >> 1u) & 1u)) + 9u * (u32(off.z) + (m >> 2u))];
      let o = (row + PERM[m]) * 3u;
      v += ke[o].xyz * xv.x + ke[o + 1u].xyz * xv.y + ke[o + 2u].xyz * xv.z;
    }
    acc += s * v;
  }
  q[n] = vec4f(acc, dot(acc, xs[13]));
}
`;

// Inverse diagonal of K (the Jacobi preconditioner) and the free-node mask.
const WGSL_DIAG = HEAD + PERM + /* wgsl */ `
@group(0) @binding(1) var<storage, read> ke: array<vec4f>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read> fixedFlag: array<u32>;
@group(0) @binding(4) var<storage, read_write> dinv: array<vec4f>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (outside(id)) { return; }
  let n = nodeIndex(id);
  let top = vec3i(i32(g.nx), i32(g.ny), i32(g.nz));
  var d = vec3f(0.0);
  for (var c = 0u; c < 8u; c++) {
    let cell = vec3i(id) + vec3i(i32(c & 1u), i32((c >> 1u) & 1u), i32(c >> 2u)) - 1;
    if (any(cell < vec3i(0)) || any(cell >= top)) { continue; }
    let s = scale[u32(cell.x) + g.nx * (u32(cell.y) + g.ny * u32(cell.z))];
    let o = PERM[7u - c] * 27u;
    d += s * vec3f(ke[o].x, ke[o + 1u].y, ke[o + 2u].z);
  }
  if (fixedFlag[n] != 0u || !(d.x > 0.0)) { dinv[n] = vec4f(0.0); } else { dinv[n] = vec4f(1.0 / d, 1.0); }
}
`;

// Start of a solve of K d = r from d = 0: r holds the right-hand side.
const WGSL_INIT = HEAD + /* wgsl */ `
@group(0) @binding(1) var<storage, read> dinv: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> r: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> p: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> u: array<vec4f>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (outside(id)) { return; }
  let n = nodeIndex(id);
  let d = dinv[n];
  let rv = r[n].xyz * d.w;
  let z = rv * d.xyz;
  r[n] = vec4f(rv, dot(rv, z));
  p[n] = vec4f(z, 0.0);
  u[n] = vec4f(0.0);
}
`;

const SCAL = /* wgsl */ `
// 0 rz, 1 pq, 2 alpha, 3 beta, 4 rr, 5 to 7 unused
@group(0) @binding(1) var<storage, read_write> scal: array<f32, 8>;
`;

// u += alpha p, r -= alpha q, and r.w = r . z for the reduction that follows.
const WGSL_UPDATE = HEAD + /* wgsl */ `
@group(0) @binding(1) var<storage, read> scal: array<f32, 8>;
@group(0) @binding(2) var<storage, read> dinv: array<vec4f>;
@group(0) @binding(3) var<storage, read> p: array<vec4f>;
@group(0) @binding(4) var<storage, read> q: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> u: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> r: array<vec4f>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (outside(id)) { return; }
  let n = nodeIndex(id);
  let alpha = scal[2];
  u[n] = vec4f(u[n].xyz + alpha * p[n].xyz, 0.0);
  let rv = r[n].xyz - alpha * q[n].xyz;
  r[n] = vec4f(rv, dot(rv, rv * dinv[n].xyz));
}
`;

// p = z + beta p
const WGSL_DIRECTION = HEAD + /* wgsl */ `
@group(0) @binding(1) var<storage, read> scal: array<f32, 8>;
@group(0) @binding(2) var<storage, read> dinv: array<vec4f>;
@group(0) @binding(3) var<storage, read> r: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> p: array<vec4f>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (outside(id)) { return; }
  let n = nodeIndex(id);
  p[n] = vec4f(r[n].xyz * dinv[n].xyz + scal[3] * p[n].xyz, 0.0);
}
`;

// Reduction, first level: sums over chunks of nodes. x = sum of w, y = sum of |xyz|^2.
const WGSL_REDUCE_NODES = /* wgsl */ `
struct Info { count: u32, mode: u32, a: u32, b: u32 }
@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let first = id.x * ${CHUNK}u;
  if (first >= info.count) { return; }
  let last = min(first + ${CHUNK}u, info.count);
  var s = vec2f(0.0);
  for (var i = first; i < last; i++) { let v = src[i]; s += vec2f(v.w, dot(v.xyz, v.xyz)); }
  dst[id.x] = s;
}
`;

// Reduction, middle levels: sums over chunks of partial sums.
const WGSL_REDUCE_PART = /* wgsl */ `
struct Info { count: u32, mode: u32, a: u32, b: u32 }
@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> src: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let first = id.x * ${CHUNK}u;
  if (first >= info.count) { return; }
  let last = min(first + ${CHUNK}u, info.count);
  var s = vec2f(0.0);
  for (var i = first; i < last; i++) { s += src[i]; }
  dst[id.x] = s;
}
`;

// Reduction, last level: one invocation finishes the sum and updates the scalars.
// mode 0: start of a solve (rz, rr). mode 1: after the product (pq, alpha).
// mode 2: after the update (new rz, rr, beta).
const WGSL_REDUCE_FINAL = /* wgsl */ `
struct Info { count: u32, mode: u32, a: u32, b: u32 }
@group(0) @binding(0) var<uniform> info: Info;
${SCAL}
@group(0) @binding(2) var<storage, read> src: array<vec2f>;

@compute @workgroup_size(1)
fn main() {
  var s = vec2f(0.0);
  for (var i = 0u; i < info.count; i++) { s += src[i]; }
  if (info.mode == 0u) {
    scal[0] = s.x; scal[4] = s.y;
  } else if (info.mode == 1u) {
    scal[1] = s.x;
    if (s.x > 0.0) { scal[2] = scal[0] / s.x; } else { scal[2] = 0.0; }
  } else {
    let old = scal[0];
    if (old > 0.0) { scal[3] = s.x / old; } else { scal[3] = 0.0; }
    scal[0] = s.x; scal[4] = s.y;
  }
}
`;

function blocksOfKE(KE) {
  // 64 node-to-node blocks, each stored as three padded columns.
  const out = new Float32Array(64 * 3 * 4);
  for (let a = 0; a < 8; a++) for (let b = 0; b < 8; b++) for (let col = 0; col < 3; col++) {
    const o = ((a * 8 + b) * 3 + col) * 4;
    for (let row = 0; row < 3; row++) out[o + row] = KE[(3 * a + row) * 24 + 3 * b + col];
  }
  return out;
}

export class GpuSolver {
  // Builds the solver for one model (grid, fixed nodes and Poisson ratio are
  // taken from it once). Throws an Error with a readable message on failure.
  static async create(model) {
    if (typeof navigator === "undefined" || !navigator.gpu) throw new Error("WebGPU is not available here");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU graphics adapter was found");
    const lim = adapter.limits;
    const { nx, ny, nz } = model;
    const nn = (nx + 1) * (ny + 1) * (nz + 1), nc = nx * ny * nz;
    const biggest = Math.min(lim.maxBufferSize, lim.maxStorageBufferBindingSize);
    if (nn * 16 > biggest) {
      throw new Error("the grid is too large for this graphics card: " + nn + " nodes need buffers of " +
        Math.ceil(nn * 16 / 1048576) + " MB but the device allows " + Math.floor(biggest / 1048576) +
        " MB (about " + Math.floor(biggest / 16) + " nodes)");
    }
    const per = lim.maxComputeWorkgroupsPerDimension;
    if (Math.ceil((Math.max(nx, ny, nz) + 1) / 4) > per || Math.ceil(nn / CHUNK / 64) > per) {
      throw new Error("the grid is too large for this graphics card (too many compute workgroups)");
    }
    const device = await adapter.requestDevice({
      requiredLimits: { maxBufferSize: lim.maxBufferSize, maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize },
    });
    const s = new GpuSolver();
    s.device = device; s.lost = null;
    device.lost.then((info) => { s.lost = (info && info.message) || "device lost"; });
    s.adapterName = "";
    try { const i = adapter.info; if (i) s.adapterName = [i.vendor, i.architecture || i.device].filter(Boolean).join(" "); } catch (e) { /* optional */ }
    s.nn = nn; s.nc = nc;
    s.patience = Math.max(300, 4 * (nx + ny + nz)); // iterations without progress before a pass gives up
    s.floor = 1e-4; // relative reduction asked of one 32-bit pass, see solve()
    s.bytes = nn * 100 + nc * 4;
    device.pushErrorScope("validation");
    device.pushErrorScope("out-of-memory");
    s._build(model);
    const oom = await device.popErrorScope(), bad = await device.popErrorScope();
    if (oom || bad) {
      s.destroy();
      throw new Error(oom ? "the graphics card ran out of memory for this grid (" + Math.ceil(s.bytes / 1048576) + " MB needed)"
        : "GPU setup failed: " + bad.message);
    }
    return s;
  }

  _build(model) {
    const d = this.device, { nn, nc } = this, { nx, ny, nz } = model;
    const U = GPUBufferUsage;
    this.buffers = [];
    const mk = (size, usage) => { const b = d.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage }); this.buffers.push(b); return b; };
    const node = (extra = 0) => mk(nn * 16, U.STORAGE | extra);
    this.u = node(U.COPY_SRC); this.r = node(U.COPY_DST); this.p = node(); this.q = node(); this.dinv = node();
    this.fixedBuf = mk(nn * 4, U.STORAGE | U.COPY_DST);
    this.scaleBuf = mk(nc * 4, U.STORAGE | U.COPY_DST);
    this.keBuf = mk(64 * 3 * 16, U.STORAGE | U.COPY_DST);
    this.scal = mk(32, U.STORAGE | U.COPY_SRC);
    this.scalStage = mk(32, U.MAP_READ | U.COPY_DST);
    this.uStage = mk(nn * 16, U.MAP_READ | U.COPY_DST);
    this.gridBuf = mk(16, U.UNIFORM | U.COPY_DST);
    d.queue.writeBuffer(this.gridBuf, 0, new Uint32Array([nx, ny, nz, 0]));
    d.queue.writeBuffer(this.keBuf, 0, blocksOfKE(model.KE));
    // a node is fixed when none of its degrees of freedom is free in the model
    const fixed = new Uint32Array(nn);
    for (let n = 0; n < nn; n++) fixed[n] = model.free[3 * n] ? 0 : 1;
    d.queue.writeBuffer(this.fixedBuf, 0, fixed);
    this.cpu = new Float32Array(nn * 4);   // upload scratch
    this.scaleCells = new Float32Array(nc);

    const pipe = (code) => d.createComputePipeline({ layout: "auto", compute: { module: d.createShaderModule({ code }), entryPoint: "main" } });
    const bind = (pipeline, bufs) => d.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const G = this.gridBuf;
    this.work = [Math.ceil((nx + 1) / 4), Math.ceil((ny + 1) / 4), Math.ceil((nz + 1) / 4)];
    const node3 = (pipeline, bufs) => ({ pipeline, group: bind(pipeline, bufs), work: this.work });
    this.passDiag = node3(pipe(WGSL_DIAG), [G, this.keBuf, this.scaleBuf, this.fixedBuf, this.dinv]);
    this.passMatvec = node3(pipe(WGSL_MATVEC), [G, this.keBuf, this.scaleBuf, this.dinv, this.p, this.q]);
    this.passInit = node3(pipe(WGSL_INIT), [G, this.dinv, this.r, this.p, this.u]);
    this.passUpdate = node3(pipe(WGSL_UPDATE), [G, this.scal, this.dinv, this.p, this.q, this.u, this.r]);
    this.passDirection = node3(pipe(WGSL_DIRECTION), [G, this.scal, this.dinv, this.r, this.p]);

    // reduction chains: nodes -> partial sums -> ... -> scalars
    const info = (count, mode) => { const b = mk(16, U.UNIFORM | U.COPY_DST); d.queue.writeBuffer(b, 0, new Uint32Array([count, mode, 0, 0])); return b; };
    const pNodes = pipe(WGSL_REDUCE_NODES), pPart = pipe(WGSL_REDUCE_PART), pFinal = pipe(WGSL_REDUCE_FINAL);
    const m1 = Math.ceil(nn / CHUNK);
    const partA = mk(m1 * 8, U.STORAGE), partB = mk(Math.ceil(m1 / CHUNK) * 8, U.STORAGE);
    const chain = (src, mode) => {
      const steps = [{ pipeline: pNodes, group: bind(pNodes, [info(nn, 0), src, partA]), work: [Math.ceil(m1 / 64), 1, 1] }];
      let count = m1, from = partA, to = partB;
      while (count > FINAL_MAX) {
        const next = Math.ceil(count / CHUNK);
        steps.push({ pipeline: pPart, group: bind(pPart, [info(count, 0), from, to]), work: [Math.ceil(next / 64), 1, 1] });
        count = next; [from, to] = [to, from];
      }
      steps.push({ pipeline: pFinal, group: bind(pFinal, [info(count, mode), this.scal, from]), work: [1, 1, 1] });
      return steps;
    };
    this.chainStart = chain(this.r, 0);
    this.chainPq = chain(this.q, 1);
    this.chainRz = chain(this.r, 2);
    this.iteration = [this.passMatvec, ...this.chainPq, this.passUpdate, ...this.chainRz, this.passDirection];
    this.dispatchesPerIteration = this.iteration.length;
  }

  _run(pass, steps) {
    for (const s of steps) { pass.setPipeline(s.pipeline); pass.setBindGroup(0, s.group); pass.dispatchWorkgroups(s.work[0], s.work[1], s.work[2]); }
  }

  _check() { if (this.lost) throw new Error("the graphics device was lost (" + this.lost + ")"); }

  async _scalars() {
    await this.scalStage.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(this.scalStage.getMappedRange().slice(0));
    this.scalStage.unmap();
    return v;
  }

  // Sends the element stiffness multipliers and rebuilds the preconditioner.
  _setScale(model) {
    const cells = this.scaleCells, { scale, cells: index, ne } = model;
    for (let e = 0; e < ne; e++) cells[index[e]] = scale[e];
    const d = this.device;
    d.queue.writeBuffer(this.scaleBuf, 0, cells);
    const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
    this._run(pass, [this.passDiag]);
    pass.end();
    d.queue.submit([enc.finish()]);
  }

  // Starts a solve of K d = rhs from d = 0. The right-hand side (4 floats per
  // node) is in this.cpu. Returns the scalars (index 4 = r . r).
  async _load() {
    const d = this.device;
    d.queue.writeBuffer(this.r, 0, this.cpu);
    const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
    this._run(pass, [this.passInit, ...this.chainStart]);
    pass.end();
    enc.copyBufferToBuffer(this.scal, 0, this.scalStage, 0, 32);
    d.queue.submit([enc.finish()]);
    return this._scalars();
  }

  // Runs n conjugate gradient iterations without leaving the GPU, then reads the scalars.
  async _batch(n) {
    const d = this.device, enc = d.createCommandEncoder(), pass = enc.beginComputePass();
    for (let i = 0; i < n; i++) this._run(pass, this.iteration);
    pass.end();
    enc.copyBufferToBuffer(this.scal, 0, this.scalStage, 0, 32);
    d.queue.submit([enc.finish()]);
    return this._scalars();
  }

  // Reads the correction d back and adds c * d to the 64-bit displacement.
  async _collect(u, c) {
    const d = this.device, nn = this.nn, enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.u, 0, this.uStage, 0, nn * 16);
    d.queue.submit([enc.finish()]);
    await this.uStage.mapAsync(GPUMapMode.READ);
    const dv = new Float32Array(this.uStage.getMappedRange());
    let sum = 0;
    for (let o = 0; o < dv.length; o += 4) sum += dv[o] + dv[o + 1] + dv[o + 2];
    const ok = Number.isFinite(sum);
    if (ok) for (let n = 0, i = 0; n < nn; n++, i += 3) { const o = 4 * n; u[i] += c * dv[o]; u[i + 1] += c * dv[o + 1]; u[i + 2] += c * dv[o + 2]; }
    this.uStage.unmap();
    if (!ok) throw new Error("the GPU solve produced a non-finite displacement");
  }

  // Solves K u = f for the model, starting from model.u (warm start) and
  // leaving the answer there. Same contract as Model.solve, but asynchronous:
  // returns { iterations, residual } with the true relative residual achieved.
  //
  // Each pass of the loop below is one 32-bit conjugate gradient solve on the
  // GPU for a correction d to the current 64-bit u. A pass ends when its
  // recurrence residual is small enough, or has stopped improving. The true
  // residual after a pass is usually much larger than the recurrence says
  // (32-bit drift); the ratio seen is remembered in this.floor so that later
  // passes and later solves do not iterate beyond what 32 bits can deliver.
  async solve(model, tol, maxIter, onProgress) {
    this._check();
    const d = this.device, { nn, cpu } = this, { u, ndof } = model, res = model._r;
    d.pushErrorScope("validation");
    d.pushErrorScope("out-of-memory");
    let total = 0, rel = 0, prev = Infinity, weak = 0;
    this.trace = []; // one entry per pass, for diagnostics
    try {
      this._setScale(model);
      for (;;) {
        rel = model.residual(res);
        if (!Number.isFinite(rel)) throw new Error("the residual is not finite");
        const last = this.trace[this.trace.length - 1];
        if (last) {
          // learn the reduction one pass really achieves (true residual after / before)
          last.achieved = rel / last.start;
          if (last.achieved > 3 * last.recurrence) this.floor = Math.max(1e-6, Math.min(1e-2, 0.1 * last.achieved));
        }
        if (!(rel > tol) || total >= maxIter || this.trace.length >= 12) break;
        // stop when two passes in a row fail to cut the true residual by a third
        if (rel > 0.67 * prev) { if (++weak >= 2) break; } else weak = 0;
        prev = rel;
        // right-hand side of unit norm, so every pass works at the same scale
        let c = 0;
        for (let i = 0; i < ndof; i++) c += res[i] * res[i];
        c = Math.sqrt(c);
        for (let n = 0, i = 0; n < nn; n++, i += 3) { const o = 4 * n; cpu[o] = res[i] / c; cpu[o + 1] = res[i + 1] / c; cpu[o + 2] = res[i + 2] / c; }
        const goal = Math.min(0.1, Math.max(0.5 * tol / rel, this.floor));
        let sc = await this._load();
        let it = 0, batch = 4, est = Math.sqrt(sc[4]), best = est, bestIt = 0, stalled = false;
        for (;;) {
          this._check();
          if (!Number.isFinite(est) || !Number.isFinite(sc[0])) throw new Error("the GPU solve produced a non-finite residual");
          if (est < 0.9 * best) { best = est; bestIt = it; }
          if (est <= goal || total + it >= maxIter) break;
          // Stall test. The residual of this iteration normally climbs above its
          // starting value for a long time (roughly three iterations per cell
          // along the part) before it drops steeply, so the test only applies
          // once the drop is under way.
          if (best < 0.1 && it - bestIt >= this.patience) { stalled = true; break; }
          const n = Math.min(batch, maxIter - total - it), t0 = performance.now(), before = est;
          sc = await this._batch(n);
          it += n; est = Math.sqrt(sc[4]);
          if (onProgress) onProgress(total + it, rel * est);
          // aim for a readback (and a progress message) about every 0.2 seconds
          const ms = Math.max(1, performance.now() - t0);
          batch = Math.max(1, Math.min(32, 2 * batch, Math.round(n * 200 / ms)));
          // near the goal, shorten the batch so few iterations run past it
          const rate = Math.log(before / est) / n;
          if (rate > 0 && est > goal) batch = Math.max(2, Math.min(batch, Math.ceil(0.75 * Math.log(est / goal) / rate)));
        }
        total += it;
        await this._collect(u, c);
        this.trace.push({ start: rel, goal, iterations: it, recurrence: est, stalled });
      }
    } finally {
      const oom = await d.popErrorScope().catch(() => null), bad = await d.popErrorScope().catch(() => null);
      if (oom || bad) throw new Error(oom ? "the graphics card ran out of memory" : "GPU error: " + bad.message);
    }
    this._check();
    return { iterations: total, residual: rel };
  }

  destroy() {
    for (const b of this.buffers || []) b.destroy();
    this.buffers = [];
    if (this.device) this.device.destroy();
  }
}
