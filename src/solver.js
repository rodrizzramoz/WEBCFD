// 3D lattice Boltzmann solver (D3Q19, regularised BGK collision with a Smagorinsky
// sub-grid model) running as a WebGPU compute shader.

const FORCE_SCALE = 131072; // fixed-point scale for the on-GPU force sum

const WGSL = /* wgsl */ `
struct Params { dims: vec4u, uin: vec4f, misc: vec4f, ports: array<vec4f, 8> };
@group(0) @binding(0) var<uniform> P: Params;
// The 19 populations of each cell are split over two buffers (0 to 9 and 10 to 18),
// so that very large grids stay under the per-buffer size limit of the GPU.
@group(0) @binding(1) var<storage, read> srcA: array<f32>;
@group(0) @binding(2) var<storage, read> srcB: array<f32>;
@group(0) @binding(3) var<storage, read_write> dstA: array<f32>;
@group(0) @binding(4) var<storage, read_write> dstB: array<f32>;
@group(0) @binding(5) var<storage, read> flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> fields: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> forceAcc: array<atomic<i32>, 4>;
@group(0) @binding(8) var<storage, read> wallQ: array<u32>;

fn fin(q: u32, n: u32, i: u32) -> f32 {
  if (q < 10u) { return srcA[q * n + i]; }
  return srcB[(q - 10u) * n + i];
}
fn fout(q: u32, n: u32, i: u32, v: f32) {
  if (q < 10u) { dstA[q * n + i] = v; } else { dstB[(q - 10u) * n + i] = v; }
}

const C = array<vec3i, 19>(
  vec3i(0, 0, 0),
  vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1),
  vec3i(1, 1, 0), vec3i(-1, -1, 0), vec3i(1, -1, 0), vec3i(-1, 1, 0),
  vec3i(1, 0, 1), vec3i(-1, 0, -1), vec3i(1, 0, -1), vec3i(-1, 0, 1),
  vec3i(0, 1, 1), vec3i(0, -1, -1), vec3i(0, 1, -1), vec3i(0, -1, 1));
const NRM = array<vec3i, 6>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1));

fn wq(q: u32) -> f32 {
  if (q == 0u) { return 1.0 / 3.0; }
  if (q < 7u) { return 1.0 / 18.0; }
  return 1.0 / 36.0;
}
fn cellIndex(p: vec3i) -> u32 { return u32(p.x) + P.dims.x * (u32(p.y) + P.dims.y * u32(p.z)); }
fn feq(q: u32, rho: f32, u: vec3f) -> f32 {
  let cu = dot(vec3f(C[q]), u);
  return wq(q) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * dot(u, u));
}
fn writeEq(i: u32, n: u32, rho: f32, u: vec3f) {
  for (var q = 0u; q < 19u; q++) { fout(q, n, i, feq(q, rho, u)); }
}

@compute @workgroup_size(4, 4, 4)
fn init(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= P.dims.xyz)) { return; }
  let n = P.dims.x * P.dims.y * P.dims.z;
  let i = id.x + P.dims.x * (id.y + P.dims.y * id.z);
  let t = flags[i] & 7u;
  var u = vec3f(0.0);
  if (t != 1u && P.misc.z > 0.5) { u = P.uin.xyz; }
  writeEq(i, n, 1.0, u);
  fields[i] = vec4f(1.0, u);
}

@compute @workgroup_size(4, 4, 4)
fn step(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= P.dims.xyz)) { return; }
  let n = P.dims.x * P.dims.y * P.dims.z;
  let i = id.x + P.dims.x * (id.y + P.dims.y * id.z);
  let flag = flags[i];
  let t = flag & 7u;
  if (t == 1u) { return; }
  let pos = vec3i(id);

  if (t >= 2u) {
    // Boundary cell: hold an equilibrium state built from the prescribed
    // values and whatever is extrapolated from the cell just inside.
    var rho = 1.0;
    var u = P.uin.xyz;
    // Inlet cells carry a port number in bits 6 to 8; each port has its own speed along its face normal.
    if (t == 3u) { u = vec3f(NRM[(flag >> 3u) & 7u]) * P.ports[(flag >> 6u) & 7u].x; }
    if (t != 2u) {
      let j = cellIndex(pos + NRM[(flag >> 3u) & 7u]);
      var rn = 1.0;
      var un = vec3f(0.0);
      if ((flags[j] & 7u) != 1u) {
        var r = 0.0;
        var m = vec3f(0.0);
        for (var q = 0u; q < 19u; q++) { let v = fin(q, n, j); r += v; m += v * vec3f(C[q]); }
        rn = r;
        un = m / r;
      }
      if (t == 3u) { rho = rn; } else {
        u = un;
        let s = length(u);
        if (s > 0.25) { u *= 0.25 / s; }
      }
    }
    writeEq(i, n, rho, u);
    fields[i] = vec4f(rho, u);
    return;
  }

  // Fluid cell: pull populations from the neighbours, bouncing back off walls.
  var f: array<f32, 19>;
  f[0] = fin(0u, n, i);
  var force = vec3f(0.0);
  // Bits 6 and up of the flag point at this cell's wall table entry (0 = none):
  // one byte per link giving how far along the link the true surface lies.
  let wi = flag >> 6u;
  for (var q = 1u; q < 19u; q++) {
    let j = cellIndex(pos - C[q]);
    if ((flags[j] & 7u) == 1u) {
      let o = q + 1u - 2u * ((q + 1u) % 2u);
      let vo = fin(o, n, i);
      var back = vo; // half-way bounce-back: the wall sits midway between the two cell centres
      if (wi > 0u) {
        let k = q - 1u;
        let b = (wallQ[(wi - 1u) * 5u + (k >> 2u)] >> (8u * (k & 3u))) & 255u;
        if (b > 0u) {
          // interpolated bounce-back at the true wall position d (fraction of the link)
          let d = f32(b) / 255.0;
          if (d < 0.5) {
            let j2 = cellIndex(pos + C[q]);
            if ((flags[j2] & 7u) != 1u) { back = 2.0 * d * vo + (1.0 - 2.0 * d) * fin(o, n, j2); }
          } else {
            back = (vo + (2.0 * d - 1.0) * fin(q, n, i)) / (2.0 * d);
          }
        }
      }
      f[q] = back;
      // Momentum handed to the wall: what left plus what came back. The rest-state
      // part (weight only) cancels over a closed surface, so leave it out for precision.
      force += ((vo - wq(o)) + (back - wq(o))) * vec3f(C[o]);
    } else {
      f[q] = fin(q, n, j);
    }
  }
  var rho = 0.0;
  var mom = vec3f(0.0);
  for (var q = 0u; q < 19u; q++) { rho += f[q]; mom += f[q] * vec3f(C[q]); }
  let u = mom / rho;

  var e: array<f32, 19>;
  var pxx = 0.0; var pyy = 0.0; var pzz = 0.0; var pxy = 0.0; var pxz = 0.0; var pyz = 0.0;
  for (var q = 0u; q < 19u; q++) {
    e[q] = feq(q, rho, u);
    let d = f[q] - e[q];
    let c = vec3f(C[q]);
    pxx += d * c.x * c.x; pyy += d * c.y * c.y; pzz += d * c.z * c.z;
    pxy += d * c.x * c.y; pxz += d * c.x * c.z; pyz += d * c.y * c.z;
  }
  let qn = sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz));
  let tau = P.uin.w;
  let omega = 2.0 / (tau + sqrt(tau * tau + P.misc.x * qn / rho));
  // Regularised collision: keep only the stress part of the non-equilibrium
  // populations. This removes the step-to-step ringing plain BGK shows at low viscosity.
  let tr = (pxx + pyy + pzz) / 3.0;
  for (var q = 0u; q < 19u; q++) {
    let c = vec3f(C[q]);
    let qp = c.x * c.x * pxx + c.y * c.y * pyy + c.z * c.z * pzz
      + 2.0 * (c.x * c.y * pxy + c.x * c.z * pxz + c.y * c.z * pyz) - tr;
    fout(q, n, i, e[q] + (1.0 - omega) * wq(q) * 4.5 * qp);
  }
  fields[i] = vec4f(rho, u);

  if (dot(force, force) > 0.0) {
    let fi = vec3i(round(force * P.misc.y));
    atomicAdd(&forceAcc[0], fi.x);
    atomicAdd(&forceAcc[1], fi.y);
    atomicAdd(&forceAcc[2], fi.z);
  }
}
`;

// Very large grids are too big to copy back whole. These two passes make a
// coarse copy of the flow for drawing (each view cell is the average of a
// block of real cells) and pick out the exact cells needed for the readouts.
const VIEW_WGSL = /* wgsl */ `
struct VP { dims: vec4u, vdims: vec4u }; // dims.w = block size, vdims.w = number of picked cells
@group(0) @binding(0) var<uniform> V: VP;
@group(0) @binding(1) var<storage, read> fields: array<vec4f>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> view: array<vec4f>;
@group(0) @binding(4) var<storage, read> pick: array<u32>;
@group(0) @binding(5) var<storage, read_write> picked: array<vec4f>;

@compute @workgroup_size(4, 4, 4)
fn shrink(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= V.vdims.xyz)) { return; }
  let s = V.dims.w;
  var sum = vec4f(0.0);
  var count = 0.0;
  for (var dz = 0u; dz < s; dz++) { for (var dy = 0u; dy < s; dy++) { for (var dx = 0u; dx < s; dx++) {
    let p = id * s + vec3u(dx, dy, dz);
    if (any(p >= V.dims.xyz)) { continue; }
    let c = p.x + V.dims.x * (p.y + V.dims.y * p.z);
    if ((flags[c] & 7u) != 1u) { sum += fields[c]; count += 1.0; }
  } } }
  let vi = id.x + V.vdims.x * (id.y + V.vdims.y * id.z);
  if (count > 0.0) { view[vi] = sum / count; } else { view[vi] = vec4f(1.0, 0.0, 0.0, 0.0); }
}

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= V.vdims.w) { return; }
  picked[id.x] = fields[pick[id.x]];
}
`;

export class Solver {
  // Grids with more cells than this are drawn from a coarser copy.
  static viewCells = 6e6;

  static async create() {
    if (!navigator.gpu) throw new Error("This browser does not support WebGPU.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU graphics adapter was found.");
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: lim.maxBufferSize,
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      },
    });
    const s = new Solver();
    s.device = device;
    // the largest single buffer holds 10 populations (40 bytes) per cell
    s.maxCells = Math.floor(Math.min(lim.maxBufferSize, lim.maxStorageBufferBindingSize) / 40);
    s.lost = false;
    device.lost.then(() => { s.lost = true; });
    const module = device.createShaderModule({ code: WGSL });
    const st = (type) => ({ buffer: { type } });
    s.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, ...st("uniform") },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, ...st("read-only-storage") },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, ...st("read-only-storage") },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, ...st("storage") },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, ...st("storage") },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, ...st("read-only-storage") },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, ...st("storage") },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, ...st("storage") },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, ...st("read-only-storage") },
      ],
    });
    const vmod = device.createShaderModule({ code: VIEW_WGSL });
    s.viewLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, ...st("uniform") },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, ...st("read-only-storage") },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, ...st("read-only-storage") },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, ...st("storage") },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, ...st("read-only-storage") },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, ...st("storage") },
      ],
    });
    const vpl = device.createPipelineLayout({ bindGroupLayouts: [s.viewLayout] });
    s.shrinkPipe = await device.createComputePipelineAsync({ layout: vpl, compute: { module: vmod, entryPoint: "shrink" } });
    s.gatherPipe = await device.createComputePipelineAsync({ layout: vpl, compute: { module: vmod, entryPoint: "gather" } });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [s.layout] });
    s.initPipe = await device.createComputePipelineAsync({ layout: pl, compute: { module, entryPoint: "init" } });
    s.stepPipe = await device.createComputePipelineAsync({ layout: pl, compute: { module, entryPoint: "step" } });
    s.buffers = [];
    return s;
  }

  // Smooth start-up factor for the inlet speed. A gentle ramp avoids setting
  // off pressure waves that slosh between the inlet and the outlet.
  static ramp(step, rampSteps) {
    if (!(rampSteps > 0) || step >= rampSteps) return 1;
    return 0.5 - 0.5 * Math.cos(Math.PI * step / rampSteps);
  }

  // Block size of the coarse drawing copy for a grid of this many cells (1 = full detail).
  static strideFor(cells) { return cells > Solver.viewCells ? Math.ceil(Math.cbrt(cells / Solver.viewCells) - 1e-9) : 1; }

  static memoryBytes(cells) { return cells * (76 * 2 + 16 + 4 + (Solver.strideFor(cells) > 1 ? 1 : 16)); }

  // grid: {nx, ny, nz, flags}; lat: {uin: [x, y, z], tau, smag, initWithFlow}
  // pickCells: cells whose exact values the readouts need (used only when the grid is drawn from a coarse copy)
  async setup(grid, lat, pickCells) {
    this.dispose();
    const d = this.device;
    d.pushErrorScope("validation");
    d.pushErrorScope("out-of-memory");
    const n = grid.nx * grid.ny * grid.nz;
    this.grid = grid; this.n = n; this.lat = lat;
    const S = GPUBufferUsage.STORAGE, CS = GPUBufferUsage.COPY_SRC, CD = GPUBufferUsage.COPY_DST;
    const mk = (size, usage) => { const b = d.createBuffer({ size, usage }); this.buffers.push(b); return b; };
    const pops = [mk(n * 40, S), mk(n * 36, S), mk(n * 40, S), mk(n * 36, S)]; // set one (two halves), set two (two halves)
    this.flagBuf = mk(n * 4, S | CD);
    this.macroBuf = mk(n * 16, S | CS);
    this.forceBuf = mk(16, S | CS | CD);
    this.paramBuf = mk(176, GPUBufferUsage.UNIFORM | CD);
    this.forceStage = mk(16, GPUBufferUsage.MAP_READ | CD);
    const MR = GPUBufferUsage.MAP_READ | CD;
    this.stride = Solver.strideFor(n);
    if (this.stride === 1) this.macroStage = mk(n * 16, MR);
    else {
      const sd = this.stride, vd = [Math.ceil(grid.nx / sd), Math.ceil(grid.ny / sd), Math.ceil(grid.nz / sd)];
      const picks = pickCells && pickCells.length ? pickCells : new Uint32Array(1);
      this.viewDims = vd; this.viewN = vd[0] * vd[1] * vd[2]; this.pickN = pickCells ? pickCells.length : 0;
      this.viewBuf = mk(this.viewN * 16, S | CS);
      this.viewStage = mk(this.viewN * 16, MR);
      this.pickBuf = mk(picks.length * 4, S | CD);
      this.pickedBuf = mk(picks.length * 16, S | CS);
      this.pickedStage = mk(picks.length * 16, MR);
      this.viewParam = mk(32, GPUBufferUsage.UNIFORM | CD);
      d.queue.writeBuffer(this.pickBuf, 0, picks);
      d.queue.writeBuffer(this.viewParam, 0, new Uint32Array([grid.nx, grid.ny, grid.nz, sd, vd[0], vd[1], vd[2], this.pickN]));
      this.viewGroup = d.createBindGroup({
        layout: this.viewLayout,
        entries: [this.viewParam, this.macroBuf, this.flagBuf, this.viewBuf, this.pickBuf, this.pickedBuf].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    }
    d.queue.writeBuffer(this.flagBuf, 0, grid.flags);
    const wallQ = grid.wallQ && grid.wallQ.length ? grid.wallQ : new Uint32Array(5);
    this.wallBuf = mk(wallQ.byteLength, S | CD);
    d.queue.writeBuffer(this.wallBuf, 0, wallQ);
    const group = (from, to) => d.createBindGroup({
      layout: this.layout,
      entries: [this.paramBuf, pops[from], pops[from + 1], pops[to], pops[to + 1], this.flagBuf, this.macroBuf, this.forceBuf, this.wallBuf]
        .map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.groups = [group(0, 2), group(2, 0)];
    this.work = [Math.ceil(grid.nx / 4), Math.ceil(grid.ny / 4), Math.ceil(grid.nz / 4)];
    this.reset();
    const oom = await d.popErrorScope(), bad = await d.popErrorScope();
    if (oom || bad) {
      this.dispose();
      throw new Error("The GPU could not allocate this grid. Lower the grid detail.");
    }
  }

  // Swaps in new lattice parameters (relaxation time tau, inlet velocity, turbulence constant)
  // without reallocating anything, so a run carries on from its current flow state.
  // They take effect on the next step, because step() writes them to the GPU each time.
  updateLattice(lat) {
    this.lat = lat;
  }

  writeParams(ramp) {
    const g = this.grid, l = this.lat;
    const buf = new ArrayBuffer(176);
    new Uint32Array(buf, 0, 4).set([g.nx, g.ny, g.nz, 0]);
    new Float32Array(buf, 16, 8).set([
      l.uin[0] * ramp, l.uin[1] * ramp, l.uin[2] * ramp, l.tau,
      l.smag, FORCE_SCALE, l.initWithFlow ? 1 : 0, 0,
    ]);
    // lattice speed of each inlet port (up to 8), one vec4 each
    const ports = new Float32Array(buf, 48, 32);
    (l.portU || []).slice(0, 8).forEach((u, i) => { ports[4 * i] = u * ramp; });
    this.device.queue.writeBuffer(this.paramBuf, 0, buf);
  }

  reset() {
    this.steps = 0; this.forceSteps = 0; this.phase = 0; this.forceSum = [0, 0, 0];
    this.writeParams(1);
    const enc = this.device.createCommandEncoder();
    for (const g of this.groups) {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.initPipe);
      pass.setBindGroup(0, g);
      pass.dispatchWorkgroups(...this.work);
      pass.end();
    }
    enc.clearBuffer(this.forceBuf);
    this.device.queue.submit([enc.finish()]);
  }

  // Runs k time steps and resolves when the GPU has finished them.
  async step(k) {
    k = Math.max(2, k - (k % 2)); // even counts, so force sums cover whole two-step cycles
    const ramp = Solver.ramp(this.steps + k, this.lat.rampSteps);
    this.writeParams(ramp);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.stepPipe);
    for (let s = 0; s < k; s++) {
      pass.setBindGroup(0, this.groups[this.phase]);
      pass.dispatchWorkgroups(...this.work);
      this.phase ^= 1;
    }
    pass.end();
    // Pull the wall force summed over these steps and zero the counter.
    enc.copyBufferToBuffer(this.forceBuf, 0, this.forceStage, 0, 16);
    enc.clearBuffer(this.forceBuf);
    this.device.queue.submit([enc.finish()]);
    this.steps += k; this.forceSteps += k;
    await this.forceStage.mapAsync(GPUMapMode.READ);
    const fi = new Int32Array(this.forceStage.getMappedRange());
    for (let a = 0; a < 3; a++) this.forceSum[a] += fi[a] / FORCE_SCALE;
    this.forceStage.unmap();
    return k;
  }

  // Copies density and velocity back to the CPU (every cell, or a coarse copy plus
  // the picked cells on very large grids), plus the
  // wall force averaged over the steps since the previous call (lattice units).
  async read() {
    const d = this.device, enc = d.createCommandEncoder();
    let macro, sample = null;
    if (this.stride === 1) {
      enc.copyBufferToBuffer(this.macroBuf, 0, this.macroStage, 0, this.n * 16);
      d.queue.submit([enc.finish()]);
      await this.macroStage.mapAsync(GPUMapMode.READ);
      macro = new Float32Array(this.macroStage.getMappedRange().slice(0));
      this.macroStage.unmap();
    } else {
      // coarse copy for drawing, exact values for the cells the readouts need
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, this.viewGroup);
      pass.setPipeline(this.shrinkPipe);
      pass.dispatchWorkgroups(Math.ceil(this.viewDims[0] / 4), Math.ceil(this.viewDims[1] / 4), Math.ceil(this.viewDims[2] / 4));
      if (this.pickN) { pass.setPipeline(this.gatherPipe); pass.dispatchWorkgroups(Math.ceil(this.pickN / 64)); }
      pass.end();
      enc.copyBufferToBuffer(this.viewBuf, 0, this.viewStage, 0, this.viewN * 16);
      if (this.pickN) enc.copyBufferToBuffer(this.pickedBuf, 0, this.pickedStage, 0, this.pickN * 16);
      d.queue.submit([enc.finish()]);
      await Promise.all([this.viewStage.mapAsync(GPUMapMode.READ), this.pickN ? this.pickedStage.mapAsync(GPUMapMode.READ) : null]);
      macro = new Float32Array(this.viewStage.getMappedRange().slice(0));
      this.viewStage.unmap();
      if (this.pickN) { sample = new Float32Array(this.pickedStage.getMappedRange().slice(0)); this.pickedStage.unmap(); }
    }
    const steps = Math.max(1, this.forceSteps);
    const force = this.forceSum.map((v) => v / steps);
    this.forceSum = [0, 0, 0]; this.forceSteps = 0;
    return { macro, sample, force };
  }

  dispose() {
    for (const b of this.buffers) b.destroy();
    this.buffers = [];
  }
}
