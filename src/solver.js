// 3D lattice Boltzmann solver (D3Q19, regularised BGK collision with a Smagorinsky
// sub-grid model) running as a WebGPU compute shader.

const FORCE_SCALE = 131072; // fixed-point scale for the on-GPU force sum

const WGSL = /* wgsl */ `
struct Params { dims: vec4u, uin: vec4f, misc: vec4f };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> fields: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> forceAcc: array<atomic<i32>, 4>;

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
  for (var q = 0u; q < 19u; q++) { dst[q * n + i] = feq(q, rho, u); }
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
    if (t != 2u) {
      let j = cellIndex(pos + NRM[(flag >> 3u) & 7u]);
      var rn = 1.0;
      var un = vec3f(0.0);
      if ((flags[j] & 7u) != 1u) {
        var r = 0.0;
        var m = vec3f(0.0);
        for (var q = 0u; q < 19u; q++) { let v = src[q * n + j]; r += v; m += v * vec3f(C[q]); }
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
  f[0] = src[i];
  var force = vec3f(0.0);
  for (var q = 1u; q < 19u; q++) {
    let j = cellIndex(pos - C[q]);
    if ((flags[j] & 7u) == 1u) {
      let o = q + 1u - 2u * ((q + 1u) % 2u);
      let v = src[o * n + i];
      f[q] = v;
      // The rest-state part (weight only) cancels over a closed surface, so leave it
      // out to keep precision for the small net force.
      force += 2.0 * (v - wq(o)) * vec3f(C[o]);
    } else {
      f[q] = src[q * n + j];
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
    dst[q * n + i] = e[q] + (1.0 - omega) * wq(q) * 4.5 * qp;
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

export class Solver {
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
    s.maxCells = Math.floor(Math.min(lim.maxBufferSize, lim.maxStorageBufferBindingSize) / 76);
    s.lost = false;
    device.lost.then(() => { s.lost = true; });
    const module = device.createShaderModule({ code: WGSL });
    const st = (type) => ({ buffer: { type } });
    s.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, ...st("uniform") },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, ...st("read-only-storage") },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, ...st("storage") },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, ...st("read-only-storage") },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, ...st("storage") },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, ...st("storage") },
      ],
    });
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

  static memoryBytes(cells) { return cells * (76 * 2 + 16 * 2 + 4); }

  // grid: {nx, ny, nz, flags}; lat: {uin: [x, y, z], tau, smag, initWithFlow}
  async setup(grid, lat) {
    this.dispose();
    const d = this.device;
    d.pushErrorScope("validation");
    d.pushErrorScope("out-of-memory");
    const n = grid.nx * grid.ny * grid.nz;
    this.grid = grid; this.n = n; this.lat = lat;
    const S = GPUBufferUsage.STORAGE, CS = GPUBufferUsage.COPY_SRC, CD = GPUBufferUsage.COPY_DST;
    const mk = (size, usage) => { const b = d.createBuffer({ size, usage }); this.buffers.push(b); return b; };
    this.fA = mk(n * 76, S);
    this.fB = mk(n * 76, S);
    this.flagBuf = mk(n * 4, S | CD);
    this.macroBuf = mk(n * 16, S | CS);
    this.forceBuf = mk(16, S | CS | CD);
    this.paramBuf = mk(48, GPUBufferUsage.UNIFORM | CD);
    this.macroStage = mk(n * 16, GPUBufferUsage.MAP_READ | CD);
    this.forceStage = mk(16, GPUBufferUsage.MAP_READ | CD);
    d.queue.writeBuffer(this.flagBuf, 0, grid.flags);
    const group = (a, b) => d.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.paramBuf } },
        { binding: 1, resource: { buffer: a } },
        { binding: 2, resource: { buffer: b } },
        { binding: 3, resource: { buffer: this.flagBuf } },
        { binding: 4, resource: { buffer: this.macroBuf } },
        { binding: 5, resource: { buffer: this.forceBuf } },
      ],
    });
    this.groups = [group(this.fA, this.fB), group(this.fB, this.fA)];
    this.work = [Math.ceil(grid.nx / 4), Math.ceil(grid.ny / 4), Math.ceil(grid.nz / 4)];
    this.reset();
    const oom = await d.popErrorScope(), bad = await d.popErrorScope();
    if (oom || bad) {
      this.dispose();
      throw new Error("The GPU could not allocate this grid. Lower the grid detail.");
    }
  }

  writeParams(ramp) {
    const g = this.grid, l = this.lat;
    const buf = new ArrayBuffer(48);
    new Uint32Array(buf, 0, 4).set([g.nx, g.ny, g.nz, 0]);
    new Float32Array(buf, 16, 8).set([
      l.uin[0] * ramp, l.uin[1] * ramp, l.uin[2] * ramp, l.tau,
      l.smag, FORCE_SCALE, l.initWithFlow ? 1 : 0, 0,
    ]);
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

  // Copies density and velocity for every cell back to the CPU, plus the
  // wall force averaged over the steps since the previous call (lattice units).
  async read() {
    const d = this.device, enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.macroBuf, 0, this.macroStage, 0, this.n * 16);
    d.queue.submit([enc.finish()]);
    await this.macroStage.mapAsync(GPUMapMode.READ);
    const macro = new Float32Array(this.macroStage.getMappedRange().slice(0));
    this.macroStage.unmap();
    const steps = Math.max(1, this.forceSteps);
    const force = this.forceSum.map((v) => v / steps);
    this.forceSum = [0, 0, 0]; this.forceSteps = 0;
    return { macro, force };
  }

  dispose() {
    for (const b of this.buffers) b.destroy();
    this.buffers = [];
  }
}
