// Runs the finite element solves off the main thread so the page stays responsive.
// The page stops a run by terminating this worker and starting a fresh one.
//
// Large models (more than GPU_MIN_ELEMENTS elements) are solved on the graphics
// card through WebGPU when it is available; everything else, and any run where
// the GPU fails, uses the CPU solver. A "backend" message tells the page which
// solver is in use. It is sent once at the start, and once more only if the
// GPU fails part way through a run and the CPU takes over. The page shows it
// as: name (detail).
import { Model, Optimiser } from "./core.js";
import { GpuSolver } from "./gpu.js";

function percentile(values, keep, p) {
  const v = [];
  for (let e = 0; e < values.length; e++) if (keep(e)) v.push(values[e]);
  if (!v.length) return 0;
  v.sort((a, b) => a - b);
  return v[Math.min(v.length - 1, Math.floor(p * v.length))];
}

function snapshot(model, x) {
  const vm = new Float64Array(model.ne);
  model.vonMises(vm);
  const solid = (e) => !x || x[e] > 0.5;
  return {
    u: Float32Array.from(model.u),
    vm: Float32Array.from(vm),
    x: x ? Float32Array.from(x) : null,
    compliance: model.compliance(),
    maxDisp: model.maxDisplacement(),
    peakStress: percentile(vm, solid, 0.99),
  };
}

const GPU_MIN_ELEMENTS = 20000;

self.onmessage = async (ev) => {
  const msg = ev.data;
  const model = new Model(msg.setup);
  const progress = (label) => (it, res) => self.postMessage({ type: "progress", label, it, res });
  const backend = (name, detail) => self.postMessage({ type: "backend", name, detail });
  // Test hooks, never sent by the page: msg.debug = { gpuMinElements, noGpu, failGpuSolve }
  // (failGpuSolve = number of the GPU solve that should fail, counted from 1).
  const debug = msg.debug || {};
  const count = model.ne.toLocaleString("en-GB") + " elements";

  let gpu = null, gpuSolves = 0;
  const minElements = debug.gpuMinElements ?? GPU_MIN_ELEMENTS;
  if (model.ne <= minElements) {
    backend("CPU", count + ", small model");
  } else {
    try {
      if (debug.noGpu) throw new Error("the GPU was switched off for this run");
      gpu = await GpuSolver.create(model);
      backend("GPU", count + ", WebGPU" + (gpu.adapterName ? " on " + gpu.adapterName : "") +
        ", about " + Math.ceil(gpu.bytes / 1048576) + " MB of graphics memory");
    } catch (err) {
      gpu = null;
      backend("CPU", count + ", GPU solver not available: " + (err && err.message ? err.message : err) + "; using the slower CPU solver");
    }
  }

  // One linear solve from the current model.u, on the GPU when possible.
  // A GPU failure switches the rest of the run to the CPU solver.
  const solve = async (tol, maxIter, onProgress) => {
    if (gpu) {
      try {
        if (++gpuSolves === debug.failGpuSolve) throw new Error("test failure requested");
        if (onProgress) onProgress(0, 1);
        return await gpu.solve(model, tol, maxIter, onProgress);
      } catch (err) {
        try { gpu.destroy(); } catch (e) { /* already gone */ }
        gpu = null;
        // the displacement may hold a partial GPU result; keep it only if it is usable
        let sum = 0;
        for (let d = 0; d < model.ndof; d++) sum += model.u[d];
        if (!Number.isFinite(sum)) model.u.fill(0);
        backend("CPU", count + ", the GPU solver failed: " + (err && err.message ? err.message : err) + "; continuing on the slower CPU solver");
      }
    }
    return model.solve(tol, maxIter, onProgress);
  };

  if (msg.type === "analyse") {
    const info = await solve(1e-5, 12000, progress("Solving"));
    self.postMessage({ type: "analysis", ...snapshot(model, null), iterations: info.iterations, residual: info.residual });
    return;
  }

  if (msg.type === "optimise") {
    // Solid part first: the reference every lighter design is compared with.
    const info = await solve(1e-5, 12000, progress("Solving the solid part"));
    self.postMessage({ type: "stage", frac: 1, ...snapshot(model, null), iterations: info.iterations });
    const opt = new Optimiser(model);
    for (const frac of msg.fractions) {
      let change = 1;
      for (let it = 0; it < msg.maxIter && change > 0.03; it++) {
        opt.applyDensities();
        // On the GPU path a single solve of a large model can take a while, so it
        // reports progress; the CPU path stays silent here as it always has.
        const note = gpu ? progress("Shaping the design with " + Math.round(frac * 100) + " % of the material, pass " + (it + 1)) : undefined;
        const r = opt.finishUpdate(frac, await solve(1e-3, 600, note));
        change = r.change;
        self.postMessage({ type: "iter", frac, it: it + 1, change, compliance: r.compliance, x: Float32Array.from(opt.x) });
      }
      opt.applyDensities();
      await solve(1e-4, 4000, progress("Checking the design at " + Math.round(frac * 100) + " % material"));
      self.postMessage({ type: "stage", frac, ...snapshot(model, opt.x) });
    }
    self.postMessage({ type: "done" });
  }
};
