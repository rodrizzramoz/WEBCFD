// Runs the finite element solves off the main thread so the page stays responsive.
// The page stops a run by terminating this worker and starting a fresh one.
import { Model, Optimiser } from "./core.js";

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

self.onmessage = (ev) => {
  const msg = ev.data;
  const model = new Model(msg.setup);
  const progress = (label) => (it, res) => self.postMessage({ type: "progress", label, it, res });

  if (msg.type === "analyse") {
    const info = model.solve(1e-5, 12000, progress("Solving"));
    self.postMessage({ type: "analysis", ...snapshot(model, null), iterations: info.iterations, residual: info.residual });
    return;
  }

  if (msg.type === "optimise") {
    // Solid part first: the reference every lighter design is compared with.
    const info = model.solve(1e-5, 12000, progress("Solving the solid part"));
    self.postMessage({ type: "stage", frac: 1, ...snapshot(model, null), iterations: info.iterations });
    const opt = new Optimiser(model);
    for (const frac of msg.fractions) {
      let change = 1;
      for (let it = 0; it < msg.maxIter && change > 0.03; it++) {
        const r = opt.update(frac);
        change = r.change;
        self.postMessage({ type: "iter", frac, it: it + 1, change, compliance: r.compliance, x: Float32Array.from(opt.x) });
      }
      opt.applyDensities();
      model.solve(1e-4, 4000, progress("Checking the design at " + Math.round(frac * 100) + " % material"));
      self.postMessage({ type: "stage", frac, ...snapshot(model, opt.x) });
    }
    self.postMessage({ type: "done" });
  }
};
