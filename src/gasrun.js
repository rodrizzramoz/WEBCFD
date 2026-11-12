// Adapter that lets the page drive the compressible gas solver through the same calls it
// uses for the incompressible one. Fields come back in real units, so the matching "lattice"
// conversion factors are simply 1 for speed and a fixed scale for pressure.
import { CompressibleSolver } from "./compressible.js";
import { RHO_AIR } from "./airjet.js";

export const P_AMB = 101325, T_AMB = 293.15;
// Gauge pressure is packed as density = 1 + 3 p / P_SCALE, the form the drawing code expects.
export const P_SCALE = 1e7;
export const GAS_REF_SPEED = 250;

// Total pressure that gives an ideal nozzle this exit speed (capped just below sonic).
export function totalPressureForSpeed(speed) {
  const cp = 1004.5, t = Math.max(T_AMB - speed * speed / (2 * cp), T_AMB / 1.2);
  return P_AMB * Math.pow(T_AMB / t, 3.5);
}

export function gasLattice(g, mu) {
  const longest = Math.max(g.nx, g.ny, g.nz);
  return {
    gas: true, uLat: GAS_REF_SPEED, dt: 0, vel: 1, pres: P_SCALE, force: 1,
    rampSteps: 6 * longest, reynolds: RHO_AIR * GAS_REF_SPEED * g.refLength / mu, mu,
  };
}

export class GasEngine {
  static memoryBytes(cells) { return CompressibleSolver.memoryBytes(cells); }
  static strideFor(cells) { return CompressibleSolver.strideFor(cells); }
  static ramp(step, rampSteps) { return CompressibleSolver.ramp(step, rampSteps); }

  static async create() {
    const e = new GasEngine();
    e.s = await CompressibleSolver.create();
    return e;
  }

  get maxCells() { return this.s.maxCells; }
  get lost() { return this.s.lost; }
  get steps() { return this.s.steps; }
  get time() { return this.s.time; }
  get stride() { return this.s.stride; }

  async setup(grid, lat) {
    this.grid = grid;
    const ports = grid.ports.map((p) => p.type === "feed"
      ? { type: "feed", totalPressure: p.pressure > 0 ? P_AMB + p.pressure : totalPressureForSpeed(p.speed), totalTemperature: T_AMB }
      : { type: "open", pressure: P_AMB, temperature: T_AMB });
    await this.s.setup(grid, { mu: lat.mu, ambientPressure: P_AMB, ambientTemperature: T_AMB, rampSteps: lat.rampSteps }, ports);
  }

  updateLattice() {}

  async step(k) {
    const done = await this.s.step(k);
    if (this.s.diverged) throw new Error("the gas solution blew up. Raise the grid detail or lower the supply pressure");
    return done;
  }

  async read() {
    const r = await this.s.read(), f = r.fields, n = f.length / 8;
    const macro = new Float32Array(4 * n), k = 3 / P_SCALE;
    for (let c = 0; c < n; c++) {
      const i = 8 * c, o = 4 * c, p = f[i + 4];
      // solid cells carry no pressure; keep them at ambient so they never trip the range check
      macro[o] = p > 0 ? 1 + k * (p - P_AMB) : 1;
      macro[o + 1] = f[i + 1]; macro[o + 2] = f[i + 2]; macro[o + 3] = f[i + 3];
    }
    return { macro, sample: null, force: r.force, gas: r };
  }

  dispose() { this.s.dispose(); }
}
