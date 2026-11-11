// Talks to the optional local OpenFOAM runner (server/flowbench_openfoam_server.py),
// which solves the current case with OpenFOAM on the user's own computer.
import { buildCase } from "./openfoam.js";
import { zipSync, strToU8 } from "three/addons/libs/fflate.module.js";

const BASE = "http://127.0.0.1:8765";

export async function runnerStatus() {
  try {
    const r = await fetch(BASE + "/status", { cache: "no-store" });
    return await r.json();
  } catch (e) { return null; }
}

// Sends the case and polls until it finishes. onStatus(text) is called as it goes.
// Returns {summary, results} or throws an Error with a readable message.
export async function solveWithOpenFoam(tris, opt, phys, token, onStatus, isCancelled) {
  const c = buildCase(tris, opt, phys, { functions: false, ascii: true, iterations: 1500 });
  const entries = {};
  for (const [path, data] of Object.entries(c.files)) entries[path] = typeof data === "string" ? strToU8(data) : data;
  onStatus("Sending the case to OpenFOAM...");
  const head = { "X-Flowbench-Token": token };
  let r;
  try { r = await fetch(BASE + "/run", { method: "POST", headers: head, body: zipSync(entries) }); }
  catch (e) { throw new Error("The OpenFOAM runner is not reachable. Start server/flowbench_openfoam_server.py on this computer first."); }
  const started = await r.json();
  if (!r.ok) throw new Error(r.status === 401 ? "The access code is wrong. Copy it from the window where the runner was started." : started.error || "the runner refused the case");
  const names = { blockMesh: "Building the background mesh", snappyHexMesh: "Fitting the mesh to the model surface", topoSet: "Marking the ports", createPatch: "Marking the ports", checkMesh: "Checking the mesh", simpleFoam: "Solving" };
  for (;;) {
    await new Promise((res) => setTimeout(res, 1000));
    if (isCancelled && isCancelled()) { fetch(BASE + "/job/" + started.job + "/cancel", { method: "POST", headers: head }).catch(() => {}); throw new Error("Stopped."); }
    const j = await (await fetch(BASE + "/job/" + started.job, { headers: head, cache: "no-store" })).json();
    if (j.state === "running") {
      onStatus("OpenFOAM: " + (names[j.stage] || j.stage) + (j.stage === "simpleFoam" ? ", iteration " + j.step + " of at most " + j.of : "") + (j.cells ? " (" + j.cells + " cells)" : ""));
      continue;
    }
    if (j.state !== "done") throw new Error(j.error || "the run failed");
    return { summary: c.summary, results: j.results, cells: j.cells };
  }
}

// Turns raw runner output into rows for display. Pressures and forces come back divided by density.
export function describe(run) {
  const { summary: s, results: r } = run, rho = s.rho, rows = [];
  const P = r.patches;
  if (s.mode === "external") {
    const f = P.model.pressureForceOverRho.map((v) => v * rho), q = 0.5 * rho * s.refSpeed ** 2 * s.refArea;
    rows.push(["Drag coefficient (pressure part)", (f[0] / q).toPrecision(3)], ["Lift coefficient (pressure part)", (f[2] / q).toPrecision(3)],
      ["Drag force (pressure part)", f[0].toPrecision(3) + " N"], ["Lift force (pressure part)", f[2].toPrecision(3) + " N"]);
  } else {
    let fed = 0, drawn = 0, out = 0;
    for (const p of s.patches) {
      const d = P[p.name];
      if (!d || p.kind === "wall") continue;
      const q = -d.flow; // OpenFOAM counts flow out of the domain as positive
      if (p.kind === "inlet") fed += q; else if (q > 0) drawn += q; else out -= q;
      rows.push([p.name + (p.kind === "inlet" ? " (fed)" : q >= 0 ? " (open, drawing in)" : " (open, blowing out)"),
        (Math.abs(q) * 60000).toPrecision(3) + " L/min, " + (d.pOverRho * rho).toPrecision(3) + " Pa"]);
    }
    if (s.mode === "ports") rows.unshift(["Amplification (air drawn in per air fed)", fed > 0 ? (drawn / fed).toPrecision(3) + " x" : "-"]);
    else {
      const inlet = s.patches.find((p) => p.kind === "inlet"), d = inlet && P[inlet.name];
      if (d) rows.unshift(["Static pressure drop, inlet to outlet", (d.pOverRho * rho).toPrecision(3) + " Pa"]);
    }
  }
  return rows;
}
