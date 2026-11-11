import "./style.css";
import * as THREE from "three";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { Viewer } from "./viewer.js";
import { Solver } from "./solver.js";
import { DEMOS } from "./demos.js";
import { buildDomain, domainBox, FACE_NAMES, NORMALS } from "./domain.js";
import { fieldStats, sliceImage, streamlines, makeSeeds, surfaceColors, internalMetrics, probe, seedCluster, LUT_CSS } from "./results.js";

const $ = (id) => document.getElementById(id);
const FLUIDS = {
  air: { rho: 1.204, mu: 1.825e-5 },
  water: { rho: 998.2, mu: 1.002e-3 },
  oil: { rho: 870, mu: 0.04 },
};

const MAX_SPEED = 1000; // m/s, far beyond where treating the fluid as incompressible makes sense

// ---------- State ----------
const state = {
  name: "ahmed", demo: "ahmed", unit: 0.001, mode: "external", fluid: "air", rho: 1.204, mu: 1.825e-5,
  speed: 40, res: 128, domainScale: 1, inletFace: 0, outletFace: 1, stlIsFluid: false,
  sliceAxis: 1, slicePos: 0.5, field: "speed", yaw: 0,
};
let source = null;            // geometry as loaded, in file units
const rot = new THREE.Matrix4();
let baked = null;             // geometry in metres, as simulated
let tris = null;
let solver = null, grid = null, lat = null, seeds = null;
let prepared = false, running = false, runToken = 0, loopPromise = null, busy = false;
let last = null, avgField = null;
let ranges = { sp: 0, pLo: 0, pHi: 0 };
let history = [];
let smooth = null;
let speedLimit = 1;          // lowered automatically when the flow runs too fast for the solver
let rate = { steps: 0, t: performance.now(), value: 0 };
let setupVersion = 0;        // goes up on every setup change, to catch changes made while a run is being prepared
let timeBase = 0, stepBase = 0; // simulated time carries on across a live speed change, which alters the time step
let smoke = [];              // smoke wand points in metres, each one becomes a small ring of streamline seeds
let sliceShown = null;       // {axis, idx} of the slice on screen, for the hover probe
let pointer = null;          // last pointer position over the 3D view
const TL_MAX = 120;
const tl = { frames: [], view: -1, playing: false, timer: 0 }; // view -1 means live

const viewer = new Viewer($("view"));
for (const b of document.querySelectorAll(".legend .bar")) b.style.background = LUT_CSS;

// ---------- Helpers ----------
function fmt(v, digits = 3) {
  if (!isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e5 || a < 1e-3) return v.toExponential(digits - 1).replace("e+", "e");
  return String(Number(v.toPrecision(digits)));
}
function fmtBig(v) { return !isFinite(v) ? "-" : v >= 1e9 ? fmt(v) : v >= 1e6 ? (v / 1e6).toFixed(2) + " M" : v >= 1e3 ? (v / 1e3).toFixed(1) + " k" : String(Math.round(v)); }
function press(container, attr, value) {
  for (const b of container.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.getAttribute(attr) === String(value)));
}
let noticeTimer = 0;
function notify(msg, sticky) {
  const n = $("notice");
  n.textContent = msg; n.classList.add("show");
  clearTimeout(noticeTimer);
  if (!sticky) noticeTimer = setTimeout(() => n.classList.remove("show"), 9000);
}
function clearNotice() { $("notice").classList.remove("show"); }
function download(name, href) {
  const a = document.createElement("a");
  a.href = href; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}
function downloadText(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  download(name, url);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// ---------- Model ----------
function setSource(geometry, name) {
  const g = geometry.index ? geometry.toNonIndexed() : geometry;
  g.deleteAttribute("uv");
  source = g;
  rot.identity();
  state.name = name;
  state.yaw = 0;
  $("yaw").value = "0"; $("yawOut").textContent = "0";
  bake(true);
}

function bake(refit) {
  const g = source.clone();
  g.applyMatrix4(rot);
  if (state.mode === "external" && state.yaw) g.rotateZ(state.yaw * Math.PI / 180);
  g.scale(state.unit, state.unit, state.unit);
  g.computeVertexNormals();
  g.computeBoundingBox();
  baked = g;
  tris = g.getAttribute("position").array;
  viewer.setModel(g);
  if (refit) viewer.fit();
  smoke = [];
  const s = g.boundingBox.getSize(new THREE.Vector3());
  $("modelInfo").textContent = state.name + ": " + fmtBig(tris.length / 9) + " triangles, " +
    [s.x, s.y, s.z].map((v) => fmt(v * 1000)).join(" x ") + " mm. You can also drop an STL onto the 3D view.";
  invalidate();
}

function loadDemo(key) {
  const d = DEMOS[key];
  Object.assign(state, d.setup, { demo: key });
  state.unit = 0.001;
  const f = FLUIDS[state.fluid];
  state.rho = f.rho; state.mu = f.mu;
  syncInputs();
  setSource(d.build(), d.label);
}

// Counts triangle edges that are not shared by exactly two triangles. A closed
// (watertight) mesh has none. Returns -1 when the mesh is too large to check quickly.
function openEdges(positions) {
  const nTri = positions.length / 9;
  if (nTri > 1.5e6) return -1;
  // Points closer together than a hundred-thousandth of the model size count as the same point,
  // so rounding in the exporting program does not show up as gaps.
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < positions.length; i++) { const v = positions[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  const q = 1e5 / Math.max(hi - lo, 1e-30);
  const ids = new Map(), vid = new Uint32Array(nTri * 3);
  for (let v = 0; v < nTri * 3; v++) {
    const key = Math.round(positions[3 * v] * q) + "," + Math.round(positions[3 * v + 1] * q) + "," + Math.round(positions[3 * v + 2] * q);
    let id = ids.get(key);
    if (id === undefined) { id = ids.size; ids.set(key, id); }
    vid[v] = id;
  }
  const N = ids.size, edges = new Map();
  for (let t = 0; t < nTri; t++) {
    for (let e = 0; e < 3; e++) {
      const a = vid[3 * t + e], b = vid[3 * t + (e + 1) % 3];
      if (a === b) continue;
      const key = a < b ? a * N + b : b * N + a;
      edges.set(key, (edges.get(key) || 0) + 1);
    }
  }
  let open = 0;
  for (const c of edges.values()) if (c === 1) open++;
  return open;
}

// Throws a readable error unless the mesh has real, finite extent.
function checkGeometry(geo) {
  const attr = geo.getAttribute("position");
  if (!attr || attr.count < 3) throw new Error("it contains no triangles");
  const a = attr.array, min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < a.length; i++) {
    const v = a[i], k = i % 3;
    if (!isFinite(v)) throw new Error("it contains coordinates that are not numbers");
    if (v < min[k]) min[k] = v;
    if (v > max[k]) max[k] = v;
  }
  const size = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  if (!(size > 0)) throw new Error("all of its points are in the same place");
  if (Math.max(Math.abs(min[0]), Math.abs(min[1]), Math.abs(min[2]), Math.abs(max[0]), Math.abs(max[1]), Math.abs(max[2])) > 1e12) throw new Error("its coordinates are too large");
}

async function loadFile(file) {
  let geo;
  try {
    if (!file.size) throw new Error("the file is empty");
    $("status").textContent = "Reading " + file.name + "...";
    const buf = await file.arrayBuffer();
    await new Promise((r) => setTimeout(r, 20)); // let the status line show before a long parse
    try { geo = new STLLoader().parse(buf); }
    catch (e) { throw new Error("it is not a valid binary or ASCII STL"); }
    checkGeometry(geo);
  } catch (e) {
    notify("That file could not be used as an STL: " + e.message + ". Export it again as a binary or ASCII STL. The model that was open is unchanged.");
    $("status").textContent = "File not loaded.";
    return;
  }
  try {
    setSource(geo, file.name);
    state.demo = "";
    $("demo").value = "";
    const gaps = openEdges(source.getAttribute("position").array);
    if (gaps > 0) {
      notify("Loaded " + file.name + ", but its surface is not closed: " + fmtBig(gaps) + " open edges were found. The solver needs a watertight STL, so the shape it builds may be wrong. Repair the mesh if results look odd.", true);
    } else notify("Loaded " + file.name + ". Check the units and the flow direction, then press Run.");
  } catch (e) {
    notify("That STL was read but could not be set up: " + e.message, true);
  }
}

// ---------- Setup ----------
function options() {
  return {
    mode: state.mode, res: state.res, domainScale: state.domainScale,
    inletFace: state.inletFace, outletFace: state.outletFace, stlIsFluid: state.stlIsFluid,
  };
}

function latticeFor(g) {
  const nu = state.mu / state.rho, U = state.speed, dx = g.dx;
  let uLat = Math.min(0.07, 0.11 / g.speedUp) * speedLimit;
  let nuLat = nu * uLat / (U * dx);
  if (nuLat > 0.3) { uLat *= 0.3 / nuLat; nuLat = 0.3; } // very viscous flow: shorten the time step instead
  const dt = uLat * dx / U;
  const longest = Math.max(g.nx, g.ny, g.nz);
  return {
    uLat, dt, nuLat, tau: 3 * nuLat + 0.5, smag: 18 * Math.SQRT2 * 0.12 * 0.12,
    uin: g.inVec.map((c) => c * uLat),
    initWithFlow: g.mode === "external",
    rampSteps: g.mode === "external" ? 0 : 20 * longest,
    vel: U / uLat,                          // lattice speed to m/s
    pres: state.rho * (U / uLat) ** 2,      // lattice pressure to Pa
    force: state.rho * dx ** 4 / dt ** 2,   // lattice force to N
    reynolds: U * g.refLength / nu,
  };
}

// Any change to the setup stops the run and shows the new domain box.
function invalidate() {
  running = false; runToken++; prepared = false; speedLimit = 1; setupVersion++;
  last = null; avgField = null; history = []; smooth = null;
  $("runBtn").textContent = "Run";
  viewer.hideSlice(); viewer.setStreamlines(null); viewer.setSurfaceColors(null);
  sliceShown = null; tlClear(); refreshProbe();
  if (!tris) return;
  const box = domainBox(tris, options());
  viewer.setDomain(box, state.mode, state.inletFace, state.outletFace);
  const cells = box.dims[0] * box.dims[1] * box.dims[2];
  showFacts(box.dims, box.dx, cells, null);
  renderTiles();
  drawChart();
  $("status").textContent = "Setup changed. Press Run to solve.";
}

function showFacts(dims, dx, cells, l) {
  const rows = [
    ["Grid", dims.join(" x ")],
    ["Cells", fmtBig(cells)],
    ["Cell size", fmt(dx * 1000) + " mm"],
    ["GPU memory", fmt(Solver.memoryBytes(cells) / 1048576) + " MB"],
  ];
  if (l) rows.push(["Time step", fmt(l.dt * 1e6) + " us"], ["Reynolds number", fmtBig(l.reynolds)]);
  $("facts").innerHTML = rows.map((r) => "<dt>" + r[0] + "</dt><dd>" + r[1] + "</dd>").join("");
}

async function prepare() {
  if (!solver) throw new Error("This browser has no WebGPU, so the solver cannot run. Use a current Chrome, Edge or Firefox on a computer with a graphics card.");
  if (!(state.speed > 0) || !(state.rho > 0) || !(state.mu > 0)) throw new Error("Speed, density and viscosity must all be above zero.");
  if (state.speed > MAX_SPEED) throw new Error("Speeds above " + MAX_SPEED + " m/s are outside what this solver can represent. Lower the speed.");
  if (state.mode === "internal" && state.inletFace === state.outletFace) throw new Error("The inlet and outlet cannot be the same face.");
  const box = domainBox(tris, options());
  const cells = box.dims[0] * box.dims[1] * box.dims[2];
  if (cells > solver.maxCells) throw new Error("This grid needs " + fmtBig(cells) + " cells, more than this GPU allows (" + fmtBig(solver.maxCells) + "). Lower the grid detail.");
  grid = buildDomain(tris, options());
  lat = latticeFor(grid);
  await solver.setup(grid, lat);
  seeds = makeSeeds(grid);
  viewer.setDomain(box, state.mode, state.inletFace, state.outletFace);
  showFacts(box.dims, box.dx, cells, lat);
  history = []; smooth = null; last = null; avgField = null;
  timeBase = 0; stepBase = 0; sliceShown = null; tlClear();
  ranges = { sp: lat.uLat * 1.4, pLo: 0, pHi: 0 };
  prepared = true;
  // Rays that enter the surface without leaving it mean the mesh has gaps.
  if (grid.openColumns > Math.max(2, 0.01 * grid.hitColumns)) {
    return "This STL is not watertight (its surface has gaps), so the solver's copy of the shape may be wrong. Repair the mesh for results you can trust.";
  }
  if (grid.portsOutside && grid.portsOutside.length) {
    return "No enclosed opening was found on the " + grid.portsOutside.map((f) => FACE_NAMES[f]).join(" and ") +
      " face, so the flow there uses the open space around the outside of the part. Pick another face if that is not what you meant.";
  }
  return "";
}

function simTime() { return timeBase + (solver.steps - stepBase) * lat.dt; }

// Speed or fluid changed. If a run is prepared it carries on with the same grid and the
// same flow state: only the relaxation time and the conversion to real units change.
function flowChanged(dragging) {
  const ok = state.speed > 0 && state.speed <= MAX_SPEED && state.rho > 0 && state.mu > 0 && isFinite(state.speed) && isFinite(state.rho) && isFinite(state.mu);
  if (!prepared || !grid || !solver || !ok) { invalidate(); return; }
  setupVersion++;
  const now = simTime(), old = lat;
  lat = latticeFor(grid);
  solver.updateLattice(lat);
  timeBase = now; stepBase = solver.steps;
  if (Math.abs(lat.uLat / old.uLat - 1) > 1e-9) {
    // Very viscous flows also change the solver's own inlet speed, so stored fields no longer match.
    const k = lat.uLat / old.uLat;
    ranges.sp *= k; ranges.pLo *= k * k; ranges.pHi *= k * k;
    avgField = null;
  }
  // Old values were in the old units, so start the chart and the smoothed readouts again.
  history = []; smooth = null;
  showFacts([grid.nx, grid.ny, grid.nz], grid.dx, grid.n, lat);
  renderTiles(); drawChart(); updateLegend();
  if (!dragging) { updateViz(false); refreshProbe(); }
  if (!running) $("status").textContent = "Flow settings changed. Press Run to carry on from the current flow with the new values.";
}

// ---------- Run loop ----------
async function runLoop(token) {
  let k = 2, lastRead = performance.now() - 150;
  const readEvery = 250 + grid.n / 1e6 * 220;
  try {
    while (running && token === runToken) {
      const t0 = performance.now();
      const done = await solver.step(k);
      const el = performance.now() - t0;
      rate.steps += done;
      k = Math.max(2, Math.min(128, Math.round(0.5 * k + 0.5 * k * 30 / Math.max(el, 0.5))));
      if (token !== runToken) break;
      if (performance.now() - lastRead > readEvery) {
        const data = await solver.read();
        lastRead = performance.now();
        if (token !== runToken) break;
        if (!ingest(data)) break;
      }
    }
  } catch (e) {
    if (token === runToken) {
      running = false; prepared = false; $("runBtn").textContent = "Run";
      notify(solver.lost ? "The GPU device was lost. Reload the page and try a lower grid detail."
        : "The solver stopped with an error: " + e.message, true);
    }
  }
}

// Stops the current run and waits until its loop has let go of the GPU buffers.
async function stopLoop() {
  running = false; runToken++;
  const p = loopPromise; loopPromise = null;
  if (p) await p;
}

function ingest(data) {
  const st = fieldStats(grid, data.macro);
  if (st.bad) {
    running = false; prepared = false;
    $("runBtn").textContent = "Run";
    notify("The solution went unstable and was stopped. Raise the grid detail or lower the speed, then press Run.", true);
    return false;
  }
  if (st.vmax > 0.17 && speedLimit > 0.15) {
    // Somewhere the flow is too close to the solver's speed limit to stay accurate.
    // Restart with a shorter time step, which lowers every speed in solver units.
    speedLimit *= Math.min(0.75, 0.11 / st.vmax);
    running = false; prepared = false; $("runBtn").textContent = "Run";
    notify("The flow speeds up more than expected in the narrowest section, so the run restarted with a shorter time step.");
    const tok = runToken;
    setTimeout(() => { if (tok === runToken && !running) toggleRun(true); }, 0); // not if the setup changed meanwhile
    return false;
  }
  if ($("avg").checked) {
    if (!avgField) avgField = data.macro.slice();
    else { const a = avgField, m = data.macro; for (let i = 0; i < a.length; i++) a[i] += (m[i] - a[i]) * 0.06; }
  } else avgField = null;
  last = data.macro;

  ranges.sp += (Math.max(st.vmax, lat.uLat * 0.5) - ranges.sp) * 0.3;
  let lo, hi;
  if (grid.mode === "external") { const q = 0.5 * lat.uLat * lat.uLat; lo = -1.2 * q; hi = q; }
  else { lo = Math.min(st.pmin, 0); hi = Math.max(st.pmax, 1e-9); }
  ranges.pLo += (lo - ranges.pLo) * 0.3; ranges.pHi += (hi - ranges.pHi) * 0.3;

  const t = simTime();
  let row;
  if (grid.mode === "external") {
    const f = data.force.map((v) => v * lat.force);
    const qA = 0.5 * state.rho * state.speed ** 2 * grid.frontalArea;
    row = { t, drag: f[0], side: f[1], lift: f[2], cd: f[0] / qA, cl: f[2] / qA };
  } else {
    const im = internalMetrics(grid, data.macro);
    const toQ = lat.vel * grid.dx * grid.dx;
    row = { t, dp: im.dp * lat.pres, q: im.qout * toQ, qin: im.qin * toQ, balance: im.balance, peak: st.vmax * lat.vel };
  }
  history.push(row);
  if (history.length > 4000) history.splice(0, 1000);
  if (!smooth) smooth = { ...row };
  else for (const key in row) smooth[key] = key === "t" ? row.t : smooth[key] + (row[key] - smooth[key]) * 0.25;

  renderTiles(); drawChart(); updateViz(true);
  const now = performance.now();
  if (now - rate.t > 800) { rate.value = rate.steps * 1000 / (now - rate.t); rate.steps = 0; rate.t = now; }
  const rampNow = Solver.ramp(solver.steps, lat.rampSteps);
  $("status").textContent = (rampNow < 1 ? "Starting up gently, inlet at " + Math.round(rampNow * 100) + " % of full speed. " : "") + "Simulated " + fmt(t * 1000) + " ms of flow, step " + solver.steps + ", " +
    Math.round(rate.value) + " steps/s, " + fmtBig(rate.value * grid.n) + " cell updates/s";
  return true;
}

// ---------- Results display ----------
function updateLegend() {
  if (!lat) return;
  $("spHi").textContent = fmt(ranges.sp * lat.vel) + " m/s";
  $("prLo").textContent = fmt(ranges.pLo * lat.pres) + " Pa";
  $("prHi").textContent = fmt(ranges.pHi * lat.pres) + " Pa";
}

// Streamline seeds: the default ones plus a ring for every smoke wand point.
function allSeeds(m) {
  if (!smoke.length) return seeds;
  const out = seeds.slice();
  for (const p of smoke) for (const v of seedCluster(grid, m, p)) out.push(v);
  return out;
}

// record: true when called with fresh solver data, so the slice image goes into the timeline.
function updateViz(record) {
  if (!last || !prepared) return;
  const m = avgField || last;
  const dims = [grid.nx, grid.ny, grid.nz];
  if ($("showSlice").checked) {
    const a = state.sliceAxis;
    const idx = Math.max(1, Math.min(dims[a] - 2, Math.round(state.slicePos * (dims[a] - 1))));
    const isP = state.field === "pressure";
    const img = sliceImage(grid, m, a, idx, state.field, isP ? ranges.pLo : 0, isP ? ranges.pHi : ranges.sp);
    sliceShown = { axis: a, idx };
    if (record === true) tlPush(img, simTime());
    const past = tl.view >= 0 ? tl.frames[tl.view] : null;
    viewer.setSlice(grid, a, idx, past ? past.img : img);
  } else { viewer.hideSlice(); sliceShown = null; }
  viewer.setStreamlines($("showLines").checked ? streamlines(grid, m, allSeeds(m), 0, ranges.sp, seeds.length) : null);
  if ($("showSurf").checked) {
    viewer.setSurfaceColors(surfaceColors(grid, m, baked.getAttribute("position").array, baked.getAttribute("normal").array, ranges.pLo, ranges.pHi));
  } else viewer.setSurfaceColors(null);
  updateLegend();
  tlRender();
  refreshProbe();
}

// ---------- Hover probe ----------
function refreshProbe() {
  const el = $("probe");
  const hide = () => { el.hidden = true; };
  if (!pointer || pointer.down || !last || !prepared || !sliceShown) return hide();
  const hit = viewer.pickSlice(pointer.x, pointer.y);
  if (!hit) return hide();
  const pr = probe(grid, avgField || last, hit, sliceShown.axis, sliceShown.idx);
  if (!pr) return hide();
  el.textContent = fmt(pr.speed * lat.vel) + " m/s\n" + fmt(pr.pressure * lat.pres) + " Pa";
  el.hidden = false;
  const box = $("viewport").getBoundingClientRect();
  const x = Math.min(pointer.x - box.left + 14, box.width - el.offsetWidth - 4);
  const y = Math.min(pointer.y - box.top + 16, box.height - el.offsetHeight - 4);
  el.style.transform = "translate(" + Math.max(0, Math.round(x)) + "px, " + Math.max(0, Math.round(y)) + "px)";
}

// ---------- Smoke wand and port picking ----------
function addSmoke(e) {
  if (!last || !prepared) { notify("Press Run first. Smoke follows the solved flow."); return; }
  let p = viewer.pickSlice(e.clientX, e.clientY);
  if (!p || !probe(grid, last, p)) {
    const q = viewer.pickInlet(e.clientX, e.clientY);
    if (q) {
      // step a few cells in from the inlet face, where the flow is defined
      const n = grid.inVec;
      p = [q[0] + n[0] * 2.5 * grid.dx, q[1] + n[1] * 2.5 * grid.dx, q[2] + n[2] * 2.5 * grid.dx];
    }
  }
  if (!p || !probe(grid, last, p)) { notify("No smoke released: Shift + click on a coloured part of the slice or on the green inlet face."); return; }
  smoke.push(p);
  if (smoke.length > 40) smoke.shift();
  $("showLines").checked = true;
  updateViz(false);
}

function pickPort(e) {
  const f = viewer.pickFace(e.clientX, e.clientY);
  if (f < 0) return;
  const wasIn = state.inletFace, wasOut = state.outletFace;
  if (e.altKey) {
    if (f === wasOut) return;
    state.outletFace = f;
    if (f === wasIn) state.inletFace = wasOut; // picking the other port's face swaps them
  } else {
    if (f === wasIn) return;
    state.inletFace = f;
    if (f === wasOut) state.outletFace = wasIn;
  }
  syncInputs(true); invalidate();
  notify("Inlet: " + FACE_NAMES[state.inletFace] + ". Outlet: " + FACE_NAMES[state.outletFace] + ". Press Run to solve.");
}

// ---------- Timeline of past slice images ----------
function tlClear() {
  tl.frames = []; tl.view = -1;
  if (tl.playing) { clearInterval(tl.timer); tl.playing = false; }
  tlRender();
}
function tlPush(img, t) {
  tl.frames.push({ img, t });
  if (tl.frames.length > TL_MAX) { tl.frames.shift(); if (tl.view > 0) tl.view--; }
}
function tlRender() {
  const n = tl.frames.length, s = $("tlPos");
  s.max = String(Math.max(0, n - 1));
  s.value = String(tl.view >= 0 ? tl.view : Math.max(0, n - 1));
  s.disabled = n < 2;
  $("tlPlay").textContent = tl.playing ? "Stop" : "Play";
  $("tlPlay").disabled = n < 2;
  const f = tl.view >= 0 ? tl.frames[tl.view] : null;
  $("tlLabel").textContent = f
    ? "Replay: image " + (tl.view + 1) + " of " + n + ", at " + fmt(f.t * 1000) + " ms. Drag to the right end for live."
    : n ? "Live, at " + fmt(tl.frames[n - 1].t * 1000) + " ms. " + n + " slice images stored."
    : "Live. No slice images stored yet.";
}
// Puts the chosen stored image (or the live one) on the slice plane.
function tlShow() {
  const f = tl.view >= 0 ? tl.frames[tl.view] : null;
  if (f && sliceShown && prepared && grid) viewer.setSlice(grid, sliceShown.axis, sliceShown.idx, f.img);
  else if (!f) updateViz(false);
  tlRender();
}
function tlToggle() {
  if (tl.playing) { clearInterval(tl.timer); tl.playing = false; tlRender(); return; }
  if (tl.frames.length < 2) return;
  if (!$("showSlice").checked) { $("showSlice").checked = true; updateViz(false); }
  tl.playing = true;
  if (tl.view < 0 || tl.view >= tl.frames.length - 1) tl.view = 0;
  tlShow();
  tl.timer = setInterval(() => {
    if (!tl.frames.length) { tlClear(); return; }
    tl.view = (tl.view + 1) % tl.frames.length;
    tlShow();
  }, 90);
}

function renderTiles() {
  const s = smooth, ext = state.mode === "external";
  const v = (key, unit) => (s ? fmt(s[key]) + (unit ? " " + unit : "") : "-");
  const re = lat && prepared ? fmtBig(lat.reynolds) : "-";
  const tiles = ext ? [
    ["a", "Drag coefficient", v("cd"), grid && prepared ? "Cd, on " + fmt(grid.frontalArea * 1e4) + " cm2 frontal area" : "Cd"],
    ["b", "Lift coefficient", v("cl"), "Cl, positive is up (+Z)"],
    ["", "Drag force", v("drag", "N"), "along the flow (+X)"],
    ["", "Lift force", v("lift", "N"), "upward (+Z)"],
    ["", "Side force", v("side", "N"), "sideways (+Y)"],
    ["", "Reynolds number", re, "based on model length"],
  ] : [
    ["a", "Pressure drop", v("dp", "Pa"), "inlet minus outlet"],
    ["b", "Flow rate", s ? fmt(s.q * 60000) + " L/min" : "-", "leaving through the outlet"],
    ["", "Mean inlet speed", s && grid && prepared ? fmt(s.qin / grid.inletArea) + " m/s" : "-", "target " + fmt(state.speed) + " m/s"],
    ["", "Peak speed", v("peak", "m/s"), "fastest point in the flow"],
    ["", "Mass balance", s ? fmt(100 * s.balance) + " %" : "-", "mass out against mass in"],
    ["", "Reynolds number", re, "based on inlet diameter"],
  ];
  $("tiles").innerHTML = tiles.map((t) =>
    '<div class="tile ' + t[0] + '"><div class="lbl">' + t[1] + '</div><div class="value">' + t[2] + '</div><div class="sub">' + t[3] + "</div></div>").join("");
  $("chartKey").innerHTML = (ext ? [["a", "Drag Cd"], ["b", "Lift Cl"]] : [["a", "Pressure drop, Pa"]])
    .map((k) => '<span class="key"><i style="background: var(--series-' + k[0] + ')"></i>' + k[1] + "</span>").join("");
}

function drawChart() {
  const c = $("chart"), dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(50, Math.round(c.clientWidth * dpr)), h = Math.round(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const g = c.getContext("2d"), cs = getComputedStyle(document.documentElement);
  g.clearRect(0, 0, w, h);
  const keys = state.mode === "external" ? ["cd", "cl"] : ["dp"];
  const rows = history.length > 600 ? history.filter((_, i) => i % Math.ceil(history.length / 600) === 0) : history;
  let lo = 0, hi = 0;
  // ignore the start-up transient when scaling
  const from = Math.floor(rows.length * 0.15);
  for (let i = from; i < rows.length; i++) for (const k of keys) { lo = Math.min(lo, rows[i][k]); hi = Math.max(hi, rows[i][k]); }
  if (hi - lo < 1e-12) { hi = lo + 1; }
  const pad = (hi - lo) * 0.12; lo -= lo < 0 ? pad : 0; hi += pad;
  const padL = 58 * dpr, padB = 16 * dpr, top = 6 * dpr;
  const X = (t, t0, t1) => padL + (w - padL - 6 * dpr) * (t1 > t0 ? (t - t0) / (t1 - t0) : 0);
  const Y = (v) => top + (h - padB - top) * (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo));
  g.font = 11 * dpr + "px 'IBM Plex Mono', Consolas, monospace";
  g.fillStyle = cs.getPropertyValue("--muted").trim();
  g.strokeStyle = cs.getPropertyValue("--line").trim();
  g.lineWidth = dpr; g.textAlign = "right"; g.textBaseline = "middle";
  for (const val of [lo, 0, hi]) {
    if (val < lo || val > hi) continue;
    g.beginPath(); g.moveTo(padL, Y(val)); g.lineTo(w, Y(val)); g.stroke();
    g.fillText(fmt(val), padL - 6 * dpr, Y(val));
  }
  if (rows.length < 2) return;
  const t0 = rows[0].t, t1 = rows[rows.length - 1].t;
  g.textAlign = "right"; g.textBaseline = "bottom";
  g.fillText(fmt(t1 * 1000) + " ms", w - 2 * dpr, h);
  keys.forEach((k, n) => {
    g.strokeStyle = g.fillStyle = cs.getPropertyValue(n ? "--series-b" : "--series-a").trim();
    g.lineWidth = 1.6 * dpr; g.lineJoin = "round";
    g.beginPath();
    rows.forEach((r, i) => { const x = X(r.t, t0, t1), y = Y(r[k]); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
    g.stroke();
    const e = rows[rows.length - 1];
    g.beginPath(); g.arc(X(e.t, t0, t1), Y(e[k]), 2.6 * dpr, 0, 6.3); g.fill();
  });
}

// ---------- Controls ----------
// Speed slider: logarithmic, 0.01 to 300 m/s over its travel.
const SLIDER_LO = -2, SLIDER_SPAN = Math.log10(300) + 2;
function sliderToSpeed(v) { return Number((10 ** (SLIDER_LO + SLIDER_SPAN * v / 1000)).toPrecision(3)); }
function speedToSlider(u) { return u > 0 ? Math.max(0, Math.min(1000, Math.round((Math.log10(u) - SLIDER_LO) / SLIDER_SPAN * 1000))) : 0; }

function syncInputs(keepGhost) {
  $("demo").value = state.demo;
  $("unit").value = String(state.unit);
  $("fluid").value = state.fluid;
  $("speed").value = state.speed;
  $("speedSlider").value = String(speedToSlider(state.speed));
  $("yaw").value = String(state.yaw); $("yawOut").textContent = String(state.yaw);
  $("slicePos").value = String(Math.round(state.slicePos * 1000));
  $("rho").value = state.rho; $("mu").value = state.mu;
  $("res").value = String(state.res);
  $("domainScale").value = String(state.domainScale);
  $("inletFace").value = String(state.inletFace);
  $("outletFace").value = String(state.outletFace);
  $("stlKind").value = state.stlIsFluid ? "fluid" : "solid";
  press($("mode"), "data-mode", state.mode);
  press($("sliceAxis"), "data-axis", state.sliceAxis);
  press($("field"), "data-field", state.field);
  const internal = state.mode === "internal";
  $("internalOpts").hidden = !internal;
  $("domainField").hidden = internal;
  $("yawField").hidden = internal;
  $("customFluid").hidden = state.fluid !== "custom";
  $("speedLabel").textContent = internal ? "Inlet speed, m/s" : "Wind speed, m/s";
  $("flowHint").textContent = internal
    ? "Flow enters through the openings on the green face and leaves through the orange face. Every other surface is a wall. Click a face of the box in the 3D view to make it the inlet, or Alt + click to make it the outlet."
    : "Flow runs along +X, from the green face to the orange face. Rotate the model to point it into the wind, or use the yaw angle for a crosswind. Z is up.";
  if (!keepGhost) {
    $("ghost").checked = internal;
    viewer.setGhost(internal);
  }
}

async function toggleRun(keepNotice) {
  if (busy) return;
  busy = true;
  try {
    if (running) { await stopLoop(); $("runBtn").textContent = "Run"; return; }
    await stopLoop();
    let warning = "";
    if (!prepared) {
      $("status").textContent = "Building the cell grid...";
      await new Promise((r) => setTimeout(r, 30));
      const ver = setupVersion;
      try { warning = await prepare(); }
      catch (e) { prepared = false; notify(e.message, true); $("status").textContent = "Not running."; return; }
      if (ver !== setupVersion) { invalidate(); return; } // the setup changed while the grid was being built
    }
    if (warning) notify(warning, true);
    else if (keepNotice !== true) clearNotice();
    running = true; $("runBtn").textContent = "Pause";
    rate = { steps: 0, t: performance.now(), value: 0 };
    loopPromise = runLoop(++runToken);
  } finally { busy = false; }
}

// Starts the run again from rest with the current setup.
async function restart() {
  if (busy) return;
  busy = true;
  try { await stopLoop(); prepared = false; $("runBtn").textContent = "Run"; }
  finally { busy = false; }
  await toggleRun();
}

// Checks a loaded setup file value by value. Returns the values to apply, or throws.
function cleanSetup(s) {
  if (!s || typeof s !== "object" || s.app !== "flowbench") throw new Error("not a setup file");
  const out = {}, bad = (k) => { throw new Error("value: " + k); };
  const oneOf = (k, list) => { if (k in s) { if (!list.includes(s[k])) bad(k); out[k] = s[k]; } };
  const number = (k, lo, hi) => { if (k in s) { if (typeof s[k] !== "number" || !isFinite(s[k]) || s[k] < lo || s[k] > hi) bad(k); out[k] = s[k]; } };
  const optionValues = (id) => [...$(id).options].map((o) => Number(o.value));
  oneOf("unit", optionValues("unit"));
  oneOf("mode", ["external", "internal"]);
  oneOf("fluid", ["air", "water", "oil", "custom"]);
  oneOf("res", optionValues("res"));
  oneOf("domainScale", optionValues("domainScale"));
  oneOf("inletFace", [0, 1, 2, 3, 4, 5]); oneOf("outletFace", [0, 1, 2, 3, 4, 5]);
  oneOf("stlIsFluid", [true, false]);
  oneOf("sliceAxis", [0, 1, 2]);
  oneOf("field", ["speed", "pressure"]);
  number("slicePos", 0, 1); number("yaw", -30, 30);
  number("speed", -1e12, 1e12); number("rho", -1e12, 1e12); number("mu", -1e12, 1e12);
  if ("rotation" in s) {
    if (!Array.isArray(s.rotation) || s.rotation.length !== 16 || !s.rotation.every((v) => typeof v === "number" && isFinite(v))) bad("rotation");
    out.rotation = s.rotation;
  }
  return out;
}

function wire() {
  $("demo").innerHTML = '<option value="">Your own STL</option>' +
    Object.entries(DEMOS).map(([k, d]) => '<option value="' + k + '">' + d.label + "</option>").join("");
  const faces = FACE_NAMES.map((n, i) => '<option value="' + i + '">' + n + "</option>").join("");
  $("inletFace").innerHTML = faces; $("outletFace").innerHTML = faces;

  $("demo").addEventListener("change", (e) => {
    if (e.target.value) loadDemo(e.target.value);
    else { e.target.value = state.demo; $("file").click(); } // "Your own STL" opens the file picker
  });
  $("openBtn").addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", (e) => { if (e.target.files[0]) loadFile(e.target.files[0]); e.target.value = ""; });
  const vp = $("viewport");
  vp.addEventListener("dragover", (e) => { e.preventDefault(); vp.classList.add("drop"); });
  vp.addEventListener("dragleave", () => vp.classList.remove("drop"));
  vp.addEventListener("drop", (e) => {
    e.preventDefault(); vp.classList.remove("drop");
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  });
  $("unit").addEventListener("change", (e) => { state.unit = +e.target.value; bake(true); });
  $("rot").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    const ax = b.dataset.axis, m = new THREE.Matrix4();
    if (ax === "x") m.makeRotationX(Math.PI / 2); else if (ax === "y") m.makeRotationY(Math.PI / 2); else m.makeRotationZ(Math.PI / 2);
    rot.premultiply(m); bake(true);
  });
  $("mode").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (state.mode === b.dataset.mode) return;
    state.mode = b.dataset.mode; syncInputs();
    if (state.yaw) bake(false); else invalidate(); // the yaw angle only applies to flow around the part
  });
  $("fluid").addEventListener("change", (e) => {
    state.fluid = e.target.value;
    if (FLUIDS[state.fluid]) Object.assign(state, FLUIDS[state.fluid]);
    syncInputs(true); flowChanged();
  });
  const num = (id, key) => $(id).addEventListener("change", (e) => {
    state[key] = e.target.value.trim() === "" ? NaN : +e.target.value;
    if (key === "speed") $("speedSlider").value = String(speedToSlider(state.speed));
    flowChanged();
  });
  num("speed", "speed"); num("rho", "rho"); num("mu", "mu");
  $("speedSlider").addEventListener("input", (e) => {
    state.speed = sliderToSpeed(+e.target.value);
    $("speed").value = state.speed;
    flowChanged(true);
  });
  $("speedSlider").addEventListener("change", () => { if (prepared) { updateViz(false); refreshProbe(); } });
  $("yaw").addEventListener("input", (e) => {
    state.yaw = +e.target.value; $("yawOut").textContent = String(state.yaw);
    bake(false);
  });
  const sel = (id, key, conv = Number) => $(id).addEventListener("change", (e) => { state[key] = conv(e.target.value); invalidate(); });
  sel("res", "res"); sel("domainScale", "domainScale"); sel("inletFace", "inletFace"); sel("outletFace", "outletFace");
  sel("stlKind", "stlIsFluid", (v) => v === "fluid");

  $("runBtn").addEventListener("click", () => toggleRun());
  $("restartBtn").addEventListener("click", restart);

  $("sliceAxis").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (state.sliceAxis === +b.dataset.axis) return;
    state.sliceAxis = +b.dataset.axis; press($("sliceAxis"), "data-axis", state.sliceAxis); tlClear(); updateViz(false);
  });
  $("field").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (state.field === b.dataset.field) return;
    state.field = b.dataset.field; press($("field"), "data-field", state.field); tlClear(); updateViz(false);
  });
  $("slicePos").addEventListener("input", (e) => { state.slicePos = +e.target.value / 1000; tlClear(); updateViz(false); });
  for (const id of ["showSlice", "showLines", "showSurf"]) $(id).addEventListener("change", () => updateViz(false));
  $("avg").addEventListener("change", () => { avgField = null; updateViz(false); });
  $("clearSmoke").addEventListener("click", () => { smoke = []; updateViz(false); });
  $("tlPlay").addEventListener("click", tlToggle);
  $("tlPos").addEventListener("input", (e) => {
    const v = +e.target.value;
    if (tl.playing) { clearInterval(tl.timer); tl.playing = false; }
    tl.view = v >= tl.frames.length - 1 ? -1 : v;
    tlShow();
  });

  // Pointer work on the 3D view: hover probe, and clicks that are not camera drags.
  const cv = $("view");
  let downAt = null, probeQueued = false;
  const queueProbe = () => {
    if (probeQueued) return;
    probeQueued = true;
    requestAnimationFrame(() => { probeQueued = false; refreshProbe(); });
  };
  cv.addEventListener("pointermove", (e) => {
    pointer = { x: e.clientX, y: e.clientY, down: e.buttons !== 0 };
    queueProbe();
  });
  cv.addEventListener("pointerleave", () => { pointer = null; queueProbe(); });
  cv.addEventListener("pointerdown", (e) => {
    downAt = e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
    pointer = { x: e.clientX, y: e.clientY, down: true };
    queueProbe();
  });
  cv.addEventListener("pointerup", (e) => {
    pointer = { x: e.clientX, y: e.clientY, down: false };
    queueProbe();
    const d = downAt; downAt = null;
    if (!d || e.button !== 0 || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) return; // a drag orbits the camera
    if (e.shiftKey) addSmoke(e);
    else if (state.mode === "internal") pickPort(e);
  });
  cv.addEventListener("pointercancel", () => { downAt = null; pointer = null; queueProbe(); });

  $("ghost").addEventListener("change", (e) => viewer.setGhost(e.target.checked));

  $("shotBtn").addEventListener("click", () => download("flowbench.png", viewer.screenshot()));
  $("csvBtn").addEventListener("click", () => {
    if (!history.length) { notify("There are no results yet. Press Run first."); return; }
    const keys = Object.keys(history[0]);
    const units = { t: "time_s", drag: "drag_N", side: "side_N", lift: "lift_N", cd: "Cd", cl: "Cl", dp: "pressure_drop_Pa", q: "flow_out_m3_per_s", qin: "flow_in_m3_per_s", balance: "mass_out_over_in", peak: "peak_speed_m_per_s" };
    const lines = [keys.map((k) => units[k] || k).join(",")].concat(history.map((r) => keys.map((k) => r[k]).join(",")));
    downloadText("flowbench-results.csv", lines.join("\n"), "text/csv");
  });
  $("saveBtn").addEventListener("click", () => {
    const out = { app: "flowbench", version: 1, ...state, rotation: rot.toArray() };
    downloadText("flowbench-setup.json", JSON.stringify(out, null, 2), "application/json");
  });
  $("loadBtn").addEventListener("click", () => $("setupFile").click());
  $("setupFile").addEventListener("change", async (e) => {
    const f = e.target.files[0]; e.target.value = "";
    if (!f) return;
    try {
      let s;
      try { s = cleanSetup(JSON.parse(await f.text())); }
      catch (err) {
        notify(String(err.message).startsWith("value: ")
          ? "That setup file has a setting that cannot be used (" + err.message.slice(7) + "), so nothing was changed."
          : "That is not a Flowbench setup file. Nothing was changed.");
        return;
      }
      if (s.rotation) { rot.fromArray(s.rotation); delete s.rotation; }
      Object.assign(state, s); // name and demo stay as they are: the setup applies to the open model
      syncInputs(); bake(true);
      notify("Setup loaded. It applies to the model that is open now; the STL itself is not stored in the setup file.");
    } catch (err) { notify("The setup could not be applied: " + err.message, true); }
  });
  window.addEventListener("resize", drawChart);
}

// ---------- Start ----------
async function start() {
  wire();
  try {
    solver = await Solver.create();
    $("gpuChip").textContent = "GPU solver ready";
    $("gpuChip").classList.add("ok");
  } catch (e) {
    $("gpuChip").textContent = "No WebGPU";
    $("gpuChip").classList.add("bad");
    notify(e.message + " You can still load and inspect models, but the solver needs WebGPU: use a current Chrome, Edge or Firefox on a computer with a graphics card.", true);
  }
  loadDemo("ahmed");
  window.flowbench = {
    state, get grid() { return grid; }, get lat() { return lat; }, get history() { return history; }, get solver() { return solver; },
    get last() { return last; }, get smoke() { return smoke; }, get timeline() { return tl; }, viewer, loadDemo, toggleRun,
  };
  if (solver) toggleRun();
}
start();
