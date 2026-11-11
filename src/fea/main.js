import "../style.css";
import "./fea.css";
import * as THREE from "three";
import { readModel, toGeometry, isCad, FORMATS_TEXT } from "../io.js";
import { voxelize } from "../domain.js";
import { LUT_CSS } from "../results.js";
import { View } from "./view.js";
import { buildAdjacency, growRegion, cellsUnderTriangles, cellNodes, voxelSkin, tetSkin, tetDensity, smoothSurface, toSTL } from "./mesh.js";
import { buildTetMesh, gridFor, facesOnTriangles, connectedTo, filterMesh } from "./tetmesh.js";

const $ = (id) => document.getElementById(id);
// E in GPa, yield in MPa, density in kg/m3
const MATERIALS = {
  pla: ["PLA (printed)", 3.5, 0.36, 50, 1240],
  petg: ["PETG (printed)", 2.1, 0.38, 50, 1270],
  abs: ["ABS", 2.2, 0.35, 40, 1050],
  pa12: ["Nylon PA12", 1.7, 0.39, 45, 1010],
  al6061: ["Aluminium 6061-T6", 69, 0.33, 276, 2700],
  s275: ["Steel S275", 210, 0.3, 275, 7850],
  ti64: ["Titanium Ti-6Al-4V", 114, 0.34, 880, 4430],
  custom: ["Custom", 3.5, 0.36, 50, 1240],
};

const state = { unit: 0.001, res: 32, mat: "pla", E: 3.5e9, nu: 0.36, yield: 50e6, rho: 1240, colourBy: "stress", viewMode: "blocks", meshType: "tet", tetSmooth: true };
let name = "", pos = null, adj = null, grid = null, cells = null;
// Tetrahedral mesh of the current run (null when the cube mesh is in use), and
// a one-entry cache so the preview and the run do not mesh the part twice.
let tet = null, tetCache = null;
let conds = [];          // {kind, tris, force}
let tool = null;         // null, "fixed" or "load"
let designs = [], sel = -1, live = null, recommended = -1;
let worker = null, running = false, stagesWanted = 1;

const view = new View($("view"));
document.querySelector(".legend .bar").style.background = LUT_CSS;

// ---------- Helpers ----------
function fmt(v, digits = 3) {
  if (!isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e5 || a < 1e-3) return v.toExponential(digits - 1).replace("e+", "e");
  return String(Number(v.toPrecision(digits)));
}
let noticeTimer = 0;
function notify(msg, sticky) {
  const n = $("notice");
  n.textContent = msg; n.classList.add("show");
  clearTimeout(noticeTimer);
  if (!sticky) noticeTimer = setTimeout(() => n.classList.remove("show"), 9000);
}
function download(fileName, blob) {
  const url = URL.createObjectURL(blob), a = document.createElement("a");
  a.href = url; a.download = fileName;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
function press(container, value) {
  for (const b of container.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.v === value));
}

// ---------- Model ----------
let srcPos = null, srcFaces = null, presetKey = "", faceTris = null, lastFile = null, backend = "";

// positions: triangle soup in file units; faces: CAD face number per triangle, or null
function setSource(positions, faces, label, key) {
  srcPos = positions; srcFaces = faces; presetKey = key || "";
  const g = toGeometry(Float32Array.from(positions));
  g.scale(state.unit, state.unit, state.unit);
  g.computeVertexNormals();
  g.computeBoundingBox();
  name = label;
  pos = g.getAttribute("position").array;
  adj = buildAdjacency(pos);
  faceTris = null;
  if (faces) {
    faceTris = new Map();
    for (let t = 0; t < faces.length; t++) { if (faces[t] < 0) continue; let l = faceTris.get(faces[t]); if (!l) faceTris.set(faces[t], l = []); l.push(t); }
  }
  const cadOpt = $("pickMode").querySelector('option[value="cad"]');
  cadOpt.hidden = !faces;
  $("pickMode").value = faces ? "cad" : "flat";
  view.setModel(g);
  conds = [];
  if (key) DEMOS[key].preset(g.boundingBox);
  const s = g.boundingBox.getSize(new THREE.Vector3());
  $("modelInfo").textContent = label + ": " + (pos.length / 9) + " triangles" + (faces ? ", " + faceTris.size + " CAD faces" : "") + ", " +
    [s.x, s.y, s.z].map((v) => fmt(v * 1000)).join(" x ") + " mm. Reads " + FORMATS_TEXT + "; you can also drop a file onto the 3D view.";
  invalidate();
  renderConds();
}

// Selects every triangle that passes a test on its normal and centre point.
function trisWhere(test) {
  const out = [];
  for (let t = 0; t < pos.length / 9; t++) {
    const o = 9 * t;
    const c = [(pos[o] + pos[o + 3] + pos[o + 6]) / 3, (pos[o + 1] + pos[o + 4] + pos[o + 7]) / 3, (pos[o + 2] + pos[o + 5] + pos[o + 8]) / 3];
    if (test([adj.normals[3 * t], adj.normals[3 * t + 1], adj.normals[3 * t + 2]], c)) out.push(t);
  }
  return Int32Array.from(out);
}

const DEMOS = {
  beam: {
    label: "Cantilever beam 100 x 20 x 20 mm",
    build: () => new THREE.BoxGeometry(100, 20, 20).translate(50, 0, 0),
    preset: () => {
      conds.push({ kind: "fixed", tris: trisWhere((n) => n[0] < -0.9), force: [0, 0, 0] });
      conds.push({ kind: "load", tris: trisWhere((n) => n[0] > 0.9), force: [0, 0, -100] });
    },
  },
  bracket: {
    label: "L bracket 80 x 80 x 30 mm",
    build: () => {
      const s = new THREE.Shape();
      s.moveTo(0, 0); s.lineTo(80, 0); s.lineTo(80, 20); s.lineTo(20, 20); s.lineTo(20, 80); s.lineTo(0, 80); s.closePath();
      return new THREE.ExtrudeGeometry(s, { depth: 30, bevelEnabled: false }).rotateX(Math.PI / 2);
    },
    preset: (bb) => {
      conds.push({ kind: "fixed", tris: trisWhere((n, c) => n[2] > 0.9 && c[2] > bb.max.z - 1e-6), force: [0, 0, 0] });
      conds.push({ kind: "load", tris: trisWhere((n, c) => n[0] > 0.9 && c[0] > bb.max.x - 1e-6), force: [0, 0, -200] });
    },
  },
};

function loadDemo(key) {
  const d = DEMOS[key];
  state.unit = 0.001; $("unit").value = "0.001"; $("demo").value = key;
  lastFile = null; $("cadField").hidden = true;
  let g = d.build();
  if (g.index) g = g.toNonIndexed();
  setSource(g.getAttribute("position").array, null, d.label, key);
}

async function loadFile(file, buffer) {
  try {
    $("status").textContent = (isCad(file.name) ? "Reading CAD geometry from " : "Reading ") + file.name + "...";
    const buf = buffer || await file.arrayBuffer();
    await new Promise((r) => setTimeout(r, 20));
    const model = await readModel(file.name, buf, $("cadQuality").value);
    for (let i = 0; i < model.positions.length; i++) if (!isFinite(model.positions[i])) throw new Error("it contains coordinates that are not numbers");
    lastFile = model.cad ? { name: file.name, buffer: buf } : null;
    $("cadField").hidden = !model.cad;
    if (model.unit) { state.unit = model.unit; $("unit").value = String(model.unit); }
    $("demo").value = "";
    setSource(model.positions, model.faceIds, file.name, "");
    notify("Loaded " + file.name + (model.cad ? " with " + model.faces + " CAD faces" : "") + ". Now pick the faces that are held fixed and the faces that carry load.");
  } catch (e) {
    notify("That file could not be used: " + (e.message || "it could not be parsed") + ". Supported files are " + FORMATS_TEXT + ".");
    $("status").textContent = "File not loaded.";
  }
}

// ---------- Supports and loads ----------
function renderConds() {
  $("conds").innerHTML = conds.length ? conds.map((c, i) => {
    const text = c.kind === "fixed" ? "Fixed support" : "Load " + fmt(Math.hypot(...c.force)) + " N (" + c.force.map((v) => fmt(v)).join(", ") + ")";
    return '<li><span class="dot ' + c.kind + '"></span><span>' + text + ", " + c.tris.length + ' triangles</span><button type="button" data-i="' + i + '">Remove</button></li>';
  }).join("") : '<li><span></span><span class="small">Nothing marked yet. Use the two buttons above, then click faces on the model.</span><span></span></li>';
  view.setMarks(pos, conds);
}

function setTool(t) {
  tool = tool === t ? null : t;
  $("toolFixed").setAttribute("aria-pressed", String(tool === "fixed"));
  $("toolLoad").setAttribute("aria-pressed", String(tool === "load"));
  $("viewport").classList.toggle("picking", !!tool);
  const tip = $("tip");
  tip.hidden = !tool;
  tip.textContent = tool === "fixed" ? "Click each face that is bolted, clamped or glued down. Click the button again when done."
    : "Click each face that carries the force set on the left. Click the button again when done.";
  if (!tool) view.setHover(null, null);
  if (tool && designs.length) { showDesign(-1); }
}

// ---------- Setup for the solver ----------
function invalidate() {
  stopWorker();
  designs = []; sel = -1; live = null; recommended = -1; grid = null; tet = null;
  view.setResult(null); view.setSmooth(null); view.setModelStyle("solid");
  $("legendBox").hidden = true;
  $("design").disabled = true; $("design").max = 0;
  $("designLabel").textContent = "No designs yet";
  $("bar").style.width = "0";
  backend = "";
  renderTiles(); drawPareto(); showFacts();
  $("status").textContent = "Ready. Run the stress analysis, or go straight to finding lighter designs.";
  view.setEdges(null);
  if (previewOn) showPreview();
}

function showFacts() {
  const rows = [];
  if (grid && tet) rows.push(["Elements", tet.vol.length + " tetrahedra"], ["Nodes", String(tet.nodes.length / 3)], ["Element size", "about " + fmt(grid.h * 1000) + " mm"],
    ["Lowest element quality", fmt(tet.quality.min, 2) + " (1 is ideal)"]);
  else if (grid) rows.push(["Elements", String(cells.length)], ["Cell size", fmt(grid.h * 1000) + " mm"], ["Grid", grid.nx + " x " + grid.ny + " x " + grid.nz]);
  if (backend) rows.push(["Solver", backend]);
  $("facts").innerHTML = rows.map((r) => "<dt>" + r[0] + "</dt><dd>" + r[1] + "</dd>").join("");
}

// Voxelises the part at the current grid detail.
function buildGrid() {
  const g = gridFor(pos, state.res), dims = [g.nx, g.ny, g.nz];
  const solid = voxelize(pos, g.origin, g.h, g.nx, g.ny, g.nz);
  if (!solid.some((v) => v)) throw new Error("The part is thinner than one cell at this grid detail. Raise the grid detail or check the file units.");
  return { g, solid, dims };
}

// Meshes the part with tetrahedra at the current grid detail.
function buildTets() {
  if (tetCache && tetCache.pos === pos && tetCache.res === state.res) return tetCache.mesh;
  const mesh = buildTetMesh(pos, gridFor(pos, state.res));
  if (!mesh) throw new Error("The part is thinner than one element at this grid detail. Raise the grid detail or check the file units.");
  tetCache = { pos, res: state.res, mesh };
  return mesh;
}

// Supports and loads on the tetrahedral mesh. Each condition is a set of
// triangles of the original surface; the mesh faces lying on them are found,
// their nodes are fixed, or share the force in proportion to the face area
// attached to each node (so a force on a face is an even pressure).
function buildTetSetup() {
  let mesh = buildTets();
  const find = () => conds.map((c) => facesOnTriangles(mesh, pos, c.tris));
  let found = find();
  // keep only material connected to a support, otherwise loose pieces make the problem unsolvable
  const seeds = [];
  conds.forEach((c, n) => { if (c.kind === "fixed") for (const b of found[n].faces) seeds.push(mesh.bfaces[b] >> 2); });
  if (!seeds.length) throw new Error("The fixed faces do not touch any material at this grid detail. Raise the grid detail.");
  const reach = connectedTo(mesh, seeds);
  if (reach.some((v) => !v)) { mesh = filterMesh(mesh, reach); found = find(); }
  const nn = mesh.nodes.length / 3, ne = mesh.vol.length;
  const fixed = new Uint8Array(nn), force = new Float64Array(3 * nn), marked = new Uint8Array(nn), passive = new Uint8Array(ne);
  conds.forEach((c, n) => {
    const f = found[n];
    if (c.kind === "load" && !f.nodes.size) throw new Error("A loaded face is not connected to the supported part of the model. Check for loose pieces, or raise the grid detail.");
    for (const [node, area] of f.nodes) {
      marked[node] = 1;
      if (c.kind === "fixed") fixed[node] = 1;
      else for (let a = 0; a < 3; a++) force[3 * node + a] += c.force[a] * area / f.area;
    }
  });
  for (let q = 0; q < mesh.tets.length; q++) if (marked[mesh.tets[q]]) passive[q >> 2] = 1;
  tet = mesh; grid = mesh.grid; cells = { length: ne };
  showFacts();
  // state.tetSmooth = false (from the console, through window.flowfea.state) gives plain linear tetrahedra, for comparison
  return { kind: "tet", nodes: mesh.nodes, tets: mesh.tets, fixed, force, passive, E: state.E, nu: state.nu, smooth: state.tetSmooth };
}

function buildSetup() {
  if (!conds.some((c) => c.kind === "fixed")) throw new Error("Mark at least one fixed face first, so the part is held somewhere.");
  if (!conds.some((c) => c.kind === "load" && Math.hypot(...c.force) > 0)) throw new Error("Mark at least one loaded face with a force that is not zero.");
  if (state.meshType === "tet") return buildTetSetup();
  tet = null;
  const { g, solid, dims } = buildGrid();
  const [nx, ny, nz] = dims, h = g.h;
  const condCells = conds.map((c) => cellsUnderTriangles(g, solid, pos, adj, c.tris));
  // keep only material connected to a support, otherwise loose pieces make the problem unsolvable
  const reach = new Uint8Array(solid.length), queue = [];
  conds.forEach((c, n) => { if (c.kind === "fixed") for (const cell of condCells[n].keys()) if (!reach[cell]) { reach[cell] = 1; queue.push(cell); } });
  if (!queue.length) throw new Error("The fixed faces do not touch any material at this grid detail. Raise the grid detail.");
  const stride = [1, nx, nx * ny];
  for (let q = 0; q < queue.length; q++) {
    const c = queue[q], ijk = [c % nx, ((c / nx) | 0) % ny, (c / (nx * ny)) | 0];
    for (let a = 0; a < 3; a++) for (const d of [-1, 1]) {
      const v = ijk[a] + d;
      if (v < 0 || v >= dims[a]) continue;
      const o = c + d * stride[a];
      if (solid[o] && !reach[o]) { reach[o] = 1; queue.push(o); }
    }
  }
  const nn = (nx + 1) * (ny + 1) * (nz + 1);
  const fixed = new Uint8Array(nn), force = new Float64Array(3 * nn), passive = new Uint8Array(solid.length);
  conds.forEach((c, n) => {
    const nodes = new Set();
    for (const [cell, face] of condCells[n]) if (reach[cell]) { passive[cell] = 1; for (const node of cellNodes(g, cell, face)) nodes.add(node); }
    if (c.kind === "fixed") for (const node of nodes) fixed[node] = 1;
    else {
      if (!nodes.size) throw new Error("A loaded face is not connected to the supported part of the model. Check for loose pieces, or raise the grid detail.");
      for (const node of nodes) for (let a = 0; a < 3; a++) force[3 * node + a] += c.force[a] / nodes.size;
    }
  });
  grid = g;
  cells = [];
  for (let c = 0; c < reach.length; c++) if (reach[c]) cells.push(c);
  cells = Int32Array.from(cells);
  showFacts();
  return { nx, ny, nz, h, solid: reach, fixed, force, passive, E: state.E, nu: state.nu };
}

// ---------- Mesh preview ----------
// Shows the elements the solver will use, with supported faces in green and loaded faces in orange.
let previewOn = false;
function showTetPreview() {
  let mesh;
  try { mesh = buildTets(); } catch (e) { notify(e.message, true); return; }
  const ne = mesh.vol.length;
  if (ne > 3e6) { notify("This mesh is too large to draw (" + ne + " elements). The solver can still try to run it."); return; }
  const marks = new Map();
  for (const c of conds) {
    const col = c.kind === "fixed" ? [0.21, 0.76, 0.42] : [0.94, 0.54, 0.24];
    for (const b of facesOnTriangles(mesh, pos, c.tris).faces) marks.set(mesh.bfaces[b], col);
  }
  const skin = tetSkin(mesh, null, null, 0, null, 0, 1, marks);
  view.setSmooth(null); view.setResult(skin); view.setEdges($("meshEdges").checked ? skin : null);
  view.setModelStyle("hidden"); $("legendBox").hidden = true;
  const g = mesh.grid, thin = Math.min(g.nx, g.ny, g.nz);
  $("status").textContent = "Mesh preview: " + ne + " tetrahedral elements, " + mesh.nodes.length / 3 + " nodes, element size about " + fmt(g.h * 1000) +
    " mm, lowest element quality " + fmt(mesh.quality.min, 2) + " (1 is ideal, average " + fmt(mesh.quality.mean, 2) + "). " +
    (thin < 4 ? "The part is under 4 elements thick in one direction, which is too coarse for bending. Raise the grid detail." : "Green faces are held, orange faces are loaded.");
}
function showPreview() {
  if (!previewOn || !pos) return;
  if (state.meshType === "tet") { showTetPreview(); return; }
  let built;
  try { built = buildGrid(); } catch (e) { notify(e.message, true); return; }
  const { g, solid } = built, list = [];
  for (let c = 0; c < solid.length; c++) if (solid[c]) list.push(c);
  if (list.length > 5e6) { notify("This mesh is too large to draw (" + list.length + " elements). The solver can still try to run it."); return; }
  const cellArr = Int32Array.from(list), rgb = new Float32Array(list.length * 3).fill(-1);
  const index = new Map(list.map((c, e) => [c, e]));
  for (const c of conds) {
    const col = c.kind === "fixed" ? [0.21, 0.76, 0.42] : [0.94, 0.54, 0.24];
    for (const cell of cellsUnderTriangles(g, solid, pos, adj, c.tris).keys()) { const e = index.get(cell); if (e !== undefined) rgb.set(col, 3 * e); }
  }
  const skin = voxelSkin(g, cellArr, new Uint8Array(list.length).fill(1), null, 0, null, 0, 1, rgb);
  view.setSmooth(null); view.setResult(skin); view.setEdges($("meshEdges").checked ? skin : null);
  view.setModelStyle("hidden"); $("legendBox").hidden = true;
  const thin = Math.min(g.nx, g.ny, g.nz);
  $("status").textContent = "Mesh preview: " + list.length + " brick elements of " + fmt(g.h * 1000) + " mm, grid " + g.nx + " x " + g.ny + " x " + g.nz + ". " +
    (thin < 4 ? "The part is under 4 elements thick in one direction, which is too coarse for bending. Raise the grid detail." : "Green cells are held, orange cells are loaded.");
}
function setPreview(on) {
  previewOn = on;
  $("meshBtn").setAttribute("aria-pressed", String(on));
  if (on) { if (tool) setTool(tool); showPreview(); }
  else { view.setEdges(null); if (designs.length) showDesign(-1); else { view.setResult(null); view.setModelStyle("solid"); } }
}

// ---------- Worker ----------
function stopWorker() {
  if (worker) { worker.terminate(); worker = null; }
  running = false; live = null;
  $("stopBtn").disabled = true; $("analyseBtn").disabled = false; $("optBtn").disabled = false;
}

function startWorker(message) {
  stopWorker();
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = onWorker;
  worker.onerror = (e) => { stopWorker(); notify("The solver stopped with an error: " + (e.message || "unknown"), true); };
  running = true;
  $("stopBtn").disabled = false; $("analyseBtn").disabled = true; $("optBtn").disabled = true;
  worker.postMessage(message);
}

function addDesign(m) {
  // vol = material volume in m3; elements of the tetrahedral mesh differ in size
  let vol = 0, full = tet ? tet.volume : cells.length * grid.h ** 3;
  if (!m.x) vol = full;
  else if (tet) for (let e = 0; e < m.x.length; e++) vol += m.x[e] * tet.vol[e];
  else { for (let e = 0; e < m.x.length; e++) vol += m.x[e]; vol *= grid.h ** 3; }
  const d = {
    frac: vol / full, x: m.x, u: m.u, vm: m.vm, compliance: m.compliance, maxDisp: m.maxDisp, peakStress: m.peakStress,
    mass: state.rho * vol,
  };
  designs.push(d);
  return d;
}

function onWorker(ev) {
  const m = ev.data;
  if (m.type === "backend") {
    backend = m.name + (m.detail ? " (" + m.detail + ")" : ""); showFacts();
  } else if (m.type === "progress") {
    $("status").textContent = m.label + ": solver pass " + m.it + ", remaining mismatch " + fmt(m.res, 2);
  } else if (m.type === "analysis") {
    designs = []; addDesign(m);
    stopWorker();
    finishRun("Stress analysis done in " + m.iterations + " solver passes.");
  } else if (m.type === "iter") {
    live = m;
    $("status").textContent = "Shaping the design with " + Math.round(m.frac * 100) + " % of the material, pass " + m.it + ".";
    showLive();
  } else if (m.type === "stage") {
    live = null;
    addDesign(m);
    $("bar").style.width = Math.round(100 * designs.length / stagesWanted) + "%";
    pickRecommended();
    setDesignSlider();
    showDesign(designs.length - 1);
  } else if (m.type === "done") {
    stopWorker();
    finishRun(recommended > 0
      ? "Done. The highlighted design is the lightest one that still has a safety factor of " + fmt(+$("targetSF").value) + " or more."
      : recommended === 0 ? "Done. No lighter design meets the safety factor, so the original part is the one highlighted."
      : "Done. Even the solid part does not reach the safety factor; lower the load, pick a stronger material or thicken the part.");
  }
}

function finishRun(text) {
  pickRecommended();
  setDesignSlider();
  showDesign(recommended >= 0 ? recommended : designs.length - 1);
  $("status").textContent = text;
}

function safety(d) { return d.peakStress > 0 ? state.yield / d.peakStress : Infinity; }

function pickRecommended() {
  const target = +$("targetSF").value || 1;
  recommended = -1;
  designs.forEach((d, i) => { if (safety(d) >= target && (recommended < 0 || d.mass < designs[recommended].mass)) recommended = i; });
}

function setDesignSlider() {
  const s = $("design");
  s.max = Math.max(0, designs.length - 1);
  s.disabled = designs.length < 2;
}

// ---------- Display ----------
function showLive() {
  if (!live) return;
  const show = new Uint8Array(cells.length);
  for (let e = 0; e < cells.length; e++) show[e] = live.x[e] > 0.5 ? 1 : 0;
  view.setSmooth(null);
  view.setResult(tet ? tetSkin(tet, show, null, 0, live.x, -0.4, 1.2) : voxelSkin(grid, cells, show, null, 0, live.x, -0.4, 1.2));
  view.setModelStyle($("showOrig").checked ? "ghost" : "hidden");
  $("legendBox").hidden = true;
}

// Smooth outer surface of a design, for the Smooth view and the STL export.
// Tetrahedral mesh: the solid part is the boundary of the mesh itself, which
// follows the real surface; an optimised design is first turned into the
// fraction of every grid cell that is filled, then smoothed like the cube mesh.
function designSurface(d) {
  if (tet && !d.x) return tetSkin(tet, null, null, 0, null, 0, 1).positions;
  let dens;
  if (tet) dens = tetDensity(tet, d.x);
  else {
    dens = new Float32Array(grid.nx * grid.ny * grid.nz);
    for (let e = 0; e < cells.length; e++) dens[cells[e]] = d.x ? d.x[e] : 1;
  }
  return smoothSurface(grid, dens).positions;
}

function showDesign(i) {
  if (i >= 0) sel = i;
  const d = designs[sel];
  if (!d || tool) {
    // while picking faces, show the plain model so every face can be clicked
    view.setResult(null); view.setSmooth(null); view.setModelStyle("solid"); $("legendBox").hidden = true;
    renderTiles(); drawPareto();
    return;
  }
  $("design").value = sel;
  $("designLabel").textContent = "Design " + (sel + 1) + " of " + designs.length + ": " + Math.round(d.frac * 100) + " % of the material";
  const ne = cells.length, show = new Uint8Array(ne);
  for (let e = 0; e < ne; e++) show[e] = !d.x || d.x[e] > 0.5 ? 1 : 0;
  view.setModelStyle($("showOrig").checked ? "ghost" : "hidden");
  if (state.viewMode === "smooth") {
    view.setResult(null);
    view.setSmooth(designSurface(d));
    $("legendBox").hidden = true;
  } else {
    const model = view.size || 1;
    const deform = d.maxDisp > 0 ? (+$("deform").value / 100) * 0.12 * model / d.maxDisp : 0;
    let values = null, lo = 0, hi = 1;
    if (state.colourBy === "stress") { values = d.vm; hi = Math.max(d.peakStress * 1.05, 1e-9); }
    else if (state.colourBy === "disp" && tet) {
      // one value per node, so the colours blend across each face
      values = new Float32Array(tet.nodes.length / 3);
      for (let n = 0; n < values.length; n++) values[n] = Math.hypot(d.u[3 * n], d.u[3 * n + 1], d.u[3 * n + 2]);
      hi = Math.max(d.maxDisp, 1e-12);
    } else if (state.colourBy === "disp") {
      values = new Float32Array(ne);
      const NX = grid.nx + 1, NY = grid.ny + 1;
      for (let e = 0; e < ne; e++) {
        const c = cells[e], n = (c % grid.nx) + NX * ((((c / grid.nx) | 0) % grid.ny) + NY * ((c / (grid.nx * grid.ny)) | 0));
        values[e] = Math.hypot(d.u[3 * n], d.u[3 * n + 1], d.u[3 * n + 2]);
      }
      hi = Math.max(d.maxDisp, 1e-12);
    }
    view.setSmooth(null);
    view.setResult(tet ? tetSkin(tet, show, d.u, deform, values, lo, hi, null, state.colourBy === "disp") : voxelSkin(grid, cells, show, d.u, deform, values, lo, hi));
    $("legendBox").hidden = !values;
    if (values) {
      const stress = state.colourBy === "stress";
      $("lgName").textContent = stress ? "Stress" : "Movement";
      $("lgHi").textContent = stress ? fmt(hi / 1e6) + " MPa" : fmt(hi * 1000) + " mm";
    }
  }
  renderTiles(); drawPareto();
}

function renderTiles() {
  const d = designs[sel], base = designs[0];
  const sf = d ? safety(d) : 0, target = +$("targetSF").value || 1;
  const tiles = [
    ["a", "Mass", d ? fmt(d.mass * 1000) + " g" : "-", d && base ? Math.round(100 * d.mass / base.mass) + " % of the solid part" : "of the selected design"],
    ["b", "Stiffness", d && base ? Math.round(100 * base.compliance / d.compliance) + " %" : "-", "compared with the solid part"],
    ["", "Largest movement", d ? fmt(d.maxDisp * 1000) + " mm" : "-", "under the applied load"],
    ["", "Peak stress", d ? fmt(d.peakStress / 1e6) + " MPa" : "-", "99th percentile, von Mises"],
    ["", "Safety factor", d ? fmt(sf, 3) : "-", d ? (sf >= target ? "meets the target of " + fmt(target) : "below the target of " + fmt(target)) : "yield strength over peak stress"],
    ["", "Material kept", d ? Math.round(d.frac * 100) + " %" : "-", sel === recommended && d ? "recommended design" : "by volume"],
  ];
  $("tiles").innerHTML = tiles.map((t) =>
    '<div class="tile ' + t[0] + '"><div class="lbl">' + t[1] + '</div><div class="value">' + t[2] + '</div><div class="sub">' + t[3] + "</div></div>").join("");
  $("chartKey").innerHTML = '<span class="key"><i style="background: var(--good)"></i>meets the safety factor</span><span class="key"><i style="background: var(--warn)"></i>too weak</span>';
}

let chartPts = [];
function drawPareto() {
  const c = $("pareto"), dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(50, Math.round(c.clientWidth * dpr)), h = Math.round(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const g = c.getContext("2d"), cs = getComputedStyle(document.documentElement);
  const tok = (n) => cs.getPropertyValue(n).trim();
  g.clearRect(0, 0, w, h);
  chartPts = [];
  const padL = 70 * dpr, padB = 34 * dpr, padT = 26 * dpr, padR = 20 * dpr;
  const base = designs[0];
  const X = (massPct) => padL + (w - padL - padR) * (massPct / 100);
  const Y = (stiffPct) => padT + (h - padT - padB) * (1 - stiffPct / 100);
  g.font = 11 * dpr + "px 'IBM Plex Mono', Consolas, monospace";
  g.strokeStyle = tok("--line"); g.fillStyle = tok("--muted"); g.lineWidth = dpr;
  g.textAlign = "right"; g.textBaseline = "middle";
  for (const v of [0, 25, 50, 75, 100]) {
    g.beginPath(); g.moveTo(padL, Y(v)); g.lineTo(w - padR, Y(v)); g.stroke();
    g.fillText(v + " %", padL - 6 * dpr, Y(v));
  }
  g.textAlign = "center"; g.textBaseline = "top";
  for (const v of [0, 25, 50, 75, 100]) g.fillText(v + " %", X(v), h - padB + 5 * dpr);
  g.fillText("mass, percent of the solid part", (padL + w - padR) / 2, h - 14 * dpr);
  g.save(); g.translate(11 * dpr, (padT + h - padB) / 2); g.rotate(-Math.PI / 2); g.fillText("stiffness", 0, 0); g.restore();
  if (!base) return;
  const target = +$("targetSF").value || 1;
  const pts = designs.map((d, i) => ({ i, x: X(100 * d.mass / base.mass), y: Y(Math.min(100, 100 * base.compliance / d.compliance)), ok: safety(d) >= target, sf: safety(d) }));
  g.strokeStyle = tok("--muted"); g.lineWidth = 1.2 * dpr; g.beginPath();
  pts.forEach((p, n) => (n ? g.lineTo(p.x, p.y) : g.moveTo(p.x, p.y)));
  g.stroke();
  for (const p of pts) {
    g.fillStyle = p.ok ? tok("--good") : tok("--warn");
    g.beginPath(); g.arc(p.x, p.y, (p.i === sel ? 7 : 5) * dpr, 0, 6.3); g.fill();
    if (p.i === sel) { g.strokeStyle = tok("--ink"); g.lineWidth = 2 * dpr; g.beginPath(); g.arc(p.x, p.y, 10 * dpr, 0, 6.3); g.stroke(); }
    g.fillStyle = tok("--muted"); g.textAlign = "center"; g.textBaseline = "bottom";
    g.fillText("SF " + fmt(p.sf, 2), p.x, p.y - 12 * dpr);
    if (p.i === recommended) { g.fillStyle = tok("--ink"); g.textBaseline = "top"; g.fillText("lightest that is strong enough", Math.min(w - 110 * dpr, Math.max(110 * dpr, p.x)), p.y + 13 * dpr); }
  }
  chartPts = pts.map((p) => ({ i: p.i, x: p.x / dpr, y: p.y / dpr }));
}

// ---------- Actions ----------
function run(kind) {
  if (!pos) return;
  if (tool) setTool(tool);
  let setup;
  try { setup = buildSetup(); } catch (e) { notify(e.message, true); return; }
  $("notice").classList.remove("show");
  designs = []; sel = -1; recommended = -1;
  setPreview(false);
  if (tet && cells.length > 250000) notify("This mesh has " + cells.length + " tetrahedral elements. They are solved on the CPU only, so this may take several minutes. A coarser grid detail, or the cube mesh type with the GPU solver, is faster.");
  else if (!tet && cells.length > 400000) notify("This grid has " + cells.length + " elements. That needs the GPU solver and may take a while; the Solver line under the button shows which solver is in use.");
  if (kind === "analyse") {
    stagesWanted = 1;
    $("status").textContent = "Solving...";
    startWorker({ type: "analyse", setup });
  } else {
    const minFrac = +$("minFrac").value, fractions = [];
    for (let f = 0.9; f > minFrac - 1e-6; f -= 0.1) fractions.push(Math.round(f * 100) / 100);
    stagesWanted = fractions.length + 1;
    $("bar").style.width = "0";
    startWorker({ type: "optimise", setup, fractions, maxIter: 18 });
  }
}

function wire() {
  $("material").innerHTML = Object.entries(MATERIALS).map(([k, m]) => '<option value="' + k + '">' + m[0] + "</option>").join("");
  const applyMat = () => {
    const m = MATERIALS[state.mat], custom = state.mat === "custom";
    $("customMat").hidden = !custom;
    if (custom) { state.E = +$("matE").value * 1e9; state.nu = Math.min(0.49, +$("matNu").value); state.yield = +$("matY").value * 1e6; state.rho = +$("matRho").value; }
    else { state.E = m[1] * 1e9; state.nu = m[2]; state.yield = m[3] * 1e6; state.rho = m[4]; }
    $("matInfo").textContent = "Stiffness " + fmt(state.E / 1e9) + " GPa, yield " + fmt(state.yield / 1e6) + " MPa, density " + fmt(state.rho) + " kg/m3.";
  };
  $("material").addEventListener("change", (e) => { state.mat = e.target.value; applyMat(); invalidate(); });
  for (const id of ["matE", "matNu", "matY", "matRho"]) $(id).addEventListener("change", () => { applyMat(); invalidate(); });
  applyMat();

  $("demo").addEventListener("change", (e) => { if (e.target.value) loadDemo(e.target.value); });
  $("openBtn").addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", (e) => { if (e.target.files[0]) loadFile(e.target.files[0]); e.target.value = ""; });
  const vp = $("viewport");
  vp.addEventListener("dragover", (e) => { e.preventDefault(); vp.classList.add("drop"); });
  vp.addEventListener("dragleave", () => vp.classList.remove("drop"));
  vp.addEventListener("drop", (e) => { e.preventDefault(); vp.classList.remove("drop"); if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); });
  $("unit").addEventListener("change", (e) => {
    state.unit = +e.target.value;
    if (srcPos) setSource(srcPos, srcFaces, name, presetKey);
  });
  const readRes = () => {
    const custom = $("res").value === "custom";
    $("resCustom").hidden = !custom;
    state.res = custom ? Math.max(8, Math.min(400, Math.round(+$("resCustom").value || 32))) : +$("res").value;
    invalidate();
  };
  $("res").addEventListener("change", readRes);
  $("resCustom").addEventListener("change", readRes);
  $("meshType").addEventListener("change", (e) => { state.meshType = e.target.value === "cube" ? "cube" : "tet"; invalidate(); });
  $("meshBtn").addEventListener("click", () => setPreview(!previewOn));
  $("meshEdges").addEventListener("change", () => { if (previewOn) showPreview(); });
  $("cadQuality").addEventListener("change", () => { if (lastFile) loadFile({ name: lastFile.name }, lastFile.buffer); });

  $("toolFixed").addEventListener("click", () => setTool("fixed"));
  $("toolLoad").addEventListener("click", () => setTool("load"));
  $("conds").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    conds.splice(+b.dataset.i, 1); renderConds(); invalidate();
  });

  // click (not drag) on the model while a tool is active marks a face
  const cv = $("view");
  let down = null, lastSeed = -1, lastRegion = null;
  const regionAt = (ev) => {
    const t = view.pick(ev);
    if (t < 0) return null;
    if (t !== lastSeed) {
      lastSeed = t;
      const mode = $("pickMode").value;
      lastRegion = mode === "cad" && srcFaces && srcFaces[t] >= 0 ? Int32Array.from(faceTris.get(srcFaces[t])) : growRegion(adj, t, mode === "cad" ? "flat" : mode);
    }
    return lastRegion;
  };
  $("pickMode").addEventListener("change", () => { lastSeed = -1; });
  cv.addEventListener("pointerdown", (ev) => { down = [ev.clientX, ev.clientY]; });
  cv.addEventListener("pointermove", (ev) => {
    if (!tool || !pos || ev.buttons) return;
    const r = regionAt(ev);
    view.setHover(pos, r, tool === "fixed" ? 0x35c26b : 0xf08a3c);
  });
  cv.addEventListener("pointerleave", () => view.setHover(null, null));
  cv.addEventListener("pointerup", (ev) => {
    if (!down || !tool || !pos) return;
    const moved = Math.hypot(ev.clientX - down[0], ev.clientY - down[1]);
    down = null;
    if (moved > 4) return;
    const r = regionAt(ev);
    if (!r) return;
    const force = tool === "load" ? [+$("fx").value || 0, +$("fy").value || 0, +$("fz").value || 0] : [0, 0, 0];
    if (tool === "load" && Math.hypot(...force) === 0) { notify("Set a force that is not zero before picking a loaded face."); return; }
    conds.push({ kind: tool, tris: r, force });
    const keep = tool;
    invalidate(); renderConds();
    tool = null; setTool(keep);
  });

  $("analyseBtn").addEventListener("click", () => run("analyse"));
  $("optBtn").addEventListener("click", () => run("optimise"));
  $("stopBtn").addEventListener("click", () => {
    stopWorker();
    if (designs.length) finishRun("Stopped. Showing the designs finished so far.");
    else { view.setResult(null); view.setModelStyle("solid"); $("status").textContent = "Stopped."; }
  });
  $("targetSF").addEventListener("change", () => { pickRecommended(); renderTiles(); drawPareto(); });

  $("colourBy").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; state.colourBy = b.dataset.v; press($("colourBy"), state.colourBy); if (!live) showDesign(-1); });
  $("viewMode").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; state.viewMode = b.dataset.v; press($("viewMode"), state.viewMode); if (!live) showDesign(-1); });
  $("deform").addEventListener("input", () => { if (!live) showDesign(-1); });
  $("showOrig").addEventListener("change", () => { if (live) showLive(); else showDesign(-1); });
  $("design").addEventListener("input", (e) => { if (!live) showDesign(+e.target.value); });
  $("pareto").addEventListener("click", (ev) => {
    const r = ev.currentTarget.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = -1, bd = 30 * 30;
    for (const p of chartPts) { const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bd) { bd = d; best = p.i; } }
    if (best >= 0 && !live) showDesign(best);
  });

  $("stlBtn").addEventListener("click", () => {
    const d = designs[sel];
    if (!d) { notify("There is no design to export yet. Run the analysis or the optimiser first."); return; }
    download("flowbench-design-" + Math.round(d.frac * 100) + "pct.stl", new Blob([toSTL(designSurface(d), 1 / state.unit)], { type: "model/stl" }));
  });
  $("csvBtn").addEventListener("click", () => {
    if (!designs.length) { notify("There are no designs yet."); return; }
    const rows = ["material_kept_pct,mass_g,stiffness_pct_of_solid,max_movement_mm,peak_stress_MPa,safety_factor,recommended"];
    designs.forEach((d, i) => rows.push([Math.round(d.frac * 100), d.mass * 1000, 100 * designs[0].compliance / d.compliance, d.maxDisp * 1000, d.peakStress / 1e6, safety(d), i === recommended ? "yes" : ""].join(",")));
    download("flowbench-designs.csv", new Blob([rows.join("\n")], { type: "text/csv" }));
  });
  $("shotBtn").addEventListener("click", () => {
    fetch(view.screenshot()).then((r) => r.blob()).then((b) => download("flowbench-structure.png", b));
  });
  window.addEventListener("resize", drawPareto);
}

wire();
loadDemo("bracket");
window.flowfea = { state, get designs() { return designs; }, get grid() { return grid; }, get conds() { return conds; }, get mesh() { return tet; }, get elements() { return cells ? cells.length : 0; }, run, loadDemo };
