// Model import shared by the flow and structure pages.
// Mesh formats (STL, OBJ, 3MF) are read with three.js loaders. CAD formats
// (STEP, IGES, BREP) are read with OpenCascade compiled to WebAssembly, which
// is only downloaded the first time a CAD file is opened.
import * as THREE from "three";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { ThreeMFLoader } from "three/addons/loaders/3MFLoader.js";
import occtWasmUrl from "occt-import-js/dist/occt-import-js.wasm?url";

export const ACCEPT = ".stl,.step,.stp,.iges,.igs,.brep,.obj,.3mf";
export const FORMATS_TEXT = "STL, STEP, IGES, BREP, OBJ or 3MF";
const CAD = { step: "ReadStepFile", stp: "ReadStepFile", iges: "ReadIgesFile", igs: "ReadIgesFile", brep: "ReadBrepFile" };

// How finely curved CAD surfaces are turned into triangles.
// linear = allowed gap between triangle and true surface, as a fraction of the part size.
export const CAD_QUALITY = {
  draft: { label: "Draft", linear: 0.004, angular: 0.6 },
  normal: { label: "Normal", linear: 0.001, angular: 0.3 },
  fine: { label: "Fine", linear: 0.0003, angular: 0.15 },
  ultra: { label: "Very fine", linear: 0.0001, angular: 0.08 },
};

let occtPromise = null;
function occt() {
  if (!occtPromise) {
    occtPromise = import("occt-import-js").then((m) => (m.default || m)({ locateFile: () => occtWasmUrl }));
    occtPromise.catch(() => { occtPromise = null; });
  }
  return occtPromise;
}

export function extension(name) {
  const m = /\.([a-z0-9]+)$/i.exec(name || "");
  return m ? m[1].toLowerCase() : "";
}
export function isCad(name) { return extension(name) in CAD; }

function fromObject(root) {
  // Flatten every mesh under a loaded object into one triangle soup.
  const parts = [];
  root.updateMatrixWorld(true);
  root.traverse((o) => {
    if (!o.isMesh || !o.geometry || !o.geometry.getAttribute("position")) return;
    const g = o.geometry.index ? o.geometry.toNonIndexed() : o.geometry.clone();
    g.applyMatrix4(o.matrixWorld);
    parts.push(g.getAttribute("position").array);
  });
  let n = 0;
  for (const p of parts) n += p.length;
  const pos = new Float32Array(n);
  let o = 0;
  for (const p of parts) { pos.set(p, o); o += p.length; }
  return pos;
}

// Reads a model file. Returns
//   positions: Float32Array, 9 numbers per triangle, in the file's own units
//   faceIds:   Int32Array with the CAD face number of each triangle, or null for mesh formats
//   faces:     number of CAD faces (0 for mesh formats)
//   cad:       true for STEP, IGES and BREP
//   unit:      metres per file unit when the format defines it (CAD files come out in millimetres), else null
export async function readModel(name, buffer, quality = "normal") {
  const ext = extension(name);
  if (ext in CAD) {
    const q = CAD_QUALITY[quality] || CAD_QUALITY.normal;
    let lib;
    try { lib = await occt(); }
    catch (e) { throw new Error("the CAD reader could not be loaded (" + (e.message || "network error") + ")"); }
    const res = lib[CAD[ext]](new Uint8Array(buffer), {
      linearUnit: "millimeter", linearDeflectionType: "bounding_box_ratio", linearDeflection: q.linear, angularDeflection: q.angular,
    });
    if (!res || !res.success || !res.meshes || !res.meshes.length) throw new Error("no solid or surface geometry was found in it");
    let tris = 0;
    for (const m of res.meshes) tris += m.index.array.length / 3;
    if (!tris) throw new Error("it contains no surfaces that could be triangulated");
    const positions = new Float32Array(tris * 9), faceIds = new Int32Array(tris).fill(-1);
    let t = 0, face = 0;
    for (const m of res.meshes) {
      const p = m.attributes.position.array, idx = m.index.array, first = t;
      for (let i = 0; i < idx.length; i++) {
        const v = 3 * idx[i];
        positions[9 * t + 3 * (i % 3)] = p[v]; positions[9 * t + 3 * (i % 3) + 1] = p[v + 1]; positions[9 * t + 3 * (i % 3) + 2] = p[v + 2];
        if (i % 3 === 2) t++;
      }
      for (const f of m.brep_faces || []) {
        for (let k = f.first; k <= f.last; k++) faceIds[first + k] = face;
        face++;
      }
    }
    return { positions, faceIds: face > 0 ? faceIds : null, faces: face, cad: true, unit: ext === "brep" ? null : 0.001 };
  }
  let positions;
  if (ext === "stl") {
    const g = new STLLoader().parse(buffer);
    const a = g.getAttribute("position");
    if (!a) throw new Error("it is not a valid binary or ASCII STL");
    positions = (g.index ? g.toNonIndexed() : g).getAttribute("position").array;
  } else if (ext === "obj") {
    positions = fromObject(new OBJLoader().parse(new TextDecoder().decode(buffer)));
  } else if (ext === "3mf") {
    positions = fromObject(new ThreeMFLoader().parse(buffer));
  } else {
    throw new Error("the file type ." + ext + " is not supported. Use " + FORMATS_TEXT);
  }
  if (!positions || positions.length < 9) throw new Error("it contains no triangles");
  return { positions, faceIds: null, faces: 0, cad: false, unit: ext === "3mf" ? 0.001 : null };
}

export function toGeometry(positions) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  return g;
}
