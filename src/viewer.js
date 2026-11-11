// three.js scene: the model, the solver domain, and the result overlays.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { NORMALS } from "./domain.js";

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0x0b0f16);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 2, 0.001, 1000);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x39424e, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(-1, -2, 3);
    this.scene.add(key);

    this.solidMat = new THREE.MeshStandardMaterial({ color: 0xc3ccd6, roughness: 0.65, metalness: 0.05, side: THREE.DoubleSide });
    this.colorMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8, metalness: 0, side: THREE.DoubleSide });
    this.model = new THREE.Mesh(new THREE.BufferGeometry(), this.solidMat);
    this.scene.add(this.model);
    this.ghost = false;

    this.domain = new THREE.Group();
    this.scene.add(this.domain);

    this.sliceTex = null;
    this.slice = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ transparent: true, side: THREE.DoubleSide, depthWrite: false }));
    this.slice.visible = false;
    this.slice.renderOrder = 2;
    this.scene.add(this.slice);

    this.lines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 }));
    this.lines.visible = false;
    this.lines.frustumCulled = false;
    this.scene.add(this.lines);

    this.raycaster = new THREE.Raycaster();
    this.box = null;
    this.inletMesh = null;

    const loop = () => {
      this.resize();
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    const pr = this.renderer.getPixelRatio();
    if (this.canvas.width !== Math.round(w * pr) || this.canvas.height !== Math.round(h * pr)) {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
  }

  setModel(geometry) {
    this.model.geometry.dispose();
    this.model.geometry = geometry;
    this.setSurfaceColors(null);
  }

  setGhost(on) {
    this.ghost = on;
    for (const m of [this.solidMat, this.colorMat]) {
      m.transparent = on; m.opacity = on ? 0.22 : 1; m.depthWrite = !on; m.needsUpdate = true;
    }
  }

  setSurfaceColors(colors) {
    const g = this.model.geometry;
    if (colors) {
      // Reuse the existing attribute where possible, so a new GPU buffer is not made on every update.
      const old = g.getAttribute("color");
      if (old && old.array.length === colors.length) { old.array.set(colors); old.needsUpdate = true; }
      else g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      this.model.material = this.colorMat;
    } else {
      this.model.material = this.solidMat;
    }
  }

  // Draws the solver box, plus the inlet and outlet faces.
  setDomain(box, mode, inletFace, outletFace) {
    for (const o of this.domain.children) {
      if (o.dispose) o.dispose();
      else { o.geometry.dispose(); o.material.dispose(); }
    }
    this.domain.clear();
    this.inletMesh = null;
    const size = box.dims.map((n) => n * box.dx);
    const c = box.origin.map((o, a) => o + size[a] / 2);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(size[0], size[1], size[2])),
      new THREE.LineBasicMaterial({ color: 0x5b6b7c }));
    edges.position.set(c[0], c[1], c[2]);
    this.domain.add(edges);
    const face = (f, color) => {
      const a = f >> 1, b = (a + 1) % 3, d = (a + 2) % 3;
      const geo = new THREE.PlaneGeometry(1, 1);
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false }));
      const p = [0, 0, 0];
      p[a] = f & 1 ? box.origin[a] + size[a] : box.origin[a];
      p[b] = c[b]; p[d] = c[d];
      m.position.set(p[0], p[1], p[2]);
      const u = [0, 0, 0], v = [0, 0, 0], w = [0, 0, 0];
      u[b] = size[b]; v[d] = size[d]; w[a] = 1;
      m.matrixAutoUpdate = false;
      m.matrix.makeBasis(new THREE.Vector3(...u), new THREE.Vector3(...v), new THREE.Vector3(...w)).setPosition(p[0], p[1], p[2]);
      this.domain.add(m);
      if (color === 0x35c26b) this.inletMesh = m;
      const len = 0.18 * Math.max(...size);
      const n = NORMALS[f], dir = new THREE.Vector3(n[0], n[1], n[2]);
      const isInlet = color === 0x35c26b;
      const start = new THREE.Vector3(p[0], p[1], p[2]).addScaledVector(dir, isInlet ? -len : 0);
      if (!isInlet) dir.negate();
      this.domain.add(new THREE.ArrowHelper(dir, start, len, color, len * 0.3, len * 0.14));
    };
    face(mode === "external" ? 0 : inletFace, 0x35c26b);
    face(mode === "external" ? 1 : outletFace, 0xf08a3c);
    this.box = { c, size, origin: box.origin.slice() };
  }

  fit() {
    const g = this.model.geometry;
    g.computeBoundingSphere();
    const s = g.boundingSphere;
    if (!s) return;
    const r = Math.max(s.radius, 1e-6);
    this.controls.target.copy(s.center);
    this.camera.position.copy(s.center).add(new THREE.Vector3(-1.6, -2.4, 1.3).multiplyScalar(r * 1.5));
    this.camera.near = r / 200; this.camera.far = r * 400;
    this.camera.updateProjectionMatrix();
  }

  // axis 0/1/2, pos = world coordinate of the plane, img = {data, w, h}
  setSlice(grid, axis, index, img) {
    const { origin, dx, nx, ny, nz } = grid;
    const p = origin[axis] + (index + 0.5) * dx;
    const x0 = origin[0], x1 = origin[0] + nx * dx, y0 = origin[1], y1 = origin[1] + ny * dx, z0 = origin[2], z1 = origin[2] + nz * dx;
    const corners = axis === 0 ? [p, y0, z0, p, y1, z0, p, y1, z1, p, y0, z1]
      : axis === 1 ? [x0, p, z0, x1, p, z0, x1, p, z1, x0, p, z1]
      : [x0, y0, p, x1, y0, p, x1, y1, p, x0, y1, p];
    const g = this.slice.geometry;
    const posAttr = g.getAttribute("position");
    if (posAttr) { posAttr.array.set(corners); posAttr.needsUpdate = true; }
    else g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(corners), 3));
    g.computeBoundingSphere(); // the raycaster (hover probe, smoke wand) needs current bounds
    if (!g.getAttribute("uv")) {
      g.setAttribute("uv", new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2));
      g.setIndex([0, 1, 2, 0, 2, 3]);
    }
    if (!this.sliceTex || this.sliceTex.image.width !== img.w || this.sliceTex.image.height !== img.h) {
      if (this.sliceTex) this.sliceTex.dispose();
      this.sliceTex = new THREE.DataTexture(img.data, img.w, img.h, THREE.RGBAFormat);
      this.sliceTex.magFilter = THREE.LinearFilter;
      this.sliceTex.minFilter = THREE.LinearFilter;
      this.sliceTex.colorSpace = THREE.SRGBColorSpace;
      this.slice.material.map = this.sliceTex;
      this.slice.material.needsUpdate = true;
    } else {
      this.sliceTex.image.data = img.data;
    }
    this.sliceTex.needsUpdate = true;
    this.slice.visible = true;
  }

  hideSlice() { this.slice.visible = false; }

  setStreamlines(data) {
    if (!data || !data.positions.length) { this.lines.visible = false; return; }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(data.colors, 3));
    this.lines.geometry.dispose();
    this.lines.geometry = g;
    this.lines.visible = true;
  }

  // Pointer position in client pixels to a ray from the camera.
  aim(clientX, clientY) {
    const r = this.canvas.getBoundingClientRect();
    if (!r.width || !r.height) return false;
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.camera.updateMatrixWorld();
    this.raycaster.setFromCamera(ndc, this.camera);
    return true;
  }

  // World point [x, y, z] where the pointer meets the slice plane, or null.
  pickSlice(clientX, clientY) {
    if (!this.slice.visible || !this.slice.geometry.getAttribute("position")) return null;
    if (!this.aim(clientX, clientY)) return null;
    this.slice.updateMatrixWorld();
    const hit = this.raycaster.intersectObject(this.slice, false)[0];
    return hit ? [hit.point.x, hit.point.y, hit.point.z] : null;
  }

  // World point where the pointer meets the green inlet face, or null.
  pickInlet(clientX, clientY) {
    if (!this.inletMesh || !this.aim(clientX, clientY)) return null;
    this.inletMesh.updateMatrixWorld(true);
    const hit = this.raycaster.intersectObject(this.inletMesh, false)[0];
    return hit ? [hit.point.x, hit.point.y, hit.point.z] : null;
  }

  // Index (0 to 5, numbered as in domain.js) of the domain box face under the pointer, or -1.
  pickFace(clientX, clientY) {
    if (!this.box || !this.aim(clientX, clientY)) return -1;
    const { origin, size } = this.box;
    const b = new THREE.Box3(new THREE.Vector3(origin[0], origin[1], origin[2]),
      new THREE.Vector3(origin[0] + size[0], origin[1] + size[1], origin[2] + size[2]));
    const p = this.raycaster.ray.intersectBox(b, new THREE.Vector3());
    if (!p) return -1;
    const q = [p.x, p.y, p.z];
    let best = -1, bestD = Infinity;
    for (let f = 0; f < 6; f++) {
      const a = f >> 1, d = Math.abs(q[a] - (f & 1 ? origin[a] + size[a] : origin[a])) / Math.max(size[a], 1e-12);
      if (d < bestD) { bestD = d; best = f; }
    }
    return best;
  }

  // Counts of live GPU resources, for leak checks.
  resources() {
    const m = this.renderer.info.memory;
    return { geometries: m.geometries, textures: m.textures };
  }

  // Coloured patches for the openings in custom port mode.
  // marks: [{face, cells, color}], box: {dims, dx, origin}
  setPortMarks(marks, box) {
    if (!this.portMarks) { this.portMarks = new THREE.Group(); this.scene.add(this.portMarks); }
    for (const o of [...this.portMarks.children]) { o.geometry.dispose(); o.material.dispose(); this.portMarks.remove(o); }
    if (!marks) return;
    const [nx, ny] = box.dims, h = box.dx;
    for (const m of marks) {
      const a = m.face >> 1, b = (a + 1) % 3, c = (a + 2) % 3, pos = new Float32Array(m.cells.length * 18);
      const plane = box.origin[a] + (m.face & 1 ? box.dims[a] * h : 0);
      m.cells.forEach((cell, n) => {
        const ijk = [cell % nx, ((cell / nx) | 0) % ny, (cell / (nx * ny)) | 0];
        const u0 = box.origin[b] + ijk[b] * h, v0 = box.origin[c] + ijk[c] * h;
        const quad = [[u0, v0], [u0 + h, v0], [u0 + h, v0 + h], [u0, v0], [u0 + h, v0 + h], [u0, v0 + h]];
        quad.forEach((q, k) => { const o = 18 * n + 3 * k; pos[o + a] = plane; pos[o + b] = q[0]; pos[o + c] = q[1]; });
      });
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: m.color, transparent: true, opacity: m.hot ? 0.85 : 0.45, side: THREE.DoubleSide, depthWrite: false }));
      mesh.renderOrder = 4;
      this.portMarks.add(mesh);
    }
  }

  // Shows the solver's cell model: skin = {positions, colors} from voxelSkin, or null to hide.
  setPreview(skin, edges) {
    if (!this.preview) {
      this.preview = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0, flatShading: true, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
      this.previewEdges = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x0b0f16, transparent: true, opacity: 0.55 }));
      this.scene.add(this.preview, this.previewEdges);
    }
    this.preview.geometry.dispose(); this.previewEdges.geometry.dispose();
    this.preview.geometry = new THREE.BufferGeometry(); this.previewEdges.geometry = new THREE.BufferGeometry();
    const on = !!(skin && skin.positions.length);
    this.preview.visible = on; this.previewEdges.visible = on && edges;
    this.model.visible = !on;
    if (!on) return;
    const g = this.preview.geometry, p = skin.positions;
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    g.setAttribute("color", new THREE.BufferAttribute(skin.colors, 3));
    g.computeVertexNormals();
    if (edges) {
      // each cell face is two triangles (6 vertices): corners are vertices 0, 1, 2 and 5
      const quads = p.length / 18, e = new Float32Array(quads * 24), order = [0, 1, 1, 2, 2, 5, 5, 0];
      for (let q = 0; q < quads; q++) for (let k = 0; k < 8; k++) {
        const src = 18 * q + 3 * order[k], dst = 24 * q + 3 * k;
        e[dst] = p[src]; e[dst + 1] = p[src + 1]; e[dst + 2] = p[src + 2];
      }
      this.previewEdges.geometry.setAttribute("position", new THREE.BufferAttribute(e, 3));
    }
  }

  screenshot() {
    this.renderer.render(this.scene, this.camera);
    return this.canvas.toDataURL("image/png");
  }
}
