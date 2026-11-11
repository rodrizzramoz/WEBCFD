// three.js scene for the structure page.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export class View {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0x0b0f16);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 2, 0.0001, 1000);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x39424e, 2.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.7);
    key.position.set(-1.5, -2.5, 3);
    this.scene.add(key);

    this.modelMat = new THREE.MeshStandardMaterial({ color: 0xc3ccd6, roughness: 0.65, metalness: 0.05, side: THREE.DoubleSide });
    this.model = new THREE.Mesh(new THREE.BufferGeometry(), this.modelMat);
    this.scene.add(this.model);

    const overlay = (color, opacity) => {
      const m = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
        color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false,
        polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
      }));
      m.visible = false; m.renderOrder = 3;
      this.scene.add(m);
      return m;
    };
    this.hover = overlay(0xffffff, 0.35);
    this.marks = new THREE.Group();
    this.scene.add(this.marks);

    this.result = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.8, metalness: 0, flatShading: true }));
    this.result.visible = false;
    this.scene.add(this.result);
    this.smooth = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({ color: 0x9fc4e8, roughness: 0.55, metalness: 0.1, side: THREE.DoubleSide }));
    this.smooth.visible = false;
    this.scene.add(this.smooth);

    this.ray = new THREE.Raycaster();
    const loop = () => {
      const w = canvas.clientWidth, h = canvas.clientHeight, pr = this.renderer.getPixelRatio();
      if (w && h && (canvas.width !== Math.round(w * pr) || canvas.height !== Math.round(h * pr))) {
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
      }
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  setModel(geometry) {
    this.model.geometry.dispose();
    this.model.geometry = geometry;
    geometry.computeBoundingSphere();
    const s = geometry.boundingSphere, r = Math.max(s.radius, 1e-6);
    this.size = r;
    this.controls.target.copy(s.center);
    this.camera.position.copy(s.center).add(new THREE.Vector3(-1.5, -2.2, 1.4).multiplyScalar(r * 1.35));
    this.camera.near = r / 200; this.camera.far = r * 400;
    this.camera.updateProjectionMatrix();
  }

  // How the original model is drawn: "solid", "ghost" (faint, behind results) or "hidden".
  setModelStyle(style) {
    this.model.visible = style !== "hidden";
    const ghost = style === "ghost";
    this.modelMat.transparent = ghost; this.modelMat.opacity = ghost ? 0.14 : 1; this.modelMat.depthWrite = !ghost;
    this.modelMat.needsUpdate = true;
  }

  // Triangle index under the pointer, or -1.
  pick(ev) {
    const r = this.canvas.getBoundingClientRect();
    const p = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(p, this.camera);
    const wasVisible = this.model.visible;
    this.model.visible = true;
    const hit = this.ray.intersectObject(this.model, false)[0];
    this.model.visible = wasVisible;
    return hit ? hit.faceIndex : -1;
  }

  static trisGeometry(pos, tris) {
    const out = new Float32Array(tris.length * 9);
    for (let n = 0; n < tris.length; n++) out.set(pos.subarray(9 * tris[n], 9 * tris[n] + 9), 9 * n);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(out, 3));
    return g;
  }

  setHover(pos, tris, color) {
    this.hover.geometry.dispose();
    if (!tris) { this.hover.visible = false; return; }
    this.hover.geometry = View.trisGeometry(pos, tris);
    this.hover.material.color.set(color);
    this.hover.visible = true;
  }

  // conditions: [{kind: "fixed" | "load", tris, force: [x, y, z]}]
  setMarks(pos, conditions) {
    for (const c of [...this.marks.children]) { if (c.geometry) c.geometry.dispose(); this.marks.remove(c); }
    for (const c of conditions) {
      const color = c.kind === "fixed" ? 0x35c26b : 0xf08a3c;
      const g = View.trisGeometry(pos, c.tris);
      const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: 0.7, side: THREE.DoubleSide, depthWrite: false,
        polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
      }));
      m.renderOrder = 2;
      this.marks.add(m);
      if (c.kind === "load") {
        g.computeBoundingBox();
        const centre = g.boundingBox.getCenter(new THREE.Vector3());
        const dir = new THREE.Vector3(...c.force);
        if (dir.lengthSq() > 0) {
          dir.normalize();
          const len = this.size * 0.45;
          // arrow ends at the loaded face, like a force pushing on it
          this.marks.add(new THREE.ArrowHelper(dir, centre.clone().addScaledVector(dir, -len), len, color, len * 0.28, len * 0.13));
        }
      }
    }
  }

  setResult(skin) {
    this.result.geometry.dispose();
    if (!skin || !skin.positions.length) { this.result.visible = false; return; }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(skin.positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(skin.colors, 3));
    g.computeVertexNormals();
    this.result.geometry = g;
    this.result.visible = true;
  }

  // Draws black cell edges over the result mesh (each cell face is two triangles: corners 0, 1, 2, 5).
  setEdges(skin) {
    if (!this.edges) {
      this.edges = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x0b0f16, transparent: true, opacity: 0.55 }));
      this.scene.add(this.edges);
      this.result.material.polygonOffset = true; this.result.material.polygonOffsetFactor = 1; this.result.material.polygonOffsetUnits = 1;
    }
    this.edges.geometry.dispose();
    this.edges.geometry = new THREE.BufferGeometry();
    this.edges.visible = !!(skin && skin.positions.length);
    if (!this.edges.visible) return;
    const p = skin.positions, quads = p.length / 18, e = new Float32Array(quads * 24), order = [0, 1, 1, 2, 2, 5, 5, 0];
    for (let q = 0; q < quads; q++) for (let k = 0; k < 8; k++) {
      const src = 18 * q + 3 * order[k], dst = 24 * q + 3 * k;
      e[dst] = p[src]; e[dst + 1] = p[src + 1]; e[dst + 2] = p[src + 2];
    }
    this.edges.geometry.setAttribute("position", new THREE.BufferAttribute(e, 3));
  }

  setSmooth(positions) {
    this.smooth.geometry.dispose();
    if (!positions || !positions.length) { this.smooth.visible = false; return; }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.computeVertexNormals();
    this.smooth.geometry = g;
    this.smooth.visible = true;
  }

  screenshot() {
    this.renderer.render(this.scene, this.camera);
    return this.canvas.toDataURL("image/png");
  }
}
