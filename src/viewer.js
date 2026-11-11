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
      g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      this.model.material = this.colorMat;
    } else {
      this.model.material = this.solidMat;
    }
  }

  // Draws the solver box, plus the inlet and outlet faces.
  setDomain(box, mode, inletFace, outletFace) {
    this.domain.clear();
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
      const len = 0.18 * Math.max(...size);
      const n = NORMALS[f], dir = new THREE.Vector3(n[0], n[1], n[2]);
      const isInlet = color === 0x35c26b;
      const start = new THREE.Vector3(p[0], p[1], p[2]).addScaledVector(dir, isInlet ? -len : 0);
      if (!isInlet) dir.negate();
      this.domain.add(new THREE.ArrowHelper(dir, start, len, color, len * 0.3, len * 0.14));
    };
    face(mode === "external" ? 0 : inletFace, 0x35c26b);
    face(mode === "external" ? 1 : outletFace, 0xf08a3c);
    this.box = { c, size };
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
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(corners), 3));
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

  screenshot() {
    this.renderer.render(this.scene, this.camera);
    return this.canvas.toDataURL("image/png");
  }
}
