// Built-in example models. All are generated in millimetres.
import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

function ahmedBody() {
  // Ahmed reference body, 25 degree rear slant: 1044 x 389 x 288 mm.
  const s = new THREE.Shape();
  s.moveTo(1044, 0);
  s.lineTo(1044, 194);
  s.lineTo(843, 288);
  s.lineTo(100, 288);
  s.quadraticCurveTo(0, 288, 0, 188);
  s.lineTo(0, 100);
  s.quadraticCurveTo(0, 0, 100, 0);
  s.closePath();
  const g = new THREE.ExtrudeGeometry(s, { depth: 389, bevelEnabled: false, curveSegments: 16 });
  g.rotateX(Math.PI / 2);
  return g;
}

function sphere() {
  return new THREE.SphereGeometry(50, 64, 40);
}

function venturi() {
  // Tube 120 mm long, 20 mm bore necking down to a 10 mm throat, 30 mm outside diameter.
  const bore = [[10, 0], [10, 25], [5, 45], [5, 55], [10, 100], [10, 120]];
  const pts = bore.map(([r, y]) => new THREE.Vector2(r, y));
  pts.push(new THREE.Vector2(15, 120), new THREE.Vector2(15, 0), new THREE.Vector2(10, 0));
  const g = new THREE.LatheGeometry(pts, 72);
  g.rotateZ(-Math.PI / 2);
  return g;
}

function elbow() {
  // 90 degree pipe bend, 16 mm bore, as the fluid volume itself.
  const path = new THREE.CurvePath();
  path.add(new THREE.LineCurve3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(40, 0, 0)));
  path.add(new THREE.QuadraticBezierCurve3(new THREE.Vector3(40, 0, 0), new THREE.Vector3(70, 0, 0), new THREE.Vector3(70, 0, 30)));
  path.add(new THREE.LineCurve3(new THREE.Vector3(70, 0, 30), new THREE.Vector3(70, 0, 70)));
  const tube = new THREE.TubeGeometry(path, 80, 8, 32, false).toNonIndexed();
  // TubeGeometry has open ends; cap them so the volume is watertight.
  const capIn = new THREE.CircleGeometry(8, 32).rotateY(-Math.PI / 2).toNonIndexed();
  const capOut = new THREE.CircleGeometry(8, 32).translate(70, 0, 70).toNonIndexed();
  return mergeGeometries([tube, capIn, capOut]);
}

function pipe() {
  // Straight pipe, 20 mm bore, 100 mm long, as the fluid volume. With oil at low
  // speed this is laminar pipe flow, which has an exact answer to check against:
  // pressure drop = 32 x viscosity x length x mean speed / diameter squared.
  return new THREE.CylinderGeometry(10, 10, 100, 64).rotateZ(-Math.PI / 2).translate(50, 0, 0);
}

function jetPump() {
  // Air mover: a 30 mm bore tube with a small 6 mm nozzle tube on its axis.
  // Compressed air fed through the nozzle drags surrounding air in through the ring around it.
  const tube = (rIn, rOut, x0, x1) => {
    const pts = [[rIn, x0], [rIn, x1], [rOut, x1], [rOut, x0], [rIn, x0]].map(([r, y]) => new THREE.Vector2(r, y));
    return new THREE.LatheGeometry(pts, 64).rotateZ(-Math.PI / 2).toNonIndexed();
  };
  return mergeGeometries([tube(15, 20, 0, 140), tube(3, 5.5, 0, 35)]);
}

export const DEMOS = {
  ahmed: {
    label: "Ahmed car body (external)", build: ahmedBody,
    setup: { mode: "external", fluid: "air", speed: 40 },
  },
  sphere: {
    label: "Sphere, 100 mm (external)", build: sphere,
    setup: { mode: "external", fluid: "air", speed: 10 },
  },
  cube: {
    label: "Cube, 100 mm, face on (external)", build: () => new THREE.BoxGeometry(100, 100, 100),
    setup: { mode: "external", fluid: "air", speed: 10 },
  },
  venturi: {
    label: "Venturi tube (internal)", build: venturi,
    setup: { mode: "internal", fluid: "water", speed: 1, inletFace: 0, outletFace: 1, stlIsFluid: false },
  },
  pipe: {
    label: "Straight pipe, oil, laminar check (internal)", build: pipe,
    setup: { mode: "internal", fluid: "oil", speed: 0.5, inletFace: 0, outletFace: 1, stlIsFluid: true },
  },
  jetpump: {
    label: "Air mover with a compressed-air nozzle (custom ports)", build: jetPump,
    setup: {
      mode: "internal", fluid: "air", speed: 30, inletFace: 0, outletFace: 1, stlIsFluid: false, portMode: "ports",
      ports: [
        { face: 0, point: [0, 0, 0], type: "feed", speed: 30 },
        { face: 0, point: [0, 0.0105, 0], type: "open", speed: 0 },
        { face: 1, point: [0.14, 0, 0], type: "open", speed: 0 },
      ],
    },
  },
  elbow: {
    label: "Pipe elbow, fluid volume (internal)", build: elbow,
    setup: { mode: "internal", fluid: "water", speed: 1, inletFace: 0, outletFace: 5, stlIsFluid: true },
  },
};
