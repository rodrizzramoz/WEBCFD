// Writes an OpenFOAM case for the current flow setup, so the same problem can
// be solved with a conventional body-fitted finite volume solver (simpleFoam
// with a snappyHexMesh mesh) and compared with the in-browser result.
//
// Written for the OpenFOAM.com series (checked with v1912). No browser APIs
// are used here, so the module also runs under Node for testing.
import { buildDomain, domainBox, FACE_NAMES, NORMALS } from "./domain.js";

const HEADER = (cls, object, location) => `/*--------------------------------*- C++ -*----------------------------------*\\
  Flowbench export for OpenFOAM
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       ${cls};
    ${location ? "location    \"" + location + "\";\n    " : ""}object      ${object};
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
`;
const PATCH = ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"];
const v3 = (a) => "(" + a.map((x) => Number(x.toPrecision(9))).join(" ") + ")";

function binarySTL(tris) {
  const nt = tris.length / 9, buf = new ArrayBuffer(84 + 50 * nt), dv = new DataView(buf);
  dv.setUint32(80, nt, true);
  for (let t = 0; t < nt; t++) {
    const o = 9 * t, b = 84 + 50 * t;
    const ux = tris[o + 3] - tris[o], uy = tris[o + 4] - tris[o + 1], uz = tris[o + 5] - tris[o + 2];
    const vx = tris[o + 6] - tris[o], vy = tris[o + 7] - tris[o + 1], vz = tris[o + 8] - tris[o + 2];
    const n = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx], l = Math.hypot(...n) || 1;
    for (let a = 0; a < 3; a++) dv.setFloat32(b + 4 * a, n[a] / l, true);
    for (let k = 0; k < 9; k++) dv.setFloat32(b + 12 + 4 * k, tris[o + k], true);
  }
  return new Uint8Array(buf);
}

// tris: model triangles in metres (as simulated). opt: the same options object the
// app passes to buildDomain. phys: {rho, mu, speed}. Returns {files, summary}.
export function buildCase(tris, opt, phys, settings = {}) {
  const level = settings.level ?? 2;                 // surface refinement levels above the background mesh
  const iterations = settings.iterations ?? 1500;
  const external = opt.mode === "external";
  const nu = phys.mu / phys.rho;
  // A coarse copy of the app's own domain tells us where the fluid is and where the ports are.
  const probeOpt = { ...opt, res: Math.min(opt.res, 96), walls: "cubes" };
  const grid = buildDomain(tris, probeOpt);
  const box = domainBox(tris, opt);
  const dxFine = box.dx, dxBase = dxFine * 2 ** level;
  let min = box.origin.slice(), max = box.origin.map((o, a) => o + box.dims[a] * box.dx);
  if (!external) {
    // Pull every face slightly inside the part so the part pokes through the box and the
    // openings become clean patches on the box faces.
    const inset = 0.3 * grid.dx;
    min = grid.bb.min.map((v) => v + inset); max = grid.bb.max.map((v) => v - inset);
  }
  const size = max.map((v, a) => v - min[a]);
  const cells = size.map((s) => Math.max(4, Math.round(s / dxBase)));

  // A point that is certainly in the fluid.
  let inside;
  if (external) inside = min.map((v, a) => v + size[a] * [0.07, 0.11, 0.13][a]);
  else {
    const { nx, ny, nz, flags, dx, origin } = grid;
    let best = -1, bestScore = -1;
    for (let k = 2; k < nz - 2; k++) for (let j = 2; j < ny - 2; j++) for (let i = 2; i < nx - 2; i++) {
      const c = i + nx * (j + ny * k);
      if ((flags[c] & 7) !== 0) continue;
      let r = 0; // how many layers of fluid surround this cell, up to 4
      outer: for (r = 1; r <= 4; r++) {
        for (const [a, b, d] of [[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]]) {
          const ii = i + a, jj = j + b, kk = k + d;
          if (ii < 0 || jj < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= nz || (flags[ii + nx * (jj + ny * kk)] & 7) === 1) break outer;
        }
      }
      if (r > bestScore) { bestScore = r; best = c; if (r > 4) break; }
    }
    if (best < 0) throw new Error("No fluid region was found to export.");
    const i = best % nx, j = ((best / nx) | 0) % ny, k = (best / (nx * ny)) | 0;
    inside = [origin[0] + (i + 0.537) * dx, origin[1] + (j + 0.519) * dx, origin[2] + (k + 0.523) * dx];
  }

  // Boundary roles. Each entry: {name, kind: "inlet" | "open" | "farfield" | "wall", speed, normal, box?}
  const patches = [];
  const sets = []; // port rectangles to carve out of box faces (custom port mode)
  let refSpeed = phys.speed, refLength, refArea = 1;
  if (external) {
    refLength = grid.refLength; refArea = grid.frontalArea;
    PATCH.forEach((name, f) => patches.push({ name, kind: f === 0 ? "inlet" : f === 1 ? "open" : "farfield", speed: phys.speed, normal: [1, 0, 0] }));
  } else if (grid.portMode) {
    refSpeed = grid.refSpeed; refLength = grid.refLength;
    PATCH.forEach((name) => patches.push({ name, kind: "wall" }));
    const { nx, ny, dx, origin } = grid;
    grid.ports.map((p, n) => ({ p, n })).sort((a, b) => a.p.area - b.p.area).forEach(({ p, n }) => {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const c of p.nbr) {
        const ijk = [c % nx, ((c / nx) | 0) % ny, (c / (nx * ny)) | 0];
        for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], origin[a] + ijk[a] * dx); hi[a] = Math.max(hi[a], origin[a] + (ijk[a] + 1) * dx); }
      }
      const a = p.face >> 1, plane = p.face & 1 ? max[a] : min[a];
      lo[a] = plane - 0.2 * dxFine; hi[a] = plane + 0.2 * dxFine;
      for (let b = 0; b < 3; b++) if (b !== a) { lo[b] -= 0.6 * dx; hi[b] += 0.6 * dx; }
      const name = (p.type === "feed" ? "feed" : "open") + (n + 1);
      sets.push({ name, lo, hi, base: PATCH[p.face] });
      patches.push({ name, kind: p.type === "feed" ? "inlet" : "open", speed: p.speed, pressure: p.pressure, normal: NORMALS[p.face] });
    });
  } else {
    refLength = grid.refLength;
    PATCH.forEach((name, f) => patches.push(f === opt.inletFace ? { name, kind: "inlet", speed: phys.speed, normal: NORMALS[f] }
      : f === opt.outletFace ? { name, kind: "open" } : { name, kind: "wall" }));
  }
  patches.push({ name: "model", kind: "wall" });

  const Re = refSpeed * refLength / nu;
  const laminar = Re < 2300;
  const k0 = 1.5 * (0.05 * refSpeed) ** 2, omega0 = Math.sqrt(k0) / (0.09 ** 0.25 * 0.07 * refLength);
  const inlets = patches.filter((p) => p.kind === "inlet"), opens = patches.filter((p) => p.kind === "open");

  const field = (name, cls, dims, internal, per) => HEADER(cls, name, "0") + `
dimensions      ${dims};
internalField   uniform ${internal};

boundaryField
{
${patches.map((p) => "    " + p.name + "\n    {\n" + per(p).split("\n").map((l) => "        " + l).join("\n") + "\n    }").join("\n")}
}
`;
  const files = {};
  const startU = external ? [phys.speed, 0, 0] : [0, 0, 0];
  files["0/U"] = field("U", "volVectorField", "[0 1 -1 0 0 0 0]", v3(startU), (p) =>
    p.kind === "inlet" ? "type            fixedValue;\nvalue           uniform " + v3(p.normal.map((c) => c * p.speed)) + ";"
    : p.kind === "open" ? "type            pressureInletOutletVelocity;\nvalue           uniform (0 0 0);"
    : p.kind === "farfield" ? "type            slip;"
    : "type            noSlip;");
  files["0/p"] = field("p", "volScalarField", "[0 2 -2 0 0 0 0]", "0", (p) =>
    p.kind === "open" ? "type            totalPressure;\np0              uniform 0;\nvalue           uniform 0;"
    : p.kind === "farfield" ? "type            slip;" : "type            zeroGradient;");
  if (external) {
    // a plain outflow is steadier than a total-pressure opening behind a bluff body
    files["0/U"] = files["0/U"].replace("type            pressureInletOutletVelocity;\n        value           uniform (0 0 0);", "type            inletOutlet;\n        inletValue      uniform (0 0 0);\n        value           uniform " + v3(startU) + ";");
    files["0/p"] = files["0/p"].replace("type            totalPressure;\n        p0              uniform 0;\n        value           uniform 0;", "type            fixedValue;\n        value           uniform 0;");
  }
  if (!laminar) {
    const turb = (name, value, wallFn) => field(name, "volScalarField", name === "k" ? "[0 2 -2 0 0 0 0]" : name === "omega" ? "[0 0 -1 0 0 0 0]" : "[0 2 -1 0 0 0 0]", String(value), (p) =>
      p.kind === "wall" ? "type            " + wallFn + ";\nvalue           uniform " + value + ";"
      : p.kind === "farfield" ? "type            slip;"
      : name === "nut" ? "type            calculated;\nvalue           uniform 0;"
      : p.kind === "inlet" ? "type            fixedValue;\nvalue           uniform " + value + ";"
      : "type            inletOutlet;\ninletValue      uniform " + value + ";\nvalue           uniform " + value + ";");
    files["0/k"] = turb("k", Number(k0.toPrecision(5)), "kqRWallFunction");
    files["0/omega"] = turb("omega", Number(omega0.toPrecision(5)), "omegaWallFunction");
    files["0/nut"] = turb("nut", 0, "nutkWallFunction");
  }

  files["constant/transportProperties"] = HEADER("dictionary", "transportProperties", "constant") + `
transportModel  Newtonian;
nu              ${Number(nu.toPrecision(6))};
`;
  files["constant/turbulenceProperties"] = HEADER("dictionary", "turbulenceProperties", "constant") + (laminar ? `
simulationType  laminar;
` : `
simulationType  RAS;
RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
`);
  files["constant/triSurface/model.stl"] = binarySTL(tris);

  const c = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]].map((q) => q.map((s, a) => (s ? max[a] : min[a])));
  const faceVerts = ["(0 4 7 3)", "(1 2 6 5)", "(0 1 5 4)", "(3 7 6 2)", "(0 3 2 1)", "(4 5 6 7)"];
  files["system/blockMeshDict"] = HEADER("dictionary", "blockMeshDict", "system") + `
scale 1;
vertices
(
${c.map((q) => "    " + v3(q)).join("\n")}
);
blocks ( hex (0 1 2 3 4 5 6 7) (${cells.join(" ")}) simpleGrading (1 1 1) );
edges ();
boundary
(
${PATCH.map((name, f) => `    ${name} { type ${patches[f].kind === "wall" ? "wall" : "patch"}; faces ( ${faceVerts[f]} ); }`).join("\n")}
);
`;
  files["system/snappyHexMeshDict"] = HEADER("dictionary", "snappyHexMeshDict", "system") + `
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{
    model.stl { type triSurfaceMesh; name model; }
}

castellatedMeshControls
{
    maxLocalCells       20000000;
    maxGlobalCells      40000000;
    minRefinementCells  10;
    nCellsBetweenLevels 3;
    features            ();
    refinementSurfaces
    {
        model { level (${level} ${level}); patchInfo { type wall; } }
    }
    resolveFeatureAngle 30;
    refinementRegions   {}
    locationInMesh      ${v3(inside)};
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      50;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;
    layers {}
    expansionRatio 1.2;
    finalLayerThickness 0.4;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}

meshQualityControls
{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality -1e30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    relaxed { maxNonOrtho 75; }
}

mergeTolerance 1e-6;
`;
  if (sets.length) {
    const done = [];
    files["system/topoSetDict"] = HEADER("dictionary", "topoSetDict", "system") + `
actions
(
${sets.map((s) => {
      const lines = [`    { name ${s.name}; type faceSet; action new; source patchToFace; patch ${s.base}; }`,
        `    { name ${s.name}Box; type faceSet; action new; source boxToFace; box ${v3(s.lo)} ${v3(s.hi)}; }`,
        `    { name ${s.name}; type faceSet; action subset; source faceToFace; set ${s.name}Box; }`];
      for (const d of done) lines.push(`    { name ${s.name}; type faceSet; action subtract; source faceToFace; set ${d}; }`);
      done.push(s.name);
      return lines.join("\n");
    }).join("\n")}
);
`;
    files["system/createPatchDict"] = HEADER("dictionary", "createPatchDict", "system") + `
pointSync false;
patches
(
${sets.map((s) => `    { name ${s.name}; patchInfo { type patch; } constructFrom set; set ${s.name}; }`).join("\n")}
);
`;
  }

  // Reporting
  const fo = [];
  if (external) {
    fo.push(`    forces
    {
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (model);
        rho             rhoInf;
        rhoInf          ${phys.rho};
        CofR            (0 0 0);
        liftDir         (0 0 1);
        dragDir         (1 0 0);
        pitchAxis       (0 1 0);
        magUInf         ${phys.speed};
        lRef            ${Number(refLength.toPrecision(6))};
        Aref            ${Number(refArea.toPrecision(6))};
    }`);
  } else {
    for (const p of [...inlets, ...opens]) {
      fo.push(`    flow_${p.name}
    {
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        log             false;
        regionType      patch;
        name            ${p.name};
        operation       sum;
        fields          (phi);
    }
    pressure_${p.name}
    {
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        log             false;
        regionType      patch;
        name            ${p.name};
        operation       areaAverage;
        fields          (p);
    }`);
    }
  }
  files["system/controlDict"] = HEADER("dictionary", "controlDict", "system") + `
application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         ${iterations};
deltaT          1;
writeControl    timeStep;
writeInterval   ${iterations};
purgeWrite      1;
writeFormat     ${settings.ascii ? "ascii" : "binary"};
writePrecision  8;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{
${settings.functions === false ? "" : fo.join("\n")}
}
`;
  files["system/fvSchemes"] = HEADER("dictionary", "fvSchemes", "system") + `
ddtSchemes      { default steadyState; }
gradSchemes     { default cellLimited Gauss linear 1; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited 0.5; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited 0.5; }
wallDist        { method meshWave; }
`;
  files["system/fvSolution"] = HEADER("dictionary", "fvSolution", "system") + `
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.05;
    }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;
    residualControl { p 1e-5; U 1e-6; "(k|omega)" 1e-6; }
}
relaxationFactors
{
    equations { U 0.8; ".*" 0.7; }
    fields    { p 0.5; }
}
`;
  files["Allrun"] = `#!/bin/sh
# Mesh and solve. Needs OpenFOAM (the openfoam.com series, v1912 or newer) on the PATH.
cd "$(dirname "$0")" || exit 1
set -e
blockMesh > log.blockMesh 2>&1
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
${sets.length ? "topoSet > log.topoSet 2>&1\ncreatePatch -overwrite > log.createPatch 2>&1\n" : ""}checkMesh > log.checkMesh 2>&1 || true
simpleFoam > log.simpleFoam 2>&1
echo "Done. Results are in postProcessing/ and the last time folder."
`;
  const compressible = !!settings.compressible;
  if (compressible && external) throw new Error("The compressible export covers flow through a part for now. Use Through the part with custom ports.");
  const summary = {
    compressible,
    mode: external ? "external" : grid.portMode ? "ports" : "internal", reynolds: Re, laminar,
    background: cells, backgroundCell: dxBase, surfaceCell: dxFine, level, inside,
    patches: patches.map((p) => ({ name: p.name, kind: p.kind, speed: p.speed || 0 })),
    refLength, refArea, refSpeed, rho: compressible ? RHO_ATM : phys.rho,
  };
  files["README.txt"] = `Flowbench export for OpenFOAM

What this is
  The same flow problem you set up in Flowbench, written as an OpenFOAM case:
  steady incompressible flow (simpleFoam), ${laminar ? "laminar" : "k-omega SST turbulence model with wall functions"},
  on a snappyHexMesh mesh that is snapped to the real surface of the model.

How to run it
  1. Install OpenFOAM (openfoam.com series, v1912 or newer). On Windows use WSL or Docker.
  2. In a terminal inside this folder:  sh Allrun
  3. Read the result:
${external ? "       postProcessing/forces/0/coefficient.dat   (Cd is the drag coefficient, Cl the lift coefficient)"
    : "       postProcessing/flow_<patch>/0/surfaceFieldValue.dat      (volume flow in m3/s, negative means into the domain)\n       postProcessing/pressure_<patch>/0/surfaceFieldValue.dat  (static pressure divided by density, in m2/s2; multiply by " + phys.rho + " for Pa)"}
  4. Open the case in ParaView (paraFoam, or create an empty file named case.foam and open it).

Setup
  Reynolds number ${Math.round(Re)}, reference speed ${refSpeed} m/s, density ${phys.rho} kg/m3, kinematic viscosity ${nu.toExponential(3)} m2/s.
  Background cells ${cells.join(" x ")} of ${(dxBase * 1000).toPrecision(3)} mm, refined ${level} levels to ${(dxFine * 1000).toPrecision(3)} mm at the model surface.
  Boundaries: ${patches.map((p) => p.name + " = " + (p.kind === "inlet" ? "inlet at " + p.speed + " m/s" : p.kind === "open" ? "open to atmosphere" : p.kind === "farfield" ? "slip wall (far field)" : "wall")).join(", ")}.

Notes
  This flow is treated as incompressible. Air jets faster than about 100 m/s need a compressible
  solver (rhoSimpleFoam or rhoPimpleFoam) with pressure and temperature boundary conditions instead.
  To add boundary layer cells, set addLayers true in system/snappyHexMeshDict and list the model patch under layers.
`;
  if (compressible) compressibleCase(files, patches, { k0, omega0, dxFine, iterations: settings.iterations ?? 4000 });
  return { files, summary };
}

// Rewrites an incompressible case as compressible air (perfect gas), solved with
// rhoPimpleFoam in local time stepping mode, which marches to a steady state and
// copes with sonic and supersonic jets. Pressures become absolute.
// Fed ports given as a supply pressure become reservoirs at that total pressure;
// fed ports given as a speed keep a fixed velocity. Open ports sit at atmospheric pressure.
const P_ATM = 101325, T_ATM = 293.15, RHO_ATM = P_ATM / (287.05 * T_ATM);
function compressibleCase(files, patches, o) {
  const field = (name, cls, dims, internal, per) => HEADER(cls, name, "0") + `
dimensions      ${dims};
internalField   uniform ${internal};

boundaryField
{
${patches.map((p) => "    " + p.name + "\n    {\n" + per(p).split("\n").map((l) => "        " + l).join("\n") + "\n    }").join("\n")}
}
`;
  const byPressure = (p) => p.kind === "inlet" && p.pressure > 0;
  files["0/p"] = field("p", "volScalarField", "[1 -1 -2 0 0 0 0]", String(P_ATM), (p) =>
    // the supply pressure is brought up over the first 500 iterations; switching it on at once makes the solver diverge
    byPressure(p) ? "type            uniformTotalPressure;\ngamma           1.4;\np0              table ((0 " + Math.round(P_ATM + 0.03 * p.pressure) + ") (500 " + Math.round(P_ATM + p.pressure) + ") (10000000 " + Math.round(P_ATM + p.pressure) + "));\nvalue           uniform " + Math.round(P_ATM + 0.03 * p.pressure) + ";"
    : p.kind === "open" ? "type            totalPressure;\ngamma           1.4;\np0              uniform " + P_ATM + ";\nvalue           uniform " + P_ATM + ";"
    : "type            zeroGradient;");
  files["0/U"] = field("U", "volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)", (p) =>
    byPressure(p) ? "type            pressureInletVelocity;\nvalue           uniform (0 0 0);"
    : p.kind === "inlet" ? "type            fixedValue;\nvalue           uniform " + v3(p.normal.map((c) => c * p.speed)) + ";"
    : p.kind === "open" ? "type            pressureInletOutletVelocity;\nvalue           uniform (0 0 0);"
    : "type            noSlip;");
  files["0/T"] = field("T", "volScalarField", "[0 0 0 1 0 0 0]", String(T_ATM), (p) =>
    byPressure(p) ? "type            totalTemperature;\ngamma           1.4;\nT0              uniform " + T_ATM + ";\nvalue           uniform " + T_ATM + ";"
    : p.kind === "inlet" ? "type            fixedValue;\nvalue           uniform " + T_ATM + ";"
    : p.kind === "open" ? "type            inletOutlet;\ninletValue      uniform " + T_ATM + ";\nvalue           uniform " + T_ATM + ";"
    : "type            zeroGradient;");
  const turb = (name, dims, value, wallFn) => field(name, "volScalarField", dims, String(value), (p) =>
    p.kind === "wall" ? "type            " + wallFn + ";\nvalue           uniform " + value + ";"
    : name === "nut" || name === "alphat" ? "type            calculated;\nvalue           uniform 0;"
    : p.kind === "inlet" ? "type            fixedValue;\nvalue           uniform " + value + ";"
    : "type            inletOutlet;\ninletValue      uniform " + value + ";\nvalue           uniform " + value + ";");
  files["0/k"] = turb("k", "[0 2 -2 0 0 0 0]", Number(o.k0.toPrecision(5)), "kqRWallFunction");
  files["0/omega"] = turb("omega", "[0 0 -1 0 0 0 0]", Number(o.omega0.toPrecision(5)), "omegaWallFunction");
  files["0/nut"] = turb("nut", "[0 2 -1 0 0 0 0]", 0, "nutkWallFunction");
  files["0/alphat"] = turb("alphat", "[1 -1 -1 0 0 0 0]", 0, "compressible::alphatWallFunction");
  delete files["constant/transportProperties"];
  files["constant/thermophysicalProperties"] = HEADER("dictionary", "thermophysicalProperties", "constant") + `
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}
mixture
{
    specie          { molWeight 28.96; }
    thermodynamics  { Cp 1005; Hf 0; }
    transport       { mu 1.825e-05; Pr 0.71; }
}
`;
  files["constant/turbulenceProperties"] = HEADER("dictionary", "turbulenceProperties", "constant") + `
simulationType  RAS;
RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
`;
  files["system/fvSchemes"] = HEADER("dictionary", "fvSchemes", "system") + `
ddtSchemes      { default localEuler; }
gradSchemes     { default cellLimited Gauss linear 1; }
divSchemes
{
    default         none;
    div(phi,U)      Gauss linearUpwind grad(U);
    div(phi,e)      Gauss limitedLinear 1;
    div(phi,K)      Gauss limitedLinear 1;
    div(phi,Ekp)    Gauss limitedLinear 1;
    div(phiv,p)     Gauss limitedLinear 1;
    div(phid,p)     Gauss limitedLinear 1;
    div(phi,k)      Gauss upwind;
    div(phi,omega)  Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited 0.5; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited 0.5; }
wallDist        { method meshWave; }
`;
  files["system/fvSolution"] = HEADER("dictionary", "fvSolution", "system") + `
solvers
{
    "(p|rho)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.05;
    }
    "(p|rho)Final"
    {
        $p;
        relTol          0;
    }
    "(U|e|k|omega)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.1;
    }
    "(U|e|k|omega)Final"
    {
        $U;
        relTol          0;
    }
}
PIMPLE
{
    momentumPredictor yes;
    transonic       yes;
    nOuterCorrectors 1;
    nCorrectors     2;
    nNonOrthogonalCorrectors 1;
    maxCo           0.2;
    rDeltaTSmoothingCoeff 0.1;
    rDeltaTDampingCoeff 1;
    maxDeltaT       ${(10 * o.dxFine / 343).toExponential(2)};
    pMinFactor      0.1;
    pMaxFactor      12;
}
relaxationFactors
{
    equations { ".*" 1; }
}
`;
  files["system/controlDict"] = files["system/controlDict"].replace("application     simpleFoam;", "application     rhoPimpleFoam;")
    .replace(/endTime         \d+;/, "endTime         " + o.iterations + ";").replace(/writeInterval   \d+;/, "writeInterval   " + o.iterations + ";");
  files["Allrun"] = files["Allrun"].replace("simpleFoam > log.simpleFoam 2>&1", "rhoPimpleFoam > log.rhoPimpleFoam 2>&1");
  files["README.txt"] = files["README.txt"].replace("steady incompressible flow (simpleFoam)", "compressible air as a perfect gas, marched to a steady state (rhoPimpleFoam with local time stepping)")
    .replace(/Notes\n[\s\S]*?To add boundary/, "Notes\n  Pressures in this case are absolute, in pascals. Flow rates on patches are mass flows in kg/s.\n  To add boundary");
}

export { FACE_NAMES };
