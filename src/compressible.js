// Compressible gas flow solver for sonic and supersonic air (nozzles, ejectors, air movers),
// running as WebGPU compute shaders. See docs/COMPRESSIBLE.md for the method and its limits.
//
// Method in short: cell-centred finite volume on the uniform grid from buildDomain,
// conserved variables (density, momentum, total energy), second-order MUSCL reconstruction
// of the primitive variables with the monotonised central limiter, HLLC flux, laminar plus Smagorinsky
// viscosity and heat conduction, two-stage strong-stability-preserving Runge-Kutta in time.
//
// Units inside the shader are non-dimensional: pressure in units of the ambient pressure,
// density in units of the ambient density, length in cells, speed in units of
// sqrt(ambient pressure / ambient density). Ambient air at rest is then density 1,
// pressure 1, temperature 1 and speed of sound sqrt(gamma), which keeps 32-bit floats accurate.

const K = 44;        // diagnostic channels per row: 12 general ones, then 4 for each of 8 ports
const KBASE = 12;
const CELL = 6;      // floats stored per cell: density, momentum x y z, total energy, eddy viscosity

// Shared by every shader: the parameter block and the gas dynamics that need no grid access.
const PHYSICS = /* wgsl */ `
struct Params {
  dims: vec4u,             // nx, ny, nz, then: number of port cells (step and ghost passes) or mode (diagnostics)
  a: vec4f,                // time step, Runge-Kutta weight of the old state, weight of the new state, gamma
  b: vec4f,                // laminar viscosity, (Smagorinsky constant x cell size) squared, 1 / Prandtl, 1 / turbulent Prandtl
  ff: vec4f,               // far field (and starting) state: density, pressure
  ffU: vec4f,              // far field velocity
  ports: array<vec4f, 8>,  // kind (0 = fed from a reservoir, 1 = open), pressure, temperature, relaxation of an opening
};
@group(0) @binding(0) var<uniform> P: Params;

struct Prim { r: f32, u: vec3f, p: f32 };
struct Flux { m: f32, mom: vec3f, e: f32 };

const AX = array<vec3i, 3>(vec3i(1, 0, 0), vec3i(0, 1, 0), vec3i(0, 0, 1));
const NRM = array<vec3i, 6>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1));

fn mirror(W: Prim, e: vec3f) -> Prim { return Prim(W.r, W.u - 2.0 * dot(W.u, e) * e, W.p); }

// Inflow from a reservoir at total pressure p0 and total temperature T0 through a face
// with inward normal n. The wave leaving the domain (Riemann invariant of the cell inside)
// and the total enthalpy of the reservoir fix the inflow speed. The inflow is at most sonic.
// If the inside pushes back harder than the reservoir, the face acts as a wall.
fn reservoir(W: Prim, n: vec3f, p0: f32, T0: f32) -> Prim {
  let g = P.a.w;
  let gm = g - 1.0;
  let ci = sqrt(g * W.p / W.r);
  let J = dot(W.u, n) - 2.0 * ci / gm;
  let c02 = g * T0;
  let qa = g + 1.0;
  let qb = 2.0 * J * gm;
  let qc = 0.5 * gm * gm * J * J - gm * c02;
  var c = (-qb + sqrt(max(qb * qb - 4.0 * qa * qc, 0.0))) / (2.0 * qa);
  var u = J + 2.0 * c / gm;
  if (u <= 0.0) { return mirror(W, n); }
  if (u > c) { c = sqrt(2.0 * c02 / (g + 1.0)); u = c; }
  let T = c * c / g;
  let p = p0 * pow(T / T0, g / gm);
  return Prim(p / T, n * u, p);
}

// Opening to surroundings at static pressure pb and temperature Tb.
// For air leaving slower than sound the outside pressure is not forced on the face at once:
// the face pressure is pulled towards pb by the fraction relax each time. In steady flow the
// pressure at the opening still ends up exactly pb, but pressure waves pass out instead of
// bouncing back, so a duct with open ends does not ring like an organ pipe. relax = 1 gives
// the plain fixed-pressure opening.
fn openGhost(W: Prim, n: vec3f, pb: f32, Tb: f32, relax: f32) -> Prim {
  let g = P.a.w;
  let un = dot(W.u, n);
  if (un > 0.0) { return reservoir(W, n, pb, Tb); }   // drawing air in from still surroundings
  let ci = sqrt(g * W.p / W.r);
  if (-un >= ci) { return W; }                        // leaving faster than sound: nothing travels back in
  let pg = W.p + relax * (pb - W.p);                  // leaving slower than sound: the outside pressure is felt
  let rg = W.r * pow(pg / W.p, 1.0 / g);
  let cg = sqrt(g * pg / rg);
  let q = -un + 2.0 * (ci - cg) / (g - 1.0);
  return Prim(rg, W.u - n * (un + q), pg);
}

// HLLC approximate Riemann solver. Returns the flux in direction e.
fn hllc(L: Prim, R: Prim, e: vec3f) -> Flux {
  let g = P.a.w;
  let gi = 1.0 / (g - 1.0);
  let qL = dot(L.u, e);
  let qR = dot(R.u, e);
  let cL = sqrt(g * L.p / L.r);
  let cR = sqrt(g * R.p / R.r);
  let EL = L.p * gi + 0.5 * L.r * dot(L.u, L.u);
  let ER = R.p * gi + 0.5 * R.r * dot(R.u, R.u);
  // bounds on the fastest waves to the left and right (Davis)
  let SL = min(qL - cL, qR - cR);
  let SR = max(qL + cL, qR + cR);
  let FL = Flux(L.r * qL, L.r * qL * L.u + L.p * e, qL * (EL + L.p));
  if (SL >= 0.0) { return FL; }
  let FR = Flux(R.r * qR, R.r * qR * R.u + R.p * e, qR * (ER + R.p));
  if (SR <= 0.0) { return FR; }
  let dL = L.r * (SL - qL);
  let dR = R.r * (SR - qR);
  let SM = (R.p - L.p + qL * dL - qR * dR) / (dL - dR);
  if (SM >= 0.0) {
    let k = dL / (SL - SM);
    let m = k * (L.u + (SM - qL) * e);
    let E = k * (EL / L.r + (SM - qL) * (SM + L.p / dL));
    return Flux(FL.m + SL * (k - L.r), FL.mom + SL * (m - L.r * L.u), FL.e + SL * (E - EL));
  }
  let k = dR / (SR - SM);
  let m = k * (R.u + (SM - qR) * e);
  let E = k * (ER / R.r + (SM - qR) * (SM + R.p / dR));
  return Flux(FR.m + SR * (k - R.r), FR.mom + SR * (m - R.r * R.u), FR.e + SR * (E - ER));
}
`;

// Grid access for the step and diagnostics shaders.
const COMMON = PHYSICS + /* wgsl */ `
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;

// bad marks a cell whose density or pressure fell steeply in the last stage ("troubled").
struct Cell { w: Prim, f: u32, mt: f32, bad: bool };

fn cidx(p: vec3i) -> u32 { return u32(p.x) + P.dims.x * (u32(p.y) + P.dims.y * u32(p.z)); }
// Flag of a cell; anything outside the grid counts as a solid wall.
fn flagAt(p: vec3i) -> u32 {
  if (any(p < vec3i(0)) || any(p >= vec3i(P.dims.xyz))) { return 1u; }
  return flags[cidx(p)];
}
fn loadPrim(c: u32) -> Prim {
  let o = c * 6u;
  let r = src[o];
  let u = vec3f(src[o + 1u], src[o + 2u], src[o + 3u]) / r;
  return Prim(r, u, (P.a.w - 1.0) * (src[o + 4u] - 0.5 * r * dot(u, u)));
}
// Solid cells are never read. Port and far field cells hold the state just outside the boundary.
fn loadCell(p: vec3i) -> Cell {
  let f = flagAt(p);
  if ((f & 7u) == 1u) { return Cell(Prim(1.0, vec3f(0.0), 1.0), f, 0.0, false); }
  let c = cidx(p);
  let v = src[c * 6u + 5u];
  // the sixth value is the eddy viscosity; a troubled cell stores it as -1 - viscosity
  if (v < -0.5) { return Cell(loadPrim(c), f, -1.0 - v, true); }
  return Cell(loadPrim(c), f, v, false);
}
`;

// Fills the port cells with the state just outside each port, worked out from the fluid cell inside.
const GHOST_WGSL = PHYSICS + /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> st: array<f32>;
@group(0) @binding(2) var<storage, read> list: array<u32>; // per port cell: its index, the fluid cell inside, its flag

@compute @workgroup_size(64)
fn ghosts(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.dims.w) { return; }
  let cell = list[3u * id.x];
  let flag = list[3u * id.x + 2u];
  let j = list[3u * id.x + 1u] * 6u;
  let r = st[j];
  let u = vec3f(st[j + 1u], st[j + 2u], st[j + 3u]) / r;
  let W = Prim(r, u, (P.a.w - 1.0) * (st[j + 4u] - 0.5 * r * dot(u, u)));
  let n = vec3f(NRM[(flag >> 3u) & 7u]);
  let port = P.ports[(flag >> 6u) & 7u];
  var G = W;
  if (port.x < 0.5) { G = reservoir(W, n, port.y, port.z); } else { G = openGhost(W, n, port.y, port.z, port.w); }
  let o = cell * 6u;
  st[o] = G.r;
  st[o + 1u] = G.r * G.u.x;
  st[o + 2u] = G.r * G.u.y;
  st[o + 3u] = G.r * G.u.z;
  st[o + 4u] = G.p / (P.a.w - 1.0) + 0.5 * G.r * dot(G.u, G.u);
  st[o + 5u] = 0.0;
}
`;

const STAGE_WGSL = COMMON + /* wgsl */ `
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<storage, read_write> events: array<atomic<u32>, 4>;

const RFLOOR = 1e-3;
const PFLOOR = 1e-3;
const DROP = 0.3;   // a cell losing more than 70 percent of its density or pressure in one stage is troubled

// Monotonised central limiter, written without branches or divisions: the limited
// difference is the central one, held to at most twice either one-sided difference,
// and zero at a peak or trough.
fn lim3(a: vec3f, b: vec3f) -> vec3f {
  let s = sign(a);
  return s * max(vec3f(0.0), min(min(2.0 * abs(a), 2.0 * s * b), 0.5 * s * (a + b)));
}
fn lim2(a: vec2f, b: vec2f) -> vec2f {
  let s = sign(a);
  return s * max(vec2f(0.0), min(min(2.0 * abs(a), 2.0 * s * b), 0.5 * s * (a + b)));
}
// State at a cell face: cell value plus s (plus or minus a half) times the limited difference.
fn recon(Wm: Prim, W0: Prim, Wp: Prim, s: f32) -> Prim {
  let rp = lim2(vec2f(W0.r - Wm.r, W0.p - Wm.p), vec2f(Wp.r - W0.r, Wp.p - W0.p));
  return Prim(W0.r + s * rp.x, W0.u + s * lim3(W0.u - Wm.u, Wp.u - W0.u), W0.p + s * rp.y);
}
// The cell beyond X, as X sees it when working out its own slope: a wall mirrors X.
fn beyond(far: Cell, X: Prim, e: vec3f) -> Prim {
  if ((far.f & 7u) == 1u) { return mirror(X, e); }
  return far.w;
}

// Flux in direction e through the face between cells L and R, which lie in a row LL, L, R, RR.
// Every face is worked out twice (once from each side) with identical inputs, so what one
// cell loses the other gains exactly.
fn faceFlux(LL: Cell, L: Cell, R: Cell, RR: Cell, e: vec3f) -> Flux {
  let tL = L.f & 7u;
  let tR = R.f & 7u;
  let wallL = tL == 1u;
  let wallR = tR == 1u;
  let g = P.a.w;
  // first order on boundary faces and next to a troubled cell (the positivity fallback)
  let first = L.bad || R.bad || tL >= 2u || tR >= 2u;
  var A = L.w;
  var B = R.w;
  if (wallR) { B = mirror(A, e); }
  if (wallL) { A = mirror(B, e); }
  var WL = A;
  var WR = B;
  if (!first) {
    if (!wallL) { WL = recon(beyond(LL, A, e), A, B, 0.5); }
    if (!wallR) { WR = recon(A, B, beyond(RR, B, e), -0.5); }
    if (wallR) { WR = mirror(WL, e); }
    if (wallL) { WL = mirror(WR, e); }
  }
  var F = hllc(WL, WR, e);
  if (wallR) {
    // Solid wall: pressure only (never suction), no heat flow, and the laminar no-slip
    // stress over the half cell to the wall. The eddy viscosity is left out here on purpose:
    // without a wall model it would overstate wall friction many times over.
    return Flux(0.0, max(dot(F.mom, e), 0.0) * e + 2.0 * P.b.x * L.w.u, 0.0);
  }
  if (wallL) {
    return Flux(0.0, max(dot(F.mom, e), 0.0) * e - 2.0 * P.b.x * R.w.u, 0.0);
  }
  // viscous stress and heat conduction from the difference across the face
  let mt = 0.5 * (L.mt + R.mt);
  let mu = P.b.x + mt;
  let kc = (P.b.x * P.b.z + mt * P.b.w) * g / (g - 1.0);
  F.mom -= mu * (B.u - A.u);
  F.e -= 0.5 * mu * (dot(B.u, B.u) - dot(A.u, A.u)) + kc * (B.p / B.r - A.p / A.r);
  return F;
}

@compute @workgroup_size(4, 4, 4)
fn init(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= P.dims.xyz)) { return; }
  let o = cidx(vec3i(id)) * 6u;
  let t = flags[o / 6u] & 7u;
  var u = P.ffU.xyz;
  if (t == 1u) { u = vec3f(0.0); }
  dst[o] = P.ff.x;
  dst[o + 1u] = P.ff.x * u.x;
  dst[o + 2u] = P.ff.x * u.y;
  dst[o + 3u] = P.ff.x * u.z;
  dst[o + 4u] = P.ff.y / (P.a.w - 1.0) + 0.5 * P.ff.x * dot(u, u);
  dst[o + 5u] = 0.0;
}

// One Runge-Kutta stage: dst = wOld x dst + wNew x (src + dt x rhs(src)).
// Only fluid cells change: walls and far field cells keep their starting values,
// port cells are filled by the ghost pass.
@compute @workgroup_size(4, 4, 4)
fn stage(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= P.dims.xyz)) { return; }
  let pos = vec3i(id);
  let i = cidx(pos);
  if ((flags[i] & 7u) != 0u) { return; }
  let o = i * 6u;
  let C0 = loadCell(pos);
  // net inflow of mass, momentum and energy, and the velocity gradient for the eddy viscosity
  var km = 0.0;
  var kmom = vec3f(0.0);
  var ke = 0.0;
  var G: array<vec3f, 3>;
  for (var a = 0; a < 3; a++) {
    let d = AX[a];
    let e = vec3f(d);
    let m1 = loadCell(pos - d);
    let p1 = loadCell(pos + d);
    let m2 = loadCell(pos - 2 * d);
    let p2 = loadCell(pos + 2 * d);
    let Fm = faceFlux(m2, m1, C0, p1, e);
    let Fp = faceFlux(m1, C0, p1, p2, e);
    km += Fm.m - Fp.m;
    kmom += Fm.mom - Fp.mom;
    ke += Fm.e - Fp.e;
    var um = m1.w.u;
    var up = p1.w.u;
    if ((m1.f & 7u) == 1u) { um = C0.w.u; }   // one-sided next to a wall, so walls do not switch the eddy viscosity on
    if ((p1.f & 7u) == 1u) { up = C0.w.u; }
    G[a] = 0.5 * (up - um);
  }
  // Smagorinsky: eddy viscosity = density x (Cs x cell size)^2 x strain rate, using the
  // shearing part of the strain only so that shocks and plain compression do not add to it.
  let tr = (G[0].x + G[1].y + G[2].z) / 3.0;
  let dxx = G[0].x - tr;
  let dyy = G[1].y - tr;
  let dzz = G[2].z - tr;
  let sxy = G[1].x + G[0].y;
  let sxz = G[2].x + G[0].z;
  let syz = G[2].y + G[1].z;
  let ss = dxx * dxx + dyy * dyy + dzz * dzz + 0.5 * (sxy * sxy + sxz * sxz + syz * syz);
  var mt = C0.w.r * P.b.y * sqrt(2.0 * ss);

  let dt = P.a.x;
  let gm = P.a.w - 1.0;
  let r0 = src[o];
  let m0 = vec3f(src[o + 1u], src[o + 2u], src[o + 3u]);
  let e0 = src[o + 4u];
  var r = r0 + dt * km;
  var m = m0 + dt * kmom;
  var E = e0 + dt * ke;
  var p = gm * (E - 0.5 * dot(m, m) / r);
  if (!(r > DROP * r0 && p > DROP * C0.w.p)) {
    // Troubled cell: its faces use first-order fluxes in the next stage.
    atomicAdd(&events[0], 1u);
    mt = -1.0 - mt;
    if (!(r > RFLOOR && p > PFLOOR)) {
      // Last resort: clamp. This is counted and reported, because it does not conserve.
      atomicAdd(&events[1], 1u);
      if (!(r > RFLOOR)) { r = max(r0 * DROP, RFLOOR); m = m0 * (r / r0); }
      let kin = 0.5 * dot(m, m) / r;
      p = gm * (E - kin);
      if (!(p > PFLOOR)) { p = max(DROP * C0.w.p, PFLOOR); }
      E = p / gm + kin;
    }
  }
  if (P.a.y > 0.0) {
    r = P.a.y * dst[o] + P.a.z * r;
    m = P.a.y * vec3f(dst[o + 1u], dst[o + 2u], dst[o + 3u]) + P.a.z * m;
    E = P.a.y * dst[o + 4u] + P.a.z * E;
  }
  dst[o] = r;
  dst[o + 1u] = m.x;
  dst[o + 2u] = m.y;
  dst[o + 3u] = m.z;
  dst[o + 4u] = E;
  dst[o + 5u] = mt;
}
`;

// Sums and extremes over the whole grid without copying the field back: one pass
// reduces every row along x, a second reduces the rows of each z slab, and only
// one short record per slab is read by the CPU.
const DIAG_WGSL = COMMON + /* wgsl */ `
@group(0) @binding(3) var<storage, read_write> rows: array<f32>;
@group(0) @binding(4) var<storage, read_write> slab: array<f32>;

const K = ${K}u;
const KBASE = ${KBASE}u;

@compute @workgroup_size(8, 8, 1)
fn reduceRows(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.dims.y || id.y >= P.dims.z) { return; }
  let full = P.dims.w == 1u;
  let g = P.a.w;
  var acc: array<f32, K>;
  acc[9] = 1e30;
  acc[10] = 1e30;
  for (var x = 0u; x < P.dims.x; x++) {
    let pos = vec3i(i32(x), i32(id.x), i32(id.y));
    let c = cidx(pos);
    if ((flags[c] & 7u) != 0u) { continue; }
    let W = loadPrim(c);
    if (!(W.r > 0.0 && W.p > 0.0 && W.r < 1e20 && W.p < 1e20)) { acc[8] += 1.0; continue; }
    let cs = sqrt(g * W.p / W.r);
    var mt = src[c * 6u + 5u];
    if (mt < -0.5) { mt = -1.0 - mt; }
    // signal speed summed over the three directions, plus the viscous limit
    let visc = 6.0 * (P.b.x * max(1.0, g * P.b.z) + mt * max(1.0, g * P.b.w)) / W.r;
    acc[0] = max(acc[0], abs(W.u.x) + abs(W.u.y) + abs(W.u.z) + 3.0 * cs + visc);
    acc[1] = max(acc[1], length(W.u) / cs);
    acc[9] = min(acc[9], W.r);
    acc[10] = min(acc[10], W.p);
    if (!full) { continue; }
    acc[2] += W.r;
    acc[3] += src[c * 6u + 4u];
    acc[7] += 1.0;
    for (var f = 0u; f < 6u; f++) {
      let d = NRM[f];
      let e = vec3f(d);
      let nf = flagAt(pos + d);
      let nt = nf & 7u;
      if (nt == 1u) {
        // pressure above ambient pushing on the wall (same flux the solver uses, at first order)
        let pw = max(dot(hllc(W, mirror(W, e), e).mom, e), 0.0) - 1.0;
        acc[4] += pw * e.x;
        acc[5] += pw * e.y;
        acc[6] += pw * e.z;
      } else if (nt >= 3u) {
        // the same flux the solver uses through this port face
        let F = hllc(W, loadPrim(cidx(pos + d)), e);
        let b = KBASE + 4u * ((nf >> 6u) & 7u);
        acc[b] -= F.m;
        acc[b + 1u] += dot(F.mom, e) - 1.0;
        acc[b + 2u] += W.p;
        acc[b + 3u] += 1.0;
      }
    }
  }
  let o = (id.x + P.dims.y * id.y) * K;
  var kn = KBASE;
  if (full) { kn = K; }
  for (var c = 0u; c < kn; c++) { rows[o + c] = acc[c]; }
}

@compute @workgroup_size(64)
fn reduceSlabs(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.dims.z) { return; }
  var kn = KBASE;
  if (P.dims.w == 1u) { kn = K; }
  for (var c = 0u; c < kn; c++) {
    var v = rows[(P.dims.y * id.x) * K + c];
    for (var y = 1u; y < P.dims.y; y++) {
      let w = rows[(y + P.dims.y * id.x) * K + c];
      if (c < 2u) { v = max(v, w); } else if (c == 9u || c == 10u) { v = min(v, w); } else { v += w; }
    }
    slab[id.x * K + c] = v;
  }
}
`;

// Coarse copy for very large grids: each view cell is the average of the conserved
// variables over the fluid cells of a block, plus the fraction of the block that is fluid.
const VIEW_WGSL = /* wgsl */ `
struct VP { dims: vec4u, vdims: vec4u }; // dims.w = block size
@group(0) @binding(0) var<uniform> V: VP;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> view: array<f32>;

@compute @workgroup_size(4, 4, 4)
fn shrink(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= V.vdims.xyz)) { return; }
  let s = V.dims.w;
  var sum: array<f32, 5>;
  var count = 0.0;
  for (var dz = 0u; dz < s; dz++) { for (var dy = 0u; dy < s; dy++) { for (var dx = 0u; dx < s; dx++) {
    let p = id * s + vec3u(dx, dy, dz);
    if (any(p >= V.dims.xyz)) { continue; }
    let c = p.x + V.dims.x * (p.y + V.dims.y * p.z);
    if ((flags[c] & 7u) != 1u) {
      for (var q = 0u; q < 5u; q++) { sum[q] += src[c * 6u + q]; }
      count += 1.0;
    }
  } } }
  let o = (id.x + V.vdims.x * (id.y + V.vdims.y * id.z)) * 6u;
  if (count > 0.0) {
    for (var q = 0u; q < 5u; q++) { view[o + q] = sum[q] / count; }
  } else {
    view[o] = 1.0; view[o + 1u] = 0.0; view[o + 2u] = 0.0; view[o + 3u] = 0.0; view[o + 4u] = 2.5;
  }
  view[o + 5u] = count / f32(s * s * s);
}
`;

async function compile(device, code, name) {
  const module = device.createShaderModule({ code });
  if (module.getCompilationInfo) {
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length) throw new Error(name + " shader: " + errors.map((m) => m.lineNum + ":" + m.linePos + " " + m.message).join("; "));
  }
  return module;
}

export class CompressibleSolver {
  // Grids with more cells than this are read back as a coarser, block-averaged copy.
  static viewCells = 6e6;

  static async create() {
    if (!navigator.gpu) throw new Error("This browser does not support WebGPU.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU graphics adapter was found.");
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: lim.maxBufferSize,
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      },
    });
    const s = new CompressibleSolver();
    s.device = device;
    // the largest single buffer holds 6 floats (24 bytes) per cell
    s.maxCells = Math.floor(Math.min(lim.maxBufferSize, lim.maxStorageBufferBindingSize) / (4 * CELL));
    s.lost = false;
    device.lost.then(() => { s.lost = true; });
    const C = GPUShaderStage.COMPUTE;
    const layout = (types) => device.createBindGroupLayout({
      entries: types.map((type, binding) => ({ binding, visibility: C, buffer: { type } })),
    });
    const RO = "read-only-storage", RW = "storage";
    s.stageLayout = layout(["uniform", RO, RO, RW, RW]);
    s.diagLayout = layout(["uniform", RO, RO, RW, RW]);
    s.viewLayout = layout(["uniform", RO, RO, RW]);
    s.ghostLayout = layout(["uniform", RW, RO]);
    const [stageMod, diagMod, viewMod, ghostMod] = await Promise.all([
      compile(device, STAGE_WGSL, "Compressible step"),
      compile(device, DIAG_WGSL, "Compressible diagnostics"),
      compile(device, VIEW_WGSL, "Compressible view"),
      compile(device, GHOST_WGSL, "Compressible ports"),
    ]);
    const pipe = (lay, module, entryPoint) => device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [lay] }), compute: { module, entryPoint },
    });
    [s.initPipe, s.stagePipe, s.rowPipe, s.slabPipe, s.shrinkPipe, s.ghostPipe] = await Promise.all([
      pipe(s.stageLayout, stageMod, "init"), pipe(s.stageLayout, stageMod, "stage"),
      pipe(s.diagLayout, diagMod, "reduceRows"), pipe(s.diagLayout, diagMod, "reduceSlabs"),
      pipe(s.viewLayout, viewMod, "shrink"), pipe(s.ghostLayout, ghostMod, "ghosts"),
    ]);
    s.buffers = [];
    return s;
  }

  // Block size of the coarse copy for a grid of this many cells (1 = every cell is read back).
  static strideFor(cells) {
    return cells > CompressibleSolver.viewCells ? Math.ceil(Math.cbrt(cells / CompressibleSolver.viewCells) - 1e-9) : 1;
  }

  // GPU memory for a grid: two copies of the state, the flags and the readback copy.
  static memoryBytes(cells) {
    return cells * (2 * 4 * CELL + 4 + (CompressibleSolver.strideFor(cells) > 1 ? 1 : 4 * CELL));
  }

  // Smooth start-up factor for the feed pressure (same shape as Solver.ramp).
  static ramp(step, rampSteps) {
    if (!(rampSteps > 0) || step >= rampSteps) return 1;
    return 0.5 - 0.5 * Math.cos(Math.PI * step / rampSteps);
  }

  // grid: {nx, ny, nz, dx (m), flags} from buildDomain.
  // gas: {gamma, R, mu, ambientPressure, ambientTemperature, cfl, farfield: {pressure, temperature, velocity},
  //       smagorinsky (constant, default 0.16), prandtl (0.72), turbulentPrandtl (0.9),
  //       rampSteps (0), dtInterval (steps between time step updates, default 8),
  //       openRelax (0.25; how firmly openings hold the outside pressure, Infinity = rigidly)}
  // ports: indexed by port number, {type: "feed", totalPressure, totalTemperature} or {type: "open", pressure, temperature}
  async setup(grid, gas, ports) {
    this.dispose();
    const d = this.device;
    d.pushErrorScope("validation");
    d.pushErrorScope("out-of-memory");
    const n = grid.nx * grid.ny * grid.nz;
    this.grid = grid; this.n = n;
    this.gas = {
      gamma: 1.4, R: 287.05, mu: 1.81e-5, ambientPressure: 101325, ambientTemperature: 293.15, cfl: 0.4,
      smagorinsky: 0.16, prandtl: 0.72, turbulentPrandtl: 0.9, rampSteps: 0, dtInterval: 8, openRelax: 0.25, ...gas,
    };
    const g = this.gas;
    // reference values: everything in the shader is in units of these
    this.pRef = g.ambientPressure;
    this.TRef = g.ambientTemperature;
    this.rhoRef = this.pRef / (g.R * this.TRef);
    this.uRef = Math.sqrt(this.pRef / this.rhoRef);
    this.tRef = grid.dx / this.uRef;
    this.ports = ports || [];

    const S = GPUBufferUsage.STORAGE, CS = GPUBufferUsage.COPY_SRC, CD = GPUBufferUsage.COPY_DST;
    const MR = GPUBufferUsage.MAP_READ | CD, UN = GPUBufferUsage.UNIFORM | CD;
    const mk = (size, usage) => { const b = d.createBuffer({ size, usage }); this.buffers.push(b); return b; };
    this.state = [mk(n * 4 * CELL, S | CS | CD), mk(n * 4 * CELL, S | CS | CD)];
    this.flagBuf = mk(n * 4, S | CD);
    this.eventBuf = mk(16, S | CS | CD);
    this.params = [mk(208, UN), mk(208, UN), mk(208, UN)]; // stage one, stage two, diagnostics
    this.rowBuf = mk(grid.ny * grid.nz * K * 4, S);
    this.slabBuf = mk(grid.nz * K * 4, S | CS);
    this.diagStage = mk(grid.nz * K * 4 + 16, MR);
    this.stride = CompressibleSolver.strideFor(n);
    if (this.stride === 1) this.fieldStage = mk(n * 4 * CELL, MR);
    else {
      const sd = this.stride, vd = [Math.ceil(grid.nx / sd), Math.ceil(grid.ny / sd), Math.ceil(grid.nz / sd)];
      this.viewDims = vd; this.viewN = vd[0] * vd[1] * vd[2];
      this.viewBuf = mk(this.viewN * 4 * CELL, S | CS);
      this.fieldStage = mk(this.viewN * 4 * CELL, MR);
      this.viewParam = mk(32, UN);
      d.queue.writeBuffer(this.viewParam, 0, new Uint32Array([grid.nx, grid.ny, grid.nz, sd, vd[0], vd[1], vd[2], 0]));
      this.viewGroup = d.createBindGroup({
        layout: this.viewLayout,
        entries: [this.viewParam, this.state[0], this.flagBuf, this.viewBuf].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    }
    d.queue.writeBuffer(this.flagBuf, 0, grid.flags);
    // Port cells (inlet and open types) with the fluid cell just inside each, for the ghost pass.
    const list = [], stride = [1, grid.nx, grid.nx * grid.ny], dims = [grid.nx, grid.ny, grid.nz];
    for (let c = 0; c < n; c++) {
      const f = grid.flags[c];
      if ((f & 7) < 3) continue;
      const face = (f >> 3) & 7, a = face >> 1, pos = a === 0 ? c % grid.nx : a === 1 ? ((c / grid.nx) | 0) % grid.ny : (c / stride[2]) | 0;
      const q = pos + (face & 1 ? -1 : 1);
      if (face > 5 || q < 0 || q >= dims[a]) continue;
      const nb = c + (face & 1 ? -stride[a] : stride[a]);
      if ((grid.flags[nb] & 7) === 0) list.push(c, nb, f);
    }
    this.portCells = list.length / 3;
    this.listBuf = mk(Math.max(12, list.length * 4), S | CD);
    d.queue.writeBuffer(this.listBuf, 0, new Uint32Array(list.length ? list : 3));
    const group = (layout, list) => d.createBindGroup({ layout, entries: list.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    // stage one reads the state and writes the scratch copy; stage two reads the scratch copy
    // and blends it into the state in place. The current state is always state[0].
    this.groups = [
      group(this.stageLayout, [this.params[0], this.state[0], this.flagBuf, this.state[1], this.eventBuf]),
      group(this.stageLayout, [this.params[1], this.state[1], this.flagBuf, this.state[0], this.eventBuf]),
    ];
    this.ghostGroups = this.state.map((st) => group(this.ghostLayout, [this.params[0], st, this.listBuf]));
    this.diagGroup = group(this.diagLayout, [this.params[2], this.state[0], this.flagBuf, this.rowBuf, this.slabBuf]);
    this.work = [Math.ceil(grid.nx / 4), Math.ceil(grid.ny / 4), Math.ceil(grid.nz / 4)];
    this.reset();
    const oom = await d.popErrorScope(), bad = await d.popErrorScope();
    if (oom || bad) {
      this.dispose();
      throw new Error("The GPU could not allocate this grid. Lower the grid detail." + (bad ? " (" + bad.message + ")" : ""));
    }
  }

  // Swaps in new port settings (for example a different feed pressure) without restarting the run.
  updatePorts(ports) { this.ports = ports || []; }

  writeParams(dt) {
    const g = this.grid, gas = this.gas, ramp = CompressibleSolver.ramp(this.steps, gas.rampSteps);
    const buf = new ArrayBuffer(208);
    const u = new Uint32Array(buf, 0, 4), f = new Float32Array(buf, 16, 48);
    const far = gas.farfield;
    const rho0 = far ? (far.pressure / this.pRef) / (far.temperature / this.TRef) : 1;
    const cs2 = (gas.smagorinsky || 0) ** 2;
    f.set([
      dt, 0, 1, gas.gamma,
      gas.mu / (this.rhoRef * this.uRef * g.dx), cs2, 1 / gas.prandtl, 1 / gas.turbulentPrandtl,
      rho0, far ? far.pressure / this.pRef : 1, 0, 0,
      ...(far ? far.velocity.map((v) => v / this.uRef) : [0, 0, 0]), 0,
    ]);
    for (let i = 0; i < 8; i++) {
      const p = this.ports[i], o = 16 + 4 * i;
      if (p && p.type === "feed") {
        f[o] = 0;
        f[o + 1] = 1 + ramp * (p.totalPressure / this.pRef - 1);
        f[o + 2] = (p.totalTemperature ?? this.TRef) / this.TRef;
      } else {
        // an opening; with no settings it opens to the far field or to ambient
        f[o] = 1;
        f[o + 1] = (p?.pressure ?? far?.pressure ?? this.pRef) / this.pRef;
        f[o + 2] = (p?.temperature ?? far?.temperature ?? this.TRef) / this.TRef;
        // fraction of the way the face pressure moves to the outside pressure per stage:
        // openRelax divided by the longest side of the grid in cells (see openGhost in the shader)
        f[o + 3] = Math.min(1, gas.openRelax / Math.max(g.nx, g.ny, g.nz));
      }
    }
    const q = this.device.queue;
    u.set([g.nx, g.ny, g.nz, this.portCells]);
    q.writeBuffer(this.params[0], 0, buf);
    f[1] = 0.5; f[2] = 0.5;
    q.writeBuffer(this.params[1], 0, buf);
    return buf;
  }

  // Back to the starting state: ambient air at rest, or the far field flow if one is given.
  reset() {
    this.steps = 0; this.time = 0; this.dt = 0; this.sinceDt = 0; this.settle = 16; this.wave = 0;
    this.diverged = false;
    this.writeParams(0);
    const enc = this.device.createCommandEncoder();
    for (const g of this.groups) {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.initPipe);
      pass.setBindGroup(0, g);
      pass.dispatchWorkgroups(...this.work);
      pass.end();
    }
    enc.clearBuffer(this.eventBuf);
    this.device.queue.submit([enc.finish()]);
  }

  // Overwrites the flow state. fields has 8 floats per cell in the layout read() returns
  // (density, velocity x y z, pressure in SI units; the rest is ignored). For tests and restarts.
  setState(fields) {
    const n = this.n, a = new Float32Array(n * CELL), gi = 1 / (this.gas.gamma - 1);
    const far = this.gas.farfield, flags = this.grid.flags;
    for (let c = 0; c < n; c++) {
      let r = fields[8 * c] / this.rhoRef, p = fields[8 * c + 4] / this.pRef;
      let u = fields[8 * c + 1] / this.uRef, v = fields[8 * c + 2] / this.uRef, w = fields[8 * c + 3] / this.uRef;
      if ((flags[c] & 7) === 2 && far) {
        // far field cells always hold the far field flow
        p = far.pressure / this.pRef; r = p / (far.temperature / this.TRef);
        [u, v, w] = far.velocity.map((q) => q / this.uRef);
      }
      a.set([r, r * u, r * v, r * w, p * gi + 0.5 * r * (u * u + v * v + w * w), 0], CELL * c);
    }
    this.device.queue.writeBuffer(this.state[0], 0, a);
    this.device.queue.writeBuffer(this.state[1], 0, a);
    this.dt = 0; this.sinceDt = 0; this.settle = 16;
  }

  // Refreshes the port cells of one state copy from the fluid cells inside them.
  ghostPass(pass, which) {
    if (!this.portCells) return;
    pass.setPipeline(this.ghostPipe);
    pass.setBindGroup(0, this.ghostGroups[which]);
    pass.dispatchWorkgroups(Math.ceil(this.portCells / 64));
  }

  // Runs the reduction passes and reads one short record per z slab.
  // full = false gives only what the time step needs.
  async measure(full) {
    const d = this.device, g = this.grid;
    const buf = this.writeParams(this.dt);
    new Uint32Array(buf, 0, 4)[3] = full ? 1 : 0;
    d.queue.writeBuffer(this.params[2], 0, buf);
    const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
    this.ghostPass(pass, 0);
    pass.setBindGroup(0, this.diagGroup);
    pass.setPipeline(this.rowPipe);
    pass.dispatchWorkgroups(Math.ceil(g.ny / 8), Math.ceil(g.nz / 8));
    pass.setPipeline(this.slabPipe);
    pass.dispatchWorkgroups(Math.ceil(g.nz / 64));
    pass.end();
    const bytes = g.nz * K * 4;
    enc.copyBufferToBuffer(this.slabBuf, 0, this.diagStage, 0, bytes);
    enc.copyBufferToBuffer(this.eventBuf, 0, this.diagStage, bytes, 16);
    d.queue.submit([enc.finish()]);
    await this.diagStage.mapAsync(GPUMapMode.READ);
    const range = this.diagStage.getMappedRange();
    const s = new Float32Array(range, 0, g.nz * K), ev = new Uint32Array(range, bytes, 4);
    const kn = full ? K : KBASE, t = new Float64Array(K);
    t[9] = Infinity; t[10] = Infinity;
    for (let z = 0; z < g.nz; z++) {
      for (let c = 0; c < kn; c++) {
        const v = s[z * K + c];
        if (c < 2) { if (!(v <= t[c])) t[c] = v; } // also lets a not-a-number through, so it is noticed
        else if (c === 9 || c === 10) { if (!(v >= t[c])) t[c] = v; }
        else t[c] += v;
      }
    }
    const events = [ev[0], ev[1]];
    this.diagStage.unmap();
    if (!(t[0] > 0) || !isFinite(t[0]) || t[8] > 0) this.diverged = true;
    return { t, events };
  }

  // Advances k time steps. The time step is the same for every cell and comes from the
  // fastest signal on the grid, which is measured on the GPU every few steps.
  async step(k) {
    const d = this.device, cfl = this.gas.cfl;
    let done = 0;
    while (done < k && !this.diverged) {
      // measure every step just after a start (the flow changes fastest then), later every few steps
      const every = this.settle > 0 ? 1 : Math.max(1, this.gas.dtInterval);
      if (!this.dt || this.sinceDt >= every) {
        const { t } = await this.measure(false);
        if (this.diverged) break;
        if (this.wave && t[0] > 1.25 * this.wave) this.settle = Math.max(this.settle, 8);
        this.wave = t[0]; this.maxMach = t[1];
        // signal speed is summed over x, y and z, so the stable limit of this sum is about 1
        this.dt = 2 * cfl / t[0];
        this.sinceDt = 0;
      }
      const m = Math.min(k - done, every - this.sinceDt);
      this.writeParams(this.dt);
      const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
      for (let s = 0; s < m; s++) {
        for (const h of [0, 1]) {
          this.ghostPass(pass, h);
          pass.setPipeline(this.stagePipe);
          pass.setBindGroup(0, this.groups[h]);
          pass.dispatchWorkgroups(...this.work);
        }
      }
      pass.end();
      d.queue.submit([enc.finish()]);
      done += m; this.steps += m; this.sinceDt += m;
      this.time += m * Math.fround(this.dt) * this.tRef;
      if (this.settle > 0) this.settle -= m;
    }
    await d.queue.onSubmittedWorkDone();
    return done;
  }

  // Totals and port flows in SI units, from the GPU sums (no field copy).
  async totals() {
    const { t, events } = await this.measure(true);
    const dx = this.grid.dx, A = dx * dx, V = dx * dx * dx;
    const ports = [];
    for (let i = 0; i < 8; i++) {
      const o = KBASE + 4 * i, faces = t[o + 3];
      if (!faces && !this.ports[i]) continue;
      ports[i] = {
        massFlow: t[o] * this.rhoRef * this.uRef * A,          // kg/s, positive into the domain
        momentumFlow: t[o + 1] * this.pRef * A,                 // N: mass flow x speed plus pressure above ambient x area
        meanPressure: faces ? (t[o + 2] / faces) * this.pRef : NaN, // Pa absolute, in the cells just inside
        area: faces * A,
      };
    }
    return {
      ports,
      force: [t[4], t[5], t[6]].map((v) => v * this.pRef * A), // N, from pressure above ambient
      maxMach: t[1],
      mass: t[2] * this.rhoRef * V,                             // kg in the fluid cells
      energy: t[3] * this.pRef * V,                             // J, internal plus kinetic
      fluidCells: t[7], bad: t[8],
      minDensity: t[9] * this.rhoRef, minPressure: t[10] * this.pRef,
      firstOrderFallbacks: events[0], clamps: events[1],
      time: this.time, steps: this.steps, dt: this.dt * this.tRef,
    };
  }

  // Copies the flow back in SI units: fields has 8 floats per cell (density, velocity x y z,
  // pressure, temperature, Mach number, 0). On very large grids fields is a block-averaged
  // copy with dims = viewDims and stride > 1 (solid-only blocks hold ambient air at rest).
  async read() {
    const d = this.device, out = await this.totals();
    const enc = d.createCommandEncoder();
    let count = this.n;
    if (this.stride === 1) enc.copyBufferToBuffer(this.state[0], 0, this.fieldStage, 0, this.n * 4 * CELL);
    else {
      count = this.viewN;
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, this.viewGroup);
      pass.setPipeline(this.shrinkPipe);
      pass.dispatchWorkgroups(Math.ceil(this.viewDims[0] / 4), Math.ceil(this.viewDims[1] / 4), Math.ceil(this.viewDims[2] / 4));
      pass.end();
      enc.copyBufferToBuffer(this.viewBuf, 0, this.fieldStage, 0, count * 4 * CELL);
    }
    d.queue.submit([enc.finish()]);
    await this.fieldStage.mapAsync(GPUMapMode.READ);
    const s = new Float32Array(this.fieldStage.getMappedRange());
    const fields = new Float32Array(count * 8), g = this.gas.gamma;
    for (let c = 0; c < count; c++) {
      const o = CELL * c, r = s[o], u = s[o + 1] / r, v = s[o + 2] / r, w = s[o + 3] / r;
      const q2 = u * u + v * v + w * w, p = (g - 1) * (s[o + 4] - 0.5 * r * q2), f = 8 * c;
      fields[f] = r * this.rhoRef;
      fields[f + 1] = u * this.uRef; fields[f + 2] = v * this.uRef; fields[f + 3] = w * this.uRef;
      fields[f + 4] = p * this.pRef;
      fields[f + 5] = (p / r) * this.TRef;
      fields[f + 6] = Math.sqrt(q2 / (g * p / r));
    }
    this.fieldStage.unmap();
    out.fields = fields;
    out.stride = this.stride;
    out.dims = this.stride === 1 ? [this.grid.nx, this.grid.ny, this.grid.nz] : this.viewDims;
    return out;
  }

  dispose() {
    for (const b of this.buffers || []) b.destroy();
    this.buffers = [];
  }
}
