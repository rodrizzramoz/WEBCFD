// Compressed-air nozzle sums for the "Air fed in" ports.
//
// The flow solver treats air as incompressible, so a jet fed at several bar
// cannot be simulated as it really is (it leaves the nozzle at the speed of
// sound and keeps expanding). What drives the air it drags along is the
// jet's thrust, so the jet is replaced by an incompressible one with the same
// thrust through the same nozzle area. The true air consumption is worked out
// here from standard nozzle theory and reported separately.
//
// Assumes dry air from a supply at 20 C, an ideal nozzle (discharge
// coefficient 1) and standard atmosphere outside.
const GAMMA = 1.4, R = 287.05, T0 = 293.15, P_ATM = 101325;
export const RHO_AIR = P_ATM / (R * T0);
export const BAR = 1e5, PSI = 6894.757;

// gauge: supply pressure above atmosphere in Pa. area: smallest nozzle area in m2.
export function nozzle(gauge, area) {
  const p0 = P_ATM + Math.max(0, gauge), ratio = P_ATM / p0;
  // speed after expanding fully to atmospheric pressure
  const vFull = Math.sqrt(2 * GAMMA / (GAMMA - 1) * R * T0 * (1 - Math.pow(ratio, (GAMMA - 1) / GAMMA)));
  const choked = ratio < Math.pow(2 / (GAMMA + 1), GAMMA / (GAMMA - 1)); // below 0.528: sonic at the nozzle
  let mdot;
  if (choked) mdot = area * p0 * Math.sqrt(GAMMA / (R * T0)) * Math.pow(2 / (GAMMA + 1), (GAMMA + 1) / (2 * (GAMMA - 1)));
  else mdot = (p0 / (R * T0)) * Math.pow(ratio, 1 / GAMMA) * vFull * area;
  const thrust = mdot * vFull;
  return {
    choked, massFlow: mdot, thrust, jetSpeed: vFull,
    freeAir: mdot / RHO_AIR,                                   // air consumption as free air, m3/s
    equivalentSpeed: Math.sqrt(thrust / (RHO_AIR * area)),     // incompressible speed through the nozzle area with the same thrust
  };
}

export const FLOW_UNITS = {
  m3h: { label: "m3/h", k: 3600 },
  ls: { label: "L/s", k: 1000 },
  lmin: { label: "L/min", k: 60000 },
  cfm: { label: "cfm", k: 2118.88 },
};
