#!/usr/bin/env python3
"""Flowbench OpenFOAM runner.

A small local service that lets the Flowbench web page solve its current case
with OpenFOAM on this computer. The page sends the case files, this program
runs the mesher and the solver, and sends back flow rates, pressures and the
pressure force on the model.

    python3 flowbench_openfoam_server.py            (listens on 127.0.0.1:8765)

Needs Python 3.8 or newer and OpenFOAM (openfoam.com series, v1912 or newer)
with blockMesh, snappyHexMesh and simpleFoam on the PATH. Standard library only.

Safety
  * It listens on this computer only (127.0.0.1).
  * Every request must carry the access code printed at start-up, so a random
    web page cannot use it.
  * It never runs scripts from the page. Only a fixed list of OpenFOAM
    dictionary files is accepted, files that could run code are rejected, and
    the commands it runs are fixed in this file.
"""
import io, json, os, re, secrets, shutil, subprocess, sys, tempfile, threading, time, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("FLOWBENCH_PORT", "8765"))
TOKEN = os.environ.get("FLOWBENCH_TOKEN") or secrets.token_urlsafe(9)
ALLOWED = {
    "0/U", "0/p", "0/k", "0/omega", "0/nut", "0/T", "0/alphat",
    "constant/transportProperties", "constant/thermophysicalProperties", "constant/turbulenceProperties", "constant/triSurface/model.stl",
    "system/blockMeshDict", "system/snappyHexMeshDict", "system/topoSetDict", "system/createPatchDict",
    "system/controlDict", "system/fvSchemes", "system/fvSolution",
}
MAX_BYTES = 400 * 1024 * 1024
JOBS = {}


def foam_env():
    """Environment with OpenFOAM set up, or None if it cannot be found."""
    if shutil.which("simpleFoam") and os.environ.get("WM_PROJECT_DIR"):
        return dict(os.environ)
    # not set up in this shell: look for the usual set-up scripts and load one
    import glob
    found = glob.glob("/usr/lib/openfoam/openfoam*/etc/bashrc") + glob.glob("/opt/openfoam*/etc/bashrc") + glob.glob(os.path.expanduser("~/OpenFOAM/OpenFOAM-*/etc/bashrc"))
    for rc in ["/usr/share/openfoam/etc/bashrc"] + sorted(found, reverse=True) + [os.environ.get("FLOWBENCH_FOAM_BASHRC", "")]:
        if rc and os.path.exists(rc):
            out = subprocess.run(["bash", "-c", ". '%s' >/dev/null 2>&1; env -0" % rc], capture_output=True)
            env = dict(kv.split("=", 1) for kv in out.stdout.decode(errors="replace").split("\0") if "=" in kv)
            if shutil.which("simpleFoam", path=env.get("PATH")):
                return env
    return dict(os.environ) if shutil.which("simpleFoam") else None


ENV = foam_env()


def read_block(text):
    m = re.search(r"\n(\d+)\s*\n?\(", text)
    return int(m.group(1)), text[m.end():]


def results(case):
    """Patch flows, area-averaged pressure and pressure force on the model, from the last time folder."""
    import math
    n, body = read_block(open(os.path.join(case, "constant/polyMesh/points")).read())
    pts = [tuple(map(float, m)) for m in re.findall(r"\(([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)\)", body)[:n]]
    n, body = read_block(open(os.path.join(case, "constant/polyMesh/faces")).read())
    faces = [list(map(int, m.split())) for m in re.findall(r"\d+\(([\d ]+)\)", body)[:n]]
    n, body = read_block(open(os.path.join(case, "constant/polyMesh/owner")).read())
    owner = list(map(int, body.split(")")[0].split()[:n]))
    bnd = open(os.path.join(case, "constant/polyMesh/boundary")).read()
    patches = {m.group(1): (int(m.group(3)), int(m.group(2))) for m in re.finditer(r"(\w+)\s*\{[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);", bnd, re.S)}
    times = sorted([d for d in os.listdir(case) if re.fullmatch(r"\d+(\.\d+)?", d) and float(d) > 0], key=float)
    if not times:
        raise RuntimeError("the solver wrote no result")
    t = times[-1]
    ptxt = open(os.path.join(case, t, "p")).read()
    m = re.search(r"internalField\s+nonuniform List<scalar>\s*(\d+)\s*\(([^;]*?)\)\s*;", ptxt, re.S)
    pint = list(map(float, m.group(2).split()))

    def boundary(name, patch):
        s = open(os.path.join(case, t, name)).read()
        i = s.index("boundaryField")
        mm = re.search(r"\n\s*" + re.escape(patch) + r"\s*\{", s[i:])
        if not mm:
            return None
        j = mm.end() + i
        seg = s[j:s.index("}", j)]
        m2 = re.search(r"value\s+nonuniform List<\w+>\s*(\d+)\s*\(([^;]*)\)\s*;", seg, re.S)
        if m2:
            return list(map(float, m2.group(2).split()))
        m2 = re.search(r"value\s+uniform\s+([-\d.e+]+)", seg)
        return float(m2.group(1)) if m2 else None

    def area_vec(face):
        p = [pts[v] for v in face]
        c = [sum(q[a] for q in p) / len(p) for a in range(3)]
        a = [0.0, 0.0, 0.0]
        for k in range(len(p)):
            u = [p[k][i] - c[i] for i in range(3)]
            w = [p[(k + 1) % len(p)][i] - c[i] for i in range(3)]
            a[0] += (u[1] * w[2] - u[2] * w[1]) / 2
            a[1] += (u[2] * w[0] - u[0] * w[2]) / 2
            a[2] += (u[0] * w[1] - u[1] * w[0]) / 2
        return a

    out = {"time": float(t), "patches": {}}
    for name, (start, nf) in patches.items():
        if nf == 0:
            continue
        sf = [area_vec(faces[start + k]) for k in range(nf)]
        mag = [math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) for a in sf]
        total = sum(mag)
        phi = boundary("phi", name)
        p = boundary("p", name)
        if p is None:
            p = [pint[owner[start + k]] for k in range(nf)]
        flow = sum(phi) if isinstance(phi, list) else (phi or 0.0) * nf
        pavg = sum(pv * a for pv, a in zip(p, mag)) / total if isinstance(p, list) else p
        entry = {"faces": nf, "area": total, "flow": flow, "pOverRho": pavg}
        if name == "model":
            pw = [pint[owner[start + k]] for k in range(nf)]
            entry["pressureForceOverRho"] = [sum(pw[k] * sf[k][a] for k in range(nf)) for a in range(3)]
        out["patches"][name] = entry
    return out


def run_job(job, files):
    case = job["dir"]
    try:
        for path, data in files.items():
            full = os.path.join(case, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(data)
        steps = [["blockMesh"], ["snappyHexMesh", "-overwrite"]]
        if "system/topoSetDict" in files:
            steps += [["topoSet"], ["createPatch", "-overwrite"]]
        control = files["system/controlDict"].decode(errors="replace")
        app = re.search(r"application\s+(\w+)", control)
        solver = app.group(1) if app and app.group(1) in ("simpleFoam", "rhoPimpleFoam") else "simpleFoam"
        steps += [["checkMesh"], [solver]]
        job["compressible"] = solver != "simpleFoam"
        m = re.search(r"endTime\s+(\d+)", control)
        job["of"] = int(m.group(1)) if m else 0
        for cmd in steps:
            if job.get("cancel"):
                raise RuntimeError("cancelled")
            job["stage"] = cmd[0]
            log = os.path.join(case, "log." + cmd[0])
            with open(log, "w") as lf:
                proc = subprocess.Popen(cmd, cwd=case, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
                job["proc"] = proc
                while proc.poll() is None:
                    time.sleep(0.5)
                    if cmd[0] == solver:
                        try:
                            with open(log, "rb") as r:
                                r.seek(max(0, os.path.getsize(log) - 4000))
                                found = re.findall(r"\nTime = (\d+)", r.read().decode(errors="replace"))
                            if found:
                                job["step"] = int(found[-1])
                        except OSError:
                            pass
                    if job.get("cancel"):
                        proc.kill()
            text = open(log, errors="replace").read()
            if cmd[0] == "checkMesh":
                c = re.search(r"cells:\s+(\d+)", text)
                job["cells"] = int(c.group(1)) if c else 0
                continue
            if proc.returncode != 0:
                tail = text[-1500:]
                fatal = text.find("FOAM FATAL")
                raise RuntimeError(cmd[0] + " failed: " + (text[fatal:fatal + 600] if fatal >= 0 else tail[-600:]).strip())
        job["results"] = results(case)
        job["results"]["compressible"] = job["compressible"]
        job["state"] = "done"
    except Exception as e:  # report every failure to the page
        job["state"] = "cancelled" if job.get("cancel") else "failed"
        job["error"] = str(e)
    finally:
        job["proc"] = None
        if not os.environ.get("FLOWBENCH_KEEP"):
            shutil.rmtree(case, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.cors()
        self.end_headers()
        self.wfile.write(body)

    def cors(self):
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin") or "*")
        self.send_header("Access-Control-Allow-Headers", "content-type, x-flowbench-token")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Vary", "Origin")

    def authorised(self):
        return secrets.compare_digest(self.headers.get("X-Flowbench-Token") or "", TOKEN)

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            # no access code needed: says only whether the service and OpenFOAM are there
            return self.reply(200, {"ok": True, "openfoam": ENV is not None, "busy": any(j["state"] == "running" for j in JOBS.values())})
        if not self.authorised():
            return self.reply(401, {"error": "wrong or missing access code"})
        m = re.fullmatch(r"/job/(\w+)", self.path)
        if m and m.group(1) in JOBS:
            j = JOBS[m.group(1)]
            return self.reply(200, {k: j.get(k) for k in ("state", "stage", "step", "of", "cells", "error", "results")})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        if not self.authorised():
            return self.reply(401, {"error": "wrong or missing access code"})
        m = re.fullmatch(r"/job/(\w+)/cancel", self.path)
        if m and m.group(1) in JOBS:
            JOBS[m.group(1)]["cancel"] = True
            return self.reply(200, {"ok": True})
        if self.path != "/run":
            return self.reply(404, {"error": "not found"})
        if ENV is None:
            return self.reply(503, {"error": "OpenFOAM was not found on this computer"})
        if any(j["state"] == "running" for j in JOBS.values()):
            return self.reply(409, {"error": "a case is already running"})
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0 or size > MAX_BYTES:
            return self.reply(413, {"error": "case is empty or too large"})
        try:
            z = zipfile.ZipFile(io.BytesIO(self.rfile.read(size)))
            files = {}
            for name in z.namelist():
                if name.endswith("/"):
                    continue
                rel = name.split("/", 1)[1] if name.startswith("flowbench-openfoam/") else name
                if rel in ("Allrun", "README.txt"):
                    continue  # never executed here
                if rel not in ALLOWED:
                    return self.reply(400, {"error": "unexpected file in case: " + rel})
                data = z.read(name)
                if not rel.endswith(".stl") and re.search(rb"#\s*(codeStream|calc|eval|include|sinclude|includeEtc|includeFunc|remove|system)|\bcoded\w*|\bsystemCall\b|\blibs\b", data):
                    return self.reply(400, {"error": "file not accepted because it could run code: " + rel})
                files[rel] = data
            for need in ("system/controlDict", "system/blockMeshDict", "system/snappyHexMeshDict", "constant/triSurface/model.stl", "0/U", "0/p"):
                if need not in files:
                    return self.reply(400, {"error": "case is missing " + need})
        except zipfile.BadZipFile:
            return self.reply(400, {"error": "not a zip file"})
        jid = secrets.token_hex(6)
        job = {"state": "running", "stage": "starting", "step": 0, "of": 0, "dir": tempfile.mkdtemp(prefix="flowbench-")}
        JOBS[jid] = job
        threading.Thread(target=run_job, args=(job, files), daemon=True).start()
        self.reply(200, {"job": jid})


if __name__ == "__main__":
    print("Flowbench OpenFOAM runner")
    print("  OpenFOAM found: " + ("yes" if ENV else "NO, install it and make sure simpleFoam is on the PATH"))
    print("  Listening on http://127.0.0.1:%d" % PORT)
    print("  Access code (paste this into the Flowbench page): " + TOKEN)
    sys.stdout.flush()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
