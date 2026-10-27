"""FastAPI backend + static frontend.   Run:  python -m uvicorn terrascope.server.app:app --port 8000"""
from __future__ import annotations

import io, json, os
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core.inference import Engine, ROOT
from ..core.simulator import generate_scene

WEB = os.path.join(os.path.dirname(__file__), "..", "web")
app = FastAPI(title="TerraScope", version="1.0")
engine = Engine()


class AnalyzeReq(BaseModel):
    source: str = "sim"                 # "sim" | "upload"
    seed: int = 7
    size: int = Field(128, ge=32, le=256)
    flood: float = Field(0.0, ge=0, le=1)
    cloud: float = Field(0.0, ge=0, le=1)
    drought: float = Field(0.0, ge=0, le=1)
    sensors: list[bool] = [True, True, True]
    scene_id: str | None = None


@app.get("/api/health")
def health():
    return dict(ready=engine.ready, params=getattr(engine, "n_params", 0))


@app.post("/api/analyze")
def analyze(r: AnalyzeReq):
    if not engine.ready:
        raise HTTPException(503, "Model checkpoint not found. Run ./run_training.sh first.")
    if len(r.sensors) != 3 or not any(r.sensors):
        raise HTTPException(400, "Enable at least one sensor (optical, SAR, DEM).")
    if r.source == "upload":
        sc = engine.uploads.get(r.scene_id or "")
        if sc is None:
            raise HTTPException(404, "Uploaded scene expired — upload again.")
        meta = dict(source="upload", scene_id=r.scene_id)
    else:
        sc = engine.sim_scene(r.seed, r.size, r.flood, r.cloud, r.drought)
        meta = dict(source="sim", seed=r.seed, size=r.size, flood=r.flood, cloud=r.cloud, drought=r.drought)
    return engine.analyze(sc, r.sensors, meta)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        z = np.load(io.BytesIO(raw))
        sid, sc = engine.register_upload({k: z[k] for k in z.files})
    except Exception as e:  # noqa
        raise HTTPException(400, f"Could not read scene: {e}")
    return dict(scene_id=sid, size=list(sc["optical"].shape[1:]))


@app.get("/api/layer/{aid}/{name}.png")
def layer(aid: str, name: str):
    d = engine.cache.get(aid)
    if not d or name not in d["layers"]:
        raise HTTPException(404)
    return Response(d["layers"][name], media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/similarity/{aid}.png")
def similarity(aid: str, x: int = 0, y: int = 0):
    if aid not in engine.cache:
        raise HTTPException(404)
    return Response(engine.similarity_png(aid, x, y), media_type="image/png")


@app.get("/api/export/{aid}.npz")
def export(aid: str):
    d = engine.cache.get(aid)
    if not d:
        raise HTTPException(404)
    buf = io.BytesIO()
    arrs = {k: v for k, v in d["sc"].items()}
    arrs.update(pred_lc=d["pred"]["lc"], pred_flood_prob=d["pred"]["flood"], pred_veg=d["pred"]["veg"])
    np.savez_compressed(buf, **arrs)
    return Response(buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": f"attachment; filename=terrascope_{aid}.npz"})


@app.get("/api/sample.npz")
def sample(seed: int = 21, flood: float = 0.8, cloud: float = 0.4, drought: float = 0.3, size: int = 128):
    """A ready-made scene in the upload format (try the 'Bring your own data' flow)."""
    sc = generate_scene(seed, size, flood=flood, cloud=cloud, drought=drought)
    buf = io.BytesIO()
    np.savez_compressed(buf, optical=sc["optical"], sar=sc["sar"], dem=sc["dem"])
    return Response(buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=terrascope_sample.npz"})


@app.get("/api/benchmarks")
def benchmarks():
    p = os.path.join(ROOT, "artifacts", "metrics.json")
    if not os.path.exists(p):
        raise HTTPException(404, "metrics.json missing — run terrascope.core.evaluate")
    return JSONResponse(json.load(open(p)))


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


app.mount("/static", StaticFiles(directory=WEB), name="static")
