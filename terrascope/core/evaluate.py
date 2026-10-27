"""Full benchmark -> artifacts/metrics.json (consumed by the web UI)."""
import json, os, torch
from .data import load
from .model import TerraFM
from .metrics import evaluate, SENSOR_SETS
from .train import CKPT, LOGS, CFG, ROOT


def read_log(name):
    p = os.path.join(LOGS, f"{name}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def load_model(name):
    m = TerraFM(**CFG)
    m.load_state_dict(torch.load(os.path.join(CKPT, f"{name}.pt"))["model"])
    return m.eval()


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 2)
    te64, te128 = load("test64"), load("test128")
    res = {"ablation": {}, "scratch": {}, "size128": {}}
    ft = load_model("finetuned")
    sc = load_model("scratch")
    for k, av in SENSOR_SETS.items():
        res["ablation"][k] = evaluate(ft, te64, av, recon=(k == "All sensors"))
        res["scratch"][k] = evaluate(sc, te64, av)
    res["size128"]["finetuned"] = evaluate(ft, te128, [True, True, True], bs=20, recon=True)
    # cloud stratified: scenes with heavy cloud cover
    heavy = te64["cloud"].mean((1, 2)) > 0.3
    sub = {k: v[heavy] for k, v in te64.items() if hasattr(v, "shape") and v.shape[0] == heavy.shape[0]}
    res["heavy_cloud"] = {"n": int(heavy.sum()),
                          "finetuned_optical": evaluate(ft, sub, [True, False, False]),
                          "finetuned_all": evaluate(ft, sub, [True, True, True]),
                          "scratch_optical": evaluate(sc, sub, [True, False, False]),
                          "scratch_all": evaluate(sc, sub, [True, True, True])}
    res["curves"] = {"pretrain": read_log("pretrain"), "finetuned": read_log("finetuned"), "scratch": read_log("scratch")}
    res["config"] = dict(CFG, params=sum(p.numel() for p in ft.parameters()), labels=int(os.environ.get("LABELS", 300)), train_scenes=3000)
    json.dump(res, open(os.path.join(ROOT, "artifacts", "metrics.json"), "w"), indent=1)
    a = res["ablation"]["All sensors"]; s = res["scratch"]["All sensors"]
    print("TerraFM  mIoU %.3f flood IoU %.3f veg MAE %.3f" % (a["miou"], a["flood_iou"], a["veg_mae"]))
    print("Scratch  mIoU %.3f flood IoU %.3f veg MAE %.3f" % (s["miou"], s["flood_iou"], s["veg_mae"]))
