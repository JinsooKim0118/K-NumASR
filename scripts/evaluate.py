import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knumasr.itn import korean_itn
from knumasr.metrics import aggregate, score_utterance

SYN_CATS = ["phone", "account", "amount", "datetime", "native"]


def load_preds(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def eval_system(rows):
    scored = [dict(score_utterance(r["ref"], r["hyp"]), category=r["category"]) for r in rows]
    syn = [s for s in scored if s["category"] in SYN_CATS]
    fl = [s for s in scored if s["category"] == "fleurs"]
    return {
        "overall": aggregate(scored),
        "synthetic": aggregate(syn),
        "fleurs": aggregate(fl),
        "by_category": {c: aggregate([s for s in scored if s["category"] == c]) for c in SYN_CATS + ["fleurs"] if any(s["category"] == c for s in scored)},
    }, scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default="predictions")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--no_itn_variants", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "details"), exist_ok=True)
    systems = {}
    for p in sorted(glob.glob(os.path.join(args.pred_dir, "*.jsonl"))):
        name = os.path.basename(p)[:-6]
        rows = load_preds(p)
        systems[name] = rows
        if not args.no_itn_variants:
            systems[name + "+K-ITN"] = [dict(r, hyp=korean_itn(r["hyp"])) for r in rows]
    board = []
    for name, rows in systems.items():
        res, scored = eval_system(rows)
        info_path = os.path.join(args.pred_dir, name.replace("+K-ITN", "") + ".info.json")
        info = json.load(open(info_path, encoding="utf-8")) if os.path.exists(info_path) else {}
        board.append({"system": name, "model": info.get("model", ""), "prompt": info.get("prompt", False),
                      "itn": name.endswith("+K-ITN"), **res})
        with open(os.path.join(args.out_dir, "details", f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["id", "category", "ref", "hyp", "nem", "nem_norm", "digit_err", "char_err"])
            for r, s in zip(rows, scored):
                wr.writerow([r["id"], r["category"], r["ref"], r["hyp"], s["nem"], s["nem_norm"], s["digit_err"], s["char_err"]])
    board.sort(key=lambda b: (-b["overall"]["NEM"], b["overall"]["DER"]))
    with open(os.path.join(args.out_dir, "leaderboard.json"), "w", encoding="utf-8") as f:
        json.dump(board, f, ensure_ascii=False, indent=1)
    head = ["rank", "system", "NEM", "NEM_syn", "NEM_fleurs", "DER", "CER", "NEM_norm"] + [f"NEM_{c}" for c in SYN_CATS]
    lines = []
    for i, b in enumerate(board, 1):
        bc = b["by_category"]
        lines.append([i, b["system"], b["overall"]["NEM"], b["synthetic"].get("NEM", float("nan")), b["fleurs"].get("NEM", float("nan")),
                      b["overall"]["DER"], b["overall"]["CER"], b["overall"]["NEM_norm"]] + [bc.get(c, {}).get("NEM", float("nan")) for c in SYN_CATS])
    with open(os.path.join(args.out_dir, "leaderboard.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(head)
        for l in lines:
            wr.writerow([l[0], l[1]] + [f"{v:.2f}" for v in l[2:]])
    with open(os.path.join(args.out_dir, "leaderboard.md"), "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
        for l in lines:
            f.write("| " + " | ".join([str(l[0]), l[1]] + [f"{v:.1f}" for v in l[2:]]) + " |\n")
    print(open(os.path.join(args.out_dir, "leaderboard.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
