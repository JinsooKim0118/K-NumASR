import argparse
import csv
import io
import os
import re
import sys
import tarfile

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = "google/fleurs"
LANG = "ko_kr"


def read_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 4:
                continue
            rows.append({"fleurs_id": cols[0], "file_name": cols[1], "raw": cols[2], "norm": cols[3]})
    return rows


def from_tar(split, out_dir):
    tsv = hf_hub_download(REPO, f"data/{LANG}/{split}.tsv", repo_type="dataset")
    tar = hf_hub_download(REPO, f"data/{LANG}/audio/{split}.tar.gz", repo_type="dataset")
    rows = [r for r in read_tsv(tsv) if re.search(r"\d", r["raw"])]
    wanted = {r["file_name"]: r for r in rows}
    found = {}
    with tarfile.open(tar) as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if name in wanted and name not in found:
                data = tf.extractfile(m).read()
                audio, sr = sf.read(io.BytesIO(data), dtype="float32")
                found[name] = (audio, sr)
    out = []
    for name, r in wanted.items():
        if name not in found:
            continue
        audio, sr = found[name]
        out.append((r, audio, sr))
    return out


def from_parquet(split):
    from datasets import Audio, load_dataset
    ds = load_dataset(REPO, LANG, split=split if split != "dev" else "validation", revision="refs/convert/parquet")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    out = []
    for ex in ds:
        if not re.search(r"\d", ex["raw_transcription"]):
            continue
        r = {"fleurs_id": str(ex["id"]), "file_name": os.path.basename(ex["path"] or f"{ex['id']}.wav"),
             "raw": ex["raw_transcription"], "norm": ex["transcription"]}
        out.append((r, np.asarray(ex["audio"]["array"], dtype="float32"), ex["audio"]["sampling_rate"]))
    return out


def to_16k(audio, sr):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr == 16000:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=sr, target_sr=16000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fleurs_num")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "audio"), exist_ok=True)
    meta = []
    for split in args.splits:
        try:
            items = from_tar(split, args.out)
        except Exception as e:
            print(f"tar download failed for {split}: {e}; falling back to parquet")
            items = from_parquet(split)
        for r, audio, sr in items:
            uid = f"fleurs_{split}_{r['fleurs_id']}_{len(meta):04d}"
            rel = os.path.join("audio", uid + ".flac")
            sf.write(os.path.join(args.out, rel), to_16k(audio, sr), 16000, format="FLAC")
            meta.append({"id": uid, "category": "fleurs", "split": split, "text": r["raw"], "fleurs_norm": r["norm"], "audio": rel})
        print(f"{split}: {sum(1 for m in meta if m['split'] == split)} numeric utterances")
    with open(os.path.join(args.out, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["id", "category", "split", "text", "fleurs_norm", "audio"])
        wr.writeheader()
        wr.writerows(meta)
    print(f"total {len(meta)} utterances -> {args.out}/metadata.csv")


if __name__ == "__main__":
    main()
