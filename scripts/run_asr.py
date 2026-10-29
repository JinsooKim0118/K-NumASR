import argparse
import csv
import json
import os
import time

import soundfile as sf
import torch

NUMERIC_PROMPT = "전화번호 010-2468-1357, 계좌 302-11-845920, 금액 47,500원, 2025년 6월 3일 오후 4시 15분, 12살, 7명."


def load_eval_set(meta_paths):
    items = []
    for p in meta_paths:
        base = os.path.dirname(p)
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                items.append({"id": r["id"], "category": r["category"], "text": r["text"], "path": os.path.join(base, r["audio"])})
    return items


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def run_whisper(model_id, items, prompt=None, batch_size=16):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device=device, dtype=dtype).eval()
    prompt_ids = proc.get_prompt_ids(prompt, return_tensors="pt").to(device) if prompt else None
    bs = 1 if prompt else batch_size
    hyps = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        feats = proc([load_audio(x["path"]) for x in chunk], sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(device, dtype=dtype)
        kwargs = {"language": "ko", "task": "transcribe", "max_new_tokens": 200, "num_beams": 1, "do_sample": False}
        if prompt_ids is not None:
            kwargs["prompt_ids"] = prompt_ids
        with torch.no_grad():
            out = model.generate(feats, **kwargs)
        texts = proc.batch_decode(out, skip_special_tokens=True)
        for t in texts:
            t = t.strip()
            if prompt and t.startswith(prompt):
                t = t[len(prompt):].strip()
            hyps.append(t)
        print(f"  {min(i + bs, len(items))}/{len(items)}", end="\r")
    print()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return hyps


def run_ctc(model_id, items):
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=model_id, device=device)
    hyps = []
    for i, x in enumerate(items):
        hyps.append(asr(load_audio(x["path"]))["text"].strip())
        print(f"  {i + 1}/{len(items)}", end="\r")
    print()
    return hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--type", choices=["whisper", "ctc"], default="whisper")
    ap.add_argument("--prompt", action="store_true")
    ap.add_argument("--meta", nargs="+", default=["data/synthetic/metadata.csv", "data/fleurs_num/metadata.csv"])
    ap.add_argument("--out_dir", default="predictions")
    args = ap.parse_args()
    items = load_eval_set(args.meta)
    t0 = time.time()
    if args.type == "whisper":
        hyps = run_whisper(args.model, items, NUMERIC_PROMPT if args.prompt else None)
    else:
        hyps = run_ctc(args.model, items)
    elapsed = time.time() - t0
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.system}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for x, h in zip(items, hyps):
            f.write(json.dumps({"id": x["id"], "category": x["category"], "ref": x["text"], "hyp": h}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, f"{args.system}.info.json"), "w", encoding="utf-8") as f:
        json.dump({"system": args.system, "model": args.model, "prompt": args.prompt, "seconds": round(elapsed, 1),
                   "device": "cuda" if torch.cuda.is_available() else "cpu", "n": len(items)}, f, ensure_ascii=False, indent=1)
    print(f"{args.system}: {len(items)} utterances in {elapsed:.1f}s -> {path}")


if __name__ == "__main__":
    main()
