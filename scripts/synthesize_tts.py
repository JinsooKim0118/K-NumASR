import argparse
import asyncio
import csv
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

RATES = ["+0%", "+15%", "-10%"]


async def _edge_voices():
    import edge_tts
    voices = await edge_tts.list_voices()
    return sorted(v["ShortName"] for v in voices if v["Locale"] == "ko-KR")


async def _edge_one(text, voice, rate, mp3_path):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate).save(mp3_path)


def mp3_to_16k(mp3_path):
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path, "-ac", "1", "-ar", "16000", tmp.name], check=True)
        audio, _ = sf.read(tmp.name, dtype="float32")
    return audio


def synth_edge(rows, out_dir):
    voices = asyncio.run(_edge_voices())
    if not voices:
        raise RuntimeError("no ko-KR voices")
    print("edge-tts voices:", voices)
    for i, r in enumerate(rows):
        voice = voices[i % len(voices)]
        rate = RATES[(i // len(voices)) % len(RATES)]
        rel = os.path.join("audio", r["id"] + ".flac")
        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
            for attempt in range(3):
                try:
                    asyncio.run(_edge_one(r["spoken"], voice, rate, tmp.name))
                    break
                except Exception:
                    if attempt == 2:
                        raise
            audio = mp3_to_16k(tmp.name)
        sf.write(os.path.join(out_dir, rel), audio, 16000, format="FLAC")
        r.update({"audio": rel, "tts": "edge-tts", "voice": voice, "rate": rate})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)}")


def synth_mms(rows, out_dir):
    import torch
    from transformers import AutoTokenizer, VitsModel
    import librosa
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
    model = VitsModel.from_pretrained("facebook/mms-tts-kor").eval()
    roman = None
    if getattr(tok, "is_uroman", False):
        import uroman as ur
        roman = ur.Uroman()
    for i, r in enumerate(rows):
        text = roman.romanize_string(r["spoken"]) if roman else r["spoken"]
        with torch.no_grad():
            wav = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
        wav = librosa.resample(wav.astype(np.float32), orig_sr=model.config.sampling_rate, target_sr=16000)
        rel = os.path.join("audio", r["id"] + ".flac")
        sf.write(os.path.join(out_dir, rel), wav, 16000, format="FLAC")
        r.update({"audio": rel, "tts": "mms-tts-kor", "voice": "mms", "rate": "+0%"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/synthetic/metadata.csv")
    ap.add_argument("--backend", choices=["edge", "mms"], default="edge")
    args = ap.parse_args()
    out_dir = os.path.dirname(args.meta)
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    with open(args.meta, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.backend == "edge":
        try:
            synth_edge(rows, out_dir)
        except Exception as e:
            print(f"edge-tts failed ({e}); falling back to MMS-TTS")
            synth_mms(rows, out_dir)
    else:
        synth_mms(rows, out_dir)
    fields = ["id", "category", "text", "spoken", "audio", "tts", "voice", "rate"]
    with open(args.meta, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows({k: r.get(k, "") for k in fields} for r in rows)
    print(f"synthesized {len(rows)} utterances")


if __name__ == "__main__":
    main()
