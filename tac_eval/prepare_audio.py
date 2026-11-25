#!/usr/bin/env python3
"""Build LongCap client manifests for LongCapBench-TAC-Long, and stitch segment runs.

Subcommands
  manifest  Verify <audio-dir>/<key>.wav against sources.jsonl and write a
            full-recording manifest for `longcap caption --manifest`.
            With --segment 600, also cut non-overlapping 10-minute WAVs
            (the paper's segment-level setting, Fig. 1) and write their manifest.
  stitch    Shift segment predictions back to recording time and merge them
            into one predictions.jsonl that score_tac.py can read.

Audio must be 16 kHz mono PCM16 WAV, e.g.
  ffmpeg -i source.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le -fflags +bitexact -map_metadata -1 KEY.wav
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_manifest(args):
    sources = load_jsonl(args.sources)
    audio_dir = Path(args.audio_dir).resolve()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, seg_rows = [], []
    ok_all = True
    for s in sources:
        key = s["key"]
        wav_path = audio_dir / f"{key}.wav"
        if not wav_path.exists():
            print(f"[missing] {wav_path}", file=sys.stderr)
            ok_all = False
            continue
        with wave.open(str(wav_path), "rb") as w:
            sr, ch, sw, nf = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        header_ok = (sr, ch, sw, nf) == (s["sample_rate"], s["channels"], s["sample_width_bytes"], s["expected_frames"])
        sha_ok = args.skip_sha or sha256(wav_path) == s["expected_wav_sha256"]
        status = "verified" if header_ok and sha_ok else "MISMATCH"
        print(f"[{status}] {key}: {sr} Hz, {ch} ch, {nf / sr:.1f} s "
              f"(header {'ok' if header_ok else 'differs'}, sha256 {'ok' if sha_ok else 'differs'})", file=sys.stderr)
        if not (header_ok and sha_ok):
            ok_all = False
            if not args.allow_mismatch:
                continue
        duration = nf / sr
        rows.append({"key": key, "audio_path": str(wav_path), "duration": duration})

        if args.segment:
            seg_dir = Path(args.segment_dir).resolve()
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg_frames = int(args.segment * sr)
            with wave.open(str(wav_path), "rb") as w:
                n = 0
                while True:
                    frames = w.readframes(seg_frames)
                    if not frames:
                        break
                    seg_key = f"{key}__seg{n:03d}"
                    seg_path = seg_dir / f"{seg_key}.wav"
                    with wave.open(str(seg_path), "wb") as o:
                        o.setnchannels(ch)
                        o.setsampwidth(sw)
                        o.setframerate(sr)
                        o.writeframes(frames)
                    seg_rows.append({"key": seg_key, "audio_path": str(seg_path),
                                     "duration": len(frames) / (sw * ch) / sr,
                                     "source_key": key, "offset": n * args.segment})
                    n += 1

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} recordings to {out}", file=sys.stderr)
    if args.segment:
        seg_out = Path(args.segment_manifest)
        with seg_out.open("w", encoding="utf-8") as f:
            for r in seg_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(seg_rows)} segments to {seg_out}", file=sys.stderr)
    return 0 if ok_all else 1


def cmd_stitch(args):
    segs = {r["key"]: r for r in load_jsonl(args.segment_manifest)}
    preds = {r["key"]: r for r in load_jsonl(args.predictions)}
    full = {}
    for seg_key, seg in sorted(segs.items()):
        rec = full.setdefault(seg["source_key"], {"key": seg["source_key"], "duration": 0.0, "events": []})
        rec["duration"] = max(rec["duration"], seg["offset"] + seg["duration"])
        p = preds.get(seg_key)
        if p is None:
            print(f"[warning] no prediction for {seg_key}", file=sys.stderr)
            continue
        for e in p.get("events", []):
            e = dict(e)
            e["start"] = float(e["start"]) + seg["offset"]
            e["end"] = float(e["end"]) + seg["offset"]
            rec["events"].append(e)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in full.values():
            rec["events"].sort(key=lambda e: (e["start"], e["end"]))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"stitched {len(segs)} segments into {len(full)} recordings: {args.out}", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--sources", required=True, help="sources.jsonl from the benchmark")
    m.add_argument("--audio-dir", required=True, help="directory with <key>.wav files")
    m.add_argument("--out", required=True, help="full-recording manifest to write")
    m.add_argument("--segment", type=float, help="also cut segments of this many seconds (paper: 600)")
    m.add_argument("--segment-dir", default="audio_segments")
    m.add_argument("--segment-manifest", default="manifest_segments.jsonl")
    m.add_argument("--skip-sha", action="store_true", help="check the header only (faster)")
    m.add_argument("--allow-mismatch", action="store_true",
                   help="keep recordings whose audio differs from the evaluated WAV (report this!)")
    s = sub.add_parser("stitch")
    s.add_argument("--segment-manifest", required=True)
    s.add_argument("--predictions", required=True, help="predictions.jsonl of the segment run")
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(cmd_manifest(args) if args.cmd == "manifest" else cmd_stitch(args))


if __name__ == "__main__":
    main()
