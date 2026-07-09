"""Prepare video dataset: extract frames, detect faces, and optionally extract audio/MFCCs.

Usage:
  python scripts/prepare_video_dataset.py --input data/video_authenticity --out data/video_authenticity/processed --face-only

This script expects folders under --input: `real/` and `ai_generated/` with video files.
It produces per-video folders with frame crops under --out and optional audio MFCCs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import shutil
import sys
import uuid
from typing import List

import cv2
import numpy as np


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def extract_frames(video_path: Path, sample_rate: int = 2) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_rate == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def detect_faces(frames: List[np.ndarray], scaleFactor=1.1, minNeighbors=4):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    crops = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        if len(rects) == 0:
            # fallback: full-frame crop
            h, w = f.shape[:2]
            crops.append(cv2.resize(f, (224, 224)))
        else:
            # take largest face
            x, y, w, h = max(rects, key=lambda r: r[2] * r[3])
            crop = f[y : y + h, x : x + w]
            crop = cv2.resize(crop, (224, 224))
            crops.append(crop)
    return crops


def extract_audio_mfcc(video_path: Path, out_npy: Path, sr=16000, n_mfcc=13):
    try:
        import librosa
    except Exception:
        print('librosa not installed; skipping audio MFCCs')
        return False
    # find ffmpeg: prefer system, fall back to workspace tools/ffmpeg/bin
    ffmpeg_bin = shutil.which('ffmpeg')
    local_ffmpeg = Path('tools/ffmpeg')
    if not ffmpeg_bin and (local_ffmpeg.exists()):
        # look for bin/ffmpeg(.exe)
        cand = local_ffmpeg / 'bin' / 'ffmpeg.exe'
        if not cand.exists():
            cand = local_ffmpeg / 'ffmpeg.exe'
        if cand.exists():
            ffmpeg_bin = str(cand)

    if not ffmpeg_bin:
        print('ffmpeg not found on PATH and no local tools/ffmpeg available; skipping audio MFCCs')
        return False

    # extract audio via ffmpeg into a temp wav
    tmp_wav = out_npy.with_suffix('.wav')
    cmd = [ffmpeg_bin, '-y', '-i', str(video_path), '-ac', '1', '-ar', str(sr), str(tmp_wav)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not tmp_wav.exists():
        return False
    y, _ = librosa.load(str(tmp_wav), sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    np.save(out_npy, mfcc)
    try:
        tmp_wav.unlink()
    except Exception:
        pass
    return True


def process_split(input_root: Path, output_root: Path, sample_rate: int, audio: bool):
    for label in ['real', 'ai_generated']:
        src = input_root / label
        if not src.exists():
            continue
        for f in sorted(src.iterdir()):
            if not f.is_file():
                continue
            vid_id = f.stem + '_' + uuid.uuid4().hex[:8]
            dest = output_root / label / vid_id
            ensure_dir(dest)
            print('Processing', f, '->', dest)
            frames = extract_frames(f, sample_rate=sample_rate)
            crops = detect_faces(frames)
            for i, c in enumerate(crops):
                outp = dest / f'frame_{i:04d}.jpg'
                cv2.imwrite(str(outp), c)
            if audio:
                mfcc_path = dest / 'audio_mfcc.npy'
                ok = extract_audio_mfcc(f, mfcc_path)
                if ok:
                    print('Saved MFCC', mfcc_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/video_authenticity')
    parser.add_argument('--out', default='data/video_authenticity/processed')
    parser.add_argument('--sample-rate', type=int, default=2)
    parser.add_argument('--audio', action='store_true')
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.out)
    ensure_dir(output_root)
    process_split(input_root, output_root, args.sample_rate, args.audio)


if __name__ == '__main__':
    main()
