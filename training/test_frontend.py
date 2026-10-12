"""Parity test: C front-end (ew_mfcc.c via ctypes) vs an independent float64
NumPy implementation of the same MFCC definition.

    python training/test_frontend.py [--test-set <speech_commands_test_set_v0.02>]

Runs on synthetic audio by default (no dataset needed, used in CI); with
--test-set it also checks real speech clips.
"""
import argparse
import sys

import numpy as np

from frontend import Frontend


def mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mfcc_reference(pcm, fe, low_hz=20.0, high_hz=4000.0):
    x = pcm.astype(np.float64) / 32768.0
    n_frames = (len(x) - fe.frame_len) // fe.hop + 1
    idx = np.arange(fe.frame_len)[None, :] + fe.hop * np.arange(n_frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(fe.frame_len) / fe.frame_len)  # periodic Hann
    power = np.abs(np.fft.rfft(x[idx] * window, n=fe.fft_len)) ** 2

    bin_mel = mel(np.arange(fe.fft_len // 2 + 1) * fe.sample_rate / fe.fft_len)
    edges = np.linspace(mel(low_hz), mel(high_hz), fe.mel_bands + 2)
    left, centre, right = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (bin_mel - left) / (centre - left)
    falling = (right - bin_mel) / (right - centre)
    fbank = np.where((bin_mel > left) & (bin_mel <= centre), rising,
                     np.where((bin_mel > centre) & (bin_mel < right), falling, 0.0))
    log_mel = np.log(power @ fbank.T + 1e-6)

    n = np.arange(fe.mel_bands)
    k = np.arange(fe.num_mfcc)[:, None]
    dct = np.cos(np.pi / fe.mel_bands * (n + 0.5) * k) * np.sqrt(2.0 / fe.mel_bands)
    dct[0] /= np.sqrt(2.0)  # orthonormal DCT-II
    return log_mel @ dct.T


def synthetic_clips(n, rng):
    t = np.arange(16000) / 16000.0
    clips = []
    for i in range(n):
        f0 = rng.uniform(100, 3000)
        chirp = np.sin(2 * np.pi * (f0 + 500 * t) * t) * rng.uniform(500, 20000)
        env = np.exp(-((t - rng.uniform(0.2, 0.8)) ** 2) / 0.02)
        noise = rng.normal(0, rng.uniform(5, 500), t.size)
        clips.append(np.clip(chirp * env + noise, -32768, 32767).astype(np.int16))
    return np.stack(clips)


def check(name, pcm, fe):
    got = fe.mfcc(pcm).astype(np.float64)
    want = np.stack([mfcc_reference(p, fe) for p in pcm])
    err = np.abs(got - want)
    rel = err / (1.0 + np.abs(want))
    print(f"{name:10s} clips={len(pcm):4d}  max|err|={err.max():.2e}  mean|err|={err.mean():.2e}  "
          f"max rel={rel.max():.2e}")
    return rel.max() < 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set")
    args = ap.parse_args()

    fe = Frontend()
    ok = check("synthetic", synthetic_clips(64, np.random.default_rng(0)), fe)
    if args.test_set:
        from data import load_official_test_set
        pcm, _ = load_official_test_set(args.test_set)
        ok &= check("speech", pcm[::10], fe)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
