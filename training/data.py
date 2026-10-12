"""Google Speech Commands v0.02 loading and augmentation (12-class task).

Classes: _silence_, _unknown_, then the 10 standard keywords. Splits follow the
dataset's validation_list.txt / testing_list.txt; the reported accuracy uses
the official pre-built v0.02 test set (4,890 clips).
"""
import hashlib
import wave
from pathlib import Path

import numpy as np

KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = ["_silence_", "_unknown_"] + KEYWORDS
SILENCE, UNKNOWN = 0, 1
SAMPLE_RATE = 16000
CLIP_LEN = 16000


def load_wav(path, length=CLIP_LEN):
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    out = np.zeros(length, np.int16)
    n = min(length, len(x))
    out[:n] = x[:n]
    return out


def load_wav_any_length(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int16)


class SpeechCommands:
    def __init__(self, root):
        self.root = Path(root)
        val = set((self.root / "validation_list.txt").read_text().split())
        test = set((self.root / "testing_list.txt").read_text().split())
        self.files = {"training": [], "validation": [], "testing": []}  # (path, label)
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if d.name.startswith("_"):
                continue
            label = LABELS.index(d.name) if d.name in KEYWORDS else UNKNOWN
            for f in sorted(d.glob("*.wav")):
                rel = f"{d.name}/{f.name}"
                split = "validation" if rel in val else "testing" if rel in test else "training"
                self.files[split].append((f, label))
        self.noises = [load_wav_any_length(p).astype(np.float32) / 32768.0
                       for p in sorted((self.root / "_background_noise_").glob("*.wav"))]

    def _noise_clip(self, rng):
        n = self.noises[rng.integers(len(self.noises))]
        start = rng.integers(0, len(n) - CLIP_LEN)
        return n[start:start + CLIP_LEN]

    def make_set(self, split, rng, augment, silence_pct=10.0, unknown_pct=10.0, limit=None):
        """Returns (pcm int16 (N, 16000), labels int64 (N,)).

        All keyword clips of the split, plus unknown-word clips and synthetic
        silence clips each sized as a percentage of the keyword count (the
        recipe from TensorFlow's speech_commands example). Unknown clips are
        re-sampled every call, so each epoch sees different unknown words.
        """
        items = self.files[split]
        kw = [it for it in items if it[1] != UNKNOWN]
        unk = [it for it in items if it[1] == UNKNOWN]
        n_unk = int(np.ceil(len(kw) * unknown_pct / 100))
        n_sil = int(np.ceil(len(kw) * silence_pct / 100))
        chosen = kw + [unk[i] for i in rng.choice(len(unk), n_unk, replace=False)]
        order = rng.permutation(len(chosen) + n_sil)
        if limit:
            order = order[:limit]

        pcm = np.zeros((len(order), CLIP_LEN), np.int16)
        labels = np.zeros(len(order), np.int64)
        for row, idx in enumerate(order):
            if idx < len(chosen):
                path, label = chosen[idx]
                x = load_wav(path).astype(np.float32) / 32768.0
                labels[row] = label
                if augment:
                    x = self._augment(x, rng)
            else:
                labels[row] = SILENCE
                x = self._noise_clip(rng) * rng.uniform(0.0, 1.0)
            pcm[row] = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
        return pcm, labels

    def _augment(self, x, rng):
        # Random time shift of up to +-100 ms, zero-filled.
        shift = int(rng.integers(-1600, 1601))
        if shift > 0:
            x = np.concatenate([np.zeros(shift, np.float32), x[:-shift]])
        elif shift < 0:
            x = np.concatenate([x[-shift:], np.zeros(-shift, np.float32)])
        # Random gain of +-6 dB: the device's microphone level won't match the dataset.
        x = x * np.float32(10.0 ** (rng.uniform(-6.0, 6.0) / 20.0))
        # Background noise on 80% of clips at up to 10% volume.
        if rng.random() < 0.8:
            x = x + self._noise_clip(rng) * np.float32(rng.uniform(0.0, 0.1))
        return np.clip(x, -1.0, 1.0)


def load_official_test_set(root):
    """The pre-built speech_commands_test_set_v0.02 (folders per label)."""
    root = Path(root)
    pcm, labels = [], []
    for label_idx, name in enumerate(LABELS):
        for f in sorted((root / name).glob("*.wav")):
            pcm.append(load_wav(f))
            labels.append(label_idx)
    return np.stack(pcm), np.array(labels, np.int64)


def split_fingerprint(files):
    """Short hash of a file list, recorded in metrics for reproducibility."""
    h = hashlib.sha1()
    for path, label in files:
        h.update(f"{Path(path).parent.name}/{Path(path).name}:{label}".encode())
    return h.hexdigest()[:12]
