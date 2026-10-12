"""Train the DS-CNN on Speech Commands v0.02 with features from the C front-end.

    python training/train.py --data /path/to/speech_commands_v0.02 --out training/runs/dscnn

Each epoch gets a freshly augmented training set (time shift, gain, background
noise, new unknown-word sample). It is produced in a background thread while
the previous epoch trains; the C feature extractor releases the GIL, so the
two overlap.
"""
import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from data import LABELS, SpeechCommands
from frontend import Frontend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="training/runs/dscnn")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    import keras
    import tensorflow as tf

    from model import build_dscnn, count_macs

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    keras.utils.set_random_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fe = Frontend()
    ds = SpeechCommands(args.data)
    print({k: len(v) for k, v in ds.files.items()})

    # Per-coefficient normalisation stats from clean training audio. Exported
    # to the firmware, where they are folded into the int8 input quantizer.
    pcm, _ = ds.make_set("training", np.random.default_rng(1), augment=False, limit=8000)
    feats = fe.mfcc(pcm).reshape(-1, fe.num_mfcc)
    mean = feats.mean(axis=0).astype(np.float32)
    std = (feats.std(axis=0) + 1e-6).astype(np.float32)
    np.savez(out / "feature_stats.npz", mean=mean, std=std)

    def features(pcm_batch):
        return ((fe.mfcc(pcm_batch) - mean) / std)[..., None]

    val_pcm, val_y = ds.make_set("validation", np.random.default_rng(2), augment=False)
    val_x = features(val_pcm)
    print("validation set:", val_x.shape)

    model = build_dscnn(input_shape=(fe.num_frames, fe.num_mfcc, 1),
                        num_classes=len(LABELS), filters=args.filters)
    model.summary(print_fn=lambda s: None)
    print(f"params={model.count_params()}  MACs/inference={count_macs(model)}")
    optimizer = keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def epoch_data(epoch):
        rng = np.random.default_rng(1000 + epoch)
        pcm_e, y_e = ds.make_set("training", rng, augment=True)
        return features(pcm_e), y_e

    best_acc, history = 0.0, []
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(epoch_data, 0)
        for epoch in range(args.epochs):
            t0 = time.time()
            x, y = pending.result()
            if epoch + 1 < args.epochs:
                pending = pool.submit(epoch_data, epoch + 1)
            # Cosine decay from lr to ~0 over the run, with a 1-epoch warm-up.
            lr = args.lr * min(1.0, (epoch + 1) / 2) * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
            optimizer.learning_rate.assign(max(lr, 1e-5))
            h = model.fit(x, y, batch_size=args.batch, epochs=1, shuffle=True, verbose=0)
            val_loss, val_acc = model.evaluate(val_x, val_y, batch_size=500, verbose=0)
            history.append({"epoch": epoch, "lr": lr, "loss": h.history["loss"][0],
                            "acc": h.history["accuracy"][0], "val_loss": val_loss, "val_acc": val_acc})
            flag = ""
            if val_acc > best_acc:
                best_acc = val_acc
                model.save(out / "best.keras")
                flag = "  *"
            print(f"epoch {epoch + 1:3d}/{args.epochs}  lr={lr:.2e}  loss={h.history['loss'][0]:.4f} "
                  f"acc={h.history['accuracy'][0]:.4f}  val_acc={val_acc:.4f}  ({time.time() - t0:.0f}s){flag}",
                  flush=True)

    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
