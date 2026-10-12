"""Replay long, realistic audio streams through the *firmware's* streaming
engine (ew_kws.c via ctypes: sliding window, VAD gate, smoothing, refractory)
with the exported int8 model as the inference callback.

    python training/stream_eval.py --data <speech_commands_v0.02> \
        --test <speech_commands_test_set_v0.02>

Each scenario is a 10-minute stream: a background track plus utterances from
the official test set (keywords and unknown words) dropped in at random gaps.
Reports, per scenario, with the VAD gate on and off:
  - keyword hit rate (right label within 1.5 s of the utterance)
  - false accepts per hour (detections that match no keyword utterance)
  - inference slots skipped by the VAD
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data import SILENCE, UNKNOWN, SpeechCommands, load_official_test_set
from export import Int8Model
from frontend import REPO, Frontend

SR = 16000

SCENARIOS = {
    # Mostly-idle device in a quiet room: a command every ~10 s on average.
    "quiet_room": {"noise": "pink_noise.wav", "noise_dbfs": -55, "gap_s": (4, 16)},
    # Same room, a command every ~5 s (busy usage).
    "quiet_room_busy": {"noise": "pink_noise.wav", "noise_dbfs": -55, "gap_s": (2, 8)},
    # Kitchen: running tap at a fairly high level, a command every ~10 s.
    "kitchen_tap": {"noise": "running_tap.wav", "noise_dbfs": -40, "gap_s": (4, 16)},
}


def word_end_s(clip):
    """End of the spoken word inside a 1 s clip: last 10 ms block whose energy
    is within 25 dB of the loudest block."""
    blocks = clip[: len(clip) // 160 * 160].astype(np.float64).reshape(-1, 160)
    e = (blocks ** 2).mean(axis=1) + 1e-9
    active = np.where(10 * np.log10(e / e.max()) > -25)[0]
    return (active[-1] + 1) * 0.01


def build_stream(ds, test_pcm, test_y, scenario, minutes, rng):
    n = int(minutes * 60 * SR)
    noise_name = scenario["noise"]
    noise = next(z for z, p in zip(ds.noises, sorted((ds.root / "_background_noise_").glob("*.wav")))
                 if p.name == noise_name)
    reps = int(np.ceil(n / len(noise)))
    bg = np.tile(noise, reps)[:n].astype(np.float64)
    bg *= 10 ** (scenario["noise_dbfs"] / 20) / (np.sqrt(np.mean(bg ** 2)) + 1e-12)
    stream = bg * 32768.0

    events = []
    speech_idx = np.where(test_y != SILENCE)[0]
    t = rng.uniform(*scenario["gap_s"])
    while t + 1.0 < minutes * 60 - 2:
        i = int(rng.choice(speech_idx))
        start = int(t * SR)
        stream[start:start + SR] += test_pcm[i].astype(np.float64)
        events.append({"start_s": t, "end_s": t + word_end_s(test_pcm[i]), "label": int(test_y[i])})
        t += 1.0 + rng.uniform(*scenario["gap_s"])
    return np.clip(np.round(stream), -32768, 32767).astype(np.int16), events


def score(det, events, hop_s):
    """Returns hits, wrong-label detections, false accepts, keyword count and,
    for each hit, the algorithmic latency: end of the hop that fired minus
    the end of the spoken word (negative = fired before the word ended)."""
    hits, wrong = 0, 0
    latencies = []
    used = set()
    keyword_events = [e for e in events if e["label"] != UNKNOWN]
    det_hops = np.where(det >= 0)[0]
    for e in keyword_events:
        lo, hi = e["start_s"], e["start_s"] + 1.5
        match = [h for h in det_hops if lo <= h * hop_s <= hi and h not in used]
        if any(det[h] == e["label"] for h in match):
            hits += 1
            h = next(h for h in match if det[h] == e["label"])
            used.add(h)
            latencies.append((h + 1) * hop_s - e["end_s"])
        elif match:
            wrong += 1
            used.add(match[0])
    false_accepts = len([h for h in det_hops if h not in used])
    return hits, wrong, false_accepts, len(keyword_events), np.array(latencies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--min-history", type=int, default=3)
    ap.add_argument("--refractory-hops", type=int, default=75)
    ap.add_argument("--out", default="stream_eval.json")
    ap.add_argument("--results", default=str(REPO / "training" / "results"))
    args = ap.parse_args()

    results = Path(args.results)
    q = Int8Model((results / "edgewake_kws_int8.tflite").read_bytes())
    mean, inv_std = load_feature_stats()

    fe = Frontend()
    ds = SpeechCommands(args.data)
    test_pcm, test_y = load_official_test_set(args.test)
    hop_s = fe.hop / SR

    report = {}
    for name, sc in SCENARIOS.items():
        rng = np.random.default_rng(2024)
        stream, events = build_stream(ds, test_pcm, test_y, sc, args.minutes, rng)
        report[name] = {"noise": sc["noise"], "noise_dbfs": sc["noise_dbfs"],
                        "utterances": len(events),
                        "keyword_utterances": sum(e["label"] != UNKNOWN for e in events)}
        for vad in (True, False):
            fe.kws_init(q.in_scale, q.in_zp, q.out_scale, q.out_zp, mean, inv_std, infer=q,
                        threshold=args.threshold, vad_enabled=vad, min_history=args.min_history,
                        refractory_hops=args.refractory_hops)
            gate, ran, det, _ = fe.kws_run(stream)
            st = fe.kws_stats()
            hits, wrong, fa, n_kw, lat = score(det, events, hop_s)
            hours = len(stream) / SR / 3600
            key = "vad_on" if vad else "vad_off"
            report[name][key] = {
                "hit_rate": round(hits / n_kw, 4),
                "wrong_label": wrong,
                "false_accepts_per_hour": round(fa / hours, 1),
                "infer_slots": st["infer_slots"],
                "inferences": st["inferences"],
                "skipped_by_vad_pct": round(100 * st["skipped_by_vad"] / st["infer_slots"], 1),
                "gate_open_pct": round(100 * gate.mean(), 1),
                "algorithmic_latency_ms": {
                    "median": round(1000 * float(np.median(lat)), 1),
                    "p90": round(1000 * float(np.percentile(lat, 90)), 1),
                    "max": round(1000 * float(lat.max()), 1),
                },
            }
            print(f"{name:16s} {key:7s} hit={hits}/{n_kw} ({hits / n_kw:.1%}) wrong={wrong} "
                  f"FA/h={fa / hours:.1f} inferences={st['inferences']}/{st['infer_slots']} "
                  f"skipped={report[name][key]['skipped_by_vad_pct']}% "
                  f"latency med/p90/max={report[name][key]['algorithmic_latency_ms']}", flush=True)
        off = report[name]["vad_off"]["inferences"]
        on = report[name]["vad_on"]["inferences"]
        report[name]["inference_reduction_pct"] = round(100 * (1 - on / off), 1)

    report["config"] = {"minutes_per_scenario": args.minutes, "threshold": args.threshold,
                        "infer_every_hops": 2, "smooth_window": 3, "min_history": args.min_history,
                        "refractory_hops": args.refractory_hops}
    (results / args.out).write_text(json.dumps(report, indent=2))


def load_feature_stats():
    """Reads mean / inv_std back from the generated firmware source, so this
    evaluates exactly what will be flashed."""
    src = (REPO / "firmware" / "main" / "model" / "model_settings.c").read_text()

    def grab(name):
        body = src.split(f"{name}[EW_NUM_MFCC] = {{")[1].split("}")[0]
        return np.array([float(v.strip().rstrip("f")) for v in body.split(",") if v.strip()], np.float32)

    return grab("ew_feat_mean"), grab("ew_feat_inv_std")


if __name__ == "__main__":
    main()
