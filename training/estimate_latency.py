"""Estimate Invoke() time on the ESP32-S3 from published ESP-NN kernel cycle
counts. This is an ESTIMATE until it is measured on hardware (INFER probe /
`timing.infer_us_*` in health telemetry, see docs/HARDWARE.md).

Method: for every layer, MACs / (MACs per cycle of the closest kernel that
Espressif benchmarked on the ESP32-S3 at 240 MHz, esp-nn README, "Kernelwise
performance on ESP32-S3"). The benchmarks put data in external PSRAM. EdgeWake's
arena is in internal SRAM, so they are treated as the pessimistic bound; the
optimistic bound assumes modestly better utilisation.

    python training/estimate_latency.py
"""
import json
from pathlib import Path

CPU_HZ = 240e6

# (benchmark description, MACs in the benchmark, cycles measured with ESP-NN)
REFERENCE = {
    "conv_few_in_ch": ("conv input(8,8) filter 64x3x3x3, pad 0", 6 * 6 * 64 * 27, 394_842),
    "conv_1x1": ("conv input(10,10) filter 64x1x1x64", 10 * 10 * 64 * 64, 331_008),
    "depthwise_3x3": ("depthwise input(18,18) filter 1x3x3x16, pad 0", 16 * 16 * 16 * 9, 184_176),
    "avg_pool": ("avg pool input(16,16) filter 3x3, 16 ch", 14 * 14 * 16 * 9, 118_052),
    "fully_connected": ("fully connected len 271, 3 ch", 271 * 3, 1_543),
    "softmax": ("softmax 8 x 32", 256, 11_107),
}

# EdgeWake DS-CNN layers: (name, kernel class, MACs or element-ops, optimistic speed-up)
LAYERS = [
    ("conv0 10x4, 1->64, out 25x5", "conv_few_in_ch", 25 * 5 * 64 * 40, 1.6),
    *[(f"dw{i} 3x3, 64 ch, 25x5", "depthwise_3x3", 25 * 5 * 64 * 9, 2.0) for i in range(1, 5)],
    *[(f"pw{i} 1x1, 64->64, 25x5", "conv_1x1", 25 * 5 * 64 * 64, 1.2) for i in range(1, 5)],
    ("avg pool 25x5, 64 ch", "avg_pool", 25 * 5 * 64, 1.0),
    ("fc 64->12", "fully_connected", 64 * 12, 1.0),
    ("softmax 12", "softmax", 12, 1.0),
]


def main():
    rows, lo, hi = [], 0.0, 0.0
    for name, kind, ops, speedup in LAYERS:
        _, ref_ops, ref_cycles = REFERENCE[kind]
        cycles = ops * ref_cycles / ref_ops
        hi += cycles
        lo += cycles / speedup
        rows.append({"layer": name, "ops": ops, "reference": REFERENCE[kind][0],
                     "cycles_pessimistic": round(cycles)})
        print(f"{name:30s} {ops:>9,d} ops  ~{cycles / 1e3:8.0f} k cycles")
    est = {"cpu_mhz": CPU_HZ / 1e6, "layers": rows,
           "invoke_ms_estimate": [round(lo / CPU_HZ * 1e3, 1), round(hi / CPU_HZ * 1e3, 1)],
           "note": "Estimate from esp-nn README kernel benchmarks (ESP32-S3, 240 MHz). Not a measurement."}
    print(f"estimated Invoke(): {est['invoke_ms_estimate'][0]}-{est['invoke_ms_estimate'][1]} ms @ 240 MHz")
    out = Path(__file__).resolve().parent / "results" / "latency_estimate.json"
    out.write_text(json.dumps(est, indent=2))


if __name__ == "__main__":
    main()
