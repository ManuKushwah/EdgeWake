# EdgeWake architecture

## Data path

```
 INMP441 ──I2S──► I2S0 + GDMA ──► 4 × 20 ms DMA buffers
 (16 kHz,                              │  i2s_channel_read() unblocks once per 20 ms
  24-bit)                              ▼
                     ┌──────── core 0 ─────────┐
                     │ audio task (prio 21)    │  int32 → DC-block → int16 (saturating)
                     └───────────┬─────────────┘
                                 │ ew_ringbuf_write()  (lock-free SPSC, 8192 samples = 512 ms)
                                 │ xTaskNotifyGive(kws)
                     ┌───────────▼─────────────┐
                     │ kws task (prio 15)      │──── core 1, nothing else pinned here
                     │  per 20 ms hop:         │
                     │   VAD energy gate       │  int64 energy → dB, μ + 3σ threshold, hangover
                     │   MFCC frame            │  Hann → 512-pt real FFT → 40 mel → log → DCT → 10
                     │   int8 feature ring     │  49 × 10, quantized once per frame
                     │  every 2nd hop, if gate │
                     │  open:                  │
                     │   TFLM Invoke()         │  int8 DS-CNN, ESP-NN kernels
                     │   smooth 3 posteriors   │
                     │   threshold+refractory  │
                     └──┬───────────────┬──────┘
          GPIO action   │               │ xQueueSend(event, timeout 0)
          (LED / relay) ▼               ▼
                                ┌──────── core 0 ─────────┐
                                │ telemetry task (prio 5) │──► esp-mqtt ──TLS 1.2 (mutual)──► AWS IoT Core
                                └─────────────────────────┘     detections / health / status / cmd
```

The platform-independent part (everything inside the kws task box, plus the
ring buffer) lives in `firmware/components/edgewake_core` and has no ESP-IDF or
TFLM dependency: inference is injected as a function pointer. The same C files
are compiled into the host unit tests and into the shared library that the
Python training and evaluation code load with ctypes.

## Tasks

| Task      | Core | Prio | Stack | Wakes on                    | Work per wake |
|-----------|------|------|-------|-----------------------------|---------------|
| audio     | 0    | 21   | 3 KB  | I2S DMA buffer complete (20 ms) | convert 320 samples, ring push, notify |
| kws       | 1    | 15   | 6 KB  | task notification            | VAD + MFCC per hop; CNN every 2nd hop when gated open |
| telemetry | 0    | 5    | 6 KB  | event queue / health period  | format JSON, `esp_mqtt_client_enqueue` |
| ota       | 0    | 4    | 8 KB  | task notification (MQTT cmd) | HTTPS download into the inactive slot |
| health    | 0    | 3    | 4 KB  | 1 s delay                    | liveness check, CPU-load window |

ESP-IDF's own tasks for reference: IPC 24, Wi-Fi 23, esp_timer 22, LwIP 18 (all
core 0 via `sdkconfig.defaults`). The audio task sits just below them: it does
~20 µs of work per 20 ms, and four DMA buffers (80 ms) absorb any Wi-Fi burst
that delays it. Core 1 runs only the kws task, so the CNN is never preempted by
networking.

**Why the ring buffer needs no lock.** The producer only writes `head`; the
consumer only writes `tail`. Each side reads the other's index with *acquire*
and publishes its own with *release*, so the sample data is visible before the
index that announces it. Indices are free-running `uint32_t` (`used = head -
tail` survives wrap-around), capacity is a power of two (`& mask` instead of
`%`). On overflow the producer drops the *new* chunk and counts it: advancing
`tail` would make it a second writer of the consumer's index. The host
stress test moves 4 M samples between two threads under ThreadSanitizer;
weakening the `head` store from release to relaxed makes TSan report a data
race on the sample `memcpy`.

## Memory

All buffers on the audio path are static. Nothing in capture, the ring, the
DSP or inference calls `malloc`. TFLM itself never allocates; its interpreter
and op resolver are placement-new'd into static storage.

| Buffer                         | Size     |
|--------------------------------|----------|
| audio ring (int16 × 8192)      | 16 KB    |
| `ew_kws_t` (MFCC tables, window, int8 feature ring, history) | 16.5 KB (16 520 B) |
| TFLM tensor arena              | 40 KB (35,516 B used, logged at boot) |
| model flatbuffer (flash, `.rodata`) | 45,992 B |
| I2S DMA (driver-owned)         | 4 × 1280 B |
| task stacks + TCBs             | 27 KB    |

Wi-Fi, LwIP, mbedTLS and esp-mqtt use the heap, outside the audio path. Health
telemetry reports free heap, minimum-ever free heap and per-task stack
high-water marks.

## Audio front-end

| Parameter | Value | Note |
|-----------|-------|------|
| Sample rate | 16 kHz | |
| Window | 512 samples (32 ms), periodic Hann | window = FFT size, no zero padding |
| Hop | 320 samples (20 ms) | = one DMA buffer |
| FFT | 512-point real FFT via 256-point complex FFT + split | radix-2 DIT, precomputed twiddles and bit-reversal |
| Mel | 40 triangular bands, 20 Hz – 4 kHz (HTK mel) | sparse weights: ~260 non-zeros instead of 40 × 257 |
| Log | natural log, ε = 1e-6 | |
| DCT | orthonormal DCT-II, first 10 coefficients | |
| Model input | 49 frames × 10 = 1 s | |

The streaming engine slides the analysis window by one hop, computes one new
frame, quantizes it straight to int8 (normalisation folded into a
per-coefficient multiply-add) and writes it into a 49-row circular buffer. At
inference time the ring is linearized oldest-first into the input tensor. MFCC
keeps running while the VAD gate is closed, so when speech starts the 1 s
window already holds fresh context.

Streaming frames end on hop boundaries, offline frames start on them. Feeding
128 samples of lead-in (`2 × 320 − 512`) aligns the two grids. The host test
`test_streaming_matches_offline_features` checks that the streaming path hands
the model byte-identical input to offline extraction. That is what allows
training on offline features.

The hot functions (`ew_rfft_power`, `ew_mfcc_compute`, VAD energy, ring
read/write) are marked `EW_IRAM` → `IRAM_ATTR`. Code that runs from flash goes
through the instruction cache. Every 20 ms the per-hop DSP runs interleaved
with TFLM kernels and the Wi-Fi stack, which evict it from cache. Keeping it in
internal SRAM removes cache-miss stalls from the per-hop timing.

## Model

DS-CNN (Zhang et al., *Hello Edge*, 2017): Conv 10×4/2 (64) → 4 × [depthwise
3×3 → pointwise 1×1 (64)] → average pool → FC(12) → softmax. BatchNorm is
folded into the convolutions at export and ReLU is fused. The model is
converted with a static batch of 1, so the flatten folds into the
fully-connected layer and the graph needs exactly five kernels: `CONV_2D,
DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, FULLY_CONNECTED, SOFTMAX`. Those five are
the ones registered in `inference.cc`.

Training features come from `ew_mfcc.c` through ctypes, so they are the
device's features. Augmentation: ±100 ms shift, ±6 dB gain, background noise
on 80 % of clips. Each epoch re-samples the unknown-word set. Post-training
int8 quantization uses per-channel weights and a 1,000-clip calibration set.
The int8 accuracy is measured through the device path: C MFCC → C quantizer →
int8 TFLite.

## Voice-activity gate

Per hop: `E = mean(x²)` (exact int64 accumulation), level `L = 10·log10(E + 1)`
dB. Speech if `L > μ + max(3σ, 6 dB)` and `L > 30 dB` (about −60 dBFS RMS),
where μ and σ are the running mean and standard deviation of the background
level. The threshold therefore scales with how much the background
*fluctuates*, not just how loud it is.

The statistics learn only from hops where the gate is fully closed, so speech
and its hangover tail don't inflate them. While the gate is open, μ only
drifts (α = 0.002), so a lasting level change is absorbed in about 7 s and the
gate cannot stick open (`test_adapts_to_louder_background`). Hangover keeps
the gate open for 300 ms after the last speech hop, so trailing consonants and
the tail of the 1 s window still get classified.

**Why this design.** The first version compared energy with 4× a minimum-
tracking floor. Replaying pink-noise streams showed it open on 72 % of hops:
the per-hop energy of low-frequency-heavy noise varies about 5×, and a floor
that hugs the quietest hops is exceeded constantly. Moving to dB and scaling
the threshold with σ cut that to 14 %. The gate was still open at the end of
the same share of spoken words. `test_fluctuating_background_stays_closed`
pins this behaviour.

Only the CNN is gated. When the gate is closed the kws task still pays for VAD
and MFCC, which is a small fraction of one inference. The gate is
runtime-switchable over MQTT (`{"vad": false}`) for A/B measurement of CPU
load on the device.

## Latency budget (keyword end → GPIO)

| Stage | Budget | Source |
|-------|--------|--------|
| DMA buffer fill (last sample waits for its 20 ms buffer) | 0–20 ms | by construction |
| ring + notify | < 0.1 ms | |
| wait for the next inference slot (every 2nd hop) | 0–20 ms | by construction |
| VAD + MFCC for the hop | ~0.3 ms (estimate) | measured on device by the DSP probe |
| `Invoke()` | 14–22 ms (estimate: `training/estimate_latency.py`) | measured by the INFER probe / `infer_us_*` |
| smoothing + threshold | 0–1 extra inference slots | measured on host replay, see results |
| GPIO write | < 1 µs | |

The algorithmic part (everything except compute time) is measured by
replaying 10-minute streams through the C engine on the host:
`training/results/stream_eval.json`, field `algorithmic_latency_ms`.

## OTA and rollback

```
 running image (VALID) ── MQTT {"ota_url":"https://…"} ──► ota task
     esp_https_ota_begin → read app descriptor → refuse same version
     esp_https_ota_perform (chunks into the inactive slot, CA-verified HTTPS)
     esp_https_ota_finish → image checksum verified, otadata → new slot (state NEW)
     esp_restart()
 bootloader: NEW → PENDING_VERIFY, boot new slot
 app_main self-test:
     golden vector: known test clip through TFLM, argmax + scores within ±2 LSB
     microphone delivers a non-constant signal within 3 s
     ≥ 100 hops processed within 5 s
     Wi-Fi connects within 30 s (if configured)
   pass → esp_ota_mark_app_valid_cancel_rollback()
   fail → esp_ota_mark_app_invalid_rollback_and_reboot()
          (if the API refuses because otadata has no record of another slot:
           esp_restart() anyway, see below)
 crash / watchdog before confirming → bootloader finds an unconfirmed image
   → marks it ABORTED and boots the previous slot
```

`esp_ota_mark_app_invalid_rollback_and_reboot()` only rolls back when otadata
holds a valid record of the other slot, and otherwise *returns*. The
bootloader is more forgiving: after an aborted image it scans every app slot
and boots the first valid one, or re-boots the same image (now ABORTED, so it
runs normally) if nothing else exists. So on refusal the firmware just resets
and lets the bootloader fall back, and the device is never left without a
bootable app.

Both paths are exercised on the emulated ESP32-S3 by
`firmware/tools/qemu_test.py rollback --new <v1.0.1.bin> [--no-previous-record]`.
It writes a second image into `ota_1` and otadata exactly as
`esp_ota_set_boot_partition()` leaves it. QEMU has no microphone, so the
self-test fails and the run must end on `ota_0`. The first version of this
code assumed the rollback call never returns; the no-record case exposed
that.

Partition table: `nvs`, `otadata`, `phy_init`, `ota_0` (1.875 MB), `ota_1`
(1.875 MB). There is no factory slot, which fits 4 MB of flash.

## Watchdogs and health

* **Task watchdog (TWDT, 5 s, panic → reboot):** subscribed by the audio, kws
  and health tasks. It also watches both idle tasks, so a runaway task that
  starves a core gets caught too.
* **Interrupt watchdog:** default ESP-IDF configuration.
* **Pipeline liveness (health task):** a task can keep feeding the TWDT while
  the data path is stuck, for example if I2S stops delivering. The health task
  checks every second that the audio chunk counter and the kws hop counter are
  both advancing, and reboots after 10 s of stall.
* **Reset reason** (`task_wdt`, `panic`, `brownout`, …) is reported in health
  telemetry, so watchdog resets are visible remotely.

## Emulator checks (QEMU)

`firmware/tools/qemu_test.py boot` boots the release image on Espressif's
QEMU `esp32s3` machine. It checks that the model loads, that the arena fits,
that the golden vector produces bit-identical scores to the PC reference
(Xtensa + ESP-NN vs TFLite reference kernels), and that the kws task starts on
core 1. The first boot of the original 32 KB arena failed `AllocateTensors()`:
ESP-NN's convolution scratch buffers needed 2.6 KB more than planned. The
arena is now 40 KB with 35.5 KB used. QEMU doesn't model the I2S microphone,
so after ~10 s the liveness monitor reboots the emulated chip, which also
exercises that path.

## Timing probes (logic analyzer)

With `CONFIG_EW_DEBUG_PROBES` (default on), four GPIOs mark the pipeline. They
are written with single `GPIO_OUT_W1TS/W1TC` register stores, so the probes
themselves add no measurable delay.

| Probe | Default GPIO | High while |
|-------|--------------|------------|
| AUDIO | 15 | capture task converts + pushes a chunk |
| DSP   | 16 | kws task processes one hop (VAD + MFCC [+ CNN]) |
| INFER | 17 | `Invoke()` |
| ACTION | 18 | keyword action (short pulse) |

With these four probes:

* AUDIO rising-edge spacing gives the 20 ms DMA cadence and its jitter.
* DSP width on non-inference hops gives front-end cost.
* INFER width gives inference time.
* AUDIO → ACTION gives the end-to-end latency.
* DSP duty cycle with the VAD on versus off gives the CPU saving.
