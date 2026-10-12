# EdgeWake

**On-device wake-word / keyword-spotting engine for the ESP32-S3.**
C/C++ · FreeRTOS · TensorFlow Lite Micro · ESP-IDF v5.5

[![CI](https://github.com/ManuKushwah/EdgeWake/actions/workflows/ci.yml/badge.svg)](https://github.com/ManuKushwah/EdgeWake/actions/workflows/ci.yml)

EdgeWake listens continuously through an I2S MEMS microphone and recognizes
10 spoken commands (*yes, no, up, down, left, right, on, off, stop, go*)
entirely on the microcontroller. Commands drive GPIOs directly. Detections and
health telemetry are published to AWS IoT Core over MQTT with mutual TLS. The
firmware updates itself over the air from an A/B partition pair and rolls back
automatically when a new image fails its self-test.

```
INMP441 ─I2S/DMA─► audio task (core 0) ─► lock-free SPSC ring ─► KWS task (core 1)
                                                                   │ energy VAD gate
                                                                   │ MFCC (C, IRAM)
                                                                   │ int8 DS-CNN (TFLM + ESP-NN)
                                                                   ▼
                                     GPIO action ◄── smoothing + threshold ──► event queue ─► MQTT/TLS ─► AWS IoT Core
```

## Highlights

* **Real-time audio path in C:** 16 kHz I2S with DMA (4 × 20 ms buffers) feeds
  a lock-free single-producer/single-consumer ring buffer (C11 acquire/release
  atomics). The producer runs on core 0 and the consumer on core 1, as
  priority-scheduled FreeRTOS tasks. Every buffer, task stack, queue and the
  tensor arena on the audio path is statically allocated.
* **MFCC front-end written from scratch in C:** 512-point real FFT computed
  as a 256-point complex FFT plus a split step, a sparse 40-band mel
  filterbank, log, and DCT-II to 10 coefficients. The hot loops are placed in
  IRAM. The same C code produces the training features via ctypes, so features
  can't differ between training and the device.
* **int8 depthwise-separable CNN** (DS-CNN, 24.3 K parameters, 2.66 M MACs per
  inference, 45 KB flatbuffer, 94.0 % on the official Speech Commands test set)
  run by the TensorFlow Lite Micro C++ runtime with ESP-NN kernels.
  Interpreter, resolver and arena are statically allocated; TFLM never uses the
  heap.
* **Energy VAD gate:** the threshold adapts to the background level *and* its
  variability, with hangover. It skips 80–90 % of CNN invocations on realistic
  streams while the MFCC window keeps running, so no context is lost when
  speech starts.
* **AWS IoT Core** over MQTT/TLS 1.2 with X.509 client certificates: detection
  events (QoS 1), periodic health (CPU load, heap, stack high-water marks, VAD
  skip rate, reset reason), LWT status, and runtime commands.
* **A/B OTA with automatic rollback:** HTTPS download into the inactive slot,
  then a first-boot self-test (golden-vector inference, mic alive, pipeline
  running, network up). The image is confirmed only if the self-test passes;
  otherwise the device rolls back. **Task watchdog** on every pipeline task,
  plus a data-path liveness monitor.
* **Logic-analyzer probes:** four GPIOs driven with single W1TS/W1TC register
  writes mark DMA chunks, per-hop DSP, `Invoke()` and actions.

## Results

Everything below is reproducible from this repo (`training/results/`,
`firmware/tools/qemu_test.py`). Each number is one of:
- **measured**: produced on a PC by the device's own C code (MFCC, quantizer,
  streaming engine) driving the int8 model;
- **verified in QEMU**: the actual ESP32-S3 firmware image booted in
  Espressif's QEMU;
- **estimate**: needs real hardware. The firmware is already instrumented to
  measure these; see [docs/HARDWARE.md](docs/HARDWARE.md).

### Accuracy: official Speech Commands v0.02 test set (4,890 clips, 12 classes), measured

| Model | Accuracy |
|-------|----------|
| float32 | 93.95 % |
| **int8 (deployed)** | **94.01 %** |

Per class (int8): yes 96.2, no 94.6, up 92.2, down 92.4, left 96.8, right 96.0,
on 91.9, off 92.8, stop 95.4, go 92.0, silence 100, unknown 87.8 (%).
Evaluated through the device path: C MFCC → C int8 quantizer → TFLite
reference kernels.

### Model

| | |
|---|---|
| int8 flatbuffer | 45,992 B (flash) |
| parameters / MACs per inference | 24.3 K / 2.66 M |
| kernels | 5 (CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, FULLY_CONNECTED, SOFTMAX) |
| TFLM arena | 35,516 B used of 40 KB (verified in QEMU) |

### Streaming, measured

Each scenario is a 10-minute replay through the firmware's streaming engine:
a background track plus test-set utterances (keywords and unknown words)
dropped in at random gaps.

| Scenario | CNN invocations skipped by VAD | Hit rate, VAD on / off | False accepts/h, VAD on / off |
|----------|-----------------------------|------------------------|-------------------------------|
| quiet room, a command every ~10 s | **85.7 %** | 94.0 % / 94.0 % | 12 / 18 |
| quiet room, a command every ~5 s  | **80.2 %** | 91.2 % / 92.3 % | 6 / 66 |
| running tap (−40 dBFS), every ~10 s | **90.4 %** | 90.0 % / 94.0 % | 12 / 24 |

**Algorithmic latency** runs from the end of the spoken word to the hop on
which the engine fires. In the quiet room: median **−161 ms**, p90 −65 ms,
max +6 ms. Negative means the keyword was recognized before the word ended.
On the device, the compute time below adds on top.

### On the ESP32-S3

| | Value | Status |
|---|---|---|
| firmware image | 756 KB, 62 % of a 1.875 MB OTA slot free | measured (`idf.py size`) |
| static internal RAM | 155 KB of 341 KB DIRAM | measured (`idf.py size`) |
| hot DSP loops in IRAM | `ew_rfft_power`, `ew_mfcc_compute`, `ew_vad_process`, ring read/write at `0x4037xxxx` | verified (ELF symbols) |
| golden-vector inference on Xtensa + ESP-NN | bit-identical to the PC reference (max \|Δq\| = 0) | verified in QEMU |
| OTA rollback after a failed self-test | rolls back to the previous slot, including when otadata has no record of it | verified in QEMU |
| `Invoke()` time | 14–22 ms at 240 MHz | estimate, from ESP-NN's published S3 kernel cycle counts (`training/estimate_latency.py`) |
| keyword → GPIO | < 100 ms budget: algorithmic part measured above + compute | partly estimate |
| CPU-load saving from the VAD | ≈ the skipped-inference share, minus the always-on front-end | to be measured: `timing.cpu_pct` with `{"vad":true}` vs `{"vad":false}` |

### Verification

* `ctest`: unit tests for the ring buffer, FFT, MFCC, VAD and streaming
  engine under ASan/UBSan, plus a 4 M-sample two-thread ring-buffer stress test
  under ThreadSanitizer.
* `training/test_frontend.py`: C MFCC against an independent NumPy
  implementation (max relative error 1.1e-4 on 489 speech clips).
* `test_streaming_matches_offline_features`: the streaming path hands the
  model byte-identical input to offline extraction.
* `firmware/tools/qemu_test.py boot | rollback`: the real firmware image on
  the emulated ESP32-S3.
* CI builds the firmware for the ESP32-S3 on every push.

## Repository layout

```
firmware/                         ESP-IDF project (target esp32s3)
  main/                           app: tasks, I2S, TFLM wrapper, MQTT, OTA, health
    model/                        generated: model_data.cc, model_settings.c
  components/edgewake_core/       portable C: ring buffer, FFT, MFCC, VAD, streaming KWS engine
host/                             host build of the core: unit tests + ctypes library
training/                         data pipeline, DS-CNN, int8 export, streaming evaluation
  results/                        metrics.json, stream_eval.json, edgewake_kws_int8.tflite
docs/ARCHITECTURE.md              tasks, memory, front-end, VAD, latency budget, OTA, watchdogs
docs/HARDWARE.md                  wiring, AWS setup, on-device measurement procedures
```

## Build

**Firmware** (ESP-IDF v5.5):

```bash
cd firmware
idf.py set-target esp32s3
idf.py menuconfig          # EdgeWake → Wi-Fi, AWS IoT endpoint, pins, thresholds
idf.py build flash monitor
python tools/qemu_test.py boot            # no board needed: boots the image in QEMU
```

Without AWS certificates in `firmware/main/certs/` the firmware builds and
runs offline (keyword actions only). See [docs/HARDWARE.md](docs/HARDWARE.md).

**Host tests** (ASan/UBSan, plus a TSan stress test for the ring buffer):

```bash
cmake -S host -B host/build && cmake --build host/build -j
ctest --test-dir host/build --output-on-failure
python training/test_frontend.py          # C MFCC vs NumPy reference
```

**Train and export the model** (Speech Commands v0.02 and its official test set):

```bash
pip install -r training/requirements.txt
python training/train.py  --data <speech_commands_v0.02> --out training/runs/dscnn
python training/export.py --run training/runs/dscnn --data <speech_commands_v0.02> \
                          --test <speech_commands_test_set_v0.02>
python training/stream_eval.py --data <speech_commands_v0.02> --test <speech_commands_test_set_v0.02>
```

`export.py` regenerates `firmware/main/model/*`, so a retrained model is
rebuilt into the firmware without manual steps.

## License

GPL-3.0, see [LICENSE](LICENSE).
