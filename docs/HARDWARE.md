# Hardware bring-up and on-device measurement

## Bill of materials

| Part | Notes |
|------|-------|
| ESP32-S3-DevKitC-1 (any N8/N16 variant) | 4 MB of flash is enough; PSRAM unused |
| INMP441 I2S MEMS microphone breakout | 24-bit, −26 dBFS sensitivity, 61 dB SNR |
| LED + 330 Ω resistor | "on" / "off" |
| 3.3 V relay module or a second LED | "go" / "stop" |
| 8-channel logic analyzer (24 MS/s class) + PulseView/sigrok or Saleae Logic | timing |

## Wiring (defaults, all changeable in `idf.py menuconfig → EdgeWake`)

| INMP441 | ESP32-S3 |
|---------|----------|
| VDD | 3V3 |
| GND | GND |
| L/R | GND (left slot) |
| SCK | GPIO 5 (BCLK, 1.024 MHz = 16 kHz × 64) |
| WS  | GPIO 6 (16 kHz frame clock) |
| SD  | GPIO 4 |

| Function | GPIO |
|----------|------|
| LED ("on"/"off") | 2 |
| Aux / relay ("go"/"stop") | 1 |
| Probe AUDIO | 15 |
| Probe DSP | 16 |
| Probe INFER | 17 |
| Probe ACTION | 18 |

Avoid the strapping pins (0, 3, 45, 46), the USB pins (19, 20) and the
flash/PSRAM pins (26–37) when re-mapping.

## Build and flash

```bash
. $IDF_PATH/export.sh            # ESP-IDF v5.5
cd firmware
idf.py set-target esp32s3
idf.py menuconfig                # EdgeWake → Wi-Fi SSID/password, AWS endpoint, thing name
idf.py build flash monitor
```

Expected boot log (offline, no Wi-Fi configured). These lines are taken from
the same image booted in QEMU (`python tools/qemu_test.py boot`):

```
main: EdgeWake 1.0.0 (ESP-IDF v5.5.5)
ota: running ota_0 @ 0x020000, fw 1.0.0, state valid
infer: model 45992 B, arena used 35516 / 40960 B, in q=(0.04952,10) out q=(0.00391,-128)
main: golden vector: got "on", expected "on", max |dq| = 0
kws: running on core 1
audio: I2S @ 16000 Hz, 4 x 320-sample DMA buffers, ring 8192 samples
```

## AWS IoT Core

1. IoT Core → Manage → Things → Create thing `edgewake-01`, auto-generate a
   certificate, download the device certificate, the private key and Amazon
   Root CA 1 into `firmware/main/certs/` (names in `certs/README.md`).
2. Attach a policy (replace region/account):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": "iot:Connect",
     "Resource": "arn:aws:iot:REGION:ACCOUNT:client/edgewake-01"},
    {"Effect": "Allow", "Action": ["iot:Publish", "iot:Receive"],
     "Resource": "arn:aws:iot:REGION:ACCOUNT:topic/edgewake/edgewake-01/*"},
    {"Effect": "Allow", "Action": "iot:Subscribe",
     "Resource": "arn:aws:iot:REGION:ACCOUNT:topicfilter/edgewake/edgewake-01/*"}
  ]
}
```

3. Set the endpoint (IoT Core → Settings) in menuconfig, rebuild, flash.
4. In the AWS MQTT test client, subscribe to `edgewake/#`. Commands go to
   `edgewake/edgewake-01/cmd`:
   * `{"vad": false}` / `{"vad": true}`: turn the inference gate off or on
   * `{"threshold": 0.85}`: change the detection threshold
   * `{"ota_url": "https://…/edgewake.bin"}`: start an OTA update

## Measurements

Each of the README's on-device numbers comes from one of these procedures.

**Inference time.** Width of the INFER pulse (min / mean / max over ≥ 200
pulses), cross-checked with `timing.infer_us_avg` / `infer_us_max` in health
telemetry.

**Front-end cost.** Width of the DSP pulse on hops where INFER stays low
(VAD + MFCC + feature quantization only).

**CPU load and VAD saving.** Leave the board in a quiet room for 5 minutes with
`{"vad": true}` and read `timing.cpu_pct` from health telemetry (busy time of
the KWS task over 5 s windows). Repeat with `{"vad": false}`. Saving =
1 − on/off. Run the same session with spoken commands at a known rate to
compare with the host replay in `training/results/stream_eval.json`.

**Keyword-to-action latency.** End-to-end = algorithmic latency + compute:
* algorithmic (end of word → end of the hop that fires) is measured by host
  replay (`stream_eval.json → algorithmic_latency_ms`);
* compute (hop available → GPIO) is the time from the AUDIO probe falling edge
  of the triggering hop to the ACTION rising edge, also logged per detection
  as `hop_to_action_us`.

For a direct measurement, play a keyword WAV from a speaker, record the speaker
signal on an analog-capable analyzer channel (or through a comparator into a
digital one), and measure from the end of the word envelope to the LED edge.

**Audio cadence and jitter.** Period of the AUDIO probe rising edges (should
be 20.000 ms), with and without Wi-Fi traffic. WS on GPIO 6 should read
16.000 kHz.

## OTA rollback demo

1. Flash 1.0.0 over USB.
2. Set `PROJECT_VER` to `1.0.1` in `firmware/CMakeLists.txt`, enable
   `EdgeWake → Actions and debug → Force the post-OTA self-test to fail`, build,
   and host `build/edgewake.bin` on any HTTPS server (an S3 pre-signed URL
   works).
3. Publish `{"ota_url": "<url>"}`. The device downloads into `ota_1`,
   reboots, logs `self-test FAILED (forced …): rolling back`, reboots into
   `ota_0`, and health telemetry shows `"part":"ota_0"`.
4. Rebuild 1.0.1 without the option. The same OTA now ends with
   `self-test passed: marking image valid` and `"part":"ota_1"`.
