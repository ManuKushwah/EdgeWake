"""Boot the built firmware in Espressif's QEMU (machine esp32s3) and check it.

    python tools/qemu_test.py boot                      # model loads, golden vector matches
    python tools/qemu_test.py rollback --new <app.bin>  # OTA image fails self-test -> rollback
    python tools/qemu_test.py rollback --new <app.bin> --no-previous-record

Run from firmware/ after `idf.py build`, inside an ESP-IDF v5.5 environment with
QEMU installed (`python $IDF_PATH/tools/idf_tools.py install qemu-xtensa`).

QEMU emulates the CPU, flash, eFuses and timers but not the I2S microphone,
so the self-test always fails at "microphone silent". That makes it a good
harness for the rollback path. `--new` is a second app image (build it with a
bumped PROJECT_VER). The script writes it into ota_1 and writes otadata the
way esp_ota_set_boot_partition() leaves it after a download.
"""
import argparse
import binascii
import os
import re
import shutil
import struct
import subprocess
import sys

ESP_OTA_IMG_NEW, ESP_OTA_IMG_VALID = 0x0, 0x2
OTADATA, OTA_1 = 0xF000, 0x200000          # from partitions.csv


def otadata_entry(seq, state):
    e = struct.pack("<I", seq) + b"\xff" * 20 + struct.pack("<I", state)
    return e + struct.pack("<I", binascii.crc32(struct.pack("<I", seq), 0xFFFFFFFF) & 0xFFFFFFFF)


def make_flash(build, new_app=None, previous_record=True):
    build = os.path.abspath(build)
    flash = os.path.join(build, "qemu_flash.bin")
    subprocess.check_call([sys.executable, "-m", "esptool", "--chip=esp32s3", "merge_bin",
                           f"--output={flash}", "--fill-flash-size=4MB", "@flash_args"],
                          cwd=build, stdout=subprocess.DEVNULL)
    if new_app:
        img = bytearray(open(flash, "rb").read())
        app = open(new_app, "rb").read()
        img[OTA_1:OTA_1 + len(app)] = app
        img[OTADATA:OTADATA + 0x2000] = b"\xff" * 0x2000
        if previous_record:  # written by the bootloader on the device's first boot
            img[OTADATA:OTADATA + 32] = otadata_entry(1, ESP_OTA_IMG_VALID)
        img[OTADATA + 0x1000:OTADATA + 0x1000 + 32] = otadata_entry(2, ESP_OTA_IMG_NEW)
        open(flash, "wb").write(img)
    return flash


def make_efuse(build):
    sys.path.insert(0, os.path.join(os.environ["IDF_PATH"], "tools"))
    from idf_py_actions.qemu_ext import QEMU_TARGETS
    path = os.path.join(build, "qemu_efuse.bin")
    open(path, "wb").write(QEMU_TARGETS["esp32s3"].default_efuse)
    return path


def run_qemu(flash, efuse, seconds):
    qemu = shutil.which("qemu-system-xtensa")
    if not qemu:
        sys.exit("qemu-system-xtensa not found (idf_tools.py install qemu-xtensa)")
    cmd = [qemu, "-M", "esp32s3", "-m", "32M", "-nographic",
           "-drive", f"file={flash},if=mtd,format=raw",
           "-drive", f"file={efuse},if=none,format=raw,id=efuse",
           "-global", "driver=nvram.esp32s3.efuse,property=drive,value=efuse",
           "-global", "driver=timer.esp32s3.timg,property=wdt_disable,value=true",
           "-serial", "mon:stdio"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=seconds, stdin=subprocess.DEVNULL).stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout or b""
    return re.sub(r"\x1b\[[0-9;]*m", "", out.decode(errors="replace"))


def expect_in_order(log, patterns):
    pos, ok = 0, True
    for p in patterns:
        m = re.compile(p).search(log, pos)
        print(f"  [{'ok' if m else 'MISSING'}] {p}")
        if m:
            pos = m.end()
        else:
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["boot", "rollback"])
    ap.add_argument("--build", default="build")
    ap.add_argument("--new", help="second app image for the rollback test")
    ap.add_argument("--no-previous-record", action="store_true")
    ap.add_argument("--seconds", type=int, default=15)
    args = ap.parse_args()

    if args.mode == "rollback" and not args.new:
        ap.error("rollback needs --new <app.bin>")
    flash = make_flash(args.build, args.new, not args.no_previous_record)
    log = run_qemu(flash, make_efuse(args.build), args.seconds)
    lines = [l for l in log.splitlines() if re.search(r" (main|ota|infer|kws|esp_ota_ops): ", l)]
    print("\n".join(lines[:20]))

    if args.mode == "boot":
        ok = expect_in_order(log, [
            r"infer: model \d+ B, arena used \d+ / \d+ B",
            # model scores, on-device MFCC + quantizer, and end-to-end decision
            r'kws: self-test: model "(\w+)" \(expected "\1"\) max \|dq\| = [01]; '
            r'front-end \d+/490 features differ \(max [01]\); end-to-end "\1"',
            r"kws: running on core 1",
        ])
    else:
        ok = expect_in_order(log, [
            r"ota: running ota_1 .* state pending_verify",
            r"ota: self-test FAILED",
            r"main: EdgeWake \S+",          # the next boot ...
            r"ota: running ota_0 ",         # ... is back on the previous slot
        ])
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
