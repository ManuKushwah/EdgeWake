"""ctypes bindings to the device's C front-end (host/build/libedgewake.so).

Training features are produced by the same ew_mfcc.c that runs on the
ESP32-S3, and evaluation quantizes them with the same ew_kws.c code, so what
the model sees in training is bit-for-bit what it sees on the device.
"""
import ctypes
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LIB = REPO / "host" / "build" / "libedgewake.so"

# Inference callback signature: int (*)(void *ctx, const int8_t *in, int8_t *out)
INFER_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8))

_f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_i16p = np.ctypeslib.ndpointer(np.int16, flags="C_CONTIGUOUS")
_i8p = np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS")
_u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
_u32p = np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS")


class Frontend:
    def __init__(self, lib_path=None):
        path = lib_path or os.environ.get("EDGEWAKE_LIB", DEFAULT_LIB)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found - build it first: cmake -S host -B host/build && cmake --build host/build")
        self.lib = lib = ctypes.CDLL(str(path))

        lib.ew_py_config.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cfg = (ctypes.c_int * 8)()
        lib.ew_py_config(cfg)
        (self.sample_rate, self.frame_len, self.hop, self.fft_len,
         self.mel_bands, self.num_mfcc, self.num_frames, self.num_classes) = list(cfg)

        lib.ew_py_mfcc_batch.argtypes = [_i16p, ctypes.c_int, ctypes.c_int, _f32p]
        lib.ew_py_mfcc_batch.restype = ctypes.c_int
        lib.ew_py_num_frames.argtypes = [ctypes.c_int]
        lib.ew_py_num_frames.restype = ctypes.c_int
        lib.ew_py_kws_init.argtypes = [
            ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_int, _f32p, _f32p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, INFER_FN]
        lib.ew_py_kws_init.restype = ctypes.c_int
        lib.ew_py_kws_quantize.argtypes = [_f32p, ctypes.c_int, _i8p]
        lib.ew_py_kws_run.argtypes = [_i16p, ctypes.c_int, _u8p, _u8p, _i16p, _f32p]
        lib.ew_py_kws_run.restype = ctypes.c_int
        lib.ew_py_kws_stats.argtypes = [_u32p]
        self._callback = None  # keep a reference so ctypes doesn't free it

    def mfcc(self, pcm):
        """pcm: (N, L) int16 -> (N, frames, num_mfcc) float32. Releases the GIL."""
        pcm = np.ascontiguousarray(pcm, dtype=np.int16)
        if pcm.ndim == 1:
            pcm = pcm[None]
        n, length = pcm.shape
        frames = self.lib.ew_py_num_frames(length)
        out = np.empty((n, frames, self.num_mfcc), np.float32)
        if self.lib.ew_py_mfcc_batch(pcm, n, length, out) != frames:
            raise RuntimeError("ew_py_mfcc_batch failed")
        return out

    def kws_init(self, in_scale, in_zp, out_scale, out_zp, mean, inv_std, infer=None,
                 infer_every_hops=2, smooth_window=3, min_history=1, threshold=0.8, refractory_hops=50,
                 silence_index=0, unknown_index=1, vad_enabled=True):
        """Configures the streaming engine. `infer(input_int8[490]) -> scores_int8[12]`."""
        num_classes = self.num_classes

        def _cb(_ctx, in_ptr, out_ptr):
            try:
                x = np.ctypeslib.as_array(in_ptr, shape=(self.num_frames * self.num_mfcc,))
                scores = infer(x.copy()) if infer else np.zeros(num_classes, np.int8)
                ctypes.memmove(out_ptr, np.ascontiguousarray(scores, np.int8).ctypes.data, num_classes)
                return 0
            except Exception as exc:  # never let an exception unwind through C
                print("infer callback failed:", exc)
                return -1

        self._callback = INFER_FN(_cb)
        rc = self.lib.ew_py_kws_init(
            float(in_scale), int(in_zp), float(out_scale), int(out_zp),
            np.ascontiguousarray(mean, np.float32), np.ascontiguousarray(inv_std, np.float32),
            infer_every_hops, smooth_window, min_history, float(threshold), refractory_hops,
            silence_index, unknown_index, int(vad_enabled), self._callback)
        if rc != 0:
            raise RuntimeError(f"ew_py_kws_init failed ({rc})")

    def quantize(self, mfcc):
        """(N, frames, num_mfcc) float32 -> int8, exactly as the device does it.
        Requires kws_init() to have set the quantization parameters."""
        mfcc = np.ascontiguousarray(mfcc, np.float32)
        out = np.empty(mfcc.shape, np.int8)
        flat_in = mfcc.reshape(-1, self.num_mfcc)
        self.lib.ew_py_kws_quantize(flat_in, flat_in.shape[0], out.reshape(-1, self.num_mfcc))
        return out

    def kws_run(self, pcm):
        """Streams int16 audio through the firmware engine; returns per-hop arrays."""
        pcm = np.ascontiguousarray(pcm, np.int16)
        hops = len(pcm) // self.hop
        vad = np.zeros(hops, np.uint8)
        ran = np.zeros(hops, np.uint8)
        det = np.zeros(hops, np.int16)
        score = np.zeros(hops, np.float32)
        self.lib.ew_py_kws_run(pcm[: hops * self.hop], hops * self.hop, vad, ran, det, score)
        return vad.astype(bool), ran.astype(bool), det, score

    def kws_stats(self):
        out = np.zeros(6, np.uint32)
        self.lib.ew_py_kws_stats(out)
        keys = ["hops", "infer_slots", "inferences", "skipped_by_vad", "detections", "infer_errors"]
        return dict(zip(keys, (int(v) for v in out)))
