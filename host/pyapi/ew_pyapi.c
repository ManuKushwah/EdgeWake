/*
 * ew_pyapi.c - flat C ABI over the core for Python (ctypes). Host only.
 *
 * Training calls ew_py_mfcc_batch() so the model learns from features made by
 * the *same* C code that runs on the ESP32-S3: no train/serve skew.
 * Evaluation calls ew_py_kws_*() to replay long audio streams through the
 * exact firmware streaming logic (VAD gate, smoothing, refractory), with the
 * int8 TFLite model plugged in as the inference callback.
 */
#include <string.h>

#include "ew_config.h"
#include "ew_kws.h"
#include "ew_mfcc.h"

static ew_mfcc_t g_mfcc;
static int g_mfcc_ready;

static ew_kws_t g_kws;
static float g_mean[EW_NUM_MFCC];
static float g_inv_std[EW_NUM_MFCC];

/* Writes the compiled-in front-end geometry so Python can assert it matches. */
void ew_py_config(int *out)
{
    out[0] = EW_SAMPLE_RATE_HZ;
    out[1] = EW_FRAME_LEN;
    out[2] = EW_FRAME_HOP;
    out[3] = EW_FFT_LEN;
    out[4] = EW_MEL_BANDS;
    out[5] = EW_NUM_MFCC;
    out[6] = EW_NUM_FRAMES;
    out[7] = EW_NUM_CLASSES;
}

int ew_py_num_frames(int n_samples)
{
    return n_samples < EW_FRAME_LEN ? 0 : (n_samples - EW_FRAME_LEN) / EW_FRAME_HOP + 1;
}

/* Offline MFCCs for `n_clips` clips of `clip_len` samples each, laid out
 * contiguously. out: n_clips * frames * EW_NUM_MFCC floats. Returns frames
 * per clip, or -1 on error. */
int ew_py_mfcc_batch(const int16_t *pcm, int n_clips, int clip_len, float *out)
{
    if (!g_mfcc_ready) {
        if (ew_mfcc_init(&g_mfcc) != 0) {
            return -1;
        }
        g_mfcc_ready = 1;
    }
    const int frames = ew_py_num_frames(clip_len);
    for (int c = 0; c < n_clips; c++) {
        const int16_t *clip = pcm + (long)c * clip_len;
        float *dst = out + (long)c * frames * EW_NUM_MFCC;
        for (int f = 0; f < frames; f++) {
            ew_mfcc_compute(&g_mfcc, clip + f * EW_FRAME_HOP, dst + f * EW_NUM_MFCC);
        }
    }
    return frames;
}

int ew_py_kws_init(float in_scale, int in_zp, float out_scale, int out_zp,
                   const float *mean, const float *inv_std,
                   int infer_every_hops, int smooth_window, float threshold,
                   int refractory_hops, int silence_index, int unknown_index,
                   int vad_enabled, ew_infer_fn infer)
{
    memcpy(g_mean, mean, sizeof(g_mean));
    memcpy(g_inv_std, inv_std, sizeof(g_inv_std));

    ew_kws_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.input_scale = in_scale;
    cfg.input_zero_point = in_zp;
    cfg.output_scale = out_scale;
    cfg.output_zero_point = out_zp;
    cfg.feat_mean = g_mean;
    cfg.feat_inv_std = g_inv_std;
    cfg.infer_every_hops = (uint32_t)infer_every_hops;
    cfg.smooth_window = (uint32_t)smooth_window;
    cfg.detect_threshold = threshold;
    cfg.refractory_hops = (uint32_t)refractory_hops;
    cfg.silence_index = silence_index;
    cfg.unknown_index = unknown_index;
    cfg.vad_enabled = vad_enabled != 0;
    ew_vad_default_config(&cfg.vad);
    cfg.infer = infer;
    cfg.infer_ctx = NULL;
    return ew_kws_init(&g_kws, &cfg);
}

/* Quantizes offline MFCC frames exactly like the device does. */
void ew_py_kws_quantize(const float *mfcc, int n_frames, int8_t *out)
{
    for (int f = 0; f < n_frames; f++) {
        ew_kws_quantize_frame(&g_kws, mfcc + f * EW_NUM_MFCC, out + f * EW_NUM_MFCC);
    }
}

/* Streams `n_samples` (a multiple of EW_FRAME_HOP) through the engine.
 * Per-hop outputs: gate state, whether the CNN ran, detected class (-1 if
 * none) and the smoothed top score. Returns the number of hops processed. */
int ew_py_kws_run(const int16_t *pcm, int n_samples, uint8_t *vad_out, uint8_t *ran_out,
                  int16_t *detect_out, float *score_out)
{
    const int hops = n_samples / EW_FRAME_HOP;
    for (int h = 0; h < hops; h++) {
        ew_kws_result_t r;
        ew_kws_process_hop(&g_kws, pcm + h * EW_FRAME_HOP, &r);
        vad_out[h] = r.vad_active;
        ran_out[h] = r.inferred;
        detect_out[h] = (int16_t)(r.detected ? r.top_class : -1);
        score_out[h] = r.inferred ? r.top_score : 0.0f;
    }
    return hops;
}

void ew_py_kws_stats(uint32_t *out)
{
    out[0] = g_kws.stats.hops;
    out[1] = g_kws.stats.infer_slots;
    out[2] = g_kws.stats.inferences;
    out[3] = g_kws.stats.skipped_by_vad;
    out[4] = g_kws.stats.detections;
    out[5] = g_kws.stats.infer_errors;
}
