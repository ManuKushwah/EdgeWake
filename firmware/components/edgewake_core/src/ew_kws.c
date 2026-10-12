#include "ew_kws.h"

#include <math.h>
#include <string.h>

#include "ew_port.h"

int ew_kws_init(ew_kws_t *k, const ew_kws_config_t *cfg)
{
    if (cfg == NULL || cfg->infer == NULL || cfg->input_scale <= 0.0f ||
        cfg->output_scale <= 0.0f || cfg->infer_every_hops == 0 ||
        cfg->smooth_window == 0 || cfg->smooth_window > EW_KWS_MAX_SMOOTH) {
        return -1;
    }
    memset(k, 0, sizeof(*k));
    k->cfg = *cfg;

    if (ew_mfcc_init(&k->mfcc) != 0) {
        return -2;
    }
    ew_vad_init(&k->vad, &cfg->vad);

    /* q = round((x - mean) * inv_std / scale) + zp
     *   = round(x * qmul + qadd)                    with per-coefficient
     * qmul/qadd precomputed here, so normalisation is free at run time. */
    for (int c = 0; c < EW_NUM_MFCC; c++) {
        const float mean = cfg->feat_mean ? cfg->feat_mean[c] : 0.0f;
        const float inv_std = cfg->feat_inv_std ? cfg->feat_inv_std[c] : 1.0f;
        k->qmul[c] = inv_std / cfg->input_scale;
        k->qadd[c] = -mean * inv_std / cfg->input_scale + (float)cfg->input_zero_point;
    }
    return 0;
}

EW_IRAM void ew_kws_quantize_frame(const ew_kws_t *k, const float *mfcc, int8_t *q)
{
    for (int c = 0; c < EW_NUM_MFCC; c++) {
        float v = roundf(mfcc[c] * k->qmul[c] + k->qadd[c]);
        if (v > 127.0f) {
            v = 127.0f;
        } else if (v < -128.0f) {
            v = -128.0f;
        }
        q[c] = (int8_t)v;
    }
}

static void run_inference(ew_kws_t *k, ew_kws_result_t *res)
{
    /* Linearize the feature ring: oldest row first. */
    const uint32_t first_rows = EW_NUM_FRAMES - k->feat_write;
    memcpy(k->input, k->feat_q[k->feat_write], first_rows * EW_NUM_MFCC);
    memcpy(k->input + first_rows * EW_NUM_MFCC, k->feat_q[0], k->feat_write * EW_NUM_MFCC);

    if (k->cfg.infer(k->cfg.infer_ctx, k->input, k->scores) != 0) {
        k->stats.infer_errors++;
        return;
    }
    k->stats.inferences++;
    res->inferred = true;

    /* Dequantize the softmax output into the smoothing history. */
    float *slot = k->hist[k->hist_pos];
    for (int c = 0; c < EW_NUM_CLASSES; c++) {
        slot[c] = (float)(k->scores[c] - k->cfg.output_zero_point) * k->cfg.output_scale;
    }
    k->hist_pos = (k->hist_pos + 1) % k->cfg.smooth_window;
    if (k->hist_count < k->cfg.smooth_window) {
        k->hist_count++;
    }

    /* Average over the history, take the argmax. */
    int best = 0;
    float best_score = -1.0f;
    for (int c = 0; c < EW_NUM_CLASSES; c++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < k->hist_count; h++) {
            acc += k->hist[h][c];
        }
        acc /= (float)k->hist_count;
        if (acc > best_score) {
            best_score = acc;
            best = c;
        }
    }
    res->top_class = best;
    res->top_score = best_score;

    const bool is_keyword = best != k->cfg.silence_index && best != k->cfg.unknown_index;
    if (is_keyword && best_score >= k->cfg.detect_threshold && k->refractory_left == 0) {
        res->detected = true;
        k->stats.detections++;
        k->refractory_left = k->cfg.refractory_hops;
        k->hist_count = 0;  /* don't let this utterance re-trigger later */
        k->hist_pos = 0;
    }
}

void ew_kws_process_hop(ew_kws_t *k, const int16_t *hop, ew_kws_result_t *out)
{
    ew_kws_result_t res = {0};
    res.top_class = -1;

    k->stats.hops++;
    if (k->refractory_left > 0) {
        k->refractory_left--;
    }

    /* 1. Slide the analysis window: keep the newest FRAME_LEN - HOP samples. */
    memmove(k->window, k->window + EW_FRAME_HOP,
            (EW_FRAME_LEN - EW_FRAME_HOP) * sizeof(int16_t));
    memcpy(k->window + (EW_FRAME_LEN - EW_FRAME_HOP), hop, EW_FRAME_HOP * sizeof(int16_t));
    k->window_fill += EW_FRAME_HOP;
    if (k->window_fill > EW_FRAME_LEN) {
        k->window_fill = EW_FRAME_LEN;
    }

    /* 2. One MFCC frame per hop once the window is full. */
    if (k->window_fill == EW_FRAME_LEN) {
        ew_mfcc_compute(&k->mfcc, k->window, k->last_mfcc);
        ew_kws_quantize_frame(k, k->last_mfcc, k->feat_q[k->feat_write]);
        k->feat_write = (k->feat_write + 1) % EW_NUM_FRAMES;
        if (k->feat_count < EW_NUM_FRAMES) {
            k->feat_count++;
        }
    }

    /* 3. VAD on the new hop only. */
    res.vad_active = k->cfg.vad_enabled ? ew_vad_process(&k->vad, hop, EW_FRAME_HOP) : true;

    /* 4./5. Gated, strided inference. */
    if (k->feat_count == EW_NUM_FRAMES && (k->stats.hops % k->cfg.infer_every_hops) == 0) {
        k->stats.infer_slots++;
        if (res.vad_active) {
            run_inference(k, &res);
        } else {
            k->stats.skipped_by_vad++;
            k->hist_count = 0;  /* posteriors from before the silence are stale */
            k->hist_pos = 0;
        }
    }

    if (out != NULL) {
        *out = res;
    }
}
