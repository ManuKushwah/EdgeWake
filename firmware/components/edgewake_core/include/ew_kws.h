/*
 * ew_kws.h - streaming keyword-spotting engine (platform independent).
 *
 * Feed it one 20 ms hop of PCM at a time. Every hop it:
 *   1. slides a 512-sample analysis window and computes one MFCC frame,
 *   2. quantizes the frame straight to int8 and stores it in a 49-row
 *      circular feature buffer (so only 10 new values are quantized per hop),
 *   3. updates the VAD gate,
 *   4. every `infer_every_hops` hops, if the gate is open, linearizes the
 *      feature ring into the model input and calls the injected `infer`
 *      callback (TFLite Micro on the device, the Python interpreter on host),
 *   5. averages the last few posteriors and reports a detection when a
 *      keyword clears the threshold outside the refractory period.
 *
 * MFCCs run even while the gate is closed: they are cheap, and it means the
 * 1 s feature window is already full of fresh context the moment speech
 * starts. Only the expensive part, the CNN, is gated.
 *
 * Inference is injected as a function pointer so this file has no TFLM or
 * ESP-IDF dependency and the exact same logic is exercised by host tests.
 */
#ifndef EW_KWS_H
#define EW_KWS_H

#include <stdbool.h>
#include <stdint.h>

#include "ew_config.h"
#include "ew_mfcc.h"
#include "ew_vad.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EW_KWS_MAX_SMOOTH 8

/* Runs the model: input is EW_INPUT_SIZE int8 values (49 x 10, oldest frame
 * first); writes EW_NUM_CLASSES int8 scores. Returns 0 on success. */
typedef int (*ew_infer_fn)(void *ctx, const int8_t *input, int8_t *scores);

typedef struct {
    /* Quantization parameters of the model's input and output tensors. */
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;

    /* Per-coefficient normalisation learned in training (EW_NUM_MFCC each).
     * Folded into the quantizer, so it costs nothing at run time. */
    const float *feat_mean;
    const float *feat_inv_std;

    uint32_t infer_every_hops;   /* 2 -> one inference per 40 ms */
    uint32_t smooth_window;      /* posteriors averaged, 1..EW_KWS_MAX_SMOOTH */
    uint32_t min_history;        /* posteriors needed before a detection may fire
                                  * (1..smooth_window); stops one transient
                                  * inference right after a reset from firing */
    float detect_threshold;      /* on the averaged probability, 0..1 */
    uint32_t refractory_hops;    /* no re-trigger for this long after a hit */
    int silence_index;           /* classes that are never "keywords" */
    int unknown_index;

    bool vad_enabled;
    ew_vad_config_t vad;

    ew_infer_fn infer;
    void *infer_ctx;
} ew_kws_config_t;

typedef struct {
    bool vad_active;     /* gate state after this hop */
    bool inferred;       /* the CNN ran on this hop */
    bool detected;       /* a keyword fired on this hop */
    int top_class;       /* argmax of the smoothed posteriors (if inferred) */
    float top_score;     /* its smoothed probability */
} ew_kws_result_t;

typedef struct {
    uint32_t hops;
    uint32_t infer_slots;     /* hops on which an inference was due */
    uint32_t inferences;      /* slots where the CNN actually ran */
    uint32_t skipped_by_vad;  /* slots skipped because the gate was closed */
    uint32_t skipped_overload;/* slots skipped because the caller was behind */
    uint32_t detections;
    uint32_t infer_errors;
} ew_kws_stats_t;

typedef struct {
    ew_kws_config_t cfg;
    ew_mfcc_t mfcc;
    ew_vad_t vad;

    int16_t window[EW_FRAME_LEN];
    uint32_t window_fill;                          /* samples, saturates */

    int8_t feat_q[EW_NUM_FRAMES][EW_NUM_MFCC];     /* circular, quantized */
    uint32_t feat_write;                           /* next row to write */
    uint32_t feat_count;                           /* valid rows, saturates */
    float last_mfcc[EW_NUM_MFCC];
    float qmul[EW_NUM_MFCC];
    float qadd[EW_NUM_MFCC];

    int8_t input[EW_INPUT_SIZE];
    int8_t scores[EW_NUM_CLASSES];
    float hist[EW_KWS_MAX_SMOOTH][EW_NUM_CLASSES];
    uint32_t hist_pos;
    uint32_t hist_count;
    uint32_t refractory_left;

    ew_kws_stats_t stats;
} ew_kws_t;

/* Returns 0 on success, negative on invalid configuration. */
int ew_kws_init(ew_kws_t *k, const ew_kws_config_t *cfg);

/* hop: exactly EW_FRAME_HOP samples. `out` may be NULL. */
void ew_kws_process_hop(ew_kws_t *k, const int16_t *hop, ew_kws_result_t *out);

/* Same, but lets the caller shed load: with allow_infer = false the front-end
 * (window, MFCC, VAD) still advances so no context is lost, while a due
 * inference is skipped and counted in stats.skipped_overload. Used when the
 * audio backlog shows the task is falling behind real time. */
void ew_kws_process_hop_ex(ew_kws_t *k, const int16_t *hop, ew_kws_result_t *out, bool allow_infer);

/* Quantizes one MFCC frame with the folded normalisation (exposed for tests
 * and for the OTA self-test). */
void ew_kws_quantize_frame(const ew_kws_t *k, const float *mfcc, int8_t *q);

#ifdef __cplusplus
}
#endif

#endif /* EW_KWS_H */
