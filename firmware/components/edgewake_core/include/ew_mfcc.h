/*
 * ew_mfcc.h - MFCC feature extractor for one analysis frame.
 *
 * Pipeline per frame (512 samples, every 20 ms):
 *   int16 PCM -> float [-1,1) * periodic Hann window
 *             -> real FFT -> power spectrum (257 bins)
 *             -> 40 triangular mel filters (sparse, mel-domain triangles)
 *             -> natural log(energy + 1e-6)
 *             -> orthonormal DCT-II, keep the first 10 coefficients
 *
 * Tables (window, mel weights, DCT matrix) are computed once in
 * ew_mfcc_init() into the context struct, so the per-frame path does no
 * trig, no allocation and only 40 logf() calls.
 */
#ifndef EW_MFCC_H
#define EW_MFCC_H

#include <stdint.h>

#include "ew_config.h"
#include "ew_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Upper bound on non-zero filterbank weights: each FFT bin falls into at
 * most two overlapping triangles. */
#define EW_MEL_WEIGHTS_MAX (2 * EW_FFT_BINS)

typedef struct {
    ew_rfft_t fft;
    float window[EW_FRAME_LEN];
    float frame[EW_FFT_LEN];
    float power[EW_FFT_BINS];
    float log_mel[EW_MEL_BANDS];
    /* Sparse filterbank: band b uses weights[offset[b] .. offset[b]+len[b])
     * applied to power[start[b] ..]. */
    uint16_t mel_start[EW_MEL_BANDS];
    uint16_t mel_len[EW_MEL_BANDS];
    uint16_t mel_offset[EW_MEL_BANDS];
    float mel_weights[EW_MEL_WEIGHTS_MAX];
    float dct[EW_NUM_MFCC][EW_MEL_BANDS];
} ew_mfcc_t;

/* Returns 0 on success, -1 if the filterbank geometry is invalid. */
int ew_mfcc_init(ew_mfcc_t *m);

/* samples: EW_FRAME_LEN int16 values. out: EW_NUM_MFCC floats. */
void ew_mfcc_compute(ew_mfcc_t *m, const int16_t *samples, float *out);

/* Exposed for tests / tooling. */
float ew_hz_to_mel(float hz);

#ifdef __cplusplus
}
#endif

#endif /* EW_MFCC_H */
