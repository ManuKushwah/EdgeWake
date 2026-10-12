/*
 * ew_fft.h - real-input FFT (N = EW_FFT_LEN) returning the one-sided power
 * spectrum |X[k]|^2, k = 0..N/2.
 *
 * Trick used: an N-point *real* FFT is computed with one N/2-point *complex*
 * FFT. Even samples go in the real part, odd samples in the imaginary part;
 * a final "split" pass separates the two half-length spectra and combines
 * them with one extra twiddle. That halves the work versus a naive complex
 * FFT of a real signal.
 *
 * All tables live inside the context struct (no heap). One twiddle table of
 * N/2 entries serves both stages because W_{N/2}^k == W_N^{2k}.
 */
#ifndef EW_FFT_H
#define EW_FFT_H

#include <stdint.h>

#include "ew_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EW_FFT_HALF (EW_FFT_LEN / 2)

typedef struct {
    float tw_re[EW_FFT_HALF];      /* cos(2*pi*k/N)  */
    float tw_im[EW_FFT_HALF];      /* -sin(2*pi*k/N) */
    uint16_t bitrev[EW_FFT_HALF];  /* bit-reversal permutation for N/2 */
    float work[2 * EW_FFT_HALF];   /* interleaved re/im scratch */
} ew_rfft_t;

void ew_rfft_init(ew_rfft_t *f);

/* in: EW_FFT_LEN real samples. power_out: EW_FFT_BINS values. */
void ew_rfft_power(ew_rfft_t *f, const float *in, float *power_out);

#ifdef __cplusplus
}
#endif

#endif /* EW_FFT_H */
