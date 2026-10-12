#include "ew_fft.h"

#include <math.h>

#include "ew_port.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* log2(N/2), computed at compile time from the config. */
#if EW_FFT_HALF == 256
#define EW_FFT_HALF_LOG2 8
#elif EW_FFT_HALF == 128
#define EW_FFT_HALF_LOG2 7
#elif EW_FFT_HALF == 512
#define EW_FFT_HALF_LOG2 9
#else
#error "EW_FFT_LEN must be 256, 512 or 1024"
#endif

void ew_rfft_init(ew_rfft_t *f)
{
    for (int k = 0; k < EW_FFT_HALF; k++) {
        /* Twiddles are computed in double then rounded once to float. */
        const double a = 2.0 * M_PI * (double)k / (double)EW_FFT_LEN;
        f->tw_re[k] = (float)cos(a);
        f->tw_im[k] = (float)-sin(a);
    }
    for (int i = 0; i < EW_FFT_HALF; i++) {
        unsigned r = 0;
        for (int b = 0; b < EW_FFT_HALF_LOG2; b++) {
            r |= ((unsigned)(i >> b) & 1u) << (EW_FFT_HALF_LOG2 - 1 - b);
        }
        f->bitrev[i] = (uint16_t)r;
    }
}

/* In-place iterative radix-2 decimation-in-time FFT of EW_FFT_HALF points. */
static EW_IRAM void complex_fft_half(ew_rfft_t *f)
{
    float *x = f->work;

    for (int i = 0; i < EW_FFT_HALF; i++) {
        const int j = f->bitrev[i];
        if (j > i) {
            float t = x[2 * i];
            x[2 * i] = x[2 * j];
            x[2 * j] = t;
            t = x[2 * i + 1];
            x[2 * i + 1] = x[2 * j + 1];
            x[2 * j + 1] = t;
        }
    }

    for (int size = 2; size <= EW_FFT_HALF; size <<= 1) {
        const int half = size >> 1;
        /* W_size^k == W_N^(k * N/size): stride through the N-point table. */
        const int stride = EW_FFT_LEN / size;
        for (int k = 0; k < half; k++) {
            const float wr = f->tw_re[k * stride];
            const float wi = f->tw_im[k * stride];
            for (int start = k; start < EW_FFT_HALF; start += size) {
                const int a = 2 * start;
                const int b = 2 * (start + half);
                const float tr = wr * x[b] - wi * x[b + 1];
                const float ti = wr * x[b + 1] + wi * x[b];
                x[b] = x[a] - tr;
                x[b + 1] = x[a + 1] - ti;
                x[a] += tr;
                x[a + 1] += ti;
            }
        }
    }
}

EW_IRAM void ew_rfft_power(ew_rfft_t *f, const float *in, float *power_out)
{
    float *z = f->work;

    /* Pack: z[n] = x[2n] + j*x[2n+1]. */
    for (int n = 0; n < EW_FFT_HALF; n++) {
        z[2 * n] = in[2 * n];
        z[2 * n + 1] = in[2 * n + 1];
    }

    complex_fft_half(f);

    /* DC and Nyquist come straight out of Z[0]. */
    const float z0r = z[0];
    const float z0i = z[1];
    power_out[0] = (z0r + z0i) * (z0r + z0i);
    power_out[EW_FFT_HALF] = (z0r - z0i) * (z0r - z0i);

    /* Split: E[k] = (Z[k] + conj(Z[M-k])) / 2   (spectrum of even samples)
     *        O[k] = (Z[k] - conj(Z[M-k])) / 2j  (spectrum of odd samples)
     *        X[k] = E[k] + W_N^k * O[k]                                   */
    for (int k = 1; k < EW_FFT_HALF; k++) {
        const int m = EW_FFT_HALF - k;
        const float ar = z[2 * k], ai = z[2 * k + 1];
        const float br = z[2 * m], bi = -z[2 * m + 1]; /* conj(Z[M-k]) */

        const float er = 0.5f * (ar + br);
        const float ei = 0.5f * (ai + bi);
        /* (a - b) / 2j  ==  ((ai - bi) - j(ar - br)) / 2 */
        const float or_ = 0.5f * (ai - bi);
        const float oi = -0.5f * (ar - br);

        const float wr = f->tw_re[k], wi = f->tw_im[k];
        const float xr = er + (wr * or_ - wi * oi);
        const float xi = ei + (wr * oi + wi * or_);
        power_out[k] = xr * xr + xi * xi;
    }
}
