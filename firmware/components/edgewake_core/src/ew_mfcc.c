#include "ew_mfcc.h"

#include <math.h>

#include "ew_port.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* HTK mel scale. */
float ew_hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static double hz_to_mel_d(double hz)
{
    return 2595.0 * log10(1.0 + hz / 700.0);
}

int ew_mfcc_init(ew_mfcc_t *m)
{
    ew_rfft_init(&m->fft);

    /* Periodic Hann window (the DFT-friendly variant used by tf.signal). */
    for (int n = 0; n < EW_FRAME_LEN; n++) {
        m->window[n] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * n / EW_FRAME_LEN));
    }
    for (int n = EW_FRAME_LEN; n < EW_FFT_LEN; n++) {
        m->frame[n] = 0.0f;  /* zero padding, unused when FRAME_LEN == FFT_LEN */
    }

    /* Mel filterbank: B triangles with centres equally spaced in mel between
     * low and high. Triangle weights are evaluated in the mel domain. */
    const double mel_lo = hz_to_mel_d(EW_MEL_LOW_HZ);
    const double mel_hi = hz_to_mel_d(EW_MEL_HIGH_HZ);
    const double mel_step = (mel_hi - mel_lo) / (EW_MEL_BANDS + 1);
    int used = 0;

    for (int b = 0; b < EW_MEL_BANDS; b++) {
        const double left = mel_lo + b * mel_step;
        const double centre = left + mel_step;
        const double right = centre + mel_step;
        int start = -1;
        int len = 0;

        for (int k = 0; k < EW_FFT_BINS; k++) {
            const double hz = (double)k * EW_SAMPLE_RATE_HZ / EW_FFT_LEN;
            const double mel = hz_to_mel_d(hz);
            double w = 0.0;
            if (mel > left && mel <= centre) {
                w = (mel - left) / (centre - left);
            } else if (mel > centre && mel < right) {
                w = (right - mel) / (right - centre);
            }
            if (w > 0.0) {
                if (start < 0) {
                    start = k;
                }
                if (used + len >= EW_MEL_WEIGHTS_MAX) {
                    return -1;
                }
                m->mel_weights[used + len] = (float)w;
                len++;
            } else if (start >= 0) {
                break;  /* triangle finished; bins are monotonic in mel */
            }
        }
        if (len == 0) {
            return -1;  /* band narrower than one FFT bin: bad config */
        }
        m->mel_start[b] = (uint16_t)start;
        m->mel_len[b] = (uint16_t)len;
        m->mel_offset[b] = (uint16_t)used;
        used += len;
    }

    /* Orthonormal DCT-II basis, truncated to the first NUM_MFCC rows. */
    for (int k = 0; k < EW_NUM_MFCC; k++) {
        const double scale = (k == 0) ? sqrt(1.0 / EW_MEL_BANDS) : sqrt(2.0 / EW_MEL_BANDS);
        for (int n = 0; n < EW_MEL_BANDS; n++) {
            m->dct[k][n] = (float)(scale * cos(M_PI / EW_MEL_BANDS * (n + 0.5) * k));
        }
    }
    return 0;
}

EW_IRAM void ew_mfcc_compute(ew_mfcc_t *m, const int16_t *samples, float *out)
{
    /* int16 -> float in [-1, 1), windowed. */
    const float scale = 1.0f / 32768.0f;
    for (int n = 0; n < EW_FRAME_LEN; n++) {
        m->frame[n] = (float)samples[n] * scale * m->window[n];
    }

    ew_rfft_power(&m->fft, m->frame, m->power);

    for (int b = 0; b < EW_MEL_BANDS; b++) {
        const float *w = &m->mel_weights[m->mel_offset[b]];
        const float *p = &m->power[m->mel_start[b]];
        float acc = 0.0f;
        for (int i = 0; i < m->mel_len[b]; i++) {
            acc += w[i] * p[i];
        }
        m->log_mel[b] = logf(acc + EW_LOG_EPS);
    }

    for (int k = 0; k < EW_NUM_MFCC; k++) {
        float acc = 0.0f;
        for (int n = 0; n < EW_MEL_BANDS; n++) {
            acc += m->dct[k][n] * m->log_mel[n];
        }
        out[k] = acc;
    }
}
