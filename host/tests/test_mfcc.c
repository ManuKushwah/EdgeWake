#include <math.h>
#include <string.h>

#include "ew_mfcc.h"
#include "ew_test.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static ew_mfcc_t mfcc;

static void test_filterbank_geometry(void)
{
    int prev_start = -1;
    for (int b = 0; b < EW_MEL_BANDS; b++) {
        EW_CHECK(mfcc.mel_len[b] >= 1);
        EW_CHECK(mfcc.mel_start[b] >= prev_start);  /* bands move up in frequency */
        prev_start = mfcc.mel_start[b];
        for (int i = 0; i < mfcc.mel_len[b]; i++) {
            const float w = mfcc.mel_weights[mfcc.mel_offset[b] + i];
            EW_CHECK(w > 0.0f && w <= 1.0f);
        }
    }
    /* Highest band must stay below 4 kHz = bin 128. */
    const int last = EW_MEL_BANDS - 1;
    EW_CHECK(mfcc.mel_start[last] + mfcc.mel_len[last] <= 129);
}

static void test_silence(void)
{
    int16_t zeros[EW_FRAME_LEN] = {0};
    float out[EW_NUM_MFCC];
    ew_mfcc_compute(&mfcc, zeros, out);
    /* log(eps) in every band -> only c0 is non-zero: sqrt(B) * log(eps). */
    EW_CHECK_NEAR(out[0], sqrt((double)EW_MEL_BANDS) * log(1e-6), 1e-3);
    for (int k = 1; k < EW_NUM_MFCC; k++) {
        EW_CHECK_NEAR(out[k], 0.0, 1e-3);
    }
}

static void test_tone_peaks_in_matching_band(void)
{
    int16_t x[EW_FRAME_LEN];
    float out[EW_NUM_MFCC];
    for (int n = 0; n < EW_FRAME_LEN; n++) {
        x[n] = (int16_t)(16000.0 * sin(2.0 * M_PI * 1000.0 * n / EW_SAMPLE_RATE_HZ));
    }
    ew_mfcc_compute(&mfcc, x, out);

    int best = 0;
    for (int b = 1; b < EW_MEL_BANDS; b++) {
        if (mfcc.log_mel[b] > mfcc.log_mel[best]) {
            best = b;
        }
    }
    /* The 1 kHz bin (32) must sit inside the winning triangle. */
    EW_CHECK(mfcc.mel_start[best] <= 32 && 32 < mfcc.mel_start[best] + mfcc.mel_len[best]);
}

/* Independent double-precision implementation (naive DFT, filterbank
 * re-derived from the formulas) to catch errors in the optimized path. */
static double ref_mel(double hz)
{
    return 2595.0 * log10(1.0 + hz / 700.0);
}

static void reference_mfcc(const int16_t *x, double *out)
{
    double power[EW_FFT_BINS];
    for (int k = 0; k < EW_FFT_BINS; k++) {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < EW_FRAME_LEN; n++) {
            const double w = 0.5 - 0.5 * cos(2.0 * M_PI * n / EW_FRAME_LEN);
            const double v = x[n] / 32768.0 * w;
            re += v * cos(2.0 * M_PI * k * n / EW_FFT_LEN);
            im -= v * sin(2.0 * M_PI * k * n / EW_FFT_LEN);
        }
        power[k] = re * re + im * im;
    }
    const double lo = ref_mel(EW_MEL_LOW_HZ), hi = ref_mel(EW_MEL_HIGH_HZ);
    const double step = (hi - lo) / (EW_MEL_BANDS + 1);
    double log_mel[EW_MEL_BANDS];
    for (int b = 0; b < EW_MEL_BANDS; b++) {
        const double l = lo + b * step, c = l + step, r = c + step;
        double acc = 0.0;
        for (int k = 0; k < EW_FFT_BINS; k++) {
            const double m = ref_mel((double)k * EW_SAMPLE_RATE_HZ / EW_FFT_LEN);
            double w = 0.0;
            if (m > l && m <= c) {
                w = (m - l) / (c - l);
            } else if (m > c && m < r) {
                w = (r - m) / (r - c);
            }
            acc += w * power[k];
        }
        log_mel[b] = log(acc + 1e-6);
    }
    for (int k = 0; k < EW_NUM_MFCC; k++) {
        const double s = k == 0 ? sqrt(1.0 / EW_MEL_BANDS) : sqrt(2.0 / EW_MEL_BANDS);
        double acc = 0.0;
        for (int n = 0; n < EW_MEL_BANDS; n++) {
            acc += log_mel[n] * cos(M_PI / EW_MEL_BANDS * (n + 0.5) * k);
        }
        out[k] = s * acc;
    }
}

static void test_matches_double_reference(void)
{
    int16_t x[EW_FRAME_LEN];
    float got[EW_NUM_MFCC];
    double want[EW_NUM_MFCC];
    for (int trial = 0; trial < 10; trial++) {
        /* Noise plus a chirp-ish tone at varying levels. */
        const double level = 200.0 * (trial + 1);
        for (int n = 0; n < EW_FRAME_LEN; n++) {
            const double tone = level * sin(2.0 * M_PI * (300.0 + 150.0 * trial) * n / EW_SAMPLE_RATE_HZ);
            const double noise = (double)((int)(ew_rand() % 2001) - 1000) * 0.2;
            x[n] = (int16_t)(tone + noise);
        }
        ew_mfcc_compute(&mfcc, x, got);
        reference_mfcc(x, want);
        for (int k = 0; k < EW_NUM_MFCC; k++) {
            EW_CHECK_NEAR(got[k], want[k], 2e-3 * (1.0 + fabs(want[k])));
        }
    }
}

int main(void)
{
    EW_CHECK(ew_mfcc_init(&mfcc) == 0);
    printf("mfcc\n");
    EW_RUN(test_filterbank_geometry);
    EW_RUN(test_silence);
    EW_RUN(test_tone_peaks_in_matching_band);
    EW_RUN(test_matches_double_reference);
    return EW_TEST_RESULT();
}
