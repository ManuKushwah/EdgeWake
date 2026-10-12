#include <math.h>

#include "ew_fft.h"
#include "ew_test.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static ew_rfft_t fft;

/* O(N^2) reference DFT in double precision. */
static void naive_power(const float *x, double *power)
{
    for (int k = 0; k < EW_FFT_BINS; k++) {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < EW_FFT_LEN; n++) {
            const double a = 2.0 * M_PI * k * n / EW_FFT_LEN;
            re += x[n] * cos(a);
            im -= x[n] * sin(a);
        }
        power[k] = re * re + im * im;
    }
}

static void test_matches_naive_dft(void)
{
    float x[EW_FFT_LEN];
    float got[EW_FFT_BINS];
    double want[EW_FFT_BINS];

    for (int trial = 0; trial < 20; trial++) {
        for (int n = 0; n < EW_FFT_LEN; n++) {
            x[n] = (float)((int)(ew_rand() % 20001) - 10000) / 10000.0f;
        }
        ew_rfft_power(&fft, x, got);
        naive_power(x, want);

        double peak = 0.0;
        for (int k = 0; k < EW_FFT_BINS; k++) {
            peak = fmax(peak, want[k]);
        }
        for (int k = 0; k < EW_FFT_BINS; k++) {
            /* float32 vs float64: error relative to the spectrum peak. */
            EW_CHECK_NEAR(got[k], want[k], 1e-4 * peak);
        }
    }
}

static void test_pure_tone_lands_in_its_bin(void)
{
    float x[EW_FFT_LEN];
    float p[EW_FFT_BINS];
    const int bin = 32;  /* 32 * 16000 / 512 = 1000 Hz */
    for (int n = 0; n < EW_FFT_LEN; n++) {
        x[n] = (float)cos(2.0 * M_PI * bin * n / EW_FFT_LEN);
    }
    ew_rfft_power(&fft, x, p);
    int argmax = 0;
    for (int k = 1; k < EW_FFT_BINS; k++) {
        if (p[k] > p[argmax]) {
            argmax = k;
        }
    }
    EW_CHECK(argmax == bin);
    /* |X[k]| = N/2 for a unit cosine exactly on a bin. */
    EW_CHECK_NEAR(sqrtf(p[bin]), EW_FFT_LEN / 2.0, 1e-2);
}

static void test_dc_and_nyquist(void)
{
    float x[EW_FFT_LEN];
    float p[EW_FFT_BINS];
    for (int n = 0; n < EW_FFT_LEN; n++) {
        x[n] = 1.0f;
    }
    ew_rfft_power(&fft, x, p);
    EW_CHECK_NEAR(p[0], (double)EW_FFT_LEN * EW_FFT_LEN, 1.0);
    EW_CHECK_NEAR(p[EW_FFT_LEN / 2], 0.0, 1e-3);

    for (int n = 0; n < EW_FFT_LEN; n++) {
        x[n] = (n & 1) ? -1.0f : 1.0f;  /* alternating = Nyquist */
    }
    ew_rfft_power(&fft, x, p);
    EW_CHECK_NEAR(p[EW_FFT_LEN / 2], (double)EW_FFT_LEN * EW_FFT_LEN, 1.0);
    EW_CHECK_NEAR(p[0], 0.0, 1e-3);
}

int main(void)
{
    ew_rfft_init(&fft);
    printf("fft\n");
    EW_RUN(test_matches_naive_dft);
    EW_RUN(test_pure_tone_lands_in_its_bin);
    EW_RUN(test_dc_and_nyquist);
    return EW_TEST_RESULT();
}
