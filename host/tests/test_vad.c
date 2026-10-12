#include <math.h>

#include "ew_config.h"
#include "ew_test.h"
#include "ew_vad.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void make_noise(int16_t *x, int amplitude)
{
    for (int i = 0; i < EW_FRAME_HOP; i++) {
        x[i] = (int16_t)((int)(ew_rand() % (2u * amplitude + 1)) - amplitude);
    }
}

static void make_tone(int16_t *x, double amplitude)
{
    for (int i = 0; i < EW_FRAME_HOP; i++) {
        x[i] = (int16_t)(amplitude * sin(2.0 * M_PI * 440.0 * i / EW_SAMPLE_RATE_HZ));
    }
}

static void test_energy(void)
{
    int16_t x[4] = {100, -100, 100, -100};
    EW_CHECK_NEAR(ew_vad_energy(x, 4), 10000.0, 1e-3);
    int16_t big[2] = {-32768, -32768};  /* worst case must not overflow */
    EW_CHECK_NEAR(ew_vad_energy(big, 2), 1073741824.0, 1.0);
}

static void test_closes_on_steady_noise_and_opens_on_speech(void)
{
    ew_vad_config_t cfg;
    ew_vad_default_config(&cfg);
    ew_vad_t v;
    ew_vad_init(&v, &cfg);
    int16_t x[EW_FRAME_HOP];

    /* Warm-up keeps the gate open, then steady background closes it. */
    bool open = true;
    for (int i = 0; i < 100; i++) {
        make_noise(x, 300);
        open = ew_vad_process(&v, x, EW_FRAME_HOP);
    }
    EW_CHECK(!open);

    /* A loud burst (~+30 dB) opens it immediately. */
    make_tone(x, 8000.0);
    EW_CHECK(ew_vad_process(&v, x, EW_FRAME_HOP));

    /* Hangover: stays open for exactly hangover_hops quiet hops. */
    for (uint32_t i = 0; i < cfg.hangover_hops; i++) {
        make_noise(x, 300);
        EW_CHECK(ew_vad_process(&v, x, EW_FRAME_HOP));
    }
    make_noise(x, 300);
    EW_CHECK(!ew_vad_process(&v, x, EW_FRAME_HOP));
}

static void test_ignores_quiet_sounds_below_absolute_floor(void)
{
    ew_vad_config_t cfg;
    ew_vad_default_config(&cfg);
    ew_vad_t v;
    ew_vad_init(&v, &cfg);
    int16_t x[EW_FRAME_HOP];
    for (int i = 0; i < 60; i++) {
        make_noise(x, 5);  /* near-digital silence */
        ew_vad_process(&v, x, EW_FRAME_HOP);
    }
    /* ~17 dB above the background but below abs_min_db (30 dB): not speech. */
    make_tone(x, 30.0);
    EW_CHECK(!ew_vad_process(&v, x, EW_FRAME_HOP));
}

/* Regression: background whose hop energy jumps around (like pink noise or a
 * room with HVAC rumble). A "4x the running minimum" rule opened on most of
 * these hops; the sigma-scaled dB threshold must stay closed. */
static void test_fluctuating_background_stays_closed(void)
{
    ew_vad_config_t cfg;
    ew_vad_default_config(&cfg);
    ew_vad_t v;
    ew_vad_init(&v, &cfg);
    int16_t x[EW_FRAME_HOP];
    int open = 0;
    const int hops = 1000;
    for (int i = 0; i < hops; i++) {
        /* level varies by up to ~10 dB from hop to hop */
        make_noise(x, 150 + (int)(ew_rand() % 301));
        open += ew_vad_process(&v, x, EW_FRAME_HOP) && i >= 100;
    }
    printf("    open on %.1f%% of fluctuating-noise hops\n", 100.0 * open / (hops - 100));
    EW_CHECK(open < (hops - 100) / 10);

    /* ...and a real 30 dB louder burst still opens it. */
    make_tone(x, 9000.0);
    EW_CHECK(ew_vad_process(&v, x, EW_FRAME_HOP));
}

/* Regression (code review): a mic that outputs exact zeros while it starts up
 * must not seed the background estimate, or the gate latches open for tens
 * of seconds once the real room noise arrives. */
static void test_startup_zeros_do_not_latch_gate(void)
{
    ew_vad_config_t cfg;
    ew_vad_default_config(&cfg);
    ew_vad_t v;
    ew_vad_init(&v, &cfg);
    int16_t x[EW_FRAME_HOP] = {0};
    for (int i = 0; i < 15; i++) {  /* 300 ms of digital silence */
        ew_vad_process(&v, x, EW_FRAME_HOP);
    }
    int open_after_warmup = 0;
    for (int i = 15; i < 15 + 500; i++) {  /* then 10 s of steady ~40 dB room noise */
        make_noise(x, 250);
        const bool open = ew_vad_process(&v, x, EW_FRAME_HOP);
        open_after_warmup += open && i >= (int)cfg.warmup_hops;
    }
    printf("    open %d hops after warm-up\n", open_after_warmup);
    EW_CHECK(open_after_warmup <= 20);
}

static void test_adapts_to_louder_background(void)
{
    ew_vad_config_t cfg;
    ew_vad_default_config(&cfg);
    ew_vad_t v;
    ew_vad_init(&v, &cfg);
    int16_t x[EW_FRAME_HOP];
    for (int i = 0; i < 100; i++) {
        make_noise(x, 300);
        ew_vad_process(&v, x, EW_FRAME_HOP);
    }
    /* A fan switches on: background energy jumps ~16x and stays there.
     * The gate opens, but the slowly rising floor must close it again. */
    int open_hops = 0;
    bool open = true;
    for (int i = 0; i < 3000 && open; i++) {
        make_noise(x, 1200);
        open = ew_vad_process(&v, x, EW_FRAME_HOP);
        open_hops += open;
    }
    EW_CHECK(!open);
    printf("    gate re-closed after %d hops (%.1f s)\n", open_hops, open_hops * 0.02);
}

int main(void)
{
    printf("vad\n");
    EW_RUN(test_energy);
    EW_RUN(test_closes_on_steady_noise_and_opens_on_speech);
    EW_RUN(test_ignores_quiet_sounds_below_absolute_floor);
    EW_RUN(test_fluctuating_background_stays_closed);
    EW_RUN(test_startup_zeros_do_not_latch_gate);
    EW_RUN(test_adapts_to_louder_background);
    return EW_TEST_RESULT();
}
