#include <math.h>
#include <string.h>

#include "ew_kws.h"
#include "ew_test.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* --- stub "model" ---------------------------------------------------------- */

typedef struct {
    int8_t last_input[EW_INPUT_SIZE];
    int calls;
    int hot_class;      /* class that gets probability `hot_prob` */
    float hot_prob;
} stub_model_t;

static int stub_infer(void *ctx, const int8_t *input, int8_t *scores)
{
    stub_model_t *m = (stub_model_t *)ctx;
    memcpy(m->last_input, input, EW_INPUT_SIZE);
    m->calls++;
    /* Output quantization used below: scale 1/256, zero point -128. */
    const float rest = (1.0f - m->hot_prob) / (EW_NUM_CLASSES - 1);
    for (int c = 0; c < EW_NUM_CLASSES; c++) {
        const float p = (c == m->hot_class) ? m->hot_prob : rest;
        long q = lroundf(p * 256.0f) - 128;
        scores[c] = (int8_t)(q > 127 ? 127 : q);
    }
    return 0;
}

static const float k_mean[EW_NUM_MFCC] = {-40, 2, -1, 0.5f, 0, 0, 0, 0, 0, 0};
static const float k_inv_std[EW_NUM_MFCC] = {0.05f, 0.2f, 0.3f, 0.3f, 0.4f, 0.4f, 0.5f, 0.5f, 0.5f, 0.5f};

static ew_kws_config_t make_config(stub_model_t *m, bool vad)
{
    ew_kws_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.input_scale = 0.05f;
    cfg.input_zero_point = 3;
    cfg.output_scale = 1.0f / 256.0f;
    cfg.output_zero_point = -128;
    cfg.feat_mean = k_mean;
    cfg.feat_inv_std = k_inv_std;
    cfg.infer_every_hops = 2;
    cfg.smooth_window = 3;
    cfg.detect_threshold = 0.8f;
    cfg.refractory_hops = 50;
    cfg.silence_index = 0;
    cfg.unknown_index = 1;
    cfg.vad_enabled = vad;
    ew_vad_default_config(&cfg.vad);
    cfg.infer = stub_infer;
    cfg.infer_ctx = m;
    return cfg;
}

static ew_kws_t kws;  /* ~20 KB: static, like on the device */

static void synth_clip(int16_t *x, int n)
{
    for (int i = 0; i < n; i++) {
        const double t = (double)i / EW_SAMPLE_RATE_HZ;
        const double env = sin(M_PI * t);  /* swells over the second */
        const double v = env * (6000.0 * sin(2 * M_PI * 400 * t) + 3000.0 * sin(2 * M_PI * 1700 * t)) +
                         (double)((int)(ew_rand() % 401) - 200);
        x[i] = (int16_t)v;
    }
}

/* The streaming path (hop by hop, sliding window, ring of quantized frames)
 * must hand the model exactly what offline extraction of the same 1 s clip
 * produces - this is what lets us train on offline features. */
static void test_streaming_matches_offline_features(void)
{
    static int16_t clip[16000];
    static int16_t stream[128 + 16000];
    synth_clip(clip, 16000);

    stub_model_t model = {.hot_class = 0, .hot_prob = 0.9f};
    ew_kws_config_t cfg = make_config(&model, false);
    cfg.infer_every_hops = 1;
    EW_CHECK(ew_kws_init(&kws, &cfg) == 0);

    /* Expected: offline frames i = 0..48 at clip offsets 320*i. */
    int8_t expected[EW_INPUT_SIZE];
    for (int i = 0; i < EW_NUM_FRAMES; i++) {
        float mf[EW_NUM_MFCC];
        ew_mfcc_compute(&kws.mfcc, &clip[i * EW_FRAME_HOP], mf);
        ew_kws_quantize_frame(&kws, mf, &expected[i * EW_NUM_MFCC]);
    }

    /* Streaming frames end on hop boundaries; a 128-sample lead-in aligns
     * them with the offline grid (512 - 2 * 320 = -128). */
    memset(stream, 0, 128 * sizeof(int16_t));
    memcpy(stream + 128, clip, sizeof(clip));
    for (int h = 0; h < 50; h++) {
        ew_kws_process_hop(&kws, &stream[h * EW_FRAME_HOP], NULL);
    }
    EW_CHECK(model.calls == 1);  /* first inference exactly when 49 frames exist */
    EW_CHECK(memcmp(model.last_input, expected, EW_INPUT_SIZE) == 0);
}

static void test_detection_and_refractory(void)
{
    stub_model_t model = {.hot_class = 5, .hot_prob = 0.95f};
    ew_kws_config_t cfg = make_config(&model, false);
    EW_CHECK(ew_kws_init(&kws, &cfg) == 0);

    int16_t hop[EW_FRAME_HOP];
    int detections = 0, last_hit = -1000, min_gap = 1 << 30;
    for (int h = 0; h < 400; h++) {
        synth_clip(hop, EW_FRAME_HOP);
        ew_kws_result_t r;
        ew_kws_process_hop(&kws, hop, &r);
        if (r.detected) {
            EW_CHECK(r.top_class == 5);
            EW_CHECK(r.top_score >= 0.8f);
            if (h - last_hit < min_gap) {
                min_gap = h - last_hit;
            }
            last_hit = h;
            detections++;
        }
    }
    EW_CHECK(detections >= 2);
    EW_CHECK(min_gap >= (int)cfg.refractory_hops);
    EW_CHECK(kws.stats.detections == (uint32_t)detections);
}

static void test_non_keywords_never_fire(void)
{
    for (int cls = 0; cls <= 1; cls++) {  /* silence, unknown */
        stub_model_t model = {.hot_class = cls, .hot_prob = 0.99f};
        ew_kws_config_t cfg = make_config(&model, false);
        EW_CHECK(ew_kws_init(&kws, &cfg) == 0);
        int16_t hop[EW_FRAME_HOP];
        for (int h = 0; h < 200; h++) {
            synth_clip(hop, EW_FRAME_HOP);
            ew_kws_process_hop(&kws, hop, NULL);
        }
        EW_CHECK(kws.stats.detections == 0);
        EW_CHECK(model.calls > 0);
    }
}

static void test_vad_gates_inference(void)
{
    stub_model_t model = {.hot_class = 0, .hot_prob = 0.9f};
    ew_kws_config_t cfg = make_config(&model, true);
    EW_CHECK(ew_kws_init(&kws, &cfg) == 0);

    int16_t hop[EW_FRAME_HOP];
    /* 6 s of quiet background: after warm-up, the CNN never runs. */
    for (int h = 0; h < 300; h++) {
        for (int i = 0; i < EW_FRAME_HOP; i++) {
            hop[i] = (int16_t)((int)(ew_rand() % 61) - 30);
        }
        ew_kws_process_hop(&kws, hop, NULL);
    }
    EW_CHECK(model.calls == 0);
    EW_CHECK(kws.stats.skipped_by_vad == kws.stats.infer_slots);

    /* 200 ms of "speech": CNN runs during the burst and the hangover. */
    for (int h = 0; h < 10; h++) {
        synth_clip(hop, EW_FRAME_HOP);
        for (int i = 0; i < EW_FRAME_HOP; i++) {
            hop[i] = (int16_t)(hop[i] + 4000.0 * sin(2 * M_PI * 300.0 * i / EW_SAMPLE_RATE_HZ));
        }
        ew_kws_process_hop(&kws, hop, NULL);
    }
    for (int h = 0; h < 60; h++) {
        for (int i = 0; i < EW_FRAME_HOP; i++) {
            hop[i] = (int16_t)((int)(ew_rand() % 61) - 30);
        }
        ew_kws_process_hop(&kws, hop, NULL);
    }
    const int expected = (int)(10 + cfg.vad.hangover_hops) / 2;
    EW_CHECK(model.calls >= expected - 1 && model.calls <= expected + 1);
    EW_CHECK(kws.stats.inferences + kws.stats.skipped_by_vad == kws.stats.infer_slots);
    printf("    slots=%u ran=%u skipped=%u\n", kws.stats.infer_slots, kws.stats.inferences,
           kws.stats.skipped_by_vad);
}

static void test_rejects_bad_config(void)
{
    stub_model_t model = {0};
    ew_kws_config_t cfg = make_config(&model, false);
    cfg.smooth_window = EW_KWS_MAX_SMOOTH + 1;
    EW_CHECK(ew_kws_init(&kws, &cfg) != 0);
    cfg = make_config(&model, false);
    cfg.infer = NULL;
    EW_CHECK(ew_kws_init(&kws, &cfg) != 0);
}

int main(void)
{
    printf("kws\n");
    EW_RUN(test_streaming_matches_offline_features);
    EW_RUN(test_detection_and_refractory);
    EW_RUN(test_non_keywords_never_fire);
    EW_RUN(test_vad_gates_inference);
    EW_RUN(test_rejects_bad_config);
    printf("    sizeof(ew_kws_t) = %zu bytes (static)\n", sizeof(ew_kws_t));
    return EW_TEST_RESULT();
}
