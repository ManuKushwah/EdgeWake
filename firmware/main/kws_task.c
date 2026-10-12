/*
 * kws_task.c - drains 20 ms hops from the ring and runs the streaming KWS
 * engine (VAD gate -> MFCC -> int8 CNN -> smoothing) on core 1.
 *
 * The task sleeps on a direct-to-task notification that the capture task
 * gives after every push; a pending notification is remembered, so there is
 * no lost-wake-up window between "ring empty" and "go to sleep".
 */
#include "kws_task.h"

#include <stdatomic.h>
#include <string.h>

#include "actions.h"
#include "app_config.h"
#include "esp_log.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "model_settings.h"
#include "probes.h"
#include "sdkconfig.h"

static const char *TAG = "kws";

#if CONFIG_EW_VAD_ENABLE
#define EW_VAD_DEFAULT true
#else
#define EW_VAD_DEFAULT false
#endif

static ew_kws_t s_kws;                 /* ~16.5 KB: MFCC tables, feature ring, history */
static int16_t s_hop[EW_FRAME_HOP];
static StaticTask_t s_tcb;
static StackType_t s_stack[EW_STACK_KWS];
static TaskHandle_t s_handle;
static ew_ringbuf_t *s_ring;
static QueueHandle_t s_events;

/* Stats are written on core 1 and snapshotted from core 0: a short spinlock
 * critical section keeps the multi-field snapshot consistent. */
static portMUX_TYPE s_lock = portMUX_INITIALIZER_UNLOCKED;
static kws_stats_t s_stats;

static _Atomic int32_t s_req_vad = -1;        /* -1: no request, else 0/1 */
static _Atomic int32_t s_req_threshold = -1;  /* -1: no request, else percent */
static uint32_t s_seq;

static int timed_infer(void *ctx, const int8_t *input, int8_t *scores)
{
    const int64_t t0 = esp_timer_get_time();
    const int rc = inference_run(ctx, input, scores);
    const uint32_t dt = (uint32_t)(esp_timer_get_time() - t0);

    portENTER_CRITICAL(&s_lock);
    s_stats.infer_us_last = dt;
    s_stats.infer_us_max = (dt > s_stats.infer_us_max) ? dt : s_stats.infer_us_max;
    s_stats.infer_us_sum += dt;
    s_stats.infer_count++;
    portEXIT_CRITICAL(&s_lock);
    return rc;
}

static void apply_requests(void)
{
    const int32_t vad = atomic_exchange(&s_req_vad, -1);
    if (vad >= 0) {
        s_kws.cfg.vad_enabled = vad != 0;
        ESP_LOGI(TAG, "VAD gate %s", vad ? "enabled" : "disabled");
    }
    const int32_t pct = atomic_exchange(&s_req_threshold, -1);
    if (pct >= 0) {
        s_kws.cfg.detect_threshold = (float)pct / 100.0f;
        ESP_LOGI(TAG, "threshold %ld%%", (long)pct);
    }
}

static void kws_task(void *arg)
{
    (void)arg;
    ESP_ERROR_CHECK(esp_task_wdt_add(NULL));
    ESP_LOGI(TAG, "running on core %d", xPortGetCoreID());

    for (;;) {
        /* Sleep until the producer signals a new chunk (timeout keeps the
         * watchdog fed if audio ever stops; health reports the stall). */
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));
        apply_requests();

        while (ew_ringbuf_read(s_ring, s_hop, EW_FRAME_HOP)) {
            const int64_t t0 = esp_timer_get_time();
            EW_PROBE_HIGH(EW_PROBE_DSP);

            ew_kws_result_t r;
            ew_kws_process_hop(&s_kws, s_hop, &r);
            if (r.detected) {
                actions_on_keyword(r.top_class);  /* GPIO first: lowest latency */
            }

            EW_PROBE_LOW(EW_PROBE_DSP);
            const uint32_t dt = (uint32_t)(esp_timer_get_time() - t0);

            bool dropped = false;
            if (r.detected) {
                const kws_event_t ev = {
                    .seq = ++s_seq,
                    .timestamp_ms = (uint32_t)(t0 / 1000),
                    .hop_to_action_us = dt,
                    .score = r.top_score,
                    .class_index = (uint8_t)r.top_class,
                };
                /* Zero timeout: networking must never stall the audio path. */
                dropped = xQueueSend(s_events, &ev, 0) != pdTRUE;
                ESP_LOGI(TAG, "\"%s\" p=%.2f (%lu us)", ew_labels[r.top_class],
                         (double)r.top_score, (unsigned long)dt);
            }

            portENTER_CRITICAL(&s_lock);
            s_stats.engine = s_kws.stats;
            s_stats.busy_us += dt;
            s_stats.hop_us_max = (dt > s_stats.hop_us_max) ? dt : s_stats.hop_us_max;
            s_stats.events_dropped += dropped;
            s_stats.vad_enabled = s_kws.cfg.vad_enabled;
            s_stats.threshold = s_kws.cfg.detect_threshold;
            portEXIT_CRITICAL(&s_lock);
        }
        esp_task_wdt_reset();
    }
}

esp_err_t kws_task_start(const inference_info_t *info, ew_ringbuf_t *ring, QueueHandle_t events)
{
    s_ring = ring;
    s_events = events;

    ew_kws_config_t cfg = {
        .input_scale = info->input_scale,
        .input_zero_point = info->input_zero_point,
        .output_scale = info->output_scale,
        .output_zero_point = info->output_zero_point,
        .feat_mean = ew_feat_mean,
        .feat_inv_std = ew_feat_inv_std,
        .infer_every_hops = CONFIG_EW_INFER_EVERY_HOPS,
        .smooth_window = CONFIG_EW_SMOOTH_WINDOW,
        .detect_threshold = (float)CONFIG_EW_DETECT_THRESHOLD_PCT / 100.0f,
        .refractory_hops = CONFIG_EW_REFRACTORY_MS / 20,
        .silence_index = EW_LABEL_SILENCE,
        .unknown_index = EW_LABEL_UNKNOWN,
        .vad_enabled = EW_VAD_DEFAULT,
        .infer = timed_infer,
        .infer_ctx = NULL,
    };
    ew_vad_default_config(&cfg.vad);
    if (ew_kws_init(&s_kws, &cfg) != 0) {
        return ESP_ERR_INVALID_ARG;
    }

    s_handle = xTaskCreateStaticPinnedToCore(kws_task, "kws", EW_STACK_KWS, NULL, EW_PRIO_KWS,
                                             s_stack, &s_tcb, EW_CORE_KWS);
    return s_handle ? ESP_OK : ESP_FAIL;
}

TaskHandle_t kws_task_handle(void)
{
    return s_handle;
}

void kws_get_stats(kws_stats_t *out)
{
    portENTER_CRITICAL(&s_lock);
    *out = s_stats;
    portEXIT_CRITICAL(&s_lock);
}

void kws_request_vad(bool enabled)
{
    atomic_store(&s_req_vad, enabled ? 1 : 0);
}

void kws_request_threshold(float threshold)
{
    int32_t pct = (int32_t)(threshold * 100.0f + 0.5f);
    pct = pct < 1 ? 1 : (pct > 99 ? 99 : pct);
    atomic_store(&s_req_threshold, pct);
}
