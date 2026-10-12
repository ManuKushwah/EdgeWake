/*
 * EdgeWake - always-on keyword spotting on the ESP32-S3.
 *
 * Boot sequence:
 *   NVS -> OTA state -> GPIO -> TFLM model + golden-vector check
 *   -> KWS task (core 1) -> audio capture task (core 0) -> health monitor
 *   -> Wi-Fi + MQTT telemetry (optional)
 *   -> if this is the first boot after an OTA: self-test, then confirm or
 *      roll back.
 */
#include <stdlib.h>

#include "actions.h"
#include "app_config.h"
#include "audio_capture.h"
#include "esp_app_desc.h"
#include "esp_log.h"
#include "health.h"
#include "inference.h"
#include "kws_task.h"
#include "model_settings.h"
#include "nvs_flash.h"
#include "ota.h"
#include "probes.h"
#include "telemetry.h"
#include "wifi.h"

static const char *TAG = "main";

/* Detection events: KWS task -> telemetry task. Statically allocated. */
static StaticQueue_t s_event_queue_mem;
static uint8_t s_event_queue_storage[EW_EVENT_QUEUE_LEN * sizeof(kws_event_t)];

/* Runs a known test-set clip through the model and compares with the scores
 * the reference int8 interpreter produced when the model was exported. Must
 * run before the KWS task starts (the interpreter is not re-entrant). */
static bool golden_vector_ok(void)
{
    int8_t scores[EW_NUM_CLASSES];
    if (inference_run(NULL, ew_golden_input, scores) != 0) {
        return false;
    }
    int best = 0, max_diff = 0;
    for (int c = 0; c < EW_NUM_CLASSES; c++) {
        best = scores[c] > scores[best] ? c : best;
        const int d = abs(scores[c] - ew_golden_scores[c]);
        max_diff = d > max_diff ? d : max_diff;
    }
    ESP_LOGI(TAG, "golden vector: got \"%s\", expected \"%s\", max |dq| = %d", ew_labels[best],
             ew_labels[ew_golden_class], max_diff);
    return best == ew_golden_class && max_diff <= 2;
}

static bool wait_until(bool (*cond)(void), uint32_t timeout_ms)
{
    for (uint32_t t = 0; t < timeout_ms; t += 100) {
        if (cond()) {
            return true;
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    return cond();
}

static bool pipeline_running(void)
{
    kws_stats_t s;
    kws_get_stats(&s);
    return s.engine.hops >= 100;  /* >= 2 s of audio processed */
}

void app_main(void)
{
    const esp_app_desc_t *app = esp_app_get_description();
    ESP_LOGI(TAG, "EdgeWake %s (ESP-IDF %s)", app->version, app->idf_ver);

    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    ota_init();
    probes_init();
    actions_init();

    inference_info_t info;
    err = inference_init(&info);
    if (err != ESP_OK) {
        ota_confirm(false, "model failed to load");  /* rolls back if this is an OTA boot */
        ESP_ERROR_CHECK(err);
    }
    const bool golden_ok = golden_vector_ok();

    QueueHandle_t events = xQueueCreateStatic(EW_EVENT_QUEUE_LEN, sizeof(kws_event_t),
                                              s_event_queue_storage, &s_event_queue_mem);
    audio_ring_init();
    ESP_ERROR_CHECK(kws_task_start(&info, audio_ring(), events));
    ESP_ERROR_CHECK(audio_capture_start(kws_task_handle()));
    ESP_ERROR_CHECK(health_start());

    const bool networked = wifi_start() == ESP_OK;
    if (networked) {
        if (telemetry_start(events) != ESP_OK) {
            ESP_LOGW(TAG, "telemetry not started");
        }
    } else {
        ESP_LOGW(TAG, "no Wi-Fi configured: running offline");
    }

    if (ota_pending_verify()) {
        ESP_LOGW(TAG, "first boot of a new image: running self-test");
        const char *failure = NULL;
#if CONFIG_EW_SELFTEST_FORCE_FAIL
        failure = "forced by CONFIG_EW_SELFTEST_FORCE_FAIL (rollback demo)";
        (void)golden_ok;
        (void)wait_until;
        (void)pipeline_running;
#else
        if (!golden_ok) {
            failure = "golden-vector inference mismatch";
        } else if (!wait_until(audio_capture_is_alive, 3000)) {
            failure = "microphone silent";
        } else if (!wait_until(pipeline_running, 5000)) {
            failure = "KWS pipeline not running";
        } else if (networked && !wifi_wait_connected(30000)) {
            /* An image that can't reach the network can't receive a fix. */
            failure = "Wi-Fi did not connect";
        }
#endif
        ota_confirm(failure == NULL, failure);
    }
    ESP_LOGI(TAG, "up: model %u B, arena %u B, ring %d samples", (unsigned)info.model_bytes,
             (unsigned)info.arena_used_bytes, EW_AUDIO_RING_SAMPLES);
}
